# src/models/attention/cross_attention.py

"""
CrossAttention — multi-head cross-attention module for the AtlasFlow pipeline.

Design notes
------------
- Cross-attention accepts two distinct tensors: a *query* sequence (e.g. the
  current window embedding) and a *context* sequence supplying keys/values
  (e.g. a memory bank or a second stream).  This generalises self-attention,
  which is the degenerate case where query == context.

- Bottleneck projection:
  Before computing Q/K/V, both query and context are passed through a shared
  bottleneck linear layer that reduces the hidden dimension to
  ``bottleneck_dim``.  This compresses redundant information early, cuts the
  cost of the subsequent Q/K/V projections (which operate on the smaller
  bottleneck representation), and acts as a regulariser.

- Scaled dot-product attention:
  Attention weights are computed as softmax(QKᵀ / √d_head) and applied to V.
  The scale factor prevents vanishing gradients when ``d_head`` is large
  (Vaswani et al., 2017).

- Output:
  The concatenated multi-head output is projected back to ``embed_dim`` by a
  final linear layer, matching the input dimension for residual connections in
  the surrounding architecture.

- CPU efficiency:
  All weight matrices are allocated once in ``__init__``.  The attention score
  computation uses pure PyTorch tensor operations (no Python loops over heads
  or sequence positions), keeping the hot path fully vectorised.

- No training logic:
  This module is purely a forward-pass building block.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttention(nn.Module):
    """
    Multi-head cross-attention with a bottleneck input projection.

    Architecture
    ------------
    ::

        query   (seq_q, embed_dim)  ─┐
                                      ├─ bottleneck ─► Q projection ─┐
        context (seq_kv, embed_dim) ─┘                               │
                                        bottleneck ─► K projection ─┤
                                                    ─► V projection ─┘
                                                                      │
                                             scaled dot-product attn  │
                                                                      │
                                              output projection        ▼
                                        (seq_q, embed_dim)

    Parameters
    ----------
    embed_dim : int
        Dimensionality of both the query and context tensors.  Must be
        divisible by ``num_heads``.  Also the dimension of the final output.
    num_heads : int
        Number of parallel attention heads.  Each head operates on
        ``d_head = bottleneck_dim // num_heads`` dimensions.  Defaults to 4.
    bottleneck_dim : int, optional
        Hidden width of the bottleneck projection applied *before* Q/K/V
        computation.  Defaults to ``embed_dim // 2``.  Must be divisible by
        ``num_heads``.
    dropout : float
        Dropout probability applied to the attention weights during training.
        Set to 0.0 (default) to disable.  No-op at eval time.

    Inputs
    ------
    query : torch.Tensor
        Shape ``(seq_q, embed_dim)``.  The sequence being attended *to*.
    context : torch.Tensor
        Shape ``(seq_kv, embed_dim)``.  The sequence supplying keys and values.

    Returns
    -------
    output : torch.Tensor
        Shape ``(seq_q, embed_dim)``.  Attended representation of the query.
    attn_weights : torch.Tensor
        Shape ``(num_heads, seq_q, seq_kv)``.  Softmax attention distributions
        per head — useful for inspection, visualisation, and debugging.

    Raises
    ------
    ValueError
        If ``embed_dim`` is not divisible by ``num_heads``.
        If ``bottleneck_dim`` is not divisible by ``num_heads``.
        If input tensors do not have shape ``(seq_len, embed_dim)``.

    Examples
    --------
    >>> attn = CrossAttention(embed_dim=64, num_heads=4)
    >>> query   = torch.randn(32, 64)   # current window
    >>> context = torch.randn(16, 64)   # memory / second stream
    >>> output, weights = attn(query, context)
    >>> output.shape
    torch.Size([32, 64])
    >>> weights.shape
    torch.Size([4, 32, 16])
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 4,
        bottleneck_dim: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        bottleneck_dim = bottleneck_dim if bottleneck_dim is not None else embed_dim // 2

        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})."
            )
        if bottleneck_dim % num_heads != 0:
            raise ValueError(
                f"bottleneck_dim ({bottleneck_dim}) must be divisible by "
                f"num_heads ({num_heads})."
            )

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.bottleneck_dim = bottleneck_dim
        self.d_head = bottleneck_dim // num_heads
        self._scale = math.sqrt(self.d_head)

        # ------------------------------------------------------------------
        # Bottleneck projections (query and context projected independently)
        # Reduces embed_dim → bottleneck_dim before Q/K/V splits.
        # No bias on the bottleneck — bias is absorbed into Q/K/V projections.
        # ------------------------------------------------------------------
        self.query_bottleneck   = nn.Linear(embed_dim, bottleneck_dim, bias=False)
        self.context_bottleneck = nn.Linear(embed_dim, bottleneck_dim, bias=False)

        # ------------------------------------------------------------------
        # Q / K / V projections (operate on bottleneck representations)
        # ------------------------------------------------------------------
        self.q_proj = nn.Linear(bottleneck_dim, bottleneck_dim)
        self.k_proj = nn.Linear(bottleneck_dim, bottleneck_dim)
        self.v_proj = nn.Linear(bottleneck_dim, bottleneck_dim)

        # ------------------------------------------------------------------
        # Output projection — maps concatenated heads back to embed_dim.
        # ------------------------------------------------------------------
        self.out_proj = nn.Linear(bottleneck_dim, embed_dim)

        # Attention weight dropout (applied before weighting values).
        self.attn_dropout = nn.Dropout(p=dropout)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        query: torch.Tensor,
        context: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute multi-head cross-attention.

        Parameters
        ----------
        query : torch.Tensor
            Shape ``(seq_q, embed_dim)``.
        context : torch.Tensor
            Shape ``(seq_kv, embed_dim)``.

        Returns
        -------
        output : torch.Tensor
            Shape ``(seq_q, embed_dim)``.
        attn_weights : torch.Tensor
            Shape ``(num_heads, seq_q, seq_kv)`` — softmax scores per head.
        """
        self._validate(query,   "query",   self.embed_dim)
        self._validate(context, "context", self.embed_dim)

        seq_q  = query.shape[0]
        seq_kv = context.shape[0]

        # --- Bottleneck ---------------------------------------------------
        # (seq_q,  bottleneck_dim) and (seq_kv, bottleneck_dim)
        q_bn = self.query_bottleneck(query)
        c_bn = self.context_bottleneck(context)

        # --- Q / K / V projections ----------------------------------------
        q = self.q_proj(q_bn)   # (seq_q,  bottleneck_dim)
        k = self.k_proj(c_bn)   # (seq_kv, bottleneck_dim)
        v = self.v_proj(c_bn)   # (seq_kv, bottleneck_dim)

        # --- Split into heads ---------------------------------------------
        # Reshape to (num_heads, seq, d_head) for batched matmul.
        q = self._split_heads(q, seq_q)    # (num_heads, seq_q,  d_head)
        k = self._split_heads(k, seq_kv)   # (num_heads, seq_kv, d_head)
        v = self._split_heads(v, seq_kv)   # (num_heads, seq_kv, d_head)

        # --- Scaled dot-product attention ---------------------------------
        # scores: (num_heads, seq_q, seq_kv)
        scores = torch.bmm(q, k.transpose(1, 2)) / self._scale

        attn_weights = F.softmax(scores, dim=-1)          # (num_heads, seq_q, seq_kv)
        attn_weights = self.attn_dropout(attn_weights)

        # context_vec: (num_heads, seq_q, d_head)
        context_vec = torch.bmm(attn_weights, v)

        # --- Merge heads & project output ---------------------------------
        # (num_heads, seq_q, d_head) → (seq_q, bottleneck_dim)
        merged = context_vec.transpose(0, 1).contiguous().view(seq_q, self.bottleneck_dim)

        output = self.out_proj(merged)   # (seq_q, embed_dim)

        return output, attn_weights

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _split_heads(self, x: torch.Tensor, seq_len: int) -> torch.Tensor:
        """
        Reshape ``(seq_len, bottleneck_dim)`` → ``(num_heads, seq_len, d_head)``.

        Parameters
        ----------
        x : torch.Tensor
            Shape ``(seq_len, bottleneck_dim)``.
        seq_len : int
            Sequence length (passed explicitly to avoid a redundant ``.shape`` call).

        Returns
        -------
        torch.Tensor
            Shape ``(num_heads, seq_len, d_head)``.
        """
        # view: (seq_len, num_heads, d_head) → transpose: (num_heads, seq_len, d_head)
        return x.view(seq_len, self.num_heads, self.d_head).transpose(0, 1)

    @staticmethod
    def _validate(tensor: torch.Tensor, name: str, expected_dim: int) -> None:
        """
        Assert that *tensor* has shape ``(seq_len, expected_dim)``.

        Parameters
        ----------
        tensor : torch.Tensor
            The tensor to validate.
        name : str
            Human-readable name used in the error message.
        expected_dim : int
            Expected size of the last dimension.

        Raises
        ------
        ValueError
            If the tensor does not satisfy the shape contract.
        """
        if tensor.dim() != 2 or tensor.shape[-1] != expected_dim:
            raise ValueError(
                f"'{name}' must have shape (seq_len, {expected_dim}), "
                f"got {tuple(tensor.shape)}."
            )

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"embed_dim={self.embed_dim}, "
            f"num_heads={self.num_heads}, "
            f"bottleneck_dim={self.bottleneck_dim}, "
            f"d_head={self.d_head})"
        )
