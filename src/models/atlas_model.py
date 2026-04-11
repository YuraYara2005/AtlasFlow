# src/models/atlas_model.py

"""
AtlasModel — the top-level inference model for the AtlasFlow pipeline.

Pipeline
--------
::

    query_tensor  (seq_q,  feature_dim)  ─► SpatioTemporalEncoder ─► (seq_q,  embed_dim) ─┐
                                                                                            ├─► CrossAttention ─► representation + attn_weights
    context_tensor (seq_kv, feature_dim) ─► SpatioTemporalEncoder ─► (seq_kv, embed_dim) ─┘

Design notes
------------
- Single entry-point (``forward``) accepts two raw feature tensors — the
  *query* (e.g. the most recent window) and the *context* (e.g. a reference
  window or memory bank) — and returns both the attended representation and
  the attention weight map for interpretability.

- Both tensors are passed through the *same* ``SpatioTemporalEncoder`` so
  that the embedding space is shared and the model generalises without
  learning separate projections per stream.

- No training loop, loss, or optimisation logic lives here.  AtlasModel is
  purely a composable forward-pass module, ready to be embedded in any
  surrounding training or serving harness.

- All sub-modules are ``nn.Module`` instances, so ``state_dict()``,
  ``.to(device)``, ``.train()``/``.eval()`` and parameter counting all
  work out of the box from the parent class.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from src.enconding.spatiotemporal_encoder import SpatioTemporalEncoder
from src.models.attention.cross_attention import CrossAttention


class AtlasModel(nn.Module):
    """
    End-to-end inference model for AtlasFlow warehouse event streams.

    Accepts two raw feature windows (query and context), encodes them into
    a shared embedding space via :class:`~src.enconding.spatiotemporal_encoder.SpatioTemporalEncoder`,
    then cross-attends the query against the context via
    :class:`~src.models.attention.cross_attention.CrossAttention`.

    Parameters
    ----------
    feature_dim : int
        Number of raw features per time step produced by
        :class:`~src.preprocessing.tensor_encoder.TensorEncoder`.
        Defaults to 4 — ``[time_delta, aisle, bin, event_type_id]``.
    embed_dim : int
        Common embedding dimension shared across encoding and attention.
        Must be a positive even integer (for sinusoidal encoding) and
        divisible by ``num_heads`` (for multi-head attention).
        Defaults to 64.
    num_heads : int
        Number of parallel attention heads in :class:`CrossAttention`.
        Defaults to 4.
    vocab_size : int
        Size of the event-type embedding vocabulary passed to
        :class:`SpatioTemporalEncoder`.  ID 0 is the ``<PAD>``/``<UNKNOWN>``
        sentinel.  Defaults to 20.
    dropout : float
        Dropout probability on attention weights.  No-op at eval time.
        Defaults to 0.0.

    Inputs
    ------
    query_tensor : torch.Tensor
        Shape ``(seq_q, feature_dim)``, ``dtype=torch.float32``.
        The window being attended *from* — typically the most recent window.
    context_tensor : torch.Tensor or None, optional
        Shape ``(seq_kv, feature_dim)``, ``dtype=torch.float32``.
        The window supplying keys/values — a reference or memory window.
        When ``None`` (the default), the model falls back to self-attention:
        the query is attended against itself, enabling cold-start inference
        when no historical context is available yet.

    Returns
    -------
    representation : torch.Tensor
        Shape ``(seq_q, embed_dim)``.  The attended query representation,
        ready for a downstream decision head or further processing.
    attn_weights : torch.Tensor
        Shape ``(num_heads, seq_q, seq_kv)``.  Per-head attention
        distributions over the context, useful for interpretability and
        diagnostics.

    Raises
    ------
    ValueError
        Propagated from sub-modules if tensor shapes or configuration
        parameters violate their contracts.

    Examples
    --------
    >>> model = AtlasModel(feature_dim=4, embed_dim=64, num_heads=4)
    >>> query   = torch.randn(32, 4)   # current window
    >>> context = torch.randn(16, 4)   # reference window
    >>> representation, weights = model(query, context)
    >>> representation.shape
    torch.Size([32, 64])
    >>> weights.shape
    torch.Size([4, 32, 16])
    """

    def __init__(
        self,
        feature_dim: int = 4,
        embed_dim: int = 64,
        num_heads: int = 4,
        vocab_size: int = 20,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.feature_dim = feature_dim
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # Shared encoder — both query and context pass through the same weights
        # so they are projected into a common embedding space.
        self.encoder = SpatioTemporalEncoder(
            input_dim=feature_dim,
            embed_dim=embed_dim,
            vocab_size=vocab_size,
        )

        # Cross-attention over the encoded representations.
        self.attention = CrossAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        query_tensor: torch.Tensor,
        context_tensor: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run the full query → encode → attend pipeline.

        Parameters
        ----------
        query_tensor : torch.Tensor
            Shape ``(seq_q, feature_dim)``.
        context_tensor : torch.Tensor or None, optional
            Shape ``(seq_kv, feature_dim)``.
            When ``None``, the model performs self-attention: the encoded
            query is used as both keys and values.  This enables cold-start
            inference before any historical context window is available.

        Returns
        -------
        representation : torch.Tensor
            Shape ``(seq_q, embed_dim)``.
        attn_weights : torch.Tensor
            Shape ``(num_heads, seq_q, seq_kv)``.
        """
        # --- Encode ---------------------------------------------------------
        # The query is always encoded through the shared SpatioTemporalEncoder.
        query_emb = self.encoder(query_tensor)    # (seq_q, embed_dim)

        # Encode the context if provided; otherwise fall back to self-attention
        # by reusing the query embedding as keys/values.  This lets the model
        # operate on a single window during the cold-start phase of a stream.
        if context_tensor is not None:
            context_emb = self.encoder(context_tensor)  # (seq_kv, embed_dim)
        else:
            context_emb = query_emb                     # self-attention fallback

        # --- Attend ---------------------------------------------------------
        # CrossAttention cross-attends query_emb against context_emb.
        # representation: (seq_q, embed_dim)
        # attn_weights:   (num_heads, seq_q, seq_kv)
        representation, attn_weights = self.attention(query_emb, context_emb)

        return representation, attn_weights

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def count_parameters(self) -> int:
        """
        Return the total number of trainable parameters.

        Useful for quick sanity checks during development.

        Returns
        -------
        int
            Sum of ``p.numel()`` for all parameters where ``p.requires_grad``.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"feature_dim={self.feature_dim}, "
            f"embed_dim={self.embed_dim}, "
            f"num_heads={self.num_heads}, "
            f"params={self.count_parameters():,})"
        )
