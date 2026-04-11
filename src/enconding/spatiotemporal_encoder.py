# src/enconding/spatiotemporal_encoder.py

"""
SpatioTemporalEncoder — sinusoidal positional encoding for warehouse tensors.

Design notes
------------
- Input:  ``(seq_len, feature_dim)`` float32 tensor produced by
  :class:`~src.preprocessing.tensor_encoder.TensorEncoder`.
  Feature layout: [time_delta, aisle, bin, event_type_id].

- Feature split:
  The input's four features are treated according to their mathematical type:
  ``[time_delta, aisle, bin]`` are continuous scalars projected via
  ``nn.Linear(3, embed_dim)``; ``event_type_id`` is a categorical integer
  looked up in an ``nn.Embedding`` table.  Conflating them in a single linear
  layer would force the model to treat event labels as ordered numerics,
  destroying its ability to learn distinct event-type representations.

- Encoding:
  Each of the three continuous spatial/temporal scalars (time_delta, aisle, bin)
  is independently expanded into ``embed_dim // 2`` sin/cos pairs using the
  standard sinusoidal scheme from "Attention Is All You Need" (Vaswani et al.,
  2017).  The three encoding blocks are summed into a single embedding of shape
  ``(seq_len, embed_dim)`` and added to the fused continuous + event output —
  a residual-style fusion that preserves the original signal while injecting
  spatial/temporal structure.

- CPU efficiency:
  All frequency buffers are pre-computed once in ``__init__`` and registered
  as non-trainable buffers (``register_buffer``), avoiding repeated allocation
  on every forward pass.  Operations are performed with broadcasted tensor
  arithmetic — no Python loops over sequence positions.

- Compatibility:
  Inherits ``nn.Module`` so it composes naturally with any downstream
  PyTorch model.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class SpatioTemporalEncoder(nn.Module):
    """
    Adds sinusoidal spatial/temporal encodings to a warehouse feature tensor,
    with a dedicated embedding table for the categorical ``event_type_id``.

    The input features are split by mathematical type before processing:

    - **Continuous** — ``[time_delta, aisle, bin]`` are passed through
      ``nn.Linear(3, embed_dim)`` and independently encoded with sinusoidal
      functions to inject structural positional bias.
    - **Categorical** — ``event_type_id`` is looked up in an ``nn.Embedding``
      table, producing a learned dense vector per event class.  This prevents
      the model from treating event labels as ordered scalars.

    The three components (``projected_continuous``, ``embedded_events``,
    ``encoding``) are summed into the final ``(seq_len, embed_dim)`` output.

    Parameters
    ----------
    input_dim : int
        Total number of features per token in the input tensor.  Must match
        the ``feature_dim`` produced by
        :class:`~src.preprocessing.tensor_encoder.TensorEncoder` (default 4).
    embed_dim : int
        Dimensionality of the output embedding.  Must be a positive even
        integer so that sin/cos pairs can be formed.  Defaults to 64.
    vocab_size : int
        Number of entries in the event-type embedding table
        (``num_embeddings``).  Must be strictly greater than the highest
        ``event_type_id`` that will appear at runtime.  Defaults to 20,
        which comfortably covers the 8 canonical event types (IDs 1–8)
        plus headroom for future additions.  ID 0 is reserved as a
        ``<PAD>``/``<UNKNOWN>`` sentinel (``padding_idx=0``).

    Input
    -----
    x : torch.Tensor
        Shape ``(seq_len, input_dim)``, ``dtype=torch.float32``.
        Feature layout: ``[time_delta, aisle, bin, event_type_id]``.
        ``event_type_id`` is cast to ``torch.long`` internally.

    Output
    ------
    torch.Tensor
        Shape ``(seq_len, embed_dim)``, ``dtype=torch.float32``.

    Raises
    ------
    ValueError
        If ``embed_dim`` is not a positive even integer.
        If the input tensor's last dimension does not match ``input_dim``.

    Examples
    --------
    >>> encoder = SpatioTemporalEncoder(input_dim=4, embed_dim=64, vocab_size=20)
    >>> x = torch.randn(32, 4)          # (seq_len=32, feature_dim=4)
    >>> out = encoder(x)
    >>> out.shape
    torch.Size([32, 64])
    """

    def __init__(
        self,
        input_dim: int = 4,
        embed_dim: int = 64,
        vocab_size: int = 20,
    ) -> None:
        super().__init__()

        if embed_dim <= 0 or embed_dim % 2 != 0:
            raise ValueError(
                f"embed_dim must be a positive even integer, got {embed_dim!r}"
            )

        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size

        # Projects the three *continuous* features [time_delta, aisle, bin]
        # into embed_dim space.  event_type_id is handled separately below
        # because it is categorical — treating it as a scalar would force the
        # model to interpret arbitrary integer IDs as ordered magnitudes.
        self.continuous_projection = nn.Linear(3, embed_dim)

        # Learned dense lookup for the categorical event_type_id.
        # padding_idx=0 keeps the <PAD>/<UNKNOWN> vector as an all-zero
        # sentinel and excludes it from gradient updates.
        self.event_embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=0,
        )

        # ------------------------------------------------------------------
        # Pre-compute sinusoidal frequency bank — shape (embed_dim // 2,)
        # Frequencies follow the geometric spacing:
        #   freq_k = 1 / 10000^(2k / embed_dim)
        # ------------------------------------------------------------------
        half_dim = embed_dim // 2
        exponents = torch.arange(half_dim, dtype=torch.float32) / half_dim
        freqs = torch.exp(-math.log(10000.0) * exponents)  # (half_dim,)

        # Registered as a buffer so it moves with .to(device) / .cuda()
        # and is excluded from optimizer parameter groups automatically.
        self.register_buffer("freqs", freqs)  # (half_dim,)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode continuous features sinusoidally and embed the categorical
        event type, then fuse all components into a single tensor.

        Parameters
        ----------
        x : torch.Tensor
            Shape ``(seq_len, input_dim)``, ``dtype=torch.float32``.
            Feature layout: ``[time_delta, aisle, bin, event_type_id]``.

        Returns
        -------
        torch.Tensor
            Shape ``(seq_len, embed_dim)``, ``dtype=torch.float32``.
            Sum of ``projected_continuous``, ``embedded_events``, and
            ``encoding`` (sinusoidal spatio-temporal bias).
        """
        if x.dim() != 2 or x.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected input of shape (seq_len, {self.input_dim}), "
                f"got {tuple(x.shape)}"
            )

        # --- Feature split ---------------------------------------------------
        # Continuous scalars: (seq_len, 3)
        continuous_features = x[:, 0:3]  # [time_delta, aisle, bin]
        # Categorical index:  (seq_len,)  — must be long for nn.Embedding
        event_ids = x[:, 3].long()       # event_type_id

        # --- Projections -----------------------------------------------------
        # Linear projection of the three continuous features: (seq_len, embed_dim)
        projected_continuous = self.continuous_projection(continuous_features)

        # Dense lookup for the categorical event type: (seq_len, embed_dim)
        embedded_events = self.event_embedding(event_ids)

        # --- Sinusoidal encoding (continuous channels only) ------------------
        # Extract each scalar as (seq_len, 1) for broadcasting.
        time_delta = x[:, 0:1]   # seconds since window start
        aisle      = x[:, 1:2]   # aisle index
        bin_idx    = x[:, 2:3]   # bin index

        # Sum sinusoidal encodings from all three spatial/temporal channels.
        encoding = (
            self._sinusoidal(time_delta)
            + self._sinusoidal(aisle)
            + self._sinusoidal(bin_idx)
        )  # (seq_len, embed_dim)

        # --- Residual fusion -------------------------------------------------
        # All three components share the same embed_dim, so element-wise
        # addition is valid and gradient flows through each branch independently.
        return projected_continuous + embedded_events + encoding

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _sinusoidal(self, values: torch.Tensor) -> torch.Tensor:
        """
        Compute sinusoidal encoding for a single scalar channel.

        Parameters
        ----------
        values : torch.Tensor
            Shape ``(seq_len, 1)`` — one scalar per sequence position.

        Returns
        -------
        torch.Tensor
            Shape ``(seq_len, embed_dim)`` — interleaved sin and cos columns.

        Notes
        -----
        The encoding for position *p* and dimension *k* follows:

        .. code-block:: text

            enc[p, 2k]   = sin(values[p] * freqs[k])
            enc[p, 2k+1] = cos(values[p] * freqs[k])

        All operations are fully vectorised over the sequence axis.
        """
        # (seq_len, half_dim) — broadcast multiply
        angles = values * self.freqs.unsqueeze(0)  # type: ignore[operator]

        # Interleave sin and cos along the feature axis.
        # Stack → (seq_len, half_dim, 2), then flatten → (seq_len, embed_dim)
        enc = torch.stack([torch.sin(angles), torch.cos(angles)], dim=-1)
        return enc.flatten(start_dim=1)  # (seq_len, embed_dim)

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:  # noqa: D105
        return (
            f"{self.__class__.__name__}("
            f"input_dim={self.input_dim}, "
            f"embed_dim={self.embed_dim}, "
            f"vocab_size={self.vocab_size})"
        )
