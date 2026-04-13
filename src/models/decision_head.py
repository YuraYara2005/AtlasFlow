# src/models/decision_head.py

"""
DecisionHead — final classification MLP for the AtlasFlow pipeline.

Sits downstream of :class:`~src.models.atlas_model.AtlasModel` and maps
the attended representation tensor to per-class logits.

Design notes
------------
- Raw logits are returned (no softmax).  This is the standard PyTorch
  convention: ``nn.CrossEntropyLoss`` applies log-softmax internally
  during training, and the serving layer can apply ``torch.softmax`` /
  ``torch.argmax`` as needed at inference time.

- The two-layer MLP (Linear → ReLU → Dropout → Linear) is intentionally
  minimal.  The AtlasModel encoder + attention already produce a rich
  contextual representation; the head only needs to collapse embed_dim
  into num_classes without overfitting.

- No training logic lives here.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class DecisionHead(nn.Module):
    """
    Two-layer MLP that maps an attended representation to warehouse-state logits.

    Architecture
    ------------
    ::

        (seq_len, embed_dim)
              │
        Linear(embed_dim → hidden_dim)
              │
            ReLU
              │
           Dropout
              │
        Linear(hidden_dim → num_classes)
              │
        (seq_len, num_classes)   ← raw logits, no softmax

    The default class mapping assumed by the surrounding pipeline is::

        0: NORMAL
        1: DELAY
        2: CRITICAL

    Override ``num_classes`` if the classification taxonomy changes.

    Parameters
    ----------
    embed_dim : int
        Dimensionality of the input representation produced by
        :class:`~src.models.atlas_model.AtlasModel`.  Defaults to 64.
    hidden_dim : int
        Width of the intermediate hidden layer.  Defaults to 32.
    num_classes : int
        Number of output logits (warehouse states).  Defaults to 3.
    dropout : float
        Dropout probability applied after the ReLU activation.
        No-op at eval time.  Defaults to 0.1.

    Examples
    --------
    >>> head = DecisionHead(embed_dim=64, hidden_dim=32, num_classes=3)
    >>> representation = torch.randn(32, 64)   # from AtlasModel
    >>> logits = head(representation)
    >>> logits.shape
    torch.Size([32, 3])
    """

    def __init__(
        self,
        embed_dim: int = 64,
        hidden_dim: int = 32,
        num_classes: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.embed_dim   = embed_dim
        self.hidden_dim  = hidden_dim
        self.num_classes = num_classes

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, representation: torch.Tensor) -> torch.Tensor:
        """
        Map the attended representation to raw class logits.

        Parameters
        ----------
        representation : torch.Tensor
            Shape ``(seq_len, embed_dim)``, ``dtype=torch.float32``.
            Typically the first return value of
            :meth:`~src.models.atlas_model.AtlasModel.forward`.

        Returns
        -------
        torch.Tensor
            Shape ``(seq_len, num_classes)``, ``dtype=torch.float32``.
            Raw logits — **no softmax applied**.  Pass directly to
            ``nn.CrossEntropyLoss`` during training or apply
            ``torch.softmax(..., dim=-1)`` at inference time.

        Raises
        ------
        ValueError
            If ``representation`` does not have shape
            ``(seq_len, embed_dim)``.
        """
        if representation.dim() != 2 or representation.shape[-1] != self.embed_dim:
            raise ValueError(
                f"Expected representation of shape (seq_len, {self.embed_dim}), "
                f"got {tuple(representation.shape)}."
            )

        return self.mlp(representation)  # (seq_len, num_classes)

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"embed_dim={self.embed_dim}, "
            f"hidden_dim={self.hidden_dim}, "
            f"num_classes={self.num_classes})"
        )
