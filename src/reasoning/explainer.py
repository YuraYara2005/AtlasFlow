# src/reasoning/explainer.py

"""
AttentionExplainer — XAI utility for the AtlasFlow pipeline.

Translates raw attention weight tensors from
:class:`~src.models.attention.cross_attention.CrossAttention` into
human-readable explanations of *which* warehouse event caused an anomaly.

Design notes
------------
- Stateless: construct once, call as many times as needed.  No internal
  state is mutated between calls, making instances safe to share across
  threads.

- Shape-agnostic: ``extract_culprit`` accepts both the 3-D shape
  ``(num_heads, seq_q, seq_kv)`` produced by our unbatched
  ``CrossAttention`` forward pass and the 4-D batched variant
  ``(batch, num_heads, seq_q, seq_kv)`` expected by external callers.
  In the batched case the first batch element is examined.

- Attention attribution strategy: scores are averaged across all heads
  before argmax.  Head-averaging reduces noise from individual heads that
  may have specialised on syntax- or position-level patterns rather than
  content, yielding a more robust culprit signal.

- Query-position focus: only the *last* query timestep (``seq_q[-1]``)
  is examined.  It corresponds to the most recently observed log entry —
  the point at which the model made its final prediction.
"""

from __future__ import annotations

from typing import Tuple

import torch

from src.ingestion.schema import WarehouseLog


class AttentionExplainer:
    """
    Explainable-AI utility that maps attention weights to actionable insights.

    Methods
    -------
    extract_culprit(attn_weights, current_window)
        Identify the single log entry that drove the anomaly prediction.
    generate_summary(status, culprit_log)
        Produce a concise, operator-ready alert string.

    Examples
    --------
    >>> explainer = AttentionExplainer()
    >>> culprit, score = explainer.extract_culprit(attn_weights, window)
    >>> print(explainer.generate_summary("CRITICAL", culprit))
    'Critical event detected: SCAN_FAILED at Aisle 4, Bin 12.'
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_culprit(
        self,
        attn_weights: torch.Tensor,
        current_window: list[WarehouseLog],
    ) -> Tuple[WarehouseLog, float]:
        """
        Identify the context log with the highest mean attention score.

        Averages attention scores across all heads at the final query
        timestep, finds the ``seq_kv`` position with the maximum score,
        and returns the corresponding :class:`~src.ingestion.schema.WarehouseLog`
        together with its normalised attention percentage.

        Parameters
        ----------
        attn_weights : torch.Tensor
            Softmax attention distributions.  Accepted shapes:

            - ``(num_heads, seq_q, seq_kv)``  — unbatched (AtlasFlow default)
            - ``(batch, num_heads, seq_q, seq_kv)``  — batched variant;
              the first batch element (index 0) is used.

        current_window : list[WarehouseLog]
            The ordered window of warehouse log objects (oldest → newest)
            that was passed through the model.  Its length must equal
            ``seq_kv`` — the last dimension of ``attn_weights``.

        Returns
        -------
        culprit_log : WarehouseLog
            The log entry that received the highest average attention across
            all heads at the last query position.
        attention_score : float
            Normalised mean attention score for the culprit position,
            rounded to 4 decimal places.  Interpretable as the fraction of
            total attention directed at that event (0.0 – 1.0).

        Raises
        ------
        ValueError
            If ``attn_weights`` is not 3-D or 4-D.
            If ``len(current_window)`` does not match ``seq_kv``.
        """
        # --- Normalise to 3-D (num_heads, seq_q, seq_kv) -----------------
        weights = self._normalise_shape(attn_weights)

        # --- Validate window length against seq_kv -----------------------
        seq_kv = weights.shape[2]
        if len(current_window) != seq_kv:
            raise ValueError(
                f"len(current_window) ({len(current_window)}) must equal "
                f"seq_kv ({seq_kv}) — the last dimension of attn_weights."
            )

        # --- Head-averaged attention at the last query position -----------
        # weights[:, -1, :] → (num_heads, seq_kv)
        # .mean(dim=0)       → (seq_kv,)
        # Averaging across heads gives a content-driven signal rather than
        # head-specific syntactic/positional biases.
        mean_scores: torch.Tensor = weights[:, -1, :].mean(dim=0)  # (seq_kv,)

        # --- Argmax → culprit index --------------------------------------
        culprit_idx: int   = int(torch.argmax(mean_scores).item())
        attention_score    = round(float(mean_scores[culprit_idx].item()), 4)

        return current_window[culprit_idx], attention_score

    @staticmethod
    def generate_summary(status: str, culprit_log: WarehouseLog) -> str:
        """
        Produce a concise, operator-ready alert string.

        Parameters
        ----------
        status : str
            Predicted warehouse state — typically ``"NORMAL"``,
            ``"DELAY"``, or ``"CRITICAL"``.  The string is title-cased
            in the output so any capitalisation convention is accepted.
        culprit_log : WarehouseLog
            The log entry identified as the primary driver of the
            anomaly, as returned by :meth:`extract_culprit`.

        Returns
        -------
        str
            A single-sentence plain-English alert, e.g.::

                "Critical event detected: SCAN_FAILED at Aisle 4, Bin 12."

        Examples
        --------
        >>> AttentionExplainer.generate_summary("CRITICAL", culprit_log)
        'Critical event detected: SCAN_FAILED at Aisle 4, Bin 12.'
        >>> AttentionExplainer.generate_summary("DELAY", culprit_log)
        'Delay event detected: RESTOCK_TRIGGERED at Aisle 2, Bin 7.'
        """
        # Title-case the status so "CRITICAL" → "Critical" regardless of
        # what the upstream policy layer uses as its internal string.
        status_title = status.title()

        return (
            f"{status_title} event detected: "
            f"{culprit_log.event_type} at "
            f"Aisle {culprit_log.aisle}, Bin {culprit_log.bin}."
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise_shape(attn_weights: torch.Tensor) -> torch.Tensor:
        """
        Ensure attention weights are 3-D ``(num_heads, seq_q, seq_kv)``.

        Accepts either the unbatched 3-D output of ``CrossAttention`` or
        a batched 4-D tensor.  For the batched case the first element
        (index 0) is extracted, consistent with the single-window
        inference pattern used across the AtlasFlow pipeline.

        Parameters
        ----------
        attn_weights : torch.Tensor
            Shape ``(num_heads, seq_q, seq_kv)`` or
            ``(batch, num_heads, seq_q, seq_kv)``.

        Returns
        -------
        torch.Tensor
            Shape ``(num_heads, seq_q, seq_kv)``.

        Raises
        ------
        ValueError
            If the tensor is not 3-D or 4-D.
        """
        ndim = attn_weights.dim()

        if ndim == 3:
            # Already the expected unbatched shape — pass through unchanged.
            return attn_weights

        if ndim == 4:
            # Batched: (batch, num_heads, seq_q, seq_kv) → take batch[0].
            return attn_weights[0]

        raise ValueError(
            f"attn_weights must be 3-D (num_heads, seq_q, seq_kv) or "
            f"4-D (batch, num_heads, seq_q, seq_kv), got {ndim}-D tensor "
            f"of shape {tuple(attn_weights.shape)}."
        )

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
