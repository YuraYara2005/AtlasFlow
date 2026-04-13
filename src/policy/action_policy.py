# src/policy/action_policy.py

"""
ActionPolicy — deterministic policy layer for the AtlasFlow pipeline.

Sits at the end of the inference chain, after
:class:`~src.models.decision_head.DecisionHead` has produced logits and
:class:`~src.models.atlas_model.AtlasModel` has produced attention weights.

Responsibilities
----------------
1. **Status resolution** — converts the last-timestep logits to a predicted
   warehouse state (NORMAL, DELAY, CRITICAL) and its confidence score.

2. **Explainability (XAI)** — for non-NORMAL states, identifies the single
   context log that received the highest mean attention score across all heads
   at the last query timestep (the "culprit").  This gives operators a direct
   pointer into the raw event stream rather than a black-box verdict.

3. **Action generation** — maps the (state, culprit) pair to a plain-English
   recommended action for the warehouse operator.

Design notes
------------
- Pure Python / PyTorch only.  No external ML libraries.
- No state is mutated after construction — the class is stateless and
  thread-safe; a single instance can be shared across goroutines.
- All operations are deterministic and produce the same output given the
  same inputs, making the policy straightforward to unit-test.
- Torch tensors are only read (via softmax / argmax / mean), never modified.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch

from src.ingestion.schema import WarehouseLog

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

# Ordered label list — index must match the num_classes layout in DecisionHead.
_LABELS: tuple[str, ...] = ("NORMAL", "DELAY", "CRITICAL")

# Plain-English action templates.
# Keys must match entries in _LABELS.
# ``{culprit}`` is substituted with a short event description when available.
_ACTION_TEMPLATES: Dict[str, str] = {
    "NORMAL":   "No action required. Warehouse operating within normal parameters.",
    "DELAY":    "Reroute traffic away from affected zone. Culprit: {culprit}",
    "CRITICAL": "Dispatch supervisor immediately. Halt operations at affected zone. Culprit: {culprit}",
}

# States that trigger XAI attribution (i.e., a culprit is surfaced).
_ALERT_STATES: frozenset[str] = frozenset({"DELAY", "CRITICAL"})


class ActionPolicy:
    """
    Deterministic policy layer that converts model outputs into operator actions.

    Accepts the raw outputs of the full AtlasFlow inference stack
    (logits from :class:`~src.models.decision_head.DecisionHead` and
    attention weights from :class:`~src.models.atlas_model.AtlasModel`)
    together with the original :class:`~src.ingestion.schema.WarehouseLog`
    window, and returns a structured decision dictionary.

    Parameters
    ----------
    confidence_threshold : float, optional
        Minimum softmax probability required to trust a non-NORMAL prediction.
        If the winning class is below this threshold the status is downgraded
        to NORMAL and no alert is raised.  Defaults to 0.0 (trust all
        predictions regardless of confidence), which is appropriate when the
        upstream model is well-calibrated.
        Set to e.g. 0.6 to add a conservative safety margin.

    Examples
    --------
    >>> policy = ActionPolicy()
    >>> result = policy.evaluate(logits, attn_weights, current_window)
    >>> result["status"]
    'CRITICAL'
    >>> result["confidence"]
    0.91
    >>> result["explanation"]
    'Highest attention on event SCAN_FAILED at Aisle 4, Bin 12 (position 7 in window).'
    >>> result["recommended_action"]
    'Dispatch supervisor immediately. Halt operations at affected zone. Culprit: SCAN_FAILED at Aisle 4, Bin 12'
    """

    def __init__(self, confidence_threshold: float = 0.0) -> None:
        if not (0.0 <= confidence_threshold < 1.0):
            raise ValueError(
                f"confidence_threshold must be in [0, 1), got {confidence_threshold!r}"
            )
        self._confidence_threshold = confidence_threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        logits: torch.Tensor,
        attn_weights: torch.Tensor,
        current_window: List[WarehouseLog],
    ) -> Dict[str, object]:
        """
        Evaluate model outputs and return a structured operator decision.

        Parameters
        ----------
        logits : torch.Tensor
            Shape ``(seq_len, num_classes)``, ``dtype=torch.float32``.
            Raw (pre-softmax) class scores from
            :class:`~src.models.decision_head.DecisionHead`.
            Class mapping: ``0=NORMAL, 1=DELAY, 2=CRITICAL``.
        attn_weights : torch.Tensor
            Shape ``(num_heads, seq_q, seq_kv)``, ``dtype=torch.float32``.
            Per-head attention distributions from
            :class:`~src.models.atlas_model.AtlasModel`.
        current_window : list[WarehouseLog]
            The ordered list of warehouse log entries that produced the
            tensors above (oldest → newest).  Length must equal ``seq_kv``
            so that attention indices map 1-to-1 to log entries.

        Returns
        -------
        dict
            A plain-English decision package with the following keys:

            ``"status"`` : str
                Predicted warehouse state — one of ``NORMAL``, ``DELAY``,
                ``CRITICAL``.
            ``"confidence"`` : float
                Softmax probability of the predicted class, rounded to 4
                decimal places.
            ``"explanation"`` : str
                Human-readable description of which event drove the
                prediction (XAI).  Empty string when status is ``NORMAL``.
            ``"recommended_action"`` : str
                Plain-English instruction for the warehouse operator.

        Raises
        ------
        ValueError
            If ``logits`` is not 2-D, or if ``len(current_window)`` does not
            match ``seq_kv`` (the last dimension of ``attn_weights``).
        """
        self._validate(logits, attn_weights, current_window)

        # --- 1. Status resolution ----------------------------------------
        # Examine only the *last* timestep — it is the most recent model
        # prediction and reflects the current state of the warehouse.
        last_logits: torch.Tensor = logits[-1]                          # (num_classes,)
        probs: torch.Tensor       = torch.softmax(last_logits, dim=-1)  # (num_classes,)
        class_idx: int            = int(torch.argmax(probs).item())
        confidence: float         = round(float(probs[class_idx].item()), 4)
        status: str               = _LABELS[class_idx]

        # Apply confidence gate: if the model is unsure, default to NORMAL
        # so the operator is not flooded with low-confidence alerts.
        if status != "NORMAL" and confidence < self._confidence_threshold:
            status     = "NORMAL"
            class_idx  = 0
            confidence = round(float(probs[0].item()), 4)

        # --- 2. XAI attribution ------------------------------------------
        culprit_log: Optional[WarehouseLog] = None
        explanation: str                    = ""

        if status in _ALERT_STATES:
            culprit_log, culprit_idx = self._find_culprit(attn_weights, current_window)
            explanation = (
                f"Highest attention on event {culprit_log.event_type} "
                f"at Aisle {culprit_log.aisle}, Bin {culprit_log.bin} "
                f"(position {culprit_idx} in window)."
            )

        # --- 3. Action generation ----------------------------------------
        recommended_action: str = self._build_action(status, culprit_log)

        return {
            "status":               status,
            "confidence":           confidence,
            "explanation":          explanation,
            "recommended_action":   recommended_action,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _find_culprit(
        attn_weights: torch.Tensor,
        window: List[WarehouseLog],
    ) -> tuple[WarehouseLog, int]:
        """
        Identify the context log that attracted the most attention.

        Averages attention scores across all heads for the *last* query
        timestep, then returns the context position (and its corresponding
        :class:`WarehouseLog`) with the maximum mean score.

        Parameters
        ----------
        attn_weights : torch.Tensor
            Shape ``(num_heads, seq_q, seq_kv)``.
        window : list[WarehouseLog]
            Context logs aligned with the ``seq_kv`` dimension.

        Returns
        -------
        culprit_log : WarehouseLog
            The log entry that received the highest average attention.
        culprit_idx : int
            Its 0-based position within the window.
        """
        # Average over heads → (seq_kv,); focus on last query position.
        # Shape: (num_heads, seq_kv) → mean → (seq_kv,)
        mean_scores: torch.Tensor = attn_weights[:, -1, :].mean(dim=0)
        culprit_idx: int          = int(torch.argmax(mean_scores).item())
        return window[culprit_idx], culprit_idx

    @staticmethod
    def _build_action(
        status: str,
        culprit_log: Optional[WarehouseLog],
    ) -> str:
        """
        Construct the plain-English operator action string.

        Parameters
        ----------
        status : str
            Predicted warehouse state (``NORMAL``, ``DELAY``, or ``CRITICAL``).
        culprit_log : WarehouseLog or None
            The log entry identified by XAI as the primary driver.
            ``None`` when the status is ``NORMAL``.

        Returns
        -------
        str
            Ready-to-display action message.
        """
        template = _ACTION_TEMPLATES.get(status, "Status unknown. Manual review required.")

        if culprit_log is not None:
            culprit_desc = (
                f"{culprit_log.event_type} at "
                f"Aisle {culprit_log.aisle}, Bin {culprit_log.bin}"
            )
            return template.format(culprit=culprit_desc)

        return template

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate(
        logits: torch.Tensor,
        attn_weights: torch.Tensor,
        window: List[WarehouseLog],
    ) -> None:
        """
        Enforce shape contracts between the three inputs.

        Parameters
        ----------
        logits : torch.Tensor
            Must be 2-D: ``(seq_len, num_classes)``.
        attn_weights : torch.Tensor
            Must be 3-D: ``(num_heads, seq_q, seq_kv)``.
        window : list[WarehouseLog]
            Length must equal ``seq_kv`` (last dim of ``attn_weights``).

        Raises
        ------
        ValueError
            On any shape mismatch.
        """
        if logits.dim() != 2:
            raise ValueError(
                f"logits must be 2-D (seq_len, num_classes), got shape {tuple(logits.shape)}."
            )
        if attn_weights.dim() != 3:
            raise ValueError(
                f"attn_weights must be 3-D (num_heads, seq_q, seq_kv), "
                f"got shape {tuple(attn_weights.shape)}."
            )
        seq_kv = attn_weights.shape[2]
        if len(window) != seq_kv:
            raise ValueError(
                f"len(current_window) ({len(window)}) must equal seq_kv "
                f"({seq_kv}) — the last dimension of attn_weights."
            )

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"confidence_threshold={self._confidence_threshold}, "
            f"labels={_LABELS})"
        )
