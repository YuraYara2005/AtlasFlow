# src/api/server.py

"""
AtlasFlow FastAPI server — Hybrid AI Warehouse Supervisor.

Architecture
------------
Incoming warehouse events flow through two phases on each POST /ingest call:

Phase 1 — The Brain (custom PyTorch pipeline):
    WarehouseLog → LogProcessor → (window full?) → AtlasModel (self-attn)
                → DecisionHead → ActionPolicy → raw decision dict

Phase 2 — The Mouth (Hugging Face text generation):
    If status is DELAY or CRITICAL:
        build a strict prompt from (status, explanation)
        → flan-t5-small → natural-language alert
        → inject into decision dict as recommended_action

NORMAL events skip Phase 2 entirely to save compute.

Startup
-------
All heavy objects (PyTorch models, HF pipeline) are initialized once via
FastAPI's lifespan context manager, stored in module-level state, and reused
across all requests.  This avoids re-loading model weights on every call.

Run locally with:
    uvicorn src.api.server:app --reload --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Dict, Optional

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import pipeline as hf_pipeline

from src.ingestion.schema import WarehouseLog
from src.models.atlas_model import AtlasModel
from src.models.decision_head import DecisionHead
from src.policy.action_policy import ActionPolicy
from src.preprocessing.log_processor import LogProcessor

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Alert states that trigger natural-language generation (Phase 2)
# ---------------------------------------------------------------------------

_ALERT_STATES: frozenset[str] = frozenset({"DELAY", "CRITICAL"})

# ---------------------------------------------------------------------------
# Module-level pipeline state
# Populated once during startup; never mutated at request time.
# ---------------------------------------------------------------------------

_state: Dict[str, Any] = {}


# ---------------------------------------------------------------------------
# Lifespan — initialise all heavy objects exactly once
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    FastAPI lifespan context manager.

    All models are loaded here on startup so that the first request is not
    penalised by model initialization latency.  Resources are released on
    shutdown (PyTorch GC handles tensors; HF pipeline holds no file handles).
    """
    logger.info("AtlasFlow server starting up — loading pipeline components…")

    # ---- Custom PyTorch pipeline ----------------------------------------
    # window_size=5: emit a tensor and run inference on every 5th log.
    # Models are set to eval() so dropout is disabled at inference time.
    _state["log_processor"] = LogProcessor(window_size=5)

    _state["atlas_model"] = AtlasModel(
        feature_dim=4,
        embed_dim=64,
        num_heads=4,
    )
    _state["atlas_model"].eval()

    _state["decision_head"] = DecisionHead(
        embed_dim=64,
        hidden_dim=32,
        num_classes=3,
    )
    _state["decision_head"].eval()

    _state["action_policy"] = ActionPolicy()

    logger.info("PyTorch pipeline ready: %s | %s | %s | %s",
                _state["log_processor"],
                _state["atlas_model"],
                _state["decision_head"],
                _state["action_policy"])

    # ---- Hugging Face text-generation pipeline ---------------------------
    # flan-t5-small is lightweight and runs comfortably on CPU.
    # We use text2text-generation (encoder-decoder) so the model can follow
    # structured prompts and output clean imperative sentences.
    logger.info("Loading Hugging Face flan-t5-small…")
    _state["hf_generator"] = hf_pipeline(
        "text2text-generation",
        model="google/flan-t5-small",
    )
    logger.info("Hugging Face pipeline ready.")

    logger.info("AtlasFlow server startup complete. Listening for events.")
    yield  # server is live

    # ---- Shutdown --------------------------------------------------------
    _state.clear()
    logger.info("AtlasFlow server shut down. Pipeline state cleared.")


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="AtlasFlow — AI Warehouse Supervisor",
    description=(
        "Real-time warehouse anomaly detection using a hybrid PyTorch + "
        "Hugging Face inference pipeline."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class LogPayload(BaseModel):
    """
    Incoming warehouse event payload.

    Mirrors the fields of :class:`~src.ingestion.schema.WarehouseLog` so
    that FastAPI validates and documents the request body automatically.
    """

    timestamp: float = Field(
        ...,
        description="Unix epoch timestamp of the event.",
        examples=[1712958000.0],
    )
    aisle: int = Field(
        ...,
        ge=0,
        description="Non-negative aisle index.",
        examples=[4],
    )
    bin: int = Field(
        ...,
        ge=0,
        description="Non-negative bin index within the aisle.",
        examples=[12],
    )
    event_type: str = Field(
        ...,
        min_length=1,
        description="Canonical event type string (e.g. SCAN_FAILED).",
        examples=["SCAN_FAILED"],
    )
    message: str = Field(
        ...,
        description="Human-readable event message.",
        examples=["Barcode unreadable at aisle 4, bin 12."],
    )


class DecisionResponse(BaseModel):
    """
    Structured response returned by POST /ingest.

    ``null`` for all fields means the sliding window is not yet full and no
    inference was run for this event.
    """

    status:               Optional[str]   = Field(None, description="Predicted warehouse state.")
    confidence:           Optional[float] = Field(None, description="Softmax probability of predicted class.")
    explanation:          Optional[str]   = Field(None, description="XAI: which event drove the prediction.")
    recommended_action:   Optional[str]   = Field(None, description="Plain-English operator instruction.")


# ---------------------------------------------------------------------------
# Endpoint: POST /ingest
# ---------------------------------------------------------------------------

@app.post(
    "/ingest",
    response_model=DecisionResponse,
    summary="Ingest a single warehouse log event",
    description=(
        "Adds the event to the sliding window. When the window is full "
        "the AI pipeline runs and returns a decision. Returns null fields "
        "while the window is still warming up."
    ),
)
async def ingest(payload: LogPayload) -> DecisionResponse:
    """
    Main inference endpoint.

    Phase 1 — Brain (PyTorch):
        Validate → construct WarehouseLog → LogProcessor → AtlasModel
        → DecisionHead → ActionPolicy → raw decision dict.

    Phase 2 — Mouth (Hugging Face):
        For DELAY / CRITICAL states, build a structured prompt and pass it
        through flan-t5-small to generate a professional alert sentence that
        replaces the templated ``recommended_action``.

    Parameters
    ----------
    payload : LogPayload
        Validated JSON body matching the WarehouseLog schema.

    Returns
    -------
    DecisionResponse
        Decision dictionary, or null-valued response if the window is not
        yet full.
    """
    # ------------------------------------------------------------------
    # Construct domain object
    # ------------------------------------------------------------------
    try:
        log = WarehouseLog(
            timestamp=payload.timestamp,
            aisle=payload.aisle,
            bin=payload.bin,
            event_type=payload.event_type,
            message=payload.message,
        )
    except ValueError as exc:
        # WarehouseLog validation failed (negative aisle, empty event_type, …)
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    logger.debug("Received log: %s", log)

    # ------------------------------------------------------------------
    # Phase 1 — The Brain
    # ------------------------------------------------------------------

    # LogProcessor returns None until the sliding window is full.
    # We return null fields to the caller while the window warms up.
    processor: LogProcessor      = _state["log_processor"]
    feature_tensor               = processor.process(log)

    if feature_tensor is None:
        logger.debug("Window not yet full — returning null response.")
        return DecisionResponse()

    # Window is full: run the full PyTorch inference stack.
    atlas:  AtlasModel    = _state["atlas_model"]
    head:   DecisionHead  = _state["decision_head"]
    policy: ActionPolicy  = _state["action_policy"]

    with torch.no_grad():
        # Self-attention mode: no external context window yet.
        # feature_tensor shape: (window_size, feature_dim)
        representation, attn_weights = atlas(feature_tensor)

        # logits shape: (window_size, num_classes)
        logits = head(representation)

    # ActionPolicy accesses the *current* window snapshot from LogProcessor.
    # We need the list of WarehouseLog objects that correspond to this tensor.
    # LogProcessor exposes its window manager; we snapshot it as a list.
    current_window = list(processor._window_manager._window)

    raw_decision: Dict[str, Any] = policy.evaluate(
        logits=logits,
        attn_weights=attn_weights,
        current_window=current_window,
    )

    logger.info(
        "Decision: status=%s confidence=%.4f",
        raw_decision["status"],
        raw_decision["confidence"],
    )

    # ------------------------------------------------------------------
    # Phase 2 — The Mouth (Hugging Face, only for alert states)
    # ------------------------------------------------------------------

    recommended_action: str = raw_decision["recommended_action"]

    if raw_decision["status"] in _ALERT_STATES:
        # Build a strict, information-dense prompt.
        # flan-t5 follows instruction-style prompts well with a leading verb.
        prompt = (
            f"Translate this into a professional warehouse alert: "
            f"Status is {raw_decision['status']}. "
            f"Explanation: {raw_decision['explanation']}"
        )

        logger.info("Calling HF generator with prompt: %s", prompt)

        try:
            hf_generator = _state["hf_generator"]
            # max_new_tokens caps output length for deterministic latency.
            hf_output = hf_generator(prompt, max_new_tokens=80)

            # hf_pipeline returns a list of dicts: [{"generated_text": "..."}]
            generated_text: str = hf_output[0]["generated_text"].strip()
            recommended_action  = generated_text or recommended_action

            logger.info("HF generated action: %s", recommended_action)

        except Exception as exc:  # noqa: BLE001
            # HF generation is best-effort: if it fails (OOM, timeout, …)
            # we fall back to the templated action from ActionPolicy so the
            # endpoint never returns a 500 due to the language model alone.
            logger.warning("HF generation failed (%s); using template fallback.", exc)

    # ------------------------------------------------------------------
    # Return structured response
    # ------------------------------------------------------------------

    return DecisionResponse(
        status=raw_decision["status"],
        confidence=raw_decision["confidence"],
        explanation=raw_decision["explanation"],
        recommended_action=recommended_action,
    )


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.get("/health", summary="Server health check")
async def health() -> Dict[str, str]:
    """Return 200 OK with pipeline readiness status."""
    ready = bool(_state)
    return {
        "status":   "ok" if ready else "initialising",
        "pipeline": "ready" if ready else "loading",
    }
