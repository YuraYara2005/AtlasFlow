# src/api/server.py

"""
AtlasFlow FastAPI server — Integrated Hybrid AI Warehouse Supervisor.

Architecture
------------
Every POST /ingest event flows through two phases:

Phase 1 — The Brain (custom PyTorch pipeline):
    WarehouseLog → LogProcessor → (window full?) → AtlasModel (self-attn)
                → DecisionHead → ActionPolicy → raw decision dict

Phase 2 — The Mouth (Hugging Face text generation):
    DELAY / CRITICAL only:
        structured prompt → flan-t5-small → natural-language alert
        replaces templated recommended_action in the response

POST /ask provides a natural-language Q&A interface:
    Operator question + last_decision context
    → WarehouseQueryProcessor → flan-t5-small → answer

Configuration
-------------
All hyperparameters are loaded from config/model_config.yaml at startup via
load_config().  The hardcoded fallback dict in config_loader.py ensures the
server starts even if the YAML file is absent.

Startup
-------
All heavy objects (PyTorch models, HF pipeline) are initialised once via
FastAPI's lifespan context manager, stored in _state, and reused across
every request — no per-request model loading.

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
from src.reasoning.explainer import AttentionExplainer
from src.reasoning.query_processor import WarehouseQueryProcessor
from src.utils.config_loader import load_config

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Alert states that trigger HF natural-language generation (Phase 2)
# ---------------------------------------------------------------------------

_ALERT_STATES: frozenset[str] = frozenset({"DELAY", "CRITICAL"})

# ---------------------------------------------------------------------------
# Module-level pipeline state
# Populated once during startup; never replaced at request time.
# _state["last_decision"] is the only key mutated per-request (under GIL).
# ---------------------------------------------------------------------------

_state: Dict[str, Any] = {}


# ---------------------------------------------------------------------------
# Lifespan — initialise all heavy objects exactly once
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncGenerator[None, None]:
    """
    FastAPI lifespan context manager.

    Loads configuration, then builds every pipeline component in dependency
    order.  All objects are stored in ``_state`` and reused across requests.
    Resources are released on shutdown via ``_state.clear()``.
    """
    logger.info("AtlasFlow server starting up — reading configuration…")

    # ---- Load configuration (YAML or built-in fallback) -----------------
    cfg = load_config()                           # config/model_config.yaml
    pipeline_cfg = cfg["pipeline"]
    arch_cfg     = cfg["model_architecture"]
    gen_cfg      = cfg["generation"]

    logger.info(
        "Config loaded: window_size=%d, embed_dim=%d, hf_model=%s",
        pipeline_cfg["window_size"],
        arch_cfg["embed_dim"],
        gen_cfg["hf_model_name"],
    )

    # ---- Custom PyTorch pipeline ----------------------------------------
    # All models set to .eval() so dropout is a no-op at inference time.

    _state["log_processor"] = LogProcessor(
        window_size=pipeline_cfg["window_size"],
    )

    _state["atlas_model"] = AtlasModel(
        feature_dim=arch_cfg["feature_dim"],
        embed_dim=arch_cfg["embed_dim"],
        num_heads=arch_cfg["num_heads"],
    )
    _state["atlas_model"].eval()

    _state["decision_head"] = DecisionHead(
        embed_dim=arch_cfg["embed_dim"],
        hidden_dim=arch_cfg["hidden_dim"],
        num_classes=arch_cfg["num_classes"],
    )
    _state["decision_head"].eval()

    _state["action_policy"] = ActionPolicy(
        confidence_threshold=pipeline_cfg["confidence_threshold"],
    )

    logger.info(
        "PyTorch pipeline ready: %s | %s | %s | %s",
        _state["log_processor"],
        _state["atlas_model"],
        _state["decision_head"],
        _state["action_policy"],
    )

    # ---- XAI + Q&A layer ------------------------------------------------
    _state["explainer"]       = AttentionExplainer()
    _state["query_processor"] = WarehouseQueryProcessor(
        max_new_tokens=gen_cfg["max_new_tokens"],
    )

    # ---- Hugging Face text-generation pipeline ---------------------------
    # flan-t5-small (encoder-decoder) follows structured prompts reliably
    # and runs comfortably on CPU.
    logger.info("Loading Hugging Face model '%s'…", gen_cfg["hf_model_name"])
    _state["hf_generator"] = hf_pipeline(
        "text2text-generation",
        model=gen_cfg["hf_model_name"],
    )
    logger.info("Hugging Face pipeline ready.")

    # ---- Shared mutable state -------------------------------------------
    # Stores the most recent inference result so /ask can reference it.
    # Initialised to None; set after the first successful /ingest inference.
    _state["last_decision"] = None

    logger.info("AtlasFlow server startup complete. Listening for events.")
    yield  # server is live — handle requests

    # ---- Shutdown --------------------------------------------------------
    _state.clear()
    logger.info("AtlasFlow server shut down. Pipeline state cleared.")


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="AtlasFlow — AI Warehouse Supervisor",
    description=(
        "Real-time warehouse anomaly detection and natural-language Q&A "
        "powered by a hybrid PyTorch + Hugging Face inference pipeline."
    ),
    version="2.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class LogPayload(BaseModel):
    """
    Incoming warehouse event payload for POST /ingest.

    Mirrors :class:`~src.ingestion.schema.WarehouseLog` so FastAPI validates
    and auto-documents the request body.
    """

    timestamp:  float = Field(..., description="Unix epoch timestamp.",         examples=[1712958000.0])
    aisle:      int   = Field(..., ge=0, description="Non-negative aisle index.", examples=[4])
    bin:        int   = Field(..., ge=0, description="Non-negative bin index.",   examples=[12])
    event_type: str   = Field(..., min_length=1, description="Event type string.", examples=["SCAN_FAILED"])
    message:    str   = Field(..., description="Human-readable event message.",  examples=["Barcode unreadable at aisle 4, bin 12."])


class DecisionResponse(BaseModel):
    """
    Structured inference response from POST /ingest.

    All fields are ``null`` while the sliding window is still warming up.
    """

    status:             Optional[str]   = Field(None, description="Predicted warehouse state (NORMAL/DELAY/CRITICAL).")
    confidence:         Optional[float] = Field(None, description="Softmax probability of the predicted class.")
    explanation:        Optional[str]   = Field(None, description="XAI: which event drove the prediction.")
    recommended_action: Optional[str]   = Field(None, description="Plain-English operator instruction.")


class QueryPayload(BaseModel):
    """Request body for POST /ask."""

    query: str = Field(
        ...,
        min_length=1,
        description="Free-form natural-language question about the warehouse status.",
        examples=["What is causing the delay?"],
    )


class QueryResponse(BaseModel):
    """Response body for POST /ask."""

    answer:         str = Field(..., description="Natural-language answer from the language model.")
    current_status: str = Field(..., description="The warehouse state that was active when /ask was called.")
    explanation:    str = Field(..., description="XAI explanation string that grounded the answer.")


# ---------------------------------------------------------------------------
# Endpoint: POST /ingest
# ---------------------------------------------------------------------------

@app.post(
    "/ingest",
    response_model=DecisionResponse,
    summary="Ingest a single warehouse log event",
    description=(
        "Adds the event to the sliding window. When the window is full, the "
        "AI pipeline runs and returns a decision (and saves it for /ask). "
        "Returns null fields while the window is still warming up."
    ),
)
async def ingest(payload: LogPayload) -> DecisionResponse:
    """
    Main inference endpoint — two-phase pipeline.

    Phase 1 — Brain (PyTorch):
        LogProcessor → AtlasModel (self-attn) → DecisionHead → ActionPolicy

    Phase 2 — Mouth (HF, alert states only):
        Structured prompt → flan-t5-small → natural-language recommended_action

    The result is persisted in ``_state["last_decision"]`` so that POST /ask
    can answer questions about the most recent warehouse state.
    """
    # ------------------------------------------------------------------
    # Construct domain object (WarehouseLog validates field constraints)
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
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    logger.debug("Received log: %s", log)

    # ------------------------------------------------------------------
    # Phase 1 — The Brain
    # ------------------------------------------------------------------

    processor: LogProcessor = _state["log_processor"]
    feature_tensor          = processor.process(log)

    # LogProcessor returns None until the sliding window is full.
    # Return null fields to the caller without touching last_decision yet.
    if feature_tensor is None:
        logger.debug("Window not yet full — returning null response.")
        return DecisionResponse()

    # Window is full: run the full PyTorch inference stack under no_grad.
    atlas:  AtlasModel   = _state["atlas_model"]
    head:   DecisionHead = _state["decision_head"]
    policy: ActionPolicy = _state["action_policy"]

    with torch.no_grad():
        # Self-attention mode (context_tensor=None): the model attends the
        # window against itself, no external memory required.
        representation, attn_weights = atlas(feature_tensor)
        logits = head(representation)

    # Snapshot the current window as a list of WarehouseLog objects so
    # ActionPolicy can map attention indices back to real log entries.
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
    # Phase 2 — The Mouth (HF generation, alert states only)
    # ------------------------------------------------------------------

    recommended_action: str = raw_decision["recommended_action"]

    if raw_decision["status"] in _ALERT_STATES:
        prompt = (
            f"Translate this into a professional warehouse alert: "
            f"Status is {raw_decision['status']}. "
            f"Explanation: {raw_decision['explanation']}"
        )
        logger.info("Calling HF generator — prompt: %s", prompt)

        try:
            hf_output      = _state["hf_generator"](prompt, max_new_tokens=80)
            generated_text = hf_output[0]["generated_text"].strip()
            recommended_action = generated_text or recommended_action
            logger.info("HF generated action: %s", recommended_action)

        except Exception as exc:  # noqa: BLE001
            # Best-effort: generation failure must never crash the endpoint.
            logger.warning("HF generation failed (%s); using template fallback.", exc)

    # ------------------------------------------------------------------
    # Persist decision for /ask — update last_decision *after* Phase 2
    # so the stored record always contains the final recommended_action.
    # ------------------------------------------------------------------
    _state["last_decision"] = {
        "status":             raw_decision["status"],
        "confidence":         raw_decision["confidence"],
        "explanation":        raw_decision["explanation"],
        "recommended_action": recommended_action,
    }

    return DecisionResponse(
        status=raw_decision["status"],
        confidence=raw_decision["confidence"],
        explanation=raw_decision["explanation"],
        recommended_action=recommended_action,
    )


# ---------------------------------------------------------------------------
# Endpoint: POST /ask
# ---------------------------------------------------------------------------

@app.post(
    "/ask",
    response_model=QueryResponse,
    summary="Ask a natural-language question about the current warehouse status",
    description=(
        "Uses the most recent AI decision (from /ingest) as context and "
        "answers the operator's free-form question via flan-t5-small. "
        "Returns a polite warm-up message if no decision has been made yet."
    ),
)
async def ask(payload: QueryPayload) -> QueryResponse:
    """
    Natural-language Q&A endpoint.

    Retrieves the last persisted decision from ``_state["last_decision"]``,
    injects it as context into a structured prompt, and passes the prompt to
    :class:`~src.reasoning.query_processor.WarehouseQueryProcessor` which
    calls the resident Hugging Face pipeline and returns a grounded answer.

    Parameters
    ----------
    payload : QueryPayload
        The operator's free-form question.

    Returns
    -------
    QueryResponse
        Generated answer alongside the status and explanation that grounded it.
    """
    # ------------------------------------------------------------------
    # Guard: no decision available yet (window still warming up)
    # ------------------------------------------------------------------
    last_decision: Optional[Dict[str, Any]] = _state.get("last_decision")

    if last_decision is None:
        # Return a polite message — do NOT raise HTTP 503 here because the
        # client should retry, not treat this as a permanent error.
        warm_up_msg = (
            "The system is still warming up and waiting for logs. "
            "Please send at least one full window of events to /ingest first."
        )
        return QueryResponse(
            answer=warm_up_msg,
            current_status="UNKNOWN",
            explanation="No inference has been run yet.",
        )

    # ------------------------------------------------------------------
    # Extract context from last decision
    # ------------------------------------------------------------------
    current_status: str = last_decision["status"]
    explanation:    str = last_decision["explanation"] or (
        # NORMAL states produce an empty explanation string — provide a
        # sensible fallback so the prompt stays well-formed.
        "No anomaly detected. The warehouse is operating normally."
    )

    # ------------------------------------------------------------------
    # Generate natural-language answer
    # ------------------------------------------------------------------
    query_processor: WarehouseQueryProcessor = _state["query_processor"]

    answer = query_processor.answer_query(
        query=payload.query,
        current_status=current_status,
        explanation=explanation,
        hf_pipeline=_state["hf_generator"],
    )

    logger.info(
        "/ask: query=%r status=%s answer=%r",
        payload.query, current_status, answer,
    )

    return QueryResponse(
        answer=answer,
        current_status=current_status,
        explanation=explanation,
    )


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.get("/health", summary="Server health check")
async def health() -> Dict[str, str]:
    """Return 200 with pipeline and last_decision readiness."""
    ready = bool(_state)
    has_decision = _state.get("last_decision") is not None
    return {
        "status":        "ok" if ready else "initialising",
        "pipeline":      "ready" if ready else "loading",
        "last_decision": "available" if has_decision else "pending",
    }
