"""
PRISM — Base Agent
==================
All 8 agents inherit from BaseAgent.  Provides:
  - Structured logging
  - Message construction helpers
  - Model loading from trained_models/ directory
  - Graceful error handling that never crashes the pipeline
"""

from __future__ import annotations
import logging
import os
import time
import traceback
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional

from core.protocol import AgentID, AgentMessage, PayloadType, PipelineState

logger = logging.getLogger("PRISM")


def _models_dir() -> Path:
    """
    Resolve the trained_models directory robustly.

    Search order:
      1. $PRISM_MODELS_DIR   (explicit env override)
      2. $PRISM_ROOT/models/trained_models
      3. <repo-root inferred from this file>/models/trained_models
      4. CWD/models/trained_models   (last resort)
    """
    # 1. Explicit env override
    env_models = os.environ.get("PRISM_MODELS_DIR")
    if env_models:
        return Path(env_models)

    # 2. PRISM_ROOT env
    env_root = os.environ.get("PRISM_ROOT")
    if env_root:
        return Path(env_root) / "models" / "trained_models"

    # 3. Infer from this file:  core/base_agent.py → ../models/trained_models
    this_file = Path(__file__).resolve()
    candidate = this_file.parent.parent / "models" / "trained_models"
    if candidate.exists():
        return candidate

    # 4. CWD fallback
    return Path.cwd() / "models" / "trained_models"


class BaseAgent(ABC):
    """Abstract base for all PRISM agents."""

    agent_id: AgentID   # must be set on every subclass

    def __init__(self, config: Dict[str, Any], output_dir: str = "data/outputs"):
        self.config     = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._model     = None          # populated by load_model()
        self.log        = logging.getLogger(f"PRISM.{self.agent_id.value}")

    # -----------------------------------------------------------------------
    # Public interface — called by the orchestrator
    # -----------------------------------------------------------------------

    def run(self, state: PipelineState) -> PipelineState:
        """
        Entry point.  Wraps _execute in try/except so one agent crash
        never aborts the whole pipeline.
        """
        self.log.info("=== %s starting ===", self.agent_id.value)
        t0 = time.time()
        try:
            state = self._execute(state)
        except Exception as exc:
            msg = f"{self.agent_id.value} FAILED: {exc}\n{traceback.format_exc()}"
            self.log.error(msg)
            state.errors.append(msg)
        elapsed = time.time() - t0
        self.log.info("=== %s done in %.1fs ===", self.agent_id.value, elapsed)
        return state

    # -----------------------------------------------------------------------
    # Abstract — subclasses implement this
    # -----------------------------------------------------------------------

    @abstractmethod
    def _execute(self, state: PipelineState) -> PipelineState:
        """Core logic.  Must update `state` and return it."""
        ...

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    def send(
        self,
        state:       PipelineState,
        recipient:   AgentID,
        payload_type:PayloadType,
        payload:     Dict[str, Any],
        confidence:  float = 1.0,
        notes:       str   = "",
    ) -> AgentMessage:
        msg = AgentMessage(
            sender_agent     = self.agent_id,
            recipient_agent  = recipient,
            payload_type     = payload_type,
            payload          = payload,
            agent_confidence = confidence,
            notes            = notes,
        )
        state.post_message(msg)
        self.log.debug("MSG → %s  conf=%.2f", recipient.value, confidence)
        return msg

    def load_model(self, model_filename: str) -> Any:
        """
        Load a pickled scikit-learn (or any picklable) model.

        Place your trained .pkl files at:
            PRISM/models/trained_models/<model_filename>.pkl

        You can override the search path via:
            export PRISM_MODELS_DIR=/absolute/path/to/trained_models

        Example:
            self._model = self.load_model("rf_ice_classifier")
            # looks for: models/trained_models/rf_ice_classifier.pkl

        HOW TO CONNECT YOUR TRAINED MODEL
        ----------------------------------
        1. Train your sklearn model and save it:
               import pickle
               with open("rf_ice_classifier.pkl", "wb") as f:
                   pickle.dump(model, f)

        2. Drop the .pkl file into:
               PRISM/models/trained_models/

        3. That's it — the agent auto-loads on startup and uses
           model.predict_proba() (Agent 2) or model.predict() (Agent 4).

        4. For the API/Render deployment, set the env var:
               PRISM_MODELS_DIR=/app/models/trained_models
        """
        import pickle

        model_dir  = _models_dir()
        model_path = model_dir / f"{model_filename}.pkl"

        if not model_path.exists():
            self.log.warning(
                "Model file not found: %s  —  agent will use physics-only fallback.\n"
                "  To fix: copy your .pkl to %s\n"
                "  Or set: export PRISM_MODELS_DIR=/your/path",
                model_path,
                model_dir,
            )
            return None

        try:
            with open(model_path, "rb") as f:
                model = pickle.load(f)
            self.log.info("Loaded model: %s", model_path)
            return model
        except Exception as exc:
            self.log.error("Failed to unpickle model %s: %s", model_path, exc)
            return None

    def output_path(self, filename: str) -> str:
        """Return absolute output path string."""
        return str(self.output_dir / filename)
