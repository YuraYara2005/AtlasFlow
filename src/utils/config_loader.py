# src/utils/config_loader.py

"""
config_loader — YAML configuration loader for the AtlasFlow pipeline.

Provides a single ``load_config`` function that reads
``config/model_config.yaml`` and returns a nested dictionary of
hyperparameters.  If the file cannot be found or parsed, a hardcoded
fallback dictionary mirroring the canonical YAML structure is returned and
a warning is logged so that the server starts up gracefully in any
deployment environment.

Usage
-----
    from src.utils.config_loader import load_config

    cfg = load_config()
    window_size = cfg["pipeline"]["window_size"]
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

import yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Fallback defaults
# ---------------------------------------------------------------------------
# Kept in strict parity with config/model_config.yaml.
# Update both locations whenever a new hyperparameter is introduced.

_DEFAULTS: Dict[str, Any] = {
    "pipeline": {
        "window_size":            5,
        "confidence_threshold":   0.0,
    },
    "model_architecture": {
        "feature_dim":  4,
        "embed_dim":    64,
        "num_heads":    4,
        "hidden_dim":   32,
        "num_classes":  3,
    },
    "generation": {
        "hf_model_name":    "google/flan-t5-small",
        "max_new_tokens":   80,
    },
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_config(
    config_path: str | Path = "config/model_config.yaml",
) -> Dict[str, Any]:
    """
    Load the AtlasFlow pipeline configuration from a YAML file.

    Reads ``config_path``, parses it with ``yaml.safe_load``, and returns
    the resulting nested dictionary.  Falls back to :data:`_DEFAULTS` if
    the file is absent or cannot be parsed, logging a warning either way so
    the caller can decide whether the fallback is acceptable.

    Parameters
    ----------
    config_path : str or Path, optional
        Path to the YAML config file, relative to the working directory or
        absolute.  Defaults to ``"config/model_config.yaml"``.

    Returns
    -------
    dict[str, Any]
        Nested configuration dictionary.  Top-level keys match the YAML
        sections: ``"pipeline"``, ``"model_architecture"``,
        ``"generation"``.

    Notes
    -----
    ``yaml.safe_load`` is used (never ``yaml.load``) to prevent arbitrary
    Python object deserialisation from untrusted YAML files.

    Examples
    --------
    >>> cfg = load_config()
    >>> cfg["pipeline"]["window_size"]
    5
    >>> cfg["model_architecture"]["embed_dim"]
    64
    >>> cfg["generation"]["hf_model_name"]
    'google/flan-t5-small'
    """
    path = Path(config_path)

    try:
        with path.open("r", encoding="utf-8") as fh:
            config: Dict[str, Any] = yaml.safe_load(fh)

        if not isinstance(config, dict):
            raise ValueError(
                f"Expected a YAML mapping at the top level, "
                f"got {type(config).__name__!r}."
            )

        logger.info("Configuration loaded from '%s'.", path)
        return config

    except FileNotFoundError:
        logger.warning(
            "Config file not found at '%s'. "
            "Using built-in defaults — consider creating the file for production use.",
            path,
        )
        return _DEFAULTS.copy()

    except (yaml.YAMLError, ValueError) as exc:
        logger.warning(
            "Failed to parse config file '%s': %s. Using built-in defaults.",
            path,
            exc,
        )
        return _DEFAULTS.copy()
