from __future__ import annotations

import glob
import importlib
import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple, Type, Union

import mlx.core as mx
import mlx.nn as nn

from mini_sglang_mlx.utils import resolve_repo

from .base import BaseModelArgs

logger = logging.getLogger(__name__)

MODEL_REMAPPING = {
    "mistral": "llama",
}


def _get_classes(config: dict) -> Tuple[Type[nn.Module], Type]:
    model_type = config["model_type"]
    model_type = MODEL_REMAPPING.get(model_type, model_type)
    try:
        arch = importlib.import_module(f"mini_sglang_mlx.models.{model_type}")
    except ImportError:
        raise ValueError(
            f"Model type {model_type!r} not supported. "
            f"No module mini_sglang_mlx.models.{model_type} found."
        )
    return arch.Model, arch.ModelArgs


def load_config(model_path: Path) -> dict:
    with open(model_path / "config.json", "r") as f:
        config = json.load(f)

    generation_config_file = model_path / "generation_config.json"
    if generation_config_file.exists():
        try:
            with open(generation_config_file, "r") as f:
                generation_config = json.load(f)
        except json.JSONDecodeError:
            generation_config = {}
        if eos_token_id := generation_config.get("eos_token_id", False):
            config["eos_token_id"] = eos_token_id

    return config


def load_model(
    model_path: Union[str, Path],
    lazy: bool = False,
    model_config: Optional[Dict[str, Any]] = None,
) -> Tuple[nn.Module, dict]:
    """Load and initialize a model from a local path or HuggingFace repo.

    Follows the mlx-lm convention: each model module exposes ``Model`` and
    ``ModelArgs``.  Weights are loaded from ``*.safetensors`` files, and
    optional mlx-native quantization is applied based on ``config.json``.

    Args:
        model_path: Local directory or HuggingFace repo ID.  For
            ModelScope-hosted models the caller is responsible for
            pre-resolving the path (use
            :func:`mini_sglang_mlx.utils.resolve_repo` with
            ``use_modelscope=True``); this loader only knows about
            local paths and HuggingFace.
        lazy: If True, defer parameter evaluation until first use.
        model_config: Extra config overrides merged into the loaded config.

    Returns:
        (model, config) tuple.
    """
    model_path = Path(resolve_repo(str(model_path), use_modelscope=False))
    config = load_config(model_path)
    if model_config is not None:
        config.update(model_config)

    weight_files = sorted(glob.glob(str(model_path / "model*.safetensors")))
    if not weight_files:
        raise FileNotFoundError(f"No safetensors found in {model_path}")

    weights: Dict[str, mx.array] = {}
    for wf in weight_files:
        weights.update(mx.load(wf))

    model_class, model_args_class = _get_classes(config)
    model_args = model_args_class.from_dict(config)
    model = model_class(model_args)

    if hasattr(model, "sanitize"):
        weights = model.sanitize(weights)

    if "quantization_config" not in config:
        text_config = config.get("text_config", {})
        if "quantization_config" in text_config:
            config["quantization_config"] = text_config["quantization_config"]

    if (quantization := config.get("quantization", None)) is not None:
        _apply_quantization(model, config, weights, quantization)
    elif quantization_config := config.get("quantization_config", False):
        quant_method = quantization_config.get("quant_method", "")
        if quant_method in ("mxfp4",):
            quantization = {"group_size": 32, "bits": 4, "mode": "mxfp4"}
            config["quantization"] = quantization
            _apply_quantization(model, config, weights, quantization)
        elif quant_method in ("compressed-tensors",):
            quantization = {"group_size": 32, "bits": 4, "mode": "affine"}
            config["quantization"] = quantization
            _apply_quantization(model, config, weights, quantization)
        elif quant_method:
            logger.warning(f"Unsupported quant_method {quant_method!r}, skipping quantization.")

    model.eval()
    model.load_weights(list(weights.items()), strict=True)

    if not lazy:
        mx.eval(model.parameters())

    return model, config


def _apply_quantization(
    model: nn.Module,
    config: dict,
    weights: Dict[str, mx.array],
    quantization: dict,
) -> None:
    def class_predicate(p, m):
        if isinstance(config.get("quantization"), dict) and p in config["quantization"]:
            return config["quantization"][p]
        if not hasattr(m, "to_quantized"):
            return False
        return f"{p}.scales" in weights

    nn.quantize(
        model,
        group_size=quantization["group_size"],
        bits=quantization["bits"],
        # mode=quantization.get("mode", "affine"),
        class_predicate=class_predicate,
    )


def create_model(
    model_path: Union[str, Path],
    lazy: bool = False,
    model_config: Optional[Dict[str, Any]] = None,
) -> Tuple[nn.Module, dict]:
    """Convenience alias for :func:`load_model`."""
    return load_model(model_path, lazy=lazy, model_config=model_config)


__all__ = [
    "BaseModelArgs",
    "create_model",
    "load_model",
    "load_config",
]
