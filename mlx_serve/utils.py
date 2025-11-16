# Adapted from https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/utils.py
import json
import logging
from pathlib import Path
from typing import Dict, Any, Callable, Tuple, Type
import mlx.core as mx
import mlx.nn as nn
from typing import Optional
import glob

from mlx_serve.models.qwen3 import Qwen3ForCausalLM, ModelArgs

def _get_classes(config: dict) -> Tuple[Type[nn.Module], Type]:
    return Qwen3ForCausalLM, ModelArgs

def load_config(model_path: Path) -> dict:
    try:
        with open(model_path / "config.json", "r") as f:
            config = json.load(f)
    except FileNotFoundError:
        logging.error(f"Config file not found in {model_path}")
        raise
    return config


def load_model(
    model_path: Path,
    lazy: bool = False,
    strict: bool = True,
    model_config: Optional[Dict[str, Any]] = None,
    get_model_classes: Callable[[dict], Tuple[Type[nn.Module], Type]] = _get_classes,
) -> Tuple[nn.Module, dict]:
    """
    Load and initialize the model from a given path.

    Args:
        model_path (Path): The path to load the model from.
        lazy (bool): If False eval the model parameters to make sure they are
            loaded in memory before returning, otherwise they will be loaded
            when needed. Default: ``False``
        strict (bool): Whether or not to raise an exception if weights don't
            match. Default: ``True``
        model_config (dict, optional): Optional configuration parameters for the
            model. Defaults to an empty dictionary.
        get_model_classes (Callable[[dict], Tuple[Type[nn.Module], Type]], optional):
            A function that returns the model class and model args class given a config.
            Defaults to the ``_get_classes`` function.

    Returns:
        Tuple[nn.Module, dict[str, Any]]: The loaded and initialized model and config.

    Raises:
        FileNotFoundError: If the weight files (.safetensors) are not found.
        ValueError: If the model class or args class are not found or cannot be instantiated.
    """
    config = load_config(model_path)
    if model_config is not None:
        config.update(model_config)

    weight_files = glob.glob(str(model_path / "model*.safetensors"))

    if not weight_files and strict:
        logging.error(f"No safetensors found in {model_path}")
        raise FileNotFoundError(f"No safetensors found in {model_path}")

    weights = {}
    for wf in weight_files:
        weights.update(mx.load(wf))

    model_class, model_args_class = get_model_classes(config=config)

    model_args = model_args_class.from_dict(config)
    model = model_class(model_args)

    if hasattr(model, "sanitize"):
        weights = model.sanitize(weights)

    def _quantize(quantization):
        def class_predicate(p, m):
            # Handle custom per layer quantizations
            if p in config["quantization"]:
                return config["quantization"][p]
            if not hasattr(m, "to_quantized"):
                return False
            return f"{p}.scales" in weights

        nn.quantize(
            model,
            group_size=quantization["group_size"],
            bits=quantization["bits"],
            mode=quantization.get("mode", "affine"),
            class_predicate=class_predicate,
        )

    if (quantization := config.get("quantization", None)) is not None:
        _quantize(quantization)

    elif quantization_config := config.get("quantization_config", False):
        # Handle legacy quantization config
        quant_method = quantization_config["quant_method"]
        if quant_method == "mxfp4":
            quantization = {"group_size": 32, "bits": 4, "mode": "mxfp4"}
            config["quantization"] = quantization
            config["quantization_config"] = quantization
            _quantize(quantization)

    model.load_weights(list(weights.items()), strict=strict)

    if not lazy:
        mx.eval(model.parameters())

    model.eval()
    return model, config
