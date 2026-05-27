from .config import EngineConfig
from .eagle_mtp_engine import EagleMTPEngine, SpecForwardOutput
from .engine import Engine, ForwardOutput
from .sample import BatchSamplingArgs


def create_engine(config: EngineConfig) -> Engine:
    """Construct the right engine variant based on ``config``.

    ``EagleMTPEngine`` is selected when a draft model path is provided;
    otherwise the standard single-model :class:`Engine` is used.
    """
    if config.mtp_model_path is not None:
        return EagleMTPEngine(config)
    return Engine(config)


__all__ = [
    "BatchSamplingArgs",
    "EagleMTPEngine",
    "Engine",
    "EngineConfig",
    "ForwardOutput",
    "SpecForwardOutput",
    "create_engine",
]
