from .config import EngineConfig
from .dflash_engine import DflashEngine
from .eagle_mtp_engine import EagleMTPEngine
from .engine import Engine, ForwardOutput
from .sample import BatchSamplingArgs
from .spec_engine import SpecEngine, SpecForwardOutput


def create_engine(config: EngineConfig) -> Engine:
    """Construct the right engine variant based on ``config``.

    Selection priority (mutually exclusive in :class:`EngineConfig`):

    * ``dflash_model_path`` set → :class:`DflashEngine`.
    * ``mtp_model_path``    set → :class:`EagleMTPEngine`.
    * otherwise              → :class:`Engine`.
    """
    if config.mtp_model_path is not None and config.dflash_model_path is not None:
        raise ValueError(
            "EngineConfig.mtp_model_path and dflash_model_path are mutually "
            "exclusive — choose one spec method at a time."
        )
    if config.dflash_model_path is not None:
        return DflashEngine(config)
    if config.mtp_model_path is not None:
        return EagleMTPEngine(config)
    return Engine(config)


__all__ = [
    "BatchSamplingArgs",
    "DflashEngine",
    "EagleMTPEngine",
    "Engine",
    "EngineConfig",
    "ForwardOutput",
    "SpecEngine",
    "SpecForwardOutput",
    "create_engine",
]
