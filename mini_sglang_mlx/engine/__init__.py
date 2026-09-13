from .config import SPEC_ALGOS, EngineConfig
from .dflash2_engine import Dflash2Engine
from .dflash_engine import DflashEngine
from .engine import Engine, ForwardOutput
from .sample import BatchSamplingArgs
from .spec_engine import SpecEngine, SpecForwardOutput


def create_engine(config: EngineConfig) -> Engine:
    """Construct the right engine variant based on ``config.spec_algo``.

    Dispatch:

    * ``spec_algo == "dflash"``  → :class:`DflashEngine`.
    * ``spec_algo == "dflash2"`` → :class:`Dflash2Engine`.
    * ``spec_algo is None``      → base :class:`Engine` (no draft).

    ``EngineConfig.__post_init__`` already validates that
    ``draft_path`` / ``num_draft_tokens`` are set when ``spec_algo``
    is non-None, so we can dispatch directly here.
    """
    if config.spec_algo is None:
        return Engine(config)
    if config.spec_algo == "dflash":
        return DflashEngine(config)
    if config.spec_algo == "dflash2":
        return Dflash2Engine(config)
    raise ValueError(
        f"Unknown spec_algo {config.spec_algo!r}; expected one of "
        f"{SPEC_ALGOS} or None."
    )


__all__ = [
    "BatchSamplingArgs",
    "Dflash2Engine",
    "DflashEngine",
    "Engine",
    "EngineConfig",
    "ForwardOutput",
    "SPEC_ALGOS",
    "SpecEngine",
    "SpecForwardOutput",
    "create_engine",
]
