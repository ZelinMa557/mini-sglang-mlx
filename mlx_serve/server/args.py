from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List

import mlx.core as mx
from mlx_serve.engine import SPEC_ALGOS
from mlx_serve.scheduler import SchedulerConfig
from mlx_serve.utils import cached_load_hf_config, init_logger, resolve_repo

logger = init_logger(__name__)


@dataclass(frozen=True)
class ServerArgs(SchedulerConfig):
    server_host: str = "127.0.0.1"
    server_port: int = 1919
    num_tokenizer: int = 0
    silent_output: bool = False
    use_modelscope: bool = False
    enable_thinking: bool = False

    @property
    def share_tokenizer(self) -> bool:
        return self.num_tokenizer == 0

    @property
    def zmq_frontend_addr(self) -> str:
        return "ipc:///tmp/mlx_serve_3" + self._unique_suffix

    @property
    def zmq_tokenizer_addr(self) -> str:
        if self.share_tokenizer:
            return self.zmq_detokenizer_addr
        result = "ipc:///tmp/mlx_serve_4" + self._unique_suffix
        assert result != self.zmq_detokenizer_addr
        return result

    @property
    def tokenizer_create_addr(self) -> bool:
        return self.share_tokenizer

    @property
    def backend_create_detokenizer_link(self) -> bool:
        return not self.share_tokenizer

    @property
    def frontend_create_tokenizer_link(self) -> bool:
        return not self.share_tokenizer

    @property
    def distributed_addr(self) -> str:
        return f"tcp://127.0.0.1:{self.server_port + 1}"


# ─────────────────────────────────────────────────────────────────────
# CLI parsing
# ─────────────────────────────────────────────────────────────────────


# CLI accepts the conventional capitalisations; internally we
# normalise to lowercase to match :data:`mlx_serve.engine.SPEC_ALGOS`.
_SPEC_ALGO_CHOICES = ["none", "mtp", "dflash"]
_SPEC_ALGO_DISPLAY = ["None", "MTP", "DFlash"]


def _normalise_spec_algo(raw: str | None) -> str | None:
    if raw is None:
        return None
    low = raw.lower()
    if low in ("none", ""):
        return None
    if low not in SPEC_ALGOS:
        raise argparse.ArgumentTypeError(
            f"--spec-algo must be one of {_SPEC_ALGO_DISPLAY} "
            f"(case-insensitive), got {raw!r}"
        )
    return low


DTYPE_MAP = {
    "float16": mx.float16,
    "bfloat16": mx.bfloat16,
    "float32": mx.float32,
}


def _resolve_dtype(dtype_str: str, model_path: str):
    if dtype_str != "auto":
        return DTYPE_MAP[dtype_str]
    hf_cfg = cached_load_hf_config(model_path)
    dtype_or_str = getattr(hf_cfg, "torch_dtype", None)
    if dtype_or_str is None:
        tc = getattr(hf_cfg, "text_config", None)
        if tc is not None:
            dtype_or_str = getattr(tc, "dtype", None)
    dtype_name = str(dtype_or_str).lower() if dtype_or_str is not None else ""
    if "bfloat16" in dtype_name:
        return mx.bfloat16
    if "float16" in dtype_name:
        return mx.float16
    return mx.float32


def parse_args(args: List[str]) -> ServerArgs:
    """Parse command-line arguments and return a :class:`ServerArgs`.

    Speculative decoding goes through a single set of flags
    (``--spec-algo``, ``--draft-path``, ``--num-draft-tokens``)
    regardless of algorithm; remote-repo paths are resolved through
    :func:`mlx_serve.utils.resolve_repo` so HF and ModelScope share
    a single code path.
    """
    from mlx_serve.kvcache import SUPPORTED_CACHE_MANAGER

    parser = argparse.ArgumentParser(description="MLX-Serve Server Arguments")

    # ── Model & runtime ────────────────────────────────────────────
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Target model weights — local folder, HuggingFace repo ID, or "
             "ModelScope repo ID (with --use-modelscope).",
    )
    parser.add_argument(
        "--use-modelscope",
        action="store_true",
        help="Resolve remote repo IDs (target + DFlash draft) via "
             "ModelScope instead of HuggingFace.  Useful in regions "
             "where HF access is slow.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
        help="Activation / weight dtype.  'auto' = FP16 for FP32/FP16 "
             "checkpoints, BF16 for BF16 checkpoints.",
    )

    # ── Capacity / scheduling ─────────────────────────────────────
    parser.add_argument(
        "--max-running-requests",
        type=int,
        dest="max_running_req",
        default=ServerArgs.max_running_req,
        help="Maximum number of concurrent running requests.",
    )
    parser.add_argument(
        "--max-seq-len-override",
        type=int,
        default=ServerArgs.max_seq_len_override,
        help="Override the model's max position embeddings.",
    )
    parser.add_argument(
        "--kv-cache-gb",
        type=float,
        dest="kv_cache_gb",
        default=ServerArgs.kv_cache_gb,
        help="KV cache size in GB.  Default: auto-determined from the "
             "model config.",
    )
    parser.add_argument(
        "--max-prefill-length",
        "--max-extend-length",
        type=int,
        dest="max_extend_tokens",
        default=ServerArgs.max_extend_tokens,
        help="Chunk prefill maximum chunk size in tokens.",
    )
    parser.add_argument(
        "--attention-backend",
        "--attn",
        type=str,
        default=ServerArgs.attention_backend,
        help="Attention backend name.",
    )
    parser.add_argument(
        "--cache-type",
        type=str,
        default=ServerArgs.cache_type,
        choices=SUPPORTED_CACHE_MANAGER.supported_names(),
        help="KV cache management strategy.",
    )
    parser.add_argument(
        "--num-mamba-slots",
        type=int,
        default=ServerArgs.num_mamba_slots,
        help="Mamba state pool size (number of slots).  When unset, "
             "auto-computed as max_running_req * (2 + spec_extra).  "
             "Exposed for hybrid models where the default heuristic "
             "may under-allocate; only meaningful for Qwen3.5-style "
             "GDN-hybrid targets.",
    )

    # ── Speculative decoding (unified) ────────────────────────────
    parser.add_argument(
        "--spec-algo",
        # argparse runs ``type`` before ``choices``; ``str.lower``
        # makes the CLI flag case-insensitive while keeping the
        # canonical capitalisations ("MTP", "DFlash") in --help.
        type=str.lower,
        default=None,
        choices=_SPEC_ALGO_CHOICES,
        metavar="{None,MTP,DFlash}",
        help="Speculative-decoding algorithm.  When set, --draft-path "
             "and --num-draft-tokens are required.  Default: no spec "
             "decoding.",
    )
    parser.add_argument(
        "--draft-path",
        type=str,
        default=None,
        help="Path or repo ID of the speculative draft.  For "
             "--spec-algo=MTP this is the MTP layer .safetensors "
             "file (use scripts/download_qwen3_5_mtp_layer.py to "
             "extract it from a Qwen3.5 checkpoint).  For "
             "--spec-algo=DFlash this is the DFlash draft model "
             "directory or a HF/ModelScope repo ID.",
    )
    parser.add_argument(
        "--num-draft-tokens",
        type=int,
        default=0,
        help="Drafts per spec iter (K).  For MTP: number of draft "
             "forwards per decode iter.  For DFlash: block_size - 1 "
             "(must match the draft checkpoint's training-time "
             "block_size minus one).",
    )

    # ── Serving frontend ──────────────────────────────────────────
    parser.add_argument(
        "--host",
        type=str,
        dest="server_host",
        default=ServerArgs.server_host,
        help="Host address for the HTTP server.",
    )
    parser.add_argument(
        "--port",
        type=int,
        dest="server_port",
        default=ServerArgs.server_port,
        help="Port for the HTTP server to listen on.",
    )
    parser.add_argument(
        "--num-tokenizer",
        "--tokenizer-count",
        type=int,
        default=ServerArgs.num_tokenizer,
        help="Number of tokenizer workers.  0 means the tokenizer is "
             "shared with the detokenizer.",
    )
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="Enable thinking / reasoning mode in the chat template.",
    )

    # ── Parse + validate ──────────────────────────────────────────
    kwargs = parser.parse_args(args).__dict__.copy()

    # Normalise spec_algo string.
    kwargs["spec_algo"] = _normalise_spec_algo(kwargs.get("spec_algo"))

    use_modelscope = bool(kwargs.get("use_modelscope", False))

    # ── Resolve target model path (local / HF / ModelScope) ──────
    kwargs["model_path"] = resolve_repo(
        kwargs["model_path"], use_modelscope=use_modelscope,
    )

    # ── Validate / resolve spec args ──────────────────────────────
    spec_algo = kwargs["spec_algo"]
    draft_path = kwargs.get("draft_path")
    num_draft = kwargs.get("num_draft_tokens", 0)

    if spec_algo is None:
        # When spec decoding is disabled, ignore (but warn about)
        # any draft args the user passed by mistake.
        if draft_path is not None:
            logger.warning(
                "--draft-path is ignored because --spec-algo is unset.",
            )
        if num_draft != 0:
            logger.warning(
                "--num-draft-tokens is ignored because --spec-algo is unset.",
            )
        kwargs["draft_path"] = None
        kwargs["num_draft_tokens"] = 0
    else:
        if draft_path is None:
            parser.error(
                f"--spec-algo={spec_algo.upper()} requires --draft-path"
            )
        if num_draft < 1:
            parser.error(
                f"--spec-algo={spec_algo.upper()} requires "
                f"--num-draft-tokens >= 1 (got {num_draft})"
            )
        if spec_algo == "mtp":
            # MTP draft is a single weights file pre-extracted by
            # scripts/download_qwen3_5_mtp_layer.py.  We don't run
            # remote resolution here: the script is the canonical
            # producer of this file and pulling a whole MTP repo
            # over the network would defeat the point.  Just
            # expand ``~`` and check existence so we fail fast.
            expanded = Path(draft_path).expanduser()
            if not expanded.exists():
                parser.error(
                    f"--draft-path {draft_path!r} does not exist; for "
                    "--spec-algo=MTP this should be a .safetensors file "
                    "produced by scripts/download_qwen3_5_mtp_layer.py."
                )
            kwargs["draft_path"] = str(expanded)
        else:  # dflash
            kwargs["draft_path"] = resolve_repo(
                draft_path, use_modelscope=use_modelscope,
            )

    # ── dtype resolution (needs the resolved local model path) ───
    kwargs["dtype"] = _resolve_dtype(kwargs["dtype"], kwargs["model_path"])

    result = ServerArgs(**kwargs)
    logger.info(f"Parsed arguments:\n{result}")
    return result
