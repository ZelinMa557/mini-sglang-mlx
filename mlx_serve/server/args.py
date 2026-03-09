from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import List, Tuple

import mlx.core as mx
from mlx_serve.scheduler import SchedulerConfig
from mlx_serve.utils import cached_load_hf_config, init_logger


@dataclass(frozen=True)
class ServerArgs(SchedulerConfig):
    server_host: str = "127.0.0.1"
    server_port: int = 1919
    num_tokenizer: int = 0
    silent_output: bool = False

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


def parse_args(args: List[str], run_shell: bool = False) -> Tuple[ServerArgs, bool]:
    """
    Parse command line arguments and return an EngineConfig.

    Args:
        args: Command line arguments (e.g., sys.argv[1:])

    Returns:
        EngineConfig instance with parsed arguments
    """
    from mlx_serve.kvcache import SUPPORTED_CACHE_MANAGER

    parser = argparse.ArgumentParser(description="MLX-Serve Server Arguments")

    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="The path of the model weights. This can be a local folder or a Hugging Face repo ID.",
    )

    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
        help="Data type for model weights and activations. 'auto' will use FP16 for FP32/FP16 models and BF16 for BF16 models.",
    )

    parser.add_argument(
        "--tensor-parallel-size",
        "--tp-size",
        type=int,
        default=1,
        help="Tensor parallelism size (currently only 1 is supported).",
    )

    parser.add_argument(
        "--max-running-requests",
        type=int,
        dest="max_running_req",
        default=ServerArgs.max_running_req,
        help="The maximum number of running requests.",
    )

    parser.add_argument(
        "--max-seq-len-override",
        type=int,
        default=ServerArgs.max_seq_len_override,
        help="The maximum sequence length override.",
    )

    parser.add_argument(
        "--kv-cache-gb",
        type=float,
        dest="kv_cache_gb",
        default=ServerArgs.kv_cache_gb,
        help="Size of the KV cache in GB. If not set, auto-determined from model config.",
    )

    parser.add_argument(
        "--host",
        type=str,
        dest="server_host",
        default=ServerArgs.server_host,
        help="The host address for the server.",
    )

    parser.add_argument(
        "--port",
        type=int,
        dest="server_port",
        default=ServerArgs.server_port,
        help="The port number for the server to listen on.",
    )

    parser.add_argument(
        "--num-tokenizer",
        "--tokenizer-count",
        type=int,
        default=ServerArgs.num_tokenizer,
        help="The number of tokenizer processes to launch. 0 means the tokenizer is shared with the detokenizer.",
    )

    parser.add_argument(
        "--max-prefill-length",
        "--max-extend-length",
        type=int,
        dest="max_extend_tokens",
        default=ServerArgs.max_extend_tokens,
        help="Chunk Prefill maximum chunk size in tokens.",
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
        help="The KV cache management strategy.",
    )

    parser.add_argument(
        "--shell-mode",
        action="store_true",
        help="Run the server in shell mode.",
    )

    # Parse arguments
    kwargs = parser.parse_args(args).__dict__.copy()

    # resolve some arguments
    run_shell |= kwargs.pop("shell_mode")
    if run_shell:
        kwargs["max_running_req"] = 1
        kwargs["silent_output"] = True

    if kwargs["model_path"].startswith("~"):
        kwargs["model_path"] = os.path.expanduser(kwargs["model_path"])

    DTYPE_MAP = {
        "float16": mx.float16,
        "bfloat16": mx.bfloat16,
        "float32": mx.float32,
    }
    if (dtype_str := kwargs["dtype"]) != "auto":
        kwargs["dtype"] = DTYPE_MAP[dtype_str]
    else:
        hf_cfg = cached_load_hf_config(kwargs["model_path"])
        dtype_or_str = getattr(hf_cfg, "torch_dtype", None)
        if dtype_or_str is None:
            tc = getattr(hf_cfg, "text_config", None)
            if tc is not None:
                dtype_or_str = getattr(tc, "dtype", None)
        dtype_name = str(dtype_or_str).lower() if dtype_or_str is not None else ""
        if "bfloat16" in dtype_name:
            kwargs["dtype"] = mx.bfloat16
        elif "float16" in dtype_name:
            kwargs["dtype"] = mx.float16
        else:
            kwargs["dtype"] = mx.float32

    tp_size = kwargs["tensor_parallel_size"]
    if tp_size != 1:
        logger = init_logger(__name__)
        logger.warning("tensor-parallel-size=%s is not supported yet, fallback to 1.", tp_size)
    del kwargs["tensor_parallel_size"]

    result = ServerArgs(**kwargs)
    logger = init_logger(__name__)
    logger.info(f"Parsed arguments:\n{result}")
    return result, run_shell
