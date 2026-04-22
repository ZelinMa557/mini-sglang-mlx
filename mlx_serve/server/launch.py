from __future__ import annotations

import logging
import multiprocessing as mp
import sys
from dataclasses import fields
from typing import TYPE_CHECKING

from mlx_serve.utils import init_logger

if TYPE_CHECKING:
    from .args import ServerArgs


def _dtype_to_name(dtype) -> str:
    import mlx.core as mx

    if dtype == mx.float16:
        return "float16"
    if dtype == mx.bfloat16:
        return "bfloat16"
    if dtype == mx.float32:
        return "float32"
    raise ValueError(f"Unsupported dtype for multiprocessing: {dtype}")


def _name_to_dtype(name: str):
    import mlx.core as mx

    mapping = {
        "float16": mx.float16,
        "bfloat16": mx.bfloat16,
        "float32": mx.float32,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported dtype name: {name}")
    return mapping[name]


def _serialize_server_args(args: "ServerArgs") -> dict:
    # NOTE: dataclasses.asdict() uses deepcopy internally, which fails on mlx.core.Dtype.
    payload = {f.name: getattr(args, f.name) for f in fields(args)}
    payload["dtype"] = _dtype_to_name(args.dtype)
    return payload


def _run_scheduler(args_payload: dict, ack_queue: mp.Queue[str]) -> None:
    from .args import ServerArgs
    from mlx_serve.scheduler import Scheduler

    args_payload = dict(args_payload)
    args_payload["dtype"] = _name_to_dtype(args_payload["dtype"])
    args = ServerArgs(**args_payload)

    scheduler = Scheduler(args)
    scheduler.sync_all_ranks()

    ack_queue.put("Scheduler is ready")

    if args.silent_output:
        logging.disable(logging.INFO)

    try:
        scheduler.run_forever()
    except KeyboardInterrupt:
        logger = init_logger(__name__)
        print()  # for a clean newline after ^C
        logger.info("Scheduler exiting gracefully...")
        scheduler.shutdown()


def launch_server(run_shell: bool = False) -> None:
    from .api_server import run_api_server
    from .args import parse_args

    server_args, run_shell = parse_args(sys.argv[1:], run_shell)
    logger = init_logger(__name__, "initializer")

    def start_subprocess() -> None:
        import multiprocessing as mp

        from mlx_serve.tokenizer import tokenize_worker

        mp.set_start_method("spawn", force=True)

        # a multiprocessing queue to receive ack from subprocesses
        # so that we can guarantee all subprocesses are ready
        ack_queue: mp.Queue[str] = mp.Queue()

        mp.Process(
            target=_run_scheduler,
            args=(_serialize_server_args(server_args), ack_queue),
            daemon=False,
            name="mlx-serve-scheduler",
        ).start()

        num_tokenizers = server_args.num_tokenizer
        # DeTokenizer, only 1
        mp.Process(
            target=tokenize_worker,
            kwargs={
                "tokenizer_path": server_args.model_path,
                "addr": server_args.zmq_detokenizer_addr,
                "backend_addr": server_args.zmq_backend_addr,
                "frontend_addr": server_args.zmq_frontend_addr,
                "local_bs": 1,
                "create": server_args.tokenizer_create_addr,
                "tokenizer_id": num_tokenizers,
                "enable_thinking": server_args.enable_thinking,
                "ack_queue": ack_queue,
            },
            daemon=False,
            name="mlx-serve-detokenizer-0",
        ).start()
        for i in range(num_tokenizers):
            mp.Process(
                target=tokenize_worker,
                kwargs={
                    "tokenizer_path": server_args.model_path,
                    "addr": server_args.zmq_tokenizer_addr,
                    "backend_addr": server_args.zmq_backend_addr,
                    "frontend_addr": server_args.zmq_frontend_addr,
                    "local_bs": 1,
                    "create": server_args.tokenizer_create_addr,
                    "tokenizer_id": i,
                    "enable_thinking": server_args.enable_thinking,
                    "ack_queue": ack_queue,
                },
                daemon=False,
                name=f"mlx-serve-tokenizer-{i}",
            ).start()

        # Wait for acknowledgments from all worker processes:
        # - world_size schedulers (but only primary rank sends ack)
        # - num_tokenizers tokenizers
        # - 1 detokenizer
        # Total acks expected: 1 + num_tokenizers + 1 = num_tokenizers + 2
        for _ in range(num_tokenizers + 2):
            logger.info(ack_queue.get())

    run_api_server(server_args, start_subprocess)


if __name__ == "__main__":
    launch_server()
