from __future__ import annotations

from typing import TYPE_CHECKING, List, NamedTuple, NoReturn, Set, TypeAlias

import mlx.core as mx
from mlx_serve.core import Batch, Req
from mlx_serve.message import (
    BaseBackendMsg,
    BatchBackendMsg,
    DetokenizeMsg,
    ExitMsg,
    UserMsg,
)
from mlx_serve.utils import init_logger
from transformers import AutoTokenizer

from .cache import CacheManager
from .config import SchedulerConfig
from .decode import DecodeManager
from .io import SchedulerIOMixin
from .prefill import ChunkedReq, PrefillManager
from .table import TableManager

from mlx_serve.engine import BatchSamplingArgs, ForwardOutput


logger = init_logger(__name__)


# For overlap scheduling, we also need to cache some other data to avoid IMA
class ForwardInput(NamedTuple):
    batch: Batch
    sample_args: BatchSamplingArgs


ForwardData: TypeAlias = tuple[ForwardInput, ForwardOutput]


class Scheduler(SchedulerIOMixin):
    def __init__(self, config: SchedulerConfig):
        from mlx_serve.engine import Engine

        self.engine = Engine(config)
        super().__init__(config)

        self.table_manager = TableManager(config.max_running_req, self.engine.page_table)
        self.cache_manager = CacheManager(None, self.engine.num_pages, config.cache_type)
        self.decode_manager = DecodeManager()
        self.prefill_manager = PrefillManager(
            self.cache_manager, self.table_manager, self.decode_manager
        )

        self.finished_reqs: Set[Req] = set()
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_path)
        self.eos_token_id = self.tokenizer.eos_token_id
        self.page_table = self.engine.page_table
        self.prefill_budget = config.max_extend_tokens

    def _process_last_data(self, last_data: ForwardData | None) -> None:
        if last_data is None:
            return
        batch, output = last_data[0].batch, last_data[1]
        next_tokens = output.next_tokens
        reply: List[DetokenizeMsg] = []

        for i, req in enumerate(batch.reqs):
            if req in self.finished_reqs or isinstance(req, ChunkedReq):
                continue

            next_token = int(next_tokens[i].item())
            req.append_host(mx.array([next_token], dtype=mx.int32))
            finished = not req.can_decode()
            if not req.sampling_params.ignore_eos:
                finished |= next_token == self.eos_token_id
            reply.append(DetokenizeMsg(uid=req.uid, next_token=next_token, finished=finished))

            # free resources if the req is finished and not ongoing
            if finished:
                self.finished_reqs.add(req)
                self.decode_manager.remove_req(req)
                logger.debug("Request %s is finished", req)

        for req in self.finished_reqs:
            self.table_manager.free(req.table_idx)
            self.cache_manager.free_and_cache_finished_req(
                req.cache_handle,
                req.input_ids[: req.cached_len],
                self.page_table[req.table_idx, : req.cached_len],
            )

        self.finished_reqs.clear()
        self.send_result(reply)

    def _process_one_msg(self, msg: BaseBackendMsg) -> None:
        if isinstance(msg, BatchBackendMsg):
            for msg in msg.data:
                self._process_one_msg(msg)
        elif isinstance(msg, ExitMsg):
            raise KeyboardInterrupt
        elif isinstance(msg, UserMsg):
            logger.debug("Received user msg: %s", msg)
            input_len, max_seq_len = len(msg.input_ids), self.engine.max_seq_len
            max_output_len = max_seq_len - input_len
            if max_output_len <= 0:
                return logger.warning(
                    f"Input sequence length {input_len} exceeds {max_seq_len}, "
                    f"request {msg.uid} is dropped."
                )
            if msg.sampling_params.max_tokens > max_output_len:
                msg.sampling_params.max_tokens = max_output_len
                logger.warning(
                    f"Adjust max_tokens to {max_output_len} for request {msg.uid}."
                )
            self.prefill_manager.add_one_req(msg)
        else:
            logger.error(f"Unknown message type: {type(msg)}")
            raise NotImplementedError

    def _prepare_batch(self, batch: Batch) -> ForwardInput:
        needed_size = sum(r.extend_len for r in batch.reqs)
        batch.out_loc = self.cache_manager.allocate(needed_size)
        batch.padded_reqs = batch.reqs
        assert all(r.device_len < self.engine.max_seq_len for r in batch.reqs)

        offset = 0
        input_chunks: List[mx.array] = []
        for req in batch.padded_reqs:
            extend_len = req.extend_len
            if extend_len > 0:
                self.page_table[req.table_idx, req.cached_len : req.device_len] = batch.out_loc[
                    offset : offset + extend_len
                ]
                input_chunks.append(req.input_ids[req.cached_len : req.device_len])
                offset += extend_len

        batch.input_ids = mx.concatenate(input_chunks) if input_chunks else mx.array([], dtype=mx.int32)
        self.engine.attn_backend.prepare_metadata(batch)
        return ForwardInput(batch=batch, sample_args=self.engine.sampler.prepare(batch))

    def _schedule_next_batch(self) -> ForwardInput | None:
        # TODO: support other policies: e.g. DECODE first
        batch = (
            self.prefill_manager.schedule_next_batch(self.prefill_budget)
            or self.decode_manager.schedule_next_batch()
        )
        return self._prepare_batch(batch) if batch else None

    def _forward(self, forward_input: ForwardInput) -> ForwardOutput:
        batch, sample_args = forward_input.batch, forward_input.sample_args
        forward_output = self.engine.forward_batch(batch, sample_args)
        mx.eval(forward_output.next_tokens)
        self.decode_manager.filter_reqs(forward_input.batch.reqs)
        return forward_output

    def run_when_idle(self) -> None:
        """Called when the scheduler is idle to perform background tasks."""
        logger.info("Scheduler is idle, waiting for new reqs...")
        self.cache_manager.check_integrity()

    def normal_loop(self) -> None:
        blocking = not (self.prefill_manager.runnable or self.decode_manager.runnable)
        for msg in self.receive_msg(blocking=blocking):
            self._process_one_msg(msg)

        forward_input = self._schedule_next_batch()
        ongoing_data = None
        if forward_input is not None:
            ongoing_data = (forward_input, self._forward(forward_input))

        self._process_last_data(ongoing_data)

    def run_forever(self) -> NoReturn:
        while True:
            self.normal_loop()

    def shutdown(self) -> None:
        self.sync_all_ranks()
        self.engine.shutdown()
