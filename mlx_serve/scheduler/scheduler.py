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

from mlx_serve.engine import (
    BatchSamplingArgs,
    ForwardOutput,
    SpecEngine,
    SpecForwardOutput,
    create_engine,
)


logger = init_logger(__name__)


# For overlap scheduling, we also need to cache some other data to avoid IMA
class ForwardInput(NamedTuple):
    batch: Batch
    sample_args: BatchSamplingArgs


ForwardData: TypeAlias = tuple[ForwardInput, ForwardOutput]


class Scheduler(SchedulerIOMixin):
    def __init__(self, config: SchedulerConfig):
        self.engine = create_engine(config)
        super().__init__(config)

        self.is_spec = isinstance(self.engine, SpecEngine)
        self.table_manager = TableManager(config.max_running_req, self.engine.page_table)
        self.is_hybrid = self.engine.is_hybrid
        if self.is_hybrid:
            from .cache import HybridCacheManager
            # Spec engines (EAGLE / DFlash) share page IDs with the
            # target via the mirrored draft KV cache, so the regular
            # HybridCacheManager + radix tree works as-is — every
            # cached page holds both target and draft K/V.
            self.cache_manager = HybridCacheManager(
                None, self.engine.num_pages, self.engine.mamba_pool,
            )
        else:
            self.cache_manager = CacheManager(
                None, self.engine.num_pages, config.cache_type,
            )

        # Spec engines reserve K extra KV pages per running req per
        # iter; bump ``DecodeManager.extra_per_req`` so the prefill
        # scheduler accounts for that peak when admitting new reqs.
        extra_per_req = self.engine.K if self.is_spec else 0  # type: ignore[attr-defined]
        self.decode_manager = DecodeManager(extra_per_req=extra_per_req)
        self.prefill_manager = PrefillManager(
            self.cache_manager, self.table_manager, self.decode_manager
        )

        if self.is_spec:
            assert isinstance(self.engine, SpecEngine)
            self.engine.set_cache_manager(self.cache_manager)

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
            is_eos = next_token == self.eos_token_id
            if not req.sampling_params.ignore_eos:
                finished |= is_eos
            reply.append(DetokenizeMsg(uid=req.uid, next_token=next_token, finished=finished))

            if finished:
                reason = "eos" if is_eos else "max_tokens"
                logger.info(
                    "[Done] uid=%d  reason=%s  total_tokens=%d",
                    req.uid, reason, req.device_len,
                )
                self.finished_reqs.add(req)
                self.decode_manager.remove_req(req)

        for req in self.finished_reqs:
            self.table_manager.free(req.table_idx)
            self.cache_manager.free_and_cache_finished_req(
                req.cache_handle,
                req.input_ids[: req.cached_len],
                self.page_table[req.table_idx, : req.cached_len],
                mamba_slot=req.mamba_slot,
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
            logger.info(
                "[NewReq] uid=%d  input_len=%d  max_tokens=%d",
                msg.uid, len(msg.input_ids), msg.sampling_params.max_tokens,
            )
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
        req_info = ", ".join(
            f"uid={r.uid}(cached={r.cached_len} dev={r.device_len} ext={r.extend_len})"
            for r in batch.reqs
        )
        logger.debug(
            "[Batch] phase=%s  reqs=%d  [%s]", batch.phase, len(batch.reqs), req_info,
        )
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
        if self.engine.gdn_backend is not None:
            self.engine.gdn_backend.prepare_batch(batch)
        return ForwardInput(batch=batch, sample_args=self.engine.sampler.prepare(batch))

    def _select_batch(self) -> Batch | None:
        """Choose the next batch (prefill preferred over decode)."""
        # TODO: support other policies: e.g. DECODE first
        return (
            self.prefill_manager.schedule_next_batch(self.prefill_budget)
            or self.decode_manager.schedule_next_batch()
        )

    def _schedule_next_batch(self) -> ForwardInput | None:
        batch = self._select_batch()
        return self._prepare_batch(batch) if batch else None

    def _forward(self, forward_input: ForwardInput) -> ForwardOutput:
        batch, sample_args = forward_input.batch, forward_input.sample_args
        forward_output = self.engine.forward_batch(batch, sample_args)
        mx.eval(forward_output.next_tokens)
        self.decode_manager.filter_reqs(forward_input.batch.reqs)
        return forward_output

    # ════════════════════════════════════════════════════════════════
    # Spec-engine output processing (EAGLE / DFlash)
    # ════════════════════════════════════════════════════════════════

    def _process_spec_output(
        self, batch: Batch, output: SpecForwardOutput,
    ) -> None:
        """Iterate ``output.accepted_tokens`` per req, handling EOS / max_tokens.

        The engine has already updated each req's ``input_ids`` /
        ``cached_len`` / ``device_len`` and stored the next ``pending_*``
        fields.  Our job is to (a) emit detokenizer messages for the new
        tokens, truncating to respect max_tokens / EOS, and (b) mark and
        clean up reqs that should stop.
        """
        # One eval so the per-req ``.tolist()`` calls below are cheap.
        mx.eval(*output.accepted_tokens)

        reply: List[DetokenizeMsg] = []
        for i, req in enumerate(batch.reqs):
            if req in self.finished_reqs or isinstance(req, ChunkedReq):
                continue

            accepted_list: List[int] = output.accepted_tokens[i].tolist()

            # Spec engines can over-commit by up to K tokens in a single
            # iter (verify accepted more than ``max_tokens -
            # generated_so_far``).  Truncate the user-visible report so
            # total generated tokens never exceed ``max_tokens``.
            # Internal req state may still carry the extras; they get
            # freed on req cleanup.
            overshoot = max(0, req.device_len - req.max_device_len)
            if overshoot >= len(accepted_list):
                accepted_list = []
            elif overshoot > 0:
                accepted_list = accepted_list[:-overshoot]

            finished = False
            finish_reason: str | None = None
            for j, tok in enumerate(accepted_list):
                is_eos = (
                    tok == self.eos_token_id
                    and not req.sampling_params.ignore_eos
                )
                is_last = j == len(accepted_list) - 1
                # Last token of the iter ends the request if we overshot
                # or used up the per-req budget (post-engine state).
                hit_budget = is_last and (overshoot > 0 or not req.can_decode())
                this_finished = is_eos or hit_budget
                reply.append(
                    DetokenizeMsg(
                        uid=req.uid, next_token=tok, finished=this_finished,
                    )
                )
                if this_finished:
                    finished = True
                    finish_reason = "eos" if is_eos else "max_tokens"
                    break

            # Defensive: if the iter produced nothing visible (full
            # overshoot truncation) but the req is out of budget, still
            # mark it finished so cleanup happens.
            if not finished and (overshoot > 0 or not req.can_decode()):
                finished = True
                finish_reason = "max_tokens"

            if finished:
                input_len = req.max_device_len - req.output_len
                logger.info(
                    "[Done] uid=%d  reason=%s  total_tokens=%d",
                    req.uid, finish_reason, req.cached_len - input_len,
                )
                self.finished_reqs.add(req)
                self.decode_manager.remove_req(req)

        # Resource cleanup for finished reqs.  Pages are SHARED between
        # target and draft (mirrored), so a single free_and_cache call
        # releases both — radix insertion stores the page IDs that hold
        # both target and draft K/V together.
        for req in self.finished_reqs:
            self.table_manager.free(req.table_idx)
            self.cache_manager.free_and_cache_finished_req(
                req.cache_handle,
                req.input_ids[: req.cached_len],
                self.page_table[req.table_idx, : req.cached_len],
                mamba_slot=req.mamba_slot,
            )

        self.finished_reqs.clear()
        self.send_result(reply)

    def run_when_idle(self) -> None:
        """Called when the scheduler is idle to perform background tasks."""
        logger.info("Scheduler is idle, waiting for new reqs...")
        self.cache_manager.check_integrity()

    def normal_loop(self) -> None:
        blocking = not (self.prefill_manager.runnable or self.decode_manager.runnable)
        for msg in self.receive_msg(blocking=blocking):
            self._process_one_msg(msg)

        if self.is_spec:
            self._spec_loop_step()
        else:
            self._non_spec_loop_step()

    def _non_spec_loop_step(self) -> None:
        forward_input = self._schedule_next_batch()
        ongoing_data = None
        if forward_input is not None:
            ongoing_data = (forward_input, self._forward(forward_input))

        self._process_last_data(ongoing_data)

    def _spec_loop_step(self) -> None:
        assert isinstance(self.engine, SpecEngine)
        batch = self._select_batch()
        if batch is None:
            return
        # The engine handles all KV / mamba alloc + free + sub-forwards
        # and returns variable-length accepted tokens per req.
        spec_output = self.engine.run_iter(batch)
        self.decode_manager.filter_reqs(batch.reqs)
        self._process_spec_output(batch, spec_output)

    def run_forever(self) -> NoReturn:
        while True:
            self.normal_loop()

    def shutdown(self) -> None:
        self.sync_all_ranks()
        self.engine.shutdown()
