from __future__ import annotations

import asyncio
import json
import sys
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Literal, Optional, Tuple

import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from mlx_serve.core import SamplingParams
from mlx_serve.env import ENV
from mlx_serve.message import (
    BaseFrontendMsg,
    BaseTokenizerMsg,
    BatchFrontendMsg,
    TokenizeMsg,
    UserReply,
)
from mlx_serve.utils import ZmqAsyncPullQueue, ZmqAsyncPushQueue, init_logger
from pydantic import BaseModel, Field
from starlette.background import BackgroundTask
from transformers import AutoTokenizer

from .args import ServerArgs

logger = init_logger(__name__, "FrontendAPI")

_GLOBAL_STATE = None

_THINK_START = "<think>"
_THINK_END = "</think>"


def _longest_tag_prefix_suffix(text: str, tags: List[str]) -> int:
    """Return max length of text suffix that may be a tag prefix."""
    max_len = 0
    for tag in tags:
        max_check = min(len(text), len(tag) - 1)
        for size in range(max_check, 0, -1):
            if text.endswith(tag[:size]):
                max_len = max(max_len, size)
                break
    return max_len


def _detect_template_injected_think(model_path: str) -> bool:
    """Detect whether chat template already appends <think> before generation."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        probe_messages = [{"role": "user", "content": "hi"}]
        rendered = tokenizer.apply_chat_template(
            probe_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        if not isinstance(rendered, str):
            return False
        # Tolerate trailing spaces/newlines after think tag.
        return rendered.rstrip().endswith(_THINK_START)
    except Exception as e:
        logger.warning("Failed to detect chat template think injection: %s", e)
        return False


def _parse_reasoning_nonstream(
    text: str, template_injected_think: bool
) -> Tuple[Optional[str], str]:
    """Split final output into reasoning_content and content."""
    if _THINK_START in text:
        _, _, after_start = text.partition(_THINK_START)
        if _THINK_END in after_start:
            reasoning_content, _, content = after_start.partition(_THINK_END)
            return reasoning_content or None, content
        # Keep fallback simple: no complete pair means treat as plain content.
        return None, text

    if _THINK_END in text:
        before_end, _, content = text.partition(_THINK_END)
        if template_injected_think:
            return before_end or None, content
        # For non-template case, this still commonly indicates reasoning-first output.
        return before_end or None, content

    return None, text


@dataclass
class ReasoningStreamParser:
    template_injected_think: bool
    state: str = field(init=False)
    pending: str = ""

    def __post_init__(self) -> None:
        self.state = "reasoning" if self.template_injected_think else "content"

    def _flush_content_safe(self) -> str:
        keep = _longest_tag_prefix_suffix(self.pending, [_THINK_START, _THINK_END])
        emit_len = len(self.pending) - keep
        if emit_len <= 0:
            return ""
        out = self.pending[:emit_len]
        self.pending = self.pending[emit_len:]
        return out

    def _flush_reasoning_safe(self) -> str:
        keep = _longest_tag_prefix_suffix(self.pending, [_THINK_END, _THINK_START])
        emit_len = len(self.pending) - keep
        if emit_len <= 0:
            return ""
        out = self.pending[:emit_len]
        self.pending = self.pending[emit_len:]
        return out

    def feed(self, text: str, *, finished: bool) -> Tuple[Optional[str], Optional[str]]:
        self.pending += text
        reasoning_delta = ""
        content_delta = ""

        while True:
            if self.state == "content":
                start_idx = self.pending.find(_THINK_START)
                end_idx = self.pending.find(_THINK_END)

                if start_idx != -1 and (end_idx == -1 or start_idx < end_idx):
                    content_delta += self.pending[:start_idx]
                    self.pending = self.pending[start_idx + len(_THINK_START) :]
                    self.state = "reasoning"
                    continue

                if end_idx != -1:
                    # Support template-injected <think>: output may only emit </think>.
                    reasoning_delta += self.pending[:end_idx]
                    self.pending = self.pending[end_idx + len(_THINK_END) :]
                    self.state = "content"
                    continue

                if finished:
                    content_delta += self.pending
                    self.pending = ""
                else:
                    content_delta += self._flush_content_safe()
                break

            # reasoning state
            end_idx = self.pending.find(_THINK_END)
            if end_idx != -1:
                reasoning_delta += self.pending[:end_idx]
                self.pending = self.pending[end_idx + len(_THINK_END) :]
                self.state = "content"
                continue

            start_idx = self.pending.find(_THINK_START)
            if start_idx != -1:
                # Drop nested/repeated start token if it appears.
                reasoning_delta += self.pending[:start_idx]
                self.pending = self.pending[start_idx + len(_THINK_START) :]
                continue

            if finished:
                reasoning_delta += self.pending
                self.pending = ""
            else:
                reasoning_delta += self._flush_reasoning_safe()
            break

        return reasoning_delta or None, content_delta or None


def get_global_state() -> FrontendManager:
    global _GLOBAL_STATE
    assert _GLOBAL_STATE is not None, "Global state is not initialized"
    return _GLOBAL_STATE


def _unwrap_msg(msg: BaseFrontendMsg) -> List[UserReply]:
    if isinstance(msg, BatchFrontendMsg):
        result = []
        for reply in msg.data:
            assert isinstance(reply, UserReply)
            result.append(reply)
        return result
    assert isinstance(msg, UserReply)
    return [msg]


class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int
    temperature: float = 1.0
    top_p: float = 1.0
    ignore_eos: bool = False


class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class OpenAICompletionRequest(BaseModel):
    """Unified request model for OpenAI-style completions and chat-completions."""

    model: str

    prompt: str | None = None
    messages: List[Message] | None = None

    max_tokens: int = 4096
    temperature: float = 1.0

    top_k: int = -1
    top_p: float = 1.0
    n: int = 1
    stream: bool = False
    stop: List[str] = []
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0

    ignore_eos: bool = False


class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "mlx-serve"
    root: str


class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelCard] = Field(default_factory=list)


@dataclass
class FrontendManager:
    config: ServerArgs
    send_tokenizer: ZmqAsyncPushQueue[BaseTokenizerMsg]
    recv_tokenizer: ZmqAsyncPullQueue[BaseFrontendMsg]
    uid_counter: int = 0
    initialized: bool = False
    ack_map: Dict[int, List[UserReply]] = field(default_factory=dict)
    event_map: Dict[int, asyncio.Event] = field(default_factory=dict)
    reasoning_parser_map: Dict[int, ReasoningStreamParser] = field(default_factory=dict)
    template_injected_think: bool = False

    def new_user(self) -> int:
        uid = self.uid_counter
        self.uid_counter += 1
        self.ack_map[uid] = []
        self.event_map[uid] = asyncio.Event()
        return uid

    async def listen(self):
        while True:
            msg = await self.recv_tokenizer.get()
            for msg in _unwrap_msg(msg):
                assert msg.uid in self.ack_map
                self.ack_map[msg.uid].append(msg)
                self.event_map[msg.uid].set()

    def _create_listener_once(self):
        if not self.initialized:
            asyncio.create_task(self.listen())
            self.initialized = True

    async def send_one(self, msg: BaseTokenizerMsg):
        self._create_listener_once()
        await self.send_tokenizer.put(msg)

    async def wait_for_ack(self, uid: int):
        event = self.event_map[uid]

        while True:
            await event.wait()
            event.clear()

            pending = self.ack_map[uid]
            self.ack_map[uid] = []
            ack = None
            for ack in pending:
                yield ack
            if ack and ack.finished:
                break

        del self.ack_map[uid]
        del self.event_map[uid]

    async def stream_generate(self, uid: int):
        async for ack in self.wait_for_ack(uid):
            yield f"data: {ack.incremental_output}\n".encode()
            if ack.finished:
                break
        yield "data: [DONE]\n".encode()
        logger.debug("Finished streaming response for user %s", uid)

    async def stream_chat_completions(self, uid: int):
        parser = self.reasoning_parser_map[uid]
        first_chunk = True
        async for ack in self.wait_for_ack(uid):
            reasoning_delta, content_delta = parser.feed(
                ack.incremental_output, finished=ack.finished
            )
            delta = {}
            if first_chunk:
                delta["role"] = "assistant"
                first_chunk = False
            if reasoning_delta is not None:
                delta["reasoning_content"] = reasoning_delta
            if content_delta is not None:
                delta["content"] = content_delta

            chunk = {
                "id": f"cmpl-{uid}",
                "object": "text_completion.chunk",
                "choices": [{"delta": delta, "index": 0, "finish_reason": None}],
            }
            yield f"data: {json.dumps(chunk)}\n\n".encode()

            if ack.finished:
                break

        self.reasoning_parser_map.pop(uid, None)
        # send final finish_reason
        end_chunk = {
            "id": f"cmpl-{uid}",
            "object": "text_completion.chunk",
            "choices": [{"delta": {}, "index": 0, "finish_reason": "stop"}],
        }
        yield f"data: {json.dumps(end_chunk)}\n\n".encode()
        yield b"data: [DONE]\n\n"
        logger.debug("Finished streaming response for user %s", uid)

    async def abort_user(self, uid: int):
        await asyncio.sleep(0.1)
        still_pending = uid in self.ack_map
        self.reasoning_parser_map.pop(uid, None)
        if uid in self.ack_map:
            del self.ack_map[uid]
        if uid in self.event_map:
            del self.event_map[uid]
        logger.warning(
            "[Abort] uid=%d  still_pending=%s (client disconnected?)",
            uid, still_pending,
        )

    def shutdown(self):
        self.send_tokenizer.stop()
        self.recv_tokenizer.stop()


@asynccontextmanager
async def lifespan(_: FastAPI):
    yield
    # shutdown code here
    global _GLOBAL_STATE
    if _GLOBAL_STATE is not None:
        _GLOBAL_STATE.shutdown()


app = FastAPI(title="MLX-Serve API Server", version="0.0.1", lifespan=lifespan)


@app.post("/generate")
async def generate(req: GenerateRequest):
    logger.debug("Received generate request %s", req)
    state = get_global_state()
    uid = state.new_user()
    await state.send_one(
        TokenizeMsg(
            uid=uid,
            text=req.prompt,
            sampling_params=SamplingParams(
                ignore_eos=req.ignore_eos,
                max_tokens=req.max_tokens,
            ),
        )
    )

    async def _abort():
        await state.abort_user(uid)

    return StreamingResponse(
        state.stream_generate(uid),
        media_type="text/event-stream",
        background=BackgroundTask(_abort),
    )


@app.api_route("/v1", methods=["GET", "POST", "HEAD", "OPTIONS"])
async def v1_root():
    return {"status": "ok"}


@app.post("/v1/chat/completions")
async def v1_completions(req: OpenAICompletionRequest):
    state = get_global_state()
    if req.messages:
        prompt = [msg.model_dump() for msg in req.messages]
    else:
        assert req.prompt is not None, "Either 'messages' or 'prompt' must be provided"
        prompt = req.prompt

    uid = state.new_user()
    template_injected_think = state.template_injected_think and (req.messages is not None)
    state.reasoning_parser_map[uid] = ReasoningStreamParser(
        template_injected_think=template_injected_think
    )
    msg_summary = ""
    if req.messages:
        msg_summary = " | ".join(f"{m.role}: {m.content[:80]}" for m in req.messages)
    else:
        msg_summary = str(prompt)[:200]
    logger.info(
        "[API] uid=%d  max_tokens=%d  temp=%.2f  stream=%s  msgs=[%s]",
        uid, req.max_tokens, req.temperature, req.stream, msg_summary,
    )
    await state.send_one(
        TokenizeMsg(
            uid=uid,
            text=prompt,
            sampling_params=SamplingParams(
                ignore_eos=req.ignore_eos,
                max_tokens=req.max_tokens,
                temperature=req.temperature,
                top_k=req.top_k,
                top_p=req.top_p,
            ),
        )
    )

    if not req.stream:
        full_text = ""
        async for ack in state.wait_for_ack(uid):
            full_text += ack.incremental_output
            if ack.finished:
                break
        state.reasoning_parser_map.pop(uid, None)
        reasoning_content, content = _parse_reasoning_nonstream(
            full_text, template_injected_think=template_injected_think
        )
        return {
            "id": f"cmpl-{uid}",
            "object": "chat.completion",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "reasoning_content": reasoning_content,
                        "content": content,
                    },
                    "index": 0,
                    "finish_reason": "stop",
                }
            ],
        }

    async def _abort():
        await state.abort_user(uid)

    return StreamingResponse(
        state.stream_chat_completions(uid),
        media_type="text/event-stream",
        background=BackgroundTask(_abort),
    )


@app.get("/v1/models")
async def available_models():
    state = get_global_state()
    return ModelList(data=[ModelCard(id=state.config.model_path, root=state.config.model_path)])


async def shell_completion(req: OpenAICompletionRequest):
    state = get_global_state()
    assert req.messages is not None, "Shell completion only supports chat-completions"
    prompt = [msg.model_dump() for msg in req.messages]

    # TODO: support more sampling parameters
    uid = state.new_user()
    await state.send_one(
        TokenizeMsg(
            uid=uid,
            text=prompt,
            sampling_params=SamplingParams(
                ignore_eos=req.ignore_eos,
                max_tokens=req.max_tokens,
                temperature=req.temperature,
                top_k=req.top_k,
                top_p=req.top_p,
            ),
        )
    )

    async def _abort():
        await state.abort_user(uid)

    return StreamingResponse(
        state.stream_generate(uid),
        media_type="text/event-stream",
        background=BackgroundTask(_abort),
    )


async def read_stdin():
    loop = asyncio.get_running_loop()
    reader = asyncio.StreamReader()
    protocol = asyncio.StreamReaderProtocol(reader)
    await loop.connect_read_pipe(lambda: protocol, sys.stdin)

    while True:
        line = await reader.readline()
        line = line.decode().rstrip("\n")


async def async_input(prompt=""):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, lambda: input(prompt))

def run_api_server(config: ServerArgs, start_backend: Callable[[], None], run_shell: bool) -> None:
    """
    Run the frontend API server (FastAPI + uvicorn) and wire it to the tokenizer process via ZMQ.

    Args:
        config: Server configuration (host/port, ZMQ IPC addresses, etc).
        start_backend: Callback that launches the backend worker processes (TP schedulers +
            tokenizer/detokenizer).
        run_shell: If True, run an interactive terminal shell instead of starting uvicorn.
    """

    global _GLOBAL_STATE

    host = config.server_host
    port = config.server_port

    assert _GLOBAL_STATE is None, "Global state is already initialized"
    _GLOBAL_STATE = FrontendManager(
        config=config,
        recv_tokenizer=ZmqAsyncPullQueue(
            config.zmq_frontend_addr,
            create=True,
            decoder=BaseFrontendMsg.decoder,
        ),
        send_tokenizer=ZmqAsyncPushQueue(
            config.zmq_tokenizer_addr,
            create=config.frontend_create_tokenizer_link,
            encoder=BaseTokenizerMsg.encoder,
        ),
        template_injected_think=_detect_template_injected_think(config.model_path),
    )

    # start the backend here
    start_backend()

    logger.info(f"API server is ready to serve on {host}:{port}")
    if not run_shell:
        uvicorn.run(app, host=host, port=port)
    else:
        asyncio.run(shell())
