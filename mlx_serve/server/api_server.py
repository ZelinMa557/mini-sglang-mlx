from __future__ import annotations

import asyncio
import json
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Literal, Optional, Union

import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from mlx_serve.core import SamplingParams
from mlx_serve.env import ENV
from mlx_serve.message import (
    BaseFrontendMsg,
    BaseTokenizerMsg,
    BatchFrontendMsg,
    PromptReadyMsg,
    TokenizeMsg,
    UserReply,
)
from mlx_serve.parser import ResponseAdapter, get_response_template
from mlx_serve.utils import ZmqAsyncPullQueue, ZmqAsyncPushQueue, init_logger
from pydantic import BaseModel, Field
from starlette.background import BackgroundTask

from .args import ServerArgs

logger = init_logger(__name__, "FrontendAPI")

_GLOBAL_STATE = None


def get_global_state() -> FrontendManager:
    global _GLOBAL_STATE
    assert _GLOBAL_STATE is not None, "Global state is not initialized"
    return _GLOBAL_STATE


FrontendMsg = Union[UserReply, PromptReadyMsg]


def _unwrap_msg(msg: BaseFrontendMsg) -> List[FrontendMsg]:
    if isinstance(msg, BatchFrontendMsg):
        data = msg.data
    else:
        data = [msg]
    result: List[FrontendMsg] = []
    for item in data:
        assert isinstance(item, (UserReply, PromptReadyMsg)), type(item)
        result.append(item)
    return result


class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int
    temperature: float = 1.0
    top_p: float = 1.0
    ignore_eos: bool = False


class FunctionCall(BaseModel):
    name: str
    arguments: Dict[str, Any] | str | None = None


class ToolCall(BaseModel):
    id: str | None = None
    type: Literal["function"] = "function"
    function: FunctionCall


class Message(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str | None = None
    reasoning_content: str | None = None
    tool_calls: List[ToolCall] | None = None
    tool_call_id: str | None = None


def normalize_message(msg: Message) -> Dict[str, Any]:
    """Shape one history message into what the chat template expects.

    OpenAI carries ``tool_calls[].function.arguments`` as a JSON *string*, but
    the Qwen template iterates it as a mapping (``tool_call.arguments|items``);
    a string raises ``TypeError: Can only get item pairs from a mapping`` inside
    the tokenizer process and takes the whole server down.
    """
    data = msg.model_dump(exclude_none=True)
    for tool_call in data.get("tool_calls", []):
        function = tool_call.get("function") or {}
        arguments = function.get("arguments")
        if isinstance(arguments, str):
            try:
                parsed = json.loads(arguments)
            except json.JSONDecodeError:
                parsed = {"value": arguments}
            function["arguments"] = parsed if isinstance(parsed, dict) else {"value": parsed}
    return data


def build_template_kwargs(req: OpenAICompletionRequest) -> Dict[str, Any]:
    """Merge per-request thinking controls into ``apply_chat_template`` kwargs.

    ``reasoning_effort`` is the standardised knob and is translated into the
    template's own vocabulary; ``chat_template_kwargs`` is the vLLM / SGLang
    compatible escape hatch and wins on conflict.
    """
    kwargs: Dict[str, Any] = {}
    effort = req.reasoning_effort
    if effort is not None:
        if effort in ("none", "minimal"):
            kwargs["enable_thinking"] = False
        else:
            kwargs["enable_thinking"] = True
            kwargs["reasoning_effort"] = "xhigh" if effort in ("high", "xhigh") else effort
    kwargs.update(req.chat_template_kwargs or {})
    # Thinking is on unless the request says otherwise -- do not rely on the
    # template's own default, which varies between Qwen releases.
    kwargs.setdefault("enable_thinking", True)
    if not kwargs["enable_thinking"]:
        # The template ignores effort when thinking is off; drop it so the
        # rendered prompt does not depend on contradictory inputs.
        kwargs.pop("reasoning_effort", None)
    # Tool schemas travel in their own field, not through chat_template_kwargs.
    kwargs.pop("tools", None)
    return kwargs


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

    tools: List[Dict[str, Any]] | None = None
    tool_choice: str | Dict[str, Any] | None = None
    # Standardised thinking controls.  "none"/"minimal" disable thinking, the
    # rest are folded into the template's own effort vocabulary.
    reasoning_effort: Literal["none", "minimal", "low", "medium", "high", "xhigh"] | None = None
    chat_template_kwargs: Dict[str, Any] = Field(default_factory=dict)


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
    response_template: Optional[Dict[str, Any]] = None
    uid_counter: int = 0
    initialized: bool = False
    ack_map: Dict[int, List[UserReply]] = field(default_factory=dict)
    event_map: Dict[int, asyncio.Event] = field(default_factory=dict)
    # Per-request response parsers, and the rendered prompt each one needs as
    # its prefix (produced by the tokenizer process, see PromptReadyMsg).
    response_map: Dict[int, ResponseAdapter] = field(default_factory=dict)
    prompt_map: Dict[int, str] = field(default_factory=dict)
    prompt_event: Dict[int, asyncio.Event] = field(default_factory=dict)

    def new_user(self) -> int:
        uid = self.uid_counter
        self.uid_counter += 1
        self.ack_map[uid] = []
        self.event_map[uid] = asyncio.Event()
        self.prompt_event[uid] = asyncio.Event()
        return uid

    async def listen(self):
        while True:
            incoming = await self.recv_tokenizer.get()
            for item in _unwrap_msg(incoming):
                if isinstance(item, PromptReadyMsg):
                    event = self.prompt_event.get(item.uid)
                    if event is None:
                        logger.debug(
                            "Dropping prompt for uid=%d (already aborted)", item.uid
                        )
                        continue
                    self.prompt_map[item.uid] = item.prompt
                    event.set()
                    continue
                if item.uid not in self.ack_map:
                    logger.debug(
                        "Dropping reply for uid=%d (already aborted)", item.uid
                    )
                    continue
                self.ack_map[item.uid].append(item)
                self.event_map[item.uid].set()

    def _create_listener_once(self):
        if not self.initialized:
            asyncio.create_task(self.listen())
            self.initialized = True

    async def send_one(self, msg: BaseTokenizerMsg):
        self._create_listener_once()
        await self.send_tokenizer.put(msg)

    async def wait_for_prompt(self, uid: int, timeout: float = 60.0) -> Optional[str]:
        """Wait for the rendered prompt of ``uid``.

        Returns ``None`` on timeout, in which case the request degrades to raw
        passthrough.  Waiting is race-free because the tokenizer sends the
        prompt before it hands the request to the backend.
        """
        event = self.prompt_event.get(uid)
        if event is None:
            return self.prompt_map.pop(uid, None)
        try:
            await asyncio.wait_for(event.wait(), timeout)
        except asyncio.TimeoutError:
            logger.warning("[API] uid=%d  timed out waiting for rendered prompt", uid)
            return None
        finally:
            self.prompt_event.pop(uid, None)
        return self.prompt_map.pop(uid, None)

    async def wait_for_ack(self, uid: int):
        event = self.event_map[uid]

        while True:
            await event.wait()
            event.clear()

            if uid not in self.ack_map:
                break
            pending = self.ack_map[uid]
            self.ack_map[uid] = []
            ack = None
            for ack in pending:
                yield ack
            if ack and ack.finished:
                break

        self.ack_map.pop(uid, None)
        self.event_map.pop(uid, None)
        self.prompt_map.pop(uid, None)
        self.prompt_event.pop(uid, None)

    async def stream_generate(self, uid: int):
        async for ack in self.wait_for_ack(uid):
            yield f"data: {ack.incremental_output}\n".encode()
            if ack.finished:
                break
        yield "data: [DONE]\n".encode()
        logger.debug("Finished streaming response for user %s", uid)

    async def stream_chat_completions(self, uid: int):
        adapter = self.response_map.get(uid)
        first_chunk = True
        finish_reason = "stop"
        try:
            async for ack in self.wait_for_ack(uid):
                deltas: List[Dict[str, Any]] = []
                if adapter is not None:
                    deltas.extend(adapter.feed(ack.incremental_output))
                    if ack.finished:
                        # Regions can close only at the very end (a truncated
                        # tool call, for instance), so these events must be
                        # routed before finish_reason is decided.
                        _, final_deltas = adapter.finalize()
                        deltas.extend(final_deltas)
                        if adapter.saw_tool_calls:
                            finish_reason = "tool_calls"
                elif ack.incremental_output:
                    deltas.append({"content": ack.incremental_output})

                if first_chunk:
                    if deltas:
                        deltas[0] = {"role": "assistant", **deltas[0]}
                    else:
                        deltas.append({"role": "assistant"})
                    first_chunk = False

                for delta in deltas:
                    chunk = {
                        "id": f"cmpl-{uid}",
                        "object": "text_completion.chunk",
                        "choices": [{"delta": delta, "index": 0, "finish_reason": None}],
                    }
                    yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n".encode()

                if ack.finished:
                    break
        finally:
            self.response_map.pop(uid, None)

        # send final finish_reason
        end_chunk = {
            "id": f"cmpl-{uid}",
            "object": "text_completion.chunk",
            "choices": [{"delta": {}, "index": 0, "finish_reason": finish_reason}],
        }
        yield f"data: {json.dumps(end_chunk)}\n\n".encode()
        yield b"data: [DONE]\n\n"
        logger.debug("Finished streaming response for user %s", uid)

    async def abort_user(self, uid: int):
        await asyncio.sleep(0.1)
        still_pending = uid in self.ack_map
        self.response_map.pop(uid, None)
        self.prompt_map.pop(uid, None)
        self.prompt_event.pop(uid, None)
        if uid in self.event_map:
            self.event_map[uid].set()
            del self.event_map[uid]
        if uid in self.ack_map:
            del self.ack_map[uid]
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
        prompt: str | List[Dict[str, Any]] = [
            normalize_message(msg) for msg in req.messages
        ]
    else:
        assert req.prompt is not None, "Either 'messages' or 'prompt' must be provided"
        prompt = req.prompt

    # "none" means the model must not call a tool; the chat template cannot
    # express that, so hide the schemas altogether.
    tools = None if req.tool_choice == "none" else req.tools

    uid = state.new_user()
    if isinstance(prompt, list):
        msg_summary = " | ".join(
            f"{m['role']}: {(m.get('content') or '')[:80]}" for m in prompt
        )
    else:
        msg_summary = prompt[:200]
    logger.info(
        "[API] uid=%d  max_tokens=%d  temp=%.2f  stream=%s  tools=%d  msgs=[%s]",
        uid, req.max_tokens, req.temperature, req.stream,
        len(tools) if tools else 0, msg_summary,
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
            tools=tools,
            template_kwargs=build_template_kwargs(req),
        )
    )

    # Response parsing needs the rendered prompt as its prefix; it comes back
    # from the tokenizer process, always before the first generated token.
    adapter = None
    if state.response_template is not None and isinstance(prompt, list):
        prefix = await state.wait_for_prompt(uid)
        if prefix is not None:
            adapter = ResponseAdapter(
                state.response_template, prefix, tools=tools, uid=uid
            )
            state.response_map[uid] = adapter

    if not req.stream:
        full_text = ""
        try:
            # Same adapter as the streaming path: feed everything, then
            # finalize.  Events emitted by finalize are ignored here -- the
            # finalized message already carries the whole reply.
            async for ack in state.wait_for_ack(uid):
                if adapter is not None:
                    adapter.feed(ack.incremental_output)
                else:
                    full_text += ack.incremental_output
                if ack.finished:
                    break
            if adapter is None:
                return _completion_response(
                    uid,
                    {"role": "assistant", "reasoning_content": None, "content": full_text},
                    finish_reason="stop",
                )
            message, _ = adapter.finalize()
            out = adapter.to_openai_message(message)
            return _completion_response(
                uid, out, finish_reason="tool_calls" if out.get("tool_calls") else "stop"
            )
        finally:
            state.response_map.pop(uid, None)

    async def _abort():
        await state.abort_user(uid)

    return StreamingResponse(
        state.stream_chat_completions(uid),
        media_type="text/event-stream",
        background=BackgroundTask(_abort),
    )


def _completion_response(
    uid: int, message: Dict[str, Any], *, finish_reason: str
) -> Dict[str, Any]:
    return {
        "id": f"cmpl-{uid}",
        "object": "chat.completion",
        "choices": [{"message": message, "index": 0, "finish_reason": finish_reason}],
    }


@app.get("/v1/models")
async def available_models():
    state = get_global_state()
    return ModelList(data=[ModelCard(id=state.config.model_path, root=state.config.model_path)])

def run_api_server(config: ServerArgs, start_backend: Callable[[], None]) -> None:
    """
    Run the frontend API server (FastAPI + uvicorn) and wire it to the tokenizer process via ZMQ.

    Args:
        config: Server configuration (host/port, ZMQ IPC addresses, etc).
        start_backend: Callback that launches the backend worker processes (TP schedulers +
            tokenizer/detokenizer).
    """

    global _GLOBAL_STATE
    host = config.server_host
    port = config.server_port

    assert _GLOBAL_STATE is None, "Global state is already initialized"
    # Resolve and validate the response template up front: a bad template must
    # fail at startup, not on the first chat request.
    response_template = get_response_template(config.model_path)
    if response_template is not None:
        from transformers.utils.chat_parsing.response_templates import (
            load_response_template,
        )

        load_response_template(response_template)
        logger.info(
            "[API] response parsing enabled with fields=%s",
            sorted(response_template.get("fields", {})),
        )
    else:
        logger.info("[API] no response template for %s; passthrough mode", config.model_path)
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
        response_template=response_template,
    )

    # start the backend here
    start_backend()

    logger.info(f"API server is ready to serve on {host}:{port}")
    uvicorn.run(app, host=host, port=port)
