"""Turn transformers response-parser events into OpenAI-shaped deltas.

``transformers.utils.chat_parsing.ResponseParser`` emits one event per region
transition; this adapter maps those onto the ``delta`` dicts an OpenAI client
expects (``content`` / ``reasoning_content`` / ``tool_calls``).
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

from mlx_serve.utils import init_logger

logger = init_logger(__name__)

_THINKING_FIELD = "thinking"
_CONTENT_FIELD = "content"
_TOOL_CALLS_FIELD = "tool_calls"


def tool_call_to_openai(value: Dict[str, Any], index: int, uid: int) -> Dict[str, Any]:
    """Convert a parsed ``tool_calls`` region into an OpenAI tool call.

    The parser yields ``{"type": "function", "function": {"name", "arguments"}}``
    with ``arguments`` as a dict; the API wants a JSON *string*.
    """
    function = value.get("function", {})
    arguments = function.get("arguments", {})
    return {
        "index": index,
        "id": f"call_{uid}_{index}",
        "type": value.get("type", "function"),
        "function": {
            "name": function.get("name", ""),
            "arguments": arguments
            if isinstance(arguments, str)
            else json.dumps(arguments, ensure_ascii=False),
        },
    }


class ResponseAdapter:
    """One stateful response parser per request, emitting OpenAI deltas.

    ``prefix`` is the rendered chat prompt: the parser truncates it at the last
    assistant turn and replays it so prefill such as an opening ``<think>`` is
    accounted for.  Those prefix events are discarded -- they are prompt, not
    answer.
    """

    def __init__(
        self,
        template: Dict[str, Any],
        prefix: str,
        tools: Optional[List[Dict[str, Any]]] = None,
        uid: int = 0,
    ) -> None:
        from transformers.utils.chat_parsing import ResponseParser

        self._parser = ResponseParser(template, prefix=prefix, tools=tools)
        self._uid = uid
        self._num_tool_calls = 0
        self._first_content = True
        self.broken = False
        # Trailing whitespace per field, withheld until the region's fate is
        # known (see _take).
        self._pending: Dict[str, str] = {}
        if self._parser.initial_events:
            logger.debug(
                "[ResponseParser] uid=%d  replayed %d prefill event(s)",
                uid, len(self._parser.initial_events),
            )

    @property
    def saw_tool_calls(self) -> bool:
        return self._num_tool_calls > 0

    def feed(self, text: str) -> List[Dict[str, Any]]:
        if self.broken:
            return [{"content": text}] if text else []
        try:
            return self._render(self._parser.feed(text))
        except Exception:  # noqa: BLE001 - a parse failure must not kill the stream
            logger.exception("[ResponseParser] uid=%d  falling back to raw text", self._uid)
            self.broken = True
            return [{"content": text}] if text else []

    def finalize(self) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """Return ``(message, deltas)`` for the end of the stream.

        Regions can close only at eos (a truncated tool call, for instance), so
        the final events must be rendered too.
        """
        # Whatever is still withheld closes at end of stream, and finalize
        # strips it from the message.
        self._pending.clear()
        try:
            message, events = self._parser.finalize()
        except Exception:  # noqa: BLE001
            logger.exception("[ResponseParser] uid=%d  finalize failed", self._uid)
            self.broken = True
            return {"role": "assistant"}, []
        return message, self._render(events)

    def to_openai_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Shape a finalized parser message into an OpenAI assistant message."""
        out: Dict[str, Any] = {"role": "assistant"}
        if message.get(_CONTENT_FIELD):
            out["content"] = message[_CONTENT_FIELD]
        if message.get(_THINKING_FIELD):
            out["reasoning_content"] = message[_THINKING_FIELD]
        tool_calls = message.get(_TOOL_CALLS_FIELD)
        if tool_calls:
            out["tool_calls"] = [
                tool_call_to_openai(value, index, self._uid)
                for index, value in enumerate(tool_calls)
            ]
        return out

    def _render(self, events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        deltas: List[Dict[str, Any]] = []
        for event in events:
            kind = event.get("type")
            field = event.get("field")

            if kind == "region_chunk":
                delta = self._chunk_delta(field, event.get("text", ""))
            elif kind == "region_close":
                # Non-streamed parsing strips the whitespace sitting just
                # before the close, so what was withheld never happened.
                self._pending.pop(field, None)
                delta = (
                    self._tool_call_delta(event.get("value"))
                    if field == _TOOL_CALLS_FIELD
                    else None
                )
            else:
                delta = None
            if delta:
                deltas.append(delta)
        return deltas

    def _take(self, field: str, text: str) -> str:
        """Withhold a chunk's trailing whitespace until its fate is known.

        A region's closing strips the whitespace in front of it, but streaming
        has already emitted anything it forwarded.  Holding trailing whitespace
        back until the next non-blank chunk lets a close drop it, so the
        streamed text ends up byte-identical to the non-streamed one.  A
        whitespace-only region therefore emits nothing at all.
        """
        text = self._pending.pop(field, "") + text
        stripped = text.rstrip()
        if not stripped:
            self._pending[field] = text
            return ""
        self._pending[field] = text[len(stripped):]
        return stripped

    def _chunk_delta(self, field: Optional[str], text: str) -> Optional[Dict[str, Any]]:
        if not text:
            return None
        if field not in (_THINKING_FIELD, _CONTENT_FIELD):
            # tool-call bodies are raw XML -- the parsed value only exists on
            # close
            return None
        if field == _CONTENT_FIELD and self._first_content:
            # The template writes "\n\n" after </think> as part of the format;
            # the full-message parse strips it too, so strip it here to keep the
            # streamed and non-streamed content identical.
            text = text.lstrip()
            if not text:
                return None
            self._first_content = False
        text = self._take(field, text)
        if not text:
            return None
        return {"reasoning_content": text} if field == _THINKING_FIELD else {"content": text}

    def _tool_call_delta(self, value: Any) -> Optional[Dict[str, Any]]:
        if not isinstance(value, dict):
            return None
        delta = tool_call_to_openai(value, self._num_tool_calls, self._uid)
        self._num_tool_calls += 1
        return {"tool_calls": [delta]}
