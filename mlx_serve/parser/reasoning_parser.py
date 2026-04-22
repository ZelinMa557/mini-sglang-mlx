from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

_THINK_START = "<think>"
_THINK_END = "</think>"


def _longest_tag_prefix_suffix(text: str, tags: list[str]) -> int:
    """Return max length of text suffix that may be a tag prefix."""
    max_len = 0
    for tag in tags:
        max_check = min(len(text), len(tag) - 1)
        for size in range(max_check, 0, -1):
            if text.endswith(tag[:size]):
                max_len = max(max_len, size)
                break
    return max_len


@dataclass
class ReasoningStreamParser:
    state: str = field(init=False)
    pending: str = ""

    def __post_init__(self) -> None:
        self.state = "reasoning"

    def _flush_reasoning(self) -> str:
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
            if self.state == "reasoning":
                end_idx = self.pending.find(_THINK_END)
                if end_idx != -1:
                    reasoning_delta += self.pending[:end_idx]
                    self.pending = self.pending[end_idx + len(_THINK_END) :]
                    self.state = "content"
                    continue

                start_idx = self.pending.find(_THINK_START)
                if start_idx != -1:
                    reasoning_delta += self.pending[:start_idx]
                    self.pending = self.pending[start_idx + len(_THINK_START) :]
                    continue

                if finished:
                    reasoning_delta += self.pending
                    self.pending = ""
                else:
                    reasoning_delta += self._flush_reasoning()
                break

            # content state
            content_delta += self.pending
            self.pending = ""
            break

        return reasoning_delta or None, content_delta or None


def parse_reasoning(text: str) -> Tuple[Optional[str], str]:
    """Split final output into reasoning_content and content."""
    if _THINK_START in text:
        _, _, after_start = text.partition(_THINK_START)
        if _THINK_END in after_start:
            reasoning_content, _, content = after_start.partition(_THINK_END)
            return reasoning_content or None, content
    return None, text
