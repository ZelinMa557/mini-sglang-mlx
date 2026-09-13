from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from .utils import deserialize_type, serialize_type


@dataclass
class BaseFrontendMsg:
    @staticmethod
    def encoder(msg: BaseFrontendMsg) -> Dict:
        return serialize_type(msg)

    @staticmethod
    def decoder(json: Dict) -> BaseFrontendMsg:
        return deserialize_type(globals(), json)


@dataclass
class BatchFrontendMsg(BaseFrontendMsg):
    data: List[BaseFrontendMsg]


@dataclass
class UserReply(BaseFrontendMsg):
    uid: int
    incremental_output: str
    finished: bool


@dataclass
class PromptReadyMsg(BaseFrontendMsg):
    """The chat-template-rendered prompt for a request.

    Response parsing needs the rendered prompt as its ``prefix`` (templates can
    prefill part of the assistant message), but rendering only happens in the
    tokenizer process -- so it is sent back to the frontend, which owns the
    per-request parser state.
    """

    uid: int
    prompt: str
