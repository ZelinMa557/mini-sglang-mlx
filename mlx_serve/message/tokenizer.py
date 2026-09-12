from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from mlx_serve.core import SamplingParams

from .utils import deserialize_type, serialize_type


@dataclass
class BaseTokenizerMsg:
    @staticmethod
    def encoder(msg: BaseTokenizerMsg) -> Dict:
        return serialize_type(msg)

    @staticmethod
    def decoder(json: Dict) -> BaseTokenizerMsg:
        return deserialize_type(globals(), json)


@dataclass
class BatchTokenizerMsg(BaseTokenizerMsg):
    data: List[BaseTokenizerMsg]


@dataclass
class DetokenizeMsg(BaseTokenizerMsg):
    uid: int
    next_token: int
    finished: bool


@dataclass
class TokenizeMsg(BaseTokenizerMsg):
    uid: int
    text: str | List[Dict[str, Any]]
    sampling_params: SamplingParams
    # Tool schemas for ``apply_chat_template(tools=...)`` and extra chat
    # template kwargs (``enable_thinking``, ``reasoning_effort``, ...).
    tools: Optional[List[Dict[str, Any]]] = None
    template_kwargs: Optional[Dict[str, Any]] = None


@dataclass
class AbortMsg(BaseTokenizerMsg):
    uid: int
