from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import mlx.core as mx
from mini_sglang_mlx.message import TokenizeMsg
from mini_sglang_mlx.utils import init_logger

if TYPE_CHECKING:
    from transformers import LlamaTokenizer

logger = init_logger(__name__)

# Keyword arguments that belong to ``apply_chat_template`` itself; letting a
# client smuggle them in through ``chat_template_kwargs`` would raise
# "got multiple values for keyword argument" and kill the worker process.
_RESERVED_TEMPLATE_KWARGS = frozenset(
    {
        "tokenize",
        "add_generation_prompt",
        "continue_final_message",
        "tools",
        "messages",
        "conversation",
        "return_dict",
        "return_tensors",
        "padding",
        "truncation",
        "max_length",
    }
)


class TokenizeManager:
    def __init__(self, tokenizer: LlamaTokenizer) -> None:
        self.tokenizer = tokenizer

    def _render(self, msg: TokenizeMsg) -> Tuple[str, str]:
        """Render a chat prompt: returns ``(prompt, apply_chat_template kwargs)``."""
        kwargs: Dict[str, Any] = {}
        for key, value in (msg.template_kwargs or {}).items():
            if key in _RESERVED_TEMPLATE_KWARGS:
                logger.warning(
                    "[Tokenize] uid=%d  ignoring reserved chat_template_kwargs %r",
                    msg.uid, key,
                )
                continue
            kwargs[key] = value
        if msg.tools:
            kwargs["tools"] = msg.tools

        try:
            prompt = self.tokenizer.apply_chat_template(
                msg.text,
                tokenize=False,
                add_generation_prompt=True,
                **kwargs,
            )
        except Exception:
            # A template error here would take down the tokenizer (and, when
            # the tokenizer is shared, the detokenizer) process, so degrade to
            # a plain render rather than propagating.
            logger.exception(
                "[Tokenize] uid=%d  chat template failed with %r; retrying plain",
                msg.uid, sorted(kwargs),
            )
            prompt = self.tokenizer.apply_chat_template(
                msg.text, tokenize=False, add_generation_prompt=True,
            )
        assert isinstance(prompt, str)
        return prompt, kwargs

    def tokenize(self, msgs: List[TokenizeMsg]) -> List[Tuple[mx.array, Optional[str]]]:
        """Tokenize a batch, returning ``(input_ids, rendered_prompt)``.

        ``rendered_prompt`` is ``None`` for raw-string prompts, which bypass the
        chat template and therefore have no response parsing.
        """
        results: List[Tuple[mx.array, Optional[str]]] = []
        for msg in msgs:
            if isinstance(msg.text, list):
                prompt, _ = self._render(msg)
                rendered: Optional[str] = prompt
            else:
                prompt = msg.text
                rendered = None
            input_ids: mx.array = mx.array(  # type: ignore
                self.tokenizer.encode(prompt, return_tensors="np")
            )
            prompt_preview = prompt[:200] + ("..." if len(prompt) > 200 else "")
            logger.info(
                "[Tokenize] uid=%d  tokens=%d  prompt=%r",
                msg.uid, len(input_ids[0]), prompt_preview,
            )
            results.append((input_ids[0], rendered))
        return results
