from __future__ import annotations

from typing import TYPE_CHECKING, List

import mlx.core as mx
from mlx_serve.message import TokenizeMsg
from mlx_serve.utils import init_logger

if TYPE_CHECKING:
    from transformers import LlamaTokenizer

logger = init_logger(__name__)


class TokenizeManager:
    def __init__(self, tokenizer: LlamaTokenizer, enable_thinking: bool = False) -> None:
        self.tokenizer = tokenizer
        self.enable_thinking = enable_thinking

    def tokenize(self, msgs: List[TokenizeMsg]) -> List[mx.array]:
        results: List[mx.array] = []
        for msg in msgs:
            if isinstance(msg.text, list):
                prompt = self.tokenizer.apply_chat_template(
                    msg.text,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=self.enable_thinking,
                )
                assert isinstance(prompt, str)
            else:
                prompt = msg.text
            input_ids: mx.array = mx.array(  # type: ignore
                self.tokenizer.encode(prompt, return_tensors="np")  
            )
            prompt_preview = prompt[:200] + ("..." if len(prompt) > 200 else "")
            logger.info(
                "[Tokenize] uid=%d  tokens=%d  prompt=%r",
                msg.uid, len(input_ids[0]), prompt_preview,
            )
            results.append(input_ids[0])
        return results
