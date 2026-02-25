from __future__ import annotations

from typing import TYPE_CHECKING, List

import mlx.core as mx
from mlx_serve.message import TokenizeMsg

if TYPE_CHECKING:
    from transformers import LlamaTokenizer


class TokenizeManager:
    def __init__(self, tokenizer: LlamaTokenizer) -> None:
        self.tokenizer = tokenizer

    def tokenize(self, msgs: List[TokenizeMsg]) -> List[mx.array]:
        results: List[mx.array] = []
        # TODO: batch tokenization
        for msg in msgs:
            if isinstance(msg.text, list):
                prompt = self.tokenizer.apply_chat_template(
                    msg.text,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                assert isinstance(prompt, str)
            else:
                prompt = msg.text
            input_ids: mx.array = mx.array(  # type: ignore
                self.tokenizer.encode(prompt, return_tensors="np")  
            )
            results.append(input_ids[0])
        return results
