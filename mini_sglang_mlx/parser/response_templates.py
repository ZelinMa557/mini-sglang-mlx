"""Declarative response templates for ``transformers`` response parsing.

A response template is the inverse of a chat template: it tells transformers
how to turn the raw token stream back into a structured assistant message
(``content`` / ``thinking`` / ``tool_calls``).  The spec format and the parser
that consumes it live in ``transformers.utils.chat_parsing`` (5.13+); this
module only ships the templates mini-sglang-mlx needs and resolves them per model.

Convention: every template here must name its fields ``thinking``,
``content`` and ``tool_calls``.  ``ResponseAdapter`` routes events by field
name, so a template using other names will not stream.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

# Qwen3.5 / Qwen3.8 chat template format.  The generation prompt already opens
# the thinking block ("<|im_start|>assistant\n<think>\n" when thinking is on,
# "...<think>\n\n</think>\n\n" when it is off), which is why the parser needs
# the rendered prompt as `prefix` to know where the message begins.
QWEN3_XML_RESPONSE_TEMPLATE: Dict[str, Any] = {
    "version": 1,
    "defaults": {"role": "assistant"},
    "start_anchor": "<|im_start|>assistant\n",
    "fields": {
        # No `open`/`close`: the implicit field that soaks up everything not
        # claimed by another region.  `<|im_end|>` is the eos token and never
        # reaches the parser (the detokenizer drops it), so content simply runs
        # to the end of the stream.
        # `repeats` + `join` concatenate the segments around a tool call
        # ("text<tool_call/>more text"); without it the parser overwrites the
        # earlier segment and the non-streamed message loses text the streamed
        # deltas already sent.
        "content": {"content": "text", "repeats": True, "join": ""},
        "thinking": {"open": "<think>", "close": "</think>", "content": "text",
                     "repeats": True, "join": ""},
        "tool_calls": {
            # <tool_call>\n<function=NAME>\n<parameter=KEY>\nVALUE\n</parameter>\n</function>\n</tool_call>
            "open_pattern": r"<tool_call>\s*<function=(?P<name>[^>\s]+)>",
            "close": "</tool_call>",
            "repeats": True,
            "content": "xml-inline",
            "content_args": {
                "tag_pattern": r"<parameter=(?P<key>[^>\s]+)>\s*(?P<value>.*?)\s*</parameter>",
                "value_parser": {"name": "json", "args": {"allow_non_json": True}},
            },
            "transform": {
                "type": "function",
                "function": {"name": "{name}", "arguments": "{content}"},
            },
        },
    },
}

# Model types sharing the Qwen3.5 template above.
_BY_MODEL_TYPE: Dict[str, Dict[str, Any]] = {
    "qwen3_5": QWEN3_XML_RESPONSE_TEMPLATE,
    "qwen3_5_text": QWEN3_XML_RESPONSE_TEMPLATE,
}


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def get_response_template(model_path: str) -> Optional[Dict[str, Any]]:
    """Resolve the response template for a (local) model directory.

    Precedence: an explicit ``response_template`` in the checkpoint's
    ``tokenizer_config.json`` (so users can override without touching code),
    then the registry keyed by the config's ``model_type``, then ``None`` --
    in which case callers must fall back to passing raw text through.
    """
    model_dir = Path(model_path)
    tokenizer_config = _read_json(model_dir / "tokenizer_config.json")
    if tokenizer_config:
        template = tokenizer_config.get("response_template")
        if isinstance(template, dict):
            return template

    config = _read_json(model_dir / "config.json")
    if config:
        model_type = config.get("model_type")
        if isinstance(model_type, str) and model_type in _BY_MODEL_TYPE:
            return _BY_MODEL_TYPE[model_type]
    return None
