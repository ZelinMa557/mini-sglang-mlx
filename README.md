# mlx-serve

mlx-serve is a (at current stage, will be a) high performance LLM inference engine for Mac GPUs.

It support the following features:

* continous batching

* paged attention

* radix tree based prefix cache

* high performance metal & cpu kernels

* open-ai compitable apis

* streaming reasoning (`reasoning_content`) and tool-call parsing

Requires `transformers>=5.16.1` (response parsing lives in
`transformers.utils.chat_parsing`; the package `__init__` does not re-export it).

## Reasoning and tool calls

Reasoning and tool calls are parsed with transformers' declarative response
parsing. Templates are resolved per model in `mlx_serve/parser/response_templates.py`:
a `response_template` key in the checkpoint's `tokenizer_config.json` wins, then
the registry lookup by `model_type`, then no parsing at all (raw passthrough).

A chat request may carry these OpenAI-compatible fields:

| Field | Effect |
| --- | --- |
| `reasoning_effort` | `"none"` / `"minimal"` disable thinking; `"low"` / `"medium"` / `"high"` / `"xhigh"` select the template's effort |
| `chat_template_kwargs` | Passed verbatim to `apply_chat_template`; overrides the above, and is the escape hatch for anything else (e.g. `preserve_thinking`) |
| `tools` / `tool_choice` | Schemas are rendered into the prompt; `tool_choice="none"` hides them. `tool_choice` is not otherwise enforced |

Thinking is **on** by default when a request does not specify any of these.
Responses put thinking in `reasoning_content` and tool calls in `tool_calls`
(as JSON strings, like OpenAI), with `finish_reason="tool_calls"` when a call
was produced. Assistant messages in the request history may carry `tool_calls`
(which are also accepted as JSON strings), and `role: "tool"` messages carry
the results.

Support models:

* qwen3, qwen3-moe

* gpt-oss

Roadmap:

* paged attention kernel, supports sliding window attention and attention sink (P0)

* model runner (P0)

* openai compitable api (P0)

* moe related kernels (P1)

* support qwen3-moe and gpt-oss (P1)