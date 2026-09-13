# mini-sglang-mlx

A high-performance LLM inference engine for Apple Silicon, ported from
[mini-sglang](https://github.com/sgl-project/mini-sglang).

mini-sglang is a compact reimplementation of SGLang's serving architecture:
a frontend / tokenizer / scheduler split across processes, radix-tree prefix
caching, chunked prefill. **mini-sglang-mlx keeps that architecture and that
code shape** — the module layout, the `Scheduler` / `Engine` / `CacheManager`
boundaries, and the ZeroMQ message plumbing are all inherited — and replaces
the CUDA execution layer with [MLX](https://github.com/ml-explore/mlx)
plus hand-written Metal kernels, so the same design runs on a Mac GPU.

The goal is a serving stack that is small enough to read and fast enough to
use: the Metal kernels, the paged KV cache, and the speculative-decoding
engine are the parts that had to be rewritten, and they are the parts this
project is really about.

## Features

* **Paged attention** — custom Metal kernels for prefill and decode, over a
  paged KV cache. The prefill kernel supports sliding-window masking, which
  the DFlash draft's block-attention layers use.
* **Radix-tree prefix caching** — shared prefixes are matched and reused
  across requests, including recurrent state for hybrid models.
* **Continuous batching** — requests join and leave the running batch between
  decode steps, and long prompts are chunked across iterations rather than
  blocking the queue. Batch selection is prefill-priority: each step takes a
  prefill batch if one is ready, otherwise a decode batch.
* **DFlash2 speculative decoding** — block-diffusion drafting with
  replay-based state rollback for gated-delta-net layers.
* **Hybrid GDN + attention targets** — Qwen3.5/3.8-style gated delta net
  layers run through fused slot-indexed Metal kernels, not a Python loop.
* **OpenAI-compatible API** — `/v1/chat/completions` and `/v1/models` with
  streaming.
* **Streaming reasoning and tool calls** — `reasoning_content` and
  `tool_calls`, parsed with transformers' declarative response parsing.

## Status

**This project is early.** It runs, it serves real requests, and the numbers
below are measured rather than aspirational — but two things are currently
holding it back, and both are known, understood, and on the roadmap.

### 1. The paged attention kernel is slower than MLX's built-in attention

The custom Metal paged-attention kernels reach roughly **0.75× the throughput
of `mx.fast.scaled_dot_product_attention`**, and the gap widens at long KV
lengths. Measured over 36 (prefill chunk × KV length) configurations:

| percentile | ratio vs. MLX built-in |
| --- | --- |
| p25 | 0.68× |
| median | 0.74× |
| p75 | 0.77× |

Paged attention buys memory efficiency and prefix sharing, not raw speed. If
your workload is a single long-context request with no prefix reuse, the
built-in attention path is faster.

### 2. Speculative decoding does not reliably turn accept length into speed

Verifying `W = K + 1` draft tokens is a single forward with `M = B × W`. MLX's
`quantized_matmul` is already at its bandwidth ceiling by `M = 2`; from
`M = 3` upward each additional verified token costs a **fixed** ~0.125 ms per
projection rather than amortising. Measured on the gate projection
(`K=5120, N=17408`, 4-bit, group size 64):

| M | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ms | 0.602 | 0.590 | 0.646 | 0.757 | 0.883 | 1.016 | 1.139 | 1.268 |
| effective GB/s | 83.3 | 84.9 | 77.7 | 66.2 | 56.8 | 49.4 | 44.0 | 39.5 |

Two regimes, with the crossover between `M = 2` and `M = 3`: up to `M = 2` the
projection is purely bandwidth-bound and a second token is **free**, and from
`M = 3` on it pays a fixed ~0.125 ms per additional token instead of
amortising.

So the marginal cost of a longer draft is nearly free, but the *baseline* cost
of a verify pass is ~2× a plain decode step regardless of how many tokens are
accepted. Speculative decoding therefore only wins once the draft reliably
lands more than ~2–3 tokens per step. On the targets we have tested DFlash2
does not consistently clear that bar, so the accept length is real but the
wall-clock win is not.

Two consequences worth stating plainly: **shrinking `K` is the wrong fix**
(draft tokens are nearly free — they are not what costs you), and so is
fusing or rebalancing projections (the bottleneck is `M`, not `N`). What does
help is batching: the per-token cost of the same projection falls from
0.602 ms at `M = 1` to 0.159 ms at `M = 8`, because one pass over the weights
serves eight tokens instead of one.

## Roadmap

1. **W4A8 quantization with matching quantized kernels.** Weight-only 4-bit
   quantization is the reason `M` saturates at 2 — the weights are already so
   compressed that there is nothing left to hide the arithmetic behind. Going
   to 4-bit weights / 8-bit activations raises arithmetic intensity at small
   `M`, which is what speculative decoding needs to actually pay off, and is
   targeted at a **~50% speedup on long-context prefill**.
2. **Continue improving the paged attention kernel.** The current kernels are
   a first working version; closing the 0.75× gap against MLX's built-in
   attention is an ongoing effort.
3. **Prepare for the Qwen4 architecture.** The model layer is deliberately
   small and registry-driven, and the next Qwen generation is expected to
   need structural changes that are cheaper to make now than later.

## Supported models

| Family | Resolves to | Notes |
| --- | --- | --- |
| Qwen3 | `models/qwen3.py` | dense |
| Qwen3-MoE | `models/qwen3_moe.py` | |
| Qwen3.5 | `models/qwen3_5.py` | GDN + attention hybrid |
| Qwen3.8 | `models/qwen3_5.py` | GDN + attention hybrid |

The draft models used by speculative decoding (`models/dflash.py`,
`models/dflash2.py`) are loaded through the same registry.

> There is also a `models/qwen3_5_moe.py` (sparse MoE on top of the Qwen3.5
> hybrid stack), but it is currently **unreachable**: `MODEL_REMAPPING` in
> `models/__init__.py` maps `model_type: "qwen3_5_moe"` to `qwen3_5`, so such a
> checkpoint is loaded by the *dense* module and the sparse one is shadowed.
> Remove that mapping entry to make it reachable again.

## Installation

Requires macOS on Apple Silicon. Two steps, in this order — the kernel
extension must be built before the Python package can import it.

### 1. Build the kernel extension

This is a C++/Metal extension and is **not on PyPI**. It is built with CMake
and [nanobind](https://github.com/wjakob/nanobind), both of which must already
be present in the environment (`--no-build-isolation` deliberately disables
pip's build sandbox).

```bash
pip install mlx nanobind "cmake>=3.25"

cd mini-sglang-mlx-kernel
pip install -e . --no-build-isolation
cd ..
```

Adding a new `.metal` or `.cpp` file under `csrc/` is picked up automatically
by a `GLOB`, so the build does not need CMake edits when kernels are added.
After changing kernel sources, re-run the same `pip install -e .` command (add
`--force-reinstall --no-deps` if the change does not appear to take).

Verify the build:

```bash
python -c "import mini_sglang_mlx_kernel; print(mini_sglang_mlx_kernel.__file__)"
```

### 2. Install the Python dependencies

```bash
pip install -r requirements.txt
```

The one dependency outside the file is `modelscope`, needed only if you pass
`--use-modelscope`; it is imported lazily and can be omitted safely.

```bash
pip install modelscope        # optional
```

## Quick start

Both commands below download the checkpoints on first run and then serve an
OpenAI-compatible API on `http://127.0.0.1:1919`.

### With speculative decoding (DFlash2)

```bash
python -m mini_sglang_mlx \
  --use-modelscope \
  --model-path mlx-community/Qwen3.8-27B-4bit \
  --draft-path z-lab/Qwen3.8-27B-DFlash2 \
  --spec-algo dflash2 \
  --num-draft-tokens 7 \
  --kv-cache-gb 2.0 \
  --max-running-requests 4
```

`--num-draft-tokens` is `K`, and must equal the draft checkpoint's
training-time `block_size` minus one — read it from the draft's `config.json`
under `dflash_config.block_size`. A mismatch does not raise; it just makes the
draft propose garbage. Drop `--use-modelscope` to resolve both repos from
HuggingFace instead. A runnable copy of this command lives in
`start_local.sh`.

### Without speculative decoding

```bash
python -m mini_sglang_mlx \
  --use-modelscope \
  --model-path mlx-community/Qwen3.8-27B-4bit \
  --kv-cache-gb 2.0 \
  --num-mamba-slots 8 \
  --max-running-requests 4
```

### Querying the server

```bash
curl -s http://127.0.0.1:1919/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3.8-27B-4bit",
    "messages": [{"role": "user", "content": "The capital of France is"}],
    "max_tokens": 32,
    "stream": true
  }'
```

### Notes on tuning

* **`--num-mamba-slots`** applies to hybrid (GDN) targets only, and is
  auto-sized to `max_running_req × (2 + spec_extra)` when unset — 12 for the
  four-request speculative command above. That is the real peak: one main slot
  per in-flight request, one scratch slot per request *simultaneously* during
  target verify, and one snapshot per request kept for the prefix cache.
  Setting it below `2 × max_running_req` leaves the pool permanently dry and
  gives the prefix cache nothing to retain.
* **DFlash limitations**, both currently structural: verification is greedy
  only (sampling params are ignored during the verify pass), and chunked
  prefill is not supported, so a prompt whose uncached portion exceeds
  `--max-prefill-length` (default 8192) raises instead of chunking.

## Reasoning and tool calls

Reasoning and tool calls are parsed with transformers' declarative response
parsing. Templates are resolved per model in
`mini_sglang_mlx/parser/response_templates.py`: a `response_template` key in
the checkpoint's `tokenizer_config.json` wins, then the registry lookup by
`model_type`, then no parsing at all (raw passthrough).

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

## Repository layout

```
mini_sglang_mlx/            Python package (namespace package, no __init__.py)
├── engine/                 Engine / SpecEngine / DFlash engines
├── scheduler/              Scheduler loop, prefill + decode scheduling, cache manager
├── kvcache/                Paged KV pool, radix-tree prefix cache, mamba state pool
├── attention/              Attention backends (paged MHA, GDN recurrence)
├── models/                 Model registry — qwen3 / qwen3_moe / qwen3_5 / dflash2
├── layers/                 Metal-backed layers (GDN, rotary, switch linear)
├── server/                 FastAPI app, CLI args, process launch
├── tokenizer/              Tokenizer + detokenizer workers
└── tests/                  Weight-free unit tests (pytest)

mini-sglang-mlx-kernel/     C++/Metal kernel extension
├── csrc/                   Kernel sources (.metal) + launch code (.cpp)
├── test_and_bench/         Correctness tests and microbenchmarks
└── docs/                   Kernel design notes
```

## Tests

```bash
pytest mini_sglang_mlx/tests/          # weight-free unit tests
cd mini-sglang-mlx-kernel && python test_and_bench/gdn_state.py
```

## Acknowledgements

This project is a port of [mini-sglang](https://github.com/sgl-project/mini-sglang)
by the SGLang team, and inherits its architecture and much of its scheduler
and cache-management code. The MLX execution layer and the Metal kernels are
original to this project. DFlash2 speculative decoding follows
[DFlash: Block Diffusion for Flash Speculative Decoding](https://github.com/z-lab/dflash).
