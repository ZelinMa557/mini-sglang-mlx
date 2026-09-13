#!/usr/bin/env bash
# Local single-process launch: Qwen3.8-27B (hybrid GDN + attention) with
# DFlash2 block-diffusion speculative decoding.
# set -euo pipefail

MODEL=mlx-community/Qwen3.8-27B-4bit
# DFlash2 draft trained against exactly this target.  Already in the
# ModelScope cache, so --use-modelscope resolves it with no download.
DRAFT=z-lab/Qwen3.8-27B-DFlash2

python -m mini_sglang_mlx \
  --use-modelscope \
  --model-path "$MODEL" \
  --kv-cache-gb 2.0 \
  --max-running-req 4 \
  --spec-algo dflash2 \
  --draft-path "$DRAFT" \
  --num-draft-tokens 7

# --num-draft-tokens 7 = the draft's training-time block_size (8) minus one:
# a single draft forward proposes K tokens, the target verifies all K+1.
# The checkpoint's own value is in its config.json (dflash_config.block_size);
# a mismatch does not error out, it just drafts garbage.
#
# --num-mamba-slots is deliberately unset, so the engine auto-sizes it to
# max_running_req * (2 + 1 for spec) = 12.  That is the real peak here: the
# 4 in-flight reqs hold a main slot each, target verify needs one scratch
# slot per req *at the same time*, and one snapshot per req stays available
# for the radix prefix cache.  Setting it below 2 * max_running_req leaves
# the pool permanently dry and the prefix cache nothing to keep.
#
# Two DFlash limitations to keep in mind (see mini_sglang_mlx/engine/dflash_engine.py):
#   * greedy verification only -- sampling params are ignored during verify;
#   * no chunked prefill, so a prompt whose uncached part exceeds
#     --max-prefill-length (default 8192) raises instead of chunking.

# Non-speculative baseline against the same target:
# python -m mini_sglang_mlx \
#   --use-modelscope \
#   --model-path "$MODEL" \
#   --kv-cache-gb 2.0 \
#   --num-mamba-slots 8 \
#   --max-running-req 4
