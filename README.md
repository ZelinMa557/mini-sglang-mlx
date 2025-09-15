# mlx-serve

mlx-serve is a (at current stage, will be a) high performance LLM inference engine for Mac GPUs.

It support the following features:

* continous batching

* paged attention

* radix tree based prefix cache

* high performance metal & cpu kernels

* open-ai compitable apis

Support models:

* qwen3, qwen3-moe

* gpt-oss

Roadmap:

* paged attention kernel, supports sliding window attention and attention sink (P0)

* model runner (P0)

* openai compitable api (P0)

* moe related kernels (P1)

* support qwen3-moe and gpt-oss (P1)