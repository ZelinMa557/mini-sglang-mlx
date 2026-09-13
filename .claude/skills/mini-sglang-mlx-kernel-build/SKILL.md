---
name: mini-sglang-mlx-kernel-build
description: How to build, test and debug the mini-sglang-mlx-kernel C++/Metal extension for the mini-sglang-mlx LLM inference engine. Use this when the user asks about compiling kernels, running kernel tests, or debugging Metal shaders in the mini-sglang-mlx project.
---

# mini-sglang-mlx-kernel Build & Debug

## Quick Build

From the `mini-sglang-mlx-kernel/` directory:

```bash
pip install -e .
```

- Requires `mlx` and `nanobind` to be installed in the current Python environment.
- Uses CMake + Metal shader compilation under the hood.
- Auto-discovers new `.cpp` and `.metal` files via `GLOB`, so adding new kernels does not require editing `CMakeLists.txt`.

## Project Layout

```
mini-sglang-mlx-kernel/
├── csrc/
│   ├── bindings.cpp          # nanobind Python bindings
│   ├── gdn_state.h/cpp       # GDN prefill/decode kernel
│   ├── gdn_verify.h/cpp      # GDN target-verify (MTP) kernel
│   ├── gdn_state.metal       # Metal shaders
│   └── ...
├── mini_sglang_mlx_kernel/
│   └── __init__.py           # Python exports
├── test_and_bench/           # Correctness & perf tests
└── CMakeLists.txt
```

## Testing

Run a specific test:

```bash
cd mini-sglang-mlx-kernel
python test_and_bench/gdn_verify.py
```

Tests use small random inputs and compare against a naive NumPy reference implementation.

## Common Issues

| Issue | Fix |
|-------|-----|
| `kernel not found` at runtime | Ensure the `.metal` file contains `instantiate_kernel(...)` for the requested template specialization. |
| `Metal backend required` | The machine is not macOS or Metal is unavailable. This extension only supports Apple Silicon. |
| Changes not picked up | Run `pip install -e . --no-build-isolation --force-reinstall --no-deps` to force a rebuild. |

## Adding a New Kernel

1. Add `.h`/`.cpp` in `csrc/` with the C++ primitive and `eval_gpu()` launch code.
2. Add the shader in `csrc/*.metal` with `instantiate_kernel(...)`.
3. Expose in `csrc/bindings.cpp` via `m.def(...)`.
4. Export in `mini_sglang_mlx_kernel/__init__.py`.
5. Add a correctness test in `test_and_bench/`.
6. Build with `pip install -e .`.
