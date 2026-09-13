# Copyright © 2023-2024 Apple Inc.

from setuptools import setup

from mlx import extension

if __name__ == "__main__":
    setup(
        name="mini_sglang_mlx_kernel",
        version="0.0.0",
        description="Kernel library for mini-sglang-mlx.",
        ext_modules=[extension.CMakeExtension("mini_sglang_mlx_kernel._ext")],
        cmdclass={"build_ext": extension.CMakeBuild},
        packages=["mini_sglang_mlx_kernel"],
        package_data={"mini_sglang_mlx_kernel": ["*.so", "*.dylib", "*.metallib"]},
        zip_safe=False,
        python_requires=">=3.10",
    )