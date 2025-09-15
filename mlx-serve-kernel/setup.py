# Copyright © 2023-2024 Apple Inc.

from setuptools import setup

from mlx import extension

if __name__ == "__main__":
    setup(
        name="mlx_serve_kernel",
        version="0.0.0",
        description="Kernel library for mlx-serve.",
        ext_modules=[extension.CMakeExtension("mlx_serve_kernel._ext")],
        cmdclass={"build_ext": extension.CMakeBuild},
        packages=["mlx_serve_kernel"],
        package_data={"mlx_serve_kernel": ["*.so", "*.dylib", "*.metallib"]},
        zip_safe=False,
        python_requires=">=3.10",
    )