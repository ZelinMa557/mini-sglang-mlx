#!/usr/bin/env python3
"""
Download MTP (Multi Token Predict) layers from Qwen3.5 model, saving only these
tensors to a single .safetensors file (MLX format, preserves bfloat16).

MTP weights are those whose names start with "mtp." This script minimizes disk usage by:
1. Fetching model.safetensors.index.json to determine which shards contain MTP weights
2. Downloading only those shard files
3. Extracting MTP tensors and saving to output
4. Deleting temporary downloaded files

For gated Hugging Face models, set HF_TOKEN env var or use huggingface-cli login.
"""

from __future__ import annotations

import argparse
import json
import logging
import tempfile
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# MTP = Multi Token Predict, weights start with "mtp."
MTP_KEY_PREFIX = "mtp."


def is_mtp_key(key: str) -> bool:
    """Check if a weight key belongs to an MTP (Multi Token Predict) layer."""
    return key.startswith(MTP_KEY_PREFIX)


def get_index_path(repo_files: list[str]) -> str | None:
    """Return the index file path if it exists."""
    for f in repo_files:
        if f == "model.safetensors.index.json":
            return f
    return None


def download_hf_file(repo_id: str, filename: str, local_dir: Path, revision: str = "main") -> Path:
    """Download a single file from Hugging Face Hub."""
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        revision=revision,
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
    )
    return Path(path)


def download_modelscope_file(
    model_id: str, file_path: str, local_dir: Path, revision: str = "master"
) -> Path:
    """Download a single file from ModelScope."""
    from modelscope.hub.file_download import model_file_download

    path = model_file_download(
        model_id=model_id,
        file_path=file_path,
        revision=revision,
        local_dir=str(local_dir),
    )
    return Path(path)


def list_repo_files_hf(repo_id: str, revision: str = "main") -> list[str]:
    """List files in a Hugging Face repo."""
    from huggingface_hub import list_repo_files

    return list(list_repo_files(repo_id, revision=revision))


def list_repo_files_modelscope(model_id: str, revision: str = "master") -> list[str]:
    """List files in a ModelScope repo."""
    from modelscope.hub.api import HubApi

    api = HubApi()
    model_files = api.get_model_files(model_id, revision=revision, recursive=True)
    if not model_files:
        return []
    # Handle both "Path" and "path" keys (API may vary by version)
    paths = []
    for f in model_files:
        p = f.get("Path") or f.get("path")
        if p:
            paths.append(p)
    return paths


def load_safetensors_mtp_keys(path: Path) -> dict:
    """Load a safetensors file and return only MTP tensors. Uses MLX (supports bfloat16)."""
    import mlx.core as mx

    all_weights = mx.load(str(path))
    return {k: v for k, v in all_weights.items() if is_mtp_key(k)}


def save_safetensors(tensors: dict, path: Path) -> None:
    """Save tensors to a safetensors file in MLX format."""
    import mlx.core as mx

    mx.save_safetensors(str(path), tensors)


def run_huggingface(model_id: str, output_path: Path, temp_dir: Path, revision: str = "main") -> None:
    """Download MTP layers from Hugging Face."""
    from huggingface_hub import hf_hub_download

    logger.info("Listing repo files on Hugging Face...")
    repo_files = list_repo_files_hf(model_id, revision)
    index_file = get_index_path(repo_files)

    if index_file:
        logger.info("Found index, downloading model.safetensors.index.json...")
        index_path = hf_hub_download(
            repo_id=model_id,
            filename=index_file,
            revision=revision,
            local_dir=str(temp_dir),
            local_dir_use_symlinks=False,
        )
        with open(index_path) as f:
            index = json.load(f)
        weight_map = index.get("weight_map", {})
    else:
        # No index: single model.safetensors file
        safetensor_files = [f for f in repo_files if f.endswith(".safetensors")]
        if not safetensor_files:
            raise FileNotFoundError("No safetensors files found in repo")
        if len(safetensor_files) > 1:
            raise ValueError(
                "Multiple safetensors files but no index - cannot determine weight mapping"
            )
        weight_map = None  # Will download single file

    if weight_map is not None:
        # Find which files contain MTP weights
        mtp_files = set()
        for key, fname in weight_map.items():
            if is_mtp_key(key):
                mtp_files.add(fname)
        if not mtp_files:
            raise ValueError("No MTP (mtp.*) weights found in model index")
        logger.info("MTP weights spread across %d file(s): %s", len(mtp_files), sorted(mtp_files))
        files_to_download = list(mtp_files)
    else:
        files_to_download = safetensor_files

    all_mtp_tensors = {}
    downloaded_paths = []

    for fname in files_to_download:
        logger.info("Downloading %s...", fname)
        path = download_hf_file(model_id, fname, temp_dir, revision)
        downloaded_paths.append(path)
        mtp_tensors = load_safetensors_mtp_keys(path)
        all_mtp_tensors.update(mtp_tensors)
        logger.info("  Extracted %d MTP tensors from this file", len(mtp_tensors))

    if not all_mtp_tensors:
        raise ValueError("No MTP tensors found in downloaded files")

    logger.info("Saving %d MTP tensors to %s", len(all_mtp_tensors), output_path)
    save_safetensors(all_mtp_tensors, output_path)

    for p in downloaded_paths:
        if p.exists():
            p.unlink()
            logger.debug("Removed %s", p)


def run_modelscope(model_id: str, output_path: Path, temp_dir: Path, revision: str = "master") -> None:
    """Download MTP layers from ModelScope."""
    logger.info("Listing repo files on ModelScope...")
    repo_files = list_repo_files_modelscope(model_id, revision)
    if not repo_files:
        raise FileNotFoundError("Could not list ModelScope repo files - check model_id")

    index_file = get_index_path(repo_files)

    if index_file:
        logger.info("Found index, downloading model.safetensors.index.json...")
        index_path = download_modelscope_file(model_id, index_file, temp_dir, revision)
        with open(index_path) as f:
            index = json.load(f)
        weight_map = index.get("weight_map", {})
    else:
        safetensor_files = [f for f in repo_files if f.endswith(".safetensors")]
        if not safetensor_files:
            raise FileNotFoundError("No safetensors files found in repo")
        if len(safetensor_files) > 1:
            raise ValueError(
                "Multiple safetensors files but no index - cannot determine weight mapping"
            )
        weight_map = None
        files_to_download = safetensor_files

    if weight_map is not None:
        mtp_files = set()
        for key, fname in weight_map.items():
            if is_mtp_key(key):
                mtp_files.add(fname)
        if not mtp_files:
            raise ValueError("No MTP (mtp.*) weights found in model index")
        logger.info("MTP weights spread across %d file(s): %s", len(mtp_files), sorted(mtp_files))
        files_to_download = list(mtp_files)

    all_mtp_tensors = {}
    downloaded_paths = []

    for fname in files_to_download:
        logger.info("Downloading %s...", fname)
        path = download_modelscope_file(model_id, fname, temp_dir, revision)
        downloaded_paths.append(path)
        mtp_tensors = load_safetensors_mtp_keys(path)
        all_mtp_tensors.update(mtp_tensors)
        logger.info("  Extracted %d MTP tensors from this file", len(mtp_tensors))

    if not all_mtp_tensors:
        raise ValueError("No MTP tensors found in downloaded files")

    logger.info("Saving %d MTP tensors to %s", len(all_mtp_tensors), output_path)
    save_safetensors(all_mtp_tensors, output_path)

    for p in downloaded_paths:
        if p.exists():
            p.unlink()
            logger.debug("Removed %s", p)


def main():
    parser = argparse.ArgumentParser(
        description="Download Qwen3.5 MTP (Multi Token Predict) layers to a single .safetensors file"
    )
    parser.add_argument(
        "--model_id",
        type=str,
        help="Model ID on Hugging Face or ModelScope (e.g. Qwen/Qwen3.5-0.5B)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output path for the .safetensors file",
    )
    parser.add_argument(
        "--use-modelscope",
        action="store_true",
        help="Use ModelScope instead of Hugging Face for download",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help="Model revision/branch. Default: 'main' for HF, 'master' for ModelScope.",
    )
    args = parser.parse_args()

    # ModelScope uses 'master' as default branch, Hugging Face uses 'main'
    revision = args.revision or ("master" if args.use_modelscope else "main")

    output_path = args.output
    if output_path.suffix != ".safetensors":
        output_path = Path(str(output_path) + ".safetensors")
    output_path = output_path.resolve()

    with tempfile.TemporaryDirectory(prefix="mtp_download_") as temp_dir:
        temp_path = Path(temp_dir)
        if args.use_modelscope:
            run_modelscope(args.model_id, output_path, temp_path, revision)
        else:
            run_huggingface(args.model_id, output_path, temp_path, revision)

    logger.info("Done. MTP layers saved to %s", output_path)


if __name__ == "__main__":
    main()
