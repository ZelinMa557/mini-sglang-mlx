from __future__ import annotations

import os
from pathlib import Path

# Files we actually need at load time — keeping the download set small
# helps a lot on flaky / slow connections.  Matches the historical
# ``mlx_serve.models.__init__._resolve_model_path`` set, plus the
# DFlash-style nested ``dflash_config`` JSON which lives in the same
# ``config.json``.
_DEFAULT_ALLOW_PATTERNS = [
    "*.json",
    "model*.safetensors",
    "*.py",
    "tokenizer.model",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
    "chat_template*",
]


def resolve_repo(
    path_or_id: str,
    use_modelscope: bool = False,
) -> str:
    """Return a local directory for ``path_or_id``.

    Resolution rules, in order:

    1. ``~`` is expanded.
    2. If the expanded path already exists on the local filesystem
       (file *or* directory), return it as-is — the caller handles
       the distinction.  This lets users mix local paths with repo
       IDs across CLI flags without a separate "is-local" flag.
    3. Otherwise treat ``path_or_id`` as a repo ID and download a
       minimal snapshot via either
       :func:`huggingface_hub.snapshot_download` (default) or
       :func:`modelscope.hub.snapshot_download.snapshot_download`
       when ``use_modelscope=True``.

    Args:
        path_or_id: A local path or remote repo ID.
        use_modelscope: Use ModelScope as the hub backend instead
            of HuggingFace.  Useful in regions where HF access is
            slow / blocked.  Only consulted when ``path_or_id`` is
            *not* a local path.

    Returns:
        Absolute local path (always a directory for remote IDs;
        whatever the user supplied for local paths).

    Raises:
        RuntimeError: ``use_modelscope=True`` but the ``modelscope``
            package isn't installed.
    """
    expanded = os.path.expanduser(str(path_or_id))
    if Path(expanded).exists():
        return expanded

    if use_modelscope:
        try:
            from modelscope.hub.snapshot_download import snapshot_download as ms_download
        except ImportError as e:
            raise RuntimeError(
                "use_modelscope=True requires the `modelscope` package. "
                "Install it with: pip install modelscope"
            ) from e
        return str(ms_download(
            str(path_or_id),
            allow_patterns=_DEFAULT_ALLOW_PATTERNS,
        ))

    from huggingface_hub import snapshot_download as hf_download

    return str(hf_download(
        str(path_or_id),
        allow_patterns=_DEFAULT_ALLOW_PATTERNS,
    ))


__all__ = ["resolve_repo"]
