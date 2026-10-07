"""Resolve benchmark video directories from LMMS_DATA_ROOT or HF_HOME."""

from __future__ import annotations

import os
from typing import Optional



def data_root() -> str:
    return os.path.expanduser(os.environ.get("LMMS_DATA_ROOT") or os.environ.get("HF_HOME") or "~/.cache/huggingface")


def resolve_cache_dir(cache_name: str, base: Optional[str] = None) -> str:
    """Resolve relative cache directories; preserve explicit absolute paths."""
    if not cache_name:
        raise ValueError("cache_name is empty")
    name = os.path.expanduser(cache_name)
    root = os.path.expanduser(base) if base is not None else data_root()
    if os.path.isabs(name):
        return name
    return os.path.join(root, name)
