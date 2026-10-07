
"""Unified optional import utilities for lmms-eval.

Usage Examples
--------------
1. Optional import with fallback:
    >>> from lmms_eval.imports import optional_import
    >>> VideoReader, HAS_DECORD = optional_import("decord", "VideoReader")

2. Check if a package is available:
    >>> from lmms_eval.imports import is_package_available
    >>> if is_package_available("textual"):
    ...     from textual.app import App

3. Require a package with helpful error message:
    >>> from lmms_eval.imports import require_package
    >>> require_package("textual", extras="tui")

4. Lazy module-level exports:
    >>> __getattr__ = make_lazy_getattr({
    ...     "run_tui": ("lmms_eval.tui.cli", "main"),
    ... })
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import warnings
from functools import lru_cache
from typing import Any, Optional, Tuple


@lru_cache(maxsize=128)
def is_package_available(package_name: str) -> bool:
    """Check if a package is installed (cached)."""
    return importlib.util.find_spec(package_name) is not None


def optional_import(
    module_name: str,
    attribute: Optional[str] = None,
    fallback: Any = None,
) -> Tuple[Any, bool]:
    """Import a module or attribute optionally, returning fallback if unavailable.

    Args:
        module_name: Full module path (e.g., "decord" or "qwen_vl_utils")
        attribute: Optional attribute to get from the module
        fallback: Value to return if import fails (default: None)

    Returns:
        Tuple of (imported_object_or_fallback, is_available)

    Examples:
        >>> VideoReader, has_decord = optional_import("decord", "VideoReader")
        >>> cpu, _ = optional_import("decord", "cpu")
    """
    try:
        module = importlib.import_module(module_name)
        if attribute is not None:
            return getattr(module, attribute), True
        return module, True
    except (ImportError, AttributeError):
        return fallback, False


def resolve_qwen_vl_process_vision_info() -> Tuple[Any, bool]:
    """Load ``process_vision_info`` for Qwen-VL / Qwen3.5 eval.

    When environment variable ``LMMS_USE_LOCAL_QWEN_VL_UTILS`` is truthy (``1``,
    ``true``, or ``yes``), import from ``model.qwen_vl_utils.vision_process`` (the
    copy under this repo). Prepend ``<repo>/saver`` to ``PYTHONPATH`` so
    the ``model`` package resolves, matching training.

    Otherwise use the PyPI package ``qwen_vl_utils``. If local import was
    requested but fails, falls back to PyPI with a warning.
    """
    prefer_local = os.environ.get("LMMS_USE_LOCAL_QWEN_VL_UTILS", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )
    if prefer_local:
        try:
            from model.qwen_vl_utils.vision_process import process_vision_info

            return process_vision_info, True
        except ImportError:
            warnings.warn(
                "LMMS_USE_LOCAL_QWEN_VL_UTILS is set but "
                "model.qwen_vl_utils.vision_process could not be imported; "
                "falling back to qwen_vl_utils (pip). "
                "Ensure PYTHONPATH includes the saver directory.",
                UserWarning,
                stacklevel=2,
            )
    return optional_import("qwen_vl_utils", "process_vision_info")


class MissingOptionalDependencyError(ImportError):
    """Raised when a required optional dependency is missing."""

    def __init__(
        self,
        package: str,
        extras: Optional[str] = None,
        feature: Optional[str] = None,
    ):
        if extras:
            install_cmd = f"pip install lmms_eval[{extras}]"
        else:
            install_cmd = f"pip install {package}"

        feature_msg = f" for {feature}" if feature else ""
        message = f"'{package}' is required{feature_msg} but not installed. Install with: {install_cmd}"
        super().__init__(message)


def require_package(
    package: str,
    extras: Optional[str] = None,
    feature: Optional[str] = None,
) -> None:
    """Require an optional package, raising helpful error if missing."""
    if not is_package_available(package):
        raise MissingOptionalDependencyError(package, extras, feature)


def make_lazy_getattr(lazy_imports: dict[str, tuple[str, str]]):
    """Create a __getattr__ function for lazy module-level imports.

    Args:
        lazy_imports: Dict mapping attribute names to (module, attribute) tuples

    Example:
        >>> __getattr__ = make_lazy_getattr({
        ...     "run_tui": ("lmms_eval.tui.cli", "main"),
        ... })
    """

    def __getattr__(name: str) -> Any:
        if name in lazy_imports:
            module_path, attr_name = lazy_imports[name]
            module = importlib.import_module(module_path)
            return getattr(module, attr_name)
        raise AttributeError(f"module has no attribute {name!r}")

    return __getattr__
