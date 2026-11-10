"""
src/rl_policy/__init__.py

Public API surface for the AetherMed 3.0 offline reinforcement learning
dosing policy module.

Imports are resolved lazily via `__getattr__` so that torch-free submodules
(e.g. `dataset.py`, used for pure data generation) remain importable and
runnable via `python -m src.rl_policy.dataset` even in environments where
PyTorch is not installed. Torch-dependent symbols (`CQLAgent`,
`OffPolicyEvaluator`, etc.) are only imported -- and thus only require
torch -- at the moment they are actually accessed.
"""

from __future__ import annotations

from typing import Any, List

__all__ = [
    "ClinicalTrajectoryDataset",
    "CQLAgent",
    "OffPolicyEvaluator",
    "normalize_action",
    "denormalize_action",
    "train_cql_main",
    "evaluate_ope_main",
]

# Maps each public symbol to the submodule it lives in and its attribute
# name within that submodule (attribute name matches every symbol here
# except the two CLI entrypoints, which are both named `main` in-module).
_LAZY_ATTRS = {
    "ClinicalTrajectoryDataset": ("src.rl_policy.dataset", "ClinicalTrajectoryDataset"),
    "CQLAgent": ("src.rl_policy.train_cql", "CQLAgent"),
    "normalize_action": ("src.rl_policy.train_cql", "normalize_action"),
    "denormalize_action": ("src.rl_policy.train_cql", "denormalize_action"),
    "train_cql_main": ("src.rl_policy.train_cql", "main"),
    "OffPolicyEvaluator": ("src.rl_policy.evaluate_ope", "OffPolicyEvaluator"),
    "evaluate_ope_main": ("src.rl_policy.evaluate_ope", "main"),
}


def __getattr__(name: str) -> Any:
    """Lazily resolve public package attributes on first access.

    Args:
        name: Attribute name being accessed on the `src.rl_policy` package.

    Returns:
        The resolved attribute (class, function, etc.).

    Raises:
        AttributeError: If `name` is not a recognized public attribute.
    """
    if name in _LAZY_ATTRS:
        module_path, attr_name = _LAZY_ATTRS[name]
        import importlib

        module = importlib.import_module(module_path)
        return getattr(module, attr_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> List[str]:
    """Expose lazily-defined attributes to `dir()` and autocompletion."""
    return sorted(list(globals().keys()) + __all__)