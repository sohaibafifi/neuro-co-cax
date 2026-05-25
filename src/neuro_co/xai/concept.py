"""Concept-bank contract + registry.

`ConceptFn` is a per-node binary labeller used by linear probes
(`fit_concept_probes` in `neuro-co-probe`) and PCA / ICA direction
matching (`discover_directions`). Problem packages define banks
and register them here; the attribution + probe layers don't know
about routing or scheduling specifics.

```python
# producer side (problem package)
from neuro_co.xai import ConceptBank, register_concept_bank

BANK = ConceptBank(
    problem="vrptw",
    concepts={"tight_tw": fn, ...},
    feature_keys=("locs", "demand", ...),
)
register_concept_bank(BANK)

# consumer side
from neuro_co.xai import concept_registry
bank = concept_registry.get("vrptw")
```
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Generic, Protocol, TypeVar, runtime_checkable

T = TypeVar("T")


@runtime_checkable
class ConceptFn(Protocol):
    """Per-node binary concept labeller.

    Implementations take an env TensorDict (or any indexable container)
    and return a `[B, N]` long tensor with values in `{0, 1, -1}`:
    `1` = concept holds, `0` = concept doesn't hold, `-1` = position
    should be ignored (e.g. depot for VRPTW, padded ops for JSSP).

    Return `None` to opt out cleanly when the TensorDict doesn't carry
    the fields this concept needs.
    """

    def __call__(self, td: Any) -> Any: ...


@dataclass(frozen=True)
class ConceptBank:
    """Per-problem bundle consumed by generic XAI tooling.

    Attributes
    ----------
    problem
        Short canonical name (`"vrptw"`, `"jssp"`, …).
    concepts
        Named per-node concept extractors.
    feature_keys
        TensorDict fields that attribution should differentiate
        against. Order is not significant.
    num_nodes_key
        Optional override for the field whose last axis equals N (the
        encoder's node count). Attribution falls back to its own
        detection logic when `None`.
    """

    problem: str
    concepts: dict[str, ConceptFn] = field(default_factory=dict)
    feature_keys: tuple[str, ...] = ()
    num_nodes_key: str | None = None
    # Optional adapter so AET / baseline drivers can report a single
    # "instance scale" integer in `energy_*.json` without hard-coding
    # routing-vs-scheduling layout. Receives `cfg.env` (a Hydra
    # DictConfig or any object with `.generator_params`).
    size_from_env_cfg: Callable[[Any], int] | None = None


class InstanceRegistry(Generic[T]):
    """Name → instance mapping. Sibling of `neuro_co.api.Registry` for
    values that aren't classes with no-arg constructors.
    """

    def __init__(self, kind: str) -> None:
        self._kind = kind
        self._items: dict[str, T] = {}

    def register(self, name: str, value: T) -> T:
        if name in self._items:
            raise ValueError(f"{self._kind} {name!r} already registered")
        self._items[name] = value
        return value

    def get(self, name: str) -> T:
        if name not in self._items:
            self._try_autoload_problems()
        if name not in self._items:
            available = ", ".join(sorted(self._items)) or "<none>"
            raise KeyError(f"{self._kind} {name!r} not registered. Available: {available}")
        return self._items[name]

    def _try_autoload_problems(self) -> None:
        """Lazy-import `neuro_co.problems` and run plug-in discovery.

        Soft dependency: when `neuro-co-problems` isn't installed (xai
        used as a standalone library) the import fails silently and
        `get()` raises its normal KeyError listing the empty registry.
        """
        import contextlib

        with contextlib.suppress(ImportError):
            from neuro_co.problems import load_plugins

            load_plugins()

    def names(self) -> list[str]:
        if not self._items:
            self._try_autoload_problems()
        return sorted(self._items)

    def __contains__(self, name: str) -> bool:
        if name not in self._items:
            self._try_autoload_problems()
        return name in self._items


concept_registry: InstanceRegistry[ConceptBank] = InstanceRegistry("concept_bank")


def register_concept_bank(bank: ConceptBank) -> ConceptBank:
    """Register `bank` under `bank.problem`. Idempotent re-registration disallowed."""
    return concept_registry.register(bank.problem, bank)


def infer_problem_name(env_target: str, *, aliases: dict[str, str] | None = None) -> str | None:
    """Substring-match `env_target` against names in `concept_registry`.

    `aliases` maps registered names to *additional* substrings to match
    (e.g. `{"vrptw": "cvrptw"}`). Returns `None` if no name matches.
    Pure: no env / config dependency, easy to test.
    """
    target = env_target.lower()
    aliases = aliases or {}
    for name in concept_registry.names():
        if name in target:
            return name
        alias = aliases.get(name)
        if alias and alias in target:
            return name
    return None
