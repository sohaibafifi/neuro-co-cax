"""Smoke tests: package imports + minimal dispatcher behaviour."""

from __future__ import annotations


def test_cax_imports() -> None:
    """The four CAX entrypoints must import without optional extras."""
    from neuro_co.cax import (  # noqa: F401
        adjudicate,
        constraint_map,
        cp_counterfactual,
        cp_minimal_subset,
        lambda_attribution,
    )


def test_xai_shim_imports() -> None:
    """The vendored xai shim exposes the symbols cax depends on."""
    from neuro_co.xai.attribution import (  # noqa: F401
        AttributionTrace,
        gradient_attribution,
        step_logits,
    )


def test_constraint_map_problems() -> None:
    """Every paper-cited problem has at least one constraint family."""
    from neuro_co.cax.constraint_map import PROBLEM_CONSTRAINTS

    for name in ("vrptw", "op", "fjsp"):
        assert name in PROBLEM_CONSTRAINTS, f"missing {name}"
        assert PROBLEM_CONSTRAINTS[name], f"empty family list for {name}"


def test_pac_sample_count_bonferroni() -> None:
    """Bonferroni-tight sample size matches the formula in the paper."""
    from math import log, ceil

    from neuro_co.cax.cp_minimal_subset import pac_sample_count

    eps, delta = 0.2, 0.2
    k_max = 25
    expected = ceil(log(2 * k_max / delta) / (2 * eps * eps))
    assert pac_sample_count(eps, delta, n_tests=k_max) == expected
    assert pac_sample_count(eps, delta, n_tests=1) < pac_sample_count(eps, delta, n_tests=k_max)


def test_lp_dual_aggregations() -> None:
    """All three aggregations are accepted by the LP backend wrapper."""
    from neuro_co.cax.duals.lp import _agg_abs_dual

    class _Row:
        def __init__(self, v: float) -> None:
            self._v = v

        def dual_value(self) -> float:
            return self._v

    rows = [_Row(0.1), _Row(0.5), _Row(0.4)]
    assert abs(_agg_abs_dual(rows, "mean") - (0.1 + 0.5 + 0.4) / 3) < 1e-9
    assert abs(_agg_abs_dual(rows, "sum") - 1.0) < 1e-9
    assert abs(_agg_abs_dual(rows, "max") - 0.5) < 1e-9
