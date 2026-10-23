"""Explicit circular-pipe friction relations; Darcy, not Fanning, factors.

This kernel leaves the rollout transport module's transitional blending intact.
An engineering caller must explicitly select its transition convention.
"""

from __future__ import annotations

from math import isfinite, log10

from chemworld.physchem.flow_meters import _engineering_root, _positive


def colebrook_friction_factor(reynolds: float, relative_roughness: float = 0.0) -> float:
    """Solve the turbulent Colebrook equation using the common residual solver.

    The equation is returned for positive Reynolds numbers without assigning a
    laminar/turbulent regime; use ``pipe_friction_factor`` for that decision.
    """
    _positive(reynolds, "Reynolds number")
    if not isfinite(relative_roughness) or not 0 <= relative_roughness < 1:
        raise ValueError("relative roughness must lie in [0, 1)")
    initial = -1.8 * log10((relative_roughness / 3.7) ** 1.11 + 6.9 / reynolds)
    inverse_sqrt = _engineering_root(
        lambda value: value + 2 * log10(relative_roughness / 3.7 + 2.51 * value / reynolds),
        max(0.1, initial),
        1e-14,
        100.0,
        name="inverse_sqrt_darcy_friction",
        unit="dimensionless",
        scale=10.0,
    )
    return inverse_sqrt**-2


def pipe_friction_factor(
    reynolds: float,
    relative_roughness: float = 0.0,
    *,
    transition_reynolds: float = 2040.0,
    method: str = "Colebrook",
    darcy: bool = True,
    check_recommended_ranges: bool = True,
) -> float:
    """64/Re below the declared transition, exact Colebrook at/above it.

    The default 2040 convention is used by the engineering two-phase reference.
    It is discontinuous and does not model intermittent transitional flow.
    """
    _positive(reynolds, "Reynolds number")
    _positive(transition_reynolds, "transition Reynolds number")
    if not isfinite(relative_roughness) or not 0 <= relative_roughness < 1:
        raise ValueError("relative roughness must lie in [0, 1)")
    from chemworld.physchem.straight_friction import (
        STRAIGHT_FRICTION_RANGES,
        straight_friction_correlation,
    )

    if method not in STRAIGHT_FRICTION_RANGES:
        raise ValueError(f"unknown friction method {method!r}")
    if reynolds < transition_reynolds or method == "laminar":
        return (64 if darcy else 16) / reynolds
    return straight_friction_correlation(
        reynolds,
        relative_roughness,
        method=method,
        darcy=darcy,
        check_recommended_ranges=check_recommended_ranges,
    )
