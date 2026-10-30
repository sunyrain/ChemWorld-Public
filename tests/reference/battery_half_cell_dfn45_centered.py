"""Explicit constant-reference cancellation in one actual positive FV gradient.

The pinned model already stores the positive potential relative to its initial
OCV. This policy removes that fixed scalar from the gradient's input before
floating evaluation. It preserves the original Variable as the BC lookup key.
"""

import math
from copy import deepcopy

import numpy as np

POLICY = "centered-positive-potential"
TARGET = "Positive electrode potential [V]"


def validate_gradient_policy(policy):
    if policy is None:
        return None
    if type(policy) is not str or policy != POLICY:
        raise ValueError("gradient policy must be None or centered-positive-potential")
    return policy


def _finite_scalar(symbol, name):
    try:
        if not symbol.is_constant() or not symbol.evaluates_to_number():
            raise ValueError("not a constant scalar")
        value = np.asarray(symbol.evaluate())
        if value.size != 1 or value.dtype.kind not in "fiu":
            raise ValueError("not a real scalar")
        result = float(value.reshape(-1)[0])
        if not math.isfinite(result):
            raise ValueError("not finite")
        return result
    except Exception as exc:
        raise ValueError(f"centered gradient requires a finite real constant {name}") from exc


def make_spatial_methods(pybamm, model, policy):
    """Return explicit copied methods, or None to leave the default path untouched."""
    if validate_gradient_policy(policy) is None:
        return None
    if pybamm.__version__ != "25.4.2":
        raise RuntimeError("centered gradient policy requires pinned PyBaMM25.4.2")
    defaults = model.default_spatial_methods
    if not isinstance(defaults, dict):
        raise ValueError("centered gradient requires actual default spatial-method mapping")
    key = "macroscale" if "macroscale" in defaults else "positive electrode"
    if key not in defaults or type(defaults[key]) is not pybamm.FiniteVolume:
        raise ValueError("centered gradient requires an unmodified default positive FV method")
    methods = deepcopy(defaults)

    class CenteredPositivePotentialFiniteVolume(pybamm.FiniteVolume):
        def __init__(self, options):
            super().__init__(options=options)
            self.centered_gradient_receipts = []

        def gradient(self, symbol, discretised_symbol, boundary_conditions):
            if not isinstance(symbol, pybamm.Variable) or symbol.name != TARGET:
                return super().gradient(symbol, discretised_symbol, boundary_conditions)
            if symbol.domain != ["positive electrode"]:
                raise ValueError("centered positive potential has an unexpected spatial domain")
            reference, scale = symbol.reference, symbol.scale
            reference_value = _finite_scalar(reference, "reference")
            scale_value = _finite_scalar(scale, "scale")
            if scale_value == 0:
                raise ValueError("centered gradient requires a nonzero coordinate scale")
            if symbol not in boundary_conditions:
                raise ValueError("centered gradient requires the original Variable BC key")
            bcs = boundary_conditions[symbol]
            if any(
                side not in bcs or len(bcs[side]) != 2 or bcs[side][1] != "Neumann"
                for side in ("left", "right")
            ):
                raise ValueError("centered gradient requires original left/right Neumann BCs")
            centered = discretised_symbol - reference
            states = {
                node.id: node
                for node in centered.pre_order()
                if isinstance(node, pybamm.StateVector)
            }
            if len(states) != 1:
                raise ValueError("centered gradient did not reduce to one original raw StateVector")
            state = next(iter(states.values()))
            expected = scale * state
            if (
                state.domains != symbol.domains
                or centered != expected
                or discretised_symbol != reference + expected
            ):
                raise ValueError(
                    "processed constant reference did not cancel exactly before gradient"
                )
            # All checks precede the original FV operation. The original symbol
            # remains its key, so its actual Neumann edge values are inserted by
            # exactly the same pinned code. No boundary datum is shifted or copied.
            result = super().gradient(symbol, centered, boundary_conditions)
            self.centered_gradient_receipts.append(
                {
                    "policy": POLICY,
                    "variable": symbol.name,
                    "domains": deepcopy(symbol.domains),
                    "original_discretised_expression": str(discretised_symbol),
                    "centered_discretised_expression": str(centered),
                    "reference_V": reference_value,
                    "scale": scale_value,
                    "boundary_lookup": "original unchanged Variable key",
                    "boundary_types": {side: bcs[side][1] for side in ("left", "right")},
                    "boundary_expressions": {side: str(bcs[side][0]) for side in ("left", "right")},
                    "scope": "positive solid Neumann gradient only; "
                    "no IC, BV, output or event rewrite",
                }
            )
            return result

    methods[key] = CenteredPositivePotentialFiniteVolume(deepcopy(defaults[key].options))
    return methods
