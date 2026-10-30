"""Excess thermodynamics and analytic responses of existing activity declarations.

These are immutable evaluations, not a second phase inventory. Energies refer
to ideal mixing at the same T, P and composition; no absolute reference is
created by an activity model.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, fields
from functools import partial
from math import exp, isfinite
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from chemworld.physchem.equilibrium import ActivityModelSpec

from chemworld.physchem.activity_kernels import R, observable, thermal_gibbs
from chemworld.physchem.caloric_math import partial_derivative
from chemworld.physchem.composition_math import (
    composition_gradient_to_moles,
    partial_molar_composition_jacobian,
    partial_molar_properties,
)


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({k: _freeze(v) for k, v in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(v) for v in value)
    return value


def _json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {k: _json(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [_json(v) for v in value]
    return value


class _ImmutableReport:
    def __post_init__(self) -> None:
        for item in fields(cast(Any, self)):
            object.__setattr__(self, item.name, _freeze(getattr(self, item.name)))

    def to_dict(self) -> dict[str, Any]:
        return {item.name: _json(getattr(self, item.name)) for item in fields(cast(Any, self))}


@dataclass(frozen=True)
class ActivityThermodynamicReport(_ImmutableReport):
    model_id: str
    model: str
    component_ids: tuple[str, ...]
    temperature_K: float
    composition: Mapping[str, float]
    gibbs_excess_J_mol: float
    enthalpy_excess_J_mol: float
    entropy_excess_J_mol_K: float
    heat_capacity_excess_J_mol_K: float
    dGE_dT: float
    d2GE_dT2: float
    d3GE_dT3: float
    dGE_dxs: tuple[float, ...]
    d2GE_dxixjs: tuple[tuple[float, ...], ...]
    d2GE_dTdxs: tuple[float, ...]
    activity_coefficients: Mapping[str, float]
    log_activity_coefficients: Mapping[str, float]
    dln_gamma_dT: Mapping[str, float]
    dln_gamma_dns: tuple[tuple[float, ...], ...]
    partial_molar_gibbs_excess_J_mol: Mapping[str, float]
    partial_molar_enthalpy_excess_J_mol: Mapping[str, float]
    partial_molar_entropy_excess_J_mol_K: Mapping[str, float]
    parameter_contract: Mapping[str, object]
    mole_derivative_basis_mol: float = 1.0
    reference: str = "ideal mixture at the same T, P and composition; excess quantities only"


@dataclass(frozen=True)
class ActivityInfiniteDilutionReport(_ImmutableReport):
    model_id: str
    temperature_K: float
    activity_coefficients: Mapping[str, float]
    log_activity_coefficients: Mapping[str, float]
    solvent_compositions: Mapping[str, Mapping[str, float]]
    limit: str = "exact zero solute mole fraction; remaining solvent ratios preserved"


def validated_state(
    spec: ActivityModelSpec, composition: Mapping[str, float], temperature_K: float
) -> tuple[float, ...]:
    # Import lazily so the existing equilibrium module can delegate its gamma
    # evaluations without a circular module initialization.
    from chemworld.physchem.equilibrium import ActivityModelSpec, _composition_vector

    if not isinstance(spec, ActivityModelSpec):
        raise TypeError("activity thermodynamics requires ActivityModelSpec")
    if isinstance(temperature_K, bool) or not isfinite(temperature_K) or temperature_K <= 0:
        raise ValueError("temperature_K must be positive and finite")
    if any(isinstance(v, bool) for v in composition.values()):
        raise ValueError("composition values must be real amounts")
    xs = _composition_vector(spec.component_ids, composition)
    if any(not isfinite(x) for x in xs) or sum(xs) <= 0:
        raise ValueError("normalized composition must be finite with positive total")
    return xs


def _parameter_contract(spec: ActivityModelSpec) -> dict[str, Any]:
    symmetric = []
    if spec.model == "margules":
        for i, left in enumerate(spec.component_ids):
            for right in spec.component_ids[i + 1 :]:
                keys = (f"A:{left}|{right}", f"A:{right}|{left}")
                if sum(k in spec.parameters for k in keys) == 1:
                    symmetric.append(next(k for k in keys if k in spec.parameters))
    return {
        "parameters": dict(spec.parameters),
        "provenance": spec.provenance,
        "margules_symmetric_single_direction": symmetric,
        "free_x_derivatives": "independent x coordinates in the declared GE expression",
        "mole_derivatives": "normalized x=n/sum(n); all other mole amounts fixed",
    }


def activity_thermodynamic_report(
    spec: ActivityModelSpec, composition: Mapping[str, float], *, temperature_K: float
) -> ActivityThermodynamicReport:
    xs = validated_state(spec, composition, temperature_K)
    ids = spec.component_ids
    xnames = tuple(f"x:{i}" for i in range(len(xs)))
    values = {"T": temperature_K, **dict(zip(xnames, xs, strict=True))}

    def energy(v: Any) -> Any:
        return thermal_gibbs(spec, [v[name] for name in xnames], v["T"])

    ge, gt, gtt, gttt = (thermal_gibbs(spec, xs, temperature_K, order=k) for k in range(4))
    gx = tuple(partial_derivative(energy, values, (name,)) for name in xnames)
    gxx = tuple(
        tuple(partial_derivative(energy, values, (left, right)) for right in xnames)
        for left in xnames
    )
    gtx = tuple(partial_derivative(energy, values, ("T", name)) for name in xnames)
    partial_g = partial_molar_properties(ge, gx, xs)
    partial_gt = partial_molar_properties(gt, gtx, xs)
    log_gamma = tuple(v / (R * temperature_K) for v in partial_g)
    try:
        gamma = tuple(exp(v) for v in log_gamma)
    except OverflowError as exc:
        raise ValueError("activity coefficient is outside numerical range") from exc
    if any(v <= 0 or not isfinite(v) for v in gamma):
        raise ValueError("activity coefficient must be positive and finite")
    mole_hessian = tuple(
        composition_gradient_to_moles(row, xs)
        for row in partial_molar_composition_jacobian(gxx, xs)
    )

    def mapping(row: Iterable[float]) -> dict[str, float]:
        return dict(zip(ids, row, strict=True))

    return ActivityThermodynamicReport(
        spec.model_id,
        spec.model,
        ids,
        temperature_K,
        mapping(xs),
        ge,
        ge - temperature_K * gt,
        -gt,
        -temperature_K * gtt,
        gt,
        gtt,
        gttt,
        gx,
        gxx,
        gtx,
        mapping(gamma),
        mapping(log_gamma),
        mapping(
            (dt - value / temperature_K) / (R * temperature_K)
            for value, dt in zip(partial_g, partial_gt, strict=True)
        ),
        tuple(tuple(v / (R * temperature_K) for v in row) for row in mole_hessian),
        mapping(partial_g),
        mapping(
            value - temperature_K * dt for value, dt in zip(partial_g, partial_gt, strict=True)
        ),
        mapping(-dt for dt in partial_gt),
        _parameter_contract(spec),
    )


def activity_infinite_dilution(
    spec: ActivityModelSpec,
    solvent_composition: Mapping[str, float],
    *,
    temperature_K: float,
    solute_id: str | None = None,
) -> ActivityInfiniteDilutionReport:
    xs = validated_state(spec, solvent_composition, temperature_K)
    if solute_id is not None and solute_id not in spec.component_ids:
        raise ValueError("infinite dilution solute must be a declared component")
    ids = spec.component_ids if solute_id is None else (solute_id,)
    gammas, logs, solvents = {}, {}, {}
    for key in ids:
        index = spec.component_ids.index(key)
        solvent = [0.0 if j == index else x for j, x in enumerate(xs)]
        total = sum(solvent)
        if total <= 0:
            raise ValueError(f"infinite dilution of {key} requires nonzero remaining solvent")
        solvent = [v / total for v in solvent]
        value = observable(spec, solvent, temperature_K, "ln_gamma:" + key)
        try:
            gamma = exp(value)
        except OverflowError as exc:
            raise ValueError(
                "infinite dilution activity coefficient is outside numerical range"
            ) from exc
        if not isfinite(gamma) or gamma <= 0:
            raise ValueError("infinite dilution activity coefficient must be positive and finite")
        gammas[key], logs[key] = gamma, value
        solvents[key] = dict(zip(spec.component_ids, solvent, strict=True))
    return ActivityInfiniteDilutionReport(spec.model_id, temperature_K, gammas, logs, solvents)


def active_activity_parameters(spec: ActivityModelSpec) -> tuple[str, ...]:
    """Explicit continuous parameters actually read by the selected equation."""
    active: set[str] = set()
    if spec.model == "unifac":
        return tuple(key for key in spec.parameters if key.startswith("psi_"))
    if spec.model in {"regular_solution", "flory_huggins", "hansen"}:
        return tuple(spec.parameters)
    if spec.model == "ideal":
        return ()
    if spec.model == "uniquac":
        active.update(f"{prefix}:{key}" for prefix in ("r", "q") for key in spec.component_ids)
        if "z" in spec.parameters:
            active.add("z")
    for left in spec.component_ids:
        for right in spec.component_ids:
            if left == right:
                continue
            pair = f":{left}|{right}"
            groups = {
                "margules": (("A", ""),),
                "wilson": (("lambda", "abcdef"),),
                "uniquac": (("tau", "abcdef"),),
                "nrtl": (("tau", "abefgh"), ("alpha", "cd")),
            }[spec.model]
            for prefix, letters in groups:
                if prefix + pair in spec.parameters:
                    active.add(prefix + pair)
                else:
                    active.update(
                        f"{prefix}_{v}{pair}"
                        for v in letters
                        if f"{prefix}_{v}{pair}" in spec.parameters
                    )
    return tuple(k for k in spec.parameters if k in active)


def activity_parameter_jacobian(
    spec: ActivityModelSpec,
    composition: Mapping[str, float],
    *,
    temperature_K: float,
    quantities: Iterable[str],
    parameters: Iterable[str],
) -> dict[str, dict[str, float]]:
    xs = validated_state(spec, composition, temperature_K)
    parameters, quantities = tuple(parameters), tuple(quantities)
    if (
        not parameters
        or len(set(parameters)) != len(parameters)
        or set(parameters) - set(active_activity_parameters(spec))
    ):
        raise ValueError("parameter Jacobian requires unique active, explicit model parameters")
    if not quantities or len(set(quantities)) != len(quantities):
        raise ValueError("parameter Jacobian requires unique requested quantities")
    return {
        quantity: {
            name: partial_derivative(
                partial(observable, spec, xs, temperature_K, quantity),
                spec.parameters,
                (name,),
            )
            for name in parameters
        }
        for quantity in quantities
    }


__all__ = [
    "ActivityInfiniteDilutionReport",
    "ActivityThermodynamicReport",
    "activity_infinite_dilution",
    "activity_parameter_jacobian",
    "activity_thermodynamic_report",
]
