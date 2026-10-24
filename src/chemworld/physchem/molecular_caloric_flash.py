"""Simultaneous two-phase PH/PS on the common molecular Helmholtz reference.

This is a declared, nonvanishing two-phase branch. Density bounds and a branch
seed are explicit; finite post-TPD searches do not prove global uniqueness.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from math import fsum, isfinite, log
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray
from scipy.special import expit

from chemworld.physchem.molecular_flash import (
    MolecularDensityBasin,
    MolecularFlashPhase,
    MolecularTPDResult,
    molecular_tp_stability,
)
from chemworld.physchem.molecular_mixture import MolecularCaloricModel, MolecularCaloricState
from chemworld.physchem.residual_system import (
    ResidualDomainError,
    ResidualVariable,
    RootPolicy,
    finite_jacobian,
    solve_root,
)


@dataclass(frozen=True)
class MolecularCaloricFlashSpec:
    model: MolecularCaloricModel
    basins: tuple[MolecularDensityBasin, ...]
    phase_basins: tuple[str, str]
    pressure_Pa: float
    temperature_range_K: tuple[float, float]
    minimum_phase_fraction: float = 1e-8
    root_policy: RootPolicy = field(default_factory=lambda: RootPolicy(2e-10, 150))
    stability_max_iterations: int = 200
    tpd_tolerance: float = 1e-8

    def __post_init__(self) -> None:
        for key in ("basins", "phase_basins", "temperature_range_K"):
            object.__setattr__(self, key, tuple(getattr(self, key)))
        names = {b.name for b in self.basins}
        if (
            not names
            or len(names) != len(self.basins)
            or len(self.phase_basins) != 2
            or set(self.phase_basins) != names
        ):
            raise ValueError("caloric flash needs two phase branches covering declared basins")
        if (
            len(self.temperature_range_K) != 2
            or not 0 < self.temperature_range_K[0] < self.temperature_range_K[1]
            or any(
                isinstance(v, bool) or not isfinite(v) or v <= 0
                for v in (self.pressure_Pa, *self.temperature_range_K, self.tpd_tolerance)
            )
            or not 0 < self.minimum_phase_fraction < 0.5
            or type(self.stability_max_iterations) is not int
            or self.stability_max_iterations < 1
        ):
            raise ValueError("caloric flash requires finite positive bounds and search budgets")

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_type": "molecular_caloric_flash_v1",
            "model": self.model.to_dict(),
            "basins": [asdict(b) for b in self.basins],
            "phase_basins": list(self.phase_basins),
            "pressure_Pa": self.pressure_Pa,
            "temperature_range_K": list(self.temperature_range_K),
            "minimum_phase_fraction": self.minimum_phase_fraction,
            "root_policy": asdict(self.root_policy),
            "stability_max_iterations": self.stability_max_iterations,
            "tpd_tolerance": self.tpd_tolerance,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> MolecularCaloricFlashSpec:
        if data.get("model_type") != "molecular_caloric_flash_v1":
            raise ValueError("unknown molecular caloric flash specification")
        values = {k: v for k, v in data.items() if k != "model_type"}
        values.update(
            model=MolecularCaloricModel.from_dict(data["model"]),
            basins=tuple(MolecularDensityBasin(**b) for b in data["basins"]),
            root_policy=RootPolicy(**data["root_policy"]),
        )
        return cls(**values)


@dataclass(frozen=True)
class MolecularCaloricFlashSeed:
    temperature_K: float
    phase_amounts_mol: tuple[dict[str, float], dict[str, float]]
    phase_densities_mol_m3: tuple[float, float]

    @classmethod
    def from_phases(cls, phases: tuple[MolecularFlashPhase, ...]) -> MolecularCaloricFlashSeed:
        if len(phases) != 2:
            raise ValueError("caloric flash seed requires two actual phases")
        return cls(
            phases[0].state.temperature_K,
            (phases[0].component_amounts_mol, phases[1].component_amounts_mol),
            (phases[0].state.molar_density_mol_m3, phases[1].state.molar_density_mol_m3),
        )


@dataclass(frozen=True)
class MolecularCaloricFlashResult:
    success: bool
    phases: tuple[MolecularFlashPhase, ...] = ()
    temperature_K: float | None = None
    enthalpy_J: float | None = None
    entropy_J_K: float | None = None
    internal_energy_J: float | None = None
    volume_m3: float | None = None
    material_balance_error_mol: float | None = None
    chemical_potential_error_RT: float | None = None
    pressure_relative_error: float | None = None
    coordinate_residual: float | None = None
    post_split_stability: tuple[MolecularTPDResult, ...] = ()
    diagnostics: dict[str, Any] = field(default_factory=dict)
    failure_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "phases": [p.to_dict() for p in self.phases],
            "scope": "declared_nonvanishing_two_phase_branch",
        }


def solve_molecular_caloric_flash(
    spec: MolecularCaloricFlashSpec,
    feed_amounts_mol: Mapping[str, float],
    *,
    coordinate: str,
    value: float,
    seed: MolecularCaloricFlashSeed,
    support_heat_capacity_J_K: float = 0.0,
    support_reference_temperature_K: float = 298.15,
    verify_stability: bool = True,
) -> MolecularCaloricFlashResult:
    """Solve T, both densities and component allocations in one residual system.

    ``verify_stability=False`` is an explicitly unqualified numerical trial for an
    outer integrator. An accepted physical endpoint must use the default full
    post-TPD checks. Support heat capacity is a declared nonmaterial apparatus.
    """
    ids = spec.model.component_ids
    if (
        set(feed_amounts_mol) != set(ids)
        or coordinate not in {"H", "S"}
        or any(isinstance(v, bool) or not isfinite(v) or v < 0 for v in feed_amounts_mol.values())
        or not isfinite(value)
        or isinstance(value, bool)
        or not isfinite(support_heat_capacity_J_K)
        or support_heat_capacity_J_K < 0
        or not isfinite(support_reference_temperature_K)
        or support_reference_temperature_K <= 0
    ):
        raise ValueError("PH/PS needs matching finite inventory, H/S and explicit support")
    feed = np.array([feed_amounts_mol[k] for k in ids])
    active = np.flatnonzero(feed > 0)
    total = float(feed.sum())
    if len(active) < 2:
        raise ValueError("multicomponent two-phase branch requires at least two active species")
    basins = tuple(next(b for b in spec.basins if b.name == name) for name in spec.phase_basins)
    if (
        len(seed.phase_amounts_mol) != 2
        or len(seed.phase_densities_mol_m3) != 2
        or not spec.temperature_range_K[0] <= seed.temperature_K <= spec.temperature_range_K[1]
        or any(
            not b.density_bounds_mol_m3[0] <= rho <= b.density_bounds_mol_m3[1]
            for b, rho in zip(basins, seed.phase_densities_mol_m3, strict=True)
        )
    ):
        raise ValueError("seed must be inside the declared T/density branch")
    initial_amounts = np.array([[p.get(k, 0.0) for k in ids] for p in seed.phase_amounts_mol])
    if (
        not np.all(np.isfinite(initial_amounts))
        or np.any(initial_amounts[:, active] <= 0)
        or np.any(initial_amounts[:, feed == 0] != 0)
        or any(set(p) - set(ids) for p in seed.phase_amounts_mol)
    ):
        raise ValueError("seed must preserve exact positive support")
    initial = np.r_[
        log(seed.temperature_K),
        np.log(seed.phase_densities_mol_m3),
        np.log(initial_amounts[0, active] / initial_amounts[1, active]),
    ]
    lower = np.r_[
        log(spec.temperature_range_K[0]),
        [log(b.density_bounds_mol_m3[0]) for b in basins],
        np.full(len(active), -70.0),
    ]
    upper = np.r_[
        log(spec.temperature_range_K[1]),
        [log(b.density_bounds_mol_m3[1]) for b in basins],
        np.full(len(active), 70.0),
    ]
    gas = spec.model.gas_constant_J_mol_K
    caloric_scale = max(
        total * gas * (seed.temperature_K if coordinate == "H" else 1.0),
        support_heat_capacity_J_K * (seed.temperature_K if coordinate == "H" else 1.0),
    )

    def unpack(
        q: NDArray[np.float64],
    ) -> tuple[float, list[NDArray[np.float64]], list[MolecularCaloricState], float, float]:
        t = float(np.exp(q[0]))
        inventories: list[NDArray[np.float64]] = []
        states: list[MolecularCaloricState] = []
        for index, sign in enumerate((1, -1)):
            ns = np.zeros(len(ids))
            ns[active] = feed[active] * expit(sign * q[3:])
            n = float(ns.sum())
            if n / total < spec.minimum_phase_fraction:
                raise ResidualDomainError("phase fraction leaves declared two-phase branch")
            try:
                state = spec.model.state(t, float(np.exp(q[index + 1])), tuple(map(float, ns / n)))
                if state.pressure_Pa <= 0 or state.log_fugacity_coefficients is None:
                    raise ValueError("nonpositive phase pressure")
            except (ValueError, ArithmeticError) as exc:
                raise ResidualDomainError(str(exc)) from exc
            inventories.append(ns)
            states.append(state)
        h = fsum(
            float(n.sum()) * s.enthalpy_J_mol for n, s in zip(inventories, states, strict=True)
        )
        h += support_heat_capacity_J_K * (t - support_reference_temperature_K)
        entropy = fsum(
            float(n.sum()) * s.entropy_J_mol_K for n, s in zip(inventories, states, strict=True)
        )
        entropy += support_heat_capacity_J_K * log(t / support_reference_temperature_K)
        return t, inventories, states, h, entropy

    def residual(q: NDArray[np.float64]) -> NDArray[np.float64]:
        _, _, states, h, entropy = unpack(q)
        mus = [
            np.log(np.array(s.mole_fractions)[active])
            + np.array(s.log_fugacity_coefficients)[active]
            + log(s.pressure_Pa / spec.pressure_Pa)
            for s in states
        ]
        return np.r_[
            [log(s.pressure_Pa / spec.pressure_Pa) for s in states],
            mus[0] - mus[1],
            ((h if coordinate == "H" else entropy) - value) / caloric_scale,
        ]

    root = solve_root(
        residual,
        initial,
        jacobian=lambda q: finite_jacobian(residual, q, np.ones(len(q)), lower=lower, upper=upper),
        variables=tuple(
            ResidualVariable(f"joint_{i}", "dimensionless", lower=float(lo), upper=float(hi))
            for i, (lo, hi) in enumerate(zip(lower, upper, strict=True))
        ),
        residual_scales=np.ones(len(initial)),
        policy=spec.root_policy,
    )
    diagnostics = {
        "root": root.diagnostics,
        "coordinate": coordinate,
        "stability_checked": verify_stability,
        "equations": "joint_T_rho1_rho2_component_allocations_and_caloric_constraint",
    }
    if not root.success:
        return MolecularCaloricFlashResult(
            False, diagnostics=diagnostics, failure_reason=root.failure_reason
        )
    t, inventories, states, h, entropy = unpack(root.value)
    phases = tuple(
        MolecularFlashPhase(b.name, dict(zip(ids, map(float, ns), strict=True)), s)
        for b, ns, s in zip(basins, inventories, states, strict=True)
    )
    distinct = max(
        abs(log(states[0].molar_density_mol_m3 / states[1].molar_density_mol_m3)),
        max(
            abs(a - b)
            for a, b in zip(states[0].mole_fractions, states[1].mole_fractions, strict=True)
        ),
    )
    post = (
        tuple(
            molecular_tp_stability(
                spec.model.residual_model,
                t,
                spec.pressure_Pa,
                p.state.mole_fractions,
                basins=spec.basins,
                max_iterations=spec.stability_max_iterations,
                tpd_tolerance=spec.tpd_tolerance,
                reference_basin=p.basin,
            )
            for p in phases
        )
        if verify_stability
        else ()
    )
    reason = (
        "trivial_identical_phases"
        if distinct < 1e-6
        else "remaining_or_unresolved_post_split_instability"
        if any(not s.complete or s.unstable for s in post)
        else ""
    )
    error = residual(root.value)
    return MolecularCaloricFlashResult(
        not reason,
        phases,
        t,
        h,
        entropy,
        fsum(
            p.amount_mol * cast(MolecularCaloricState, p.state).internal_energy_J_mol
            for p in phases
        )
        + support_heat_capacity_J_K * (t - support_reference_temperature_K),
        fsum(p.volume_m3 for p in phases),
        max(
            abs(feed_amounts_mol[k] - fsum(p.component_amounts_mol[k] for p in phases)) for k in ids
        ),
        float(np.max(np.abs(error[2:-1]))),
        max(abs(s.pressure_Pa / spec.pressure_Pa - 1) for s in states),
        (h if coordinate == "H" else entropy) - value,
        post,
        diagnostics,
        reason,
    )
