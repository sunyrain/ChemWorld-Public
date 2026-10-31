"""Multiphase CEF heat through the original device, phases and ledger."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from math import fsum, isclose, isfinite
from typing import Any, cast

import numpy as np

from chemworld.foundation.state_ledgers import EquipmentLedger, PhaseLedger, PhaseRecord
from chemworld.physchem.calphad import CefPhaseModel
from chemworld.physchem.calphad_caloric_equilibrium import (
    CalphadCaloricProblem,
    CalphadCaloricResult,
    CalphadCaloricSpec,
)
from chemworld.physchem.calphad_equilibrium_search import PhaseSurface
from chemworld.physchem.elements import ELEMENTS
from chemworld.physchem.heat_transfer_units import ThermalDeviceSpec
from chemworld.physchem.solver_backend import ODESolverPolicy, solve_ode
from chemworld.physchem.tdb_expressions import TDB_GAS_CONSTANT_J_MOL_K as R
from chemworld.runtime.advance_paths import advance_ledger
from chemworld.runtime.calphad_phase_inventory import CalphadPhaseTarget
from chemworld.runtime.coupled_thermal import connected_device
from chemworld.runtime.kernel_contracts import ModelProviderResult


@dataclass(frozen=True)
class CalphadThermalSpec:
    caloric: CalphadCaloricSpec
    phase_targets: tuple[CalphadPhaseTarget, ...]
    support_heat_capacity_J_K: float = 0.02
    reference_temperature_K: float = 600.0
    minimum_temperature_K: float = 510.0
    maximum_temperature_K: float = 950.0
    ode_method: str = "BDF"
    loss_conductance_W_K: float = 0.0
    environment_temperature_K: float = 600.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "phase_targets", tuple(self.phase_targets))
        if (
            not self.phase_targets
            or len({p.phase_id for p in self.phase_targets}) != len(self.phase_targets)
            or {p.phase_name for p in self.phase_targets}
            != {m.phase_name for m in self.caloric.equilibrium.models}
            or not self.caloric.temperature_range_K[0]
            < self.minimum_temperature_K
            < self.maximum_temperature_K
            < self.caloric.temperature_range_K[1]
        ):
            raise ValueError(
                "CALPHAD thermal needs explicit distinct targets and interior operating bounds"
            )
        if (
            any(
                isinstance(v, bool) or not isfinite(v) or v < 0
                for v in (self.support_heat_capacity_J_K, self.loss_conductance_W_K)
            )
            or any(
                isinstance(v, bool) or not isfinite(v) or v <= 0
                for v in (self.reference_temperature_K, self.environment_temperature_K)
            )
            or self.ode_method not in {"BDF", "Radau"}
        ):
            raise ValueError("CALPHAD thermal support, loss and method must be explicit")
        _ = self.element_symbols

    @property
    def pressure_Pa(self) -> float:  # noqa: N802
        return self.caloric.pressure_Pa

    @property
    def component_ids(self) -> tuple[str, ...]:
        return self.caloric.component_ids

    @property
    def element_symbols(self) -> dict[str, str]:
        symbols = {s.upper(): s for s in ELEMENTS}
        if set(self.component_ids) - symbols.keys():
            raise ValueError("CALPHAD thermal stock requires actual elemental counters")
        return {e: symbols[e] for e in self.component_ids}

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_type": "calphad_multiphase_thermal_v1",
            "caloric": self.caloric.to_dict(),
            "phase_targets": [asdict(p) for p in self.phase_targets],
            **{
                k: getattr(self, k)
                for k in (
                    "support_heat_capacity_J_K",
                    "reference_temperature_K",
                    "minimum_temperature_K",
                    "maximum_temperature_K",
                    "ode_method",
                    "loss_conductance_W_K",
                    "environment_temperature_K",
                )
            },
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CalphadThermalSpec:
        from chemworld.data.scientific_inputs import _keys

        _keys(
            data,
            (
                "model_type",
                "caloric",
                "phase_targets",
                "support_heat_capacity_J_K",
                "reference_temperature_K",
                "minimum_temperature_K",
                "maximum_temperature_K",
                "ode_method",
                "loss_conductance_W_K",
                "environment_temperature_K",
            ),
        )
        if data["model_type"] != "calphad_multiphase_thermal_v1":
            raise ValueError("unknown CALPHAD thermal encoding")
        return cls(
            **{
                **{k: v for k, v in data.items() if k != "model_type"},
                "caloric": CalphadCaloricSpec.from_dict(data["caloric"]),
                "phase_targets": tuple(CalphadPhaseTarget(**p) for p in data["phase_targets"]),
            }
        )


def _inventory(state: Any, spec: CalphadThermalSpec) -> tuple[dict[str, float], float, float]:
    targets = {p.phase_id: p for p in spec.phase_targets}
    models = {m.phase_name: m for m in spec.caloric.equilibrium.models}
    if (
        state.state_slots.slots
        or not state.phases.phases
        or set(state.phases.phases) - targets.keys()
        or set(state.species_amounts) != set(spec.component_ids)
        or not isclose(state.pressure_Pa, spec.pressure_Pa, rel_tol=0, abs_tol=1e-7)
        or not spec.minimum_temperature_K - 1e-6
        <= state.temperature_K
        <= spec.maximum_temperature_K + 1e-6
    ):
        raise ValueError(
            "CALPHAD thermal requires its declared original phases, pressure and T range"
        )
    feed = dict.fromkeys(spec.component_ids, 0.0)
    enthalpy = spec.support_heat_capacity_J_K * (state.temperature_K - spec.reference_temperature_K)
    volume = 0.0
    for key, phase in state.phases.phases.items():
        target = targets[key]
        if (
            phase.vessel_id != state.vessel_id
            or phase.phase_type != target.phase_type
            or set(phase.species_amounts_mol) - set(spec.component_ids)
            or phase.solvents.volume_L != 0
            or any(not isfinite(v) or v < 0 for v in phase.species_amounts_mol.values())
        ):
            raise ValueError("CALPHAD actual phase owner, atoms or solvent mismatch")
        n = fsum(phase.species_amounts_mol.values())
        if n <= 0:
            raise ValueError("empty CALPHAD phase must be absent from original ledger")
        model = models[target.phase_name]
        sites = model._fractions(phase.metadata["site_fractions"])
        projected = model._composition(sites)[1]
        if phase.metadata.get("tdb_phase") != target.phase_name or any(
            abs(n * projected.get(e, 0.0) - phase.species_amounts_mol.get(e, 0.0)) > 2e-9 * n
            for e in set(projected) | set(spec.component_ids)
        ):
            raise ValueError("CALPHAD site configuration disagrees with actual elemental inventory")
        if not isclose(
            phase.volume_L, n * target.molar_volume_L_mol_atoms, rel_tol=2e-9, abs_tol=1e-14
        ):
            raise ValueError("CALPHAD actual volume disagrees with declared phase volume")
        properties = model.evaluate(
            sites, state.temperature_K, state.pressure_Pa, pressure_derivative=False
        )
        enthalpy += n * properties.enthalpy_J_mol
        volume += phase.volume_L
        for e in spec.component_ids:
            feed[e] += phase.species_amounts_mol.get(e, 0.0)
    if (
        any(
            not isclose(feed[e], state.species_amounts[e], rel_tol=2e-9, abs_tol=1e-14)
            for e in feed
        )
        or not isclose(volume, state.volume_L, rel_tol=2e-9, abs_tol=1e-14)
        or volume > state.vessels.vessels[state.vessel_id].max_volume_L + 1e-12
    ):
        raise ValueError("CALPHAD total atoms, volume or finite vessel mismatch")
    return feed, enthalpy, enthalpy - spec.pressure_Pa * volume * 0.001


def calphad_inventory_enthalpy_j(state: Any, spec: CalphadThermalSpec) -> float:
    return _inventory(state, spec)[1]


class _CalphadClosureError(ValueError):
    def __init__(self, result: CalphadCaloricResult) -> None:
        self.result = result
        super().__init__("CALPHAD PH unresolved: " + result.failure_reason)


def _solve(
    problem: CalphadCaloricProblem, spec: CalphadThermalSpec, enthalpy: float
) -> CalphadCaloricResult:
    result = problem.solve(
        enthalpy,
        support_heat_capacity_J_K=spec.support_heat_capacity_J_K,
        support_reference_temperature_K=spec.reference_temperature_K,
    )
    if not result.success:
        raise _CalphadClosureError(result)
    return result


def validate_calphad_thermal_state(state: Any, spec: CalphadThermalSpec) -> None:
    feed, enthalpy, _ = _inventory(state, spec)
    result = _solve(CalphadCaloricProblem(spec.caloric, feed), spec, enthalpy)
    if abs(cast(float, result.temperature_K) - state.temperature_K) > 2e-5:
        raise ValueError("CALPHAD current phase assemblage is not at its declared PH equilibrium")
    candidate = install_calphad_thermal_state(state, spec, result)
    if set(candidate.phases.phases) != set(state.phases.phases) or any(
        abs(
            p.species_amounts_mol.get(e, 0.0)
            - state.phases.phases[k].species_amounts_mol.get(e, 0.0)
        )
        > 2e-7 * fsum(feed.values())
        for k, p in candidate.phases.phases.items()
        for e in feed
    ):
        raise ValueError("CALPHAD committed phase amount/configuration is not equilibrium")
    # Equal total composition and H do not determine internal sublattice order:
    # an ideal redistribution can change only entropy. Validate the committed
    # configuration itself against the accepted common tangent. Comparing site
    # arrays directly would incorrectly reject symmetry-equivalent minima.
    potentials = cast(dict[str, float], result.element_potentials_J_mol)
    elements = tuple(potentials)
    mu = np.array([potentials[e] for e in elements])
    models = {m.phase_name: m for m in spec.caloric.equilibrium.models}
    scale = R * state.temperature_K
    for phase in state.phases.phases.values():
        original = models[phase.metadata["tdb_phase"]]
        model = CefPhaseModel(
            original.database,
            original.phase_name,
            elements,
            original.extrapolation,
            original.unspecified_endmembers,
            original.activate_suppressed,
        )
        surface = PhaseSurface(
            model,
            elements,
            state.temperature_K,
            spec.pressure_Pa,
            spec.caloric.equilibrium.policy.site_floor,
        )
        q = np.array(
            [
                phase.metadata["site_fractions"][j][s]
                for j, members in enumerate(model.constituents)
                for s in members[:-1]
            ]
        )
        values, first, second = surface.response(q)
        tangent = float(values[0] - mu @ values[1:])
        gradient = (first[0] - mu @ first[1:]) / scale
        curvature = 0.0
        if len(q):
            hessian = (second[0] - np.einsum("i,ijk->jk", mu, second[1:])) / scale
            curvature = float(np.min(np.linalg.eigvalsh(hessian)))
        if (
            abs(tangent) > spec.caloric.equilibrium.policy.tangent_tolerance_J_mol
            or float(np.max(np.abs(gradient), initial=0.0)) > 1e-7
            or curvature < -1e-7
        ):
            raise ValueError(
                "CALPHAD committed configuration is not a local equilibrium tangent point"
            )


def install_calphad_thermal_state(
    state: Any, spec: CalphadThermalSpec, result: CalphadCaloricResult
) -> Any:
    if not result.success or not result.diagnostics.get("stability_checked"):
        raise ValueError("cannot install unchecked or unresolved CALPHAD PH")
    available = {
        m.phase_name: [t for t in spec.phase_targets if t.phase_name == m.phase_name]
        for m in spec.caloric.equilibrium.models
    }
    phases = {}
    for phase in result.phases:
        if not available[phase.phase_name]:
            raise ValueError("insufficient original target identities for CALPHAD phase split")
        target = available[phase.phase_name].pop(0)
        phases[target.phase_id] = PhaseRecord(
            target.phase_id,
            state.vessel_id,
            target.phase_type,
            phase.amount_mol_atoms * target.molar_volume_L_mol_atoms,
            phase.element_amounts_mol,
            selected=target.selected,
            metadata={
                "carrier_representation": "species_inventory",
                "tdb_phase": phase.phase_name,
                "site_fractions": phase.site_fractions,
                "molar_basis": "nonvacant_atoms",
            },
        )
    volume = fsum(p.volume_L for p in phases.values())
    if volume > state.vessels.vessels[state.vessel_id].max_volume_L + 1e-12:
        raise ValueError("CALPHAD equilibrium exceeds finite vessel capacity")
    return state.replace(
        phases=PhaseLedger(phases),
        species_amounts={
            e: fsum(p.species_amounts_mol.get(e, 0.0) for p in phases.values())
            for e in spec.component_ids
        },
        temperature_K=result.temperature_K,
        pressure_Pa=spec.pressure_Pa,
        volume_L=volume,
        phase="solid" if all(p.phase_type == "solid" for p in phases.values()) else "liquid",
        process=replace(state.process, last_observation_record=None),
    )


class CalphadThermalRuntime:
    def __init__(self, composition: Any = None) -> None:
        self.composition = composition

    def supports(self, state: Any) -> bool:
        return (
            self.composition is not None
            and self.composition.physics is not None
            and isinstance(
                self.composition.physics.models_of("calphad_material").get(state.vessel_id),
                CalphadThermalSpec,
            )
        )

    def _spec(self, state: Any) -> CalphadThermalSpec:
        return self.composition.physics.models_of("calphad_material")[state.vessel_id]

    def temperature_bounds_k(self, state: Any) -> tuple[float, float]:
        spec = self._spec(state)
        return spec.minimum_temperature_K, spec.maximum_temperature_K

    def validate_state(self, state: Any) -> None:
        validate_calphad_thermal_state(state, self._spec(state))

    def validate_world(self) -> None:
        if self.composition is None or self.composition.physics is None:
            return
        physics = self.composition.physics
        for state in physics.initial_states.values():
            if not self.supports(state):
                continue
            spec = self._spec(state)
            components = [c for c in self.composition.components if c.vessel == state.vessel_id]
            records = [
                r
                for r in state.equipment.equipment.values()
                if r.equipment_type == "calphad_material"
                and r.attached_vessel_id == state.vessel_id
            ]
            if (
                physics.network.reactions
                or len(records) != 1
                or records[0].settings
                != {"model_id": "calphad_material", "vessel": state.vessel_id}
                or sum(c.kind == "calphad_phase" for c in components) != 1
                or any(self.composition.modules.components()[c.kind].processes for c in components)
                or {s.species_id: s.formula for s in physics.network.species}
                != spec.element_symbols
                or any(s.charge != 0 for s in physics.network.species)
            ):
                raise ValueError(
                    "CALPHAD thermal needs one material reference, atoms and no competing kinetics"
                )
            self.validate_state(state)

    def reconcile_initialization(self, transition: Any) -> Any:
        if transition.receipt.get("scope") not in {"material", "batch"}:
            return transition
        active = transition.state
        locals_ = {active.vessel_id: active.replace(inactive_vessels={}), **active.inactive_vessels}
        changed = dict(transition.receipt.get("solid_equilibration", {}))
        for name, receipt in transition.receipt.get("vessels", {}).items():
            state = locals_[name]
            if not self.supports(state):
                continue
            spec = self._spec(state)
            retained = spec.support_heat_capacity_J_K * (
                receipt["previous_temperature_K"] - state.temperature_K
            )
            feed, enthalpy, _ = _inventory(state, spec)
            result = _solve(CalphadCaloricProblem(spec.caloric, feed), spec, enthalpy + retained)
            locals_[name] = install_calphad_thermal_state(state, spec, result)
            changed[name] = {
                "retained_support_heat_relative_to_template_J": retained,
                "total_enthalpy_J": enthalpy + retained,
                "temperature_K": result.temperature_K,
            }
        if not changed:
            return transition
        local = locals_.pop(active.vessel_id)
        return replace(
            transition,
            state=local.replace(inactive_vessels=locals_),
            receipt={**transition.receipt, "solid_equilibration": changed},
        )

    def advance(
        self,
        state: Any,
        action: Mapping[str, Any],
        *,
        heat: bool = False,
        jacket_program: Any = None,
    ) -> ModelProviderResult:
        spec = self._spec(state)
        try:
            record, device = connected_device(state, required=False)
            if jacket_program is not None or (device is not None and device.mode != "finite_power"):
                raise ValueError("multiphase CALPHAD requires its original finite-power control")
            if heat:
                if device is None:
                    raise ValueError("heat requires original connected thermal device")
                device = replace(
                    device, enabled=True, setpoint_K=float(action["target_temperature_K"])
                )
            if (
                device is not None
                and not spec.minimum_temperature_K
                <= device.setpoint_K
                <= spec.maximum_temperature_K
            ):
                raise ValueError("CALPHAD device setpoint outside operating domain")
            duration = float(action["duration_s"])
            if not isfinite(duration) or duration <= 0:
                raise ValueError("CALPHAD thermal duration must be finite and positive")
        except (ValueError, KeyError) as exc:
            return ModelProviderResult(
                {},
                success=False,
                outcome="rejected",
                failure_reason="invalid_calphad_control",
                diagnostics={"control_error": str(exc)},
            )
        try:
            feed, initial_h, initial_u = _inventory(state, spec)
            problem = CalphadCaloricProblem(spec.caloric, feed)
            cache: dict[float, CalphadCaloricResult] = {}

            def trial(value: float) -> CalphadCaloricResult:
                value = float(value)
                if value not in cache:
                    cache[value] = _solve(problem, spec, value)
                return cache[value]

            original = trial(initial_h)
            if abs(cast(float, original.temperature_K) - state.temperature_K) > 2e-5:
                raise ValueError("initial CALPHAD inventory is not equilibrated")

            def volume(result: CalphadCaloricResult) -> float:
                available = {
                    m.phase_name: [t for t in spec.phase_targets if t.phase_name == m.phase_name]
                    for m in spec.caloric.equilibrium.models
                }
                total = 0.0
                for phase in result.phases:
                    if not available[phase.phase_name]:
                        raise ValueError("insufficient declared CALPHAD phase targets")
                    total += (
                        phase.amount_mol_atoms
                        * available[phase.phase_name].pop(0).molar_volume_L_mol_atoms
                    )
                return total

            def rhs(time: float, vector: Any) -> Any:
                t = cast(float, trial(vector[0]).temperature_K)
                q = 0.0 if device is None else device.delivered_power_w(t)
                loss = spec.loss_conductance_W_K * (t - spec.environment_temperature_K)
                return np.array(
                    [q - loss, max(q, 0.0) + max(-loss, 0.0), max(-q, 0.0) + max(loss, 0.0)]
                )

            def capacity(time: float, vector: Any) -> float:
                return state.vessels.vessels[state.vessel_id].max_volume_L - volume(
                    trial(vector[0])
                )

            def hot(time: float, vector: Any) -> float:
                return min(
                    spec.maximum_temperature_K,
                    state.vessels.vessels[state.vessel_id].max_temperature_K,
                ) - cast(float, trial(vector[0]).temperature_K)

            def cold(time: float, vector: Any) -> float:
                return cast(float, trial(vector[0]).temperature_K) - spec.minimum_temperature_K

            events = (capacity, hot, cold)
            names = ("vessel_capacity", "temperature_limit", "minimum_temperature")
            for event in events:
                cast(Any, event).terminal, cast(Any, event).direction = True, -1
            report = solve_ode(
                rhs,
                [initial_h, 0.0, 0.0],
                time_span_s=(0.0, duration),
                events=events,
                policy=ODESolverPolicy(
                    "calphad_multiphase_enthalpy_v1",
                    spec.ode_method,
                    2e-9,
                    2e-11,
                    max_step_s=min(20.0, duration),
                    event_names=names,
                ),
            )
            report.raise_for_failure("CALPHAD finite thermal advance")
            final_time, vector = float(report.t[-1]), report.y[:, -1]
            stopped = None
            if report.terminal_point is not None:
                final_time, vector, index = report.terminal_point
                stopped = names[index]
            result = trial(vector[0])
            candidate = install_calphad_thermal_state(state, spec, result)
            work = spec.pressure_Pa * (candidate.volume_L - state.volume_L) * 0.001
            net_heat = float(vector[1] - vector[2])
            final_u = cast(float, result.enthalpy_J) - spec.pressure_Pa * candidate.volume_L * 0.001
            residual = final_u - initial_u - net_heat + work
            if abs(residual) > 2e-7 * max(1.0, abs(net_heat)):
                raise ValueError("CALPHAD same-reference first law failed")
            equipment = dict(state.equipment.equipment)
            if record is not None:
                device = cast(ThermalDeviceSpec, device)
                equipment[record.equipment_id] = replace(
                    record, settings={**record.settings, "thermal_device": device.to_dict()}
                )
            candidate = candidate.replace(
                equipment=EquipmentLedger(equipment),
                ledger=advance_ledger(
                    state.ledger,
                    final_time,
                    energy_deltas={
                        "energy_jacket_J": float(vector[1]),
                        "heat_loss_J": float(vector[2]),
                        "boundary_work_J": work,
                    },
                ),
            )
        except (ValueError, ArithmeticError, RuntimeError) as exc:
            return ModelProviderResult(
                {},
                success=False,
                outcome="numerical_failure",
                failure_reason="calphad_thermal_unresolved",
                diagnostics={
                    "solver_error": str(exc),
                    "failed_caloric_closure": exc.result.to_dict()
                    if isinstance(exc, _CalphadClosureError)
                    else None,
                },
            )
        return ModelProviderResult(
            {},
            state=candidate,
            actual_duration_s=final_time,
            outcome="stopped" if stopped else "success",
            diagnostics={
                "solver": report.diagnostic.to_dict(),
                "caloric": result.to_dict(),
                "first_law_residual_J": residual,
                "boundary_work_J": work,
                "stopping_event": stopped,
                "caloric_trial_count": len(cache),
            },
        )
