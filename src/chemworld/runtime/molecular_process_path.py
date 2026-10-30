"""Consume a declared steady process path in the original finite receiver.

The source is explicitly external and prescribed, not an untracked finite tank.
The existing CoupledSystem owns integration; WorldState owns accepted inventory,
time and utility receipts. This adapter returns a normal provider candidate for
the caller's transaction, and never commits trial evaluations itself.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from math import fsum, isclose, isfinite
from typing import Any, Protocol, cast

from chemworld.foundation import WorldState
from chemworld.foundation.state_ledgers import PhaseLedger, VesselLedger
from chemworld.physchem.coupled_system import (
    CoupledEvent,
    CoupledProcess,
    CoupledState,
    CoupledStateView,
    CoupledSystem,
    MaterialPort,
)
from chemworld.physchem.molecular_mixture import MolecularIsochoricThermodynamics, MolecularStream
from chemworld.physchem.molecular_processes import (
    MolecularHeatDuty,
    MolecularPressureStage,
    MolecularProcessDomain,
    MolecularProcessResult,
)
from chemworld.physchem.solver_backend import ODESolverPolicy
from chemworld.runtime.advance_paths import advance_ledger
from chemworld.runtime.kernel_contracts import ModelProviderResult

PROCESS_PATH_POLICY = ODESolverPolicy(
    "molecular-process-receiver", method="BDF", rtol=1e-10, atol=1e-12
)


class MolecularSteadyProcessPath(Protocol):
    """Pure steady producer accepted by the existing finite receiver boundary."""

    @property
    def domain(self) -> MolecularProcessDomain: ...

    @property
    def source_id(self) -> str: ...

    @property
    def inlet(self) -> MolecularStream: ...

    def evaluate(self) -> MolecularProcessResult: ...


@dataclass(frozen=True)
class MolecularProcessPath:
    domain: MolecularProcessDomain
    source_id: str
    inlet: MolecularStream
    devices: tuple[MolecularPressureStage | MolecularHeatDuty, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "devices", tuple(self.devices))
        if not isinstance(self.source_id, str) or not self.source_id:
            raise ValueError("process path requires an explicit external source identity")
        self.domain.validate_stream(self.inlet)
        if any(d.domain != self.domain for d in self.devices):
            raise ValueError("all process devices must share the same model and caloric reference")

    def evaluate(self) -> MolecularProcessResult:
        self.domain.validate_stream(self.inlet)
        stream, heat, work, reports = self.inlet, 0.0, 0.0, []
        for device in self.devices:
            result = device.evaluate(stream)
            reports.append(result.to_dict())
            if not result.success:
                return MolecularProcessResult(
                    False,
                    failure_reason=result.failure_reason,
                    diagnostics={"failed_device_index": len(reports) - 1, "devices": reports},
                )
            if result.outlet is None:
                raise AssertionError("invariant violated: result.outlet is not None")
            stream = result.outlet
            heat += result.heat_duty_W
            work += result.shaft_work_W
        return MolecularProcessResult(
            True,
            (stream,),
            heat,
            work,
            diagnostics={
                "source_id": self.source_id,
                "source_scope": "external_prescribed_reservoir",
                "external_feed_enthalpy_rate_W": self.inlet.enthalpy_rate_W,
                "energy_residual_W": stream.enthalpy_rate_W
                - self.inlet.enthalpy_rate_W
                - heat
                - work,
                "devices": reports,
            },
        )

    def to_dict(self) -> dict[str, Any]:
        s = self.inlet.state
        return {
            "model": "molecular-process-path-v1",
            "domain": self.domain.to_dict(),
            "source_id": self.source_id,
            "source_scope": "external_prescribed_reservoir",
            "inlet": {
                "temperature_K": s.temperature_K,
                "molar_density_mol_m3": s.molar_density_mol_m3,
                "mole_fractions": s.mole_fractions,
                "molar_rate_mol_s": self.inlet.molar_rate_mol_s,
            },
            "devices": [d.to_dict() for d in self.devices],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> MolecularProcessPath:
        if (
            data.get("model") != "molecular-process-path-v1"
            or data.get("source_scope") != "external_prescribed_reservoir"
        ):
            raise ValueError("unknown process path or undeclared finite source")
        domain = MolecularProcessDomain.from_dict(data["domain"])
        v = data["inlet"]
        inlet = MolecularStream(
            domain.state(v["temperature_K"], v["molar_density_mol_m3"], v["mole_fractions"]),
            v["molar_rate_mol_s"],
        )
        devices = []
        for d in data["devices"]:
            if d.get("kind") not in {"pressure", "heat"}:
                raise ValueError("unknown device in molecular process path")
            devices.append(
                (
                    MolecularPressureStage if d["kind"] == "pressure" else MolecularHeatDuty
                ).from_dict(d)
            )
        return cls(domain, data["source_id"], inlet, tuple(devices))


def build_molecular_process_receiver(
    path: MolecularSteadyProcessPath,
    *,
    receiver_id: str,
    volume_L: float,
    maximum_amount_mol: float,
    maximum_pressure_Pa: float,
    maximum_temperature_K: float | None = None,
    numerical_form: str = "ode",
    policy: ODESolverPolicy = PROCESS_PATH_POLICY,
) -> tuple[CoupledSystem | None, MolecularProcessResult]:
    """Metered external inlet through devices, stopped at receiver/backpressure limits.

    Equipment work and heat are already in the outlet enthalpy. They are not
    also applied as receiver heat/work. No finite upstream inventory is implied.
    """
    if (
        not receiver_id
        or receiver_id == path.source_id
        or any(
            not isfinite(v) or v <= 0 for v in (volume_L, maximum_amount_mol, maximum_pressure_Pa)
        )
    ):
        raise ValueError("process receiver requires identity and finite positive limits")
    evaluated = path.evaluate()
    if not evaluated.success:
        return None, evaluated
    if evaluated.outlet is None:
        raise AssertionError("invariant violated: evaluated.outlet is not None")
    stream = evaluated.outlet
    maximum_temperature = path.domain.temperature_range_K[1]
    if maximum_temperature_K is not None:
        if not isfinite(maximum_temperature_K) or maximum_temperature_K <= 0:
            raise ValueError("receiver temperature limit must be positive")
        maximum_temperature = min(maximum_temperature, maximum_temperature_K)
    thermal = MolecularIsochoricThermodynamics(path.domain.model, volume_L)
    ports = tuple(
        MaterialPort(f"process-feed:{k}", k, path.source_id, receiver_id, "in")
        for k in path.domain.model.component_ids
    )
    contribution = stream.contribution({p.inventory_key: p.port_id for p in ports})
    max_amount = min(
        maximum_amount_mol, path.domain.density_basin.density_bounds_mol_m3[1] * volume_L * 1e-3
    )
    max_pressure = min(maximum_pressure_Pa, stream.state.pressure_Pa)

    def validate(s: CoupledStateView) -> None:
        thermal.validate_state(s)
        total = fsum(s.amounts_mol.values())
        path.domain.state(
            s.temperature_K,
            total / (volume_L * 1e-3),
            tuple(s.amounts_mol[k] / total for k in thermal.inventory_keys),
        )
        if (
            total > max_amount * (1 + 1e-9)
            or thermal.pressure_pa(s) > max_pressure * (1 + 1e-9)
            or s.temperature_K > maximum_temperature * (1 + 1e-9)
        ):
            raise ValueError("receiver exceeds inventory, hardware or inlet backpressure limit")

    events = (
        CoupledEvent("receiver_amount_limit", lambda s: max_amount - fsum(s.amounts_mol.values())),
        CoupledEvent("receiver_pressure_limit", lambda s: max_pressure - thermal.pressure_pa(s)),
        CoupledEvent("receiver_temperature_max", lambda s: maximum_temperature - s.temperature_K),
        CoupledEvent(
            "receiver_temperature_min",
            lambda s: s.temperature_K - path.domain.temperature_range_K[0],
        ),
    )
    process = CoupledProcess(
        "molecular-process-path", lambda _: contribution, validate_state=validate
    )
    system = CoupledSystem(
        thermal,
        (process,),
        ports=ports,
        events=events,
        conservation_laws={k: {k: 1.0} for k in thermal.inventory_keys},
        numerical_form=numerical_form,
        policy=policy,
    )
    return system, evaluated


def advance_molecular_process_receiver(
    state: WorldState,
    path: MolecularSteadyProcessPath,
    duration_s: float,
    *,
    phase_id: str,
    maximum_amount_mol: float,
    numerical_form: str = "ode",
    policy: ODESolverPolicy = PROCESS_PATH_POLICY,
) -> ModelProviderResult:
    """Produce an atomic candidate using original phase, clock and utility ledgers.

    Single homogeneous receiver only. Registry installation and multi-vessel
    dynamic source ownership are outside this standalone provider's scope.
    """
    try:
        phases = cast(PhaseLedger, state.phases)
        vessels = cast(VesselLedger, state.vessels)
        if (
            state.terminated
            or state.quenched
            or state.inactive_vessels
            or state.state_slots.slots
            or set(phases.phases) != {phase_id}
            or set(state.species_amounts) != set(path.domain.model.component_ids)
            or not isfinite(duration_s)
            or duration_s < 0
        ):
            raise ValueError(
                "receiver needs one active homogeneous native inventory and valid time"
            )
        phase = phases.phases[phase_id]
        if (
            phase.vessel_id != state.vessel_id
            or phase.phase_id != phase_id
            or phase.solvents.volume_L > 0
            or phase.volume_L != state.volume_L
            or phase.species_amounts_mol != state.species_amounts
        ):
            raise ValueError("receiver phase identity, volume or sole material ownership disagrees")
        hardware = vessels.vessels[state.vessel_id]
        if state.volume_L > hardware.max_volume_L:
            raise ValueError("receiver volume exceeds original hardware")
        system, evaluated = build_molecular_process_receiver(
            path,
            receiver_id=state.vessel_id,
            volume_L=state.volume_L,
            maximum_amount_mol=maximum_amount_mol,
            maximum_pressure_Pa=hardware.max_pressure_Pa,
            maximum_temperature_K=hardware.max_temperature_K,
            numerical_form=numerical_form,
            policy=policy,
        )
        if system is None:
            return ModelProviderResult(
                outputs={},
                diagnostics={"steady_path": evaluated.to_dict()},
                success=False,
                failure_reason="molecular_process_path_failed",
                outcome="numerical_failure",
            )
        initial = CoupledState(
            state.species_amounts,
            state.temperature_K,
            state.volume_L,
            state.vessel_elapsed_s.get(state.vessel_id, state.ledger.time_s),
        )
        thermal = cast(MolecularIsochoricThermodynamics, system.thermodynamics)
        if not isclose(thermal.pressure_pa(initial), state.pressure_Pa, rel_tol=1e-9, abs_tol=1e-5):
            raise ValueError("receiver pressure disagrees with actual inventory and declared EOS")
        result = system.advance(initial, duration_s)
        final, elapsed = result.final_state, result.actual_duration_s
        heat, work = evaluated.heat_duty_W * elapsed, evaluated.shaft_work_W * elapsed
        incoming = fsum(final.exchange_energy_J.values())
        source_h = path.inlet.enthalpy_rate_W * elapsed
        energy_residual = incoming - source_h - heat - work
        if abs(energy_residual) > 1e-6 + 1e-8 * max(
            abs(incoming), abs(source_h), abs(heat), abs(work)
        ):
            raise ValueError("process path utility/stream energy balance failed")
        amounts = dict(final.amounts_mol)
        candidate = state.replace(
            species_amounts=amounts,
            phases=PhaseLedger({phase_id: replace(phase, species_amounts_mol=amounts)}),
            temperature_K=final.temperature_K,
            pressure_Pa=thermal.pressure_pa(final),
            vessel_elapsed_s={**state.vessel_elapsed_s, state.vessel_id: final.time_s},
            ledger=advance_ledger(
                state.ledger,
                elapsed,
                energy_deltas={"flow_heat_J": heat, "flow_shaft_work_J": work},
            ),
        )
        return ModelProviderResult(
            outputs={"actual_exchanges": result.actual_exchanges},
            state=candidate,
            actual_duration_s=elapsed,
            outcome=result.outcome,
            diagnostics={
                "source_id": path.source_id,
                "source_scope": "external_prescribed_reservoir",
                "steady_path": evaluated.to_dict(),
                "stopping_event": result.stopping_event,
                "external_feed_enthalpy_J": source_h,
                "stream_heat_J": heat,
                "stream_shaft_work_J": work,
                "received_enthalpy_J": incoming,
                "path_energy_residual_J": energy_residual,
                "receiver_energy_residual_J": result.energy_balance_residual_J,
                "component_balance_mol": dict(result.conservation_residuals),
                "solver": dict(result.solver_diagnostic),
                "heat_scope": "upstream_device_utility_already_in_received_enthalpy",
            },
        )
    except (ValueError, ArithmeticError, RuntimeError) as exc:
        return ModelProviderResult(
            outputs={},
            diagnostics={"error": str(exc)},
            success=False,
            failure_reason="molecular_process_receiver_failed",
            outcome="numerical_failure",
        )
