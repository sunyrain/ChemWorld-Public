"""Select an existing physical strategy and settle its actual advance once.

This is not another experiment/session loop. Transactions, observations and RNG
remain in ChemWorldRuntime. Conservative mixed-tank enthalpy and spatial PFR
packing are retained numerical strategies; their physical clocks are explicit.
"""

from dataclasses import dataclass, replace
from math import isfinite
from typing import Any

from chemworld.foundation.gas import GasBoundary
from chemworld.foundation.state import Ledger, WorldState
from chemworld.physchem.gas_control import advance_gas_with_report
from chemworld.runtime.kernel_contracts import ModelProviderResult

_ENERGY_FIELDS = frozenset(
    {
        "energy_jacket_J",
        "heat_reaction_J",
        "heat_loss_J",
        "gas_pump_work_J",
        "gas_bath_heat_J",
        "flow_pump_work_J",
        "flow_electrical_work_J",
        "flow_heat_J",
        "flow_shaft_work_J",
        "boundary_work_J",
    }
)


def advance_ledger(
    ledger: Ledger,
    actual_duration_s: float,
    *,
    cost_delta: float = 0.0,
    energy_deltas: dict[str, float] | None = None,
) -> Ledger:
    """Apply one accepted interval; subsystem utility settlement uses zero dt."""
    if (
        not isfinite(actual_duration_s)
        or actual_duration_s < 0
        or not isfinite(cost_delta)
        or cost_delta < 0
    ):
        raise ValueError("advance accounting requires finite nonnegative time and cost")
    energies = energy_deltas or {}
    if set(energies) - _ENERGY_FIELDS or any(not isfinite(v) for v in energies.values()):
        raise ValueError("advance accounting requires declared finite energy increments")
    return ledger.with_updates(
        time_s=ledger.time_s + actual_duration_s,
        cost=ledger.cost + cost_delta,
        **{key: getattr(ledger, key) + value for key, value in energies.items()},
    )


@dataclass(frozen=True)
class ControlStageAdvance:
    provider: ModelProviderResult
    gas: GasBoundary
    exchanges: tuple[dict, ...] = ()


def select_advance_strategy(strategies, state, *, bindings=None, required=False):
    """Shared compile/query/runtime selection. A binding cannot hide a conflict."""
    candidates = [
        (key, strategy) for key, strategy in strategies.items() if strategy.supports(state)
    ]
    if len(candidates) > 1:
        raise ValueError(
            f"multiple physical advance strategies claim {state.vessel_id!r}: "
            f"{[key for key, _ in candidates]}"
        )
    expected = (bindings or {}).get(state.vessel_id)
    if expected is not None and (not candidates or candidates[0][0] != expected):
        raise ValueError(
            f"advance binding {expected!r} does not own physical system {state.vessel_id!r}"
        )
    if required and not candidates:
        raise ValueError(f"no declared physical advance strategy for {state.vessel_id!r}")
    return candidates[0] if candidates else None


class PhysicalAdvanceCoordinator:
    def __init__(self, thermal: Any, *, modules=None, composition=None):
        self.thermal = thermal
        self.specs = {} if modules is None else modules.advance_specs()
        self.strategies = (
            {} if modules is None
            else modules.build_advance_strategies(composition, thermal=thermal)
        )
        self.bindings = (
            {} if composition is None or composition.physics is None
            else composition.physics.advance_bindings
        )

    @property
    def declared(self):
        return tuple(self.strategies.items())

    def validate_snapshot_state(self, state):
        """Check installed state owners before accepting saved continuation data."""
        locals_ = {
            s.vessel_id: s.replace(inactive_vessels={})
            for s in (state, *state.inactive_vessels.values())
        }
        for identity, local in locals_.items():
            selected = select_advance_strategy(self.strategies, local, bindings=self.bindings)
            validator = None if selected is None else getattr(selected[1], "validate_state", None)
            if validator is not None:
                # A multi-vessel owner must validate this saved system, including
                # current members when its anchor is not the active vessel.
                validator(local.replace(inactive_vessels={
                    name: member for name, member in locals_.items() if name != identity
                }))

    @property
    def preserved_source_vessels(self):
        """Finite sources owned by declared strategies keep their actual stock."""
        return set().union(
            *(getattr(s, "preserved_source_vessels", set()) for _, s in self.declared)
        )

    @property
    def protected_lifecycle_vessels(self):
        """Consumed supplies cannot be replenished by a local entity boundary."""
        return set().union(
            *(getattr(s, "protected_lifecycle_vessels", set()) for _, s in self.declared)
        )

    def reconcile_initialization(self, transition, *, previous=None):
        """Reclose retained apparatus and replacement material without time advance."""
        for _, strategy in self.declared:
            with_previous = getattr(strategy, 'reconcile_initialization_from', None)
            if with_previous is not None:
                transition = with_previous(transition, previous)
                continue
            reconcile = getattr(strategy, "reconcile_initialization", None)
            if reconcile is not None:
                transition = reconcile(transition)
        return transition

    def reconcile_material_change(self, before, after):
        """Let an existing strategy invalidate its solver memory after real sampling."""
        selected = self.strategy(before)
        reconcile = None if selected is None else getattr(
            selected[1], 'reconcile_material_change', None)
        return after if reconcile is None else reconcile(before, after)

    def strategy(self, state, *, required=False):
        return select_advance_strategy(
            self.strategies, state, bindings=self.bindings, required=required
        )

    def supports(self, state):
        return self.strategy(state) is not None

    def supports_measurement_clock(self, state):
        selected = self.strategy(state)
        return selected is not None and self.specs[selected[0]].timed_measurements

    def batch(self, state, action, *, heat, jacket_program=None):
        from chemworld.runtime.control_program import advance_electrical_program

        controls = [
            record.settings["electrical_control"]
            for record in state.equipment.equipment.values()
            if record.attached_vessel_id == state.vessel_id
            and "electrical_control" in record.settings
        ]
        if any(
            c.get("status") == "running"
            and c.get("stage_index", 0) - c.get("stage_offset", 0) < len(c.get("stages", []))
            for c in controls
        ):
            return advance_electrical_program(
                state,
                action,
                lambda candidate, interval, stage, index: self._batch_interval(
                    candidate,
                    interval,
                    heat=heat,
                    jacket_program=jacket_program,
                    control_stage=(stage, index),
                ),
            )
        return self._batch_interval(state, action, heat=heat, jacket_program=jacket_program)

    def _batch_interval(self, state, action, *, heat, jacket_program=None, control_stage=None):
        strategy, runner = self.strategy(state, required=True)
        if control_stage is None:
            result = runner.advance(state, action, heat=heat, jacket_program=jacket_program)
        else:
            result = runner.advance_controlled(
                state, action, control_stage, heat=heat, jacket_program=jacket_program
            )
        return replace(result, diagnostics={**result.diagnostics, "advance_strategy": strategy})

    def control_stage(
        self,
        state: WorldState,
        action,
        *,
        jacket_program,
        gas: GasBoundary,
        pressure_setpoint_Pa: float,
        atmosphere: int,
    ) -> ControlStageAdvance:
        provider = self.batch(state, action, heat=True, jacket_program=jacket_program)
        if provider.state is None or provider.outcome in {"rejected", "numerical_failure"}:
            return ControlStageAdvance(provider, gas)
        dt = provider.actual_duration_s
        if dt == 0:
            return ControlStageAdvance(provider, gas)
        try:
            report = advance_gas_with_report(
                gas,
                duration_s=dt,
                final_temperature_K=provider.state.temperature_K,
                pressure_setpoint_Pa=pressure_setpoint_Pa,
                atmosphere=atmosphere,
            )
        except (ArithmeticError, RuntimeError, ValueError) as exc:
            return ControlStageAdvance(
                ModelProviderResult(
                    outputs={},
                    diagnostics={"solver_error": str(exc), "error_type": type(exc).__name__},
                    success=False,
                    outcome="numerical_failure",
                    failure_reason="plenum_solver_failed",
                ),
                gas,
            )
        updated = report.boundary
        pump, bath = updated.pump_work_J - gas.pump_work_J, updated.bath_heat_J - gas.bath_heat_J
        gas_input = sum(updated.added_mol) - sum(gas.added_mol)
        candidate = provider.state.replace(
            pressure_Pa=updated.pressure_pa,
            ledger=advance_ledger(
                provider.state.ledger,
                0.0,
                cost_delta=gas_input * 0.5 + pump / 250000,
                energy_deltas={"gas_pump_work_J": pump, "gas_bath_heat_J": bath},
            ),
        )
        if abs(candidate.ledger.time_s - state.ledger.time_s - dt) > 1e-7:
            raise ValueError("control physical interval was settled more than once")
        return ControlStageAdvance(
            replace(
                provider,
                state=candidate,
                diagnostics={
                    **provider.diagnostics,
                    "plenum_solver": report.solver_diagnostic,
                },
            ),
            updated,
            report.exchanges,
        )

    def mixed_tank(self, **kwargs):
        from chemworld.physchem.coupled_flow import integrate_flow

        return integrate_flow(**kwargs)

    def plug_flow(self, model, inlet_concentrations, **kwargs):
        # PFR's independent coordinate is distance, not elapsed experiment time.
        # The caller settles its requested run once; a residence-time solve does
        # not add a second interval to the campaign clock.
        return model.simulate(inlet_concentrations, **kwargs)
