"""Separate work ports share the existing finite n/H solve and stop boundary."""

from dataclasses import dataclass, field, replace
from types import SimpleNamespace

import numpy as np
import pytest

from chemworld.physchem.phase_inventory import PhaseInventoryRates, integrate_phase_inventory
from chemworld.physchem.transport_devices import TransportInventory


@dataclass(frozen=True)
class WorkInventory(TransportInventory):
    shaft_import_energy_J: float = 7.0
    shaft_export_energy_J: float = 11.0
    outlet_amounts_mol: dict = field(default_factory=lambda: {"a": 0.5})
    outlet_enthalpy_J: float = 13.0


def integrate(state, contribution, **kwargs):
    return integrate_phase_inventory(
        SimpleNamespace(properties=SimpleNamespace(component_ids=("a",))),
        state,
        duration_s=2,
        rates=lambda _state, _phases: contribution,
        closure_indices=(),
        stop_functions=kwargs.pop("stop_functions", ()),
        enthalpy_bounds=lambda i, n: (-1e6, 1e6),
        **kwargs,
    )


def rates(**kwargs):
    return replace(PhaseInventoryRates(np.zeros((1, 1)), np.array([3.0]), 2, 1, (),
                                       shaft_import_W=5, shaft_export_W=3), **kwargs)


def test_work_heat_and_inventory_are_one_conserved_solve():
    state = WorkInventory(({"a": 2.0},), (10.0,), heater_energy_J=17,
                          condenser_energy_J=19)
    result = integrate(state, rates(), include_shaft_work=True)
    final = result.final_state
    assert final.enthalpies_J == pytest.approx((16,))
    assert final.heater_energy_J == pytest.approx(21)
    assert final.condenser_energy_J == pytest.approx(21)
    assert final.shaft_import_energy_J == pytest.approx(17)
    assert final.shaft_export_energy_J == pytest.approx(17)
    assert abs(result.enthalpy_residual_J) < 1e-11
    assert state.enthalpies_J == (10.0,)


def test_outlet_and_work_integrals_do_not_overlap():
    state = WorkInventory(({"a": 2.0},), (10.0,))
    contribution = rates(amounts_mol_s=np.array([[-0.1]]), enthalpy_W=np.array([1.0]),
                         outlet_amounts_mol_s={"a": 0.1}, outlet_enthalpy_W=2)
    result = integrate(state, contribution, include_shaft_work=True, open_outlet=True)
    final = result.final_state
    assert final.amounts_mol[0]["a"] == pytest.approx(1.8)
    assert final.enthalpies_J == pytest.approx((12,))
    assert final.outlet_amounts_mol == pytest.approx({"a": 0.7})
    assert final.outlet_enthalpy_J == pytest.approx(17)
    assert final.shaft_import_energy_J == pytest.approx(17)
    assert final.shaft_export_energy_J == pytest.approx(17)
    assert abs(result.enthalpy_residual_J) < 1e-11
    assert abs(result.component_residuals_mol["a"]) < 1e-11


def test_work_stops_with_actual_inventory_time():
    state = WorkInventory(({"a": 2.0},), (10.0,))
    result = integrate(state, rates(), include_shaft_work=True,
                       stop_functions=(("enthalpy_limit", lambda s, p: 13-s.enthalpies_J[0]),))
    assert result.actual_duration_s == pytest.approx(1)
    assert result.stopping_event == "enthalpy_limit"
    assert result.final_state.shaft_import_energy_J == pytest.approx(12)
    assert result.final_state.shaft_export_energy_J == pytest.approx(14)


def test_existing_no_work_path_keeps_original_coordinates():
    state = TransportInventory(({"a": 2.0},), (10.0,))
    result = integrate(state, rates(shaft_import_W=0, shaft_export_W=0,
                                    enthalpy_W=np.array([1.0])))
    assert result.final_state.enthalpies_J == pytest.approx((12,))
    assert len(result.solver_diagnostic["minimum_trial_utility_integrals_J"]) == 2
    assert "shaft_accounting" not in result.solver_diagnostic
    assert not hasattr(result.final_state, "shaft_import_energy_J")


def test_nonzero_work_requires_explicit_accounting():
    with pytest.raises(ValueError, match="explicitly enabled"):
        integrate(TransportInventory(({"a": 2.0},), (10.0,)), rates())


@pytest.mark.parametrize("value", [-1.0, float("nan"), float("inf"), True])
def test_invalid_work_ports_are_rejected(value):
    with pytest.raises(ValueError, match="finite nonnegative"):
        integrate(WorkInventory(({"a": 2.0},), (10.0,)),
                  rates(shaft_import_W=value), include_shaft_work=True)


@pytest.mark.parametrize("state", [TransportInventory(({"a": 2.0},), (10.0,)),
                                   WorkInventory(({"a": 2.0},), (10.0,),
                                                 shaft_export_energy_J=-1)])
def test_work_needs_explicit_valid_owned_counters(state):
    with pytest.raises(ValueError, match="owned nonnegative"):
        integrate(state, rates(), include_shaft_work=True)


def test_wrong_enthalpy_rate_cannot_hide_behind_work_counters():
    with pytest.raises(RuntimeError, match="conservation"):
        integrate(WorkInventory(({"a": 2.0},), (10.0,)),
                  rates(enthalpy_W=np.array([2.0])), include_shaft_work=True)
