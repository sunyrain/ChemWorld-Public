"""Versioned finite metal conversion through the original battery consumers."""

import json
from dataclasses import asdict, replace
from math import fsum

import pytest

from chemworld.data.scientific_inputs import (
    InputValue,
    QuantityColumn,
    ScientificRow,
    ScientificTable,
)
from chemworld.foundation.state_ownership import StateSlots
from chemworld.physchem.battery_degradation import SEISpec
from chemworld.physchem.battery_impedance import battery_impedance, battery_impedance_system
from chemworld.physchem.battery_models import (
    BatteryThermalSpec,
    advance_battery,
    battery_chemical_energy_j,
    battery_chemical_potentials,
    battery_dissipation_w,
    battery_readout,
    battery_sei_affinity_j_mol,
    battery_spec_from_dict,
)
from chemworld.physchem.battery_plating import LithiumPlatingSpec
from chemworld.physchem.battery_soh import battery_spec_at_soc, soh_from_battery
from chemworld.physchem.reactor_shared import ScalarControlProgram
from chemworld.runtime import ChemWorldRuntime
from chemworld.runtime.battery import initial_battery_slot, unpack_battery
from chemworld.world.battery_world import battery_world_spec
from test_battery_hysteresis_analysis import operational
from test_battery_plating_compatibility import METAL, old_spec, plating

OLD_VERSION = "pybamm-25.4.2-spm-plating-v1"
PARTIAL_VERSION = "pybamm-25.4.2-constant-decay-plating-v1"


def partial(*, gamma=.2, rate=1e-10, plated=2e-5, dead=3e-5, unit="1/s"):
    old = plating(rate=rate, plated=plated, dead=dead)
    table, row = old.parameters, old.parameters.rows[0]
    parameters = ScientificTable(
        "partial-plating-compatibility", "parameters",
        (*table.columns, QuantityColumn(
            "gamma_s_inv", unit, "parameter", (-1., 1e6), "Synthetic constant conversion rate"
        )),
        table.sources,
        (ScientificRow(
            row.row_id, row.source_id, row.group_id, row.sample_id, row.split,
            {**row.values, "gamma_s_inv": InputValue(gamma, unit, None, "Synthetic constant")},
        ),),
    )
    return LithiumPlatingSpec(parameters, row.row_id, "partially reversible", plated, dead)


def model(**changes):
    return replace(old_spec(), plating=partial(**changes))


@pytest.mark.parametrize("unit", ("1/s", "s^-1"))
def test_partial_parameter_units_source_and_version_roundtrip(unit):
    spec = model(unit=unit)
    encoded = json.loads(json.dumps(spec.to_dict(), allow_nan=False))
    declaration = encoded["plating"]
    assert declaration["model_version"] == PARTIAL_VERSION
    assert declaration["mode"] == "partially reversible"
    assert set(spec.plating.native_parameters) == {
        "k_m_s", "alpha", "ce_mol_m3", "negative_electrode_volume_m3", "gamma_s_inv"
    }
    assert spec.plating.decay_rate_s_inv == .2
    assert spec.plating.deactivation_rate_mol_s(2e-5) == pytest.approx(4e-6, rel=1e-15)
    restored = battery_spec_from_dict(encoded)
    assert restored.to_dict() == encoded
    assert restored.plating.parameters.to_dict() == spec.plating.parameters.to_dict()
    assert restored.initial_state() == spec.initial_state()
    with pytest.raises(TypeError):
        restored.plating.native_parameters["gamma_s_inv"] = 99.
    assert spec.plating.decay_rate_s_inv == .2


@pytest.mark.parametrize("mode", ("reversible", "irreversible"))
def test_old_mode_still_uses_exact_four_column_declaration_and_version(mode):
    declaration = plating(mode=mode)
    encoded = declaration.to_dict()
    assert encoded["model_version"] == OLD_VERSION
    assert len(declaration.native_parameters) == 4
    assert "gamma_s_inv" not in declaration.native_parameters
    assert LithiumPlatingSpec.from_dict(json.loads(json.dumps(encoded))).to_dict() == encoded
    wrong_version = {**encoded, "model_version": PARTIAL_VERSION}
    with pytest.raises(ValueError):
        LithiumPlatingSpec.from_dict(wrong_version)


@pytest.mark.parametrize("defect", (
    "old-version", "unknown-version", "reversible-mode", "irreversible-mode",
    "unknown-mode", "unknown-field",
))
def test_partial_codec_refuses_mixed_version_mode_and_unknown_keys(defect):
    encoded = partial().to_dict()
    if defect == "old-version":
        encoded["model_version"] = OLD_VERSION
    elif defect == "unknown-version":
        encoded["model_version"] = "unregistered-decay-v9"
    elif defect.endswith("-mode"):
        encoded["mode"] = defect.removesuffix("-mode")
    else:
        encoded["inferred_sei_thickness"] = 0.
    with pytest.raises(ValueError):
        LithiumPlatingSpec.from_dict(encoded)


@pytest.mark.parametrize("mode", ("reversible", "irreversible"))
def test_gamma_is_not_silently_accepted_by_original_modes(mode):
    new = partial()
    with pytest.raises(ValueError):
        LithiumPlatingSpec(new.parameters, new.parameter_row_id, mode)


def test_partial_cannot_infer_a_missing_decay_parameter_or_accept_negative_gamma():
    old = plating()
    with pytest.raises(ValueError):
        LithiumPlatingSpec(old.parameters, old.parameter_row_id, "partially reversible")
    with pytest.raises(ValueError):
        partial(gamma=-.1)


def test_zero_decay_retains_the_exact_reversible_numerical_path():
    combined = model(gamma=0.)
    reversible = replace(combined, plating=plating())
    left, right = combined.initial_state(), reversible.initial_state()
    assert left == right
    for current, duration in ((-.1, .2), (0., .15), (.08, .2)):
        control = ScalarControlProgram(((left.time_s, current),), "A")
        actual = advance_battery(
            combined, left, control, duration, evaluation_times_s=(0., duration/2, duration)
        )
        expected = advance_battery(
            reversible, right, control, duration, evaluation_times_s=(0., duration/2, duration)
        )
        assert actual.outcome == expected.outcome == "success"
        assert actual.stopping_event == expected.stopping_event
        assert actual.actual_duration_s == expected.actual_duration_s
        assert actual.trajectory == expected.trajectory
        assert actual.final_state == expected.final_state
        assert actual.solver_diagnostic["plating_mode"] == "partially reversible"
        assert expected.solver_diagnostic["plating_mode"] == "reversible"
        assert {
            k: v for k, v in actual.solver_diagnostic.items() if k != "plating_mode"
        } == {
            k: v for k, v in expected.solver_diagnostic.items() if k != "plating_mode"
        }
        assert actual.conservation_residuals == expected.conservation_residuals
        assert actual.roundoff_corrections_mol == expected.roundoff_corrections_mol
        assert asdict(battery_readout(combined, actual.final_state, current)) == asdict(
            battery_readout(reversible, expected.final_state, current)
        )
        left, right = actual.final_state, expected.final_state


def test_nonzero_initial_dead_inventory_survives_original_json_restore_once():
    spec = model(rate=0.)
    runtime = ChemWorldRuntime(composition=battery_world_spec(spec))
    runtime.reset(seed=46)
    restored = None
    try:
        action = {"operation": "wait", "duration_s": 1.}
        first = runtime.execute(action)
        assert first.info["transaction_status"] == "committed", first.info.get("error_message")
        encoded = json.loads(json.dumps(runtime.snapshot().to_dict(), allow_nan=False))
        restored = ChemWorldRuntime.from_snapshot(encoded)
        record, slot, decoded, physical, _ = unpack_battery(restored._state)
        assert physical.amounts_mol["dead:Li"] > spec.plating.initial_dead_mol > 0
        assert physical.amounts_mol["plating:Li"] < spec.plating.initial_plated_mol
        assert tuple(decoded.inventory_keys[-2:]) == METAL
        assert set(slot.values) == set(spec.inventory_keys) | set(spec.coordinate_keys)
        assert slot.spec.owner_kind == "equipment" and slot.spec.owner_id == record.equipment_id
        assert slot.spec.batch_reset == "preserve"
        assert initial_battery_slot(record, decoded).values["dead:Li"] == 3e-5
        assert all(value == 0 for value in restored._state.species_amounts.values())
        assert all(not p.species_amounts_mol for p in restored._state.phases.phases.values())
        inventory = restored.constitution.material_inventory(restored._state)
        assert inventory == StateSlots((slot,)).species_amounts()
        assert inventory == pytest.approx(
            {"Li": fsum(spec.initial_state().amounts_mol.values())}, rel=0, abs=2e-17
        )
        before = restored.snapshot().to_dict()
        assert unpack_battery(restored._state)[3] == physical
        assert restored.snapshot().to_dict() == before
        for engine in (runtime, restored):
            assert engine.execute(action).info["transaction_status"] == "committed"
        assert json.loads(json.dumps(runtime.snapshot().to_dict())) == json.loads(
            json.dumps(restored.snapshot().to_dict())
        )
    finally:
        runtime.close()
        if restored is not None:
            restored.close()


@pytest.mark.parametrize("analysis", (
    "eis-system", "eis", "soh", "soc", "energy", "chemical-potentials", "dissipation", "sei",
))
def test_zero_rate_partial_declaration_still_refuses_undeclared_analysis(analysis):
    spec = model(gamma=0., rate=0., plated=0., dead=0.)
    state, encoded = spec.initial_state(), spec.to_dict()
    calls = {
        "eis-system": lambda: battery_impedance_system(spec, state),
        "eis": lambda: battery_impedance(spec, [1.], state=state),
        "soh": lambda: soh_from_battery(spec, state),
        "soc": lambda: battery_spec_at_soc(spec, None, .5),
        "energy": lambda: battery_chemical_energy_j(spec, state),
        "chemical-potentials": lambda: battery_chemical_potentials(spec, state),
        "dissipation": lambda: battery_dissipation_w(spec, state, 0.),
        "sei": lambda: battery_sei_affinity_j_mol(spec, state, 0.),
    }
    with pytest.raises(ValueError, match=r"(?i)plating"):
        calls[analysis]()
    assert state == spec.initial_state() and spec.to_dict() == encoded


@pytest.mark.parametrize("unsupported", (
    "MPM", "DFN", "SPMe", "thermal", "sei", "current_sigmoid", "axen", "ode",
))
def test_partial_scope_is_not_inherited_by_other_families_or_joint_physics(unsupported):
    spec = old_spec(unsupported) if unsupported in {"MPM", "DFN", "SPMe"} else old_spec()
    options = {"plating": partial()}
    if unsupported == "thermal":
        options["thermal"] = BatteryThermalSpec()
    elif unsupported == "sei":
        options["sei"] = SEISpec()
    elif unsupported in {"current_sigmoid", "axen"}:
        options["negative_ocp"] = operational("negative", unsupported)
    elif unsupported == "ode":
        options["numerical_form"] = "ode"
    with pytest.raises(ValueError, match=r"(?i)plating"):
        replace(spec, **options)


def test_old_mpm_decoder_refuses_partial_instead_of_discarding_its_mode():
    encoded = old_spec("MPM").to_dict()
    encoded["plating"] = partial().to_dict()
    with pytest.raises(ValueError, match=r"(?i)plating"):
        battery_spec_from_dict(encoded)
