"""Legacy MPM contracts across extraction of the common particle-electrode kernel.

These checks complement the existing MPM/hysteresis physics tests: no golden
trajectory generated from the new implementation is used as its own reference.
"""

import json
from collections.abc import Mapping
from dataclasses import fields, replace
from math import exp

import numpy as np
import pytest

from chemworld.physchem.battery_models import F, battery_spec_from_dict
from chemworld.physchem.battery_mpm import (
    _closure_jacobian,
    _fields,
    mpm_algebraic,
    mpm_fields,
    mpm_system,
)
from chemworld.physchem.coupled_system import CoupledState, CoupledStateView
from chemworld.physchem.reactor_shared import ScalarControlProgram
from test_battery_hysteresis import ocp
from test_battery_mpm import psd


def legacy_payload(negative_mode="single", positive_mode="single"):
    """Pre-composite public fields, deliberately unequal electrodes and PSDs."""
    value = {
        "model_family": "mpm-isothermal-v1",
        "negative": {"cells": 3},
        "positive": {"cells": 5},
        "negative_distribution": psd(
            (5e-6, 10e-6, 15e-6), (1e-5, 0.7, 0.29999)
        ).to_dict(),
        "positive_distribution": psd((8e-6, 14e-6), (0.25, 0.75)).to_dict(),
    }
    for name, mode in (("negative", negative_mode), ("positive", positive_mode)):
        if mode != "single":
            value[name + "_ocp"] = ocp(name, mode, h=0.15).to_dict()
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def program(current=2e-5):
    return ScalarControlProgram(((0.0, current),), "A")


def checkpoint(state):
    # Existing caller-owned state schema; sorted JSON also changes map order.
    payload = {
        field.name: (dict(value) if isinstance(value, Mapping) else value)
        for field in fields(state)
        for value in (getattr(state, field.name),)
    }
    return CoupledState(**json.loads(json.dumps(payload, sort_keys=True, allow_nan=False)))


def test_old_mpm_payload_keeps_exact_inventory_algebraic_and_event_identities():
    spec = battery_spec_from_dict(legacy_payload())
    expected_inventory = tuple(
        f"{name}:size:{size}:Li:{cell}"
        for name, sizes, cells in (("negative", 3, 3), ("positive", 2, 5))
        for size in range(sizes)
        for cell in range(cells)
    )
    assert spec.inventory_keys == expected_inventory
    assert set(spec.inventory_species.values()) == {"Li"}
    assert spec.counts == (3, 2)
    assert spec.algebraic_names == (
        "battery:negative:size-current:0_A",
        "battery:negative:size-current:2_A",
        "battery:positive:size-current:0_A",
        "battery:mpm:electrolyte-potential_V",
        "battery:current_A",
        "battery:voltage_V",
    )
    assert spec.coordinate_keys == ("discharged_charge_C", "delivered_work_J")
    system = mpm_system(spec, program())
    assert tuple(p.process_id for p in system.processes) == (
        "battery:negative:size:0",
        "battery:negative:size:1",
        "battery:negative:size:2",
        "battery:positive:size:0",
        "battery:positive:size:1",
        "battery:mpm-intercalation",
    )
    assert tuple(event.event_id for event in system.events) == (
        "battery:negative:depleted",
        "battery:negative:saturated",
        "battery:positive:depleted",
        "battery:positive:saturated",
        "battery:minimum-voltage",
        "battery:maximum-voltage",
    )
    closure = mpm_algebraic(spec, program())
    assert tuple(variable.name for variable in closure.variables) == spec.algebraic_names
    assert tuple(equation.name for equation in closure.equations) == (
        "battery:bv:negative:size:0",
        "battery:bv:negative:size:1",
        "battery:bv:negative:size:2",
        "battery:bv:positive:size:0",
        "battery:bv:positive:size:1",
        "battery:control",
    )
    encoded = spec.to_dict()
    assert encoded["model_family"] == "mpm-isothermal-v1"
    assert "negative_ocp" not in encoded and "positive_ocp" not in encoded
    assert not any("phase" in key or "population" in key for key in encoded)
    restored = battery_spec_from_dict(json.loads(json.dumps(encoded, allow_nan=False)))
    assert restored == spec
    assert restored.inventory_keys == expected_inventory
    assert restored.initial_state() == spec.initial_state()
    assert restored.negative_distribution.source == spec.negative_distribution.source


@pytest.mark.parametrize(
    "modes",
    [("single", "single"), ("current_sigmoid", "current_sigmoid"),
     ("axen", "axen"), ("axen", "current_sigmoid")],
)
def test_legacy_private_closure_and_full_jacobian_follow_named_packed_order(modes):
    spec = battery_spec_from_dict(legacy_payload(*modes))
    state = spec.initial_state()
    control = program()
    solved = mpm_fields(spec, state, program=control)
    view = CoupledStateView(
        state.amounts_mol, state.temperature_K, state.volume_L,
        coordinates=state.coordinates, algebraic_values=solved["algebraic_values"],
    )
    raw = _fields(spec, view, view.algebraic_values)
    assert set(raw) == set(solved) - {"closure_diagnostic"}
    for key in raw:
        if key == "algebraic_values":
            assert raw[key] == solved[key]
        else:
            np.testing.assert_array_equal(raw[key], solved[key])

    identities = (*spec.inventory_keys, *spec.coordinate_keys, *spec.algebraic_names)
    permutation = np.r_[np.arange(len(identities))[1::2], np.arange(len(identities))[::2]]
    reordered = tuple(identities[i] for i in permutation)
    direct = _closure_jacobian(spec, view, control, identities).toarray()
    shuffled = _closure_jacobian(spec, view, control, reordered).toarray()
    np.testing.assert_array_equal(shuffled, direct[:, permutation])
    # Extra caller-owned coordinates must have zero influence on electrical closure.
    extra = _closure_jacobian(spec, view, control, (*reordered, "caller:tracer")).toarray()
    np.testing.assert_array_equal(extra[:, :-1], shuffled)
    np.testing.assert_array_equal(extra[:, -1], 0)
    system = mpm_system(spec, control)
    actual = system.residual_rhs_jacobian(view, reordered).toarray()
    canonical = system.residual_rhs_jacobian(view, identities).toarray()
    np.testing.assert_array_equal(actual, canonical[np.ix_(permutation, permutation)])


@pytest.mark.parametrize("axen_electrode", ["negative", "positive"])
def test_mixed_ocp_old_snapshot_continues_every_psd_inventory_and_one_shared_h(axen_electrode):
    modes = ("axen", "current_sigmoid") if axen_electrode == "negative" else (
        "current_sigmoid", "axen"
    )
    spec = battery_spec_from_dict(legacy_payload(*modes))
    initial = spec.initial_state()
    amounts = dict(initial.amounts_mol)
    for name, size, particle, keys, _ in spec.particle_bins:
        # Deliberately distinct sizes and radial profiles: averaging loses information.
        base = (0.69 if name == "negative" else 0.29) + 0.003 * size
        theta = base + np.linspace(0, 0.002, particle.cells)
        amounts.update(zip(
            keys, particle.mesh.cell_volumes_m3 * particle.maximum_concentration_mol_m3 * theta,
            strict=True,
        ))
    initial = replace(initial, amounts_mol=amounts)
    current, duration = 2e-5, 0.1
    control = program(current)
    first = mpm_system(spec, control).advance(initial, duration)
    assert first.outcome == "success"
    saved = checkpoint(first.final_state)
    restored_spec = battery_spec_from_dict(
        json.loads(json.dumps(spec.to_dict(), sort_keys=True, allow_nan=False))
    )
    assert saved == first.final_state
    actual = mpm_system(restored_spec, control).advance(saved, duration)
    expected = mpm_system(spec, control).advance(first.final_state, duration)
    assert actual.outcome == expected.outcome == "success"
    assert actual.final_state == expected.final_state
    final = actual.final_state
    assert set(final.amounts_mol) == set(spec.inventory_keys)
    assert len(final.amounts_mol) == 19
    for key in spec.inventory_keys:
        assert final.amounts_mol[key] > 0
    h_key = "battery:" + axen_electrode + ":hysteresis"
    assert spec.hysteresis_keys == (h_key,)
    assert set(final.coordinates) == {"discharged_charge_C", "delivered_work_J", h_key}
    sign = 1 if axen_electrode == "negative" else -1
    decay = 40 if sign > 0 else 20
    h_expected = sign + (initial.coordinates[h_key] - sign) * exp(
        -decay * current * (2 * duration) / (F * getattr(spec, axen_electrode).capacity_mol)
    )
    assert final.coordinates[h_key] == pytest.approx(h_expected, abs=3e-10)
    assert final.coordinates["discharged_charge_C"] == pytest.approx(
        current * 2 * duration, abs=1e-12
    )
    for name, sign in (("negative", -1), ("positive", 1)):
        keys = [k for k in spec.inventory_keys if k.startswith(name + ":")]
        delta = sum(final.amounts_mol[k] - initial.amounts_mol[k] for k in keys)
        assert F * delta == pytest.approx(sign * current * 2 * duration, abs=1e-10)
    # The tiny negative tail is a real changed inventory, not an omitted/reinflated bin.
    tail = [key for key in spec.inventory_keys if key.startswith("negative:size:0:")]
    assert any(final.amounts_mol[key] != initial.amounts_mol[key] for key in tail)
