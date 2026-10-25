"""Original two-tank stocks, true shaft work, thermal ownership and exact replay."""

import copy
import json
from dataclasses import replace
from math import exp, log

import pytest

from chemworld.examples.gas_shaft_machine import (
    gas_shaft_example_document,
    gas_shaft_example_inputs,
    gas_shaft_example_network,
    gas_shaft_machine_world,
)
from chemworld.physchem.gas_reactor_network import (
    GasReactorNetworkSpec,
    advance_gas_reactor_network,
    gas_network_coordinates,
    pack_gas_reactor_network_state,
)
from chemworld.physchem.gas_shaft_machine import GasShaftMachineSpec, gas_shaft_operating_point
from chemworld.physchem.reaction_network import R_J_PER_MOL_K as R
from chemworld.runtime import ChemWorldRuntime


def point(mode="compressor", **changes):
    spec, network, _ = gas_shaft_example_inputs(mode)
    inputs = {
        "inlet_temperature_K": 400.0,
        "inlet_pressure_Pa": 100000.0,
        "outlet_pressure_Pa": 200000.0 if mode == "compressor" else 50000.0,
        "mole_fractions": {"A": 0.7, "B": 0.3},
        "requested_molar_rate_mol_s": 0.001,
    }
    inputs.update(changes)
    return gas_shaft_operating_point(
        spec.shaft_machine, network.imported_gas_thermochemistry, **inputs
    )


@pytest.mark.parametrize("mode", ["compressor", "expander"])
def test_constant_cp_closure_is_same_reference_ps_eta_ph(mode):
    report = point(mode)
    cp, ratio = 34.4, 2.0 if mode == "compressor" else 0.5
    ts = 400.0 * exp(R / cp * log(ratio))
    change = (ts - 400.0) / 0.8 if mode == "compressor" else (ts - 400.0) * 0.8
    assert report["isentropic_temperature_K"] == pytest.approx(ts, abs=1e-8)
    assert report["outlet_temperature_K"] == pytest.approx(400.0 + change, abs=1e-8)
    assert report["shaft_work_J_mol"] == pytest.approx(cp * change, abs=1e-6)
    assert report["entropy_generation_J_mol_K"] >= -1e-8


@pytest.mark.parametrize("mode", ["compressor", "expander"])
def test_power_cap_limits_real_moles_not_just_reported_power(mode):
    spec, network, amounts = gas_shaft_example_inputs(mode, maximum_power_W=0.5)
    result, detail = advance_gas_reactor_network(
        spec, network, pack_gas_reactor_network_state(spec, network, amounts), 1.0
    )
    flow = detail["connections"]["shaft-feed"]
    assert abs(detail["shaft_work_J"]) == pytest.approx(0.5, abs=2e-8)
    assert sum(flow["amounts_mol"].values()) < 0.001
    assert flow["delivered_enthalpy_J"] - flow["enthalpy_J"] == pytest.approx(
        detail["shaft_work_J"], abs=2e-8
    )
    assert detail["nodes"]["source"]["internal_energy_change_J"] == pytest.approx(
        -flow["enthalpy_J"], abs=2e-6
    )
    assert detail["nodes"]["receiver"]["internal_energy_change_J"] == pytest.approx(
        flow["delivered_enthalpy_J"], abs=2e-6
    )
    assert result.energy_balance_residual_J == pytest.approx(0.0, abs=2e-6)


@pytest.mark.parametrize("mode", ["compressor", "expander"])
def test_source_receiver_anchor_swap_preserves_actual_process(mode):
    answers = []
    for anchor in ("source", "receiver"):
        spec, network, amounts = gas_shaft_example_inputs(mode, anchor=anchor)
        _, report = advance_gas_reactor_network(
            spec, network, pack_gas_reactor_network_state(spec, network, amounts), 1.0
        )
        answers.append(report)
    for name in ("source", "receiver"):
        for key in ("temperature_K", "pressure_Pa", "internal_energy_change_J"):
            assert answers[0]["nodes"][name][key] == pytest.approx(
                answers[1]["nodes"][name][key], abs=1e-5
            )
        assert answers[0]["nodes"][name]["amounts_mol"] == pytest.approx(
            answers[1]["nodes"][name]["amounts_mol"], abs=1e-10
        )
    assert answers[0]["shaft_work_J"] == pytest.approx(answers[1]["shaft_work_J"], abs=2e-7)


@pytest.mark.parametrize("mode", ["compressor", "expander"])
def test_species_energy_reference_shifts_cancel_in_material_ports(mode):
    spec, network, amounts = gas_shaft_example_inputs(mode)
    document = gas_shaft_example_document()
    for row, shift in zip(document["species"], (10000.0, -7000.0), strict=True):
        row["thermo"]["h0"] = f"{float(row['thermo']['h0'].split()[0]) + shift} J/mol"
    shifted = gas_shaft_example_network(document)
    answers = [
        advance_gas_reactor_network(
            spec, model, pack_gas_reactor_network_state(spec, model, amounts), 1.0
        )[1]
        for model in (network, shifted)
    ]
    for name in ("source", "receiver"):
        assert answers[0]["nodes"][name]["temperature_K"] == pytest.approx(
            answers[1]["nodes"][name]["temperature_K"], abs=1e-7
        )
        assert answers[0]["nodes"][name]["amounts_mol"] == pytest.approx(
            answers[1]["nodes"][name]["amounts_mol"], abs=1e-12
        )
    assert answers[0]["shaft_work_J"] == pytest.approx(answers[1]["shaft_work_J"], abs=2e-7)
    flow0, flow1 = [a["connections"]["shaft-feed"] for a in answers]
    offset = flow0["amounts_mol"]["A"] * 10000.0 - flow0["amounts_mol"]["B"] * 7000.0
    for key in ("enthalpy_J", "delivered_enthalpy_J"):
        assert flow1[key] - flow0[key] == pytest.approx(offset, abs=2e-7)


@pytest.mark.parametrize(
    "change",
    [
        {"isentropic_efficiency": 0},
        {"isentropic_efficiency": 1.1},
        {"maximum_shaft_power_W": 0},
        {"maximum_pressure_ratio": 1},
        {"mode": "valve"},
        {"maximum_outlet_temperature_K": float("inf")},
        {"temperature_range_K": (3000.0, 100.0)},
    ],
)
def test_machine_requires_explicit_physical_limits(change):
    with pytest.raises(ValueError):
        replace(gas_shaft_example_inputs()[0].shaft_machine, **change)


def test_no_machine_keeps_original_codec_and_coordinate_layout():
    spec, network, _ = gas_shaft_example_inputs()
    original = replace(spec, shaft_machine=None)
    payload = original.to_dict()
    assert "shaft_machine" not in payload
    assert payload["model"] == "finite-thermal-gas-network-v1"
    assert GasReactorNetworkSpec.from_dict(json.loads(json.dumps(payload))) == original
    assert all("gas_shaft" not in c.name for c in gas_network_coordinates(original, network))
    assert GasReactorNetworkSpec.from_dict(json.loads(json.dumps(spec.to_dict()))) == spec
    assert GasShaftMachineSpec.from_dict(spec.shaft_machine.to_dict()) == spec.shaft_machine


@pytest.mark.parametrize("defect", ["isothermal", "reactive", "valve", "wrong-edge", "extra-node"])
def test_network_rejects_unqualified_machine_attachments(defect):
    spec, _, _ = gas_shaft_example_inputs()
    with pytest.raises(ValueError):
        if defect == "isothermal":
            replace(spec, energy_nodes=())
        elif defect == "reactive":
            replace(spec, nodes=(replace(spec.nodes[0], chemistry_enabled=True), spec.nodes[1]))
        elif defect == "valve":
            replace(spec, connections=(replace(spec.connections[0], mode="pressure-valve"),))
        elif defect == "wrong-edge":
            replace(spec, shaft_machine=replace(spec.shaft_machine, connection_id="unknown"))
        else:
            replace(spec, nodes=(*spec.nodes, replace(spec.nodes[0], vessel_id="third")))


def test_off_wrong_direction_and_zero_head_do_not_invent_work():
    assert point(enabled=False)["molar_rate_mol_s"] == 0
    assert point(requested_molar_rate_mol_s=0)["shaft_power_W"] == 0
    assert point(outlet_pressure_Pa=50000.0)["status"] == "blocked_pressure_direction"
    equal = point(outlet_pressure_Pa=100000.0)
    assert equal["outlet_temperature_K"] == 400.0
    assert equal["shaft_work_J_mol"] == 0


def test_original_machine_world_resource_receipt_and_exact_json():
    session = ChemWorldRuntime(composition=gas_shaft_machine_world())
    session.reset(seed=31)
    replay = None
    try:
        action = session.execute({"operation": "wait", "duration_s": 1.0})
        assert action.info["transaction_status"] == "committed", action.info
        assert session._state.ledger.flow_shaft_work_J > 0
        assert session._state.ledger.boundary_work_J == session._state.ledger.energy_jacket_J == 0
        payload = json.loads(json.dumps(session.snapshot().to_dict(), allow_nan=False))
        replay = ChemWorldRuntime.from_snapshot(payload)
        for runtime in (session, replay):
            result = runtime.execute({"operation": "wait", "duration_s": 1.0})
            assert result.info["transaction_status"] == "committed", result.info
        assert session._state.to_dict() == replay._state.to_dict()
        assert session._state.ledger.time_s == 2.0
        bad = copy.deepcopy(payload)

        # A physical model edited only in the runtime state cannot bypass the
        # composition's original shaft declaration during snapshot restore.
        def corrupt(value):
            if isinstance(value, dict):
                if value.get("model") == "finite-gas-shaft-machine-v1":
                    value["isentropic_efficiency"] = 0.5
                    return True
                return any(corrupt(v) for v in value.values())
            if isinstance(value, list):
                return any(corrupt(v) for v in value)
            return False

        assert corrupt(bad["world_state"])
        with pytest.raises(ValueError):
            ChemWorldRuntime.from_snapshot(bad)
    finally:
        session.close()
        if replay:
            replay.close()
