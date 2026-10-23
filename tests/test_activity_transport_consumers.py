"""Excess enthalpy is consumed by existing finite liquid n/H inventories."""

import json
from dataclasses import replace

import pytest

from chemworld.physchem import ActivityModelSpec, regular_solution_model
from chemworld.physchem.activity_thermodynamics import activity_thermodynamic_report
from chemworld.physchem.transport_devices import FiniteTransportSpec, LiquidTransportProperties
from chemworld.runtime import ChemWorldRuntime
from chemworld.runtime.session_snapshot import SessionSnapshot
from chemworld.runtime.transport_devices import unpack_transport
from chemworld.world.transport_world import synthetic_transport_properties, transport_world_spec


def regular_properties():
    original = synthetic_transport_properties()
    model = regular_solution_model(
        "synthetic-endothermic-liquid", original.component_ids,
        molar_volumes_m3_mol={"water": 1e-4, "ethanol": 1e-4},
        solubility_parameters_Pa_half={"water": 18000, "ethanol": 16000},
        provenance={"scope": "synthetic homogeneous liquid consumer"},
    )
    return replace(original, activity_model=model)


def test_stream_and_inventory_share_excess_calorics_and_original_volume():
    properties = regular_properties()
    amounts = {"water": .6, "ethanol": .4}
    temperature = 315.0
    expected_h = (75 * .6 + 110 * .4) * (temperature - 298.15) + 400 * .6 * .4
    assert properties.enthalpy(amounts, temperature) == pytest.approx(expected_h)
    assert properties.heat_capacity(amounts, temperature) == pytest.approx(75 * .6 + 110 * .4)
    fluid, h, volume = properties.evaluate(amounts, temperature)
    assert h == pytest.approx(expected_h)
    ideal = replace(properties, activity_model=None).evaluate(amounts, temperature)
    assert volume == ideal[2] and fluid.density_kg_m3 == ideal[0].density_kg_m3
    assert fluid.viscosity_Pa_s == ideal[0].viscosity_Pa_s
    restored = LiquidTransportProperties.from_dict(json.loads(json.dumps(properties.to_dict())))
    assert restored.activity_model.to_dict() == properties.activity_model.to_dict()
    assert restored.enthalpy(amounts, temperature) == properties.enthalpy(amounts, temperature)
    assert "activity_model" not in replace(properties, activity_model=None).to_dict()


def test_temperature_dependent_excess_capacity_reaches_heat_transfer():
    original = synthetic_transport_properties()
    activity = ActivityModelSpec("temperature-dependent", original.component_ids, "wilson",
        {"lambda_b:water|ethanol": 150, "lambda_b:ethanol|water": -90})
    properties = replace(original, activity_model=activity)
    amounts = {"water": .6, "ethanol": .4}
    excess = activity_thermodynamic_report(activity, amounts, temperature_K=315)
    assert abs(excess.heat_capacity_excess_J_mol_K) > .01
    expected = 75 * .6 + 110 * .4 + excess.heat_capacity_excess_J_mol_K
    assert properties.heat_capacity(amounts, 315) == pytest.approx(expected)
    fluid = properties.evaluate(amounts, 315)[0]
    mass = sum(p.component.molecular_weight_g_mol * amounts[p.component.identifier] / 1000
               for p in original.packages)
    assert fluid.heat_capacity_J_kg_K == pytest.approx(expected / mass)
    derivative = (properties.enthalpy(amounts, 315.001)
                  - properties.enthalpy(amounts, 314.999)) / .002
    assert derivative == pytest.approx(expected, rel=1e-8)
    with pytest.raises(ValueError, match="identities"):
        replace(original, activity_model=ActivityModelSpec("foreign", ("foreign",)))


def execute(runtime, operation, **kwargs):
    result = runtime.execute({"operation": operation, **kwargs})
    assert result.info["transaction_status"] == "committed", result.info
    return result


@pytest.mark.parametrize("algorithm", ("residual", "brentq"))
def test_actual_nonideal_mixing_cools_receiver_and_restores_same_inventory(algorithm, tmp_path):
    spec = FiniteTransportSpec(regular_properties(), caloric_algorithm=algorithm)
    spec = replace(spec, path=replace(spec.path, exchanger_area_m2=0))
    world = transport_world_spec(spec, feed_temperature_K=300, receiver_temperature_K=300,
        feed_mole_fractions={"water": 1.0, "ethanol": 0.0},
        receiver_mole_fractions={"water": 0.0, "ethanol": 1.0})
    runtime = ChemWorldRuntime(composition=world)
    runtime.reset(seed=24)
    _, _, start = unpack_transport(runtime._state)
    execute(runtime, "set_flow_rate", flow_rate_mL_min=5)
    result = execute(runtime, "run_flow", duration_s=5)
    _, _, end = unpack_transport(runtime._state)
    for key in spec.properties.component_ids:
        assert sum(n[key] for n in end.amounts_mol) == pytest.approx(
            sum(n[key] for n in start.amounts_mol), abs=1e-12)
    receiver = end.amounts_mol[1]
    nw, ne = receiver["water"], receiver["ethanol"]
    independent_temperature = 298.15 + (end.enthalpies_J[1] - 400 * nw * ne / (nw + ne)) / (
        75 * nw + 110 * ne + spec.capacities_j_k[1])
    assert runtime._state.temperature_K == pytest.approx(independent_temperature, abs=1e-7)
    assert runtime._state.temperature_K < 300
    diagnostics = result.info["outcome_diagnostics"]["transport"]
    assert sum(end.enthalpies_J) - sum(start.enthalpies_J) == pytest.approx(
        diagnostics["hydraulic_work_J"], abs=1e-7)
    path = tmp_path / "snapshot.json"
    runtime.snapshot().save(path)
    resumed = ChemWorldRuntime.from_snapshot(SessionSnapshot.load(path))
    execute(runtime, "wait", duration_s=5)
    execute(resumed, "wait", duration_s=5)
    assert unpack_transport(resumed._state)[2] == unpack_transport(runtime._state)[2]
