"""Independent pressure work, unchanged gross pumps, original worlds and replay."""

import json
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from chemworld.physchem.distributed_ro import DistributedROSpec, distributed_ro_profile
from chemworld.physchem.pressure_recovery import (
    IsothermalLiquidPressureRecoverySpec as Recovery,
)
from chemworld.physchem.pressure_recovery import (
    evaluate_isothermal_liquid_pressure_recovery as evaluate,
)
from chemworld.runtime import ChemWorldRuntime
from chemworld.world.distributed_ro_world import distributed_ro_world_spec
from reference.ro47_fixtures import feed, spec
from reference.ro_pressure_recovery47_reference import axial, recovery


@pytest.mark.parametrize(
    "family,eta,cap",
    [
        ("ideal", 0.0, 1.0),
        ("ideal", 0.75, 1.0),
        ("ideal", 1.0, 1.0),
        ("ideal", 0.75, 0.02),
        ("Davies", 0.75, 1.0),
        ("Davies", 1.0, 1.0),
    ],
)
def test_same_table_isothermal_work_heat_and_entropy(family, eta, cap):
    model = spec(family)
    rates = feed(model, water_mol=0.01)
    device = Recovery(eta, cap, 1.5e6, 1e-6)
    result = evaluate(
        device,
        model.properties,
        rates,
        temperature_K=310.0,
        inlet_pressure_Pa=6e5,
        outlet_pressure_Pa=1e5,
    )
    ref = recovery(family, rates, 310.0, 6e5, 1e5, eta, cap)
    for name, key in (
        ("pressure_availability_W", "availability"),
        ("recovered_shaft_power_W", "work"),
        ("device_heat_into_liquid_W", "heat"),
        ("entropy_production_W_K", "entropy"),
    ):
        assert result[name] == pytest.approx(ref[key], abs=1e-9)
    ds = result["outlet"]["entropy_J_K"] - result["inlet"]["entropy_J_K"]
    assert ds - result["device_heat_into_liquid_W"] / 310 == pytest.approx(
        ref["entropy"], abs=1e-12
    )
    if family == "Davies" and eta == 1:
        assert abs(result["device_heat_into_liquid_W"]) > 1e-10


def test_zero_flow_has_no_fabricated_liquid_or_density():
    model = spec()
    rates = dict.fromkeys(model.properties.component_ids, 0.0)
    result = evaluate(
        Recovery(0.8, 1.0, 1.5e6, 1e-6),
        model.properties,
        rates,
        temperature_K=310.0,
        inlet_pressure_Pa=6e5,
        outlet_pressure_Pa=1e5,
    )
    assert result["no_material"] and result["inlet"] is result["outlet"] is None
    assert result["recovered_shaft_power_W"] == result["pressure_availability_W"] == 0


def test_v1_exact_default_and_v2_codec_and_gross_pump_path():
    old = spec()
    before = old.to_dict()
    new = replace(old, energy_recovery=Recovery(0.8, 0.02, 1.5e6, 1e-6))
    assert before["schema"] == "declared-distributed-ro-v1" and "energy_recovery" not in before
    assert new.to_dict()["schema"] == "declared-distributed-ro-v2"
    assert (
        DistributedROSpec.from_dict(json.loads(json.dumps(new.to_dict()))).to_dict()
        == new.to_dict()
    )
    a, b = (distributed_ro_profile(s, feed(s), 10 / 6e7) for s in (old, new))
    for key in (
        "stations",
        "inlet_mol_s",
        "permeate_mol_s",
        "retentate_mol_s",
        "shaft_power_W",
        "hydraulic_power_W",
        "actual_flow_m3_s",
    ):
        assert a[key] == b[key]
    assert old.to_dict() == before
    with pytest.raises(ValueError, match="legacy"):
        DistributedROSpec.from_dict({**before, "energy_recovery": new.energy_recovery.to_dict()})


def physical(runtime):
    return {
        name: {
            "n": tank.species_amounts,
            "H": tank.state_slots.to_dict(),
            "T": tank.temperature_K,
            "P": tank.pressure_Pa,
            "V": tank.volume_L,
            "phases": tank.phases.to_dict(),
        }
        for name, tank in {
            runtime._state.vessel_id: runtime._state,
            **runtime._state.inactive_vessels,
        }.items()
    }


def setup(shared=False, event=False, rating=None):
    model = replace(spec(), energy_recovery=rating or Recovery(0.8, 1.0, 1.5e6, 1e-6))
    n = feed(model)
    if event:
        rate = axial(model, n)["recovery"]["volume"]
        model = replace(model, retentate_capacity_L=1000 * rate * 1.5)
    runtime = ChemWorldRuntime(composition=distributed_ro_world_spec(model, n, shared_clock=shared))
    runtime.reset(seed=47)
    outcome = runtime.execute({"operation": "set_flow_rate", "flow_rate_mL_min": 10})
    assert outcome.info["transaction_status"] == "committed"
    return runtime


@pytest.mark.parametrize("shared", [False, True])
def test_actual_three_tank_original_flow_heat_work_and_shared_clock(shared):
    runtime = setup(shared)
    result = runtime.execute({"operation": "wait" if shared else "run_flow", "duration_s": 2.0})
    assert result.info["transaction_status"] == "committed", result.info
    ledger = runtime._state.ledger
    assert ledger.time_s == pytest.approx(2.0) and ledger.flow_shaft_work_J < 0
    assert ledger.flow_pump_work_J > 0 and ledger.flow_electrical_work_J == 0
    assert ledger.cost == pytest.approx(ledger.flow_pump_work_J / 250000)
    assert all(
        t.ledger.time_s == pytest.approx(2.0) for t in runtime._state.inactive_vessels.values()
    )


def test_full_new_process_continuation_at_receiver_event(tmp_path):
    runtime = setup(shared=True, event=True)
    first = runtime.execute({"operation": "wait", "duration_s": 0.5})
    assert first.info["transaction_status"] == "committed", first.info.get("error_message")
    assert runtime._state.ledger.time_s == pytest.approx(0.5)
    source, output = tmp_path / "snapshot.json", tmp_path / "continued.json"
    source.write_text(json.dumps(runtime.snapshot().to_dict()), encoding="utf-8")
    result = runtime.execute({"operation": "wait", "duration_s": 2.0})
    assert result.info["transaction_status"] == "committed", result.info
    assert result.info["operation_outcome"] == "stopped"
    assert runtime._state.ledger.time_s == pytest.approx(1.5, abs=2e-7)
    completed = subprocess.run(
        [
            "uv",
            "run",
            "--no-sync",
            "python",
            str(Path(__file__).parent / "reference" / "ro_pressure_recovery47_resume.py"),
            str(source),
            str(output),
            "2",
        ],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(output.read_text(encoding="utf-8"))["snapshot"] == json.loads(
        json.dumps(runtime.snapshot().to_dict())
    )


@pytest.mark.parametrize(
    "rating,reason",
    [
        (Recovery(0.8, 1.0, 1e4, 1e-6), "pressure-drop"),
        (Recovery(0.8, 1.0, 1.5e6, 1e-12), "inlet-volume-flow"),
    ],
)
def test_rating_rejection_is_atomic(rating, reason):
    runtime = setup(rating=rating)
    before = physical(runtime)
    result = runtime.execute({"operation": "wait", "duration_s": 1.0})
    assert result.info["operation_outcome"] == "numerical_failure"
    assert reason in str(result.info["outcome_diagnostics"])
    assert physical(runtime) == before and runtime._state.ledger.time_s == 0
    assert runtime._state.ledger.flow_shaft_work_J == runtime._state.ledger.flow_pump_work_J == 0


@pytest.mark.parametrize(
    "temperature,pin,pout,reason",
    [(310.0, 1e5, 6e5, "forward"), (350.0, 6e5, 1e5, "domain|range|outside")],
)
def test_physical_direction_and_temperature_domain_reject(temperature, pin, pout, reason):
    model = spec()
    with pytest.raises(ValueError, match=reason):
        evaluate(
            Recovery(0.8, 1.0, 1.5e6, 1e-6),
            model.properties,
            feed(model, water_mol=0.01),
            temperature_K=temperature,
            inlet_pressure_Pa=pin,
            outlet_pressure_Pa=pout,
        )
