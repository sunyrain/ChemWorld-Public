"""New multisolute path must work through the original finite-world consumer."""

import json
from dataclasses import replace

import pytest

from chemworld.physchem.distributed_ro import DistributedROSpec, distributed_ro_profile
from chemworld.runtime import ChemWorldRuntime
from chemworld.world.distributed_ro_world import distributed_ro_world_spec
from reference.ro47close_fixtures import mcas_feed, mcas_spec, multisolute_feed, multisolute_spec


@pytest.mark.parametrize("shared", [False, True])
def test_multisolute_original_world_and_resume(shared):
    model = multisolute_spec()
    assert DistributedROSpec.from_dict(model.to_dict()).to_dict() == model.to_dict()
    runtime = ChemWorldRuntime(
        composition=distributed_ro_world_spec(model, multisolute_feed(model), shared_clock=shared)
    )
    restored = None
    try:
        runtime.reset(seed=47)
        for action in (
            {"operation": "set_flow_rate", "flow_rate_mL_min": 10},
            {"operation": "wait", "duration_s": 2},
        ):
            result = runtime.execute(action)
            assert result.info["transaction_status"] == "committed", result.info
        restored = ChemWorldRuntime.from_snapshot(
            json.loads(json.dumps(runtime.snapshot().to_dict()))
        )
        for current in (runtime, restored):
            result = current.execute({"operation": "wait", "duration_s": 2})
            assert result.info["transaction_status"] == "committed", result.info
        assert runtime._state.to_dict() == restored._state.to_dict()
        for tank in (runtime._state, *runtime._state.inactive_vessels.values()):
            model.properties.validate_amounts(tank.species_amounts)
    finally:
        runtime.close()
        if restored:
            restored.close()


def test_multi_salt_fluxes_keep_every_ion_and_local_neutrality():
    model = multisolute_spec()
    report = distributed_ro_profile(model, multisolute_feed(model), 10e-6 / 60)
    for key, amount in report["inlet_mol_s"].items():
        assert report["permeate_mol_s"][key] + report["retentate_mol_s"][key] == pytest.approx(
            amount
        )
    assert report["pressure_drop_Pa"] > 0
    for station in report["stations"]:
        assert station["charge_flux_mol_m2_s"] == 0
        assert abs(station["water_flux_residual_m_s"]) < 1e-12


@pytest.mark.parametrize("shared", [False, True])
def test_mcas_real_neutral_solute_and_ions_original_world(shared):
    spec = replace(
        mcas_spec(), membrane_area_m2=0.001, channel=replace(mcas_spec().channel, length_m=0.05)
    )
    runtime = ChemWorldRuntime(
        composition=distributed_ro_world_spec(spec, mcas_feed(spec), shared_clock=shared)
    )
    try:
        runtime.reset(seed=47)
        for action in (
            {"operation": "set_flow_rate", "flow_rate_mL_min": 10},
            {"operation": "wait", "duration_s": 2},
        ):
            result = runtime.execute(action)
            assert result.info["transaction_status"] == "committed", result.info
        assert runtime._state.inactive_vessels["permeate"].species_amounts["urea"] > 0
    finally:
        runtime.close()


def test_mcas_zero_solute_limit_and_no_fictitious_gibbs_recovery():
    from chemworld.physchem.pressure_recovery import (
        IsothermalLiquidPressureRecoverySpec,
        evaluate_isothermal_liquid_pressure_recovery,
    )

    model = mcas_spec()
    feed = model.properties.amounts(50.0, 0.0)
    profile = distributed_ro_profile(model, feed, model.properties.volume(feed, 298.15, 101325.0))
    assert all(profile["permeate_mol_s"][key] == 0.0 for key in model.properties.solute_ids)
    liquid = model.properties.liquid(feed, 298.15, 5e6)
    assert not hasattr(liquid, "gibbs_energy_J")
    recovery = IsothermalLiquidPressureRecoverySpec(0.7, 1.0, 10e6, 1.0)
    with pytest.raises(ValueError, match=r"G|Gibbs"):
        replace(model, energy_recovery=recovery)
    with pytest.raises(ValueError, match="Gibbs"):
        evaluate_isothermal_liquid_pressure_recovery(
            recovery,
            model.properties,
            feed,
            temperature_K=298.15,
            inlet_pressure_Pa=5e6,
            outlet_pressure_Pa=101325.0,
        )
