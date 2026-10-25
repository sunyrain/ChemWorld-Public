"""Source-selected multifluid declarations consume the existing actual phase ledger."""

import json
from dataclasses import replace

import pytest
from tests.test_multifluid_catalog import catalog_payload, selected

from chemworld.data.multifluid_catalog import load_multifluid_catalog
from chemworld.foundation.state_ledgers import PhaseLedger, PhaseRecord
from chemworld.physchem.molecular_flash import MolecularDensityBasin
from chemworld.runtime.molecular_phase_inventory import (
    MolecularPhaseEquilibriumSpec,
    MolecularPhaseTarget,
    equilibrate_molecular_phase_inventory,
    molecular_phase_stream,
)


def specification(count=4):
    model = selected(load_multifluid_catalog(catalog_payload(count)))
    return MolecularPhaseEquilibriumSpec(
        model,
        (MolecularDensityBasin("vapor", (1.0, 2000.0), 40.0),),
        (MolecularPhaseTarget("gas", "vapor", "gas", selected=True),),
        model.caloric_model(),
    )


@pytest.mark.parametrize("count", [4, 8])
def test_source_model_real_inventory_codec_enthalpy_and_live_stream(count):
    spec = specification(count)
    restored = MolecularPhaseEquilibriumSpec.from_dict(json.loads(json.dumps(spec.to_dict())))
    assert restored == spec
    feed = dict.fromkeys(spec.residual_model.component_ids, 1.0 / count)
    stock = PhaseLedger(
        {
            "feed": PhaseRecord("feed", "vessel", "gas", 20.0, feed),
            "external": PhaseRecord("external", "elsewhere", "solid", 0.01, {"other": 0.1}),
        }
    )
    result = equilibrate_molecular_phase_inventory(
        stock,
        restored,
        vessel_id="vessel",
        source_phase_ids=("feed",),
        temperature_K=450.0,
        pressure_Pa=1e5,
    )
    assert result.success, result.to_dict()
    assert result.ledger.total_amounts_mol() == pytest.approx(stock.total_amounts_mol(), rel=5e-13)
    assert result.ledger.phases["external"] == stock.phases["external"]
    assert result.total_enthalpy_J == pytest.approx(sum(result.phase_enthalpies_J.values()))
    stream = molecular_phase_stream(
        result.ledger, restored, vessel_id="vessel", temperature_K=450.0, molar_rate_mol_s=0.02
    )
    assert stream.enthalpy_rate_W == pytest.approx(0.02 * result.total_enthalpy_J, rel=2e-10)
    rates = stream.contribution({k: f"port:{k}" for k in feed}).port_rates_mol_s
    assert sum(rates.values()) == pytest.approx(0.02)
    # Source selection does not become a second, cached composition owner.
    gas = result.ledger.phases["gas"]
    changed = dict(gas.species_amounts_mol)
    changed[spec.residual_model.component_ids[0]] *= 2
    modified = PhaseLedger(
        {**result.ledger.phases, "gas": replace(gas, species_amounts_mol=changed)}
    )
    updated = molecular_phase_stream(
        modified, restored, vessel_id="vessel", temperature_K=450.0, molar_rate_mol_s=0.02
    )
    assert updated.state.mole_fractions != stream.state.mole_fractions
    json.dumps(result.to_dict(), allow_nan=False)


def test_failed_source_domain_leaves_actual_inventory_untouched():
    spec = specification()
    feed = dict.fromkeys(spec.residual_model.component_ids, 0.25)
    stock = PhaseLedger({"feed": PhaseRecord("feed", "vessel", "gas", 20.0, feed)})
    result = equilibrate_molecular_phase_inventory(
        stock,
        spec,
        vessel_id="vessel",
        source_phase_ids=("feed",),
        temperature_K=50.0,
        pressure_Pa=1e5,
    )
    assert not result.success and result.ledger is stock
    assert result.failure_reason == "no_reference_pressure_root"


def test_density_trial_pressure_does_not_override_valid_source_target():
    from tests.test_multifluid import synthetic_model

    from chemworld.physchem.molecular_eos import molecular_density_from_pressure

    original = synthetic_model(2, ideal=True)
    model = replace(
        original,
        fluids=tuple(replace(f, maximum_pressure_Pa=1e5) for f in original.fluids),
    )
    # The declared Newton seed exceeds the source pressure ceiling; the
    # requested physical endpoint is well inside it. It is not a new EOS domain.
    with pytest.raises(ValueError, match="pressure outside common pure-source validity"):
        model.state(300.0, 700.0, (0.3, 0.7))
    result = molecular_density_from_pressure(
        model, 300.0, 5e4, (0.3, 0.7), initial_density_mol_m3=700.0,
        density_bounds_mol_m3=(1.0, 3000.0),
    )
    assert result.pressure_Pa == pytest.approx(5e4, rel=2e-12)
    assert result == model.state(300.0, result.molar_density_mol_m3, (0.3, 0.7))
    assert all(fluid.maximum_pressure_Pa == 1e5 for fluid in model.fluids)


def test_density_trial_keeps_source_temperature_domain():
    from tests.test_multifluid import synthetic_model

    from chemworld.physchem.molecular_eos import molecular_density_from_pressure

    with pytest.raises(ValueError, match="temperature outside common pure-source validity"):
        molecular_density_from_pressure(
            synthetic_model(2, ideal=True), 50.0, 5e4, (0.3, 0.7),
            initial_density_mol_m3=700.0, density_bounds_mol_m3=(1.0, 3000.0),
        )


@pytest.mark.parametrize("propane_fraction", [0.5, 0.7])
def test_real_source_two_phase_inventory_uses_live_selected_stream(propane_fraction):
    pytest.importorskip("teqp")
    from tests.reference.run_multifluid_catalog import select, source_catalog

    model = select(source_catalog(), ("n-Propane", "n-Butane"))
    spec = MolecularPhaseEquilibriumSpec(
        model,
        (MolecularDensityBasin("vapor", (1.0, 3000.0), 500.0),
         MolecularDensityBasin("liquid", (7000.0, 22000.0), 12000.0)),
        (MolecularPhaseTarget("gas", "vapor", "gas", selected=True),
         MolecularPhaseTarget("liquid", "liquid", "liquid")),
        model.caloric_model(),
    )
    spec = MolecularPhaseEquilibriumSpec.from_dict(json.loads(json.dumps(spec.to_dict())))
    amounts = dict(zip(model.component_ids, (propane_fraction, 1 - propane_fraction), strict=True))
    stock = PhaseLedger({"feed": PhaseRecord("feed", "vessel", "gas", 10.0, amounts)})
    result = equilibrate_molecular_phase_inventory(
        stock, spec, vessel_id="vessel", source_phase_ids=("feed",),
        temperature_K=300.0, pressure_Pa=6e5, max_iterations=120,
    )
    assert result.success and result.flash.phase_status == "two_phase", result.to_dict()
    assert result.ledger.total_amounts_mol() == pytest.approx(amounts, rel=5e-13)
    assert set(result.installed_phase_ids) == {"gas", "liquid"}
    assert all(s.complete and not s.unstable for s in result.flash.post_split_stability)
    assert result.flash.chemical_potential_error_RT <= 2e-9
    for phase_id in ("gas", "liquid"):
        phase = result.ledger.phases[phase_id]
        total = sum(phase.species_amounts_mol.values())
        assert 0 < total < 1
        stream = molecular_phase_stream(
            result.ledger, spec, vessel_id="vessel", phase_id=phase_id,
            temperature_K=300.0, molar_rate_mol_s=0.02,
        )
        assert stream.enthalpy_rate_W == pytest.approx(
            0.02 * result.phase_enthalpies_J[phase_id] / total, rel=2e-10,
        )
    gas = molecular_phase_stream(
        result.ledger, spec, vessel_id="vessel", temperature_K=300.0, molar_rate_mol_s=0.02,
    )
    liquid_selected = PhaseLedger({
        key: replace(phase, selected=key == "liquid")
        for key, phase in result.ledger.phases.items()
    })
    liquid = molecular_phase_stream(
        liquid_selected, spec, vessel_id="vessel", temperature_K=300.0, molar_rate_mol_s=0.02,
    )
    assert gas.state.mole_fractions != liquid.state.mole_fractions
    assert gas.state.molar_density_mol_m3 < liquid.state.molar_density_mol_m3
