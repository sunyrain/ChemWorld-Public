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
