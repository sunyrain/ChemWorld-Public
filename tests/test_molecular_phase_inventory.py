"""Actual PhaseLedger material ownership, calorics, streams and recovery."""

import json
from dataclasses import dataclass, replace

import pytest

from chemworld.foundation.solvents import SolventInventory
from chemworld.foundation.state_ledgers import PhaseLedger, PhaseRecord
from chemworld.foundation.state_serialization import _phases
from chemworld.physchem.cpa import CPAComponent, CPAModel
from chemworld.physchem.helmholtz import HelmholtzFluid
from chemworld.physchem.helmholtz_terms import HelmholtzTerm
from chemworld.physchem.molecular_flash import MolecularDensityBasin
from chemworld.physchem.molecular_mixture import MolecularCaloricModel, MolecularStream
from chemworld.runtime.molecular_phase_inventory import (
    MolecularPhaseEquilibriumSpec,
    MolecularPhaseTarget,
    equilibrate_molecular_phase_inventory,
    molecular_phase_stream,
)


def specification(caloric=True):
    R = 8.31446261815324
    components = tuple(CPAComponent(name, .45724*R*R*tc*tc/pc, .07780*R*tc/pc, tc,
        .37464+1.54226*omega-.26992*omega*omega, "declared PR test parameters")
        for name, tc, pc, omega in (("propane", 369.83, 4.248e6, .152),
                                    ("butane", 425.12, 3.796e6, .2)))
    residual = CPAModel(components, ((0, 0), (0, 0)), (200, 500), "explicit PR",
                         cubic_model="peng_robinson")
    refs = tuple(HelmholtzFluid(c.component_id, "", .044+.014*i, R, 300, 1000,
        200, 500, 1e9, (HelmholtzTerm("IdealGasHelmholtzLead", {"a1": 1+i, "a2": 2+i}),
                       HelmholtzTerm("IdealGasHelmholtzLogTau", {"a": 5+i})), (),
        "explicit ideal caloric test reference") for i, c in enumerate(components))
    return MolecularPhaseEquilibriumSpec(residual,
        (MolecularDensityBasin("vapor", (1, 2000), 250),
         MolecularDensityBasin("liquid", (7500, 13500), 10000)),
        (MolecularPhaseTarget("gas", "vapor", "gas", selected=True),
         MolecularPhaseTarget("condensate", "liquid", "liquid")),
        MolecularCaloricModel(residual, refs) if caloric else None)


def ledger():
    return PhaseLedger({
        "feed": PhaseRecord("feed", "vessel", "liquid", 1, {"propane": .5, "butane": .5}),
        "solid": PhaseRecord("solid", "vessel", "solid", .01, {"inert": .03},
                             metadata={"untouched": [1, 2]}),
        "external": PhaseRecord("external", "other-vessel", "liquid", 2, {"propane": 3})})


def close(stock, spec, **kwargs):
    return equilibrate_molecular_phase_inventory(stock, spec, vessel_id="vessel",
        source_phase_ids=("feed",), temperature_K=300, pressure_Pa=6e5, **kwargs)


def test_real_ledger_closure_calorics_and_json_recovery():
    original, spec = ledger(), specification()
    restored = MolecularPhaseEquilibriumSpec.from_dict(json.loads(json.dumps(spec.to_dict())))
    assert restored == spec
    result = close(original, restored)
    assert result.success, result.to_dict()
    assert result.ledger.total_amounts_mol() == pytest.approx(original.total_amounts_mol())
    assert result.ledger.phases["solid"] == original.phases["solid"]
    assert result.ledger.phases["external"] == original.phases["external"]
    assert "feed" in original.phases and "feed" not in result.ledger.phases
    assert set(result.installed_phase_ids) == {"gas", "condensate"}
    assert all(result.ledger.phases[k].vessel_id == "vessel" for k in result.installed_phase_ids)
    assert result.total_enthalpy_J == pytest.approx(sum(result.phase_enthalpies_J.values()))
    decoded = _phases(json.loads(json.dumps(result.ledger.to_dict())))
    assert decoded == result.ledger
    rerun = equilibrate_molecular_phase_inventory(decoded, restored, vessel_id="vessel",
        source_phase_ids=result.installed_phase_ids, temperature_K=300, pressure_Pa=6e5)
    assert rerun.success and rerun.total_enthalpy_J == pytest.approx(result.total_enthalpy_J)
    json.dumps(result.to_dict(), allow_nan=False)


def test_selected_stream_uses_current_inventory_and_original_port_contract():
    spec = specification()
    result = close(ledger(), spec)
    stream = molecular_phase_stream(result.ledger, spec, vessel_id="vessel", temperature_K=300,
                                     molar_rate_mol_s=.02)
    assert isinstance(stream, MolecularStream)
    gas = result.ledger.phases["gas"]
    assert stream.state.enthalpy_J_mol*sum(gas.species_amounts_mol.values()) == pytest.approx(
        result.phase_enthalpies_J["gas"])
    contribution = stream.contribution({"propane": "port_c3", "butane": "port_c4"})
    assert sum(contribution.port_rates_mol_s.values()) == pytest.approx(.02)
    assert stream.enthalpy_rate_W == pytest.approx(.02*stream.state.enthalpy_J_mol)
    changed = replace(gas, species_amounts_mol={"propane": gas.species_amounts_mol["propane"]*.8,
                                               "butane": gas.species_amounts_mol["butane"]*1.2})
    fresh = molecular_phase_stream(PhaseLedger({**result.ledger.phases, "gas": changed}), spec,
        vessel_id="vessel", temperature_K=300, molar_rate_mol_s=.02)
    assert fresh.state.mole_fractions[0] == pytest.approx(changed.species_amounts_mol["propane"]
                                                       /sum(changed.species_amounts_mol.values()))
    assert fresh.state.mole_fractions != stream.state.mole_fractions
    with pytest.raises(ValueError, match="ownership"):
        molecular_phase_stream(result.ledger, spec, vessel_id="wrong", temperature_K=300,
                                 phase_id="gas", molar_rate_mol_s=.02)
    with pytest.raises(ValueError, match="basin"):
        molecular_phase_stream(PhaseLedger({**result.ledger.phases,
            "gas": replace(gas, volume_L=1e6)}), spec, vessel_id="vessel", temperature_K=300,
            phase_id="gas", molar_rate_mol_s=.02)


@dataclass(frozen=True)
class ExcessModel:
    component_ids: tuple[str, ...] = ("a", "b")
    gas_constant_J_mol_K: float = 8.31446261815324

    def alphar(self, temperature, density, fractions):
        return 3*fractions[0]*fractions[1]


def test_same_basin_lle_has_distinct_real_phase_ids_and_no_manufactured_energy():
    spec = MolecularPhaseEquilibriumSpec(ExcessModel(),
        (MolecularDensityBasin("liquid", (1, 100), 40),),
        (MolecularPhaseTarget("liquid_a", "liquid", "liquid"),
         MolecularPhaseTarget("liquid_b", "liquid", "liquid", selected=True)))
    original = PhaseLedger({"feed": PhaseRecord("feed", "vessel", "liquid", 1, {"a": .3,"b": .7})})
    result = equilibrate_molecular_phase_inventory(original, spec, vessel_id="vessel",
        source_phase_ids=("feed",), temperature_K=300, pressure_Pa=1e5)
    assert result.success and set(result.ledger.phases) == {"liquid_a", "liquid_b"}
    assert result.phase_enthalpies_J is None and result.total_enthalpy_J is None
    assert result.ledger.total_amounts_mol() == pytest.approx(original.total_amounts_mol())
    with pytest.raises(ValueError, match="caloric"):
        molecular_phase_stream(result.ledger, spec, vessel_id="vessel", temperature_K=300,
                                 molar_rate_mol_s=.02)
    failed = equilibrate_molecular_phase_inventory(original,
        replace(spec, phase_targets=spec.phase_targets[:1]), vessel_id="vessel",
        source_phase_ids=("feed",), temperature_K=300, pressure_Pa=1e5)
    assert not failed.success and failed.ledger is original
    assert failed.failure_reason == "insufficient_distinct_phase_targets_for_basin"


@pytest.mark.parametrize("trace", [0., 1e-15])
def test_trace_material_retained_through_native_phase_ledger(trace):
    spec = specification(False)
    spec = replace(spec, basins=(MolecularDensityBasin("vapor", (1, 100), 10),),
                   phase_targets=spec.phase_targets[:1])
    original = PhaseLedger({"feed": PhaseRecord("feed", "vessel", "gas", 1,
                                              {"propane": 1, "butane": trace})})
    result = equilibrate_molecular_phase_inventory(original, spec, vessel_id="vessel",
        source_phase_ids=("feed",), temperature_K=450, pressure_Pa=1e4)
    assert result.success, result.to_dict()
    assert result.ledger.total_amounts_mol()["butane"] == trace


def test_solver_and_caloric_failure_preserve_original_ledger():
    original, spec = ledger(), specification()
    before = original.to_dict()
    failed = close(original, spec, max_iterations=1)
    assert not failed.success and failed.ledger is original and original.to_dict() == before
    limited = MolecularCaloricModel(spec.residual_model,
        tuple(replace(r, maximum_temperature_K=280) for r in spec.caloric_model.ideal_references))
    failed = close(original, replace(spec, caloric_model=limited))
    assert not failed.success and failed.ledger is original and original.to_dict() == before
    assert failed.failure_reason.startswith("caloric_closure_failed")


def test_ownership_collision_unknown_species_and_model_mismatch_rejected():
    original, spec = ledger(), specification()
    with pytest.raises(ValueError, match="ownership"):
        equilibrate_molecular_phase_inventory(original, spec, vessel_id="vessel",
            source_phase_ids=("feed", "external"), temperature_K=300, pressure_Pa=6e5)
    with pytest.raises(ValueError, match="overwrite"):
        close(original, replace(spec, phase_targets=(
            replace(spec.phase_targets[0], phase_id="solid"), spec.phase_targets[1])))
    with pytest.raises(ValueError, match="undeclared"):
        close(PhaseLedger({"feed": replace(original.phases["feed"], species_amounts_mol={"x": 1})}),
              spec)
    with pytest.raises(ValueError, match="legacy solvent"):
        close(PhaseLedger({"feed": replace(original.phases["feed"],
                                            solvents=SolventInventory.pure(0, 1))}), spec)
    changed = replace(spec.residual_model, interaction=((0, .01), (.01, 0)))
    with pytest.raises(ValueError, match="same residual"):
        replace(spec, residual_model=changed)
