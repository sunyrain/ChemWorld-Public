"""Native quadruplet physics and shared Gibbs/material contracts."""

import json

import numpy as np
import pytest

from chemworld.physchem.calphad_material import CefMaterialSpec
from chemworld.physchem.mqmqa import MQMQAPhaseModel
from chemworld.runtime import ChemWorldRuntime
from chemworld.runtime.solid_solutions import solid_enthalpy
from chemworld.world.solid_world import calphad_world_spec
from reference.run_mqmqa_group_block import cases
from test_calphad_material import check_native_entrypoints
from test_chem_sage import MIXED_VALENCE, database


def model():
    return MQMQAPhaseModel(database(MIXED_VALENCE), "SLAG")


@pytest.mark.parametrize("family", ["SUBG", "SUBQ"])
@pytest.mark.parametrize("lattice", ["cation", "anion"])
def test_additional_principal_ion_is_outside_nu_and_gamma(family, lattice):
    # With equal six-quad fractions and group vector 111, Chi_i=Chi_j=1/3,
    # Y_m=1/3, and nu=gamma=empty. Eq.25's off-group multiplier is 1/27;
    # p=1,q=2 gives another 1/27 and the Eq.17 weight is 1/6.
    text = dict(cases())[f"{family}-{lattice}-G-111-1.dat"]
    phase = MQMQAPhaseModel(database(text), "MIX")
    sites = [dict.fromkeys(phase.constituents[0], 1/6)]
    atoms_per_quad, _ = phase._composition(sites)
    excess = phase.evaluate(sites, 1050).contributions_J_mol["excess"]
    assert excess*atoms_per_quad == pytest.approx(1000/(2*6*27*27), abs=1e-12)


def test_mixed_valence_real_atom_basis_charge_and_common_analytic_response():
    m = model()
    sites = [dict.fromkeys(m.constituents[0], 1 / 3)]
    total, x = m._composition(sites)
    assert total == pytest.approx((1+11/12+5/6)/3)
    assert set(x) == {"FE", "O"} and sum(x.values()) == pytest.approx(1)
    assert m._constraints(sites) == pytest.approx((0, 0))
    response = m.derivatives(sites, 1000.)
    assert response.hessian.shape == (5, 5, 5)
    assert np.all(np.isfinite(response.hessian))
    p = m.evaluate(sites, 1000.)
    assert set(p.contributions_J_mol) == {"reference", "ideal", "excess"}
    assert p.enthalpy_J_mol == pytest.approx(0, abs=1e-9)
    assert m.temperature_response(sites, 1000.).fixed_constitution_heat_capacity_J_mol_K == 0
    assert MQMQAPhaseModel.from_dict(json.loads(json.dumps(m.to_dict()))).to_dict() == m.to_dict()


def test_actual_quadruplet_material_uses_shared_energy_and_snapshot_without_ion_inventory():
    m = model()
    sites = (dict.fromkeys(m.constituents[0], 1 / 3),)
    x = m._composition(sites)[1]
    material = CefMaterialSpec(m, sites, 900, 1100, 1000, .01, phase_type="liquid")
    stock = {e: .001*v for e, v in x.items()}
    world = calphad_world_spec(material, element_amounts_mol=stock, temperature_K=1000.)
    runtime = ChemWorldRuntime(composition=world)
    runtime.reset(seed=40)
    initial_h = solid_enthalpy(runtime._state, material)
    restored = ChemWorldRuntime.from_snapshot(runtime.snapshot())
    result = restored.execute({"operation": "heat", "target_temperature_K": 1050,
                               "duration_s": 1})
    assert result.info["transaction_status"] == "committed"
    assert restored._state.species_amounts == stock
    assert abs(solid_enthalpy(restored._state, material)-initial_h
               -restored._state.ledger.energy_jacket_J) < 1e-7
    assert 1000 < restored._state.temperature_K < 1050
    assert CefMaterialSpec.from_dict(json.loads(json.dumps(material.to_dict()))).to_dict() == (
        material.to_dict())


@pytest.mark.parametrize("noise", ["sequential", "keyed"])
def test_mqmqa_public_gym_cli_agent_history_and_new_process_snapshot(tmp_path, capsys, noise):
    m = model()
    sites = (dict.fromkeys(m.constituents[0], 1/3),)
    material = CefMaterialSpec(m, sites, 500, 900, 600, .01, phase_type="liquid")
    stock = {e: .001*x for e, x in m._composition(sites)[1].items()}
    spec = calphad_world_spec(material, element_amounts_mol=stock, temperature_K=600)
    check_native_entrypoints(spec, tmp_path, capsys, noise)


def test_bad_pair_coordination_and_unimplemented_physics_are_rejected():
    with pytest.raises(ValueError, match="coordination"):
        MQMQAPhaseModel(database(MIXED_VALENCE.replace("4 4 4 4", "4 4 4 5")), "SLAG")
    with pytest.raises(ValueError, match="active ion topology"):
        MQMQAPhaseModel(database(MIXED_VALENCE), "SLAG", components=("FE",))


def test_nonzero_unconsumed_source_auxiliary_balance_is_retained_and_rejected():
    text = (MIXED_VALENCE.replace("2 2 0 3 0", "3 2 0 3 0")
            .replace("FE O\n", "FE O E(SLAG)\n")
            .replace("55.845 15.999\n", "55.845 15.999 0\n")
            .replace("1 1 1 1\n", "1 1 1 1 1\n")
            .replace("1 1 2 3\n", "1 1 2 3 -1\n"))
    db = database(text)
    assert db.phase_records["SLAG"].endmembers[0].stoichiometry[-1] == 1
    with pytest.raises(ValueError, match="source auxiliary"):
        MQMQAPhaseModel(db, "SLAG")


@pytest.mark.parametrize("replacement", ["type", "code", "unused_exponent", "wrong_lattice"])
def test_unconsumed_interaction_declarations_reject_the_whole_phase(replacement):
    record = "3 G 1 2 3 3 1 0 0 0\n"+"0 "*12+"\n0 0 1000\n0\n"
    changes = {"type": ("3 G", "5 G"), "code": ("3 G", "3 B"),
               "unused_exponent": ("1 0 0 0", "1 0 2 0"),
               "wrong_lattice": ("0 0 1000", "0 3 1000")}
    valid = MIXED_VALENCE[:-2]+record
    phase = MQMQAPhaseModel(database(valid), "SLAG")
    assert phase.evaluate([dict.fromkeys(phase.constituents[0], 1/3)], 1000
                         ).contributions_J_mol["excess"] > 0
    malformed = MIXED_VALENCE[:-2]+record.replace(*changes[replacement])
    with pytest.raises(ValueError, match=r"interaction type|mixing code|unconsumed"):
        MQMQAPhaseModel(database(malformed), "SLAG")


def test_pure_endmembers_and_mixed_cp_use_real_atoms_and_source_formula_basis():
    text = MIXED_VALENCE.replace("2000 0 0 0 0 0 0", "2000 -10000 0 -40 0 0 0", 1)
    text = text.replace("2000 0 0 0 0 0 0", "2000 -50000 0 -100 0 0 0", 1)
    m = MQMQAPhaseModel(database(text), "SLAG")
    first, _, last = m.constituents[0]
    for selected, expected_h in ((first, -5000), (last, -10000)):
        sites = [dict.fromkeys(m.constituents[0], 0.)]
        sites[0][selected] = 1.
        assert m.evaluate(sites, 1000).enthalpy_J_mol == pytest.approx(
            expected_h+20*1000, abs=1e-9)
        assert m.temperature_response(sites, 1000).fixed_constitution_heat_capacity_J_mol_K == (
            pytest.approx(20))
        with pytest.raises(ValueError, match="positive"):
            m.site_partials(sites, 1000)
    mixed = [dict.fromkeys(m.constituents[0], 1/3)]
    assert m.temperature_response(mixed, 1000).fixed_constitution_heat_capacity_J_mol_K == (
        pytest.approx(20))


def test_virtual_vacancy_formula_is_not_real_ionic_stock_and_active_subset_preserves_indices():
    text = """Synthetic virtual-anion basis
2 2 0 3 0
FE NI
55.845 58.69
6 1 2 3 4 5 6
1 1
LIQUID
SUBQ
2 3
FE
1 1 1 0
2000 0 0 0 0 0 0
1 1 0 0 0
2.4
NI
1 1 0 1
2000 0 0 0 0 0 0
1 1 0 0 0
2.4
2 1
FE NI
VA
2 2
1 2
1
1
1 2
1 1
1 1 3 3 6 6 3 3
1 2 3 3 6 6 3 3
2 2 3 3 6 6 3 3
0
"""
    db = database(text)
    assert db.species["VA"].constituents == (("VA", 1.),)
    m = MQMQAPhaseModel(db, "LIQUID")
    sites = [dict.fromkeys(m.constituents[0], 1/3)]
    total, x = m._composition(sites)
    assert total == pytest.approx(1/3)
    assert x == pytest.approx({"FE": .5, "NI": .5})
    assert m._constraints(sites) == pytest.approx((0, 0))
    submodel = MQMQAPhaseModel(db, "LIQUID", components=("FE", "VA"))
    assert submodel.constituents == (("Q1_1_3_3",),)
    assert submodel.gibbs([{"Q1_1_3_3": 1.}], 1000) == 0
