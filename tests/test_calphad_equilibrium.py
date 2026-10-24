"""Physical multicomponent equilibrium and explicit native stock boundaries."""

import importlib.util
import json
from dataclasses import replace
from pathlib import Path

import pytest

from chemworld.foundation.solvents import SolventInventory
from chemworld.foundation.state_ledgers import PhaseLedger, PhaseRecord
from chemworld.physchem.calphad_equilibrium import (
    CalphadEquilibriumSpec,
    calphad_tp_equilibrium,
)
from chemworld.runtime.calphad_phase_inventory import (
    CalphadPhaseTarget,
    equilibrate_calphad_phase_inventory,
)

_spec = importlib.util.spec_from_file_location(
    "calphad_cases", Path(__file__).parent / "reference/calphad_equilibrium_models.py")
_models = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_models)


@pytest.fixture(scope="module")
def binary():
    return _models.native_spec(_models.cases()[0])


def test_gap_keeps_two_constitutions_and_changes_amount_not_composition(binary):
    result = calphad_tp_equilibrium(binary, 600, 101325, {"FE": .3, "NI": .7})
    assert result.success, result.to_dict()
    assert len(result.phases) == 2
    assert result.phases[0].phase_name == result.phases[1].phase_name == "ALPHA"
    assert result.phases[0].element_fractions["FE"] < .2
    assert result.phases[1].element_fractions["FE"] > .8
    assert sum(p.element_amounts_mol["FE"] for p in result.phases) == pytest.approx(.3)
    scaled = calphad_tp_equilibrium(binary, 600, 101325, {"FE": 3., "NI": 7.})
    assert scaled.success
    assert scaled.total_gibbs_J == pytest.approx(10*result.total_gibbs_J)
    assert [p.fraction for p in scaled.phases] == pytest.approx([p.fraction for p in result.phases])


def test_three_components_actually_select_three_phases():
    case = _models.cases()[3]
    result = calphad_tp_equilibrium(_models.native_spec(case), case["T"], 101325, case["feed"])
    assert result.success, result.to_dict()
    assert {p.phase_name for p in result.phases} == {"FE_PREF", "NI_PREF", "CR_PREF"}
    assert len(result.phases) == 3
    assert all(p.fraction > 0 for p in result.phases)
    assert len(result.element_potentials_J_mol) == 3


def test_phase_disappearance_and_exact_zero_support(binary):
    hot = calphad_tp_equilibrium(binary, 1000, 101325, {"FE": .3, "NI": .7})
    assert hot.success and len(hot.phases) == 1
    pure = calphad_tp_equilibrium(binary, 600, 101325, {"FE": 2., "NI": 0.})
    assert pure.success, pure.to_dict()
    assert pure.phases[0].element_amounts_mol == {"FE": 2.}
    assert pure.phases[0].site_fractions == ({"FE": 1.},)


def test_specification_codec_and_bad_declaration(binary):
    payload = json.loads(json.dumps(binary.to_dict()))
    clone = CalphadEquilibriumSpec.from_dict(payload)
    assert clone.to_dict() == binary.to_dict()
    with pytest.raises(ValueError):
        CalphadEquilibriumSpec(binary.models*2)
    with pytest.raises(ValueError):
        CalphadEquilibriumSpec.from_dict({**payload, "invented": True})
    for feed in ({"FE": -1.}, {"FE": 0.}, {"FE": 1., "UNKNOWN": 0.}, {"FE": True}):
        with pytest.raises(ValueError):
            calphad_tp_equilibrium(binary, 600, 101325, feed)


def inventory(binary, *, feed=None, targets=None, policy=None, ledger=None, **kwargs):
    stock = PhaseLedger({"feed": PhaseRecord("feed", "vessel", "solid", .02,
        feed or {"FE": .3, "NI": .7}),
        "other": PhaseRecord("other", "another", "solid", .01, {"CR": 2.})})
    specification = replace(binary, policy=policy) if policy is not None else binary
    return equilibrate_calphad_phase_inventory(ledger or stock, specification,
        vessel_id="vessel", source_phase_ids=("feed",), temperature_K=600., pressure_Pa=101325.,
        phase_targets=targets or (CalphadPhaseTarget("left", "ALPHA", .007),
                                 CalphadPhaseTarget("right", "ALPHA", .009)), **kwargs)


def test_actual_inventory_is_single_owner_and_unrelated_stock_untouched(binary):
    result = inventory(binary)
    assert result.success, result.to_dict()
    assert set(result.ledger.phases) == {"left", "right", "other"}
    assert result.ledger.phases["other"].species_amounts_mol == {"CR": 2.}
    assert result.ledger.total_amounts_mol() == pytest.approx({"FE": .3, "NI": .7, "CR": 2.})
    assert result.ledger.phases["left"].volume_L == pytest.approx(
        result.equilibrium.phases[0].amount_mol_atoms*.007)
    changed = inventory(binary, feed={"FE": .6, "NI": .4})
    assert changed.success
    assert changed.ledger.total_amounts_mol()["FE"] == pytest.approx(.6)
    assert changed.equilibrium.phases[0].fraction != pytest.approx(
        result.equilibrium.phases[0].fraction)


def test_collision_ownership_and_insufficient_target_rollback(binary):
    with pytest.raises(ValueError, match="overwrite"):
        inventory(binary, targets=(CalphadPhaseTarget("other", "ALPHA", .01),))
    original = PhaseLedger({"feed": PhaseRecord(
        "feed", "wrong", "solid", .01, {"FE": .3, "NI": .7})})
    with pytest.raises(ValueError, match="ownership"):
        inventory(binary, ledger=original)
    with pytest.raises(ValueError, match="volume"):
        CalphadPhaseTarget("bad", "ALPHA", 0.)
    result = inventory(binary, targets=(CalphadPhaseTarget("one", "ALPHA", .01),))
    assert not result.success
    assert result.failure_reason == "insufficient_distinct_phase_targets_for_model"
    assert set(result.ledger.phases) == {"feed", "other"}


def test_failed_search_retains_original_ledger_and_diagnostics(binary):
    policy = replace(binary.policy, maximum_iterations=1)
    result = inventory(binary, policy=policy)
    assert not result.success
    assert set(result.ledger.phases) == {"feed", "other"}
    assert result.equilibrium.diagnostics["rounds"]


def test_stoichiometric_underdetermined_potentials_are_not_fabricated():
    case = _models.cases()[9]
    spec = _models.native_spec({**case, "phases": ["COMPOUND"]})
    result = calphad_tp_equilibrium(spec, 500, 101325, {"FE": .5, "NI": .5})
    assert not result.success
    assert result.failure_reason == "underdetermined_element_potentials"
    assert result.element_potentials_J_mol is None


def test_negative_tangent_discovers_unsampled_phase_and_honors_column_budget():
    tdb = "\n".join((_models.ELEMENTS, _models.solution("ALPHA", ("FE", "NI"), (0, 0)),
                     _models.solution("BETA", ("FE", "NI"), (1000, 1000)),
                     "PARAMETER L(BETA,FE,NI;0) 100 -8000; 2500 N !"))
    case = {"tdb": tdb, "phases": ["ALPHA", "BETA"]}
    specification = _models.native_spec(case, intervals=1)
    result = calphad_tp_equilibrium(specification, 700, 101325, {"FE": .5, "NI": .5})
    assert result.success, result.to_dict()
    assert [p.phase_name for p in result.phases] == ["BETA"]
    rounds = result.diagnostics["rounds"]
    assert len(rounds) > 1 and rounds[0]["negative_minima"] > 0
    assert rounds[1]["columns"] > rounds[0]["columns"]
    limited = replace(specification, policy=replace(specification.policy, maximum_column_rounds=1))
    failure = calphad_tp_equilibrium(limited, 700, 101325, {"FE": .5, "NI": .5})
    assert not failure.success and failure.failure_reason == "column_search_budget_exhausted"
    assert failure.phases == ()


def test_condensed_phase_scope_rejects_gas_and_vacancy():
    case = _models.cases()[0]
    with pytest.raises(ValueError, match="neutral"):
        _models.native_spec({**case, "tdb": case["tdb"].replace(
            "PHASE ALPHA %", "PHASE ALPHA:G %")})
    with pytest.raises(ValueError):
        CalphadPhaseTarget("gas", "ALPHA", .02, phase_type="gas")
    text = (_models.ELEMENTS + "ELEMENT VA VACUUM 0 0 0 !\n" +
            _models.solution("ALPHA", ("FE", "VA"), (0, 0)))
    with pytest.raises(ValueError, match="vacancy"):
        _models.native_spec({"tdb": text, "phases": ["ALPHA"]})


def test_exact_absent_element_excludes_incompatible_compound():
    specification = _models.native_spec(_models.cases()[9])
    result = calphad_tp_equilibrium(specification, 500, 101325, {"FE": 1., "NI": 0.})
    assert result.success
    assert [p.phase_name for p in result.phases] == ["FCC"]
    assert result.phases[0].element_amounts_mol == {"FE": 1.}
    assert result.diagnostics["inactive_phases"][0]["phase"] == "COMPOUND"


@pytest.mark.parametrize("amounts", ({"UNKNOWN": 1.}, {"FE": float("nan")}, {"FE": -.1}))
def test_invalid_actual_stock_is_not_silently_normalized(binary, amounts):
    ledger = PhaseLedger({"feed": PhaseRecord("feed", "vessel", "solid", .01, amounts)})
    with pytest.raises(ValueError, match="atomic stock"):
        inventory(binary, ledger=ledger)


def test_legacy_solvent_is_not_discarded(binary):
    ledger = PhaseLedger({"feed": PhaseRecord("feed", "vessel", "solid", .01,
        {"FE": .3, "NI": .7}, solvents=SolventInventory.pure(0, .01))})
    with pytest.raises(ValueError, match="legacy solvents"):
        inventory(binary, ledger=ledger)
