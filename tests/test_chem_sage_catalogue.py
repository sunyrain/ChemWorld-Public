"""Catalogue bookkeeping contracts; synthetic reports are not scientific evidence."""

import hashlib
import importlib.util
from copy import deepcopy
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("catalogue_contract_driver",
    Path(__file__).parent/"reference/run_chem_sage_catalogue.py")
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)


def reports():
    hashes = {name: hashlib.sha256(name.encode()).hexdigest() for name in driver.FILES}
    catalogue = {"reference_version": "0.11.2", "files": [], "phases": []}
    cef, mq = {"units": []}, {"units": []}
    families = [family for family, count in driver.FAMILIES.items() for _ in range(count)]
    rejections = 0
    for i, family in enumerate(families):
        file, phase = driver.FILES[i % 7], f"PHASE_{i}"
        mqmqa = family in {"SUBG", "SUBQ"}
        rejected = mqmqa and rejections < 3
        issue = "pair_element_identity" if rejections < 2 else "quadruplet_charge"
        case = {"database": file, "phase": phase, "family": family, "scheduled_points": 6,
                "temperatures": [400., 700., 1000.],
                "expected": "source_refusal" if rejected else "numerical",
                "source_issues": [{"kind": issue}] if rejected else []}
        catalogue["phases"].append(case)
        row = {"database": file, "phase": phase, "passed": not rejected,
               "failures": [], "points": []}
        if rejected:
            message = ("ValueError: MQMQA pair source atoms disagree" if rejections < 2 else
                       "ValueError: MQMQA quadruplet coordination violates charge neutrality")
            row["failures"] = [message+"\n"]
            rejections += 1
        else:
            row["points"] = [{"T": t, "profile": p, "passed": True, "failures": [],
                              "values": {"H": 1., "response:G|temperature": 0.}}
                             for p in ("uniform", "tilted") for t in case["temperatures"]]
        (mq if mqmqa else cef)["units"].append(row)
    structure = {"files": []}
    for i, file in enumerate(driver.FILES):
        n = sum(c["database"] == file for c in catalogue["phases"])
        catalogue["files"].append({"name": file, "phases": n, "sha256": hashes[file]})
        structure["files"].append({"name": file, "phases": n, "sha256": hashes[file],
            "passed": True, "failures": [], "differences": [],
            "endmembers": 202 if i == 0 else 0, "parameters": 170 if i == 0 else 0})
    for report, count in ((cef, 93), (mq, 21)):
        report.update(expected_phases=count, expected_points=count*6,
            passed_points=sum(len(u["points"]) for u in report["units"]), failed_points=0,
            unexecuted_points=count*6-sum(len(u["points"]) for u in report["units"]))
    return catalogue, structure, cef, mq, hashes


def test_source_refusals_never_turn_into_original_numerical_passes():
    result = driver.assess(*reports())
    assert result["complete_catalogue_qualified"]
    assert result["valid_phases_qualified"] == 111
    assert result["source_refusals_confirmed"] == 3
    assert result["qualified_valid_points"] == 666
    assert result["unexecuted_original_points"] == 18
    assert result["all_original_phases_numerically_qualified"] is False
    assert result["all_original_sources_physics_consumable"] is False
    assert driver.quantity_unit("response:G|temperature|pressure") == (
        "J/mol_nonvacant_atoms/K/Pa")


@pytest.mark.parametrize("corruption", ["missing", "duplicate", "counter", "repeated_point",
                                       "wrong_refusal", "invalid_accepted", "changed_bytes",
                                       "reference_version"])
def test_incomplete_or_inconsistent_evidence_cannot_qualify(corruption):
    catalogue, structure, cef, mq, hashes = deepcopy(reports())
    if corruption == "missing":
        cef["units"].pop()
    elif corruption == "duplicate":
        cef["units"][-1] = deepcopy(cef["units"][0])
    elif corruption == "counter":
        mq["passed_points"] = 126
    elif corruption == "repeated_point":
        cef["units"][0]["points"][-1] = deepcopy(cef["units"][0]["points"][0])
    elif corruption == "wrong_refusal":
        mq["units"][0]["failures"] = ["RuntimeError: unrelated numerical failure"]
    elif corruption == "invalid_accepted":
        mq["units"][0].update(passed=True, failures=[], points=deepcopy(cef["units"][0]["points"]))
    elif corruption == "changed_bytes":
        catalogue["files"][0]["sha256"] = "different"
    else:
        catalogue["reference_version"] = "unverified"
    assert not driver.assess(catalogue, structure, cef, mq, hashes)["complete_catalogue_qualified"]
