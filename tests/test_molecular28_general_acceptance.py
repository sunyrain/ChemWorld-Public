"""Qualification accounting and independent common-tangent controls for parent28."""

import ast
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from reference import molecular28_general_reference as reference
from reference import run_molecular28_general as runner


def test_reference_does_not_import_native_physics():
    tree = ast.parse(Path(reference.__file__).read_text(encoding="utf-8"))
    imports = [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    imports += [
        name.name for node in ast.walk(tree) if isinstance(node, ast.Import) for name in node.names
    ]
    assert not any(name and name.startswith("chemworld") for name in imports)


@pytest.mark.parametrize(
    "z,count",
    [((1 / 3,) * 3, 3), ((0.49, 0.49, 0.02), 2), ((0.98, 0.01, 0.01), 1), ((0.5, 0.5, 0), 2)],
)
def test_analytic_reference_tangents_and_material(z, count):
    result = reference.regular_state(z, 300.0)
    amounts = np.array(result["amounts"])
    assert len(amounts) == count
    assert amounts.sum(axis=0) == pytest.approx(z, abs=2e-14)
    fractions = amounts / amounts.sum(axis=1)[:, None]
    active = np.array(z) > 0
    a = 9000 / (reference.R * 300)
    # Independent chemical-potential derivative of the authored molar Gibbs.
    potentials = [
        np.log(x[active])
        + a * (1 - x[active] - sum(x[i] * x[j] for i in range(3) for j in range(i + 1, 3)))
        for x in fractions
    ]
    assert max(abs(potentials - potentials[0]).ravel()) < 1e-10


def test_fixed_accounting_preserves_all_failed_units_and_refuses_overwrite(tmp_path, monkeypatch):
    matrix = runner.cases()
    assert len(matrix) == len({c["id"] for c in matrix}) == 67
    assert sum(runner.planned_comparisons(c) for c in matrix) == 1296
    assert sum(c.get("family") == "vlle" for c in matrix) == 12
    assert {c["coordinate"] for c in matrix if c.get("family") == "vle"} == {
        "TP",
        "PH",
        "PS",
        "UV",
        "TV",
        "Pmass_density",
        "Ph_mass",
        "Ps_mass",
    }

    def fail(case, context):
        raise RuntimeError("retained qualification failure " + case["id"])

    monkeypatch.setattr(runner, "execute", fail)
    path = tmp_path / "failed"
    summary = runner.run(path)
    assert summary["attempted"] == summary["failed"] == len(summary["failures"]) == 67
    assert not summary["passed_block"]
    records = [json.loads(line) for line in (path / "records.jsonl").read_text().splitlines()]
    assert [r["case"]["id"] for r in records] == [c["id"] for c in matrix]
    assert all("RuntimeError" in r["traceback"] for r in records)
    with pytest.raises(FileExistsError):
        runner.run(path)


def test_positive_trace_loss_cannot_hide_in_absolute_material_tolerance():
    feed = {"a": 0.5 - 5e-16, "b": 0.5 - 5e-16, "c": 1e-15}
    ref = reference.regular_state(tuple(feed.values()), 300.0)
    phases = []
    for amounts, rho in zip(ref["amounts"], ref["densities"], strict=True):
        actual = dict(zip(feed, amounts, strict=True))
        actual["c"] = 0.0
        phases.append(
            SimpleNamespace(
                component_amounts_mol=actual, state=SimpleNamespace(molar_density_mol_m3=rho)
            )
        )
    native = SimpleNamespace(
        success=True,
        phases=phases,
        temperature_K=ref["T"],
        pressure_Pa=ref["P"],
        enthalpy_J=ref["H"],
        entropy_J_K=ref["S"],
        internal_energy_J=ref["U"],
        volume_m3=ref["V"],
        material_balance_error_mol=1e-15,
        chemical_potential_error_RT=0.0,
        pressure_relative_error=0.0,
    )
    comparisons = runner.state_comparisons(native, ref, feed)
    failed = {row["name"] for row in comparisons if not row["passed"]}
    assert failed == {"trace_conservation_c", "trace_positive_all_phases_c"}


@pytest.mark.parametrize("pressure", [5e6, 6e6])
def test_source_dew_precision_is_tighter_than_comparison_contract(pressure):
    source = reference.thermo_boundary((0.8, 0.2), "dew", family="envelope", P=pressure)
    assert source["source_chemical_potential_error_RT"] <= 3e-7
    assert source["rhoL"] > 1.5 * source["rhoV"]
