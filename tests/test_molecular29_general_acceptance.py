"""Independent profile-oracle semantics and failure accounting for parent29."""

import ast
import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("phasepy", reason="Optional independent reference dependency")

from reference import molecular29_source_reference as source
from reference import run_molecular29_general as runner
from reference.molecular29_cases import fixed_cases, native_influence, native_model


def test_source_expected_values_do_not_import_native_physics():
    tree = ast.parse(Path(source.__file__).read_text(encoding="utf-8"))
    imports = [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    imports += [
        name.name for node in ast.walk(tree) if isinstance(node, ast.Import) for name in node.names
    ]
    assert not any(name and name.startswith("chemworld") for name in imports)


def profile_fixture():
    z = np.linspace(-5, 5, 401)
    fraction = 0.5 * (1 + np.tanh(z))
    left, right = np.array([1.0, 2.0]), np.array([3.0, 5.0])
    rho = left + (right - left) * fraction[:, None]
    # A genuine second-component adsorption contribution is retained.
    rho[:, 1] += 0.2 / np.cosh(z) ** 2
    return {"z": z, "rho": rho, "left": left, "right": right}


def test_profile_alignment_accepts_translation_but_rejects_stretch_and_missing_adsorption():
    expected = profile_fixture()
    translated = expected | {"z": expected["z"] + 100}
    metrics = runner.compare_profiles(translated, expected)
    assert metrics["spatial_shape_relative"] < 1e-12
    stretched = expected | {"z": expected["z"] * 1.2}
    metrics = runner.compare_profiles(stretched, expected)
    assert metrics["width_relative"] == pytest.approx(0.2)
    linear = expected["rho"].copy()
    linear[:, 1] = 2 + 3 * (linear[:, 0] - 1) / 2
    metrics = runner.compare_profiles(expected | {"rho": linear}, expected)
    assert metrics["density_path_relative"] > 0.06


def test_multiple_interface_crossings_cannot_be_sorted_into_a_passing_profile():
    expected = profile_fixture()
    triple = np.vstack((expected["rho"], expected["rho"][::-1], expected["rho"]))
    with pytest.raises(ValueError, match=r"multiple|monotone"):
        runner.profile_coordinates(
            np.arange(len(triple)), triple, expected["left"], expected["right"]
        )


def test_same_eos_factory_and_declared_rank_are_explicit():
    model = native_model("ternary")
    assert model.component_ids == ("propane", "butane", "pentane")
    assert model.gas_constant_J_mol_K == 8.314
    influence = native_influence("ternary", 0)
    matrix = np.array(influence.matrix_at(320))
    assert influence.declared_rank == np.linalg.matrix_rank(matrix) == 1
    assert np.diag(matrix) == pytest.approx((1e-20, 1.5e-20, 2e-20))


def test_failed_groups_preserve_116_checks_and_create_only_outputs(tmp_path, monkeypatch):
    cases = fixed_cases()
    assert len(cases) == 10
    assert sum(1 if c["family"] == "pure" else 6 for c in cases) == 45
    source_data = {"groups": [{"case": case, "runs": []} for case in cases]}

    def fail(_group):
        raise RuntimeError("retained source/reference failure")

    monkeypatch.setattr(runner, "source_group_checks", fail)
    summary = runner.run_native(source_data, tmp_path / "failure")
    assert not summary["success"]
    assert summary["expected_checks"] == summary["evaluated_checks"] == 116
    assert summary["passed_checks"] == summary["completed_runs"] == 0
    assert all(
        "retained source/reference failure" in group["exception"] for group in summary["groups"]
    )
    saved = json.loads((tmp_path / "failure" / "SUMMARY.json").read_text())
    assert saved["expected_runs"] == 45
    with pytest.raises(FileExistsError):
        runner.run_native(source_data, tmp_path / "failure")


def test_missing_source_runs_do_not_pass_vacuous_comparisons():
    for case in fixed_cases():
        checks, _ = runner.source_group_checks({"case": case, "runs": []})
        assert not any(checks.values())
