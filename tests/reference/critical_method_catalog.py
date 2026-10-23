"""D4 evidence index, not a second production method or source registry."""

import importlib
import inspect
import json
from pathlib import Path

from tests.reference.run_virial_source_block import catalogue

from chemworld.physchem.volume_kernels import VOLUME_METHODS
from chemworld.physchem.volume_mixtures import MIXTURE_VOLUME_METHODS


def entry(api, evidence, **details):
    return {"api": api, "evidence": evidence, **details}


def method_catalog():
    p = "chemworld.physchem."
    db = "chemworld.data.constants.ConstantDatabase."
    critical = {}
    for key in ("Tc", "Pc", "Vc", "Zc"):
        critical[key] = entry(
            db + "value",
            ["B3/all_sources", "D4/constant"],
            field=key,
            semantics="Explicit source selection; Zc source value remains distinct from PcVc/(RTc)",
        )
        critical[key + "_methods"] = entry(
            db + "methods",
            ["B3/all_sources"],
            field=key,
            semantics="Explicit fields; derived Zc candidates use intersection of Tc/Pc/Vc methods",
        )
    for name in ("Ihmels", "Meissner", "Grigoras", "critical_surface", "third_property"):
        critical[name] = entry(
            p + "critical_properties.estimate_critical_property",
            ["D1/core:surface", "D4/derived"],
            selectors=([name] if name in {"Ihmels", "Meissner", "Grigoras"}
                       else ["Ihmels", "Meissner", "Grigoras"]),
            semantics="Explicit pair, output and surface; no automatic source rank",
        )
    critical["critical_surface_methods"] = entry(
        p + "critical_kernels.critical_surface",
        ["D1/core:surface"],
        selectors=["Ihmels", "Meissner", "Grigoras"],
        semantics="Native explicit selector contract, supported for any declared missing Tc/Pc/Vc",
    )
    for name, native in (
        ("Hekayati_Raeissi", "hekayati_raeissi_critical"),
        ("Mersmann_Kind_predictor", "mersmann_kind_volume"),
        ("Tb_Tc_relationship", "boiling_critical_temperature"),
    ):
        critical[name] = entry(
            p + "critical_kernels." + native,
            ["D1/Hekayati" if name == "Hekayati_Raeissi" else "D1/core", "D4/derived"],
        )
    for name in (
        "Li",
        "Chueh_Prausnitz_Tc",
        "Chueh_Prausnitz_Vc",
        "Grieves_Thodos",
        "modified_Wilson_Tc",
        "modified_Wilson_Vc",
    ):
        critical[name] = entry(
            p + "critical_kernels.mixture_critical_property",
            ["D1/core:mixture"],
            selector=name,
            semantics="Explicit ordered interactions; pseudocritical mixture estimate",
        )
    acentric = {
        "omega": entry(db + "value", ["B3/all_sources", "D4/constant"], field="omega"),
        "omega_methods": entry(db + "methods", ["B3/all_sources"], field="omega"),
        "LK_omega": entry(p + "critical_kernels.lee_kesler_acentric_factor", ["D1/core:acentric"]),
        "omega_definition": entry(
            p + "critical_kernels.acentric_factor_from_pressure", ["D1/core:acentric"]
        ),
        "Stiel_polar_factor": entry(
            p + "critical_kernels.stiel_polar_factor", ["D1/core:acentric"]
        ),
    }
    volume = {
        name: entry(
            p + "volume_kernels.volume_value",
            [
                "D2B/mixture_compressed"
                if spec.uses_pressure and name != "ideal_gas"
                else "D2A/pure",
                "D4/pure",
            ],
            selector=name,
            phase=spec.phase,
            raw_quantity=spec.native_output,
        )
        for name, spec in VOLUME_METHODS.items()
    }
    volume.update(
        {
            name: entry(
                p + "volume_mixtures.mixture_volume_evaluation",
                ["D2B/mixture_compressed", "D4/mixture"],
                selector=name,
                semantics="Current MixtureSpec T/P/x; transport enthalpy is separately declared",
            )
            for name in MIXTURE_VOLUME_METHODS
        }
    )
    virial = {}
    for row in catalogue():
        native = row["native"]
        assert native is not None, row
        function = native.split("(")[0].split(".")[0].split(";")[0]
        if " / " in function:
            function = function.split(" / ")[0]
        virial[row["public"]] = entry(
            p + function,
            [row["evidence"] + "/" + row["block"], "D3C/source_gas"],
            semantic_mapping=native,
            semantics=row["semantics"],
        )
    return {"acentric": acentric, "critical": critical, "volume": volume, "virial": virial}


def verify_catalog():
    catalog = method_catalog()
    for module, mapping in catalog.items():
        upstream = importlib.import_module("chemicals." + module)
        names = {name for name in upstream.__all__ if inspect.isfunction(getattr(upstream, name))}
        assert names == set(mapping), (module, names - set(mapping), set(mapping) - names)
        for row in mapping.values():
            parts = row["api"].split(".")
            for end in range(len(parts) - 1, 1, -1):
                try:
                    target = importlib.import_module(".".join(parts[:end]))
                except ModuleNotFoundError:
                    continue
                for part in parts[end:]:
                    target = getattr(target, part)
                assert callable(target), row
                break
            else:
                raise AssertionError(row)
    assert [len(v) for v in catalog.values()] == [5, 23, 22, 56]
    return catalog


def reused_evidence():
    """Read actual attempts, including failed history; never infer from a filename."""
    root = Path(__file__).resolve().parents[2] / "workstreams/world_foundation"
    reports = {
        key: json.loads((root / f"NATIVE_CONV22{key}_RESULTS.json").read_text(encoding="utf8"))
        for key in ("D1", "D2A", "D2B", "D2C", "D3A", "D3B", "D3C")
    }
    rows = []

    def add(key, path, result, total):
        actual = result.get("units", result.get("total", result.get("passed")))
        assert actual == total and result["passed"] == total
        assert not result.get("failed", 0) and not result.get("failed_units", [])
        rows.append(
            {
                "artifact": f"NATIVE_CONV22{key}_RESULTS.json",
                "selector": path,
                "total": total,
                "passed": total,
                "comparisons": result.get("comparisons"),
                "scope": "Reused mathematical/source evidence; failures retained in artifact",
            }
        )

    for i, total in ((3, 119), (5, 125)):
        add("D1", f"scientific_attempts/{i}", reports["D1"]["scientific_attempts"][i], total)
    for i, total in ((2, 221), (3, 18)):
        add("D2A", f"scientific_rounds/{i}", reports["D2A"]["scientific_rounds"][i], total)
    add("D2B", "scientific_attempts/0", reports["D2B"]["scientific_attempts"][0], 160)
    add("D2C", "source_block_attempts/2", reports["D2C"]["source_block_attempts"][2], 3640)
    for i, total in ((1, 321), (2, 22)):
        add("D3A", f"science_attempts/{i}", reports["D3A"]["science_attempts"][i], total)
    add("D3B", "attempts/1/result", reports["D3B"]["attempts"][1]["result"], 92)
    add("D3C", "science_attempts/2/result", reports["D3C"]["science_attempts"][2]["result"], 447)
    return rows


if __name__ == "__main__":
    print(
        json.dumps(
            {
                "public_functions": {k: len(v) for k, v in verify_catalog().items()},
                "reused_evidence": reused_evidence(),
            },
            indent=2,
        )
    )
