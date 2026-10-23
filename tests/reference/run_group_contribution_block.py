"""Fixed whole-table 23D group estimates and original property consumers."""

import argparse
import json
import math
import time
from pathlib import Path

import mpmath as mp
import thermo.group_contribution.bondi as bondi
import thermo.group_contribution.fedors as fedors
import thermo.group_contribution.wilson_jasperson as wj
from run_activity_thermodynamics_block import compare, new_record
from thermo.group_contribution import JOBACK_GROUPS, Joback

from chemworld.data.scientific_inputs import InputValue
from chemworld.physchem.critical_properties import eos_component_from_critical_inputs
from chemworld.physchem.data_conflicts import DataSourceProvenance
from chemworld.physchem.group_contribution import (
    JOBACK_FIELDS,
    GroupContributionTable,
    GroupEstimateReport,
    bondi_group_properties,
    fedors_critical_volume,
    joback_estimate,
    joback_group_values,
    joback_property_correlations,
    wilson_jasperson_critical,
)
from chemworld.physchem.property_equations import evaluate_correlation
from chemworld.physchem.specs import ComponentSpec


def tables():
    rows = {
        "Joback": {
            str(k): {f: getattr(v, f) for f in JOBACK_FIELDS} for k, v in JOBACK_GROUPS.items()
        },
        "Fedors": {k: {"Vc": v} for k, v in fedors.fedors_contributions.items()},
        "Wilson-Jasperson": {
            **{
                k: {"Tc": v, "Pc": wj.Wilson_Jasperson_Pc_increments[k]}
                for k, v in wj.Wilson_Jasperson_Tc_increments.items()
            },
            **{
                "group:" + k: {"Tc": v, "Pc": wj.Wilson_Jasperson_Pc_groups[k]}
                for k, v in wj.Wilson_Jasperson_Tc_groups.items()
            },
        },
        "Bondi": {str(k): {"Vw": v.Vw, "Aw": v.Aw} for k, v in bondi.BONDI_GROUPS.items()},
    }
    return {
        name: GroupContributionTable(
            name,
            values,
            DataSourceProvenance(
                name,
                0,
                "thermo0.6 public " + name + " coefficient data",
                version="0.6.0",
                license="upstream source attribution retained",
            ),
        )
        for name, values in rows.items()
    }


def nullable_compare(rec, actual, expected):
    for key, target in expected.items():
        value = actual[key]
        compare(
            rec,
            {key + "/available": float(value is not None)},
            {key + "/available": float(target is not None)},
        )
        if value is not None and target is not None:
            compare(rec, {key: value}, {key: target})


def high_wj(atoms, groups, ring_count, tb, second_order):
    with mp.workdps(80):

        def total(table, counts):
            values = [table.get(k) for k in counts]
            return (
                None
                if any(v is None for v in values)
                else mp.fsum(
                    mp.mpf(str(v)) * n for v, n in zip(values, counts.values(), strict=True)
                )
            )

        t = total(wj.Wilson_Jasperson_Tc_increments, atoms)
        p = total(wj.Wilson_Jasperson_Pc_increments, atoms)
        if second_order:
            if t is not None:
                t += total(wj.Wilson_Jasperson_Tc_groups, groups)
            if p is not None:
                p += total(wj.Wilson_Jasperson_Pc_groups, groups)
        denominator = None if t is None else mp.mpf(".048271") - mp.mpf(".019846") * ring_count + t
        tc = (
            None
            if denominator is None or denominator <= 0
            else mp.mpf(str(tb)) / denominator ** mp.mpf(".2")
        )
        denominator = (
            None
            if p is None
            else mp.exp(-mp.mpf(".00922295") - mp.mpf(".0290403") * ring_count + mp.mpf(".041") * p)
            - mp.mpf(".96601")
        )
        pc = (
            None
            if tc is None or denominator is None or denominator <= 0
            else 100000 * mp.mpf(".0186233") * tc / denominator
        )
        return {"Tc": None if tc is None else float(tc), "Pc": None if pc is None else float(pc)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    data = tables()
    for name, table in tuple(data.items()):
        path = args.output / (name + ".json")
        path.write_text(json.dumps(table.to_dict(), indent=2), encoding="utf-8")
        data[name] = GroupContributionTable.from_dict(json.loads(path.read_text()))
    result = {
        "block": "23D native group property estimates",
        "mode": "development",
        "expected_units": 485,
        "units": [],
        "source_rows": {name: len(table.coefficients) for name, table in data.items()},
        "missing_increments": {
            name: sum(v is None for row in table.coefficients.values() for v in row.values())
            for name, table in data.items()
        },
    }

    def unit(kind, method, **kwargs):
        rec = new_record(kind, method, 1, **kwargs)
        result["units"].append(rec)
        return rec

    for key in JOBACK_GROUPS:
        for count in (1, 2, 5):
            for tb in (None, 400.0):
                rec = unit("coefficient_row", "Joback", group=key, count=count, Tb_input=tb)
                try:
                    actual = joback_group_values(
                        {str(key): count}, data["Joback"], atom_count=40, boiling_temperature_K=tb
                    )
                    counts = {key: count}
                    expected = {
                        name: getattr(Joback, name)(counts)
                        for name in ("Tb", "Tm", "Vc", "Hf", "Gf", "Hfus", "Hvap")
                    }
                    expected.update(
                        Tc=Joback.Tc(counts, Tb=tb),
                        Pc=Joback.Pc(counts, 40),
                        Cpig_coefficients=Joback.Cpig_coeffs(counts),
                        viscosity_coefficients=Joback.mul_coeffs(counts),
                    )
                    # A negative critical prediction is outside the model input domain.
                    if expected["Tc"] is not None and expected["Tc"] <= 0:
                        rec["reference_out_of_domain"] = {"Tc": expected["Tc"]}
                        expected["Tc"] = None
                    nullable_compare(rec, actual, expected)
                except Exception as exc:
                    rec["failures"].append({"exception": str(exc)})
    print("Joback source equations 246/485 complete", flush=True)
    for key, increment in fedors.fedors_contributions.items():
        rec = unit("coefficient_row", "Fedors", group=key)
        try:
            atoms = {"C": 6, "H": 14}
            options = {}
            expected_increment = mp.mpf(str(increment))
            if key in {"C", "H", "O", "N", "F", "Cl", "Br", "I", "S"}:
                atoms = {key: 1}
                expected = 1e-6 * float(mp.mpf("26.6") + expected_increment)
            else:
                if key == "O_alcohol":
                    atoms["O"] = 1
                    options["alcohol_oxygens"] = 1
                elif key == "N_amine":
                    atoms["N"] = 1
                    options["amine_nitrogens"] = 1
                elif key.endswith("_ring"):
                    options["rings_by_size"] = {int(key[0]): 1}
                else:
                    options[
                        {
                            "double_bond": "double_bonds",
                            "triple_bond": "triple_bonds",
                            "ring_ring_bonds": "ring_ring_bonds",
                        }[key]
                    ] = 1
                with mp.workdps(80):
                    expected = float(
                        (
                            mp.mpf("26.6")
                            + 6 * mp.mpf("34.426")
                            + 14 * mp.mpf("9.172")
                            + expected_increment
                        )
                        * mp.mpf("1e-6")
                    )
            actual = fedors_critical_volume(atoms, data["Fedors"], **options)
            compare(rec, actual, expected)
        except Exception as exc:
            rec["failures"].append({"exception": str(exc)})
    rec = unit("public_example", "Fedors", molecule="sec-butanol")
    compare(
        rec,
        fedors_critical_volume({"C": 4, "H": 10, "O": 1}, data["Fedors"], alcohol_oxygens=1),
        0.000274024,
    )
    for element in wj.Wilson_Jasperson_Tc_increments:
        for second in (False, True):
            rec = unit("element", "Wilson-Jasperson", element=element, second_order=second)
            try:
                atoms = {element: 1}
                groups = {"OH_large": 1}
                actual = wilson_jasperson_critical(
                    atoms,
                    data["Wilson-Jasperson"],
                    boiling_temperature_K=400.0,
                    functional_groups=groups,
                    second_order=second,
                )
                nullable_compare(rec, actual, high_wj(atoms, groups, 0, 400.0, second))
            except Exception as exc:
                rec["failures"].append({"exception": str(exc)})
    for group in wj.Wilson_Jasperson_Tc_groups:
        for count in (1, 2):
            rec = unit("functional_group", "Wilson-Jasperson", group=group, count=count)
            try:
                atoms = {"C": 6, "H": 14, "O": 2}
                groups = {group: count}
                actual = wilson_jasperson_critical(
                    atoms,
                    data["Wilson-Jasperson"],
                    boiling_temperature_K=400.0,
                    functional_groups=groups,
                )
                nullable_compare(rec, actual, high_wj(atoms, groups, 0, 400.0, True))
            except Exception as exc:
                rec["failures"].append({"exception": str(exc)})
    for second in (False, True):
        rec = unit(
            "public_example", "Wilson-Jasperson", second_order=second, molecule="2-ethylphenol"
        )
        atoms = {"C": 8, "H": 10, "O": 1}
        groups = {"OH_large": 1}
        actual = wilson_jasperson_critical(
            atoms,
            data["Wilson-Jasperson"],
            boiling_temperature_K=477.67,
            ring_count=1,
            functional_groups=groups,
            second_order=second,
        )
        nullable_compare(rec, actual, high_wj(atoms, groups, 1, 477.67, second))
        rec["public_rounded_values"] = {
            "Tc": 693.567 if second else 702.883,
            "Pc": 3743819.6667 if second else 3794106.49,
        }
    print("Critical source equations 379/485 complete", flush=True)
    for key, group in bondi.BONDI_GROUPS.items():
        for count in (1, 2):
            rec = unit("coefficient_row", "Bondi", group=key, count=count)
            try:
                actual = bondi_group_properties({str(key): count}, data["Bondi"])
                with mp.workdps(80):
                    volume = mp.mpf(str(group.Vw)) * count
                    area = None if group.Aw is None else mp.mpf(str(group.Aw)) * count
                    expected = {
                        "V_vdw": float(volume * mp.mpf("1e-6")),
                        "A_vdw": None if area is None else float(area * 100000),
                        "R": float(volume / mp.mpf("15.17")),
                        "Q": None if area is None else float(area / mp.mpf("2.5")),
                    }
                nullable_compare(rec, actual, expected)
                if area is None:
                    rec["reference_partial_area_policy"] = (
                        "original Bondi helper silently omits this missing area; "
                        "native A/Q stay unavailable"
                    )
            except Exception as exc:
                rec["failures"].append({"exception": str(exc)})
    rec = unit("public_example", "Bondi", molecule="decane")
    nullable_compare(
        rec,
        bondi_group_properties({"4": 2, "3": 8}, data["Bondi"]),
        {"V_vdw": 0.00010918, "A_vdw": 1504000.0, "R": 109.18 / 15.17, "Q": 15.04 / 2.5},
    )
    rec = unit("native_consumer", "Joback", molecule="acetone")
    try:
        report = joback_estimate(
            "acetone",
            {"1": 2, "24": 1},
            data["Joback"],
            atom_count=10,
            molecular_weight_g_mol=58.041864812,
        )
        restored = GroupEstimateReport.from_dict(json.loads(json.dumps(report.to_dict())))
        compare(
            rec, {"roundtrip": float(restored.to_dict() == report.to_dict())}, {"roundtrip": 1.0}
        )
        cp, mu = joback_property_correlations(report, temperature_range_K=(298, 500))
        coefficients = Joback.Cpig_coeffs({1: 2, 24: 1})
        ab = Joback.mul_coeffs({1: 2, 24: 1})
        for T in (300.0, 350.0, 400.0):
            compare(
                rec,
                {
                    "cp": evaluate_correlation(cp, temperature_K=T).value,
                    "mu": evaluate_correlation(mu, temperature_K=T).value,
                },
                {
                    "cp": sum(v * T**i for i, v in enumerate(coefficients)),
                    "mu": 58.041864812 * math.exp(ab[0] / T + ab[1]),
                },
            )
        component = ComponentSpec("acetone", "C3H6O")
        eos = eos_component_from_critical_inputs(
            component,
            Tc=report.critical_estimate("Tc"),
            Pc=report.critical_estimate("Pc"),
            omega=InputValue(0.3, "dimensionless", None, "explicit authored omega"),
        )
        compare(
            rec,
            {"Tc": eos.critical_temperature_K, "Pc": eos.critical_pressure_Pa},
            {"Tc": Joback.Tc({1: 2, 24: 1}), "Pc": Joback.Pc({1: 2, 24: 1}, 10)},
        )
    except Exception as exc:
        rec["failures"].append({"exception": str(exc)})
    result["comparisons"] = sum(r["comparisons"] for r in result["units"])
    result["failures"] = [
        {"unit": i, "model": r["model"], **f}
        for i, r in enumerate(result["units"])
        for f in r["failures"]
    ]
    result["elapsed_seconds"] = time.monotonic() - start
    result["passed"] = not result["failures"] and len(result["units"]) == 485
    (args.output / "summary.json").write_text(
        json.dumps(result, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(
        json.dumps({k: result[k] for k in ("comparisons", "passed", "elapsed_seconds")}), flush=True
    )
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
