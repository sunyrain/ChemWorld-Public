"""Fixed CONV23I phase consumer, VLE response and analytic azeotrope block."""

import argparse
import json
import math
import time
from pathlib import Path

import mpmath
import thermo
from activity_reference_cases import COMPOSITIONS, TEMPERATURES, high_precision_report
from activity_reference_cases import model_case as activity_case
from regular_solution_high_precision import expected as regular_reference
from run_activity_thermodynamics_block import new_record
from run_regular_solution_block import expected as public_object_reference
from run_regular_solution_block import model_case as regular_case
from run_unifac_block import case as unifac_case
from run_unifac_block import upstream as unifac_reference

from chemworld.physchem import (
    ActivityModelSpec,
    CaloricReference,
    ComponentPropertyPackage,
    ComponentSpec,
    PhaseEvaluation,
    PhasePropertyModel,
    PropertyCorrelation,
)
from chemworld.physchem.equilibrium import (
    binary_azeotrope_diagnostic_report,
    bubble_pressure_pa,
    dew_pressure_pa,
    gamma_phi_k_value_report,
)

MODELS = (
    "ideal",
    "margules",
    "wilson",
    "nrtl",
    "uniquac",
    "regular_solution",
    "flory_huggins",
    "hansen",
    *(f"unifac-{v}" for v in range(6)),
)
R = 8.31446261815324
QUANTITIES = ("enthalpy", "cp", "entropy", "activity_coefficient", "chemical_potential")
EXPECTED_UNITS = 195
EXPECTED_COMPARISONS = 3042


def declarations(model, n):
    if model.startswith("unifac-"):
        return unifac_case(int(model[-1]), n)
    if model in ("regular_solution", "flory_huggins", "hansen"):
        return regular_case(model, n)
    return activity_case(model, n)


def reference(spec, T, xs):
    if spec.model == "unifac":
        return public_object_reference(unifac_reference(spec, T, xs))
    if spec.model in ("regular_solution", "flory_huggins", "hansen"):
        return regular_reference(spec, T, xs)
    return high_precision_report(spec, T, xs)


def packages(n):
    return tuple(
        ComponentPropertyPackage(
            ComponentSpec(key, formula),
            (
                PropertyCorrelation(
                    f"23I-Cp-{key}",
                    "cp_polynomial",
                    {"a": 100 + 20 * i},
                    {"temperature": "K"},
                    "J/(mol*K)",
                    property_id="liquid_heat_capacity",
                    validity_ranges={"temperature": (250, 550)},
                    source_note="Explicit fixed synthetic liquid Cp for CONV23I",
                ),
            ),
        )
        for i, (key, formula) in enumerate(zip("ABC"[:n], ("C2H6", "C3H8", "C4H10"), strict=False))
    )


def phase_model(spec):
    return PhasePropertyModel(
        packages(len(spec.component_ids)),
        phase="liquid",
        activity_model=spec,
        references={
            key: CaloricReference(
                phase="liquid",
                temperature_K=300,
                pressure_Pa=1e5,
                energy_J_mol=100 * i,
                entropy_J_mol_K=200 + 20 * i,
                source_note="CONV23I explicit synthetic datum",
            )
            for i, key in enumerate(spec.component_ids)
        },
    )


def vapor_pressures(ids, T):
    return {key: math.exp(24 + 0.2 * i - (4000 + 200 * i) / T) for i, key in enumerate(ids)}


def compare(record, name, value, target, *, atol=2e-8, rtol=2e-9):
    error = abs(value - target)
    tolerance = atol + rtol * abs(target)
    record["comparisons"] += 1
    record["max_absolute_error"] = max(record["max_absolute_error"], error)
    record.setdefault("values", {})[name] = {"actual": value, "expected": target}
    if not math.isfinite(value) or error > tolerance:
        record["failures"].append(
            {
                "quantity": name,
                "actual": value,
                "expected": target,
                "absolute_error": error,
                "tolerance": tolerance,
            }
        )


def equality(record, name, actual, expected):
    compare(record, name, int(actual == expected), 1, atol=0, rtol=0)


def liquid_unit(record, spec, model, T, xs):
    ids, n = spec.component_ids, len(xs)
    x = dict(zip(ids, xs, strict=True))
    amounts = {key: 7 * fraction for key, fraction in x.items()}
    result = model.evaluate(amounts, temperature_K=T, pressure_Pa=1e5, quantities=QUANTITIES)
    ref = reference(spec, T, xs)
    cp = [100 + 20 * i for i in range(n)]
    h = [100 * i + cp[i] * (T - 300) for i in range(n)]
    s = [200 + 20 * i + cp[i] * math.log(T / 300) for i in range(n)]
    expected_h = sum(a * b for a, b in zip(xs, h, strict=True)) + ref["HE"]
    expected_s = sum(a * b for a, b in zip(xs, s, strict=True))
    expected_s += -R * sum(v * math.log(v) for v in xs) + ref["SE"]
    compare(record, "phase_H", result.value("enthalpy"), expected_h)
    compare(
        record,
        "phase_Cp",
        result.value("cp"),
        sum(a * b for a, b in zip(xs, cp, strict=True)) + ref["CpE"],
    )
    compare(record, "phase_S", result.value("entropy"), expected_s)
    for i, key in enumerate(ids):
        compare(
            record,
            "gamma_" + key,
            result.value("activity_coefficient", component=key),
            ref["gamma"][i],
        )
        compare(
            record,
            "mu_" + key,
            result.value("chemical_potential", component=key),
            h[i] - T * s[i] + R * T * math.log(xs[i]) + ref["partial_GE"][i],
        )

    psat = vapor_pressures(ids, T)
    expected_p = sum(xs[i] * ref["gamma"][i] * psat[key] for i, key in enumerate(ids))
    bubble = bubble_pressure_pa(x, vapor_pressures_Pa=psat, activity_model=spec, temperature_K=T)
    compare(record, "bubble_P", bubble, expected_p)
    phi_v = {key: 0.9 + 0.03 * i for i, key in enumerate(ids)}
    phi_l = {key: 0.98 + 0.01 * i for i, key in enumerate(ids)}
    poynting = {key: 1.01 + 0.005 * i for i, key in enumerate(ids)}
    corrected = gamma_phi_k_value_report(
        spec,
        x,
        vapor_pressures_Pa=psat,
        pressure_Pa=1e5,
        temperature_K=T,
        vapor_fugacity_coefficients=phi_v,
        liquid_reference_fugacity_coefficients=phi_l,
        poynting_factors=poynting,
    )
    for i, key in enumerate(ids):
        compare(
            record,
            "corrected_K_" + key,
            corrected.k_values[key],
            ref["gamma"][i] * psat[key] * phi_l[key] * poynting[key] / (1e5 * phi_v[key]),
        )
    bubble_k = gamma_phi_k_value_report(
        spec, x, vapor_pressures_Pa=psat, pressure_Pa=bubble, temperature_K=T
    ).k_values
    compare(record, "bubble_sum_xK", sum(x[k] * bubble_k[k] for k in ids), 1)

    def bubble_at(temp):
        return bubble_pressure_pa(
            x,
            vapor_pressures_Pa=vapor_pressures(ids, temp),
            activity_model=spec,
            temperature_K=temp,
        )

    step = 0.02
    response = (
        bubble_at(T - 2 * step)
        - 8 * bubble_at(T - step)
        + 8 * bubble_at(T + step)
        - bubble_at(T + 2 * step)
    ) / (12 * step)
    expected_response = sum(
        xs[i] * ref["gamma"][i] * psat[key] * (ref["dln_gamma_dT"][i] + (4000 + 200 * i) / T**2)
        for i, key in enumerate(ids)
    )
    compare(record, "bubble_dP_dT", response, expected_response, atol=2e-5, rtol=2e-7)
    compare(
        record,
        "phase_G_equals_sum_xmu",
        result.value("enthalpy") - T * result.value("entropy"),
        sum(x[key] * result.value("chemical_potential", component=key) for key in ids),
    )
    restored = PhasePropertyModel.from_dict(json.loads(json.dumps(model.to_dict())))
    equality(
        record,
        "model_JSON_evaluation",
        restored.evaluate(
            amounts,
            temperature_K=T,
            pressure_Pa=1e5,
            quantities=QUANTITIES,
        ).to_dict(),
        result.to_dict(),
    )
    equality(
        record,
        "snapshot_JSON",
        PhaseEvaluation.from_dict(json.loads(json.dumps(result.to_dict()))).to_dict(),
        result.to_dict(),
    )


def azeotrope_unit(record, A, delta, T):
    spec = ActivityModelSpec("23I-azeotrope", ("A", "B"), "margules", {"A:A|B": A, "A:B|A": A})
    x = (1 + delta / A) / 2
    expected_x = {"A": x, "B": 1 - x}
    p_b = math.exp(24 - 4000 / T)
    psat = {"A": math.exp(delta) * p_b, "B": p_b}
    pressure = math.exp(A * (1 - x) ** 2) * psat["A"]
    scan = binary_azeotrope_diagnostic_report(
        spec,
        vapor_pressures_Pa=psat,
        pressure_Pa=pressure,
        temperature_K=T,
        light_component_id="A",
        grid_size=41,
        composition_bounds=(0.001, 0.999),
        residual_tolerance=1e-10,
    )
    equality(
        record,
        "found_crossing",
        scan.status in ("endpoint_near_crossing", "relative_volatility_crossing"),
        True,
    )
    found = scan.estimated_azeotrope_composition
    if found is None:
        raise ValueError("analytic azeotrope crossing was not found")
    for key in ("A", "B"):
        compare(record, "azeotrope_x_" + key, found[key], expected_x[key])
    report = gamma_phi_k_value_report(
        spec, found, vapor_pressures_Pa=psat, pressure_Pa=pressure, temperature_K=T
    )
    compare(record, "actual_log_K_ratio", math.log(report.k_values["A"] / report.k_values["B"]), 0)
    compare(
        record,
        "azeotrope_bubble_P",
        bubble_pressure_pa(found, vapor_pressures_Pa=psat, activity_model=spec, temperature_K=T),
        pressure,
    )
    compare(
        record,
        "azeotrope_dew_P",
        dew_pressure_pa(found, vapor_pressures_Pa=psat, activity_model=spec, temperature_K=T),
        pressure,
    )
    normalization = sum(found[key] * report.k_values[key] for key in found)
    for key in ("A", "B"):
        compare(record, "azeotrope_K_" + key, report.k_values[key], 1)
        compare(
            record,
            "azeotrope_y_" + key,
            found[key] * report.k_values[key] / normalization,
            expected_x[key],
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    units, start = [], time.monotonic()

    def run(record, function, *inputs):
        unit_start = time.monotonic()
        try:
            function(record, *inputs)
        except Exception as error:
            record["failures"].append({"exception": type(error).__name__, "message": str(error)})
        record["elapsed_seconds"] = time.monotonic() - unit_start
        units.append(record)
        with (args.output / "units.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")
        elapsed = max(time.monotonic() - start, 1e-9)
        print(
            f"23I {record['kind']} {len(units)}/{EXPECTED_UNITS}; "
            f"{len(units) / elapsed:.2f}/s; "
            f"ETA {(EXPECTED_UNITS - len(units)) * elapsed / len(units):.1f}s",
            flush=True,
        )

    for model_name in MODELS:
        for n in (2, 3):
            spec = declarations(model_name, n)
            model = phase_model(spec)
            for T in TEMPERATURES:
                for xs in COMPOSITIONS[n][:2]:
                    run(
                        new_record("liquid", model_name, n, T=T, x=xs),
                        liquid_unit,
                        spec,
                        model,
                        T,
                        xs,
                    )
    for A in (0.5, 1.25, 1.9):
        for delta in (-0.3, 0, 0.4):
            for T in (290.0, 330.0, 370.0):
                run(
                    new_record("azeotrope", "margules", 2, A=A, delta=delta, T=T),
                    azeotrope_unit,
                    A,
                    delta,
                    T,
                )
    failures = [
        {
            "unit": {k: v for k, v in u.items() if k not in {"failures", "values"}},
            "failure": failure,
        }
        for u in units
        for failure in u["failures"]
    ]
    summary = {
        "block": "23I unified phases and equilibrium responses",
        "mode": "development",
        "versions": {"thermo": thermo.__version__, "mpmath": mpmath.__version__},
        "expected_units": EXPECTED_UNITS,
        "expected_comparisons": EXPECTED_COMPARISONS,
        "units": units,
        "comparisons": sum(u["comparisons"] for u in units),
        "failures": failures,
        "elapsed_seconds": time.monotonic() - start,
    }
    summary["passed"] = (
        not failures
        and len(units) == EXPECTED_UNITS
        and summary["comparisons"] == EXPECTED_COMPARISONS
    )
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print({k: v for k, v in summary.items() if k not in {"units", "failures"}}, flush=True)
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
