"""Every Bell2018 row reaches alpha derivatives and the original PR/SRK state."""

import argparse
import json
import time
from pathlib import Path

import chemicals
import mpmath as mp
from run_activity_thermodynamics_block import compare, new_record
from thermo.eos_alpha_functions import Twu91_a_alpha

from chemworld.data.phase_change import PhaseChangeDatabase
from chemworld.data.twu_eos import twu_eos_component_from_record
from chemworld.physchem.cubic_alpha import cubic_alpha_response
from chemworld.physchem.eos import CubicEOSSpec, cubic_pure_parameters, evaluate_cubic_eos


def high_alpha(T, Tc, coefficients):
    with mp.workdps(80):
        T, Tc = mp.mpf(str(T)), mp.mpf(str(Tc))
        ell, m, n = (mp.mpf(str(v)) for v in coefficients)
        power = m * n
        exponent = n * (m - 1)
        reduced = T / Tc
        value = reduced**exponent * mp.exp(ell * (1 - reduced**power))
        f1 = (exponent - ell * power * reduced**power) / T
        f2 = (-exponent - ell * power * (power - 1) * reduced**power) / T**2
        f3 = (2 * exponent - ell * power * (power - 1) * (power - 2) * reduced**power) / T**3
        return value, value * f1, value * (f2 + f1 * f1), value * (f3 + 3 * f1 * f2 + f1**3)


def high_state(component, model, T, P):
    with mp.workdps(80):
        temp, pressure, Tc, Pc = (
            mp.mpf(str(v))
            for v in (T, P, component.critical_temperature_K, component.critical_pressure_Pa)
        )
        R = mp.mpf("8.31446261815324")
        wa, wb = (
            (mp.mpf(".45724"), mp.mpf(".07780"))
            if model == "peng_robinson"
            else (mp.mpf(".42747"), mp.mpf(".08664"))
        )
        a0 = wa * R * R * Tc * Tc / Pc
        alpha = high_alpha(T, component.critical_temperature_K, component.alpha_coefficients)
        a, at, att, attt = (a0 * v for v in alpha)
        b = wb * R * Tc / Pc
        s1, s2 = (
            (1 + mp.sqrt(2), 1 - mp.sqrt(2)) if model == "peng_robinson" else (mp.mpf(1), mp.mpf(0))
        )

        def equation(z):
            volume = z * R * temp / pressure
            return R * temp / (volume - b) - a / ((volume + s1 * b) * (volume + s2 * b)) - pressure

        z = mp.findroot(
            equation, (mp.mpf(".95"), mp.mpf("1.05")), solver="secant", tol=mp.mpf("1e-65")
        )
        volume = z * R * temp / pressure
        logterm = mp.log((volume + s1 * b) / (volume + s2 * b))
        h = R * temp * (z - 1) + (temp * at - a) / ((s1 - s2) * b) * logterm
        entropy = R * mp.log(z - pressure * b / (R * temp)) + at / ((s1 - s2) * b) * logterm
        gibbs = h - temp * entropy
        return dict(
            zip(
                ("Z", "V", "phi", "H", "S", "G", "a", "da", "d2a", "d3a"),
                map(
                    float,
                    (z, volume, mp.exp(gibbs / (R * temp)), h, entropy, gibbs, a, at, att, attt),
                ),
                strict=True,
            )
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    database = PhaseChangeDatabase(Path(chemicals.__file__).parent)
    records = database.records("BELL_TWU")
    result = {
        "block": "23E Bell2018 Twu native EOS",
        "mode": "development",
        "source_rows": len(records),
        "expected_rows": 2571,
        "expected_units": 15426,
        "expected_comparisons": 97698,
        "units": [],
    }
    for index, record in enumerate(records, 1):
        component = twu_eos_component_from_record(record)
        identifier = component.component_id
        Tc, Pc = component.critical_temperature_K, component.critical_pressure_Pa
        for reduced in (0.7, 1.0, 1.3):
            rec = new_record(
                "alpha", "twu91", 1, record_id=record.record_id, reduced_temperature=reduced
            )
            result["units"].append(rec)
            try:
                T = reduced * Tc
                actual = cubic_alpha_response(
                    temperature_K=T,
                    critical_temperature_K=Tc,
                    model="twu91",
                    coefficients=component.alpha_coefficients,
                )
                ref = Twu91_a_alpha()
                ref.Tc = Tc
                ref.a = 1.0
                ref.alpha_coeffs = component.alpha_coefficients
                first = ref.a_alpha_and_derivatives_pure(T)
                target = (*first, float(high_alpha(T, Tc, component.alpha_coefficients)[3]))
                compare(
                    rec,
                    (actual.alpha, actual.d_alpha_dT, actual.d2_alpha_dT2, actual.d3_alpha_dT3),
                    target,
                )
            except Exception as exc:
                rec["failures"].append({"exception": str(exc)})
        for model in ("peng_robinson", "srk"):
            rec = new_record("EOS", model, 1, record_id=record.record_id)
            result["units"].append(rec)
            try:
                T, P = 1.2 * Tc, 0.1 * Pc
                spec = CubicEOSSpec("Bell-source", model, (component,))
                native = evaluate_cubic_eos(spec, {identifier: 1.0}, temperature_K=T, pressure_Pa=P)
                pure = cubic_pure_parameters(component, model=model, temperature_K=T)
                actual = {
                    "Z": native.compressibility_factor,
                    "V": native.molar_volume_m3_mol,
                    "phi": native.fugacity_coefficients[identifier],
                    "H": native.molar_residual_enthalpy_J_mol,
                    "S": native.molar_residual_entropy_J_mol_K,
                    "G": native.molar_residual_gibbs_J_mol,
                    "a": pure.a_alpha,
                    "da": pure.da_alpha_dT,
                    "d2a": pure.d2a_alpha_dT2,
                    "d3a": pure.d3a_alpha_dT3,
                }
                compare(rec, actual, high_state(component, model, T, P))
            except Exception as exc:
                rec["failures"].append({"exception": str(exc)})
        rec = new_record("source_roundtrip", "twu91", 1, record_id=record.record_id)
        result["units"].append(rec)
        try:
            spec = CubicEOSSpec("source", "peng_robinson", (component,))
            restored = CubicEOSSpec.from_dict(json.loads(json.dumps(spec.to_dict()))).components[0]
            compare(
                rec,
                {
                    "values": (
                        restored.critical_temperature_K,
                        restored.critical_pressure_Pa,
                        *restored.alpha_coefficients,
                    ),
                    "source_preserved": float(
                        restored.alpha_provenance == component.alpha_provenance
                    ),
                },
                {
                    "values": tuple(
                        float(record.raw(key)) for key in ("Tc", "Pc", "c0", "c1", "c2")
                    ),
                    "source_preserved": 1.0,
                },
            )
        except Exception as exc:
            rec["failures"].append({"exception": str(exc)})
        if index % 100 == 0:
            elapsed = max(time.monotonic() - start, 1e-9)
            print(
                f"Bell actual EOS rows {index}/2571; {index / elapsed:.2f}/s; "
                f"ETA {(2571 - index) * elapsed / index:.1f}s",
                flush=True,
            )
        if index % 500 == 0:
            (args.output / "partial.json").write_text(
                json.dumps(result, indent=2, allow_nan=False), encoding="utf-8"
            )
    result["comparisons"] = sum(r["comparisons"] for r in result["units"])
    result["failures"] = [
        {"unit": i, "model": r["model"], "record_id": r["record_id"], **f}
        for i, r in enumerate(result["units"])
        for f in r["failures"]
    ]
    result["elapsed_seconds"] = time.monotonic() - start
    result["passed"] = (
        not result["failures"] and len(result["units"]) == 15426 and result["comparisons"] == 97698
    )
    (args.output / "summary.json").write_text(
        json.dumps(result, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(
        json.dumps({k: result[k] for k in ("comparisons", "passed", "elapsed_seconds")}), flush=True
    )
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
