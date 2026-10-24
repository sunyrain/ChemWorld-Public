"""Fixed 39A mathematical block and a real public StatePropertyReport example."""

import argparse
import json
import math
import time
from itertools import product
from pathlib import Path

from aqueous39_fixtures import VALUES, activity, database, inventory
from scipy.integrate import quad

from chemworld.physchem.property_reports import R_J_PER_MOL_K

FAMILIES = ("ideal", "DebyeHuckelLimiting", "DebyeHuckel", "Davies", "SIT", "Pitzer")
TEMPERATURES, PRESSURES, MOLALITIES = (280.0, 298.15, 330.0), (1e5, 1e6), (0.001, 0.1, 1.0)


def dilute_gamma(family, m):
    root = math.sqrt(m)
    if family == "ideal":
        return 0
    if family == "DebyeHuckelLimiting":
        return -math.log(10) * 0.509 * root
    if family == "Davies":
        return -math.log(10) * 0.509 * (root / (1 + root) - 0.3 * m)
    return -math.log(10) * 0.509 * root / (1 + 1.5 * root) + (
        math.log(10) * 0.06 * m if family == "SIT" else 0
    )


def binary_reference(family, m):
    if family != "Pitzer":
        gamma = dilute_gamma(family, m)
        phi = 1 + gamma - quad(lambda s: dilute_gamma(family, s), 0, m, epsabs=1e-13)[0] / m
        return gamma, phi
    root, b = math.sqrt(m), 1.2
    gamma = (
        -0.392 * (root / (1 + b * root) + 2 / b * math.log(1 + b * root)) + 2 * m * VALUES["beta0"]
    )
    phi = 1 - 0.392 * root / (1 + b * root) + m * VALUES["beta0"]
    for beta, alpha in ((VALUES["beta1"], 2.0), (VALUES["beta2"], 12.0)):
        x = alpha * root
        g = 2 * quad(lambda y, x=x: y * math.exp(-x * y), 0, 1, epsabs=1e-13)[0]
        gamma += m * beta * (g + math.exp(-x))
        phi += m * beta * math.exp(-x)
    x = VALUES["omega"] * root
    h = quad(lambda y: y**3 * math.exp(-x * y), 0, 1, epsabs=1e-13)[0]
    gamma += m * m * (3 * VALUES["C0"] + 4 * VALUES["C1"] * h + 2 * VALUES["C1"] * math.exp(-x))
    phi += 2 * m * m * (VALUES["C0"] + VALUES["C1"] * math.exp(-x))
    return gamma, phi


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    grid = [
        ("activities", family, t, p, m)
        for family, t, p, m in product(FAMILIES, TEMPERATURES, PRESSURES, MOLALITIES)
    ]
    grid += [("database", None, t, p, None) for t, p in product(TEMPERATURES, PRESSURES)]
    (output / "units.json").write_text(json.dumps(grid, indent=2))
    db = database()
    amounts = inventory(db)
    failures, count, passed = [], 0, 0
    started = time.monotonic()
    print(f"start units=0/{len(grid)} comparisons=576", flush=True)

    def compare(unit, label, actual, expected):
        nonlocal count, passed
        ok = math.isfinite(actual) and math.isclose(actual, expected, rel_tol=2e-10, abs_tol=2e-10)
        record = {
            "unit": unit,
            "quantity": label,
            "actual": actual,
            "expected": expected,
            "passed": ok,
        }
        count += 1
        passed += ok
        if not ok:
            failures.append(record)
        return record

    with (output / "records.jsonl").open("w", encoding="utf-8") as stream:
        for unit, (kind, family, t, p, m) in enumerate(grid, 1):
            records = []
            if kind == "activities":
                model = activity(family)
                actual = model.evaluate(
                    {"Na+": m, "Cl-": m},
                    temperature_K=t,
                    pressure_Pa=p,
                    solvent_molar_mass_kg_mol=0.018,
                )
                gamma, phi = binary_reference(family, m)
                for ion in ("Na+", "Cl-"):
                    records.append(
                        compare(
                            unit, "log_gamma_" + ion, actual.log_activity_coefficients[ion], gamma
                        )
                    )
                records.append(
                    compare(unit, "osmotic_coefficient", actual.osmotic_coefficient, phi)
                )
            else:
                report = db.chemical_potentials(
                    amounts, temperature_K=t, pressure_Pa=p, gas_fugacities_Pa={"CO2g": 2e4}
                )
                for species in db.species:
                    state = report.standard_states[species.species_id]
                    ref = species.standard_state.thermo.evaluate(298.15)
                    shift = (species.standard_state.molar_volume_m3_mol or 0) * (p - 1e5)
                    h = ref.enthalpy_J_mol + ref.cp_J_mol_K * (t - 298.15) + shift
                    s = ref.entropy_J_mol_K + ref.cp_J_mol_K * math.log(t / 298.15)
                    for name, expected in (
                        ("enthalpy_J_mol", h),
                        ("entropy_J_mol_K", s),
                        ("gibbs_J_mol", h - t * s),
                    ):
                        records.append(
                            compare(
                                unit,
                                species.species_id + ":" + name,
                                getattr(state, name),
                                expected,
                            )
                        )
                for key, reaction in db.reaction_affinities(report).items():
                    expected = (
                        reaction["delta_gibbs_J_mol"]
                        + R_J_PER_MOL_K * t * reaction["log_reaction_quotient"]
                    )
                    records.append(
                        compare(
                            unit,
                            key + ":reaction_gibbs_J_mol",
                            reaction["reaction_gibbs_J_mol"],
                            expected,
                        )
                    )
            stream.write(
                json.dumps({"unit": unit, "definition": grid[unit - 1], "comparisons": records})
                + "\n"
            )
    liquid = {
        s.species_id: amounts[s.species_id] for s in db.species if s.phase in ("aqueous", "solvent")
    }
    state = db.property_report(liquid, temperature_K=330.0, pressure_Pa=1e6)
    (output / "example_database.json").write_text(
        json.dumps(db.to_dict(), indent=2), encoding="utf-8"
    )
    (output / "example_state_report.json").write_text(
        json.dumps(state.to_dict(), indent=2), encoding="utf-8"
    )
    summary = {
        "mode": "development",
        "source": "explicit synthetic data, no material-data claim",
        "units": len(grid),
        "comparisons": count,
        "passed": passed,
        "failed": count - passed,
        "failures": failures,
        "elapsed_seconds": time.monotonic() - started,
        "example": "example_state_report.json generated by AqueousDatabase.property_report",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"complete units={len(grid)}/{len(grid)} passed={passed}/{count}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
