"""Fixed property-proximity and independent flux-equation development comparison."""

import argparse
import itertools
import json
import time
from pathlib import Path

import cantera as ct
import yaml
from gas_transport36_cases import (
    COLLISION_MODELS,
    COMPOSITIONS,
    IDS,
    PRESSURES,
    TEMPERATURES,
    network_from_species,
    public_species,
)

from chemworld.physchem.elements import molecular_weight
from chemworld.physchem.gas_transport import gas_transport_from_network, gas_transport_response
from chemworld.physchem.gas_transport_fluxes import gas_transport_fluxes


def source_phase(species):
    elements = sorted({element for s in species for element in s["composition"]})
    payload = {
        "elements": [{"symbol": e, "atomic-weight": molecular_weight({e: 1})} for e in elements],
        "phases": [
            {
                "name": "gas",
                "thermo": "ideal-gas",
                "transport": "mixture-averaged",
                "elements": elements,
                "species": list(IDS),
            }
        ],
        "species": species,
    }
    return ct.Solution(yaml=yaml.safe_dump(payload), name="gas")


def source_properties(phase, temperature, pressure, composition):
    mu, conductivity = [], []
    for species in IDS:
        phase.TPX = temperature, pressure, {species: 1}
        mu.append(float(phase.viscosity))
        conductivity.append(float(phase.thermal_conductivity))
    phase.TPX = temperature, pressure, composition
    return {
        "species_viscosities_Pa_s": dict(zip(IDS, mu, strict=True)),
        "species_conductivities_W_m_K": dict(zip(IDS, conductivity, strict=True)),
        "binary_diffusivities_m2_s": {
            s: dict(zip(IDS, map(float, phase.binary_diff_coeffs[i]), strict=True))
            for i, s in enumerate(IDS)
        },
        "viscosity_Pa_s": float(phase.viscosity),
        "conductivity_W_m_K": float(phase.thermal_conductivity),
        "partial_enthalpies_J_mol": dict(
            zip(IDS, map(float, phase.partial_molar_enthalpies / 1000), strict=True)
        ),
        "diffusion_coefficients_m2_s": {
            key: dict(zip(IDS, map(float, values), strict=True))
            for key, values in (
                ("molar_mole", phase.mix_diff_coeffs_mole),
                ("mass_mole", phase.mix_diff_coeffs),
                ("mass_mass", phase.mix_diff_coeffs_mass),
            )
        },
    }


def scalar_flux_reference(response, gradient, pressure_gradient, model, frame):
    """Independent scalar Cramer solution; no native matrix assembler or solver."""
    x = [response.mole_fractions[s] for s in IDS]
    masses = [response.molecular_weights_kg_mol[s] for s in IDS]
    mean_mass = sum(a * b for a, b in zip(x, masses, strict=True))
    y = [a * b / mean_mass for a, b in zip(x, masses, strict=True)]
    d = [g + (a - b) * pressure_gradient for g, a, b in zip(gradient, x, y, strict=True)]
    D = [[response.binary_diffusivities_m2_s[s][t] for t in IDS] for s in IDS]
    c = response.molar_density_mol_m3
    weights = masses if frame == "mass" else [1, 1, 1]
    if model == "mixture-averaged":
        coefficients = [
            sum(y[j] for j in range(3) if j != i) / sum(x[j] / D[i][j] for j in range(3) if j != i)
            for i in range(3)
        ]
        provisional = [-c * a * b for a, b in zip(coefficients, d, strict=True)]
        shift = sum(w * n for w, n in zip(weights, provisional, strict=True)) / sum(
            w * a for w, a in zip(weights, x, strict=True)
        )
        return [n - a * shift for n, a in zip(provisional, x, strict=True)]
    # Two friction balances plus the selected frame eliminate N_2 exactly.
    a11, a12, a13 = x[1] / D[0][1] + x[2] / D[0][2], -x[0] / D[0][1], -x[0] / D[0][2]
    a21, a22, a23 = -x[1] / D[1][0], x[0] / D[1][0] + x[2] / D[1][2], -x[1] / D[1][2]
    b11, b12 = a11 - a13 * weights[0] / weights[2], a12 - a13 * weights[1] / weights[2]
    b21, b22 = a21 - a23 * weights[0] / weights[2], a22 - a23 * weights[1] / weights[2]
    rhs1, rhs2 = -c * d[0], -c * d[1]
    determinant = b11 * b22 - b12 * b21
    n0 = (rhs1 * b22 - b12 * rhs2) / determinant
    n1 = (b11 * rhs2 - rhs1 * b21) / determinant
    return [n0, n1, -(weights[0] * n0 + weights[1] * n1) / weights[2]]


def compare(label, actual, expected, *, rtol, atol=1e-15):
    actual, expected = float(actual), float(expected)
    tolerance = atol + rtol * abs(expected)
    return {
        "quantity": label,
        "native": actual,
        "reference": expected,
        "absolute_error": abs(actual - expected),
        "relative_error": None if expected == 0 else abs(actual - expected) / abs(expected),
        "tolerance": tolerance,
        "passed": abs(actual - expected) <= tolerance,
    }


def run(output):
    species = public_species()
    network, phase = network_from_species(species), source_phase(species)
    property_rows, flux_rows, failures = [], [], []
    started = time.perf_counter()
    states = list(itertools.product(TEMPERATURES, PRESSURES, COMPOSITIONS))
    for collision in COLLISION_MODELS:
        for temperature, pressure, composition in states:
            key = {
                "collision": collision,
                "temperature_K": temperature,
                "pressure_Pa": pressure,
                "mole_fractions": dict(zip(IDS, composition, strict=True)),
            }
            row = {**key, "checks": []}
            try:
                spec = gas_transport_from_network(
                    network, thermal_diffusion="off", collision_method=collision
                )
                native = gas_transport_response(
                    spec,
                    network,
                    temperature_K=temperature,
                    pressure_Pa=pressure,
                    mole_fractions=key["mole_fractions"],
                )
                source = source_properties(phase, temperature, pressure, composition)
                row.update(native=native.to_dict(), reference=source)
                checks = row["checks"]
                for quantity, limit in (
                    ("species_viscosities_Pa_s", 0.02),
                    ("species_conductivities_W_m_K", 0.25),
                ):
                    for s in IDS:
                        checks.append(
                            compare(
                                f"{quantity}:{s}",
                                getattr(native, quantity)[s],
                                source[quantity][s],
                                rtol=limit,
                            )
                        )
                for s in IDS:
                    for target in IDS:
                        checks.append(
                            compare(
                                f"D:{s}:{target}",
                                native.binary_diffusivities_m2_s[s][target],
                                source["binary_diffusivities_m2_s"][s][target],
                                rtol=0.02,
                            )
                        )
                for quantity, limit in (("viscosity_Pa_s", 0.02), ("conductivity_W_m_K", 0.25)):
                    checks.append(
                        compare(quantity, getattr(native, quantity), source[quantity], rtol=limit)
                    )
                for basis in ("molar_mole", "mass_mole", "mass_mass"):
                    for s in IDS:
                        checks.append(
                            compare(
                                f"{basis}:{s}",
                                native.diffusion_coefficients_m2_s[basis][s],
                                source["diffusion_coefficients_m2_s"][basis][s],
                                rtol=0.02,
                            )
                        )
                if collision == COLLISION_MODELS[0]:
                    for gradient, pressure_gradient in (
                        ((0.01, -0.01, 0), 0),
                        ((0, 0, 0), 0.2),
                        ((0.01, -0.01, 0), 0.2),
                    ):
                        for model, frame in itertools.product(
                            ("mixture-averaged", "stefan-maxwell"), ("mass", "molar")
                        ):
                            record = {
                                **key,
                                "flux_model": model,
                                "frame": frame,
                                "gradient_per_m": gradient,
                                "pressure_log_gradient_per_m": pressure_gradient,
                                "temperature_gradient_K_m": 3,
                                "checks": [],
                            }
                            try:
                                actual = gas_transport_fluxes(
                                    native,
                                    mole_fraction_gradients_per_m=dict(
                                        zip(IDS, gradient, strict=True)
                                    ),
                                    pressure_log_gradient_per_m=pressure_gradient,
                                    temperature_gradient_K_m=3,
                                    model=model,
                                    frame=frame,
                                    thermal_diffusion="off",
                                )
                                expected = scalar_flux_reference(
                                    native, gradient, pressure_gradient, model, frame
                                )
                                record.update(
                                    native=actual.to_dict(),
                                    reference_molar_fluxes=dict(zip(IDS, expected, strict=True)),
                                )
                                for i, s in enumerate(IDS):
                                    record["checks"].append(
                                        compare(
                                            f"N:{s}",
                                            actual.molar_fluxes_mol_m2_s[s],
                                            expected[i],
                                            rtol=2e-10,
                                            atol=1e-12,
                                        )
                                    )
                                    record["checks"].append(
                                        compare(
                                            f"j:{s}",
                                            actual.mass_fluxes_kg_m2_s[s],
                                            expected[i] * native.molecular_weights_kg_mol[s],
                                            rtol=2e-10,
                                            atol=1e-12,
                                        )
                                    )
                                enthalpy = sum(
                                    n * source["partial_enthalpies_J_mol"][s]
                                    for s, n in zip(IDS, expected, strict=True)
                                )
                                conduction = -native.conductivity_W_m_K * 3
                                for quantity, value in (
                                    ("diffusion_enthalpy_flux_W_m2", enthalpy),
                                    ("fourier_heat_flux_W_m2", conduction),
                                    ("total_heat_flux_W_m2", enthalpy + conduction),
                                    ("frame_residual", 0),
                                ):
                                    record["checks"].append(
                                        compare(
                                            quantity,
                                            getattr(actual, quantity),
                                            value,
                                            rtol=2e-10,
                                            atol=1e-12,
                                        )
                                    )
                            except Exception as exc:
                                record["error"] = f"{type(exc).__name__}: {exc}"
                            flux_rows.append(record)
            except Exception as exc:
                row["error"] = f"{type(exc).__name__}: {exc}"
            property_rows.append(row)
            completed = len(property_rows)
            elapsed = time.perf_counter() - started
            if completed % 9 == 0:
                print(
                    f"36A properties {completed}/54; fluxes {len(flux_rows)}/324; "
                    f"{completed / max(elapsed, 1e-9):.1f} states/s; "
                    f"ETA {elapsed / completed * (54 - completed):.1f}s",
                    flush=True,
                )
    for group, rows in (
        ("approximate_properties", property_rows),
        ("matched_coefficient_flux", flux_rows),
    ):
        for i, row in enumerate(rows):
            if "error" in row:
                failures.append({"block": group, "case_index": i, "error": row["error"]})
            failures.extend(
                {"block": group, "case_index": i, **check}
                for check in row["checks"]
                if not check["passed"]
            )
    summary = {
        "status": "development",
        "source": f"Cantera {ct.__version__} public gri30 species; matched native atomic weights",
        "property_model_scope": "LJ fit + Eucken/WHZ proximity, not kinetic-model equivalence",
        "flux_reference_scope": (
            "independent scalar Cramer/explicit frame equations with "
            "declared native D matrix; original source enthalpy"
        ),
        "expected_property_cases": 54,
        "completed_property_cases": sum("error" not in row for row in property_rows),
        "expected_flux_cases": 324,
        "completed_flux_cases": sum("error" not in row for row in flux_rows),
        "expected_checks": 4644,
        "completed_checks": sum(len(row["checks"]) for row in (*property_rows, *flux_rows)),
        "passed_checks": sum(
            c["passed"] for row in (*property_rows, *flux_rows) for c in row["checks"]
        ),
        "failures": failures,
        "property_rows": property_rows,
        "flux_rows": flux_rows,
        "elapsed_s": time.perf_counter() - started,
    }
    serialized = json.dumps(summary, indent=2, allow_nan=False)
    with output.open("x", encoding="utf-8") as handle:
        handle.write(serialized)
    print(
        json.dumps(
            {
                k: v
                for k, v in summary.items()
                if k not in {"property_rows", "flux_rows", "failures"}
            }
        ),
        flush=True,
    )
    return int(bool(failures) or summary["completed_checks"] != 4644)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    raise SystemExit(run(parser.parse_args().output))
