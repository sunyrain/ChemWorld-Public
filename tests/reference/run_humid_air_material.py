"""Fixed 24 parent states, 48 phase selections, 216 amount/energy comparisons."""

import argparse
import json
from collections import Counter
from dataclasses import replace
from pathlib import Path

from tests.reference.run_incompressible import compare
from tests.test_humid_air_volume import thermal_inputs

from chemworld.physchem.humid_air_thermal import DRY_AIR_MOLE_FRACTIONS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("never overwrite material selection evidence")
    cases, rows = [], []
    for phase, temperatures in (("liquid", (280.0, 300.0, 340.0)), ("ice", (220.0, 240.0, 260.0))):
        for T in temperatures:
            for p in (101325.0, 1e6):
                for extra in (0.0, 0.01):
                    thermal, state = thermal_inputs(T, p, phase, dry=0.002, extra=extra)
                    current = thermal.equilibrium_state(state)
                    dry_molar_mass = sum(
                        x * c.molecular_weight_g_mol / 1000
                        for x, c in zip(DRY_AIR_MOLE_FRACTIONS, thermal.components[:3], strict=True)
                    )
                    for outlet in ("gas", "condensate"):
                        inputs = {
                            "phase": phase,
                            "T_K": T,
                            "p_Pa": p,
                            "extra": extra,
                            "dry_mass_kg": 0.002,
                            "outlet": outlet,
                            "fraction": 0.25,
                            "state": dict(state.amounts_mol),
                            "volume_L": state.volume_L,
                        }
                        model = replace(thermal, material_outlet_phase=outlet)
                        result, error = None, None
                        try:
                            result = model.material_outlet(state, 0.25)
                        except Exception as exc:
                            error = f"{type(exc).__name__}: {exc}"
                        absent = outlet == "condensate" and extra == 0
                        status = (
                            "expected_no_phase_rejection"
                            if absent and error and "no condensed water" in error
                            else "unexpected_acceptance"
                            if absent and not error
                            else "selection_error"
                            if error
                            else "returned"
                        )
                        cases.append(
                            {"inputs": inputs, "status": status, "result": result, "error": error}
                        )
                        if not absent:
                            expected = dict.fromkeys(thermal.inventory_keys, 0.0)
                            if outlet == "gas":
                                for key, x in zip(
                                    thermal.inventory_keys[:3], DRY_AIR_MOLE_FRACTIONS, strict=True
                                ):
                                    expected[key] = 0.002 / dry_molar_mass * x * 0.25
                                vapor = 0.002 * current.gas.humidity_ratio_kg_water_kg_dry_air
                                expected[thermal.water.identifier] = (
                                    vapor * 0.25 / (thermal.water.molecular_weight_g_mol / 1000)
                                )
                                mass = (0.002 + vapor) * 0.25
                                condensed_u = (
                                    0.0
                                    if current.condensate is None
                                    else current.condensed_water_mass_kg
                                    * current.condensate.enthalpy_J_kg
                                    - current.gas.pressure_Pa * current.condensed_volume_m3
                                )
                                energy = 0.25 * (
                                    current.internal_energy_J
                                    - condensed_u
                                    + current.gas.pressure_Pa * current.gas_volume_m3
                                )
                            else:
                                mass = 0.002 * extra * 0.25
                                expected[thermal.water.identifier] = mass / (
                                    thermal.water.molecular_weight_g_mol / 1000
                                )
                                energy = 0.25 * (
                                    current.internal_energy_J
                                    - current.gas_mass_kg * current.gas.internal_energy_J_kg
                                    + current.gas.pressure_Pa * current.condensed_volume_m3
                                )
                            values = [(k, v, 1e-12) for k, v in expected.items()]
                            values += [
                                ("withdrawn_mass_kg", mass, 1e-14),
                                ("enthalpy_J", energy, 1e-7),
                            ]
                            for key, expected_value, atol in values:

                                def actual(key=key, result=result, error=error, expected=expected):
                                    if error:
                                        raise ValueError(error)
                                    return (
                                        result["amounts_mol"][key]
                                        if key in expected
                                        else result[key]
                                    )

                                rows.append(
                                    compare(
                                        inputs,
                                        key,
                                        actual,
                                        lambda value=expected_value: value,
                                        atol,
                                        2e-10,
                                    )
                                )
                        print(
                            f"humid phase selection {len(cases)}/48; comparisons {len(rows)}/216",
                            flush=True,
                        )
    assert len(cases) == 48 and len(rows) == 216
    payload = {
        "evidence_mode": "development",
        "reference": "phase mass and U+pV identity",
        "cases": cases,
        "comparisons": rows,
        "selection_counts": dict(Counter(c["status"] for c in cases)),
        "comparison_counts": dict(Counter(r["status"] for r in rows)),
    }
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(payload["selection_counts"], payload["comparison_counts"])


if __name__ == "__main__":
    main()
