"""Independent analytic pressure work and original authored RK4 RO reference."""

import sys
from pathlib import Path

# The existing RO47 oracle is also a directly executable reference script.
sys.path.insert(0, str(Path(__file__).parent))
from ro47_fixtures import ROWS, WATER_MASS
from ro47_reference import R, davies_terms, profile, properties


def recovery(family, rates, temperature, inlet, outlet, efficiency, maximum):
    if not any(rates.values()):
        return {
            "volume": 0.0,
            "availability": 0.0,
            "enthalpy_change": 0.0,
            "work": 0.0,
            "dissipation": 0.0,
            "heat": 0.0,
            "entropy": 0.0,
            "power_limited": False,
        }
    water, salt = rates["H2O"], rates["Na+"]
    kg = water * WATER_MASS
    excess = davies_terms(salt / kg)[1] if family == "Davies" else 0.0
    volume = sum(rates[key] * v for key, _, _, _, _, v in ROWS)
    volume += R * temperature * kg * 1e-9 * excess
    available = (inlet - outlet) * volume
    dh = (
        properties(family, rates, temperature, outlet)["enthalpy_J"]
        - properties(family, rates, temperature, inlet)["enthalpy_J"]
    )
    work = min(efficiency * available, maximum)
    return {
        "volume": volume,
        "availability": available,
        "enthalpy_change": dh,
        "work": work,
        "dissipation": available - work,
        "heat": dh + work,
        "entropy": (available - work) / temperature,
        "power_limited": efficiency * available > maximum,
    }


def axial(model, amounts, requested=10 / 6e7, cells=128):
    drop = model.inlet_pressure_Pa - model.reservoir_pressure_Pa
    pump = model.pump
    q = (
        0.0
        if pump.maximum_pressure_rise_Pa <= drop
        else min(requested, pump.maximum_shaft_power_W * pump.efficiency / drop)
    )
    if q == 0:
        n = dict.fromkeys(amounts, 0.0)
        result = {
            "inlet_mol_s": n,
            "retentate_mol_s": n,
            "permeate_mol_s": n,
            "pressure_Pa": model.reservoir_pressure_Pa,
        }
    else:
        result = profile(model, amounts, q, cells)
    result.update(flow=q, hydraulic=drop * q, shaft=drop * q / pump.efficiency)
    device = model.energy_recovery
    result["recovery"] = recovery(
        model.properties.database.activity_model.family,
        result["retentate_mol_s"],
        model.temperature_K,
        result["pressure_Pa"],
        model.reservoir_pressure_Pa,
        device.efficiency,
        device.maximum_shaft_power_W,
    )
    return result


def finite(model, initial, duration):
    result = axial(model, initial[0])
    flows = [
        {k: -v for k, v in result["inlet_mol_s"].items()},
        result["permeate_mol_s"],
        result["retentate_mol_s"],
    ]
    final = [{k: n[k] + duration * f[k] for k in n} for n, f in zip(initial, flows, strict=True)]
    family, t, p = (
        model.properties.database.activity_model.family,
        model.temperature_K,
        model.reservoir_pressure_Pa,
    )
    before = [properties(family, n, t, p) for n in initial]
    after = [properties(family, n, t, p) for n in final]
    dh = sum(s["enthalpy_J"] for s in after) - sum(s["enthalpy_J"] for s in before)
    result.update(
        amounts=final,
        initial=before,
        final=after,
        enthalpy_change=dh,
        bath=dh - result["hydraulic"] * duration + result["recovery"]["work"] * duration,
    )
    return result
