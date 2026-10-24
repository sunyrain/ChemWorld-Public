"""Fixed source cards for 38C; parameters are authored synthetic inputs."""

from copy import deepcopy

R = 8.31446261815324
SOURCE = {
    "source_id": "authored-rrkm38-reference",
    "priority": 0,
    "citation": "Explicit synthetic harmonic states, not real molecular data",
    "version": "2026-10-24",
    "license": "CC0-1.0",
    "checksum": "",
}
COUNT_CASES = ("one-mode", "two-mode", "sparse", "degeneracy-zpe", "nearest", "empty-ts")
PHYSICAL_CASES = (
    "pressure-low",
    "pressure-mid",
    "pressure-high",
    "temperature-low",
    "temperature-high",
    "finite-flow",
)


def state(name, quanta, energy=0.0, degeneracy=1, basis="zero-point-corrected"):
    return {
        "state_id": name,
        "vibrational_quanta_J_mol": list(quanta),
        "reference_energy_J_mol": energy,
        "energy_basis": basis,
        "electronic_degeneracy": degeneracy,
        "source": deepcopy(SOURCE),
    }


def base():
    return {
        "model": "harmonic-rrkm-grain-v1",
        "network_id": "rrkm38-fixed",
        "molecular_formula": "C6H6",
        "baseline": {
            "cp_J_mol_K": 30.0,
            "h_ref_J_mol": 0.0,
            "s_ref_J_mol_K": 40.0,
            "excluded_internal_modes": [f"A:vibration:{i}" for i in range(30)],
            "reference_temperature_K": 298.15,
            "min_temperature_K": 200.0,
            "max_temperature_K": 1500.0,
        },
        "well": state("A", [1000.0] * 30, -15000.0, basis="electronic"),
        "grid": {
            "grain_width_J_mol": 1000.0,
            "maximum_grain_index": 14,
            "quantization": "exact",
            "maximum_quantization_error_J_mol": 0.0,
        },
        "baths": [
            {
                "bath_species_id": s,
                "collision_coefficient_L_mol_s": k,
                "alpha_J_mol": a,
                "reference_temperature_K": 300.0,
                "collision_exponent": 0.5,
                "alpha_exponent": 0.0,
            }
            for s, k, a in (("He", 3000.0, 3000.0), ("Ar", 1000.0, 4000.0))
        ],
        "bath_gases": [
            {
                "species_id": s,
                "formula": s,
                "cp_J_mol_K": 2.5 * R,
                "h_ref_J_mol": 0.0,
                "s_ref_J_mol_K": 20.0,
            }
            for s in ("He", "Ar")
        ],
        "channels": [
            {
                "channel_id": name,
                "transition_state": ts,
                "imaginary_frequency_cm_inverse": -500.0,
                "path_degeneracy": d,
                "product": {"species_id": p, "energy_J_mol": energy, "statistical_weight": 1.0},
            }
            for name, ts, d, p, energy in (
                (
                    "to-P",
                    state("TS-P", [1000.0] * 29, -2500.0, basis="electronic"),
                    1,
                    "P",
                    -4000.0,
                ),
                (
                    "to-Q",
                    state("TS-Q", [1000.0] * 28 + [2000.0], -2000.0, basis="electronic"),
                    2,
                    "Q",
                    -2000.0,
                ),
            )
        ],
    }


def count_card(name):
    card = base()
    card["channels"] = card["channels"][:1]
    quanta = {
        "one-mode": [1000.0],
        "two-mode": [1000.0, 2000.0],
        "sparse": [2000.0, 3000.0, 4000.0],
        "degeneracy-zpe": [1000.0, 2000.0, 3000.0],
        "nearest": [1100.0, 1800.0, 2900.0],
        "empty-ts": [1000.0],
    }[name]
    well = state("A", quanta)
    transition = state("TS", quanta[:-1], 4000.0)
    if name == "degeneracy-zpe":
        well = state("A", quanta, -3000.0, 3, "electronic")
        transition = state("TS", quanta[:-1], 2500.0, 2, "electronic")
        card["channels"][0]["path_degeneracy"] = 4
    if name == "nearest":
        card["grid"].update(quantization="nearest", maximum_quantization_error_J_mol=250.0)
        transition = state("TS", [1100.0, 1800.0], 4200.0)
    if name == "empty-ts":
        transition = state("TS", [], 20000.0)
    card["well"] = well
    card["channels"][0]["transition_state"] = transition
    card["baseline"]["excluded_internal_modes"] = [f"A:vibration:{i}" for i in range(len(quanta))]
    return card


def physical_card(name):
    card = base()
    temperature = {"temperature-low": 300.0, "temperature-high": 900.0}.get(name, 600.0)
    pressure = {"pressure-low": 1000.0, "pressure-high": 100000.0}.get(name, 10000.0)
    total_bath = pressure / (1000 * R * temperature)
    initial = {
        "A:grain:0": 0.0009,
        "A:grain:14": 0.0001,
        "He": 0.8 * total_bath,
        "Ar": 0.2 * total_bath,
    }
    return card, temperature, initial, 0.2


def flow_card():
    card, temperature, initial, _ = physical_card("finite-flow")
    return card, temperature, {"cell": initial, "feed": {"He": 0.02}}, 0.0002


def refinement_cards(name):
    original = count_card("two-mode" if name == "sparse-refinement" else "nearest")
    result = []
    for width in (1000.0, 500.0, 250.0):
        card = deepcopy(original)
        card["grid"].update(grain_width_J_mol=width, maximum_grain_index=int(20000 / width))
        if name == "quantized-refinement":
            card["grid"]["maximum_quantization_error_J_mol"] = width / 2
        result.append(card)
    return result
