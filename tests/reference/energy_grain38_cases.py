"""Fixed finite declarations for the prospective 30-unit 38B development block."""

from copy import deepcopy

R = 8.31446261815324
TEMPERATURES = (300.0, 600.0, 1200.0)
PRESSURES = (1e3, 1e4, 1e5)
ALPHAS = (1500.0, 6000.0)
TIMES = (0.05, 0.2, 1.0)


def grain(name, energy, weight=1.0):
    return {"species_id": name, "energy_J_mol": energy, "statistical_weight": weight}


def declaration(alpha=6000.0, topology="branch"):
    value = {
        "model": "finite-energy-grain-master-equation-v1",
        "network_id": f"declared-grains-{topology}",
        "molecular_formula": "H2O",
        "baseline": {
            "cp_J_mol_K": 30.0,
            "h_ref_J_mol": 0.0,
            "s_ref_J_mol_K": 40.0,
            "excluded_internal_modes": ["explicit synthetic vibration/chemical activation states"],
            "reference_temperature_K": 298.15,
            "min_temperature_K": 200.0,
            "max_temperature_K": 1500.0,
        },
        "wells": [
            {
                "well_id": "A",
                "grains": [grain("A0", 0), grain("A1", 4000, 3), grain("A2", 10000, 5)],
                "baths": [
                    {
                        "bath_species_id": "He",
                        "collision_coefficient_L_mol_s": 3000.0,
                        "alpha_J_mol": alpha,
                        "reference_temperature_K": 300.0,
                        "collision_exponent": 0.5,
                        "alpha_exponent": 0.2,
                    },
                    {
                        "bath_species_id": "Ar",
                        "collision_coefficient_L_mol_s": 800.0,
                        "alpha_J_mol": alpha * 0.6,
                        "reference_temperature_K": 300.0,
                        "collision_exponent": 0.25,
                        "alpha_exponent": 0.0,
                    },
                ],
            }
        ],
        "bath_gases": [
            {
                "species_id": name,
                "formula": name,
                "cp_J_mol_K": 2.5 * R,
                "h_ref_J_mol": 0.0,
                "s_ref_J_mol_K": 20.0,
            }
            for name in ("He", "Ar")
        ],
        "channels": [
            {
                "channel_id": "relaxed-product",
                "source_grain_id": "A1",
                "target_species_id": "P",
                "rate_s": 0.7,
            },
            {
                "channel_id": "activated-product",
                "source_grain_id": "A2",
                "target_species_id": "Q",
                "rate_s": 2.0,
            },
        ],
        "sinks": [grain("P", -4000), grain("Q", 0)],
        "source_note": "fixed finite synthetic states; no material-parameter prediction claim",
    }
    if topology == "two-level":
        value["wells"][0]["grains"] = value["wells"][0]["grains"][:2]
        value["channels"], value["sinks"] = [], []
    elif topology == "two-well":
        second = deepcopy(value["wells"][0])
        second["well_id"] = "B"
        second["grains"] = [grain("B0", 2000, 2), grain("B1", 10000, 7)]
        value["wells"].append(second)
        value["channels"].append(
            {
                "channel_id": "isomerize",
                "source_grain_id": "A2",
                "target_species_id": "B1",
                "rate_s": 1.3,
            }
        )
        value["channels"].append(
            {
                "channel_id": "B-product",
                "source_grain_id": "B0",
                "target_species_id": "P",
                "rate_s": 0.4,
            }
        )
    elif topology == "six-grain":
        value["wells"][0]["grains"] = [grain(f"A{i}", 2000 * i, (i + 1) ** 2) for i in range(6)]
        value["channels"][1]["source_grain_id"] = "A5"
    elif topology == "chain":
        # Exact three-state irreversible competition, A2 -> P and A2 -> Q.
        value["wells"][0]["grains"] = [grain("A2", 10000)]
        value["wells"][0]["baths"] = []
        value["channels"] = [
            {
                "channel_id": "sink-P",
                "source_grain_id": "A2",
                "target_species_id": "P",
                "rate_s": 0.7,
            },
            {
                "channel_id": "sink-Q",
                "source_grain_id": "A2",
                "target_species_id": "Q",
                "rate_s": 2.0,
            },
        ]
    elif topology != "branch":
        raise ValueError(topology)
    return value


def initial_amounts(value, temperature=600.0, pressure=1e4, volume_L=1.0):
    amounts = {g["species_id"]: 0.0 for w in value["wells"] for g in w["grains"]}
    amounts.update({s["species_id"]: 0.0 for s in value["sinks"]})
    amounts[value["wells"][0]["grains"][-1]["species_id"]] = 0.01
    bath_total = pressure * volume_L / (1000 * R * temperature)
    amounts.update({"He": 0.7 * bath_total, "Ar": 0.3 * bath_total})
    return amounts


def flow_inputs():
    value = declaration(topology="two-well")
    feed = initial_amounts(value, pressure=2e5)
    feed.update({name: 0.0 for name in feed if name not in {"He", "Ar"}})
    amounts = {"feed": feed, "reactor": initial_amounts(value, pressure=1e3)}
    return value, amounts, 600.0, 0.003, 1.0
