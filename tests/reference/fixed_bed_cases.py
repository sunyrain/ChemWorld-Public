"""Original synthetic fixed cases; no calibrated material accuracy is implied."""

import numpy as np

R = 8.31446261815324
IDS = ("CO2", "H2O", "N2")


def cases():
    base = {
        "temperature": 298.15,
        "source_pressure": 105000.0,
        "collector_pressure": 100000.0,
        "tank_volume_L": 10.0,
        "feed": (0.2, 0.05, 0.75),
        "collector": (0.0, 0.0, 1.0),
        "rates": (0.3, 0.6),
        "loading": (0.0, 0.0),
        "stages": (("source", 30.0),),
    }
    return [
        {**base, **update}
        for update in (
            {"id": "no-adsorption", "feed": (0.0, 0.0, 1.0), "rates": (0.0, 0.0)},
            {"id": "single-adsorbate", "feed": (0.2, 0.0, 0.8)},
            {"id": "competitive"},
            {"id": "preloaded-purge", "feed": (0.0, 0.0, 1.0), "loading": (0.3, 0.2)},
            {
                "id": "rich-purge-rich",
                "stages": (("source", 30.0), ("purge", 30.0), ("source", 30.0)),
            },
            {
                "id": "finite-small",
                "tank_volume_L": 0.02,
                "source_pressure": 120000.0,
                "collector_pressure": 80000.0,
            },
            {
                "id": "reverse",
                "source_pressure": 100000.0,
                "collector_pressure": 105000.0,
                "feed": (0.0, 0.0, 1.0),
                "collector": (0.2, 0.0, 0.8),
            },
            {"id": "snapshot", "stages": (("source", 15.0), ("source", 15.0)), "snapshot": True},
        )
    ]


def specification(case, cells):
    from chemworld.physchem.coupled_system import SpeciesInternalEnergy
    from chemworld.physchem.fixed_bed import FixedBedSpec
    from chemworld.physchem.sorbent_kinetics import SorbentKineticsSpec

    sorbent = SorbentKineticsSpec(
        "synthetic-fixed-bed",
        1.0,
        {"CO2": 100.0, "H2O": 500.0},
        dict(zip(IDS[:2], case["rates"], strict=True)),
        {"CO2": -20000.0, "H2O": -30000.0},
    )
    return FixedBedSpec(
        "bed",
        tuple(f"cell_{i}" for i in range(cells)),
        tuple(np.linspace(0.0, 0.1, cells + 1)),
        1e-4,
        0.4,
        1e-4,
        0.001,
        sorbent,
        dict(zip(IDS, (0.044, 0.018, 0.028), strict=True)),
        {k: SpeciesInternalEnergy(0.0, cv) for k, cv in zip(IDS, (28.0, 25.0, 20.8), strict=True)},
        1.8e-5,
    )


def initial_data(case, cells):
    volumes = np.r_[
        np.full(cells, 0.004 / cells), case["tank_volume_L"], case["tank_volume_L"], 10.0
    ]
    centers = (np.arange(cells) + 0.5) / cells
    pressure = np.r_[
        case["source_pressure"] + centers * (case["collector_pressure"] - case["source_pressure"]),
        case["source_pressure"],
        case["collector_pressure"],
        110000.0,
    ]
    fractions = np.array(
        [(0.0, 0.0, 1.0)] * cells + [case["feed"], case["collector"], (0.0, 0.0, 1.0)]
    )
    gas = pressure[:, None] * volumes[:, None] * 0.001 / (R * case["temperature"]) * fractions
    bound = np.tile(np.array(case["loading"]) * 0.001 / cells, (cells, 1))
    return volumes, gas, bound


def native_world(case, cells):
    from chemworld.foundation.state import WorldState
    from chemworld.foundation.state_ledgers import (
        EquipmentLedger,
        EquipmentRecord,
        PhaseLedger,
        PhaseRecord,
        VesselLedger,
        VesselRecord,
    )
    from chemworld.foundation.state_ownership import StateSlots
    from chemworld.physchem.fixed_bed import fixed_bed_slots

    spec = specification(case, cells)
    volumes, gas, _ = initial_data(case, cells)
    names = (*spec.cell_phase_ids, "source", "collector", "purge")
    phases = PhaseLedger(
        {
            name: PhaseRecord(
                name,
                "assembly",
                "gas",
                float(v),
                dict(zip(IDS, map(float, n), strict=True)),
                metadata={"carrier_representation": "species_inventory"},
            )
            for name, v, n in zip(names, volumes, gas, strict=True)
        }
    )
    sensor = float(gas[cells - 1].sum() * R * case["temperature"] / (volumes[cells - 1] * 0.001))
    slots = fixed_bed_slots(
        spec,
        "assembly",
        initial_loading_mol_kg=[dict(zip(IDS[:2], case["loading"], strict=True))] * cells,
    )
    state = WorldState(
        phases.total_amounts_mol(),
        float(sum(volumes)),
        case["temperature"],
        sensor,
        "gas",
        "assembly",
        phases=phases,
        state_slots=StateSlots(slots),
        equipment=EquipmentLedger(
            {"bed": EquipmentRecord("bed", "fixed_bed", "assembly", batch_reset="preserve")}
        ),
        vessels=VesselLedger(
            {
                "assembly": VesselRecord(
                    "assembly", "fixed_bed_assembly", float(sum(volumes)) + 0.01, 430.0, 160000.0
                )
            }
        ),
    )
    return spec, state
