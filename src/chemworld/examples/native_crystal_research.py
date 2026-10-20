"""Authored synthetic crystal phase and caloric inputs, without reference libraries."""

from chemworld.physchem.crystal_thermal_kinetics import CrystalThermalSpec
from chemworld.physchem.crystallization_units import (
    CrystallizationKineticsSpec,
    SolubilityCurveSpec,
)
from chemworld.physchem.property_packages import PropertyPackageThermodynamics, SpeciesCaloricModel
from chemworld.physchem.property_reports import CaloricReference
from chemworld.physchem.specs import PropertyCorrelation


def synthetic_crystal_model(molecular_weight_kg_mol, *, nucleation=2e7):
    return CrystalThermalSpec(
        "P",
        "I",
        SolubilityCurveSpec("synthetic-Cs", 1.0, 298.15, 20000, 250, 430, "synthetic-liquid-solid"),
        CrystallizationKineticsSpec(
            "synthetic-growth",
            nucleation,
            2,
            2e-8,
            1,
            1200,
            molecular_weight_kg_mol,
            8e-6,
            impurity_occlusion_mol_per_mol=0.02,
            supersaturation_occlusion_factor=0.5,
            provenance_id="synthetic-liquid-solid",
        ),
        {"P": "solid:P", "I": "solid:I"},
    )


def synthetic_crystal_thermodynamics():
    properties = {}
    for name, cv, energy, phase in (
        ("P", 100, 0, "liquid"),
        ("I", 120, 0, "liquid"),
        ("solid:P", 100, -20000, "solid"),
        ("solid:I", 120, 0, "solid"),
    ):
        properties[name] = SpeciesCaloricModel(
            PropertyCorrelation(
                name + ":synthetic-Cv",
                "cp_polynomial",
                {"a": cv},
                {"temperature": "K"},
                "J/(mol*K)",
                property_id="isochoric_heat_capacity",
                validity_ranges={"temperature": (250, 430)},
                source_note="Synthetic Cv, no gas conversion",
            ),
            CaloricReference(phase=phase, energy_basis="internal_energy", energy_J_mol=energy),
        )
    return PropertyPackageThermodynamics(properties, 20, 298.15)


def run_example():
    """Advance a finite slurry, retain its filtrate, and warm its wet crystals."""
    from chemworld.foundation import equipment_settings
    from chemworld.physchem.elements import molecular_weight
    from chemworld.runtime import ChemWorldRuntime
    from chemworld.world.crystal_world import crystal_world_spec

    model = synthetic_crystal_model(molecular_weight({"C": 6, "H": 12, "O": 1}) / 1000)
    world = crystal_world_spec(
        model, synthetic_crystal_thermodynamics(), {"P": "C6H12O", "I": "C3H8O"}
    )
    session = ChemWorldRuntime(composition=world)
    session.reset(seed=23)
    actions = [
        {"operation": "seed_crystals", "seed_mass_g": 0.001},
        {"operation": "cool_crystallize", "target_temperature_K": 278.15, "duration_s": 300},
        {"operation": "measure", "instrument": "phase_state"},
        {"operation": "filter_crystals"},
        {"operation": "resuspend_crystals", "volume_L": 0.01, "solvent": 0},
        {
            "operation": "heat",
            "target_temperature_K": 350,
            "duration_s": 60,
            "stirring_speed_rpm": 600,
        },
        {"operation": "wait", "duration_s": 60, "stirring_speed_rpm": 600},
        {"operation": "terminate"},
    ]
    try:
        for action in actions:
            result = session.execute(action)
            if result.info["transaction_status"] != "committed":
                raise RuntimeError(result.info)
        state = session._state
        return {
            "classification": "synthetic development example",
            "actions": len(actions),
            "physical_time_s": state.ledger.time_s,
            "actual_temperature_K": state.temperature_K,
            "actual_phases": state.phases.to_dict(),
            "cohort_count": len(
                equipment_settings(state.equipment, "crystallizer")["population_cohorts"]
            ),
            "retained_filtrate_count": len(state.samples.samples),
            "energy_jacket_J": state.ledger.energy_jacket_J,
            "heat_loss_J": state.ledger.heat_loss_J,
        }
    finally:
        session.close()


if __name__ == "__main__":
    import json

    print(json.dumps(run_example(), indent=2, allow_nan=False))
