"""Requested properties and a native reactor with explicitly varying Cv.

Run ``python -m chemworld.examples.state_property_research`` after installation.
The gas Cp table is the existing curated model. The runtime demonstration uses
synthetic Cv(T)=Cv350*(1/2+T/700 K), retaining the native U datum at 350 K.
"""

import json
from dataclasses import replace

from chemworld.physchem.curated_properties import curated_property_package
from chemworld.physchem.property_packages import PropertyPackageThermodynamics, SpeciesCaloricModel
from chemworld.physchem.property_reports import CaloricReference
from chemworld.physchem.specs import PropertyCorrelation
from chemworld.runtime import ChemWorldRuntime
from chemworld.world.surface_world import surface_world_spec


def variable_caloric_world():
    """Retain the existing apparatus/chemistry and declare a synthetic caloric model."""
    spec = surface_world_spec(include_surface=False, include_bulk=True)
    models = {}
    for vessel, model in spec.physics.models_of("thermal").items():
        old = model.thermodynamics
        species = {
            key: SpeciesCaloricModel(
                PropertyCorrelation(
                    correlation_id=f"synthetic-variable-cv:{key}",
                    property_id="isochoric_heat_capacity", equation_id="cp_polynomial",
                    coefficients={"a": value.cv_J_mol_K / 2, "b": value.cv_J_mol_K / 700},
                    input_units={"temperature": "K"}, output_unit="J/(mol*K)",
                    validity_ranges={"temperature": (250, 800)},
                    source_note="Synthetic Cv(T)=Cv350*(1/2+T/700 K)",
                ),
                CaloricReference(
                    temperature_K=old.reference_temperature_K, energy_basis="internal_energy",
                    energy_J_mol=value.u_ref_J_mol,
                    source_note="Retained authored native U datum at 350 K",
                ),
            ) for key, value in old.species.items()
        }
        models[vessel] = replace(model, thermodynamics=PropertyPackageThermodynamics(
            species, old.support_heat_capacity_J_K, old.reference_temperature_K))
    return replace(
        spec, physics=replace(spec.physics, models={**spec.physics.models, "thermal": models})
    )


def run_example():
    pure = curated_property_package("7732-18-5").evaluate_state(
        temperature_K=400, pressure_Pa=1000, phase="gas",
        quantities=("cp", "cv", "enthalpy", "internal_energy", "density"))
    session = ChemWorldRuntime(composition=variable_caloric_world())
    session.reset(seed=12)
    history = []
    try:
        for action in (
            {"operation": "add_component", "component": "CO-feed", "amount_mol": 6e-5},
            {"operation": "add_component", "component": "O2-feed", "amount_mol": 3e-5},
            {"operation": "heat", "target_temperature_K": 380, "duration_s": 20,
             "stirring_speed_rpm": 600},
            {"operation": "measure", "instrument": "gas_composition"},
            {"operation": "terminate"},
        ):
            result = session.execute(action)
            if result.info["transaction_status"] != "committed":
                raise RuntimeError(f"Property example could not commit {action['operation']}")
            history.append({"action": action, "outcome": result.info["operation_outcome"],
                            "duration_s": result.info["actual_duration_s"]})
        return {"pure_gas_properties": pure.to_dict(), "steps": history,
                "readings": session.observe().to_dict(), "ended": result.terminated,
                "physical_time_s": session._state.ledger.time_s}
    finally:
        session.close()


def main() -> None:
    print(json.dumps(run_example(), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
