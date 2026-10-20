"""Explicit synthetic VLE data shared by the runnable example and checks."""

from math import log

from chemworld.physchem.flash_units import VLEPropertyPackage, ph_flash
from chemworld.physchem.property_packages import ComponentPropertyPackage
from chemworld.physchem.property_reports import R_J_PER_MOL_K, CaloricReference
from chemworld.physchem.specs import ComponentSpec, PropertyCorrelation


def synthetic_vle_properties():
    packages = []
    references = {}
    for name, formula, cp, latent, boiling in (
        ("light", "C2H6O", 80, 30000, 330), ("heavy", "C3H8O", 100, 40000, 370),
    ):
        def correlation(property_id, equation, coefficients, unit, name=name):
            return PropertyCorrelation(
                correlation_id=f"synthetic-vle:{name}:{property_id}", property_id=property_id,
                equation_id=equation, coefficients=coefficients,
                input_units={"temperature": "K"}, output_unit=unit,
                validity_ranges={"temperature": (250, 500)},
                source_note="Synthetic constant-latent-heat Raoult model; not real material data")
        packages.append(ComponentPropertyPackage(
            ComponentSpec(name, formula, metadata={"synthetic": True}),
            (correlation("ideal_gas_heat_capacity", "cp_polynomial", {"a": cp}, "J/(mol*K)"),
             correlation("vapor_pressure", "dippr101_vapor_pressure",
                         {"A": log(1e5) + latent / R_J_PER_MOL_K / boiling,
                          "B": -latent / R_J_PER_MOL_K, "C": 0, "D": 0, "E": 1}, "Pa"))))
        references[name] = CaloricReference(
            phase="gas", source_note="Synthetic ideal gas H=0 at 298.15 K")
    return VLEPropertyPackage(tuple(packages), references, (250, 500))


def run_example():
    from chemworld.runtime import ChemWorldRuntime
    from chemworld.world.vle_world import vle_world_spec

    properties = synthetic_vle_properties()
    result = ph_flash(synthetic_vle_properties(), {"light": .5, "heavy": .5},
                      pressure_Pa=1e5, enthalpy_J=-15000)
    session = ChemWorldRuntime(composition=vle_world_spec(properties))
    session.reset(seed=12)
    try:
        history = []
        for action in (
            {"operation": "add_component", "component": "mixture-feed", "amount_mol": 1e-4},
            {"operation": "heat", "target_temperature_K": 410, "duration_s": 20,
             "stirring_speed_rpm": 600},
            {"operation": "measure", "instrument": "phase_state"},
            {"operation": "route_material", "connection": "reactor-to-receiver",
             "transfer_fraction": .2, "mixing": 1},
            {"operation": "terminate"},
        ):
            step = session.execute(action)
            if step.info["transaction_status"] != "committed":
                raise RuntimeError(f"Flash example failed: {step.info['error_message']}")
            history.append({"action": action, "outcome": step.info["operation_outcome"],
                            "actual_duration_s": step.info["actual_duration_s"]})
        return {"PH_query": result.to_dict(), "steps": history, "ended": step.terminated,
                "reading": session.observe().to_dict(), "phases": session._state.phases.to_dict(),
                "physical_time_s": session._state.ledger.time_s}
    finally:
        session.close()


def main():
    import json

    print(json.dumps(run_example(), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
