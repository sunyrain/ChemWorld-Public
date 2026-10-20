"""Explicit synthetic constitutive data for finite sorbent research."""

from chemworld.physchem.property_packages import PropertyPackageThermodynamics, SpeciesCaloricModel
from chemworld.physchem.property_reports import CaloricReference
from chemworld.physchem.sorbent_kinetics import SorbentKineticsSpec
from chemworld.physchem.specs import PropertyCorrelation


def synthetic_sorbent_spec():
    return SorbentKineticsSpec(
        "synthetic-sorbent",
        1.0,
        {"A": 1.0, "B": 3.0},
        {"A": 0.01, "B": 0.02},
        {"A": -10000.0, "B": -20000.0},
    )


def synthetic_sorbent_thermodynamics(spec):
    models = {}
    for key, cv, slope in (("A", 50.0, 0.02), ("B", 70.0, 0.03)):
        for bound in (False, True):
            identifier = f"{key}_ads" if bound else key
            models[identifier] = SpeciesCaloricModel(
                PropertyCorrelation(
                    identifier + ":synthetic-Cv",
                    "cp_polynomial",
                    {"a": cv - slope * 298.15, "b": slope},
                    {"temperature": "K"},
                    "J/(mol*K)",
                    property_id="isochoric_heat_capacity",
                    validity_ranges={"temperature": (280.0, 430.0)},
                    source_note="Synthetic condensed Cv; no Cp-to-Cv or gas conversion",
                ),
                CaloricReference(
                    phase="solid" if bound else "liquid",
                    energy_basis="internal_energy",
                    energy_J_mol=(spec.adsorption_energy_J_mol[key] if bound else 0.0),
                ),
            )
    return PropertyPackageThermodynamics(models, 20.0, 298.15)


def run_example():
    """Load, wash and warm the same carrier through existing runtime operations."""
    from chemworld.runtime import ChemWorldRuntime
    from chemworld.world.sorbent_world import sorbent_world_spec

    spec = synthetic_sorbent_spec()
    world = sorbent_world_spec(
        spec, synthetic_sorbent_thermodynamics(spec), {"A": "C2H6O", "B": "C3H8O"})
    session = ChemWorldRuntime(composition=world)
    session.reset(seed=22)
    actions = [
        {"operation": "dry"},
        {"operation": "route_material", "connection": "reactor-to-waste",
         "transfer_fraction": 1.0, "mixing": 1},
        {"operation": "add_solvent", "solvent": 0, "volume_L": 0.01},
        {"operation": "heat", "target_temperature_K": 350, "duration_s": 60,
         "stirring_speed_rpm": 600},
        {"operation": "measure", "instrument": "phase_state"},
        {"operation": "pause_control"},
        {"operation": "wait", "duration_s": 10, "stirring_speed_rpm": 600},
        {"operation": "terminate"},
    ]
    try:
        for action in actions:
            result = session.execute(action)
            if result.info["transaction_status"] != "committed":
                raise RuntimeError(result.info)
        state = session._state
        return {"classification": "synthetic development example",
                "actions": len(actions), "physical_time_s": state.ledger.time_s,
                "temperature_K": state.temperature_K,
                "fluid_moles": dict(state.species_amounts),
                "owned_carrier": state.state_slots.slots[0].to_dict(),
                "remaining_wash_L": state.inactive_vessels["wash_supply"].volume_L,
                "carrier_packets": state.metadata["carrier_packet_history"]}
    finally:
        session.close()


if __name__ == "__main__":
    import json

    print(json.dumps(run_example(), indent=2, allow_nan=False))
