"""Explicit synthetic volatile/liquid-only caloric data for concentration."""

from chemworld.examples.native_flash_research import synthetic_vle_properties
from chemworld.physchem.flash_units import VLEPropertyPackage
from chemworld.physchem.property_packages import ComponentPropertyPackage
from chemworld.physchem.property_reports import CaloricReference
from chemworld.physchem.specs import ComponentSpec, PropertyCorrelation


def synthetic_concentration_properties(pressure_Pa=1e5, *, binary_volatile=False):
    volatile = synthetic_vle_properties()
    packages = volatile.packages if binary_volatile else volatile.packages[:1]
    target = ComponentPropertyPackage(
        ComponentSpec("target", "C6H12O6", metadata={"synthetic": True}),
        (
            PropertyCorrelation(
                "synthetic-concentration:target:cp",
                "cp_polynomial",
                {"a": 120},
                {"temperature": "K"},
                "J/(mol*K)",
                property_id="liquid_heat_capacity",
                validity_ranges={"temperature": (250, 500)},
                source_note="Synthetic liquid-only target; no gas or vapor-pressure parameters",
            ),
        ),
    )
    return VLEPropertyPackage(
        (*packages, target),
        {p.component.identifier: volatile.gas_references[p.component.identifier] for p in packages},
        (250, 500),
        nonvolatile_references={
            "target": CaloricReference(
                phase="liquid",
                pressure_Pa=pressure_Pa,
                source_note="Synthetic pressure-specific liquid H=0 at 298.15 K; zero-volume model",
            )
        },
    )


def synthetic_concentration_spec(pressure_Pa=1e5, *, binary_volatile=False):
    from chemworld.physchem.concentration_units import VacuumConcentratorSpec
    from chemworld.physchem.coupled_concentration import CoupledConcentrationSpec

    properties = synthetic_concentration_properties(pressure_Pa, binary_volatile=binary_volatile)
    equipment = VacuumConcentratorSpec(
        "synthetic-finite-concentrator",
        0.1,
        1e-5,
        10,
        0.01,
        0.96,
        5000,
        430,
        3600,
        "synthetic-state-dependent-concentration-v1",
    )
    return CoupledConcentrationSpec(
        properties,
        equipment,
        tuple(properties.gas_references),
        target_component_id="target",
        pressure_Pa=pressure_Pa,
    )


def initial_concentration_state(spec):
    from chemworld.physchem.coupled_concentration import CoupledConcentrationState

    amounts = dict.fromkeys(spec.properties.component_ids, 0.0)
    amounts["light"] = 0.012 if "heavy" in amounts else 0.02
    if "heavy" in amounts:
        amounts["heavy"] = 0.008
    amounts["target"] = 0.002
    empty = dict.fromkeys(amounts, 0.0)
    return CoupledConcentrationState(
        (amounts, empty),
        (spec.at_tp(0, amounts, 295).total_enthalpy_j, spec.at_tp(1, empty, 270).total_enthalpy_j),
    )


def run_example():
    from chemworld.runtime import ChemWorldRuntime
    from chemworld.world.concentration_world import concentration_world_spec

    spec = synthetic_concentration_spec(30000, binary_volatile=True)
    session = ChemWorldRuntime(
        composition=concentration_world_spec(spec, initial_concentration_state(spec))
    )
    session.reset(seed=21)
    try:
        steps = []
        for action in (
            {"operation": "concentrate", "duration_s": 10},
            {"operation": "measure", "instrument": "phase_state"},
            {"operation": "pause_control"},
            {"operation": "wait", "duration_s": 2, "stirring_speed_rpm": 600},
            {"operation": "terminate"},
        ):
            result = session.execute(action)
            if result.info["transaction_status"] != "committed":
                raise RuntimeError(f"Concentration example failed: {result.info['error_message']}")
            steps.append(
                {
                    "action": action,
                    "outcome": result.info["operation_outcome"],
                    "actual_duration_s": result.info["actual_duration_s"],
                }
            )
        return {
            "steps": steps,
            "ended": result.terminated,
            "physical_time_s": session._state.ledger.time_s,
            "phases": session._state.phases.to_dict(),
            "outlet_integral": session._state.metadata["concentration_outlet_integral"],
            "advance_receipt": session._state.metadata["last_concentration_advance"],
            "reading": session.observe().to_dict(),
        }
    finally:
        session.close()


def main():
    import json

    print(json.dumps(run_example(), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
