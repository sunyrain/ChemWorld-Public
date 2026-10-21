"""Authored synthetic closed-system combinations on the existing laboratory clock."""

from copy import deepcopy

from chemworld.examples.native_domains import _rename, mixed_clock_world
from chemworld.foundation.lab_clock import LabClockSpec
from chemworld.world.composition import WorldCompositionSpec


def _fragment(world, old, name):
    payload = _rename(world.to_dict(), old, name)
    payload["vessels"] = [v for v in payload["vessels"] if v["id"] == name]
    payload["components"] = [c for c in payload["components"] if c["vessel"] == name]
    physics = payload["physics"]
    physics["initial_states"] = {name: physics["initial_states"][name]}
    equipment = physics["initial_states"][name]["equipment"]
    for identity, record in list(equipment.items()):
        if record["equipment_type"] == "thermal_device" and identity == "heater":
            del equipment[identity]
            record["equipment_id"] = name + ":heater"
            equipment[record["equipment_id"]] = record
    for model_id, descriptor in physics["models"].items():
        if model_id in {"thermal", "crystal"}:
            descriptor["bindings"] = {
                key: value for key, value in descriptor["bindings"].items() if key == name
            }
    return payload


def _combine(fragments, *, composition_id, heater_id, analyzer=None):
    """Merge declared recipes, refusing conflicting chemical/model identities."""
    payload = deepcopy(fragments[0])
    physics = payload["physics"]
    for fragment in fragments[1:]:
        other = fragment["physics"]
        payload["vessels"] += deepcopy(fragment["vessels"])
        payload["components"] += deepcopy(fragment["components"])
        if set(physics["initial_states"]) & set(other["initial_states"]):
            raise ValueError("closed laboratory fragments need distinct vessel identities")
        physics["initial_states"].update(deepcopy(other["initial_states"]))
        species = {s["species_id"]: s for s in physics["network"]["species"]}
        for candidate in other["network"]["species"]:
            prior = species.get(candidate["species_id"])
            if prior is None:
                physics["network"]["species"].append(deepcopy(candidate))
            elif any(prior.get(key) != candidate.get(key)
                     for key in ("formula", "phase", "charge")):
                raise ValueError("conflicting chemical identities in closed laboratory")
        if other["network"]["reactions"]:
            raise ValueError("these authored fragments provide kinetics in the first recipe")
        for model_id, descriptor in other["models"].items():
            if model_id not in physics["models"]:
                physics["models"][model_id] = deepcopy(descriptor)
            else:
                target = physics["models"][model_id]
                if (target["version"] != descriptor["version"]
                        or set(target["bindings"]) & set(descriptor["bindings"])):
                    raise ValueError("conflicting model bindings in closed laboratory")
                target["bindings"].update(deepcopy(descriptor["bindings"]))
    payload["composition_id"] = composition_id
    payload["connections"] = []
    physics["feed_sources"] = {}
    for state in physics["initial_states"].values():
        for identity, record in list(state["equipment"].items()):
            if record["equipment_type"] == "thermal_device":
                if identity != heater_id:
                    del state["equipment"][identity]
                else:
                    record["settings"]["thermal_device"]["enabled"] = False
    for vessel in physics["initial_states"]:
        if not any(c["vessel"] == vessel and c["kind"] == "lab_schedule"
                   for c in payload["components"]):
            payload["components"].append({
                "id": vessel + ":shared-clock", "vessel": vessel, "kind": "lab_schedule",
                "parameters": {"duration_range_s": [1e-6, 14400]},
            })
    physics["lab_clock"] = LabClockSpec(
        heater_id=heater_id, analyzer=analyzer,
        vessel_capacity=len(physics["initial_states"]),
    ).to_dict()
    payload["task"].update(
        budget=120, objective="characterize declared independent synthetic systems",
        operations=["select_vessel", "queue_lab_heat", "wait", "terminate"]
        + (["queue_lab_measure"] if analyzer else []),
        instruments=[] if analyzer is None else [analyzer],
    )
    return WorldCompositionSpec.from_dict(payload)


def reactive_closed_lab(*, battery_spec=None):
    """Thermal A/B reaction, thermal SPM, finite adsorption and independent SSA.

    A/B are the same synthetic CH2O isomer counters in the reaction and contact
    vessels; the separately authored caloric/contact parameters are not fitted
    material data. The SPM lithium and adsorbed stocks retain their own owners.
    """
    from chemworld.examples.native_sorbent_research import (
        synthetic_sorbent_spec,
        synthetic_sorbent_thermodynamics,
    )
    from chemworld.physchem.battery_models import BatterySPMSpec, BatteryThermalSpec
    from chemworld.physchem.reactor_shared import ScalarControlProgram
    from chemworld.world.battery_world import battery_world_spec
    from chemworld.world.sorbent_world import sorbent_world_spec

    base = _rename(_rename(mixed_clock_world().to_dict(), "reactor", "ssa"),
                   "continuous", "reactor")
    sorbent = synthetic_sorbent_spec()
    contact = _fragment(sorbent_world_spec(
        sorbent, synthetic_sorbent_thermodynamics(sorbent), {"A": "CH2O", "B": "CH2O"},
    ), "reactor", "adsorber")
    contact["physics"]["models"]["thermal"]["bindings"]["adsorber"][
        "condensed_background"
    ]["source_vessel_id"] = None
    battery = _fragment(battery_world_spec(
        battery_spec or BatterySPMSpec(thermal=BatteryThermalSpec()),
    ), "reactor", "cell")
    battery["physics"]["initial_states"]["cell"]["equipment"]["battery"]["settings"][
        "electrical_control"
    ]["program"] = ScalarControlProgram(((0, .02),), "A").to_dict()
    return _combine(
        (base, contact, battery), composition_id="synthetic-reactive-closed-lab",
        heater_id="adsorber:heater", analyzer="molecule_capture",
    )


def phase_closed_lab():
    """Existing finite crystal growth and Al/Cu solid equilibrium in separate vessels."""
    from chemworld.examples.native_crystal_research import (
        synthetic_crystal_model,
        synthetic_crystal_thermodynamics,
    )
    from chemworld.physchem.elements import molecular_weight
    from chemworld.world.crystal_world import crystal_world_spec
    from chemworld.world.solid_world import solid_world_spec

    crystal = _fragment(crystal_world_spec(
        synthetic_crystal_model(molecular_weight({"C": 6, "H": 12, "O": 1}) / 1000),
        synthetic_crystal_thermodynamics(), {"P": "C6H12O", "I": "C3H8O"},
    ), "reactor", "crystallizer")
    solid = _fragment(solid_world_spec(), "reactor", "specimen")
    return _combine(
        (crystal, solid), composition_id="synthetic-two-phase-models-closed-lab",
        heater_id="specimen:heater",
    )
