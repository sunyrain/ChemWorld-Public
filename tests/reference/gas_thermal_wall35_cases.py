"""Fixed authored inputs for the 35F development block."""

from dataclasses import replace

from chemworld.examples.gas_thermal_wall import example_inputs
from chemworld.physchem.gas_thermal_wall import GasWallContact
from chemworld.physchem.heat_transfer_units import ThermalDeviceSpec

PHYSICAL = (
    "one-insulated",
    "two-insulated",
    "grey-cooling",
    "grey-heating",
    "black-wall",
    "zero-emissivity",
    "reactive-closed",
    "finite-flow",
    "stiff-contact",
    "heater-pulse",
    "wall-low-event",
    "wall-high-event",
)


def native(name):
    if name not in PHYSICAL:
        raise ValueError(name)
    connected = name in {"two-insulated", "finite-flow"}
    reactive = name in {"reactive-closed", "finite-flow"}
    spec, network, amounts = example_inputs(connected=connected, reactive=reactive)
    wall = spec.thermal_wall
    segments = [(3.0, None)]
    if name in {"one-insulated", "two-insulated"}:
        wall = replace(wall, emissivity=0.0, loss_conductance_W_K=0.0)
        spec = replace(spec, connections=())
    if name in {"grey-heating", "wall-high-event"}:
        wall = replace(wall, initial_temperature_K=400.0, environment_temperature_K=900.0)
    if name == "black-wall":
        wall = replace(wall, emissivity=1.0)
    if name == "zero-emissivity":
        wall = replace(wall, emissivity=0.0)
    if reactive:
        segments = [(0.8, None)]
    if name == "stiff-contact":
        wall = replace(wall, heat_capacity_J_K=0.02, contacts=(GasWallContact("reactor", 150.0),))
        segments = [(0.3, None)]
    if name == "heater-pulse":
        wall = replace(
            wall, initial_temperature_K=350.0, contacts=(GasWallContact("reactor", 0.3),)
        )
        heater = ThermalDeviceSpec(
            conductance_W_K=10.0,
            maximum_power_W=50.0,
            setpoint_K=1000.0,
            minimum_setpoint_K=250.0,
            maximum_setpoint_K=1200.0,
        )
        segments = [(0.3, heater), (0.5, None)]
    if name == "wall-low-event":
        wall = replace(wall, minimum_temperature_K=790.0)
    if name == "wall-high-event":
        wall = replace(wall, maximum_temperature_K=410.0)
    return replace(spec, thermal_wall=wall), network, amounts, segments


def card(name):
    spec, _, amounts, segments = native(name)
    return {
        "name": name,
        "nodes": [n.vessel_id for n in spec.nodes],
        "amounts": amounts,
        "temperatures": spec.initial_temperatures_K,
        "volumes_L": {n.vessel_id: n.volume_L for n in spec.nodes},
        "supports": {n.vessel_id: 0.5 for n in spec.nodes},
        "reactive_nodes": ["reactor"] if name in {"reactive-closed", "finite-flow"} else [],
        "flows": [
            (e.source_vessel_id, e.target_vessel_id, e.coefficient) for e in spec.connections
        ],
        "wall": spec.thermal_wall.to_dict(),
        "segments": [
            (duration, 0.0 if heater is None else heater.maximum_power_W)
            for duration, heater in segments
        ],
    }
