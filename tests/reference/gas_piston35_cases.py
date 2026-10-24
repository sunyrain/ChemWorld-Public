"""Fixed authored 35E inputs shared as data, with no native-derived reference output."""

from copy import deepcopy
from dataclasses import replace

from chemworld.examples.gas_piston import example_inputs, mechanism_document
from chemworld.physchem.gas_network_energy_specs import GasThermalLinkSpec
from chemworld.physchem.heat_transfer_units import ThermalDeviceSpec

PHYSICAL = (
    "isothermal-closed",
    "thermal-closed",
    "isothermal-flow",
    "thermal-flow",
    "mass",
    "valve",
    "reverse-valve",
    "nonanchor",
    "heater",
    "cooling",
    "compression",
    "conductive",
)
EVENTS = ("maximum-volume", "minimum-volume", "temperature", "inventory")


def native(name):
    thermal = not name.startswith("isothermal") and name not in {
        "maximum-volume",
        "minimum-volume",
        "inventory",
        "compression",
    }
    connected = name not in {
        "isothermal-closed",
        "thermal-closed",
        "maximum-volume",
        "temperature",
        "heater",
        "cooling",
    }
    spec, network, amounts = example_inputs(thermal=thermal, connected=connected)
    device = None
    duration = 1.0
    if name in {"mass", "valve", "reverse-valve"}:
        mode = "mass-flow" if name == "mass" else "pressure-valve"
        edges = tuple(
            replace(e, mode=mode, coefficient=(1e-6 if mode == "mass-flow" else 2e-11))
            for e in spec.connections
        )
        if name == "reverse-valve":
            edges = tuple(
                replace(e, source_vessel_id=e.target_vessel_id, target_vessel_id=e.source_vessel_id)
                for e in edges
            )
        spec = replace(spec, connections=edges)
    if name == "nonanchor":
        spec = replace(spec, anchor_vessel_id="feed", temperature_K=700.0)
    if name in {"heater", "cooling"}:
        device = ThermalDeviceSpec(
            conductance_W_K=1.0,
            maximum_power_W=8.0,
            setpoint_K=1100.0 if name != "cooling" else 350.0,
            enabled=True,
            minimum_setpoint_K=300.0,
            maximum_setpoint_K=1200.0,
            maximum_cooling_power_W=6.0,
        )
    if name == "conductive":
        spec = replace(spec, heat_links=(GasThermalLinkSpec("heat", "feed", "piston", 0.01),))
    if name in {"minimum-volume", "compression"}:
        spec = replace(
            spec,
            nodes=tuple(replace(n, chemistry_enabled=False) for n in spec.nodes),
            connections=tuple(
                replace(e, coefficient=0.0 if e.connection_id == "inlet" else 0.002)
                for e in spec.connections
            ),
        )
    if name == "maximum-volume":
        spec = replace(spec, pressure_control=replace(spec.pressure_control, maximum_volume_L=1.08))
        duration = 4.0
    if name == "minimum-volume":
        spec = replace(spec, pressure_control=replace(spec.pressure_control, minimum_volume_L=0.97))
        duration = 2.0
    if name == "temperature":
        spec = replace(
            spec,
            energy_nodes=tuple(replace(n, minimum_temperature_K=598.0) for n in spec.energy_nodes),
        )
        duration = 2.0
    if name == "inventory":
        spec = replace(
            spec,
            nodes=tuple(
                replace(n, minimum_amount_mol=sum(amounts[n.vessel_id].values()) - 0.0003)
                if n.vessel_id == "feed"
                else n
                for n in spec.nodes
            ),
        )
        duration = 2.0
    return spec, network, amounts, device, duration


def card(name):
    spec, _, amounts, device, duration = native(name)
    energy = {n.vessel_id: n for n in spec.energy_nodes}
    return {
        "name": name,
        "duration_s": duration,
        "mechanism": deepcopy(mechanism_document()),
        "thermal": bool(energy),
        "anchor": spec.anchor_vessel_id,
        "piston": spec.pressure_control.to_dict(),
        "amounts": deepcopy(amounts),
        "nodes": [
            {
                "id": n.vessel_id,
                "volume_L": n.volume_L,
                "temperature_K": spec.initial_temperatures_K[n.vessel_id],
                "chemistry": n.chemistry_enabled,
                "support": energy[n.vessel_id].support_heat_capacity_J_K if energy else 0.0,
                "min_T": energy[n.vessel_id].minimum_temperature_K if energy else 200.0,
                "max_T": energy[n.vessel_id].maximum_temperature_K if energy else 2000.0,
                "min_n": n.minimum_amount_mol,
                "max_P": n.maximum_pressure_Pa,
            }
            for n in spec.nodes
        ],
        "edges": [e.__dict__ for e in spec.connections],
        "heat_links": [e.__dict__ for e in spec.heat_links],
        "device": device.to_dict() if device else None,
    }
