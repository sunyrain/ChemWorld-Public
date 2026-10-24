"""Fixed two-vessel originals for finite humid-air material acceptance."""

from dataclasses import replace

from tests.test_humid_air_volume import thermal_inputs

from chemworld.foundation import WorldState
from chemworld.foundation.state_ledgers import (
    EquipmentLedger,
    EquipmentRecord,
    VesselLedger,
    VesselRecord,
)
from chemworld.physchem.coupled_system import CoupledState
from chemworld.physchem.heat_transfer_units import ThermalDeviceSpec, ThermalSystemSpec
from chemworld.physchem.reaction_network import ReactionNetworkSpec, SpeciesSpec
from chemworld.world.physical_world import CompositionPhysics


def world(phase, outlet):
    dew = 300.0 if phase == "liquid" else 240.0
    thermal, saturated = thermal_inputs(dew, 101325.0, phase, extra=0)
    low, high, stop = (270.0, 473.0, 460.0) if phase == "liquid" else (193.0, 273.16, 272.0)
    states, models, components = {}, {}, []
    for name, shift, water_factor in (("reactor", 10.0, 1.0), ("receiver", 5.0, 0.25)):
        thermodynamics = replace(
            thermal,
            support_heat_capacity_J_K=1.0,
            material_outlet_phase=outlet if name == "reactor" else None,
        )
        model = ThermalSystemSpec(
            thermodynamics,
            pressure_law="humid_air",
            loss_conductance_W_K=0.0,
            minimum_temperature_K=low,
            maximum_temperature_K=high,
            stop_temperature_K=stop,
            maximum_pressure_Pa=5e6,
        )
        n = {**saturated.amounts_mol, "water": saturated.amounts_mol["water"] * water_factor}
        initial = CoupledState(n, dew + shift, thermal.volume_L)
        p = thermodynamics.pressure_pa(initial)
        heater = name + ":heater"
        device = ThermalDeviceSpec(
            conductance_W_K=100.0,
            maximum_power_W=2.0,
            maximum_cooling_power_W=2.0,
            setpoint_K=dew + shift,
            enabled=False,
            minimum_setpoint_K=low,
            maximum_setpoint_K=high,
        )
        states[name] = WorldState(
            n,
            initial.volume_L,
            initial.temperature_K,
            p,
            "fluid",
            name,
            units={
                "amount": "mol",
                "volume": "L",
                "temperature": "K",
                "pressure": "Pa",
                "time": "s",
                "cost": "currency",
                "risk": "risk",
            },
            equipment=EquipmentLedger(
                {
                    heater: EquipmentRecord(
                        heater,
                        "thermal_device",
                        attached_vessel_id=name,
                        settings={"thermal_device": device.to_dict()},
                    )
                }
            ),
            vessels=VesselLedger(
                {
                    name: VesselRecord(
                        name,
                        "humid-air-vessel",
                        initial.volume_L,
                        high,
                        5e6,
                        ("fluid",),
                        initial.temperature_K,
                        p,
                    )
                }
            ),
        )
        models[name] = model
        components += [
            {"id": name + ":thermal", "kind": "thermal", "vessel": name, "parameters": {}},
            {"id": name + ":material", "kind": "material_input", "vessel": name, "parameters": {}},
        ]
    network = ReactionNetworkSpec(
        "humid-material-stock",
        tuple(SpeciesSpec(c.identifier, c.formula, phase="gas") for c in thermal.components),
        (),
    )
    payload = {
        "schema_version": "chemworld-world-composition-0.1",
        "composition_id": "humid-air-finite-material",
        "components": components,
        "vessels": [{"id": k, "capacity_L": thermal.volume_L} for k in states],
        "connections": [
            {
                "id": "reactor-to-receiver",
                "unit": "mol",
                "source": {"component": "reactor:material", "port": "out"},
                "target": {"component": "receiver:material", "port": "in"},
            }
        ],
        "physics": CompositionPhysics(network, states, models={"thermal": models}).to_dict(),
        "task": {
            "objective": "mix selected humid phases and separate condensed water",
            "budget": 40,
            "operations": [
                "heat",
                "wait",
                "pause_control",
                "resume_control",
                "route_material",
                "select_vessel",
                "terminate",
            ],
            "instruments": [],
        },
    }
    return payload, models, dew


def packed(state):
    return CoupledState(state.species_amounts, state.temperature_K, state.volume_L)


def record(state, models):
    locals_ = {state.vessel_id: state, **state.inactive_vessels}
    rows = {}
    for name, local in locals_.items():
        thermo = models[name].thermodynamics
        point = thermo.equilibrium_state(packed(local))
        rows[name] = {
            "amounts_mol": local.species_amounts,
            "temperature_K": local.temperature_K,
            "volume_L": local.volume_L,
            "pressure_Pa": local.pressure_Pa,
            "energy_J": thermo.internal_energy_j(packed(local)),
            "equilibrium": point.to_dict(),
            "thermodynamics": thermo.to_dict(),
        }
    return {
        "vessels": rows,
        "time_s": state.ledger.time_s,
        "jacket_heat_J": state.ledger.energy_jacket_J,
        "cost": state.ledger.cost,
    }
