"""Author a finite imported gas batch using the shared thermal apparatus."""

from chemworld.foundation import WorldState
from chemworld.foundation.state_ledgers import (
    EquipmentLedger,
    EquipmentRecord,
    VesselLedger,
    VesselRecord,
)
from chemworld.physchem.gas_mechanism_batch import IdealGasThermodynamics
from chemworld.physchem.heat_transfer_units import ThermalDeviceSpec, ThermalSystemSpec
from chemworld.physchem.reaction_network import R_J_PER_MOL_K
from chemworld.world.composition import WorldCompositionSpec
from chemworld.world.physical_world import CompositionPhysics


def gas_mechanism_world(
    network,
    amounts_mol,
    *,
    temperature_K=700.0,
    volume_L=0.01,
    minimum_temperature_K=600.0,
    maximum_temperature_K=1800.0,
):
    """Initial inventory is authored explicitly; no upstream state is inferred."""
    thermo = IdealGasThermodynamics(network.imported_gas_thermochemistry)
    amounts = {key: float(amounts_mol.get(key, 0)) for key in network.species_ids}
    if set(amounts_mol) - set(amounts):
        raise ValueError("unknown initial gas species")
    pressure = sum(amounts.values()) * R_J_PER_MOL_K * temperature_K * 1000 / volume_L
    device = ThermalDeviceSpec(
        setpoint_K=temperature_K,
        minimum_setpoint_K=minimum_temperature_K,
        maximum_setpoint_K=maximum_temperature_K,
        maximum_power_W=5,
        conductance_W_K=0.2,
    )
    state = WorldState(
        amounts,
        volume_L,
        temperature_K,
        pressure,
        "gas",
        "reactor",
        units={"amount": "mol", "volume": "L", "temperature": "K", "pressure": "Pa", "time": "s"},
        vessels=VesselLedger(
            {
                "reactor": VesselRecord(
                    "reactor",
                    "sealed",
                    volume_L,
                    maximum_temperature_K,
                    1e7,
                    ("gas",),
                    temperature_K,
                    pressure,
                )
            }
        ),
        equipment=EquipmentLedger(
            {
                "heater": EquipmentRecord(
                    "heater",
                    "thermal_device",
                    attached_vessel_id="reactor",
                    settings={"thermal_device": device.to_dict()},
                )
            }
        ),
    )
    model = ThermalSystemSpec(
        thermo,
        minimum_temperature_K=minimum_temperature_K,
        maximum_temperature_K=maximum_temperature_K,
        stop_temperature_K=maximum_temperature_K - 1,
        maximum_pressure_Pa=1e7,
        loss_conductance_W_K=0,
    )
    # The existing gas_composition instrument is specifically CO/O2/CO2;
    # it is not a generic assay for an arbitrary imported mechanism.
    instruments = ["gas_pressure"]
    return WorldCompositionSpec.from_dict(
        {
            "schema_version": "chemworld-world-composition-0.1",
            "composition_id": f"gas-mechanism:{network.network_id}",
            "vessels": [{"id": "reactor", "capacity_L": volume_L}],
            "components": [
                {"kind": "reaction", "parameters": {}},
                {
                    "kind": "thermal",
                    "parameters": {
                        "temperature_range_K": [minimum_temperature_K, maximum_temperature_K],
                        "duration_range_s": [1, 2],
                    },
                },
                {"kind": "observation", "parameters": {"instruments": instruments}},
            ],
            "physics": CompositionPhysics(
                network, {"reactor": state}, models={"thermal": {"reactor": model}}
            ).to_dict(),
            "task": {
                "objective": "characterize finite gas mechanism",
                "budget": 20,
                "operations": ["heat", "wait", "measure", "terminate"],
                "instruments": instruments,
            },
        }
    )
