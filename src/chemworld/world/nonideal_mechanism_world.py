"""A finite homogeneous reacting phase using the original thermal world."""

from collections.abc import Mapping

from chemworld.foundation import WorldState
from chemworld.foundation.lab_clock import LabClockSpec
from chemworld.foundation.state_ledgers import (
    EquipmentLedger,
    EquipmentRecord,
    VesselLedger,
    VesselRecord,
)
from chemworld.physchem.coupled_system import CoupledState
from chemworld.physchem.heat_transfer_units import ThermalDeviceSpec, ThermalSystemSpec
from chemworld.physchem.molecular_eos import ResidualHelmholtzModel
from chemworld.physchem.molecular_mixture import MolecularIsochoricThermodynamics
from chemworld.physchem.molecular_reaction import (
    mechanism_caloric_model,
    molecular_reaction_process,
)
from chemworld.physchem.reaction_network import ReactionNetworkSpec
from chemworld.world.composition import WorldCompositionSpec
from chemworld.world.physical_world import CompositionPhysics


def nonideal_mechanism_world(
    network: ReactionNetworkSpec,
    residual_model: ResidualHelmholtzModel,
    amounts_mol: Mapping[str, float],
    *,
    temperature_K: float = 600.0,
    volume_L: float = 1.0,
    minimum_temperature_K: float = 400.0,
    maximum_temperature_K: float = 1400.0,
    maximum_pressure_Pa: float = 1e8,
    support_heat_capacity_J_K: float = 0.0,
    thermal_device: ThermalDeviceSpec | None = None,
    budget: int = 30,
    shared_clock: bool = False,
) -> WorldCompositionSpec:
    """Author explicit PR/source thermo; no critical-data inference or phase selection.

    The normal heat/wait/measure operations and shared clock consume one finite
    phase inventory. Only homogeneous local stability is checked; the caller
    chooses the phase's physical validity domain.
    """
    thermo = MolecularIsochoricThermodynamics(
        mechanism_caloric_model(network, residual_model),
        volume_L,
        support_heat_capacity_J_K=support_heat_capacity_J_K,
    )
    molecular_reaction_process(network, thermo)
    if set(amounts_mol) - set(network.species_ids):
        raise ValueError("unknown initial molecular species")
    amounts = {key: float(amounts_mol.get(key, 0)) for key in network.species_ids}
    pressure = thermo.pressure_pa(CoupledState(amounts, temperature_K, volume_L))
    device = thermal_device or ThermalDeviceSpec(
        setpoint_K=temperature_K,
        minimum_setpoint_K=minimum_temperature_K,
        maximum_setpoint_K=maximum_temperature_K,
        maximum_power_W=20.0,
        conductance_W_K=0.5,
        enabled=False,
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
                    maximum_pressure_Pa,
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
    thermal = ThermalSystemSpec(
        thermo,
        pressure_law="molecular",
        minimum_temperature_K=minimum_temperature_K,
        maximum_temperature_K=maximum_temperature_K,
        stop_temperature_K=maximum_temperature_K - 1,
        maximum_pressure_Pa=maximum_pressure_Pa,
        loss_conductance_W_K=0,
    )
    return WorldCompositionSpec.from_dict(
        {
            "schema_version": "chemworld-world-composition-0.1",
            "composition_id": f"nonideal-mechanism:{network.network_id}",
            "vessels": [{"id": "reactor", "capacity_L": volume_L}],
            "components": [
                {"kind": "reaction", "parameters": {}},
                {
                    "kind": "thermal",
                    "parameters": {
                        "temperature_range_K": [minimum_temperature_K, maximum_temperature_K],
                        "duration_range_s": [1e-6, 3600],
                    },
                },
                *(
                    [{"kind": "lab_schedule", "parameters": {"duration_range_s": [1e-6, 3600]}}]
                    if shared_clock
                    else []
                ),
                *(
                    [{"kind": "observation", "parameters": {"instruments": ["gas_pressure"]}}]
                    if not shared_clock
                    else []
                ),
            ],
            "physics": CompositionPhysics(
                network,
                {"reactor": state},
                models={"thermal": {"reactor": thermal}},
                lab_clock=(
                    LabClockSpec(heater_ids=("heater",), analyzers=(), vessel_capacity=1)
                    if shared_clock
                    else None
                ),
            ).to_dict(),
            "task": {
                "objective": "study a finite nonideal reacting phase",
                "budget": budget,
                "operations": (
                    ["queue_lab_heat", "wait", "terminate"]
                    if shared_clock
                    else ["heat", "wait", "measure", "terminate"]
                ),
                "instruments": [] if shared_clock else ["gas_pressure"],
            },
        }
    )
