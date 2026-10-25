"""Declared original finite heaters with continuous feedback on real temperature."""

from chemworld.foundation import WorldState
from chemworld.foundation.lab_clock import LabClockSpec
from chemworld.foundation.state_ledgers import (
    EquipmentLedger,
    EquipmentRecord,
    VesselLedger,
    VesselRecord,
)
from chemworld.physchem.coupled_system import ConstantCvThermodynamics, SpeciesInternalEnergy
from chemworld.physchem.heat_transfer_units import ThermalDeviceSpec, ThermalSystemSpec
from chemworld.physchem.reaction_network import ReactionNetworkSpec, SpeciesSpec
from chemworld.physchem.thermal_feedback import ThermalFeedbackSpec
from chemworld.runtime.thermal_feedback import install_thermal_feedback
from chemworld.world.composition import WorldCompositionSpec
from chemworld.world.physical_world import CompositionPhysics


def thermal_feedback_world(
    controller_type: str = "PI",
    *,
    shared: bool = False,
) -> WorldCompositionSpec:
    """Synthetic ideal-gas caloric example; not a fitted physical controller."""
    if controller_type not in {"P", "PI", "PD", "PID"}:
        raise ValueError("example requires P, PI, PD or PID")
    network = ReactionNetworkSpec(
        "thermal-feedback-inert", (SpeciesSpec("A", "H2", phase="gas"),), ()
    )
    device = ThermalDeviceSpec(
        conductance_W_K=2.0,
        maximum_power_W=30.0,
        maximum_cooling_power_W=20.0,
        setpoint_K=350.0,
        minimum_setpoint_K=280.0,
        maximum_setpoint_K=430.0,
        enabled=not shared,
    )
    spec = ThermalFeedbackSpec(
        proportional_gain=2.0,
        integral_gain_s_inv=0.2 if "I" in controller_type else 0.0,
        derivative_gain_s=0.4 if "D" in controller_type else 0.0,
        derivative_filter_time_s=0.5,
        antiwindup_rate_s_inv=1.0 if "I" in controller_type else 0.0,
        bias_temperature_K=350.0,
        minimum_command_K=290.0,
        maximum_command_K=420.0,
    )
    names = ("reactor", "witness") if shared else ("reactor",)
    states, models = {}, {}
    for name in names:
        temperature = 300.0 if name == "reactor" else 310.0
        records = (
            {}
            if name != "reactor"
            else {
                "feedback-heater": EquipmentRecord(
                    "feedback-heater",
                    "thermal_device",
                    name,
                    settings={"thermal_device": device.to_dict()},
                ),
            }
        )
        state = WorldState(
            {"A": 0.01},
            1.0,
            temperature,
            0.01 * 8.31446261815324 * temperature * 1000,
            "gas",
            name,
            units={
                "amount": "mol",
                "volume": "L",
                "temperature": "K",
                "pressure": "Pa",
                "time": "s",
            },
            equipment=EquipmentLedger(records),
            vessels=VesselLedger(
                {
                    name: VesselRecord(
                        name,
                        "finite-thermal-vessel",
                        1.0,
                        430.0,
                        1e6,
                        ("gas",),
                        temperature,
                        0.01 * 8.31446261815324 * temperature * 1000,
                    )
                }
            ),
        )
        if name == "reactor":
            state = install_thermal_feedback(
                state,
                "feedback-heater",
                spec,
                initial_integral_K=0.0,
                initial_filtered_temperature_K=temperature,
            )
        states[name] = state
        models[name] = ThermalSystemSpec(
            ConstantCvThermodynamics({"A": SpeciesInternalEnergy(0.0, 20.0)}, 20.0),
            environment_temperature_K=300.0,
            loss_conductance_W_K=0.2,
            minimum_temperature_K=280.0,
            maximum_temperature_K=430.0,
            stop_temperature_K=425.0,
            maximum_pressure_Pa=1e6,
        )
    return WorldCompositionSpec.from_dict(
        {
            "schema_version": "chemworld-world-composition-0.1",
            "composition_id": "finite-thermal-feedback-example",
            "vessels": [{"id": name, "capacity_L": 1.0} for name in names],
            "components": [
                {
                    "id": name + ":" + kind,
                    "kind": kind,
                    "vessel": name,
                    "parameters": {
                        "temperature_range_K": [280.0, 430.0],
                        "duration_range_s": [1e-6, 14400.0],
                    }
                    if kind == "thermal"
                    else {},
                }
                for name in names
                for kind in (("thermal", "lab_schedule") if shared else ("thermal",))
            ],
            "physics": CompositionPhysics(
                network,
                states,
                models={"thermal": models},
                lab_clock=LabClockSpec(heater_ids=("feedback-heater",)) if shared else None,
            ).to_dict(),
            "task": {
                "objective": "control actual sample temperature with one original finite heater",
                "budget": 100,
                "operations": ["wait", "terminate"]
                + (
                    ["queue_lab_heat", "select_vessel"]
                    if shared
                    else ["heat", "pause_control", "resume_control"]
                ),
                "instruments": [],
                "termination": "explicit-stop-or-budget",
            },
        }
    )
