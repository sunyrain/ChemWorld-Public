"""Original synthetic two finite gas tanks with a bounded shaft machine."""

from typing import Any

from chemworld.physchem.gas_network_energy_specs import GasNodeEnergySpec
from chemworld.physchem.gas_reactor_network import (
    GasFlowConnectionSpec,
    GasReactorNetworkSpec,
    GasReactorNodeSpec,
)
from chemworld.physchem.gas_shaft_machine import GasShaftMachineSpec
from chemworld.physchem.heat_transfer_units import ThermalDeviceSpec
from chemworld.physchem.reaction_network import ReactionNetworkSpec, SpeciesSpec
from chemworld.world.composition import WorldCompositionSpec
from chemworld.world.gas_reactor_network import gas_reactor_network_world


def gas_shaft_example_document() -> dict[str, Any]:
    """CC0 authored teaching identities; not fitted hydrogen-isomer chemistry."""
    return {
        "description": "Original ChemWorld CC0 synthetic nonreactive shaft-machine teaching model",
        "units": {"length": "m", "quantity": "kmol", "activation-energy": "J/kmol"},
        "phases": [
            {
                "name": "shaft-mixture",
                "thermo": "ideal-gas",
                "kinetics": "gas",
                "elements": ["H"],
                "species": ["A", "B"],
            }
        ],
        "species": [
            {
                "name": name,
                "composition": {"H": 2},
                "thermo": {
                    "model": "constant-cp",
                    "T0": 298.15,
                    "h0": f"{h} J/mol",
                    "s0": "40 J/mol/K",
                    "cp0": f"{cp} J/mol/K",
                    "T-min": 150.0,
                    "T-max": 2000.0,
                },
            }
            for name, h, cp in (("A", 0.0, 32.0), ("B", -5000.0, 40.0))
        ],
        "reactions": [],
    }


def gas_shaft_example_network(document: dict[str, Any] | None = None) -> ReactionNetworkSpec:
    document = gas_shaft_example_document() if document is None else document
    return ReactionNetworkSpec(
        "shaft-mixture",
        tuple(SpeciesSpec(row["name"], "H2", phase="gas") for row in document["species"]),
        (),
        metadata={
            "source": document["description"],
            "gas_thermochemistry": {
                row["name"]: {
                    "thermo": row["thermo"],
                    "composition": row["composition"],
                    "reference_pressure_Pa": 101325.0,
                    "units": document["units"],
                }
                for row in document["species"]
            },
        },
    )


def gas_shaft_example_inputs(
    mode: str = "compressor", *, anchor: str = "source", maximum_power_W: float = 10.0
) -> tuple[GasReactorNetworkSpec, ReactionNetworkSpec, dict[str, dict[str, float]]]:
    if mode not in {"compressor", "expander"}:
        raise ValueError("example mode must be compressor or expander")
    source, receiver = (0.02, 0.08) if mode == "compressor" else (0.08, 0.02)
    amounts = {
        "source": {"A": source * 0.7, "B": source * 0.3},
        "receiver": {"A": receiver * 0.4, "B": receiver * 0.6},
    }
    spec = GasReactorNetworkSpec(
        (
            GasReactorNodeSpec("source", 1.0, False, 1e-6),
            GasReactorNodeSpec("receiver", 2.0, False, 1e-6),
        ),
        (GasFlowConnectionSpec("shaft-feed", "source", "receiver", coefficient=0.001),),
        400.0 if anchor == "source" else 350.0,
        anchor,
        (GasNodeEnergySpec("source", 400.0, 0.5), GasNodeEnergySpec("receiver", 350.0, 0.8)),
        shaft_machine=GasShaftMachineSpec("shaft-feed", mode, 0.8, maximum_power_W, 12.0),
    )
    return spec, gas_shaft_example_network(), amounts


def gas_shaft_machine_world(
    mode: str = "compressor", *, anchor: str = "source", with_heater: bool = False
) -> WorldCompositionSpec:
    spec, network, amounts = gas_shaft_example_inputs(mode, anchor=anchor)
    heater = (
        ThermalDeviceSpec(
            conductance_W_K=0.2,
            maximum_power_W=2.0,
            maximum_cooling_power_W=2.0,
            setpoint_K=600.0,
            minimum_setpoint_K=200.0,
            maximum_setpoint_K=1500.0,
            enabled=False,
        )
        if with_heater
        else None
    )
    return gas_reactor_network_world(spec, network, amounts, anchor_thermal_device=heater)
