"""Nonisothermal reacting grid with finite feed/product tanks and original heater."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from chemworld.examples.gas_flow_column import example_inputs as isothermal_inputs
from chemworld.physchem.gas_network_energy_specs import GasNodeEnergySpec, GasThermalLinkSpec
from chemworld.physchem.heat_transfer_units import ThermalDeviceSpec
from chemworld.runtime import ChemWorldRuntime
from chemworld.world.gas_reactor_network import gas_reactor_network_world


def example_inputs(*, cells=4, non_equimolar=False, flux_model="stefan-maxwell"):
    """Synthetic A/B gas, authored temperature gradient, finite support and heat link."""
    spec, network, amounts = isothermal_inputs(
        cells=cells, non_equimolar=non_equimolar, flux_model=flux_model
    )
    temperatures = {
        "feed": 650.0,
        "receiver": 500.0,
        **{
            name: 600.0 + 60 * i / max(cells - 1, 1)
            for i, name in enumerate(spec.spatial_transport.cell_vessel_ids)
        },
    }
    spec = replace(
        spec,
        spatial_transport=replace(spec.spatial_transport, energy_equation=True),
        energy_nodes=tuple(
            GasNodeEnergySpec(
                n.vessel_id,
                temperatures[n.vessel_id],
                0.15,
                loss_conductance_W_K=0.0001,
                maximum_temperature_K=1500.0,
            )
            for n in spec.nodes
        ),
        heat_links=(GasThermalLinkSpec("support-link", "feed", "receiver", 0.0002),),
    )
    amounts = {
        name: {s: n * 600 / temperatures[name] for s, n in row.items()}
        for name, row in amounts.items()
    }
    return spec, network, amounts


def example_world(**kwargs):
    spec, network, amounts = example_inputs(**kwargs)
    return gas_reactor_network_world(
        spec,
        network,
        amounts,
        anchor_thermal_device=ThermalDeviceSpec(
            maximum_power_W=0.2,
            setpoint_K=800.0,
            enabled=False,
            minimum_setpoint_K=250.0,
            maximum_setpoint_K=1400.0,
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("preserve attempts; choose a fresh output path")
    runtime = ChemWorldRuntime(composition=example_world())
    try:
        runtime.reset(seed=36)
        outcomes = []
        for command in (
            {"operation": "wait", "duration_s": 0.03},
            {"operation": "queue_lab_heat", "target_temperature_K": 800.0, "duration_s": 1.0},
            {"operation": "wait", "duration_s": 0.02},
        ):
            outcome = runtime.execute(command)
            outcomes.append(dict(outcome.info))
            print(
                json.dumps(
                    {
                        "operation": command["operation"],
                        "status": outcome.info["transaction_status"],
                        "time_s": runtime._state.ledger.time_s,
                    }
                ),
                flush=True,
            )
        args.output.write_text(
            json.dumps(
                {"outcomes": outcomes, "state": runtime._state.to_dict()}, indent=2, allow_nan=False
            ),
            encoding="utf-8",
        )
        if any(row["transaction_status"] != "committed" for row in outcomes):
            raise RuntimeError("a world operation failed; see the saved complete outcomes")
    finally:
        runtime.close()


if __name__ == "__main__":
    main()
