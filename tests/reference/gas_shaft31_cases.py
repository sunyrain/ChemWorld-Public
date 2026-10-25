"""Fixed 31D source cards and native adaptation; no outcome-dependent selection."""

from dataclasses import replace
from math import exp, log

from chemworld.examples.gas_shaft_machine import (
    gas_shaft_example_document,
    gas_shaft_example_inputs,
    gas_shaft_example_network,
)
from chemworld.physchem.gas_reactor_network import GasReactorNetworkSpec

R = 8.31446261815324


def document(family="constant", shifts=None):
    value = gas_shaft_example_document()
    shifts = shifts or {"A": 0.0, "B": 0.0}
    if family == "nasa":
        for row, coefficients in zip(
            value["species"],
            (
                [3.8, 0.0012, -1e-7, 0.0, 0.0, -1200.0, 2.0],
                [4.2, 0.0018, 0.0, 0.0, 0.0, -2000.0, 4.0],
            ),
            strict=True,
        ):
            coefficients[5] += shifts[row["name"]] / R
            row["thermo"] = {
                "model": "NASA7",
                "temperature-ranges": [150.0, 2000.0],
                "data": [coefficients],
            }
    else:
        for row in value["species"]:
            row["thermo"]["h0"] = (
                f"{float(row['thermo']['h0'].split()[0]) + shifts[row['name']]} J/mol"
            )
    return value


def case(family="constant", mode="compressor", regime="flow", *, anchor="source"):
    spec, _, amounts = gas_shaft_example_inputs(
        mode, anchor=anchor, maximum_power_W=10.0 if regime == "flow" else 0.5
    )
    if regime == "power":
        spec = replace(
            spec,
            connections=(replace(spec.connections[0], mode="mass-flow", coefficient=2.016e-6),),
        )
    return {
        "id": f"{family}-{mode}-{regime}",
        "document": document(family),
        "spec": spec.to_dict(),
        "amounts": amounts,
        "duration_s": 1.0,
        "sample_times_s": [0.2, 0.5, 1.0],
        "heat_power_W": 0.0,
    }


def native(card):
    return (
        GasReactorNetworkSpec.from_dict(card["spec"]),
        gas_shaft_example_network(card["document"]),
        card["amounts"],
    )


def event_case(kind):
    mode = "expander" if kind in {"outlet-low-T", "direction"} else "compressor"
    card = case(mode=mode)
    spec, _, _ = native(card)
    nodes = {n.vessel_id: n for n in spec.nodes}
    energy = {n.vessel_id: n for n in spec.energy_nodes}
    machine = spec.shaft_machine
    expected = ""
    if kind == "inventory":
        nodes["source"] = replace(nodes["source"], minimum_amount_mol=0.0196)
        expected = "gas:source:minimum_amount"
    elif kind == "pressure":
        pressure = 0.08 * R * 350.0 * 1000 / 2.0
        nodes["receiver"] = replace(nodes["receiver"], maximum_pressure_Pa=pressure * 1.01)
        expected = "gas:receiver:maximum_pressure"
    elif kind == "source-low-T":
        energy["source"] = replace(energy["source"], minimum_temperature_K=399.5)
        expected = "gas:source:minimum_temperature"
    elif kind == "receiver-high-T":
        energy["receiver"] = replace(energy["receiver"], maximum_temperature_K=350.2)
        expected = "gas:receiver:maximum_temperature"
    elif kind == "ratio":
        machine = replace(machine, maximum_pressure_ratio=1.78)
        expected = "shaft:shaft-feed:maximum_pressure_ratio"
    elif kind in {"outlet-high-T", "outlet-low-T"}:
        ratio = 1.75 if mode == "compressor" else 1 / ((0.08 * 400.0) / (0.02 * 350.0 / 2.0))
        ts = 400.0 * exp(R / 34.4 * log(ratio))
        tout = 400.0 + ((ts - 400.0) / 0.8 if mode == "compressor" else (ts - 400.0) * 0.8)
        if kind == "outlet-high-T":
            machine = replace(machine, maximum_outlet_temperature_K=tout + 0.5)
            expected = "shaft:shaft-feed:maximum_outlet_temperature"
        else:
            machine = replace(machine, minimum_outlet_temperature_K=tout - 0.5)
            energy["source"] = replace(energy["source"], loss_conductance_W_K=0.5)
            expected = "shaft:shaft-feed:minimum_outlet_temperature"
    elif kind == "direction":
        card["duration_s"] = 80.0
        expected = "shaft:shaft-feed:pressure_direction"
    else:
        raise ValueError(kind)
    card.update(
        id=f"event-{kind}",
        spec=replace(
            spec,
            nodes=tuple(nodes.values()),
            energy_nodes=tuple(energy.values()),
            shaft_machine=machine,
        ).to_dict(),
        sample_times_s=[],
        expected_event=expected,
    )
    return card


def dynamic_cases():
    return [
        case(family, mode, regime)
        for family in ("constant", "nasa")
        for mode in ("compressor", "expander")
        for regime in ("flow", "power")
    ] + [
        event_case(kind)
        for kind in (
            "inventory",
            "pressure",
            "source-low-T",
            "receiver-high-T",
            "ratio",
            "outlet-high-T",
            "outlet-low-T",
            "direction",
        )
    ]
