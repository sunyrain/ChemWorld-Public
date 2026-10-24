"""Fixed raw cards and separate native assembly for the 36C development block."""

from dataclasses import replace

R = 8.31446261815324
TIMES = (0.0, 0.1, 0.3, 0.8)


def case(number):
    non = number in (5, 6, 7, 8, 18)
    cells = {4: 1, 9: 4, 10: 8, 11: 16, 12: 32}.get(number, 4)
    card = {
        "number": number,
        "cells": cells,
        "non_equimolar": non,
        "temperature": 600.0,
        "pressure": 100000.0,
        "length": 0.04,
        "area": 0.002,
        "feed_volume": 0.001,
        "receiver_volume": 0.002,
        "feed_pressure": 160000.0,
        "receiver_pressure": 40000.0,
        "feed_x": 0.8,
        "receiver_x": 0.1,
        "fractions": [
            {"A": 0.1 + 0.6 * (i + 0.5) / cells, "B": 0.9 - 0.6 * (i + 0.5) / cells}
            for i in range(cells)
        ],
        "Fin": 0.0004,
        "Fout": 0.0004,
        "k": 0.0 if number in (3, 13, 14, 15, 16) else 0.3,
        "model": "mixture-averaged" if number in (2, 6) else "stefan-maxwell",
        "maximum_mach": 0.05,
        "minimum_feed": 1e-12,
        "maximum_grid": 1e8,
        "maximum_receiver": 1e8,
        "species": {
            "A": {"mass": 0.002016, "cp": 30.0, "h": 0.0, "sigma": 3.5, "epsilon": 100.0},
            "B": {
                "mass": 0.004032 if non else 0.002016,
                "cp": 44.0,
                "h": -9000.0,
                "sigma": 3.5,
                "epsilon": 100.0,
            },
        },
        "times": TIMES,
    }
    if number == 7:
        card.update(Fin=0.0002, Fout=0.0008)
    if number == 8:
        card.update(
            Fin=0.0, Fout=0.0, fractions=[{"A": x, "B": 1 - x} for x in (0.9, 0.9, 0.1, 0.1)]
        )
    if number in (9, 10, 11, 12):
        card["times"] = (0.0, 30.0)
    if number == 13:
        card["minimum_feed"] = (
            card["feed_pressure"] * card["feed_volume"] / (R * card["temperature"]) - 0.00012
        )
    if number == 14:
        card["maximum_receiver"] = 40700.0
    if number == 15:
        card.update(Fin=0.0008, maximum_grid=110000.0)
    if number == 16:
        card.update(
            Fin=0.0,
            Fout=0.0008,
            maximum_mach=2e-5,
            feed_x=0.5,
            receiver_x=0.5,
            fractions=[{"A": 0.5, "B": 0.5} for _ in range(cells)],
        )
    if number in (13, 14, 15, 16):
        card["times"] = (0.0, 1.5)
    return card


def native(card):
    from chemworld.examples.gas_flow_column import example_inputs

    spec, network, _ = example_inputs(
        cells=card["cells"], non_equimolar=card["non_equimolar"], flux_model=card["model"]
    )
    reaction = network.reactions[0]
    network = replace(
        network,
        reactions=(
            replace(
                reaction,
                rate_law=replace(
                    reaction.rate_law, parameters={**reaction.rate_law.parameters, "A": card["k"]}
                ),
            ),
        ),
    )
    spec = replace(
        spec,
        connections=tuple(
            replace(e, coefficient=card["Fin" if e.connection_id == "inlet" else "Fout"])
            for e in spec.connections
        ),
        spatial_transport=replace(spec.spatial_transport, maximum_mach=card["maximum_mach"]),
        nodes=tuple(
            replace(
                n,
                minimum_amount_mol=card["minimum_feed"] if n.vessel_id == "feed" else 1e-12,
                maximum_pressure_Pa=card["maximum_receiver"]
                if n.vessel_id == "receiver"
                else card["maximum_grid"]
                if n.vessel_id.startswith("cell")
                else n.maximum_pressure_Pa,
            )
            for n in spec.nodes
        ),
    )
    cells = spec.spatial_transport.cell_vessel_ids
    fractions = {
        "feed": {"A": card["feed_x"], "B": 1 - card["feed_x"]},
        "receiver": {"A": card["receiver_x"], "B": 1 - card["receiver_x"]},
        **dict(zip(cells, card["fractions"], strict=True)),
    }
    pressure = {"feed": card["feed_pressure"], "receiver": card["receiver_pressure"]}
    amounts = {
        n.vessel_id: {
            s: x
            * pressure.get(n.vessel_id, card["pressure"])
            * n.volume_L
            / (1000 * R * card["temperature"])
            for s, x in fractions[n.vessel_id].items()
        }
        for n in spec.nodes
    }
    return spec, network, amounts
