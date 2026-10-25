"""Authored constant-Cp cards; native assembly is separate from reference equations."""

from dataclasses import replace

import gas_spatial36_cases as sealed
import gas_spatial_flow36_cases as flowing

R = 8.31446261815324


def case(number):
    if number <= 4:
        card = sealed.case(
            "equimolar" if number <= 2 else "non-equimolar",
            "mixture-averaged" if number % 2 == 0 else "stefan-maxwell",
        )
        card.update(
            flow=False,
            temperatures=[570, 590, 620, 650],
            support=0.05,
            loss=0.0001,
            link=0.0002,
            duration=0.06,
        )
    else:
        card = flowing.case(5 if number == 5 else 7)
        card.update(
            flow=True,
            temperatures=[670, 570, 590, 620, 650, 510],
            support=0.05,
            loss=0.0001,
            link=0.0002,
            duration=0.06,
        )
    return card


def native(card):
    from chemworld.physchem.gas_network_energy_specs import GasNodeEnergySpec, GasThermalLinkSpec

    spec, network, amounts = (flowing if card["flow"] else sealed).native(card)
    names = list(spec.spatial_transport.cell_vessel_ids)
    if card["flow"]:
        names = ["feed", *names, "receiver"]
    temperatures = dict(zip(names, card["temperatures"], strict=True))
    spec = replace(
        spec,
        temperature_K=temperatures[spec.anchor_vessel_id],
        spatial_transport=replace(spec.spatial_transport, energy_equation=True),
        energy_nodes=tuple(
            GasNodeEnergySpec(
                n.vessel_id,
                temperatures[n.vessel_id],
                card["support"],
                loss_conductance_W_K=card["loss"],
                maximum_temperature_K=1500,
            )
            for n in spec.nodes
        ),
        heat_links=(GasThermalLinkSpec("link", names[0], names[-1], card["link"]),)
        if len(names) > 1
        else (),
    )
    amounts = {
        name: {s: n * card["temperature"] / temperatures[name] for s, n in row.items()}
        for name, row in amounts.items()
    }
    return spec, network, amounts


def pure(cells=4, amplitude=0.01):
    import math

    card = sealed.case("pure", cells=cells)
    card.update(
        flow=False,
        temperatures=[
            600 + amplitude * math.cos(math.pi * (i + 0.5) / cells) for i in range(cells)
        ],
        support=0.0,
        loss=0.0,
        link=0.0,
        duration=0.03,
    )
    return card
