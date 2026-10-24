"""Fixed 33A declarations and independent ideal one-loop equations."""

from chemworld.foundation import WorldState
from chemworld.foundation.state_ledgers import PhaseLedger, PhaseRecord, VesselLedger, VesselRecord
from chemworld.physchem.molecular_processes import MolecularHeatDuty, MolecularPressureStage
from chemworld.physchem.molecular_recycle import MolecularRecyclePath
from reference.run_molecular_processes import R, domain, stream


def path(*, x=(0.25, 0.75), recycle=0.2, pressure_cycle=False, start=350.0, real=False):
    model = domain(real)
    devices = (MolecularHeatDuty(model, 5.0 if real else 10.0),)
    if pressure_cycle:
        devices += (
            MolecularPressureStage(model, "compressor", 4e5, 0.8),
            MolecularPressureStage(model, "expander", 2e5, 0.85),
        )
    return MolecularRecyclePath(
        model,
        "external-feed",
        stream(model, fractions=x),
        devices,
        recycle,
        2e5,
        tuple(0.01 * v for v in x),
        start,
    )


def expected(x, recycle, pressure_cycle):
    cp = R * (3.5 * x[0] + 4.5 * x[1])
    compress = 1 + (2 ** (R / cp) - 1) / 0.8 if pressure_cycle else 1.0
    expand = 1 - 0.85 * (1 - 2 ** (-R / cp)) if pressure_cycle else 1.0
    a = compress * expand
    flow = 0.05 / (1 - recycle)
    tout = a * (1 - recycle) * (400 + 10 / (0.05 * cp)) / (1 - a * recycle)
    tmix = (1 - recycle) * 400 + recycle * tout
    heated = tmix + 10 / (flow * cp)
    work = flow * cp * (compress * heated - heated + tout - compress * heated)
    return {
        "tear_flow": recycle * flow,
        "product_flow": 0.05,
        "outlet_T": tout,
        "mixer_T": tmix,
        "heat": 10.0,
        "work": work,
        "composition": x,
    }


def receiver(model):
    temperature, volume, total = 350.0, 1.0, 0.05
    x = model.inlet.state.mole_fractions
    s = model.domain.model.state(temperature, 50.0, x)
    amounts = {key: total * v for key, v in zip(model.domain.model.component_ids, x, strict=True)}
    phase = PhaseRecord(
        "bulk",
        "receiver",
        "gas",
        volume,
        amounts,
        metadata={"carrier_representation": "species_inventory"},
    )
    hardware = VesselRecord(
        "receiver", "tank", volume, 900.0, 2e5, ("gas",), temperature, s.pressure_Pa
    )
    return WorldState(
        amounts,
        volume,
        temperature,
        s.pressure_Pa,
        "gas",
        "receiver",
        phases=PhaseLedger({"bulk": phase}),
        vessels=VesselLedger({"receiver": hardware}),
    )
