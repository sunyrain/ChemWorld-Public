"""Explicit synthetic ideal finite ED declaration, reusing the original ion table."""

from chemworld.physchem.electrodialysis import ElectrodialysisSpec
from chemworld.physchem.ion_exchange_pore import IonExchangePoreSpec
from reference.nanofiltration48_fixtures import amounts as nf_amounts
from reference.nanofiltration48_fixtures import spec as nf_spec


def spec(ions=3, segments=16):
    native = nf_spec(ions)
    diff = tuple((k, 1e-10 * (1 + 0.15 * i)) for i, k in enumerate(native.properties.ion_ids))
    aem = IonExchangePoreSpec(100, 2e-5, 1e-12, diff, native.source_id, segments)
    cem = IonExchangePoreSpec(-100, 2e-5, 1e-12, diff, native.source_id, segments)
    return ElectrodialysisSpec(
        native.properties,
        aem,
        cem,
        1e-5,
        310,
        1e5,
        (0.04, 0.04),
        0.001,
        0.01,
        2.0,
        native.source_id,
    )


def amounts(model, dilution=1.0, water=1.0):
    return nf_amounts(nf_spec(len(model.properties.ion_ids)), water, dilution)


def inventories(model, direction=1):
    return (
        amounts(model, dilution=0.5 if direction < 0 else 1),
        amounts(model, dilution=1 if direction < 0 else 0.5),
    )
