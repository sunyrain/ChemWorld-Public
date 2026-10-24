"""Caller-owned neutral kinetics and liquid properties in a real finite CSTR."""

from collections.abc import Mapping

from chemworld.physchem.reactive_water_treatment import ReactiveWaterTreatmentSpec
from chemworld.world.composition import WorldCompositionSpec
from chemworld.world.reactive_water_treatment_world import reactive_water_treatment_world_spec


def reactive_water_treatment_example(
    spec: ReactiveWaterTreatmentSpec,
    feed: Mapping[str, float],
    reactor: Mapping[str, float],
    product: Mapping[str, float] | None = None,
    *,
    shared_clock: bool = True,
) -> WorldCompositionSpec:
    """Set inlet flow, then wait; zero flow retains original batch chemistry."""
    return reactive_water_treatment_world_spec(
        spec, feed, reactor, product, shared_clock=shared_clock
    )
