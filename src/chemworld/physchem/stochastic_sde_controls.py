"""Resolve current inputs against immutable channel/parameter declarations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, TypedDict, cast

from chemworld.foundation.dimensions import DimensionVector
from chemworld.physchem.stochastic_sde_specs import finite_real

if TYPE_CHECKING:
    from chemworld.physchem.reaction_network import ReactionNetworkSpec
    from chemworld.physchem.stochastic_sde_specs import ReactionNoise, StochasticSDESpec


class SDEControlMetadata(TypedDict):
    reaction_id: str
    target: str
    parameter: str | None
    initial_value: float
    bounds: list[float]
    dimension: dict[str, float]
    unit: str


def resolve_sde_controls(
    network: ReactionNetworkSpec,
    spec: StochasticSDESpec,
    noises: Sequence[ReactionNoise],
    values: Mapping[str, float] | None,
) -> tuple[
    dict[str, float], dict[str, SDEControlMetadata], dict[str, float],
    dict[str, dict[str, float]], dict[str, dict[str, float]],
]:
    metadata: dict[str, SDEControlMetadata] = {}
    defaults: dict[str, float] = {}
    rate_scales: dict[str, float] = {}
    flux_parameters: dict[str, dict[str, float]] = {}
    noise_parameters: dict[str, dict[str, float]] = {}
    fluxes = {f.reaction_id: f for f in spec.fluxes}
    noise_map = {n.reaction_id: n for n in noises}
    for control in spec.controls:
        if control.reaction_id not in network.reaction_ids:
            raise ValueError("SDE control must bind an existing reaction")
        dimension = DimensionVector()
        if control.target == "rate_multiplier":
            initial = 1.0
        else:
            declaration = (fluxes if control.target == "flux_parameter" else noise_map).get(
                control.reaction_id
            )
            parameter = next(
                (p for p in getattr(declaration, "parameters", ()) if p.name == control.parameter),
                None,
            )
            if parameter is None:
                raise ValueError("SDE control requires an existing typed parameter")
            initial, dimension = parameter.value, parameter.dimension
        if not control.lower <= initial <= control.upper:
            raise ValueError("SDE control bounds exclude the declared initial value")
        defaults[control.name] = initial
        metadata[control.name] = {
            "reaction_id": control.reaction_id,
            "target": control.target,
            "parameter": control.parameter,
            "initial_value": initial,
            "bounds": [control.lower, control.upper],
            "dimension": dimension.to_dict(),
            "unit": "dimensionless" if dimension == DimensionVector() else "coherent_SI",
        }
    if values is None:
        values = defaults
    if not isinstance(values, Mapping) or values.keys() != defaults.keys():
        raise ValueError("SDE current controls must cover exactly their declared inputs")
    current = {}
    for control in spec.controls:
        value = finite_real(values[control.name], "SDE current control")
        if not control.lower <= value <= control.upper:
            raise ValueError("SDE control value is outside its declared bounds")
        current[control.name] = value
        if control.target == "rate_multiplier":
            rate_scales[control.reaction_id] = value
        else:
            parameters = flux_parameters if control.target == "flux_parameter" else noise_parameters
            # Parameter controls validate their nonempty name during construction.
            parameters.setdefault(control.reaction_id, {})[cast(str, control.parameter)] = value
    return current, metadata, rate_scales, flux_parameters, noise_parameters
