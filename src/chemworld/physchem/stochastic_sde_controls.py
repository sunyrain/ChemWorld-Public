"""Resolve current inputs against immutable channel/parameter declarations."""

from collections.abc import Mapping

from chemworld.foundation.dimensions import DimensionVector
from chemworld.physchem.stochastic_sde_specs import finite_real


def resolve_sde_controls(network, spec, noises, values):
    metadata, defaults, rate_scales, flux_parameters, noise_parameters = {}, {}, {}, {}, {}
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
            parameters.setdefault(control.reaction_id, {})[control.parameter] = value
    return current, metadata, rate_scales, flux_parameters, noise_parameters
