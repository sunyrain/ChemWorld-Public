"""Regular, Flory-Huggins and Hansen liquid models on the shared GE contract.

Molar volumes and cohesive parameters are explicit constant reference inputs.
Their temperature dependence is not invented. Source vectors remain immutable
input records and never create a separate material state.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import isfinite
from typing import TYPE_CHECKING, Any, Literal

from chemworld.data.scientific_inputs import InputValue
from chemworld.physchem.caloric_math import logarithm, scalar_value
from chemworld.physchem.solubility import HansenParameters

if TYPE_CHECKING:
    from chemworld.physchem.equilibrium import ActivityModelSpec

R = 8.31446261815324
REGULAR_MODELS = frozenset({"regular_solution", "flory_huggins", "hansen"})


def _number(value: Any, name: str, *, positive: bool = False) -> Any:
    scalar = scalar_value(value)
    if isinstance(scalar, bool) or not isfinite(scalar) or scalar < 0 or (positive and scalar == 0):
        raise ValueError(f"{name} must be finite and {'positive' if positive else 'nonnegative'}")
    return value


def validate_regular_parameters(spec: ActivityModelSpec) -> None:
    p, ids = spec.parameters, spec.component_ids
    prefixes = ("delta_d", "delta_p", "delta_h") if spec.model == "hansen" else ("delta",)
    required = {f"{prefix}:{key}" for prefix in ("V", *prefixes) for key in ids}
    if spec.model == "hansen":
        required.add("alpha")
    else:
        required.update(f"lambda:{left}|{right}" for left in ids for right in ids if left != right)
    if set(p) != required:
        raise ValueError(
            f"{spec.model} parameter fields: missing={sorted(required - set(p))}, "
            f"unknown={sorted(set(p) - required)}"
        )
    for key, value in p.items():
        if key.startswith("lambda:"):
            if isinstance(value, bool) or not isfinite(value):
                raise ValueError("regular-solution interactions must be finite")
        else:
            _number(value, key, positive=key.startswith("V:"))


def regular_reduced_gibbs(
    spec: ActivityModelSpec, xs: Any, temperature: Any, parameters: Mapping[str, Any]
) -> Any:
    p, ids = parameters, spec.component_ids
    volumes = [_number(p[f"V:{key}"], "molar volume", positive=True) for key in ids]
    weighted = [x * v for x, v in zip(xs, volumes, strict=True)]
    vbar = sum(weighted)
    _number(vbar, "mixture reference volume", positive=True)
    n = len(ids)
    cohesive = 0.0
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if spec.model == "hansen":
                alpha = _number(p["alpha"], "Hansen alpha")
                # The double sum counts each unlike pair twice. thermo 0.6.0
                # Hansen examples and Aijs use alpha/2; its class formula
                # omits that factor (recorded in the CONV23B source note).
                a = 0.5 * alpha * sum(
                    weight
                    * (
                        _number(p[f"{prefix}:{ids[i]}"], prefix)
                        - _number(p[f"{prefix}:{ids[j]}"], prefix)
                    )
                    ** 2
                    for prefix, weight in (("delta_d", 1.0), ("delta_p", 0.25), ("delta_h", 0.25))
                )
            else:
                left = _number(p[f"delta:{ids[i]}"], "solubility parameter")
                right = _number(p[f"delta:{ids[j]}"], "solubility parameter")
                a = 0.5 * (left - right) ** 2 + left * right * p[f"lambda:{ids[i]}|{ids[j]}"]
            cohesive += weighted[i] * weighted[j] * a
    result = cohesive / (vbar * R * temperature)
    if spec.model != "regular_solution":
        result += sum(x * logarithm(v / vbar) for x, v in zip(xs, volumes, strict=True))
    return result


def regular_solution_model(
    model_id: str,
    component_ids: Sequence[str],
    *,
    molar_volumes_m3_mol: Mapping[str, float],
    solubility_parameters_Pa_half: Mapping[str, float],
    interaction_parameters: Mapping[tuple[str, str], float] | None = None,
    model: Literal["regular_solution", "flory_huggins"] = "regular_solution",
    provenance: Mapping[str, object] | None = None,
) -> ActivityModelSpec:
    """Compile explicit constants, including all zero corrections if none are requested."""
    from chemworld.physchem.equilibrium import ActivityModelSpec

    ids = tuple(component_ids)
    if set(molar_volumes_m3_mol) != set(ids) or set(solubility_parameters_Pa_half) != set(ids):
        raise ValueError("regular-solution inputs must cover exactly the declared components")
    pairs = {(left, right) for left in ids for right in ids if left != right}
    if interaction_parameters is not None and set(interaction_parameters) != pairs:
        raise ValueError("interaction corrections must contain every directed component pair")
    interactions = (
        dict.fromkeys(pairs, 0.0) if interaction_parameters is None else interaction_parameters
    )
    parameters = {
        **{f"V:{key}": molar_volumes_m3_mol[key] for key in ids},
        **{f"delta:{key}": solubility_parameters_Pa_half[key] for key in ids},
        **{
            f"lambda:{left}|{right}": value for (left, right), value in sorted(interactions.items())
        },
    }
    return ActivityModelSpec(
        model_id,
        ids,
        model,
        parameters,
        {
            **({} if provenance is None else dict(provenance)),
            "constant_input_units": {"V": "m^3/mol", "delta": "Pa^0.5", "lambda": "dimensionless"},
            "interaction_mode": "explicit_zero_corrections"
            if interaction_parameters is None
            else "explicit_directed_corrections",
            "temperature_dependence": "constant reference volumes and cohesive parameters",
        },
    )


def hansen_activity_model(
    model_id: str,
    component_parameters: Sequence[HansenParameters],
    *,
    molar_volumes: Mapping[str, InputValue],
    alpha: float,
    volume_reference: str,
) -> ActivityModelSpec:
    """Consume original source-bearing Hansen vectors and authored volume inputs."""
    from chemworld.physchem.equilibrium import ActivityModelSpec

    if not component_parameters or any(
        not isinstance(v, HansenParameters) for v in component_parameters
    ):
        raise TypeError("Hansen activity inputs require source-bearing HansenParameters")
    ids = tuple(v.component_id for v in component_parameters)
    if len(set(ids)) != len(ids) or set(molar_volumes) != set(ids):
        raise ValueError("Hansen vectors and volumes need the same unique component identities")
    if not isinstance(volume_reference, str) or not volume_reference.strip():
        raise ValueError("reference-volume inputs require a declared reference description")
    parameters = {"alpha": alpha}
    for vector in component_parameters:
        volume = molar_volumes[vector.component_id]
        if not isinstance(volume, InputValue) or volume.quality != "valid" or volume.value is None:
            raise ValueError("Hansen reference volumes must be explicit valid InputValues")
        parameters[f"V:{vector.component_id}"] = volume.to("m^3/mol").value
        parameters.update(
            {
                f"{prefix}:{vector.component_id}": value
                for prefix, value in zip(
                    ("delta_d", "delta_p", "delta_h"), vector.values_pa_half(), strict=True
                )
            }
        )
    return ActivityModelSpec(
        model_id,
        ids,
        "hansen",
        parameters,
        {
            "hansen_sources": [v.to_dict() for v in component_parameters],
            "molar_volume_inputs": {key: value.to_dict() for key, value in molar_volumes.items()},
            "volume_reference": volume_reference,
            "constant_input_units": {
                "V": "m^3/mol",
                "delta_d/p/h": "Pa^0.5",
                "alpha": "dimensionless",
            },
            "temperature_dependence": "constant reference volumes and cohesive parameters",
        },
    )


__all__ = ["hansen_activity_model", "regular_solution_model"]
