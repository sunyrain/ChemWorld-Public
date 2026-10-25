"""Rate-law evaluation helpers for reaction networks."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import replace
from itertools import pairwise
from math import exp, isfinite, log10, ulp
from typing import Any, Protocol, TypeVar, cast

from chemworld.physchem.reaction_network_specs import ReactionSpec

R_J_PER_MOL_K = 8.31446261815324


class ReactionNetworkLike(Protocol):
    @property
    def reactions(self) -> tuple[ReactionSpec, ...]: ...


NetworkT = TypeVar("NetworkT", bound=ReactionNetworkLike)
ThermochemicalReverseRate = Callable[
    [ReactionSpec, Mapping[str, object], float, Mapping[str, Any]],
    float,
]


def evaluate_rate_law(
    reaction: ReactionSpec,
    *,
    concentrations_mol_L: Mapping[str, float],
    temperature_K: float,
    species_thermo: Mapping[str, Any] | None = None,
    activity_coefficients: Mapping[str, float] | None = None,
    thermochemical_reverse_rate_constant: ThermochemicalReverseRate | None = None,
) -> float:
    if temperature_K <= 0:
        raise ValueError("temperature_K must be positive")
    params = reaction.rate_law.parameters
    _declared_temperature_validity(params, temperature_K)
    if reaction.rate_law.equation_id == "energy_grain_collision":
        from chemworld.physchem.energy_grain_master_equation import (
            evaluate_energy_grain_collision_rate,
        )

        return evaluate_energy_grain_collision_rate(
            reaction,
            concentrations_mol_L=concentrations_mol_L,
            temperature_K=temperature_K,
            activity_coefficients=activity_coefficients,
        )
    if params.get("mechanism_schema") == "cantera-gas-v2":
        from chemworld.physchem.mechanism_rates import evaluate_mechanism_rate

        if activity_coefficients is not None:
            raise ValueError(
                "imported ideal-gas mechanisms do not use liquid activity coefficients"
            )
        return evaluate_mechanism_rate(
            reaction, concentrations_mol_L=concentrations_mol_L,
            temperature_K=temperature_K, species_thermo=species_thermo)
    equation_id = reaction.rate_law.equation_id
    if equation_id == "runtime_owned":
        runtime_model_id = str(params["runtime_model_id"])
        raise ValueError(
            "runtime_owned pathway cannot be evaluated by the generic reaction network: "
            f"reaction_id={reaction.reaction_id!r}, runtime_model_id={runtime_model_id!r}"
        )
    driving_quantities = rate_driving_quantities(
        reaction,
        concentrations_mol_L,
        activity_coefficients=activity_coefficients,
    )
    if (
        equation_id in {"pressure_log", "chebyshev"}
        or params.get("reverse_basis") == "ideal_gas_nasa7"
    ):
        # Only explicitly imported ideal-gas laws infer pressure from inventory.
        # Liquid concentration kinetics retain their existing standard-state contract.
        k = gas_forward_rate_constant(reaction, concentrations_mol_L, temperature_K)
        forward = mass_action_rate(reaction.kinetic_forward_orders, driving_quantities, k)
        if not reaction.reversible:
            return forward
        if params.get("reverse_basis") != "ideal_gas_nasa7" or species_thermo is None:
            raise ValueError("reversible gas kinetics require declared NASA7 thermochemistry")
        from chemworld.physchem.reaction_network import (
            thermochemical_concentration_equilibrium_constant,
        )

        standard_pressure = float(params["reference_pressure_Pa"])
        if any(
            value.reference_pressure_Pa != standard_pressure for value in species_thermo.values()
        ):
            raise ValueError("gas thermochemistry reference pressures must agree")
        kc, _, _ = thermochemical_concentration_equilibrium_constant(
            reaction,
            species_thermo=species_thermo,
            temperature_K=temperature_K,
            standard_concentration_mol_L=standard_pressure / (1000 * R_J_PER_MOL_K * temperature_K),
        )
        return forward - mass_action_rate(
            reaction.kinetic_reverse_orders, driving_quantities, k / kc
        )
    forward_enabled = all(
        concentrations_mol_L.get(species_id, 0.0) > 0.0 for species_id in reaction.reactants
    )
    if equation_id != "reversible_arrhenius" and not forward_enabled:
        return 0.0
    if equation_id == "mass_action":
        return mass_action_rate(
            reaction.kinetic_forward_orders,
            driving_quantities,
            float(params["k"]),
        )
    if equation_id in {"arrhenius", "modified_arrhenius"}:
        k = arrhenius_k(params, temperature_K)
        return mass_action_rate(reaction.kinetic_forward_orders, driving_quantities, k)
    if equation_id == "third_body_arrhenius":
        k = arrhenius_k(params, temperature_K)
        third_body = effective_third_body_concentration(
            concentrations_mol_L,
            efficiencies=third_body_efficiencies(params),
            default_efficiency=float_param(params, "default_efficiency", default=1.0),
        )
        return mass_action_rate(
            reaction.kinetic_forward_orders,
            driving_quantities,
            k * third_body,
        )
    if equation_id == "lindemann_falloff":
        k = lindemann_falloff_rate_constant(params, concentrations_mol_L, temperature_K)
        return mass_action_rate(reaction.kinetic_forward_orders, driving_quantities, k)
    if equation_id == "troe_falloff":
        k = troe_falloff_rate_constant(params, concentrations_mol_L, temperature_K)
        return mass_action_rate(reaction.kinetic_forward_orders, driving_quantities, k)
    if equation_id == "reversible_arrhenius":
        forward_rate_constant = arrhenius_k(params, temperature_K)
        forward = (
            mass_action_rate(
                reaction.kinetic_forward_orders,
                driving_quantities,
                forward_rate_constant,
            )
            if forward_enabled
            else 0.0
        )
        reverse_k = reverse_rate_constant(
            reaction,
            params,
            temperature_K=temperature_K,
            forward_rate_constant=forward_rate_constant,
            species_thermo=species_thermo,
            thermochemical_reverse_rate_constant=thermochemical_reverse_rate_constant,
        )
        reverse_enabled = all(
            concentrations_mol_L.get(species_id, 0.0) > 0.0 for species_id in reaction.products
        )
        reverse = (
            mass_action_rate(
                reaction.kinetic_reverse_orders,
                driving_quantities,
                reverse_k,
            )
            if reverse_enabled
            else 0.0
        )
        return forward - reverse
    if equation_id == "catalytic_activity":
        base = mass_action_rate(
            reaction.kinetic_forward_orders,
            driving_quantities,
            arrhenius_k(params, temperature_K),
        )
        catalyst_species = str(params.get("catalyst_species", ""))
        reference = float(params.get("reference_concentration_mol_L", 1.0))
        exponent = float(params.get("activity_order", 1.0))
        if not catalyst_species:
            raise ValueError("catalytic_activity requires catalyst_species")
        if reference <= 0.0 or not isfinite(reference):
            raise ValueError("reference_concentration_mol_L must be finite and positive")
        activity = (
            max(driving_quantities.get(catalyst_species, 0.0), 0.0)
            if reaction.rate_law.uses_activities
            else max(concentrations_mol_L.get(catalyst_species, 0.0), 0.0) / reference
        )
        return base * activity**exponent
    if equation_id == "catalyst_deactivation":
        species = str(params.get("species", ""))
        if not species:
            raise ValueError("catalyst_deactivation requires species")
        return arrhenius_k(params, temperature_K) * max(
            concentrations_mol_L.get(species, 0.0),
            0.0,
        )
    if equation_id == "langmuir_hinshelwood":
        k = arrhenius_k(params, temperature_K)
        numerator = mass_action_rate(
            reaction.kinetic_forward_orders,
            driving_quantities,
            k,
        )
        adsorption = params.get("adsorption", {})
        if not isinstance(adsorption, dict):
            raise ValueError("langmuir_hinshelwood adsorption must be a mapping")
        denominator = 1.0 + sum(
            float(K) * max(driving_quantities.get(str(species_id), 0.0), 0.0)
            for species_id, K in adsorption.items()
        )
        power = float(params.get("denominator_power", 1.0))
        return numerator / denominator**power
    if equation_id == "michaelis_menten":
        substrate = str(params["substrate"])
        concentration = max(concentrations_mol_L.get(substrate, 0.0), 0.0)
        return float(params["vmax"]) * concentration / (float(params["Km"]) + concentration)
    raise ValueError(f"Unsupported rate law: {equation_id}")


def validate_pressure_rate(parameters: Mapping[str, Any], equation_id: str) -> None:
    """Strict finite-domain contract shared by import and evaluation."""
    for key in ("temperature_range_K", "pressure_range_Pa"):
        values = parameters.get(key, ())
        if (
            not isinstance(values, (list, tuple))
            or len(values) != 2
            or any(not isfinite(float(v)) or float(v) <= 0 for v in values)
            or float(values[0]) >= float(values[1])
        ):
            raise ValueError(f"{key} requires two ordered positive finite endpoints")
    if equation_id == "pressure_log":
        terms = parameters.get("rate_constants", ())
        if not isinstance(terms, (list, tuple)) or not terms:
            raise ValueError("P-log requires rate_constants")
        pressures = []
        for term in terms:
            if not isinstance(term, Mapping) or set(term) != {
                "pressure_Pa",
                "A",
                "b",
                "Ea_J_per_mol",
            }:
                raise ValueError("P-log term must declare pressure_Pa, A, b, Ea_J_per_mol")
            pressure = float(term["pressure_Pa"])
            if not isfinite(pressure) or pressure <= 0 or float(term["A"]) <= 0:
                raise ValueError("finite P-log slice requires positive pressure and A")
            arrhenius_k(term, float(parameters["temperature_range_K"][0]))
            pressures.append(pressure)
        if [min(pressures), max(pressures)] != list(parameters["pressure_range_Pa"]):
            raise ValueError("P-log pressure domain must match its tabulated endpoints")
    elif equation_id == "chebyshev":
        rows = parameters.get("coefficients", ())
        if (
            not isinstance(rows, (list, tuple))
            or not rows
            or any(not isinstance(row, (list, tuple)) or not row for row in rows)
            or len({len(row) for row in rows}) != 1
            or any(not isfinite(float(x)) for row in rows for x in row)
        ):
            raise ValueError("Chebyshev coefficients must be a finite rectangular matrix")
    else:
        raise ValueError("unsupported pressure rate law")


def pressure_rate_constant(
    parameters: Mapping[str, Any], equation_id: str, temperature_K: float, pressure_Pa: float
) -> float:
    """P-log / Chebyshev in native mol/L, s, K, Pa; never extrapolate."""
    validate_pressure_rate(parameters, equation_id)
    resolved = []
    for value, key in ((temperature_K, "temperature_range_K"), (pressure_Pa, "pressure_range_Pa")):
        lower, upper = parameters[key]
        # nRT/V reconstructed from a declared endpoint can differ by a few ULPs.
        # This is arithmetic boundary normalization, not physical extrapolation.
        if isfinite(value) and lower - 8 * ulp(float(lower)) <= value < lower:
            value = lower
        if isfinite(value) and upper < value <= upper + 8 * ulp(float(upper)):
            value = upper
        if not isfinite(value) or not lower <= value <= upper:
            raise ValueError(f"{key}: state outside declared validity domain")
        resolved.append(value)
    temperature_K, pressure_Pa = resolved
    if equation_id == "pressure_log":
        grouped: dict[float, float] = {}
        for term in parameters["rate_constants"]:
            pressure = float(term["pressure_Pa"])
            grouped[pressure] = grouped.get(pressure, 0.0) + arrhenius_k(term, temperature_K)
        if pressure_Pa in grouped:
            return grouped[pressure_Pa]
        pressures = sorted(grouped)
        for low, high in pairwise(pressures):
            if low < pressure_Pa < high:
                weight = (log10(pressure_Pa) - log10(low)) / (log10(high) - log10(low))
                return 10 ** ((1 - weight) * log10(grouped[low]) + weight * log10(grouped[high]))
        raise ValueError("P-log pressure cannot be bracketed")
    t_min, t_max = parameters["temperature_range_K"]
    p_min, p_max = parameters["pressure_range_Pa"]
    t_red = (2 / temperature_K - 1 / t_min - 1 / t_max) / (1 / t_max - 1 / t_min)
    p_red = (2 * log10(pressure_Pa) - log10(p_min) - log10(p_max)) / (log10(p_max) - log10(p_min))

    def polynomials(x, count):
        values = [1.0, x]
        for _ in range(2, count):
            values.append(2 * x * values[-1] - values[-2])
        return values[:count]

    rows = parameters["coefficients"]
    ts, ps = polynomials(t_red, len(rows)), polynomials(p_red, len(rows[0]))
    value = 10 ** sum(
        float(a) * ts[i] * ps[j] for i, row in enumerate(rows) for j, a in enumerate(row)
    )
    if not isfinite(value) or value <= 0:
        raise ValueError("Chebyshev rate must be finite and positive")
    return value


def gas_forward_rate_constant(
    reaction: ReactionSpec, concentrations: Mapping[str, float], temperature_K: float
) -> float:
    params, law = reaction.rate_law.parameters, reaction.rate_law.equation_id
    if params.get("mechanism_schema") == "cantera-gas-v2":
        from chemworld.physchem.mechanism_rates import mechanism_forward_rate_constant

        return mechanism_forward_rate_constant(reaction, concentrations, temperature_K)
    if law in {"pressure_log", "chebyshev"}:
        if params.get("pressure_basis") != "ideal_gas":
            raise ValueError("network pressure kinetics require explicit ideal_gas pressure basis")
        pressure = sum(concentrations.values()) * 1000 * R_J_PER_MOL_K * temperature_K
        return pressure_rate_constant(params, law, temperature_K, pressure)
    if law == "troe_falloff":
        return troe_falloff_rate_constant(params, concentrations, temperature_K)
    if law == "lindemann_falloff":
        return lindemann_falloff_rate_constant(params, concentrations, temperature_K)
    if law not in {"arrhenius", "modified_arrhenius", "third_body_arrhenius"}:
        raise ValueError("unsupported ideal-gas forward law")
    k = arrhenius_k(params, temperature_K)
    if law == "third_body_arrhenius":
        k *= effective_third_body_concentration(
            concentrations,
            efficiencies=third_body_efficiencies(params),
            default_efficiency=float(params.get("default_efficiency", 1)),
        )
    return k


def mass_action_rate(
    reactants: Mapping[str, float],
    concentrations_mol_L: Mapping[str, float],
    k: float,
    *,
    allow_signed_trial: bool = False,
) -> float:
    if k < 0 or not isfinite(k):
        raise ValueError("Rate coefficient must be finite and nonnegative")
    rate = k
    if type(allow_signed_trial) is not bool:
        raise ValueError("signed mass-action trial policy must be explicitly Boolean")
    for species_id, order in reactants.items():
        value = float(concentrations_mol_L.get(species_id, 0.0))
        if allow_signed_trial:
            if (not isfinite(value) or isinstance(order, bool)
                    or not isfinite(order) or order < 0 or not float(order).is_integer()):
                raise ValueError("signed mass-action trials require finite integer powers")
            concentration = value
        else:
            concentration = max(value, 0.0)
        rate *= concentration**order
    return rate


def rate_driving_quantities(
    reaction: ReactionSpec,
    concentrations_mol_L: Mapping[str, float],
    *,
    activity_coefficients: Mapping[str, float] | None = None,
) -> dict[str, float]:
    """Return either concentration powers or dimensionless activities.

    Activity-based laws use ``a_i = gamma_i C_i / C_standard``.  The explicit
    switch prevents silently mixing activity and concentration rate constants.
    """

    quantities: dict[str, float] = {}
    coefficients = {} if activity_coefficients is None else activity_coefficients
    standard = reaction.rate_law.standard_concentration_mol_L
    for species_id, concentration in concentrations_mol_L.items():
        value = float(concentration)
        if value < 0.0 or not isfinite(value):
            raise ValueError(f"concentration for {species_id!r} must be finite and nonnegative")
        if not reaction.rate_law.uses_activities:
            quantities[species_id] = value
            continue
        coefficient = float(coefficients.get(species_id, 1.0))
        if coefficient <= 0.0 or not isfinite(coefficient):
            raise ValueError(f"activity coefficient for {species_id!r} must be finite and positive")
        quantities[species_id] = coefficient * value / standard
    return quantities


def _declared_temperature_validity(params: Mapping[str, Any], temperature_K: float) -> None:
    """Optional source validity survives generation/export, including zero reactant flux."""
    if "valid_temperature_range_K" not in params:
        return
    bounds = params["valid_temperature_range_K"]
    if (not isinstance(bounds, (tuple, list)) or len(bounds) != 2
            or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not isfinite(v)
                   for v in bounds) or not 0 < bounds[0] < bounds[1]):
        raise ValueError("invalid explicitly declared kinetic temperature range")
    if not isfinite(temperature_K) or not bounds[0] <= temperature_K <= bounds[1]:
        raise ValueError("temperature outside explicitly declared kinetic validity")


def arrhenius_k(params: Mapping[str, object], temperature_K: float) -> float:
    _declared_temperature_validity(params, temperature_K)
    if temperature_K <= 0.0 or not isfinite(temperature_K):
        raise ValueError("temperature_K must be finite and positive")
    A = float_param(params, "A", default=float_param(params, "k", default=0.0))
    b = float_param(params, "b", default=0.0)
    Ea = float_param(
        params,
        "Ea_J_per_mol",
        default=float_param(params, "Ea", default=0.0),
    )
    if A < 0.0 or not isfinite(A):
        raise ValueError("Arrhenius A must be finite and nonnegative")
    if not isfinite(b):
        raise ValueError("Arrhenius b must be finite")
    if not isfinite(Ea):
        raise ValueError("Arrhenius activation energy must be finite")
    value = A * temperature_K**b * exp(-Ea / (R_J_PER_MOL_K * temperature_K))
    if value < 0.0 or not isfinite(value):
        raise ValueError("Arrhenius rate coefficient is not finite and nonnegative")
    return value


def effective_third_body_concentration(
    concentrations_mol_L: Mapping[str, float],
    *,
    efficiencies: Mapping[str, float] | None = None,
    default_efficiency: float = 1.0,
) -> float:
    """Return effective third-body concentration ``[M]_eff``.

    The compact slice follows the usual collision-efficiency form:

    ``[M]_eff = sum_i alpha_i C_i``.
    """

    if default_efficiency < 0.0 or not isfinite(default_efficiency):
        raise ValueError("default_efficiency must be finite and nonnegative")
    efficiency_map = {} if efficiencies is None else dict(efficiencies)
    total = 0.0
    for species_id, concentration in concentrations_mol_L.items():
        value = max(float(concentration), 0.0)
        efficiency = float(efficiency_map.get(species_id, default_efficiency))
        if efficiency < 0.0 or not isfinite(efficiency):
            raise ValueError(f"third-body efficiency for {species_id!r} is invalid")
        total += efficiency * value
    return total


def third_body_efficiencies(params: Mapping[str, object]) -> dict[str, float]:
    """Parse ``third_body_efficiencies`` from rate-law parameters."""

    payload = params.get("third_body_efficiencies", params.get("efficiencies", {}))
    if payload is None:
        return {}
    if not isinstance(payload, Mapping):
        raise ValueError("third_body_efficiencies must be a mapping")
    result: dict[str, float] = {}
    for species_id, efficiency in payload.items():
        value = float(efficiency)
        if value < 0.0 or not isfinite(value):
            raise ValueError(
                f"third-body efficiency for {species_id!r} must be finite and nonnegative"
            )
        result[str(species_id)] = value
    return result


def lindemann_falloff_rate_constant(
    params: Mapping[str, object],
    concentrations_mol_L: Mapping[str, float],
    temperature_K: float,
) -> float:
    """Return Lindemann pressure-dependent effective rate constant.

    ``k_eff = k_inf * Pr / (1 + Pr)``, where ``Pr = k0 [M]_eff / k_inf``.
    """

    if temperature_K <= 0.0:
        raise ValueError("temperature_K must be positive")
    low_k = arrhenius_k(prefixed_arrhenius_params(params, "low"), temperature_K)
    high_k = arrhenius_k(prefixed_arrhenius_params(params, "high"), temperature_K)
    if high_k <= 0.0 or not isfinite(high_k):
        raise ValueError("high-pressure Arrhenius rate must be finite and positive")
    third_body = effective_third_body_concentration(
        concentrations_mol_L,
        efficiencies=third_body_efficiencies(params),
        default_efficiency=float_param(params, "default_efficiency", default=1.0),
    )
    if third_body <= 0.0:
        return 0.0
    reduced_pressure = low_k * third_body / high_k
    if reduced_pressure < 0.0 or not isfinite(reduced_pressure):
        raise ValueError("falloff reduced pressure must be finite and nonnegative")
    return high_k * reduced_pressure / (1.0 + reduced_pressure)


def troe_falloff_rate_constant(
    params: Mapping[str, object],
    concentrations_mol_L: Mapping[str, float],
    temperature_K: float,
) -> float:
    """Return Troe-broadened falloff effective rate constant."""

    lindemann = lindemann_falloff_rate_constant(
        params,
        concentrations_mol_L,
        temperature_K,
    )
    if lindemann <= 0.0:
        return 0.0
    reduced_pressure = falloff_reduced_pressure(
        params,
        concentrations_mol_L,
        temperature_K,
    )
    return lindemann * troe_broadening_factor(params, temperature_K, reduced_pressure)


def falloff_reduced_pressure(
    params: Mapping[str, object],
    concentrations_mol_L: Mapping[str, float],
    temperature_K: float,
) -> float:
    """Return ``Pr = k0 [M]_eff / k_inf`` for falloff rate laws."""

    low_k = arrhenius_k(prefixed_arrhenius_params(params, "low"), temperature_K)
    high_k = arrhenius_k(prefixed_arrhenius_params(params, "high"), temperature_K)
    if high_k <= 0.0 or not isfinite(high_k):
        raise ValueError("high-pressure Arrhenius rate must be finite and positive")
    third_body = effective_third_body_concentration(
        concentrations_mol_L,
        efficiencies=third_body_efficiencies(params),
        default_efficiency=float_param(params, "default_efficiency", default=1.0),
    )
    return low_k * third_body / high_k


def troe_broadening_factor(
    params: Mapping[str, object],
    temperature_K: float,
    reduced_pressure: float,
) -> float:
    """Return compact Troe broadening factor ``F``.

    The implementation uses the common four-parameter form with optional
    ``troe_T2``. It is intentionally small but keeps all numerical choices
    explicit for model-card validation.
    """

    if temperature_K <= 0.0:
        raise ValueError("temperature_K must be positive")
    if reduced_pressure < 0.0 or not isfinite(reduced_pressure):
        raise ValueError("reduced_pressure must be finite and nonnegative")
    if reduced_pressure == 0.0:
        return 1.0
    alpha = float_param(params, "troe_a")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("troe_a must lie in [0, 1]")
    t3 = float_param(params, "troe_T3")
    t1 = float_param(params, "troe_T1")
    if t3 <= 0.0 or t1 <= 0.0:
        raise ValueError("troe_T1 and troe_T3 must be positive")
    f_cent = (1.0 - alpha) * exp(-temperature_K / t3) + alpha * exp(-temperature_K / t1)
    if "troe_T2" in params:
        t2 = float_param(params, "troe_T2")
        if t2 <= 0.0:
            raise ValueError("troe_T2 must be positive when provided")
        f_cent += exp(-t2 / temperature_K)
    if f_cent <= 0.0 or not isfinite(f_cent):
        raise ValueError("Troe F_cent must be finite and positive")
    log_f_cent = log10(f_cent)
    log_pr = log10(max(reduced_pressure, 1.0e-300))
    c_value = -0.4 - 0.67 * log_f_cent
    n_value = 0.75 - 1.27 * log_f_cent
    denominator = n_value - 0.14 * (log_pr + c_value)
    if abs(denominator) < 1.0e-300:
        raise ValueError("Troe denominator is numerically singular")
    broadening_log = log_f_cent / (1.0 + ((log_pr + c_value) / denominator) ** 2)
    factor = 10.0**broadening_log
    if factor <= 0.0 or not isfinite(factor):
        raise ValueError("Troe broadening factor must be finite and positive")
    return factor


def prefixed_arrhenius_params(
    params: Mapping[str, object],
    prefix: str,
) -> dict[str, object]:
    """Extract low/high Arrhenius parameter groups from a falloff law."""

    aliases = {
        "low": {
            "A": ("low_A", "A0", "A_low"),
            "b": ("low_b", "b0", "b_low"),
            "Ea_J_per_mol": (
                "low_Ea_J_per_mol",
                "Ea0_J_per_mol",
                "Ea_low_J_per_mol",
                "low_Ea",
                "Ea0",
                "Ea_low",
            ),
        },
        "high": {
            "A": ("high_A", "A_inf", "A_high", "Ainf"),
            "b": ("high_b", "b_inf", "b_high", "binf"),
            "Ea_J_per_mol": (
                "high_Ea_J_per_mol",
                "Ea_inf_J_per_mol",
                "Ea_high_J_per_mol",
                "high_Ea",
                "Ea_inf",
                "Ea_high",
                "Eainf",
            ),
        },
    }
    if prefix not in aliases:
        raise ValueError(f"unknown Arrhenius prefix: {prefix}")
    extracted: dict[str, object] = {}
    for target_key, candidate_keys in aliases[prefix].items():
        for candidate in candidate_keys:
            if candidate in params:
                extracted[target_key] = params[candidate]
                break
        if target_key == "A" and target_key not in extracted:
            raise ValueError(f"{prefix}-pressure Arrhenius group requires A")
    extracted.setdefault("b", 0.0)
    extracted.setdefault("Ea_J_per_mol", 0.0)
    return extracted


def reaction_order_delta(reaction: ReactionSpec) -> float:
    """Return stoichiometric ``sum(nu_i)`` for thermodynamic Kc conversion."""

    return sum(reaction.stoichiometry.values())


def reverse_rate_constant(
    reaction: ReactionSpec,
    params: Mapping[str, object],
    *,
    temperature_K: float,
    forward_rate_constant: float,
    species_thermo: Mapping[str, Any] | None,
    thermochemical_reverse_rate_constant: ThermochemicalReverseRate | None,
) -> float:
    if "A_reverse" in params:
        return arrhenius_k(reverse_params(params), temperature_K)
    if "K_eq" in params:
        K_eq = float_param(params, "K_eq")
        if K_eq <= 0:
            raise ValueError("K_eq must be positive")
        return reverse_rate_constant_from_equilibrium(
            forward_rate_constant=forward_rate_constant,
            concentration_equilibrium_constant=K_eq,
        )
    source = str(params.get("K_eq_source", params.get("equilibrium_source", ""))).lower()
    if source in {"nasa7", "species_thermo", "thermochemistry"}:
        if species_thermo is None:
            raise ValueError("NASA7 reversible_arrhenius requires species_thermo")
        if thermochemical_reverse_rate_constant is None:
            raise ValueError(
                "NASA7 reversible_arrhenius requires thermochemical reverse-rate callback"
            )
        return thermochemical_reverse_rate_constant(
            reaction,
            params,
            temperature_K,
            species_thermo,
        )
    raise ValueError("reversible_arrhenius requires A_reverse, K_eq, or K_eq_source='nasa7'")


def reverse_params(params: Mapping[str, object]) -> dict[str, object]:
    if "A_reverse" in params:
        return {
            "A": params["A_reverse"],
            "b": params.get("b_reverse", 0.0),
            "Ea_J_per_mol": params.get(
                "Ea_reverse_J_per_mol",
                params.get("Ea_reverse", 0.0),
            ),
        }
    if "K_eq" in params:
        K_eq = float_param(params, "K_eq")
        if K_eq <= 0:
            raise ValueError("K_eq must be positive")
        return {
            "A": float_param(params, "A", default=0.0) / K_eq,
            "b": params.get("b", 0.0),
            "Ea_J_per_mol": params.get("Ea_J_per_mol", params.get("Ea", 0.0)),
        }
    raise ValueError("reversible_arrhenius requires A_reverse or K_eq")


def reverse_rate_constant_from_equilibrium(
    *,
    forward_rate_constant: float,
    concentration_equilibrium_constant: float,
) -> float:
    """Return ``k_reverse = k_forward / K_c`` with explicit validation."""

    if forward_rate_constant < 0 or not isfinite(forward_rate_constant):
        raise ValueError("forward_rate_constant must be finite and nonnegative")
    if concentration_equilibrium_constant <= 0 or not isfinite(concentration_equilibrium_constant):
        raise ValueError("concentration_equilibrium_constant must be finite and positive")
    return forward_rate_constant / concentration_equilibrium_constant


def positive_reaction_parameter(
    network: ReactionNetworkLike,
    reaction_id: str,
    parameter_name: str,
) -> float:
    reaction = reaction_by_id(network, reaction_id)
    if parameter_name not in reaction.rate_law.parameters:
        raise ValueError(f"Reaction {reaction_id!r} has no parameter {parameter_name!r}")
    value = reaction.rate_law.parameters[parameter_name]
    if not isinstance(value, int | float | str):
        raise ValueError(f"Reaction parameter {reaction_id}.{parameter_name} must be numeric")
    numeric = float(value)
    if numeric <= 0 or not isfinite(numeric):
        raise ValueError(
            f"Reaction parameter {reaction_id}.{parameter_name} must be finite and positive"
        )
    return numeric


def with_reaction_parameter(
    network: NetworkT,
    reaction_id: str,
    parameter_name: str,
    value: float,
) -> NetworkT:
    if value <= 0 or not isfinite(value):
        raise ValueError("perturbed reaction parameter must be finite and positive")
    reactions: list[ReactionSpec] = []
    found = False
    for reaction in network.reactions:
        if reaction.reaction_id != reaction_id:
            reactions.append(reaction)
            continue
        found = True
        if parameter_name not in reaction.rate_law.parameters:
            raise ValueError(f"Reaction {reaction_id!r} has no parameter {parameter_name!r}")
        params = dict(reaction.rate_law.parameters)
        params[parameter_name] = value
        reactions.append(
            replace(
                reaction,
                rate_law=replace(reaction.rate_law, parameters=params),
            )
        )
    if not found:
        raise ValueError(f"Unknown reaction id: {reaction_id}")
    return cast(NetworkT, replace(cast(Any, network), reactions=tuple(reactions)))


def reaction_by_id(network: ReactionNetworkLike, reaction_id: str) -> ReactionSpec:
    for reaction in network.reactions:
        if reaction.reaction_id == reaction_id:
            return reaction
    raise ValueError(f"Unknown reaction id: {reaction_id}")


def float_param(
    params: Mapping[str, object],
    key: str,
    *,
    default: float | None = None,
) -> float:
    if key not in params:
        if default is None:
            raise ValueError(f"Missing numeric rate-law parameter: {key}")
        return default
    value = params[key]
    if isinstance(value, int | float | str):
        return float(value)
    raise ValueError(f"Rate-law parameter {key} must be numeric, got {type(value).__name__}")


__all__ = [
    "R_J_PER_MOL_K",
    "ReactionNetworkLike",
    "ThermochemicalReverseRate",
    "arrhenius_k",
    "effective_third_body_concentration",
    "evaluate_rate_law",
    "falloff_reduced_pressure",
    "float_param",
    "lindemann_falloff_rate_constant",
    "mass_action_rate",
    "positive_reaction_parameter",
    "prefixed_arrhenius_params",
    "rate_driving_quantities",
    "reaction_by_id",
    "reaction_order_delta",
    "reverse_params",
    "reverse_rate_constant",
    "reverse_rate_constant_from_equilibrium",
    "third_body_efficiencies",
    "troe_broadening_factor",
    "troe_falloff_rate_constant",
    "with_reaction_parameter",
]
