"""Axial multisolute RO on original G/H/V and exactly neutral ionic fluxes."""

from collections.abc import Mapping
from dataclasses import replace
from itertools import pairwise
from math import exp, isfinite
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from scipy.optimize import brentq, least_squares

from chemworld.physchem.elements import molecular_weight
from chemworld.physchem.multisolute_ro_properties import MultisoluteROProperties
from chemworld.physchem.pressure_recovery import evaluate_isothermal_liquid_pressure_recovery
from chemworld.physchem.solver_backend import ODESolverPolicy, solve_ode
from chemworld.physchem.transport import FluidState, pipe_pressure_drop, pump_operating_point

if TYPE_CHECKING:
    from chemworld.physchem.distributed_ro import DistributedROSpec


def local_multisolute_ro_flux(
    spec: "DistributedROSpec",
    concentrations_mol_m3: Mapping[str, float],
    pressure_Pa: float,
    *,
    film_coefficients_m_s: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    prop = spec.properties
    if not isinstance(prop, MultisoluteROProperties):
        raise ValueError("multiple-channel flux needs declared multisolute properties")
    cb = dict(concentrations_mol_m3)
    if (
        set(cb) != {c.channel_id for c in prop.channels}
        or any(not isfinite(v) or v < 0 for v in cb.values())
        or sum(cb.values()) > spec.maximum_interface_concentration_mol_m3
        or pressure_Pa < spec.reservoir_pressure_Pa
    ):
        raise ValueError("multi-solute RO inputs lie outside the declared forward domain")
    t, vw, total = spec.temperature_K, prop.water_volume_m3_mol, sum(cb.values())
    a = spec.water_permeability.evaluate(t, pressure_Pa, total)
    b = {
        c.channel_id: c.permeability.evaluate(t, pressure_Pa, total) if c.permeability else 0.0
        for c in prop.channels
    }
    k = {c.channel_id: c.mass_transfer.evaluate(t, pressure_Pa, total) for c in prop.channels}
    if film_coefficients_m_s is not None:
        if set(film_coefficients_m_s) != set(k) or any(
            not isfinite(v) or v <= 0 for v in film_coefficients_m_s.values()
        ):
            raise ValueError(
                "calculated film coefficients require every channel and positive values"
            )
        k = dict(film_coefficients_m_s)
    if prop.property_approximation == "mcas-ideal-volume":
        from chemworld.physchem.mcas_ro_flux import local_mcas_ro_flux

        return local_mcas_ro_flux(spec, cb, pressure_Pa, k)

    def evaluate(j: float) -> dict[str, Any]:
        cp = {
            key: value * b[key] / (b[key] + j * exp(-j / k[key])) if b[key] else 0.0
            for key, value in cb.items()
        }
        cm = {
            key: cp[key] * (1 + j / b[key]) if b[key] else value * exp(j / k[key])
            for key, value in cb.items()
        }
        if sum(cm.values()) > spec.maximum_interface_concentration_mol_m3:
            raise ValueError("polarized multisolute interface exceeds its declared domain")
        nf, np_ = (
            prop.amounts(1.0, {key: c * vw for key, c in values.items()}) for values in (cm, cp)
        )
        mf = prop.database.chemical_potentials(nf, temperature_K=t, pressure_Pa=pressure_Pa)
        mp = prop.database.chemical_potentials(
            np_, temperature_K=t, pressure_Pa=spec.reservoir_pressure_Pa
        )
        delta = cast(float, mf.chemical_potentials_J_mol[prop.water_id]) - cast(
            float, mp.chemical_potentials_J_mol[prop.water_id]
        )
        species_flux = prop.amounts(j / vw, {key: j * c for key, c in cp.items()})
        return {
            "water_flux_m_s": j,
            "component_flux_mol_m2_s": species_flux,
            "channel_flux_mol_m2_s": {key: j * c for key, c in cp.items()},
            "film_coefficients_m_s": k,
            "interface_concentrations_mol_m3": cm,
            "permeate_concentrations_mol_m3": cp,
            "water_chemical_potential_difference_J_mol": delta,
            "driving_pressure_Pa": delta / vw,
            "water_flux_residual_m_s": j - a * delta / vw,
            "charge_flux_mol_m2_s": sum(
                prop.charges[key] * species_flux[key] for key in prop.solute_ids
            ),
        }

    zero = evaluate(0.0)
    if zero["driving_pressure_Pa"] <= 0:
        return {**zero, "stopping_event": "zero_or_reverse_water_drive"}
    upper = min(max(2 * a * zero["driving_pressure_Pa"], 1e-12), 40 * min(k.values()))
    lower, boundary = 0.0, upper
    for _ in range(80):
        try:
            if evaluate(upper)["water_flux_residual_m_s"] >= 0:
                break
            lower = upper
        except ValueError:
            boundary = upper
        upper = (lower + boundary) / 2
    else:
        raise ValueError("forward multi-solute root lies beyond its declared domain")
    j = brentq(
        lambda value: evaluate(value)["water_flux_residual_m_s"], 0.0, upper, xtol=1e-15, rtol=1e-11
    )
    return {**evaluate(j), "stopping_event": None}


def multisolute_ro_profile(
    spec: "DistributedROSpec",
    feed_amounts: Mapping[str, float],
    requested_flow_m3_s: float,
    *,
    samples: int = 17,
    rtol: float = 1e-7,
) -> dict[str, Any]:
    prop = spec.properties
    if not isinstance(prop, MultisoluteROProperties):
        raise ValueError("multi-solute profile needs multisolute properties")
    prop.validate_amounts(feed_amounts)
    if type(samples) is not int or samples < 2:
        raise ValueError("axial reporting needs at least two stations")
    t, pr = spec.temperature_K, spec.reservoir_pressure_Pa
    point = pump_operating_point(
        spec.pump, requested_flow_m3_s, lambda _: spec.inlet_pressure_Pa - pr
    )
    q = point["actual_flow_m3_s"]
    incoming = {key: n * q / prop.volume(feed_amounts, t, pr) for key, n in feed_amounts.items()}
    channels = tuple(c.channel_id for c in prop.channels)
    w0 = incoming[prop.water_id]
    s0 = prop.channel_amounts(incoming)
    mass = {
        s.species_id: molecular_weight(dict(s.composition)) / 1000 for s in prop.database.species
    }
    unit_pipe = replace(spec.channel, length_m=1.0)
    if q:
        scales = np.array([w0, *(max(s0[key], w0 * 1e-5) for key in channels), 1e6])

        def station(vector: Any) -> dict[str, Any]:
            raw = vector * scales
            w, p = float(raw[0]), float(raw[-1])
            if w <= w0 * 1e-5 or min(raw[1:-1]) < 0:
                raise ValueError("axial multi-solute recovery exhausted the feed stream")
            n = prop.amounts(w, {key: float(v) for key, v in zip(channels, raw[1:-1], strict=True)})
            liquid = prop.liquid(n, t, p)
            cb = prop.channel_concentrations(n)
            if prop.property_approximation == "mcas-ideal-volume":
                cb = {key: v / liquid.volume_m3 for key, v in prop.channel_amounts(n).items()}
            fluid = FluidState(
                sum(n[key] * mass[key] for key in n) / liquid.volume_m3,
                prop.viscosity.evaluate(t, p, sum(cb.values())),
                1.0,
                1.0,
            )
            film = None
            if spec.flat_sheet_channel is None:
                gradient = pipe_pressure_drop(
                    unit_pipe, fluid, volumetric_flow_m3_s=liquid.volume_m3
                ).pressure_drop_total_Pa
            else:
                geometry = spec.flat_sheet_channel
                h, epsilon = geometry.height_m, geometry.spacer_porosity
                dh = 4 * epsilon / (2 / h + (1 - epsilon) * 8 / h)
                cross = h * spec.membrane_area_m2 / spec.channel.length_m * epsilon
                rho, mu = fluid.density_kg_m3, fluid.viscosity_Pa_s
                re = liquid.volume_m3 * rho * dh / (cross * mu)
                film = {}
                for key, law in geometry.diffusivities:
                    diffusion = law.evaluate(t, p, sum(cb.values()))
                    film[key] = diffusion * 0.46 * (re * mu / (rho * diffusion)) ** 0.36 / dh
                gradient = (0.42 + 189.3 / re) * rho * (liquid.volume_m3 / cross) ** 2 / (2 * dh)
            return {
                **local_multisolute_ro_flux(spec, cb, p, film_coefficients_m_s=film),
                "component_mol_s": n,
                "bulk_concentrations_mol_m3": cb,
                "pressure_Pa": p,
                "volume_flow_m3_s": liquid.volume_m3,
                "pressure_gradient_Pa_m": gradient,
            }

        def rhs(x: float, y: Any) -> Any:
            row = station(y)
            return (
                np.array(
                    [
                        -spec.membrane_area_m2 * row["component_flux_mol_m2_s"][prop.water_id],
                        *(
                            -spec.membrane_area_m2 * row["channel_flux_mol_m2_s"][key]
                            for key in channels
                        ),
                        -spec.channel.length_m * row["pressure_gradient_Pa_m"],
                    ]
                )
                / scales
            )

        initial = np.array([w0, *(s0[key] for key in channels), spec.inlet_pressure_Pa]) / scales
        if spec.flat_sheet_channel is None:
            solved = solve_ode(
                rhs,
                initial,
                time_span_s=(0.0, 1.0),
                evaluation_times_s=np.linspace(0.0, 1.0, samples),
                policy=ODESolverPolicy(
                    "axial_multisolute_ro_v1", "DOP853", rtol, 1e-10, max_step_s=0.125
                ),
            )
            solved.raise_for_failure("distributed multisolute reverse osmosis")
            coordinates, vectors = solved.t, list(solved.y.T)
            diagnostic = solved.diagnostic.to_dict()
        else:
            coordinates = np.round(
                np.linspace(0.0, 1.0, spec.flat_sheet_channel.finite_elements + 1), 6
            )
            vectors = [initial]
            evaluations = 0
            for left, right in pairwise(coordinates):
                previous = vectors[-1]
                lower = np.array([w0 * 1e-5, *([0.0] * len(channels)), pr]) / scales
                start = previous.copy()
                active = np.flatnonzero(previous > lower)

                def expand(y: Any, previous: Any = previous, active: Any = active) -> Any:
                    vector = previous.copy()
                    vector[active] = y
                    return vector

                def residual(
                    y: Any,
                    previous: Any = previous,
                    dx: float = right - left,
                    right: float = right,
                    active: Any = active,
                ) -> Any:
                    vector = expand(y)
                    return (vector - previous - dx * rhs(right, vector))[active]

                result = least_squares(
                    residual,
                    start[active],
                    bounds=(lower[active], previous[active]),
                    jac="3-point",
                    xtol=1e-14,
                    ftol=1e-14,
                    gtol=1e-14,
                    max_nfev=500,
                )
                error = max(abs(result.fun))
                if not result.success or error > 1e-8:
                    raise RuntimeError(f"multi-solute RO axial cell did not close: {error}")
                evaluations += result.nfev
                vectors.append(expand(result.x))
            diagnostic = {
                "method": "backward-finite-volume",
                "cells": len(coordinates) - 1,
                "residual_evaluations": evaluations,
            }
        stations = [
            {"axial_fraction": float(x), **station(y)}
            for x, y in zip(coordinates, vectors, strict=True)
        ]
        retained = stations[-1]["component_mol_s"]
        final_p, final_q = stations[-1]["pressure_Pa"], stations[-1]["volume_flow_m3_s"]
    else:
        retained, stations, final_p, final_q, diagnostic = incoming, [], pr, 0.0, {}
    permeate = {key: incoming[key] - retained[key] for key in incoming}
    if min(permeate.values()) < -1e-12:
        raise RuntimeError("axial RO generated negative component collection")
    result = {
        **point,
        "inlet_mol_s": incoming,
        "permeate_mol_s": permeate,
        "retentate_mol_s": retained,
        "stations": stations,
        "pressure_drop_Pa": spec.inlet_pressure_Pa - final_p if q else 0.0,
        "backpressure_dissipation_W": (final_p - pr) * final_q,
        "water_recovery": permeate[prop.water_id] / w0 if w0 else 0.0,
        "solver": diagnostic,
    }
    if spec.energy_recovery is not None:
        recovery = evaluate_isothermal_liquid_pressure_recovery(
            spec.energy_recovery,
            prop,
            retained,
            temperature_K=t,
            inlet_pressure_Pa=final_p,
            outlet_pressure_Pa=pr,
        )
        result["pressure_recovery"] = recovery
        result["backpressure_dissipation_W"] = recovery["irreversibility_W"]
    return result
