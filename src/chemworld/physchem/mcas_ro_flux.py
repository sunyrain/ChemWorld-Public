"""Explicit MCAS ideal solution-volume SD/film approximation, without a G claim."""

from collections.abc import Mapping
from math import expm1
from typing import TYPE_CHECKING, Any

from scipy.optimize import brentq

from chemworld.physchem.multisolute_ro_properties import MultisoluteROProperties
from chemworld.physchem.property_reports import R_J_PER_MOL_K as R

if TYPE_CHECKING:
    from chemworld.physchem.distributed_ro import DistributedROSpec


def local_mcas_ro_flux(
    spec: "DistributedROSpec",
    cb: Mapping[str, float],
    pressure_Pa: float,
    film: Mapping[str, float],
) -> dict[str, Any]:
    prop = spec.properties
    if (
        not isinstance(prop, MultisoluteROProperties)
        or prop.property_approximation != "mcas-ideal-volume"
    ):
        raise ValueError("MCAS flux requires an explicit independent property approximation")
    t, total = spec.temperature_K, sum(cb.values())
    a = spec.water_permeability.evaluate(t, pressure_Pa, total)
    b = {
        c.channel_id: c.permeability.evaluate(t, pressure_Pa, total) if c.permeability else 0.0
        for c in prop.channels
    }
    weights = prop.component_mass_kg_mol
    volumes = {
        c.channel_id: sum(weights[k] * v for k, v in c.stoichiometry) / 1000.0
        for c in prop.channels
    }
    numbers = {c.channel_id: sum(v for _, v in c.stoichiometry) for c in prop.channels}

    def evaluate(j: float) -> dict[str, Any]:
        exponential = {key: expm1(j / film[key]) for key in cb}

        def solute_flux(qp: float) -> dict[str, float]:
            return {
                key: value * (1 + exponential[key]) / (1 / b[key] + 1 / qp + exponential[key] / j)
                if b[key]
                else 0.0
                for key, value in cb.items()
            }

        upper = j + sum(
            volumes[key] * b[key] * value * (1 + exponential[key]) for key, value in cb.items()
        )
        if upper > j:
            qp = brentq(
                lambda q: (q - j - sum(volumes[key] * v for key, v in solute_flux(q).items())) / j,
                j,
                upper,
                xtol=max(j * 1e-13, 1e-25),
                rtol=1e-13,
            )
        else:
            qp = j
        js = solute_flux(qp)
        cp = {key: v / qp for key, v in js.items()}
        cm = {
            key: cp[key] + js[key] / b[key] if b[key] else cb[key] * (1 + exponential[key])
            for key in cb
        }
        if sum(cm.values()) > spec.maximum_interface_concentration_mol_m3:
            raise ValueError("MCAS polarized interface outside declared concentration domain")
        pi_f = R * t * sum(numbers[key] * v for key, v in cm.items())
        pi_p = R * t * sum(numbers[key] * v for key, v in cp.items())
        drive = pressure_Pa - spec.reservoir_pressure_Pa - pi_f + pi_p
        component_flux = prop.amounts(j * 1000 / weights[prop.water_id], js)
        return {
            "water_flux_m_s": j,
            "component_flux_mol_m2_s": component_flux,
            "channel_flux_mol_m2_s": js,
            "film_coefficients_m_s": dict(film),
            "interface_concentrations_mol_m3": cm,
            "permeate_concentrations_mol_m3": cp,
            "interface_osmotic_pressure_Pa": pi_f,
            "permeate_osmotic_pressure_Pa": pi_p,
            "driving_pressure_Pa": drive,
            "water_flux_residual_m_s": j - a * drive,
            "charge_flux_mol_m2_s": sum(
                prop.charges[key] * component_flux[key] for key in prop.solute_ids
            ),
            "property_approximation": "mcas-ideal-volume",
            "permeate_volume_flux_m_s": qp,
        }

    low = 1e-16
    upper = min(2 * a * (pressure_Pa - spec.reservoir_pressure_Pa), 30 * min(film.values()))
    if upper <= low:
        raise ValueError("MCAS RO needs a forward finite water flux")
    lower, boundary = low, upper
    for _ in range(80):
        try:
            if evaluate(upper)["water_flux_residual_m_s"] >= 0:
                break
            lower = upper
        except ValueError:
            boundary = upper
        upper = (lower + boundary) / 2
    else:
        raise ValueError("MCAS forward root is outside declared property bounds")
    j = brentq(lambda x: evaluate(x)["water_flux_residual_m_s"], low, upper, xtol=1e-16, rtol=1e-12)
    return {**evaluate(j), "stopping_event": None}
