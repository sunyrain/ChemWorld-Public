"""Extensive aqueous properties from the declared standard states and excess G.

This is a query over caller-owned moles. It supplies no phase selection, state
store, empirical density correction or extra thermodynamic reference.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from math import fsum, isfinite
from typing import Any, cast

from chemworld.physchem.aqueous_database import AqueousDatabase
from chemworld.physchem.caloric_math import partial_derivative
from chemworld.physchem.property_reports import R_J_PER_MOL_K as R


@dataclass(frozen=True)
class AqueousPhaseProperties:
    temperature_K: float
    pressure_Pa: float
    volume_m3: float
    enthalpy_J: float
    internal_energy_J: float
    gibbs_energy_J: float
    entropy_J_K: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def aqueous_phase_properties(
    database: AqueousDatabase,
    amounts_mol: Mapping[str, float],
    *,
    temperature_K: float,
    pressure_Pa: float,
) -> AqueousPhaseProperties:
    """Use G_P for V and G-T*G_T for H at fixed amounts, including finite water.

    Every water/solute standard state must supply its actual Gibbs molar volume.
    Negative solute partial volumes are valid; total volume must be positive.
    The existing molality ideal-mixing term is linear in T and independent of P,
    so it contributes to G and S, but not to H or V. Nonideal contributions are
    differentiated through the original excess-G function, including T/P laws.
    """
    liquid = {s.species_id: s for s in database.species if s.phase in {"solvent", "aqueous"}}
    if set(amounts_mol) - liquid.keys():
        raise ValueError("aqueous phase properties accept only declared water and solutes")
    report = database.chemical_potentials(
        amounts_mol, temperature_K=temperature_K, pressure_Pa=pressure_Pa
    )
    if any(report.standard_states[k].molar_volume_m3_mol is None for k in liquid):
        raise ValueError(
            "aqueous inventory requires explicit volumes from constant-volume or aqueous-gibbs")
    amounts = report.amounts_mol
    water_kg = report.solvent_mass_kg
    molalities = {
        key: amounts[key] / water_kg for key, s in liquid.items() if s.phase == "aqueous"
    }

    def excess(values: Mapping[str, Any]) -> Any:
        return database.activity_model.excess_gibbs_RT_mol_kg(
            molalities, values["T"], values["P"]
        )

    coordinates = {"T": temperature_K, "P": pressure_Pa}
    dg_dt = partial_derivative(excess, coordinates, ("T",))
    dg_dp = partial_derivative(excess, coordinates, ("P",))
    h = fsum(amounts[k] * report.standard_states[k].enthalpy_J_mol for k in liquid)
    h -= R * temperature_K**2 * water_kg * dg_dt
    v = fsum(
        amounts[k] * cast(float, report.standard_states[k].molar_volume_m3_mol) for k in liquid
    ) + R * temperature_K * water_kg * dg_dp
    g = fsum(
        amounts[k] * cast(float, report.chemical_potentials_J_mol[k])
        for k in liquid if amounts[k] > 0
    )
    if not all(isfinite(value) for value in (h, v, g)) or v <= 0:
        raise ValueError("aqueous phase requires finite thermodynamic properties and positive V")
    return AqueousPhaseProperties(
        temperature_K, pressure_Pa, float(v), float(h), float(h - pressure_Pa * v),
        float(g), float((h - g) / temperature_K),
    )
