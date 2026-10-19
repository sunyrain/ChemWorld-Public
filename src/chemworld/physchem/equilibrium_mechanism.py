"""Coupled weak-acid, ion-pair, and precipitation equilibrium networks."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, log10
from typing import Literal

import numpy as np
from scipy.optimize import brentq

EquilibriumMechanismFamily = Literal[
    "direct_free_ion_precipitation",
    "aqueous_ion_pair_intermediate",
]


@dataclass(frozen=True)
class CoupledEquilibriumResult:
    """A mass-conserving solution of one registered EQ-S reaction network."""

    mechanism_family: EquilibriumMechanismFamily
    branch: Literal["dissolved", "solid_present"]
    pH: float
    hydrogen_mol_L: float
    undissociated_acid_mol_L: float
    free_conjugate_base_mol_L: float
    free_cation_mol_L: float
    aqueous_pair_mol_L: float
    precipitated_mol: float
    acid_dissociation_fraction: float
    precipitation_signal: float
    equilibrium_residual: float
    ion_product: float
    ksp: float

    def __post_init__(self) -> None:
        values = (
            self.pH,
            self.hydrogen_mol_L,
            self.undissociated_acid_mol_L,
            self.free_conjugate_base_mol_L,
            self.free_cation_mol_L,
            self.aqueous_pair_mol_L,
            self.precipitated_mol,
            self.acid_dissociation_fraction,
            self.precipitation_signal,
            self.equilibrium_residual,
            self.ion_product,
            self.ksp,
        )
        if not all(isfinite(value) for value in values):
            raise ValueError("coupled equilibrium result must be finite")
        if min(values[1:7]) < 0.0:
            raise ValueError("coupled equilibrium species amounts must be nonnegative")
        if not 0.0 <= self.acid_dissociation_fraction <= 1.0:
            raise ValueError("acid dissociation fraction must be in [0, 1]")
        if not 0.0 <= self.precipitation_signal <= 1.0:
            raise ValueError("precipitation signal must be in [0, 1]")


def solve_coupled_weak_acid_precipitation(
    *,
    acid_total_mol: float,
    volume_L: float,
    pka: float,
    log10_ksp: float,
    mechanism_family: EquilibriumMechanismFamily,
    cation_fraction: float = 0.15,
    association_beta_L_per_mol: float | None = None,
    activity_coefficient_ratio: float = 1.0,
) -> CoupledEquilibriumResult:
    """Solve the two registered EQ-S algebraic mechanism families.

    ``direct_free_ion_precipitation`` omits ``MA(aq)`` from the species graph.
    ``aqueous_ion_pair_intermediate`` includes ``[MA] = beta [M+][A-]``.
    Both branches enforce acid and cation balances, electroneutrality, acid
    dissociation, and the usual solid complementarity condition.
    """

    numeric = {
        "acid_total_mol": acid_total_mol,
        "volume_L": volume_L,
        "pka": pka,
        "log10_ksp": log10_ksp,
        "cation_fraction": cation_fraction,
        "activity_coefficient_ratio": activity_coefficient_ratio,
    }
    if not all(isfinite(float(value)) for value in numeric.values()):
        raise ValueError("coupled equilibrium inputs must be finite")
    if acid_total_mol <= 0.0 or volume_L <= 0.0:
        raise ValueError("acid amount and volume must be positive")
    if not 0.0 < cation_fraction < 1.0:
        raise ValueError("cation_fraction must be in (0, 1)")
    if activity_coefficient_ratio <= 0.0:
        raise ValueError("activity_coefficient_ratio must be positive")
    if mechanism_family not in {
        "direct_free_ion_precipitation",
        "aqueous_ion_pair_intermediate",
    }:
        raise ValueError(f"unsupported equilibrium mechanism family: {mechanism_family}")

    if mechanism_family == "direct_free_ion_precipitation":
        if association_beta_L_per_mol not in {None, 0.0}:
            raise ValueError("direct network cannot carry an aqueous association constant")
        beta = 0.0
    else:
        if association_beta_L_per_mol is None or association_beta_L_per_mol <= 0.0:
            raise ValueError("ion-pair network requires a positive association constant")
        beta = float(association_beta_L_per_mol)

    acid_total = float(acid_total_mol) / float(volume_L)
    cation_total = cation_fraction * acid_total
    spectator_anion = cation_total
    ka = 10.0 ** (-float(pka))
    ksp = 10.0 ** float(log10_ksp)
    kw = 1.0e-14

    def species(hydrogen: float) -> tuple[float, float, float, float, float]:
        deprotonated_total = hydrogen - kw / hydrogen
        undissociated = acid_total - deprotonated_total
        free_base = (
            ka
            * undissociated
            / (activity_coefficient_ratio * hydrogen)
        )
        bound_or_solid = deprotonated_total - free_base
        free_cation = cation_total - bound_or_solid
        return (
            deprotonated_total,
            undissociated,
            free_base,
            bound_or_solid,
            free_cation,
        )

    upper_log_h = log10(max(1.2 * acid_total, 1.0e-2))
    log_h_grid = np.linspace(-12.0, upper_log_h, 320)

    def find_branch(
        branch: Literal["dissolved", "solid_present"],
    ) -> tuple[float, float, float, float, float, float, float] | None:
        def residual(log_h: float) -> float:
            hydrogen = 10.0**log_h
            _, _, free_base, bound_or_solid, free_cation = species(hydrogen)
            if branch == "dissolved":
                return bound_or_solid - beta * free_cation * free_base
            return free_cation * free_base - ksp

        previous: tuple[float, float] | None = None
        for log_h in log_h_grid:
            value = residual(float(log_h))
            if not isfinite(value):
                previous = None
                continue
            if previous is not None and value * previous[1] <= 0.0:
                try:
                    root = brentq(
                        residual,
                        previous[0],
                        float(log_h),
                        xtol=1.0e-14,
                        rtol=1.0e-12,
                    )
                except ValueError:
                    previous = (float(log_h), value)
                    continue
                hydrogen = 10.0**root
                (
                    deprotonated_total,
                    undissociated,
                    free_base,
                    bound_or_solid,
                    free_cation,
                ) = species(hydrogen)
                tolerance = 1.0e-9 * max(acid_total, 1.0)
                if (
                    min(
                        undissociated,
                        free_base,
                        bound_or_solid,
                        free_cation,
                    )
                    < -tolerance
                    or deprotonated_total > acid_total + tolerance
                ):
                    previous = (float(log_h), value)
                    continue
                pair = beta * free_cation * free_base
                if branch == "dissolved":
                    if free_cation * free_base > ksp * (1.0 + 1.0e-7):
                        previous = (float(log_h), value)
                        continue
                    solid = 0.0
                else:
                    pair = beta * ksp
                    solid = bound_or_solid - pair
                    if solid < -tolerance:
                        previous = (float(log_h), value)
                        continue
                    solid = max(solid, 0.0)
                return (
                    hydrogen,
                    max(undissociated, 0.0),
                    max(free_base, 0.0),
                    max(free_cation, 0.0),
                    max(pair, 0.0),
                    solid,
                    free_cation * free_base,
                )
            previous = (float(log_h), value)
        return None

    solution = find_branch("dissolved")
    branch: Literal["dissolved", "solid_present"] = "dissolved"
    if solution is None:
        solution = find_branch("solid_present")
        branch = "solid_present"
    if solution is None:
        raise ValueError("could not solve the registered coupled equilibrium network")

    hydrogen, undissociated, free_base, free_cation, pair, solid, ion_product = solution
    acid_mass_residual = abs(
        undissociated + free_base + pair + solid - acid_total
    ) / max(acid_total, 1.0e-30)
    cation_mass_residual = abs(
        free_cation + pair + solid - cation_total
    ) / max(cation_total, 1.0e-30)
    charge_residual = abs(
        hydrogen + free_cation - kw / hydrogen - free_base - spectator_anion
    ) / max(acid_total, 1.0e-30)
    acid_law_residual = abs(
        hydrogen
        * free_base
        * activity_coefficient_ratio
        / max(undissociated, 1.0e-30)
        / ka
        - 1.0
    )
    pair_law_residual = (
        0.0
        if beta == 0.0
        else abs(pair / max(beta * free_cation * free_base, 1.0e-30) - 1.0)
    )
    solid_law_residual = (
        max(ion_product / ksp - 1.0, 0.0)
        if branch == "dissolved"
        else abs(ion_product / ksp - 1.0)
    )
    equilibrium_residual = max(
        acid_mass_residual,
        cation_mass_residual,
        charge_residual,
        acid_law_residual,
        pair_law_residual,
        solid_law_residual,
    )

    return CoupledEquilibriumResult(
        mechanism_family=mechanism_family,
        branch=branch,
        pH=-log10(hydrogen),
        hydrogen_mol_L=hydrogen,
        undissociated_acid_mol_L=undissociated,
        free_conjugate_base_mol_L=free_base,
        free_cation_mol_L=free_cation,
        aqueous_pair_mol_L=pair,
        precipitated_mol=solid * volume_L,
        acid_dissociation_fraction=float(np.clip(free_base / acid_total, 0.0, 1.0)),
        precipitation_signal=float(np.clip(solid / acid_total, 0.0, 1.0)),
        equilibrium_residual=float(equilibrium_residual),
        ion_product=ion_product,
        ksp=ksp,
    )


__all__ = [
    "CoupledEquilibriumResult",
    "EquilibriumMechanismFamily",
    "solve_coupled_weak_acid_precipitation",
]
