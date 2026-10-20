"""Electrochemical thermodynamics and charge-accounting kernels.

This module implements a compact, auditable electrochemistry core for the
shared ChemWorld law.  The sign convention follows the usual electrode-kinetics
form: positive overpotential gives positive anodic Butler-Volmer current, while
negative overpotential gives cathodic current. Reaction direction is explicit;
charge magnitude is never silently reinterpreted as forward conversion.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import exp, isfinite, log
from typing import Any

from chemworld.physchem.electrochemistry_cards import electrochemistry_model_cards
from chemworld.physchem.reaction_network import R_J_PER_MOL_K

FARADAY_C_PER_MOL = 96485.33212
_ACTIVITY_FLOOR = 1.0e-30
_EXPONENT_LIMIT = 80.0


@dataclass(frozen=True)
class ElectrodeReactionSpec:
    """Single electrode reaction contract.

    ``reaction_quotient_exponents`` stores the stoichiometric exponents used in
    ``Q = prod(activity_i ** exponent_i)``.  For a reduction-like virtual
    conversion ``A -> P``, use ``{"P": 1, "A": -1}``.
    """

    reaction_id: str
    electrons_transferred: float
    standard_potential_V: float
    reaction_quotient_exponents: dict[str, float]
    exchange_current_density_A_m2: float
    electrode_area_m2: float
    alpha_anodic: float = 0.5
    alpha_cathodic: float = 0.5
    faradaic_efficiency_ref: float = 0.90
    product_selectivity_ref: float = 0.88
    overpotential_selectivity_sensitivity_V_inv: float = 0.55
    forward_current_sign: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.reaction_id.strip():
            raise ValueError("reaction_id cannot be empty")
        _positive(self.electrons_transferred, "electrons_transferred")
        _finite(self.standard_potential_V, "standard_potential_V")
        if not self.reaction_quotient_exponents:
            raise ValueError("reaction_quotient_exponents cannot be empty")
        for key, value in self.reaction_quotient_exponents.items():
            if not key.strip():
                raise ValueError("reaction quotient species ids cannot be empty")
            _finite(value, f"reaction_quotient_exponents[{key!r}]")
            if value == 0.0:
                raise ValueError("reaction quotient exponents cannot be zero")
        _nonnegative(self.exchange_current_density_A_m2, "exchange_current_density_A_m2")
        _positive(self.electrode_area_m2, "electrode_area_m2")
        _positive(self.alpha_anodic, "alpha_anodic")
        _positive(self.alpha_cathodic, "alpha_cathodic")
        _fraction(self.faradaic_efficiency_ref, "faradaic_efficiency_ref")
        _fraction(self.product_selectivity_ref, "product_selectivity_ref")
        _nonnegative(
            self.overpotential_selectivity_sensitivity_V_inv,
            "overpotential_selectivity_sensitivity_V_inv",
        )
        if isinstance(self.forward_current_sign, bool) or self.forward_current_sign not in {-1, 1}:
            raise ValueError("forward_current_sign must be either -1 or 1")

    @property
    def exchange_current_a(self) -> float:
        return self.exchange_current_density_A_m2 * self.electrode_area_m2

    def to_dict(self) -> dict[str, Any]:
        return {
            "reaction_id": self.reaction_id,
            "electrons_transferred": self.electrons_transferred,
            "standard_potential_V": self.standard_potential_V,
            "reaction_quotient_exponents": dict(self.reaction_quotient_exponents),
            "exchange_current_density_A_m2": self.exchange_current_density_A_m2,
            "electrode_area_m2": self.electrode_area_m2,
            "exchange_current_A": self.exchange_current_a,
            "alpha_anodic": self.alpha_anodic,
            "alpha_cathodic": self.alpha_cathodic,
            "faradaic_efficiency_ref": self.faradaic_efficiency_ref,
            "product_selectivity_ref": self.product_selectivity_ref,
            "overpotential_selectivity_sensitivity_V_inv": (
                self.overpotential_selectivity_sensitivity_V_inv
            ),
            "forward_current_sign": self.forward_current_sign,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class ElectrolyteResistanceSpec:
    """Lumped electrolyte and contact-resistance contract.

    The compact cell model uses ``R = L / (kappa A) + R_contact`` where ``L`` is
    the uncompensated electrolyte path length, ``kappa`` is ionic conductivity,
    and ``A`` is effective electrode area.  It is deliberately a cell-level
    lumped resistance rather than a porous-electrode transport model.
    """

    electrolyte_conductivity_S_m: float
    electrode_gap_m: float
    electrode_area_m2: float
    contact_resistance_ohm: float = 0.0
    voltage_window_V: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _positive(
            self.electrolyte_conductivity_S_m,
            "electrolyte_conductivity_S_m",
        )
        _positive(self.electrode_gap_m, "electrode_gap_m")
        _positive(self.electrode_area_m2, "electrode_area_m2")
        _nonnegative(self.contact_resistance_ohm, "contact_resistance_ohm")
        if self.voltage_window_V is not None:
            _positive(self.voltage_window_V, "voltage_window_V")

    @property
    def electrolyte_resistance_ohm(self) -> float:
        return self.electrode_gap_m / (self.electrolyte_conductivity_S_m * self.electrode_area_m2)

    @property
    def total_resistance_ohm(self) -> float:
        return self.electrolyte_resistance_ohm + self.contact_resistance_ohm

    def to_dict(self) -> dict[str, Any]:
        return {
            "electrolyte_conductivity_S_m": self.electrolyte_conductivity_S_m,
            "electrode_gap_m": self.electrode_gap_m,
            "electrode_area_m2": self.electrode_area_m2,
            "contact_resistance_ohm": self.contact_resistance_ohm,
            "electrolyte_resistance_ohm": self.electrolyte_resistance_ohm,
            "total_resistance_ohm": self.total_resistance_ohm,
            "voltage_window_V": self.voltage_window_V,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class OhmicDropResult:
    """Measured/interfacial potential split for a lumped cell resistance."""

    measured_potential_V: float
    interfacial_potential_V: float
    current_A: float
    electrolyte_resistance_ohm: float
    contact_resistance_ohm: float
    total_resistance_ohm: float
    uncompensated_voltage_drop_V: float
    ohmic_loss_J: float
    voltage_window_exceeded: bool
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "measured_potential_V": self.measured_potential_V,
            "interfacial_potential_V": self.interfacial_potential_V,
            "current_A": self.current_A,
            "electrolyte_resistance_ohm": self.electrolyte_resistance_ohm,
            "contact_resistance_ohm": self.contact_resistance_ohm,
            "total_resistance_ohm": self.total_resistance_ohm,
            "uncompensated_voltage_drop_V": self.uncompensated_voltage_drop_V,
            "ohmic_loss_J": self.ohmic_loss_J,
            "voltage_window_exceeded": self.voltage_window_exceeded,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class ElectrolysisResult:
    """Terminal electrolysis accounting for one operation."""

    reaction_id: str
    equilibrium_potential_V: float
    measured_potential_V: float
    interfacial_potential_V: float
    overpotential_V: float
    kinetic_current_A: float
    actual_current_A: float
    reaction_direction: int
    charge_C: float
    faradaic_charge_C: float
    extent_mol: float
    converted_mol: float
    product_mol: float
    byproduct_mol: float
    faradaic_efficiency: float
    product_selectivity: float
    electrical_work_J: float
    interfacial_work_J: float
    ohmic_loss_J: float
    reversible_work_J: float
    energy_efficiency: float
    electrolyte_resistance_ohm: float
    contact_resistance_ohm: float
    total_resistance_ohm: float
    uncompensated_voltage_drop_V: float
    voltage_window_exceeded: bool
    capacitive_charge_C: float = 0.0
    side_reaction_charge_C: float = 0.0
    charge_balance_residual_C: float = 0.0
    energy_balance_residual_J: float = 0.0
    signed_terminal_work_J: float = 0.0
    signed_interfacial_work_J: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.reaction_direction not in {-1, 0, 1}:
            raise ValueError("reaction_direction must be -1, 0, or 1")
        for name, value in (
            ("charge_C", self.charge_C),
            ("faradaic_charge_C", self.faradaic_charge_C),
            ("capacitive_charge_C", self.capacitive_charge_C),
            ("side_reaction_charge_C", self.side_reaction_charge_C),
            ("extent_mol", self.extent_mol),
            ("converted_mol", self.converted_mol),
            ("product_mol", self.product_mol),
            ("byproduct_mol", self.byproduct_mol),
            ("electrical_work_J", self.electrical_work_J),
            ("ohmic_loss_J", self.ohmic_loss_J),
        ):
            _nonnegative(value, name)
        _finite(self.charge_balance_residual_C, "charge_balance_residual_C")
        _finite(self.energy_balance_residual_J, "energy_balance_residual_J")
        _finite(self.signed_terminal_work_J, "signed_terminal_work_J")
        _finite(self.signed_interfacial_work_J, "signed_interfacial_work_J")
        if abs(self.product_mol + self.byproduct_mol - self.converted_mol) > 1.0e-12:
            raise ValueError("electrolysis material balance does not close")
        if (
            abs(
                self.charge_C
                - self.faradaic_charge_C
                - self.capacitive_charge_C
                - self.side_reaction_charge_C
            )
            > 1.0e-9
        ):
            raise ValueError("electrolysis charge balance does not close")

    def to_dict(self) -> dict[str, Any]:
        return {
            "reaction_id": self.reaction_id,
            "equilibrium_potential_V": self.equilibrium_potential_V,
            "measured_potential_V": self.measured_potential_V,
            "interfacial_potential_V": self.interfacial_potential_V,
            "overpotential_V": self.overpotential_V,
            "kinetic_current_A": self.kinetic_current_A,
            "actual_current_A": self.actual_current_A,
            "reaction_direction": self.reaction_direction,
            "charge_C": self.charge_C,
            "faradaic_charge_C": self.faradaic_charge_C,
            "extent_mol": self.extent_mol,
            "converted_mol": self.converted_mol,
            "product_mol": self.product_mol,
            "byproduct_mol": self.byproduct_mol,
            "faradaic_efficiency": self.faradaic_efficiency,
            "product_selectivity": self.product_selectivity,
            "electrical_work_J": self.electrical_work_J,
            "interfacial_work_J": self.interfacial_work_J,
            "ohmic_loss_J": self.ohmic_loss_J,
            "reversible_work_J": self.reversible_work_J,
            "energy_efficiency": self.energy_efficiency,
            "electrolyte_resistance_ohm": self.electrolyte_resistance_ohm,
            "contact_resistance_ohm": self.contact_resistance_ohm,
            "total_resistance_ohm": self.total_resistance_ohm,
            "uncompensated_voltage_drop_V": self.uncompensated_voltage_drop_V,
            "voltage_window_exceeded": self.voltage_window_exceeded,
            "capacitive_charge_C": self.capacitive_charge_C,
            "side_reaction_charge_C": self.side_reaction_charge_C,
            "charge_balance_residual_C": self.charge_balance_residual_C,
            "energy_balance_residual_J": self.energy_balance_residual_J,
            "signed_terminal_work_J": self.signed_terminal_work_J,
            "signed_interfacial_work_J": self.signed_interfacial_work_J,
            "metadata": dict(self.metadata),
        }


def reaction_quotient_from_activities(
    activities: dict[str, float],
    exponents: dict[str, float],
) -> float:
    """Return ``Q`` from dimensionless activities and stoichiometric exponents."""

    log_q = reaction_quotient_log_from_activities(activities, exponents)
    return exp(_clamp(log_q, -_EXPONENT_LIMIT, _EXPONENT_LIMIT))


def reaction_quotient_log_from_activities(
    activities: dict[str, float],
    exponents: dict[str, float],
) -> float:
    """Return ``ln(Q)`` with a positive activity floor."""

    if not exponents:
        raise ValueError("exponents cannot be empty")
    total = 0.0
    for species_id, exponent in exponents.items():
        _finite(exponent, f"exponent[{species_id!r}]")
        activity = max(float(activities.get(species_id, _ACTIVITY_FLOOR)), _ACTIVITY_FLOOR)
        total += exponent * log(activity)
    return total


def equilibrium_potential_from_delta_g(
    *,
    delta_g_reaction_J_mol: float,
    electrons_transferred: float,
) -> float:
    """Return ``E_eq = -Delta G_rxn / (n F)`` in volts."""

    _finite(delta_g_reaction_J_mol, "delta_g_reaction_J_mol")
    _positive(electrons_transferred, "electrons_transferred")
    return -delta_g_reaction_J_mol / (electrons_transferred * FARADAY_C_PER_MOL)


def nernst_potential(
    spec: ElectrodeReactionSpec,
    activities: dict[str, float],
    *,
    temperature_K: float = 298.15,
) -> float:
    """Return the Nernst equilibrium potential for an electrode reaction."""

    _positive(temperature_K, "temperature_K")
    log_q = reaction_quotient_log_from_activities(
        activities,
        spec.reaction_quotient_exponents,
    )
    return (
        spec.standard_potential_V
        - (R_J_PER_MOL_K * temperature_K / (spec.electrons_transferred * FARADAY_C_PER_MOL)) * log_q
    )


def butler_volmer_current(
    spec: ElectrodeReactionSpec,
    *,
    electrode_potential_V: float,
    activities: dict[str, float],
    temperature_K: float = 298.15,
) -> float:
    """Return signed Butler-Volmer current in amperes."""

    _finite(electrode_potential_V, "electrode_potential_V")
    _positive(temperature_K, "temperature_K")
    equilibrium_potential = nernst_potential(spec, activities, temperature_K=temperature_K)
    overpotential = electrode_potential_V - equilibrium_potential
    anodic, cathodic = butler_volmer_factors(
        overpotential, temperature_K=temperature_K,
        electrons_transferred=spec.electrons_transferred,
        alpha_anodic=spec.alpha_anodic, alpha_cathodic=spec.alpha_cathodic)
    return spec.exchange_current_a * (anodic - cathodic)


def butler_volmer_factors(
    overpotential_V: float, *, temperature_K: float = 298.15,
    electrons_transferred: float = 1.0, alpha_anodic: float = 0.5,
    alpha_cathodic: float = 0.5,
) -> tuple[float, float]:
    """Common signed BV activation factors; concentrations belong to the caller."""
    _finite(overpotential_V, "overpotential_V")
    for name, value in (("temperature_K", temperature_K),
                        ("electrons_transferred", electrons_transferred),
                        ("alpha_anodic", alpha_anodic), ("alpha_cathodic", alpha_cathodic)):
        _positive(value, name)
    coefficient = electrons_transferred * FARADAY_C_PER_MOL / (R_J_PER_MOL_K * temperature_K)
    anodic = exp(
        _clamp(
            alpha_anodic * coefficient * overpotential_V,
            -_EXPONENT_LIMIT,
            _EXPONENT_LIMIT,
        )
    )
    cathodic = exp(
        _clamp(
            -alpha_cathodic * coefficient * overpotential_V,
            -_EXPONENT_LIMIT,
            _EXPONENT_LIMIT,
        )
    )
    return anodic, cathodic


def faradaic_extent_mol(
    *,
    current_A: float,
    duration_s: float,
    electrons_transferred: float,
    faradaic_efficiency: float = 1.0,
) -> float:
    """Convert electrical charge to reaction extent through Faraday's law."""

    _finite(current_A, "current_A")
    _nonnegative(duration_s, "duration_s")
    _positive(electrons_transferred, "electrons_transferred")
    _fraction(faradaic_efficiency, "faradaic_efficiency")
    charge_C = abs(current_A) * duration_s
    return charge_C * faradaic_efficiency / (electrons_transferred * FARADAY_C_PER_MOL)


def electrolyte_resistance_ohm(spec: ElectrolyteResistanceSpec) -> float:
    """Return lumped electrolyte plus contact resistance in ohms."""

    return spec.total_resistance_ohm


def ohmic_drop(
    *,
    measured_potential_V: float,
    current_A: float,
    duration_s: float,
    resistance: ElectrolyteResistanceSpec | None,
) -> OhmicDropResult:
    """Return uncompensated voltage drop and interfacial potential.

    Positive current lowers the interfacial potential relative to the measured
    terminal/control potential by ``iR``.  Negative current shifts it upward by
    the same Ohm-law convention.
    """

    _finite(measured_potential_V, "measured_potential_V")
    _finite(current_A, "current_A")
    _nonnegative(duration_s, "duration_s")
    if resistance is None:
        total_resistance = 0.0
        electrolyte_resistance = 0.0
        contact_resistance = 0.0
        voltage_window = None
    else:
        total_resistance = resistance.total_resistance_ohm
        electrolyte_resistance = resistance.electrolyte_resistance_ohm
        contact_resistance = resistance.contact_resistance_ohm
        voltage_window = resistance.voltage_window_V
    voltage_drop = current_A * total_resistance
    interfacial_potential = measured_potential_V - voltage_drop
    return OhmicDropResult(
        measured_potential_V=measured_potential_V,
        interfacial_potential_V=interfacial_potential,
        current_A=current_A,
        electrolyte_resistance_ohm=electrolyte_resistance,
        contact_resistance_ohm=contact_resistance,
        total_resistance_ohm=total_resistance,
        uncompensated_voltage_drop_V=voltage_drop,
        ohmic_loss_J=current_A * current_A * total_resistance * duration_s,
        voltage_window_exceeded=(
            voltage_window is not None and abs(measured_potential_V) > voltage_window
        ),
        metadata={"model_id": "lumped_electrolyte_ohmic_drop_v1"},
    )


def run_electrolysis(
    spec: ElectrodeReactionSpec,
    *,
    electrode_potential_V: float,
    duration_s: float,
    activities: dict[str, float],
    available_substrate_mol: float,
    available_reverse_substrate_mol: float = 0.0,
    temperature_K: float = 298.15,
    applied_current_A: float | None = None,
    electrolyte_resistance: ElectrolyteResistanceSpec | None = None,
    useful_charge_limit_C: float | None = None,
    capacitive_charge_C: float = 0.0,
) -> ElectrolysisResult:
    """Run a terminal electrolysis accounting step.

    The kinetic Butler-Volmer current sets the sign and kinetic upper bound.  If
    ``applied_current_A`` is provided, the actual current magnitude is capped by
    the kinetic current magnitude, approximating a finite cell/power-supply
    constraint without solving a full coupled electrochemical cell.
    """

    _finite(electrode_potential_V, "electrode_potential_V")
    _nonnegative(duration_s, "duration_s")
    _nonnegative(available_substrate_mol, "available_substrate_mol")
    _nonnegative(available_reverse_substrate_mol, "available_reverse_substrate_mol")
    _positive(temperature_K, "temperature_K")
    if applied_current_A is not None:
        _finite(applied_current_A, "applied_current_A")
    if useful_charge_limit_C is not None:
        _nonnegative(useful_charge_limit_C, "useful_charge_limit_C")
    _nonnegative(capacitive_charge_C, "capacitive_charge_C")

    equilibrium_potential = nernst_potential(spec, activities, temperature_K=temperature_K)
    actual_current = 0.0 if applied_current_A is None else float(applied_current_A)
    kinetic_current = 0.0
    interfacial_potential = electrode_potential_V
    iteration_count = 0
    for _ in range(12):
        iteration_count += 1
        drop = ohmic_drop(
            measured_potential_V=electrode_potential_V,
            current_A=actual_current,
            duration_s=duration_s,
            resistance=electrolyte_resistance,
        )
        interfacial_potential = drop.interfacial_potential_V
        kinetic_current = butler_volmer_current(
            spec,
            electrode_potential_V=interfacial_potential,
            activities=activities,
            temperature_K=temperature_K,
        )
        kinetic_magnitude = abs(kinetic_current)
        if applied_current_A is None:
            next_current = kinetic_current
        else:
            requested_magnitude = abs(applied_current_A)
            actual_magnitude = min(requested_magnitude, kinetic_magnitude)
            sign_source = kinetic_current if kinetic_magnitude > 0.0 else applied_current_A
            next_current = _copy_sign(actual_magnitude, sign_source)
        if abs(next_current - actual_current) <= 1.0e-12:
            actual_current = next_current
            break
        actual_current = next_current

    drop = ohmic_drop(
        measured_potential_V=electrode_potential_V,
        current_A=actual_current,
        duration_s=duration_s,
        resistance=electrolyte_resistance,
    )
    interfacial_potential = drop.interfacial_potential_V
    overpotential = interfacial_potential - equilibrium_potential
    kinetic_current = butler_volmer_current(
        spec,
        electrode_potential_V=interfacial_potential,
        activities=activities,
        temperature_K=temperature_K,
    )

    overpotential_stress = max(abs(overpotential) - 0.15, 0.0)
    intrinsic_faradaic_efficiency = _clip01(
        spec.faradaic_efficiency_ref
        * exp(-spec.overpotential_selectivity_sensitivity_V_inv * overpotential_stress)
    )
    product_selectivity = _clip01(
        spec.product_selectivity_ref
        * exp(-0.5 * spec.overpotential_selectivity_sensitivity_V_inv * overpotential_stress)
    )
    charge = abs(actual_current) * duration_s
    if capacitive_charge_C > charge + 1.0e-12:
        raise ValueError("capacitive_charge_C cannot exceed total charge")
    if abs(actual_current) <= 1.0e-15:
        reaction_direction = 0
    elif actual_current * spec.forward_current_sign > 0.0:
        reaction_direction = 1
    else:
        reaction_direction = -1
    directional_inventory = (
        available_substrate_mol
        if reaction_direction >= 0
        else available_reverse_substrate_mol
    )
    productive_charge = charge * intrinsic_faradaic_efficiency
    productive_charge = min(
        productive_charge,
        max(charge - capacitive_charge_C, 0.0),
        directional_inventory * spec.electrons_transferred * FARADAY_C_PER_MOL,
    )
    if useful_charge_limit_C is not None:
        productive_charge = min(productive_charge, useful_charge_limit_C)
    faradaic_efficiency = 0.0 if charge <= 0.0 else productive_charge / charge
    extent = productive_charge / (spec.electrons_transferred * FARADAY_C_PER_MOL)
    converted = extent
    product_mol = converted * product_selectivity
    byproduct_mol = converted - product_mol
    faradaic_charge = productive_charge
    side_reaction_charge = max(charge - capacitive_charge_C - faradaic_charge, 0.0)
    charge_balance_residual = charge - faradaic_charge - capacitive_charge_C - side_reaction_charge
    electrical_work = abs(electrode_potential_V * actual_current * duration_s)
    interfacial_work = abs(interfacial_potential * actual_current * duration_s)
    signed_terminal_work = electrode_potential_V * actual_current * duration_s
    signed_interfacial_work = interfacial_potential * actual_current * duration_s
    energy_balance_residual = signed_terminal_work - signed_interfacial_work - drop.ohmic_loss_J
    reversible_work = min(
        abs(equilibrium_potential) * faradaic_charge,
        max(electrical_work - drop.ohmic_loss_J, 0.0),
    )
    energy_efficiency = 0.0
    if electrical_work > 0.0:
        energy_efficiency = _clip01((reversible_work / electrical_work) * product_selectivity)
    return ElectrolysisResult(
        reaction_id=spec.reaction_id,
        equilibrium_potential_V=equilibrium_potential,
        measured_potential_V=electrode_potential_V,
        interfacial_potential_V=interfacial_potential,
        overpotential_V=overpotential,
        kinetic_current_A=kinetic_current,
        actual_current_A=actual_current,
        reaction_direction=reaction_direction,
        charge_C=charge,
        faradaic_charge_C=faradaic_charge,
        extent_mol=extent,
        converted_mol=converted,
        product_mol=product_mol,
        byproduct_mol=byproduct_mol,
        faradaic_efficiency=faradaic_efficiency,
        product_selectivity=product_selectivity,
        electrical_work_J=electrical_work,
        interfacial_work_J=interfacial_work,
        ohmic_loss_J=drop.ohmic_loss_J,
        reversible_work_J=reversible_work,
        energy_efficiency=energy_efficiency,
        electrolyte_resistance_ohm=drop.electrolyte_resistance_ohm,
        contact_resistance_ohm=drop.contact_resistance_ohm,
        total_resistance_ohm=drop.total_resistance_ohm,
        uncompensated_voltage_drop_V=drop.uncompensated_voltage_drop_V,
        voltage_window_exceeded=drop.voltage_window_exceeded,
        capacitive_charge_C=capacitive_charge_C,
        side_reaction_charge_C=side_reaction_charge,
        charge_balance_residual_C=charge_balance_residual,
        energy_balance_residual_J=energy_balance_residual,
        signed_terminal_work_J=signed_terminal_work,
        signed_interfacial_work_J=signed_interfacial_work,
        metadata={
            "model_id": "nernst_butler_volmer_faradaic_v1",
            "ohmic_drop_model_id": drop.metadata["model_id"],
            "temperature_K": temperature_K,
            "applied_current_A": applied_current_A,
            "kinetic_current_limited": (
                applied_current_A is not None and abs(applied_current_A) > abs(kinetic_current)
            ),
            "ohmic_iteration_count": iteration_count,
            "intrinsic_faradaic_efficiency": intrinsic_faradaic_efficiency,
            "reaction_direction": reaction_direction,
            "forward_current_sign": spec.forward_current_sign,
            "useful_charge_limit_C": useful_charge_limit_C,
        },
    )


def _finite(value: float, name: str) -> None:
    if not isfinite(value):
        raise ValueError(f"{name} must be finite")


def _positive(value: float, name: str) -> None:
    _finite(value, name)
    if value <= 0.0:
        raise ValueError(f"{name} must be positive")


def _nonnegative(value: float, name: str) -> None:
    _finite(value, name)
    if value < 0.0:
        raise ValueError(f"{name} must be nonnegative")


def _fraction(value: float, name: str) -> None:
    _finite(value, name)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1")


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _copy_sign(magnitude: float, sign_source: float) -> float:
    if sign_source < 0.0:
        return -magnitude
    return magnitude


__all__ = [
    "FARADAY_C_PER_MOL",
    "ElectrodeReactionSpec",
    "ElectrolysisResult",
    "ElectrolyteResistanceSpec",
    "OhmicDropResult",
    "butler_volmer_current",
    "electrochemistry_model_cards",
    "electrolyte_resistance_ohm",
    "equilibrium_potential_from_delta_g",
    "faradaic_extent_mol",
    "nernst_potential",
    "ohmic_drop",
    "reaction_quotient_from_activities",
    "reaction_quotient_log_from_activities",
    "run_electrolysis",
]
