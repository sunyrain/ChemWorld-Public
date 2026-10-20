"""Finite carbonate/water/calcite equilibrium at a declared fixed temperature.

Mass-action elimination leaves scalar charge, mineral and ionic-strength roots.
Water is actual material. The invariant n(H2O)+n(OH)-n(CO2) eliminates its
amount without treating the solvent as an infinite source. No ion is repaired.
"""

from dataclasses import asdict, dataclass, field
from math import isclose, isfinite, log10

from scipy.optimize import brentq

from chemworld.physchem.elements import molecular_weight, parse_formula
from chemworld.physchem.equilibrium import davies_aqueous_activity_coefficient
from chemworld.physchem.equilibrium_chemistry import (
    ionic_strength_from_amounts,
    net_charge_equivalents,
)

FORMULAS = {
    "H2O": "H2O",
    "H+": "H",
    "OH-": "OH",
    "CO2*": "CO2",
    "HCO3-": "HCO3",
    "CO3--": "CO3",
    "Ca++": "Ca",
    "Na+": "Na",
    "Cl-": "Cl",
    "Calcite": "CaCO3",
}
CHARGES = dict(zip(FORMULAS, (0, 1, -1, 0, -1, -2, 2, 1, -1, 0), strict=True))
COMPOSITIONS = {key: parse_formula(value) for key, value in FORMULAS.items()}
WATER_MOLAR_MASS_KG_MOL = molecular_weight(COMPOSITIONS["H2O"]) / 1000


@dataclass(frozen=True)
class CarbonateEquilibriumSpec:
    model_id: str = "synthetic-carbonate-calcite-v1"
    pKw: float = 14.0
    pKa1: float = 6.35
    pKa2: float = 10.33
    log10_Ksp: float = -8.48
    davies_A: float = 0.509
    davies_b: float = 0.3
    maximum_ionic_strength_mol_kg: float = 0.1
    allow_calcite: bool = True
    temperature_K: float = 298.15
    maximum_pressure_Pa: float = 200000.0

    def __post_init__(self):
        if (
            not self.model_id
            or any(
                not isfinite(v)
                for v in (
                    self.pKw,
                    self.pKa1,
                    self.pKa2,
                    self.log10_Ksp,
                    self.davies_A,
                    self.davies_b,
                    self.maximum_ionic_strength_mol_kg,
                    self.temperature_K,
                    self.maximum_pressure_Pa,
                )
            )
            or self.davies_A < 0
            or self.davies_b < 0
            or not 0 < self.maximum_ionic_strength_mol_kg <= 0.1
            or self.temperature_K != 298.15
            or self.maximum_pressure_Pa <= 0
            or not isinstance(self.allow_calcite, bool)
        ):
            raise ValueError("carbonate model requires finite, fixed-298.15-K dilute data")

    def validate_conditions(self, temperature_K, pressure_Pa):
        if not isclose(temperature_K, self.temperature_K, rel_tol=0, abs_tol=1e-8):
            raise ValueError("carbonate constants have no declared temperature dependence")
        if not isfinite(pressure_Pa) or not 0 < pressure_Pa <= self.maximum_pressure_Pa:
            raise ValueError("carbonate equilibrium requires its declared low-pressure domain")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        return cls(**value)


@dataclass(frozen=True)
class CarbonateEquilibriumResult:
    amounts_mol: dict[str, float]
    solvent_mass_kg: float
    hydrogen_activity: float
    ionic_strength_mol_kg: float
    saturation_ratio: float
    mineral_status: str
    element_residuals_mol: dict[str, float]
    charge_residual_eq: float
    mass_action_log10_residuals: dict[str, float | None]
    metadata: dict = field(default_factory=dict)

    @property
    def pH(self):  # noqa: N802 - existing ChemWorld pH result protocol
        return -log10(self.hydrogen_activity)

    @property
    def calcite_mol(self):
        return self.amounts_mol["Calcite"]

    @property
    def charge_balance_error_eq(self):
        return self.metadata["final_charge_eq"]

    @classmethod
    def from_dict(cls, payload):
        return cls(
            **{k: v for k, v in payload.items() if k not in {"pH", "log10_saturation_ratio"}}
        )

    @property
    def log10_saturation_ratio(self):
        return log10(self.saturation_ratio) if self.saturation_ratio > 0 else None

    def to_dict(self):
        return {
            **asdict(self),
            "pH": self.pH,
            "log10_saturation_ratio": self.log10_saturation_ratio,
        }


def element_amounts(amounts):
    return {
        element: sum(
            amounts.get(key, 0) * composition.get(element, 0)
            for key, composition in COMPOSITIONS.items()
        )
        for element in ("H", "O", "C", "Ca", "Na", "Cl")
    }


def carbonate_feed(
    *,
    water_kg,
    sodium_bicarbonate_mol=0.0,
    calcium_chloride_mol=0.0,
    sodium_chloride_mol=0.0,
    hydrochloric_acid_mol=0.0,
    sodium_hydroxide_mol=0.0,
    calcite_mol=0.0,
):
    """Expand actual neutral reagent inputs; never add a balancing counterion."""
    values = (
        water_kg,
        sodium_bicarbonate_mol,
        calcium_chloride_mol,
        sodium_chloride_mol,
        hydrochloric_acid_mol,
        sodium_hydroxide_mol,
        calcite_mol,
    )
    if any(not isfinite(v) or v < 0 for v in values) or water_kg <= 0:
        raise ValueError(
            "aqueous inputs require positive actual water and finite nonnegative stock"
        )
    return {
        **dict.fromkeys(FORMULAS, 0.0),
        "H2O": water_kg / WATER_MOLAR_MASS_KG_MOL,
        "H+": hydrochloric_acid_mol,
        "OH-": sodium_hydroxide_mol,
        "HCO3-": sodium_bicarbonate_mol,
        "Ca++": calcium_chloride_mol,
        "Na+": sodium_bicarbonate_mol + sodium_chloride_mol + sodium_hydroxide_mol,
        "Cl-": 2 * calcium_chloride_mol + sodium_chloride_mol + hydrochloric_acid_mol,
        "Calcite": calcite_mol,
    }


def solve_carbonate_equilibrium(
    model, amounts_mol, *, temperature_K=298.15, pressure_Pa=101325.0, fixed_calcite_mol=None
):
    """Solve the allowed phases, or just fast water equilibrium at a fixed solid stock.

    The latter boundary has no mineral stability assertion and introduces no
    kinetic law. It lets a physical caller hold an actual finite mineral amount.
    All returned molecules still belong to the same input elemental inventory.
    """
    model.validate_conditions(temperature_K, pressure_Pa)
    if set(amounts_mol) - FORMULAS.keys():
        raise ValueError("unsupported species in the bounded carbonate system")
    initial = {key: float(amounts_mol.get(key, 0)) for key in FORMULAS}
    if any(not isfinite(v) or v < 0 for v in initial.values()) or initial["H2O"] <= 0:
        raise ValueError("carbonate equilibrium requires finite nonnegative material and water")
    initial_charge = net_charge_equivalents(initial, CHARGES)
    if abs(initial_charge) > 1e-12:
        raise ValueError("aqueous inputs must be electroneutral; no counterion repair is allowed")
    if not model.allow_calcite and initial["Calcite"] > 0:
        raise ValueError("an excluded calcite phase cannot hold initial mineral stock")
    totals = element_amounts(initial)
    carbon, calcium = totals["C"], totals["Ca"]
    limit = min(carbon, calcium)
    if fixed_calcite_mol is not None and (
        not isfinite(fixed_calcite_mol)
        or not 0 <= fixed_calcite_mol <= limit
        or (not model.allow_calcite and fixed_calcite_mol > 0)
    ):
        raise ValueError("fixed mineral stock exceeds actual available elements or phase domain")
    # Calcite and both carbonate ions have O-3C=0. Water, OH and CO2
    # provide this independent elemental invariant without subtracting big H/O totals.
    anchor = initial["H2O"] + initial["OH-"] - initial["CO2*"]
    kw, k1, k2, ksp = (10.0**v for v in (-model.pKw, -model.pKa1, -model.pKa2, model.log10_Ksp))
    evaluations = 0

    def at_ionic_strength(log_ionic):
        nonlocal evaluations
        ionic = 10.0**log_ionic
        gamma = {
            z: davies_aqueous_activity_coefficient(
                ionic_strength_mol_kg=ionic,
                charge=z,
                temperature_K=temperature_K,
                debye_huckel_A=model.davies_A,
                davies_b=model.davies_b,
            ).activity_coefficient
            for z in (1, 2)
        }
        g1, g2 = gamma[1], gamma[2]

        def water_state(log_h, solid):
            nonlocal evaluations
            evaluations += 1
            h = 10.0**log_h
            r1, r2 = k1 / (h * g1), k1 * k2 / (h * h * g2)
            f0 = 1.0 / (1.0 + r1 + r2)
            co2 = (carbon - solid) * f0
            oh_m = kw / (h * g1)
            water_kg = (
                WATER_MOLAR_MASS_KG_MOL * (anchor + co2) / (1 + WATER_MOLAR_MASS_KG_MOL * oh_m)
            )
            if water_kg <= 0:
                raise ValueError("water inventory is insufficient for this trial speciation")
            amounts = {
                "H2O": water_kg / WATER_MOLAR_MASS_KG_MOL,
                "H+": water_kg * h / g1,
                "OH-": water_kg * oh_m,
                "CO2*": co2,
                "HCO3-": (carbon - solid) * r1 * f0,
                "CO3--": (carbon - solid) * r2 * f0,
                "Ca++": calcium - solid,
                "Na+": initial["Na+"],
                "Cl-": initial["Cl-"],
                "Calcite": solid,
            }
            return amounts, water_kg, h

        def aqueous(solid):
            low = -18.0
            if anchor <= 0:

                def available(log_h):
                    h = 10.0**log_h
                    f0 = 1 / (1 + k1 / (h * g1) + k1 * k2 / (h * h * g2))
                    return anchor + (carbon - solid) * f0

                if available(1.0) <= 0:
                    raise ValueError("no positive water phase is available in the carbonate domain")
                low = brentq(available, low, 1.0) + 1e-10

            def charge(log_h):
                candidate, _, _ = water_state(log_h, solid)
                return net_charge_equivalents(candidate, CHARGES) - initial_charge

            log_h = brentq(charge, low, 1.0, xtol=1e-13, rtol=1e-14)
            candidate, water, h = water_state(log_h, solid)
            omega = g2 * g2 * (candidate["Ca++"] / water) * (candidate["CO3--"] / water) / ksp
            return candidate, water, h, omega

        if fixed_calcite_mol is not None:
            candidate = aqueous(fixed_calcite_mol)
        else:
            candidate = aqueous(0.0)
            if model.allow_calcite and limit > 0 and candidate[3] > 1:
                solid = brentq(
                    lambda n: aqueous(n)[3] - 1,
                    0,
                    limit,
                    xtol=max(limit * 1e-13, 5e-324),
                    rtol=1e-14,
                )
                candidate = aqueous(solid)
        return (*candidate, ionic, gamma)

    def ionic_residual(value):
        candidate, water, *_ = at_ionic_strength(value)
        actual = ionic_strength_from_amounts(candidate, CHARGES, solvent_mass_kg=water)
        return log10(actual) - value

    upper = log10(model.maximum_ionic_strength_mol_kg)
    if ionic_residual(upper) > 0:
        raise ValueError("aqueous state exceeds the declared ionic-strength applicability domain")
    log_ionic = brentq(ionic_residual, -16.0, upper, xtol=1e-13, rtol=1e-14)
    final, water, h, omega, ionic, gamma = at_ionic_strength(log_ionic)
    actual_ionic = ionic_strength_from_amounts(final, CHARGES, solvent_mass_kg=water)
    elements = element_amounts(final)
    residuals = {key: elements[key] - value for key, value in totals.items()}
    charge = net_charge_equivalents(final, CHARGES)
    g1, g2 = gamma[1], gamma[2]
    ah, aoh = g1 * final["H+"] / water, g1 * final["OH-"] / water
    aco2, ahco3, aco3 = (
        final["CO2*"] / water,
        g1 * final["HCO3-"] / water,
        g2 * final["CO3--"] / water,
    )
    action_errors = {
        "water": log10(ah * aoh / kw),
        "carbonate1": log10(ah * ahco3 / aco2 / k1) if aco2 > 0 else None,
        "carbonate2": log10(ah * aco3 / ahco3 / k2) if ahco3 > 0 else None,
    }
    if (
        max(abs(residuals[key]) / max(abs(value), 1e-12) for key, value in totals.items()) > 1e-10
        or abs(charge - initial_charge) > 1e-11
        or abs(charge) > 1e-11
        or max(abs(v) for v in action_errors.values() if v is not None) > 1e-8
        or abs(actual_ionic - ionic) > 1e-10 * max(actual_ionic, 1e-12)
    ):
        raise RuntimeError("carbonate element, charge, activity or ionic-strength closure failed")
    status = (
        "held"
        if fixed_calcite_mol is not None
        else "excluded"
        if not model.allow_calcite
        else "present"
        if final["Calcite"] > 0
        else "absent"
    )
    if (status == "present" and abs(omega - 1) > 1e-8) or (status == "absent" and omega > 1 + 1e-8):
        raise RuntimeError("finite calcite complementarity failed")
    return CarbonateEquilibriumResult(
        final,
        water,
        h,
        actual_ionic,
        omega,
        status,
        residuals,
        charge - initial_charge,
        action_errors,
        {
            "model_id": model.model_id,
            "temperature_K": temperature_K,
            "pressure_Pa": pressure_Pa,
            "water_activity": 1.0,
            "neutral_activity_coefficient": 1.0,
            "initial_charge_eq": initial_charge,
            "final_charge_eq": charge,
            "function_evaluations": evaluations,
            "activity_basis": "molality",
            "caloric_closure": "unavailable",
            "volume_model": "declared geometric holdup",
        },
    )
