"""Source-declared pure virial gas properties and the existing thermal integrator.

Residual Helmholtz energy at fixed T,V is RT(B/V+C/(2V**2)); all caloric
corrections and response derivatives use that same potential. No material
accuracy, mixture cross rule, phase equilibrium or open-flow model is inferred.
"""

from dataclasses import dataclass
from math import isfinite

from chemworld.physchem.property_packages import (
    ComponentPropertyPackage,
    SpeciesCaloricModel,
    evaluate_caloric_correlation,
)
from chemworld.physchem.property_reports import (
    R_J_PER_MOL_K as R,
)
from chemworld.physchem.property_reports import (
    PropertyEvaluation,
    StatePropertyReport,
)
from chemworld.physchem.specs import ComponentSpec, PropertyCorrelation
from chemworld.physchem.virial_correlations import virial_temperature_report
from chemworld.physchem.virial_gas import density_virial_state, virial_pressure


def _coefficients(package, temperature_K):
    results, curves = [], []
    for prefix, unit in (("second", "m^3/mol"), ("third", "m^6/mol^2")):
        candidates = package.by_property(prefix + "_virial_coefficient")
        if prefix == "third" and not candidates:
            results.append((0.0,) * 4)
            continue
        if len(candidates) != 1:
            raise ValueError(f"pure virial package requires exactly one {prefix} coefficient")
        curve = candidates[0]
        if "temperature" not in curve.validity_ranges:
            raise ValueError("virial consumer requires an explicit temperature range")
        evaluations = virial_temperature_report(curve, temperature_K=temperature_K)
        results.append(
            tuple(
                value.to(unit if i == 0 else f"({unit})/K" + (f"^{i}" if i > 1 else "")).value
                for i, value in enumerate(evaluations)
            )
        )
        curves.append(curve)
    return (*results, tuple(curves))


def _state_values(
    package,
    temperature_K,
    *,
    pressure_Pa=None,
    molar_volume_m3_mol=None,
    ideal_caloric=None,
    entropy=False,
):
    b, c, curves = _coefficients(package, temperature_K)
    t = temperature_K
    if (pressure_Pa is None) == (molar_volume_m3_mol is None):
        raise ValueError("supply exactly one of pressure or molar volume")
    root = None
    if pressure_Pa is not None:
        root = density_virial_state(t, pressure_Pa, (b[0], c[0]))
        v = root.molar_volume_m3_mol
    else:
        v = molar_volume_m3_mol
        pressure_Pa = virial_pressure(t, v, (b[0], c[0]))
    p = pressure_Pa
    stiffness = 1 + 2 * b[0] / v + 3 * c[0] / v**2
    if not isfinite(stiffness) or stiffness <= 1e-10:
        raise ValueError("virial state has nonpositive or singular mechanical stiffness")
    pv = -R * t / v**2 * stiffness
    pt = R / v * (1 + (b[0] + t * b[1]) / v + (c[0] + t * c[1]) / v**2)
    values = {
        "molar_volume": v,
        "density": package.component.molecular_weight_g_mol / 1000 / v,
        "compressibility_factor": p * v / (R * t),
        "dV_dT_P": -pt / pv,
        "dV_dP_T": 1 / pv,
    }
    if ideal_caloric is not None:
        if ideal_caloric.reference.phase != "gas":
            raise ValueError("virial caloric reference must explicitly describe ideal gas")
        requested = ("cv", "internal_energy", "entropy") if entropy else ("cv", "internal_energy")
        ideal = evaluate_caloric_correlation(
            ideal_caloric.heat_capacity,
            reference=ideal_caloric.reference,
            temperature_K=t,
            pressure_Pa=R * t / v,
            quantities=requested,
            validity_policy="raise",
        )
        ut = b[1] / v + c[1] / (2 * v**2)
        utt = b[2] / v + c[2] / (2 * v**2)
        u = ideal["internal_energy"].value - R * t * t * ut
        cv = ideal["cv"].value - R * (2 * t * ut + t * t * utt)
        cp = cv - t * pt * pt / pv
        if not isfinite(cv) or cv <= 0 or not isfinite(cp) or cp <= 0:
            raise ValueError("virial caloric state requires positive finite Cv and Cp")
        values.update(internal_energy=u, enthalpy=u + p * v, cv=cv, cp=cp, dH_dT_P=cp, dU_dT_V=cv)
        if entropy:
            s = ideal["entropy"].value - R * (
                (b[0] + t * b[1]) / v + (c[0] + t * c[1]) / (2 * v * v)
            )
            values.update(entropy=s, chemical_potential=u + p * v - t * s)
    return (
        values,
        p,
        {
            "B_temperature_derivatives": b,
            "C_temperature_derivatives": c,
            "coefficient_declarations": [curve.to_dict() for curve in curves],
            "truncation_order": 3 if len(curves) == 2 else 2,
            "root": None if root is None else root.to_dict(),
            "mechanical_stiffness": stiffness,
            "caloric_basis": "explicit ideal gas datum plus fixed-density residual Helmholtz",
            "scope": "pure single gas; local mechanical stability; no phase equilibrium",
        },
    )


_UNITS = {
    "molar_volume": "m^3/mol",
    "density": "kg/m^3",
    "compressibility_factor": "dimensionless",
    "dV_dT_P": "m^3/(mol*K)",
    "dV_dP_T": "m^3/(mol*Pa)",
    "internal_energy": "J/mol",
    "enthalpy": "J/mol",
    "entropy": "J/(mol*K)",
    "chemical_potential": "J/mol",
    "cv": "J/(mol*K)",
    "cp": "J/(mol*K)",
    "dH_dT_P": "J/(mol*K)",
    "dU_dT_V": "J/(mol*K)",
}


def evaluate_virial_gas_state(
    package, *, temperature_K, pressure_Pa, quantities, ideal_caloric=None
):
    """Unit-bearing original state report; no silent ideal-gas caloric fallback."""
    requested = set(quantities)
    if not requested or requested - _UNITS.keys():
        raise ValueError("unsupported virial gas property request")
    values, p, details = _state_values(
        package,
        temperature_K,
        pressure_Pa=pressure_Pa,
        ideal_caloric=ideal_caloric,
        entropy=bool(requested & {"entropy", "chemical_potential"}),
    )
    if requested - values.keys():
        raise ValueError(
            "virial caloric properties require an explicit ideal heat capacity/reference"
        )
    name = "pure-density-virial-gas-v1"
    props = {
        key: PropertyEvaluation(
            key,
            name,
            "residual_helmholtz_virial",
            values[key],
            _UNITS[key],
            {"temperature": temperature_K, "pressure": p},
        )
        for key in quantities
    }
    identifier = package.component.identifier
    return StatePropertyReport(
        name,
        "gas",
        temperature_K,
        p,
        {identifier: 1.0},
        props,
        {} if ideal_caloric is None else {identifier: ideal_caloric.reference},
        model_details=details,
    )


@dataclass(frozen=True)
class VirialGasThermodynamics:
    """Pure closed rigid-vessel U/Cv adapter; the caller owns inventory and clock.

    Mass-transfer/reaction rates are rejected. A moving volume or flow model
    needs its own consistent boundary-work/partial-energy declaration.
    """

    package: ComponentPropertyPackage
    ideal_caloric: SpeciesCaloricModel

    def __post_init__(self):
        if self.ideal_caloric.reference.phase != "gas":
            raise ValueError("virial caloric reference must explicitly describe ideal gas")
        _coefficients(self.package, self.ideal_caloric.reference.temperature_K)

    @property
    def inventory_keys(self):
        return (self.package.component.identifier,)

    def _at(self, state):
        (key,) = self.inventory_keys
        if set(state.amounts_mol) != {key} or state.amounts_mol[key] <= 0:
            raise ValueError("pure virial thermal state requires one positive component amount")
        n = state.amounts_mol[key]
        values, p, _ = _state_values(
            self.package,
            state.temperature_K,
            molar_volume_m3_mol=state.volume_L * 1e-3 / n,
            ideal_caloric=self.ideal_caloric,
        )
        return n, values, p

    def heat_capacity_j_k(self, state):
        n, values, _ = self._at(state)
        return n * values["cv"]

    def internal_energy_j(self, state):
        n, values, _ = self._at(state)
        return n * values["internal_energy"]

    def pressure_pa(self, state):
        return self._at(state)[2]

    def validate_state(self, state):
        self._at(state)

    def chemical_energy_rate_w(self, state, derivatives, contributions):
        if any(derivatives.values()) or any(
            item.legacy_reaction_energy_W is not None
            or item.amount_rates_mol_s
            or item.port_rates_mol_s
            or item.internal_transfers
            or item.reaction_progress_mol_s
            for item in contributions
        ):
            raise ValueError("pure closed virial thermal model does not support material changes")
        return 0.0

    def to_dict(self):
        return {
            "model": "pure-virial-thermal-v1",
            "package": self.package.to_dict(),
            "ideal_caloric": self.ideal_caloric.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload):
        if set(payload) != {"model", "package", "ideal_caloric"} or (
            payload["model"] != "pure-virial-thermal-v1"
        ):
            raise ValueError("unsupported virial thermal declaration")
        package = payload["package"]
        return cls(
            ComponentPropertyPackage(
                ComponentSpec.from_dict(package["component"]),
                tuple(PropertyCorrelation.from_dict(c) for c in package["correlations"]),
            ),
            SpeciesCaloricModel.from_dict(payload["ideal_caloric"]),
        )


def virial_gas_batch_system(thermodynamics, *, heat_flux=None, policy=None):
    """Compile a closed fixed-volume heating/cooling block on CoupledSystem."""
    from chemworld.physchem.coupled_system import CoupledSystem

    options = {}
    if heat_flux is not None:
        options["heat_flux"] = heat_flux
    if policy is not None:
        options["policy"] = policy
    return CoupledSystem(thermodynamics, **options)
