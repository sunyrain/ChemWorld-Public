"""Native solid/liquid solubility and cohesive-energy parameter calculations.

Equations: Gmehling (2012), Barton (1991), Hansen's three-parameter distance.
These are explicit equilibrium/affinity models, not experimental solubility
guarantees. Source records and input quantities retain their original owners.
"""

from dataclasses import dataclass
from math import isfinite

from chemworld.data.scientific_inputs import InputValue
from chemworld.physchem.caloric_math import exponential, logarithm, scalar_value
from chemworld.physchem.data_conflicts import DataSourceProvenance
from chemworld.physchem.property_reports import R_J_PER_MOL_K as R
from chemworld.physchem.property_reports import PropertyEvaluation
from chemworld.physchem.viscosity_kernels import _positive

HANSEN_SOURCES = ("ALSHEHRI_HANSEN", "HSPIPY_HANSEN", "RUBEN_HANSEN", "SCHRIER_HANSEN")


def solid_solubility(T, Tm, Hm, Cpl=0, Cps=0, gamma=1):
    """Eutectic solid saturation mole fraction; Cp and Hm are molar SI.

    Constant Cp difference and a supplied liquid activity coefficient are
    assumptions of this formula. Superheated solid and x>1 reject; no clipping.
    """
    for name, v in (
        ("temperature", T),
        ("melting temperature", Tm),
        ("fusion enthalpy", Hm),
        ("activity coefficient", gamma),
    ):
        _positive(v, name)
    _positive(Cpl, "liquid molar heat capacity", zero=True)
    _positive(Cps, "solid molar heat capacity", zero=True)
    if scalar_value(T) > scalar_value(Tm):
        raise ValueError("solid solubility requires T <= Tm")
    dcp = Cpl - Cps
    exponent = -Hm / R * (1 / T - 1 / Tm) + dcp / R * (Tm / T - 1 + logarithm(T / Tm))
    try:
        x = exponential(exponent) / gamma
    except OverflowError as exc:
        raise ValueError("solid solubility overflows") from exc
    _positive(x, "solid solubility", zero=True)
    if scalar_value(x) > 1:
        raise ValueError("solid solubility exceeds a mole fraction; no implicit clipping")
    return x


def freezing_point_depression(
    Tm, Hm, *, mole_fraction=None, molality_mol_kg=None, solvent_molecular_weight_g_mol=None
):
    """Ideal dilute-solute depression of the solvent melting point, K.

    Hm and Tm belong to the solvent; molality needs its MW. The molality
    expression is the dilute limit, not an exact conversion to mole fraction.
    """
    _positive(Tm, "solvent melting temperature")
    _positive(Hm, "solvent fusion enthalpy")
    if mole_fraction is not None:
        if molality_mol_kg is not None or solvent_molecular_weight_g_mol is not None:
            raise ValueError("choose mole fraction or molality with solvent MW")
        _positive(mole_fraction, "solute mole fraction", zero=True)
        if mole_fraction >= 1:
            raise ValueError("solute mole fraction must be below one")
        x = mole_fraction
    else:
        _positive(molality_mol_kg, "solute molality", zero=True)
        _positive(solvent_molecular_weight_g_mol, "solvent molecular weight")
        x = molality_mol_kg * solvent_molecular_weight_g_mol / 1000
    return _positive(R * Tm**2 / Hm * x, "freezing point depression", zero=True)


def hildebrand_parameter(T, *, vaporization_enthalpy_J_mol, liquid_molar_volume_m3_mol):
    """sqrt((Hvap-RT)/Vm), Pa**.5. No real value if cohesive energy is negative."""
    _positive(T, "temperature")
    _positive(vaporization_enthalpy_J_mol, "molar vaporization enthalpy", zero=True)
    _positive(liquid_molar_volume_m3_mol, "liquid molar volume")
    cohesive = vaporization_enthalpy_J_mol - R * T
    _positive(cohesive, "cohesive molar energy", zero=True)
    try:
        return (cohesive / liquid_molar_volume_m3_mol) ** 0.5
    except ZeroDivisionError as exc:
        raise ValueError("Hildebrand derivative does not exist at zero cohesive energy") from exc


@dataclass(frozen=True)
class HansenParameters:
    """Source parameter vector, with the original input values and provenance.

    This is immutable configuration, not a material or experiment state.
    Missing source values remain missing; no temperature dependence is inferred.
    """

    component_id: str
    dispersion: InputValue
    polar: InputValue
    hydrogen_bonding: InputValue
    source: DataSourceProvenance
    record_id: str

    def __post_init__(self) -> None:
        if not self.component_id or not self.record_id or not self.source.source_id:
            raise ValueError("Hansen parameters need component and source identities")
        for value in (self.dispersion, self.polar, self.hydrogen_bonding):
            converted = value.to("Pa^0.5")
            if converted.value is not None and converted.value < 0:
                raise ValueError("Hansen parameters must be nonnegative")

    @classmethod
    def from_record(cls, record):
        if record.source_id not in HANSEN_SOURCES:
            raise ValueError("select one of the four explicit Hansen sources")
        return cls(
            record.cas_number or record.source_key,
            *(record.value("HANSEN_DELTA_" + key).input_value("Pa^0.5") for key in "DPH"),
            record.source.provenance,
            record.record_id,
        )

    def values_pa_half(self):
        values = (self.dispersion, self.polar, self.hydrogen_bonding)
        if any(v.quality != "valid" or v.value is None for v in values):
            raise ValueError("Hansen comparison requires all three valid source values")
        return tuple(v.to("Pa^0.5").value for v in values)

    def to_dict(self):
        return {
            "model": "hansen-parameters-1",
            "component_id": self.component_id,
            "dispersion": self.dispersion.to_dict(),
            "polar": self.polar.to_dict(),
            "hydrogen_bonding": self.hydrogen_bonding.to_dict(),
            "source": self.source.to_dict(),
            "record_id": self.record_id,
        }

    @classmethod
    def from_dict(cls, payload):
        expected = {
            "model",
            "component_id",
            "dispersion",
            "polar",
            "hydrogen_bonding",
            "source",
            "record_id",
        }
        if set(payload) != expected or payload["model"] != "hansen-parameters-1":
            raise ValueError("unknown Hansen parameter schema")
        return cls(
            payload["component_id"],
            *(
                InputValue.from_dict(payload[k])
                for k in ("dispersion", "polar", "hydrogen_bonding")
            ),
            DataSourceProvenance.from_dict(payload["source"]),
            payload["record_id"],
        )


def hansen_distance(solute, solvent, *, interaction_radius=None):
    """Source vectors -> Hansen distance or explicit relative energy distance.

    With a supplied radius, RED<1 means inside the declared compatibility
    sphere. It does not prove solubility or authorize a laboratory operation.
    """
    a, b = solute.values_pa_half(), solvent.values_pa_half()
    distance = (
        sum(weight * (x - y) ** 2 for weight, x, y in zip((4, 1, 1), a, b, strict=True)) ** 0.5
    )
    inputs = {
        **{f"solute_{k}_Pa_half": v for k, v in zip("DPH", a, strict=True)},
        **{f"solvent_{k}_Pa_half": v for k, v in zip("DPH", b, strict=True)},
    }
    prop, unit = "hansen_distance", "Pa^0.5"
    if interaction_radius is not None:
        if interaction_radius.quality != "valid" or interaction_radius.value is None:
            raise ValueError("interaction radius requires a valid declared value")
        radius = _positive(interaction_radius.to("Pa^0.5").value, "Hansen interaction radius")
        inputs["interaction_radius_Pa_half"] = radius
        distance /= radius
        prop, unit = "hansen_relative_energy_distance", "dimensionless"
    if not isfinite(distance):
        raise ValueError("Hansen distance is nonfinite")
    return PropertyEvaluation(
        prop, f"{solute.record_id}/{solvent.record_id}", "Hansen_4_1_1", distance, unit, inputs
    )
