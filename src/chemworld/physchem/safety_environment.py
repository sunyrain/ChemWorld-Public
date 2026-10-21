"""Published safety correlations and consumers of explicit source assessments.

These are bounded numerical models and historical source comparisons, not
runtime permissions or a certification against a complete safety standard.
"""

from dataclasses import replace
from math import fsum, isclose, isfinite
from types import MappingProxyType

from chemworld.physchem.caloric_math import scalar_value
from chemworld.physchem.equilibrium import liquid_liquid_split
from chemworld.physchem.property_reports import R_J_PER_MOL_K, PropertyEvaluation
from chemworld.physchem.viscosity_kernels import _positive

ESTIMATION_METHODS = ("Suzuki (1994)", "Crowl and Louvar (2001)")
ISO_10156_DILUENT_COEFFICIENTS = MappingProxyType(
    {
        "7782-44-7": 1.0,
        "7727-37-9": 1.0,
        "124-38-9": 1.5,
        "7440-59-7": 0.9,
        "7440-37-1": 0.55,
        "7440-01-9": 0.7,
        "7439-90-9": 0.5,
        "7440-63-3": 0.5,
        "7446-09-5": 1.5,
        "2551-62-4": 4.0,
        "75-73-0": 2.0,
        "76-19-7": 1.5,
        "354-33-6": 3.5,
    }
)


def _fraction(value, label):
    _positive(value, label, zero=True)
    if scalar_value(value) > 1:
        raise ValueError(f"{label} exceeds unity")
    return value


def ppmv_to_mass_concentration(ppmv, molecular_weight_g_mol, *, temperature_K, pressure_Pa):
    """Ideal gas mg/m3; all state inputs are explicit, no STP inference."""
    _positive(ppmv, "volumetric concentration", zero=True)
    _fraction(ppmv / 1e6, "volumetric concentration")
    for v, label in (
        (molecular_weight_g_mol, "molecular weight"),
        (temperature_K, "temperature"),
        (pressure_Pa, "pressure"),
    ):
        _positive(v, label)
    return ppmv * pressure_Pa * molecular_weight_g_mol / (R_J_PER_MOL_K * temperature_K * 1000)


def mass_concentration_to_ppmv(mg_m3, molecular_weight_g_mol, *, temperature_K, pressure_Pa):
    _positive(mg_m3, "mass concentration", zero=True)
    unit_value = ppmv_to_mass_concentration(
        1, molecular_weight_g_mol, temperature_K=temperature_K, pressure_Pa=pressure_Pa
    )
    result = mg_m3 / unit_value
    _fraction(result / 1e6, "volumetric concentration")
    return result


def suzuki_flammability_limit(heat_of_combustion_J_mol, *, limit):
    """Suzuki 1994: negative Hc, MJ/mol polynomial, percentage -> fraction.

    Public equations: DOI 10.1002/fam.810180509 and 10.1002/fam.810180608.
    The lower-limit expression continues across two displayed lines in the
    public documentation: -3.42/h + .569*h + .0538*h**2 + 1.80.
    The caller must establish applicability to the compound class; this
    scalar heat input cannot prove the absence of halogens.
    """
    h = -_positive(-heat_of_combustion_J_mol, "magnitude of negative combustion enthalpy") / 1e6
    if limit == "LFL":
        result = (-3.42 / h + 0.569 * h + 0.0538 * h * h + 1.80) / 100
    elif limit == "UFL":
        result = (6.3 * h + 0.567 * h * h + 23.5) / 100
    else:
        raise ValueError("select LFL or UFL")
    return _fraction(result, "predicted flammability limit")


def crowl_louvar_flammability_limit(atoms, *, limit):
    """CHO compounds; no implicit neglect of nitrogen, sulfur or halogens."""
    if set(atoms) - {"C", "H", "O"}:
        raise ValueError("Crowl-Louvar supports only the declared CHO model")
    for symbol, count in atoms.items():
        _positive(count, symbol + " count", zero=True)
    _positive(atoms.get("C", 0), "carbon count")
    denominator = 4.76 * atoms.get("C", 0) + 1.19 * atoms.get("H", 0) - 2.38 * atoms.get("O", 0) + 1
    _positive(denominator, "Crowl-Louvar denominator")
    if limit not in {"LFL", "UFL"}:
        raise ValueError("select LFL or UFL")
    return _fraction(
        (0.55 if limit == "LFL" else 3.5) / denominator, "predicted flammability limit"
    )


def flammability_estimate(*, limit, method, heat_of_combustion_J_mol=None, atoms=None):
    """Explicit method; unavailable required input is never an inert gas."""
    if method == ESTIMATION_METHODS[0]:
        if heat_of_combustion_J_mol is None:
            raise ValueError("Suzuki requires combustion enthalpy")
        return suzuki_flammability_limit(heat_of_combustion_J_mol, limit=limit)
    if method == ESTIMATION_METHODS[1]:
        if atoms is None:
            raise ValueError("Crowl-Louvar requires atom counts")
        return crowl_louvar_flammability_limit(atoms, limit=limit)
    raise ValueError("unknown flammability estimation method")


def fire_mixture_limit(mole_fractions, limits):
    """Le Chatelier rule for explicitly normalized combustible components."""
    if not mole_fractions or len(mole_fractions) != len(limits):
        raise ValueError("fractions and limits must be nonempty and aligned")
    for y in mole_fractions:
        _fraction(y, "mole fraction")
    if not isclose(fsum(mole_fractions), 1, abs_tol=1e-12, rel_tol=0):
        raise ValueError("combustible mole fractions must sum to one")
    denominator = 0
    for y, value in zip(mole_fractions, limits, strict=True):
        if y == 0:
            continue
        _positive(value, "flammability limit")
        _fraction(value, "flammability limit")
        denominator += y / value
    return 1 / denominator


def iso_10156_lower_limit(mole_fractions, lower_limits, cas_numbers):
    """ISO 10156 (2017) correlation in a consistent mole-fraction basis.

    Let A/B be total combustible/diluent fractions, f the combustible-only
    Le Chatelier limit and K the normalized diluent coefficient. The public
    equation's common correction is c = 1 + (K-1)*(B/A)*f/(1-f), giving c*f/A.
    Its printed denominator '100-f' mixes percentage and fraction conventions;
    every unity here uses the fraction basis, including the numerator.
    Source: https://chemicals.readthedocs.io/chemicals.safety.html

    The 13 coefficients are scientific source data. Oxygen's inclusion in
    this correlation is not a general assertion of chemical inertness or
    a model of oxygen-enriched combustion. Unknown positive-fraction species
    with missing LFL are not silently treated as nitrogen.
    """
    if not mole_fractions or not len(mole_fractions) == len(lower_limits) == len(cas_numbers):
        raise ValueError("fractions, lower limits and CAS identities must be nonempty and aligned")
    for y in mole_fractions:
        _fraction(y, "mole fraction")
    if not isclose(fsum(mole_fractions), 1, abs_tol=1e-12, rel_tol=0):
        raise ValueError("mole fractions must sum to one")
    combustible, diluent, weighted, inverse = 0, 0, 0, 0
    for y, value, cas in zip(mole_fractions, lower_limits, cas_numbers, strict=True):
        if y == 0:
            continue
        if cas in ISO_10156_DILUENT_COEFFICIENTS:
            if value is not None:
                raise ValueError("a declared ISO diluent must have no combustible LFL input")
            diluent += y
            weighted += y * ISO_10156_DILUENT_COEFFICIENTS[cas]
        else:
            if value is None:
                raise ValueError("missing LFL for a species without an ISO diluent coefficient")
            _positive(value, "lower flammability limit")
            if value >= 1:
                raise ValueError("ISO combustible lower limit must be below one")
            combustible += y
            inverse += y / value
    if combustible == 0:
        raise ValueError("no combustible fraction: no finite lower flammability limit")
    f = combustible / inverse
    correction = 1 if diluent == 0 else 1 + (weighted - diluent) / combustible * f / (1 - f)
    _positive(correction, "ISO correction factor")
    return _fraction(correction / inverse, "ISO predicted mixture limit")


def nfpa_30_classification(
    flash_temperature_K, *, boiling_temperature_K=None, vapor_pressure_at_100F_Pa=None
):
    """Historical NFPA 30 liquid classes; supplied closed-cup flash point."""
    _positive(flash_temperature_K, "flash temperature")
    cut73, cut100, cut140, cut200 = ((f - 32) * 5 / 9 + 273.15 for f in (73, 100, 140, 200))
    criteria = []
    if boiling_temperature_K is not None:
        _positive(boiling_temperature_K, "normal boiling temperature")
        criteria.append(boiling_temperature_K < cut100)
    if vapor_pressure_at_100F_Pa is not None:
        _positive(vapor_pressure_at_100F_Pa, "saturation pressure at 100 F", zero=True)
        criteria.append(vapor_pressure_at_100F_Pa > 101325)
    if criteria and any(v != criteria[0] for v in criteria):
        raise ValueError("boiling point and saturation-pressure criteria disagree")
    if flash_temperature_K < cut73:
        if not criteria:
            raise ValueError("class IA/IB needs a boiling point or vapor pressure at 100 F")
        return "IA" if criteria[0] else "IB"
    for upper, result in ((cut100, "IC"), (cut140, "II"), (cut200, "IIIA")):
        if flash_temperature_K < upper:
            return result
    return "IIIB"


def _source_number(assessment, property_id):
    if assessment.property_id != property_id or assessment.quality != "valid":
        raise ValueError(f"consumer requires a valid {property_id} source assessment")
    return assessment.input_value().value


def emission_impact(masses, assessments):
    """Explicit emitted masses -> kg reference-substance equivalent.

    Missing factors, mixed horizons/versions or different response metrics
    reject the calculation. A material inventory is not implicitly emitted.
    """
    if not masses or set(masses) != set(assessments):
        raise ValueError("emitted mass and factor identities must match exactly")
    first = next(iter(assessments.values()))
    if first.property_id not in {"GWP", "GTP", "ODP"}:
        raise ValueError("select an explicit environmental impact property")
    if first.unit is None:
        raise ValueError("an ODP interval text is not a scalar impact factor")
    total, inputs, records, warnings = 0, {}, [], []
    for identity, mass in masses.items():
        factor = assessments[identity]
        if (
            factor.component_id != identity
            or factor.property_id != first.property_id
            or factor.method != first.method
        ):
            raise ValueError("impact identity, property, version and horizon must agree")
        m = _positive(mass.to("kg").value, "emitted mass", zero=True)
        v = _source_number(factor, first.property_id)
        total += m * v
        inputs[f"mass_kg:{identity}"] = m
        inputs[f"factor:{identity}"] = v
        records.append(factor.record_id)
        warnings.extend(factor.warnings)
    if not isfinite(total):
        raise ValueError("nonfinite impact total")
    warnings.append(
        "Equal-mass reference equivalent under the selected method; not a dynamic climate model"
    )
    return PropertyEvaluation(
        first.property_id + "_mass_equivalent",
        "/".join(records),
        first.method,
        total,
        "kg",
        inputs,
        tuple(dict.fromkeys(warnings)),
    )


def octanol_water_partition(
    feed_amounts_mol, assessments, *, water_volume_L, octanol_volume_L, temperature_K
):
    """Original finite LLE split using neutral-species source logP values."""
    if not feed_amounts_mol or set(feed_amounts_mol) != set(assessments):
        raise ValueError("logP sources and finite feed identities must match exactly")
    _positive(water_volume_L, "water volume")
    _positive(octanol_volume_L, "octanol volume")
    _positive(temperature_K, "declared temperature")
    coefficients = {}
    for identity, amount in feed_amounts_mol.items():
        _positive(amount, "feed amount", zero=True)
        value = assessments[identity]
        if value.component_id != identity:
            raise ValueError("logP source identity differs from feed")
        coefficients[identity] = _positive(
            10 ** _source_number(value, "logP"), "octanol/water partition coefficient"
        )
    result = liquid_liquid_split(
        feed_amounts_mol,
        partition_coefficients=coefficients,
        aqueous_volume_L=water_volume_L,
        organic_volume_L=octanol_volume_L,
        temperature_K=temperature_K,
    )
    return replace(
        result,
        stability_diagnostic={
            **result.stability_diagnostic,
            "source_assessments": {k: v.to_dict() for k, v in assessments.items()},
            "application": (
                "Neutral solute only; source temperature unknown; no temperature correction"
            ),
        },
    )


def gas_exposure_comparison(mixture, identity, assessment, *, source_identity, averaging_window):
    """Explicit gas-state concentration versus a matching source/window.

    The caller supplies the TWA/STEL/Ceiling window represented by the
    concentration. This does not calculate an unobserved exposure history.
    """
    if mixture.phase_label != "gas" or identity not in mixture.component_ids:
        raise ValueError("exposure comparison requires the selected component in a gas mixture")
    if assessment.component_id != source_identity:
        raise ValueError("declared source identity does not match the exposure limit")
    if averaging_window not in {"TWA", "STEL", "Ceiling"}:
        raise ValueError("select TWA, STEL or Ceiling averaging window")
    limit = _positive(_source_number(assessment, averaging_window), "exposure limit")
    index = mixture.component_ids.index(identity)
    ppm = mixture.mole_fractions[index] * 1e6
    if assessment.unit == "ppmv":
        concentration = ppm
    elif assessment.unit == "mg/m^3":
        concentration = ppmv_to_mass_concentration(
            ppm,
            mixture.molecular_weights_g_mol[index],
            temperature_K=mixture.temperature_K,
            pressure_Pa=mixture.pressure_Pa,
        )
    else:
        raise ValueError("unsupported source concentration unit")
    return PropertyEvaluation(
        "exposure_limit_ratio",
        assessment.record_id,
        assessment.method,
        concentration / limit,
        "dimensionless",
        {
            "concentration": concentration,
            "limit": limit,
            "temperature": mixture.temperature_K,
            "pressure": mixture.pressure_Pa,
        },
        (
            *assessment.warnings,
            f"Caller-declared {averaging_window} concentration; no exposure history inferred",
        ),
    )


def source_flammable_mixture(
    mixture, assessments, *, source_identities, limit, method="Le_Chatelier"
):
    """Original MixtureSpec -> source-selected combustible-mixture limit."""
    if (
        mixture.phase_label != "gas"
        or set(assessments) != set(mixture.component_ids)
        or set(source_identities) != set(mixture.component_ids)
    ):
        raise ValueError("gas mixture, source assessments and identities must match exactly")
    if limit not in {"LFL", "UFL"}:
        raise ValueError("select LFL or UFL")
    if method not in {"Le_Chatelier", "ISO_10156_2017"}:
        raise ValueError("unknown flammable mixture method")
    if method == "ISO_10156_2017" and limit != "LFL":
        raise ValueError("ISO 10156 correlation supports only LFL")
    values, warnings, records = [], [], []
    for identity in mixture.component_ids:
        source = assessments[identity]
        if source is None:
            if method != "ISO_10156_2017":
                raise ValueError("Le Chatelier requires every combustible source assessment")
            values.append(None)
            continue
        if source.component_id != source_identities[identity]:
            raise ValueError("flammability source identity mismatch")
        values.append(_source_number(source, limit))
        warnings.extend(source.warnings)
        records.append(source.record_id)
    if method == "ISO_10156_2017":
        result = iso_10156_lower_limit(
            mixture.mole_fractions,
            values,
            tuple(source_identities[k] for k in mixture.component_ids),
        )
        warnings.append(
            "ISO 10156 source correlation; specified diluents only; no temperature/pressure "
            "correction or oxygen-enriched combustion model"
        )
    else:
        result = fire_mixture_limit(mixture.mole_fractions, values)
        warnings.append(
            "Combustible components only; no inert-gas or temperature/pressure correction"
        )
    if limit == "UFL":
        warnings.append("Le Chatelier upper-limit rule has lower empirical reliability")
    return PropertyEvaluation(
        limit,
        "/".join(records),
        method,
        result,
        "dimensionless",
        dict(zip(mixture.component_ids, mixture.mole_fractions, strict=True)),
        tuple(dict.fromkeys(warnings)),
    )
