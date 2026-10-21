"""Liquid free-volume and compressed viscosity equations with explicit inputs."""

from math import isfinite

from chemworld.physchem.caloric_math import scalar_value
from chemworld.physchem.viscosity_kernels import _positive


def lucas_liquid_viscosity(
    T,
    pressure_Pa,
    *,
    critical_temperature_K,
    critical_pressure_Pa,
    acentric_factor,
    saturation_pressure_Pa,
    saturation_viscosity_Pa_s,
    below_saturation="reject",
):
    """Lucas pressure correction, Pa s. No saturation pressure/viscosity is guessed.

    Below Psat the default rejects use of a compressed-liquid relation.
    below_saturation='saturated_limit' explicitly selects the public reference
    API's zero-pressure-correction behavior without asserting phase stability.
    """
    for value, label in (
        (T, "temperature"),
        (pressure_Pa, "pressure"),
        (critical_temperature_K, "critical temperature"),
        (critical_pressure_Pa, "critical pressure"),
        (saturation_viscosity_Pa_s, "saturation viscosity"),
    ):
        _positive(value, label)
    _positive(saturation_pressure_Pa, "saturation pressure", zero=True)
    if (
        isinstance(acentric_factor, bool)
        or not isinstance(acentric_factor, int | float)
        or not isfinite(acentric_factor)
    ):
        raise ValueError("acentric factor must be finite")
    if below_saturation not in {"reject", "saturated_limit"}:
        raise ValueError("unknown below-saturation policy")
    tr = T / critical_temperature_K
    if tr > 1:
        raise ValueError("Lucas liquid requires T <= Tc")
    difference = (pressure_Pa - saturation_pressure_Pa) / critical_pressure_Pa
    if difference < 0:
        if below_saturation == "reject":
            raise ValueError("pressure is below saturation pressure")
        difference = 0
    a = 0.9991 - 4.674e-4 / (1.0523 * tr**-0.03877 - 1.0513)
    d = 0.3257 / (1.0039 - tr**2.573) ** 0.2906 - 0.2086
    c = sum(
        value * tr**i
        for i, value in enumerate(
            (-0.07921, 2.1616, -13.4040, 44.1706, -84.8291, 96.1209, -59.8127, 15.6719)
        )
    )
    denominator = _positive(1 + c * acentric_factor * difference, "Lucas denominator")
    result = saturation_viscosity_Pa_s * (1 + d * (difference / 2.118) ** a) / denominator
    return _positive(result, "Lucas compressed viscosity")


def przedziecki_sridhar_viscosity(
    molar_volume_m3_mol,
    *,
    melting_volume_m3_mol,
    melting_temperature_K,
    critical_temperature_K,
    critical_pressure_Pa,
    critical_volume_m3_mol,
    acentric_factor,
    molecular_weight_g_mol,
):
    """Przedziecki-Sridhar free-volume form with actual and melting volumes.

    Equation documented in Chempak 8.5.4 / Reid: internal cm³/mol, bar, cP.
    No T argument is ignored: temperature affects actual volume, supplied by
    the caller. The melting volume is a separate material parameter; it is
    not silently equated to the volume at the operating temperature.
    """
    for value, label in (
        (molar_volume_m3_mol, "actual molar volume"),
        (melting_volume_m3_mol, "melting molar volume"),
        (melting_temperature_K, "melting temperature"),
        (critical_temperature_K, "critical temperature"),
        (critical_pressure_Pa, "critical pressure"),
        (critical_volume_m3_mol, "critical volume"),
        (molecular_weight_g_mol, "molecular weight"),
    ):
        _positive(value, label)
    if (
        isinstance(acentric_factor, bool)
        or not isinstance(acentric_factor, int | float)
        or not isfinite(acentric_factor)
    ):
        raise ValueError("acentric factor must be finite")
    trm = melting_temperature_K / critical_temperature_K
    denominator = (
        12.94
        + 0.10 * molecular_weight_g_mol
        - 0.23 * critical_pressure_Pa / 1e5
        + 0.0424 * melting_temperature_K
        - 11.58 * trm
    )
    if denominator == 0:
        raise ValueError("Przedziecki E denominator is singular")
    e = -1.12 + critical_volume_m3_mol * 1e6 / denominator
    v0 = (
        0.0085 * acentric_factor * critical_temperature_K
        - 2.02
        + melting_volume_m3_mol * 1e6 / (0.342 * trm + 0.894)
    )
    _positive(e, "Przedziecki E")
    _positive(v0, "Przedziecki free volume")
    _positive(molar_volume_m3_mol * 1e6 - v0, "Przedziecki V-V0")
    return 0.001 * v0 / (e * (molar_volume_m3_mol * 1e6 - v0))


def przedziecki_sridhar_temperature_viscosity(
    T,
    *,
    melting_volume_m3_mol,
    melting_temperature_K,
    critical_temperature_K,
    critical_pressure_Pa,
    critical_volume_m3_mol,
    acentric_factor,
    molecular_weight_g_mol,
    volume_scale_m3_mol=None,
):
    """Pa s using the original paper's quartic Gunn-Yamada volume estimate.

    Przedziecki-Sridhar (1985) eqs. 8-12. The optional scale defaults to Vc,
    the approximation used by chemicals' convenience API. Its Vm argument
    corresponds to the MELTING volume, despite its docstring saying volume
    at T. Actual operating volume is computed here; use the free-volume
    kernel directly when an independently known operating volume is available.
    No volume or density is stored in a separate physical state.
    """
    _positive(T, "temperature")
    _positive(critical_temperature_K, "critical temperature")
    tr = T / critical_temperature_K
    if not 0.2 <= scalar_value(tr) <= 1 or scalar_value(T) < melting_temperature_K:
        raise ValueError("Przedziecki estimate requires T >= Tm and 0.2 <= T/Tc <= 1")
    scale = critical_volume_m3_mol if volume_scale_m3_mol is None else volume_scale_m3_mol
    _positive(scale, "Gunn-Yamada volume scale")
    reduced = 0.33593 - 0.33953 * tr + 1.51941 * tr**2 - 2.02512 * tr**3 + 1.11422 * tr**4
    gamma = 0.29607 - 0.09045 * tr - 0.04842 * tr**2
    volume = scale * reduced * (1 - acentric_factor * gamma)
    return przedziecki_sridhar_viscosity(
        volume,
        melting_volume_m3_mol=melting_volume_m3_mol,
        melting_temperature_K=melting_temperature_K,
        critical_temperature_K=critical_temperature_K,
        critical_pressure_Pa=critical_pressure_Pa,
        critical_volume_m3_mol=critical_volume_m3_mol,
        acentric_factor=acentric_factor,
        molecular_weight_g_mol=molecular_weight_g_mol,
    )


def lucas_liquid_viscosity_report(
    saturation_report,
    *,
    temperature_K,
    pressure_Pa,
    critical_temperature_K,
    critical_pressure_Pa,
    acentric_factor,
    saturation_pressure_Pa,
    below_saturation="reject",
):
    """Consume an existing saturation report, returning the same report type.

    The saturated value must belong to this temperature and, if it records a
    pressure, this Psat. The correction consumes the actual requested P; it
    does not alter the source correlation's saturation interval or invent EOS.
    """
    from chemworld.physchem.property_reports import PropertyEvaluation
    from chemworld.physchem.transport_properties import TransportPropertyReport

    source = saturation_report.evaluation
    if saturation_report.phase != "liquid" or source.property_id != "liquid_viscosity":
        raise ValueError("Lucas requires a liquid saturation-viscosity report")
    if source.inputs.get("temperature") != temperature_K:
        raise ValueError("saturation report must use the same temperature in K")
    if source.inputs.get("pressure", saturation_pressure_Pa) != saturation_pressure_Pa:
        raise ValueError("saturation report pressure must equal Psat in Pa")
    value = lucas_liquid_viscosity(
        temperature_K,
        pressure_Pa,
        critical_temperature_K=critical_temperature_K,
        critical_pressure_Pa=critical_pressure_Pa,
        acentric_factor=acentric_factor,
        saturation_pressure_Pa=saturation_pressure_Pa,
        saturation_viscosity_Pa_s=source.to("Pa*s").value,
        below_saturation=below_saturation,
    )
    return TransportPropertyReport(
        PropertyEvaluation(
            "liquid_viscosity",
            source.correlation_id + ":Lucas",
            "Lucas",
            value,
            "Pa*s",
            {
                "temperature": temperature_K,
                "pressure": pressure_Pa,
                "saturation_pressure": saturation_pressure_Pa,
            },
            source.warnings,
        ),
        "liquid",
        "Lucas compressed liquid",
        "out_of_range" if saturation_report.validity_status == "out_of_range" else "estimated",
        uncertainty_note="Pressure correction; source uncertainty is not a combined uncertainty.",
        reference_reading=(
            *saturation_report.reference_reading,
            "Reid et al., Properties of Gases and Liquids, Lucas",
        ),
    )
