"""Interface mixing and capillary response on existing mixtures and reports.

No inferred phase split, EOS or private state. Pure values supplied to mixing
rules must belong to their stated temperatures; parachors use SI
N**.25*m**2.75/mol, molar densities mol/m³, surface tensions N/m.
"""

from math import cos, isclose, isfinite, pi

from chemworld.physchem.conductivity_mixtures import _vectors
from chemworld.physchem.interface_kernels import _power
from chemworld.physchem.property_reports import PropertyEvaluation
from chemworld.physchem.viscosity_kernels import _positive


def _aligned(fractions, *vectors):
    _vectors(fractions)
    if any(len(v) != len(fractions) for v in vectors):
        raise ValueError("interface vectors must be aligned")
    for vector in vectors:
        for value in vector:
            _positive(value, "component interface property", zero=True)


def winterfeld_scriven_davis(mole_fractions, surface_tensions_N_m, molar_densities_mol_m3):
    _aligned(mole_fractions, surface_tensions_N_m, molar_densities_mol_m3)
    for rho in molar_densities_mol_m3:
        _positive(rho, "molar density")
    volumes = [x / rho for x, rho in zip(mole_fractions, molar_densities_mol_m3, strict=True)]
    return (
        sum(v * s**0.5 for v, s in zip(volumes, surface_tensions_N_m, strict=True)) / sum(volumes)
    ) ** 2


def diguilio_teja(
    T, mole_fractions, tensions_at_boiling_N_m, boiling_temperatures_K, critical_temperatures_K
):
    _positive(T, "temperature")
    _aligned(
        mole_fractions, tensions_at_boiling_N_m, boiling_temperatures_K, critical_temperatures_K
    )
    for tb, tc in zip(boiling_temperatures_K, critical_temperatures_K, strict=True):
        if not 0 < tb < tc:
            raise ValueError("Diguilio-Teja requires 0 < Tb < Tc for each component")
    tb, tc, sigma = (
        sum(x * v for x, v in zip(mole_fractions, values, strict=True))
        for values in (boiling_temperatures_K, critical_temperatures_K, tensions_at_boiling_N_m)
    )
    return 1.002855 * _power((tc / T - 1) / (tc / tb - 1), 1.118091) * T / tb * sigma


def weinaug_katz(
    parachors_SI,
    liquid_molar_volume_m3_mol,
    vapor_molar_volume_m3_mol,
    liquid_mole_fractions,
    vapor_mole_fractions,
):
    """Liquid-vapor fourth-power rule. Vmg=None explicitly neglects vapor density."""
    _aligned(liquid_mole_fractions, parachors_SI, vapor_mole_fractions)
    _vectors(vapor_mole_fractions)
    _positive(liquid_molar_volume_m3_mol, "liquid molar volume")
    if vapor_molar_volume_m3_mol is not None:
        _positive(vapor_molar_volume_m3_mol, "vapor molar volume")
    vapor_density = 0 if vapor_molar_volume_m3_mol is None else 1 / vapor_molar_volume_m3_mol
    value = (
        sum(
            p * (x / liquid_molar_volume_m3_mol - y * vapor_density)
            for p, x, y in zip(
                parachors_SI, liquid_mole_fractions, vapor_mole_fractions, strict=True
            )
        )
        ** 4
    )
    return _positive(value, "surface tension", zero=True)


def water_oil_interfacial_tension(
    T, *, water_density_kg_m3, oil_density_kg_m3, oil_critical_temperature_K
):
    """Meybodi-Daryasafar-Karimi water/hydrocarbon empirical fit, N/m.

    Scientific A1..A10 from the reference's literal coefficient table and its
    A10=3.5 comment; density contrast is absolute, internally g/mL. This is
    not a universal two-liquid model and does not infer coexistence.
    """
    for name, value in (
        ("temperature", T),
        ("water density", water_density_kg_m3),
        ("oil density", oil_density_kg_m3),
        ("oil Tc", oil_critical_temperature_K),
    ):
        _positive(value, name)
    d = abs(water_density_kg_m3 - oil_density_kg_m3) / 1000
    numerator = -0.13687340042 - 0.30391828884 * d + 0.56225871072 * d * d - 0.33074367079 * d**3
    denominator = (
        -3.0050179309
        + 5.8914210205e-5 * T**-4.1388901263 / oil_critical_temperature_K
        + 3.008429903 * T**-0.0038203072876
    )
    if denominator == 0:
        raise ValueError("water-oil interface fit denominator is zero")
    return _positive(0.001 * _power(numerator / denominator, 3.5), "interfacial tension", zero=True)


def _mixture_inputs(mixture, phase):
    if mixture.phase_label != phase:
        raise ValueError(f"interface requires an explicit {phase} mixture")
    _vectors(mixture.mole_fractions, mixture.molecular_weights_g_mol)
    _vectors(mixture.mass_fractions)
    for x, w, mw in zip(
        mixture.mole_fractions, mixture.mass_fractions, mixture.molecular_weights_g_mol, strict=True
    ):
        if not isclose(
            w, x * mw / mixture.average_molecular_weight_g_mol, rel_tol=1e-9, abs_tol=1e-12
        ):
            raise ValueError("mass and mole fractions must describe the same composition")
    _positive(mixture.temperature_K, "temperature")
    _positive(mixture.pressure_Pa, "pressure")
    return {
        "temperature": mixture.temperature_K,
        "pressure": mixture.pressure_Pa,
        **{
            f"x:{key}": x
            for key, x in zip(mixture.component_ids, mixture.mole_fractions, strict=True)
        },
    }


def mixture_surface_tension(
    mixture,
    *,
    method,
    component_tensions_N_m,
    molar_densities_mol_m3=None,
    boiling_temperatures_K=None,
    critical_temperatures_K=None,
):
    """Existing liquid MixtureSpec -> PropertyEvaluation; explicit mixing rule."""
    inputs = _mixture_inputs(mixture, "liquid")
    extras = {
        "molar_densities_mol_m3": molar_densities_mol_m3,
        "boiling_temperatures_K": boiling_temperatures_K,
        "critical_temperatures_K": critical_temperatures_K,
    }
    required = {
        "Winterfeld_Scriven_Davis": {"molar_densities_mol_m3"},
        "Diguilio_Teja": {"boiling_temperatures_K", "critical_temperatures_K"},
    }
    if (
        method not in required
        or {k for k, v in extras.items() if v is not None} != required[method]
    ):
        raise ValueError("interface method requires exactly its declared input vectors")
    if method == "Winterfeld_Scriven_Davis":
        value = winterfeld_scriven_davis(
            mixture.mole_fractions, component_tensions_N_m, molar_densities_mol_m3
        )
        sigma_label = "sigma_at_T_N_m"
    else:
        value = diguilio_teja(
            mixture.temperature_K,
            mixture.mole_fractions,
            component_tensions_N_m,
            boiling_temperatures_K,
            critical_temperatures_K,
        )
        sigma_label = "sigma_at_Tb_N_m"
    for key, values in {sigma_label: component_tensions_N_m, **extras}.items():
        if values is not None:
            inputs.update(
                {
                    f"{key}:{identity}": v
                    for identity, v in zip(mixture.component_ids, values, strict=True)
                }
            )
    return PropertyEvaluation("surface_tension", method, method, value, "N/m", inputs)


def parachor_surface_tension(
    liquid, vapor, *, parachors_SI, liquid_molar_volume_m3_mol, vapor_molar_volume_m3_mol
):
    """Use two actual, aligned phase compositions; never infer a vapor phase."""
    inputs = _mixture_inputs(liquid, "liquid")
    gas_inputs = _mixture_inputs(vapor, "gas")
    if (
        liquid.component_ids != vapor.component_ids
        or liquid.molecular_weights_g_mol != vapor.molecular_weights_g_mol
        or liquid.temperature_K != vapor.temperature_K
        or liquid.pressure_Pa != vapor.pressure_Pa
    ):
        raise ValueError("parachor phases require aligned identities and the same T/P")
    inputs.update({f"vapor:{k}": v for k, v in gas_inputs.items()})
    inputs.update(Vml_m3_mol=liquid_molar_volume_m3_mol, Vmg_m3_mol=vapor_molar_volume_m3_mol)
    inputs.update(
        {f"parachor_SI:{key}": p for key, p in zip(liquid.component_ids, parachors_SI, strict=True)}
    )
    value = weinaug_katz(
        parachors_SI,
        liquid_molar_volume_m3_mol,
        vapor_molar_volume_m3_mol,
        liquid.mole_fractions,
        vapor.mole_fractions,
    )
    return PropertyEvaluation(
        "surface_tension", "Weinaug_Katz", "Weinaug_Katz", value, "N/m", inputs
    )


def _tension(evaluation):
    if evaluation.property_id not in {"surface_tension", "interfacial_tension"}:
        raise ValueError("capillary response requires an interface-property evaluation")
    return _positive(evaluation.to("N/m").value, "surface tension", zero=True)


def laplace_pressure(evaluation, *, curvature_1_per_m, curvature_2_per_m):
    """Young-Laplace signed pressure jump; sphere curvatures are both 1/r."""
    if any(isinstance(v, bool) or not isfinite(v) for v in (curvature_1_per_m, curvature_2_per_m)):
        raise ValueError("principal curvatures must be finite")
    value = _tension(evaluation) * (curvature_1_per_m + curvature_2_per_m)
    return PropertyEvaluation(
        "laplace_pressure",
        evaluation.correlation_id,
        "Young_Laplace",
        value,
        "Pa",
        {
            **evaluation.inputs,
            "curvature_1_per_m": curvature_1_per_m,
            "curvature_2_per_m": curvature_2_per_m,
        },
        evaluation.warnings,
    )


def capillary_force(evaluation, *, wetted_perimeter_m, contact_angle_rad):
    """Signed force along a wetted perimeter, sigma*perimeter*cos(theta)."""
    _positive(wetted_perimeter_m, "wetted perimeter", zero=True)
    if (
        isinstance(contact_angle_rad, bool)
        or not isfinite(contact_angle_rad)
        or not 0 <= contact_angle_rad <= pi
    ):
        raise ValueError("contact angle must be between zero and pi radians")
    value = _tension(evaluation) * wetted_perimeter_m * cos(contact_angle_rad)
    return PropertyEvaluation(
        "capillary_force",
        evaluation.correlation_id,
        "Wilhelmy",
        value,
        "N",
        {
            **evaluation.inputs,
            "wetted_perimeter_m": wetted_perimeter_m,
            "contact_angle_rad": contact_angle_rad,
        },
        evaluation.warnings,
    )
