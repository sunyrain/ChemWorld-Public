"""Published conductivity mixing rules on explicit mass/mole composition.

Conductivities are W/(m K), molar volumes m³/mol, viscosities Pa s, molecular
weights g/mol, temperatures K. These kernels do not infer phases or pure
properties. The ledger adapter consumes the existing MixtureSpec unchanged.
"""

from math import isclose, sqrt

from chemworld.physchem.viscosity_kernels import _positive
from chemworld.physchem.viscosity_mixtures import herning_zipperer_viscosity


def _vectors(fractions, *vectors):
    if not len(fractions) or any(len(v) != len(fractions) for v in vectors):
        raise ValueError("mixture vectors must be nonempty and aligned")
    for x in fractions:
        _positive(x, "fraction", zero=True)
    if not isclose(sum(fractions), 1, rel_tol=0, abs_tol=1e-9):
        raise ValueError("fractions must sum to one; no implicit normalization")
    for vector in vectors:
        for value in vector:
            _positive(value, "component property")


def dippr9h_conductivity_terms(mass_fractions, conductivities_W_m_K):
    """Shared inverse-square terms for the scalar and original DIPPR9H ledger."""
    _vectors(mass_fractions, conductivities_W_m_K)
    return tuple(w / k**2 for w, k in zip(mass_fractions, conductivities_W_m_K, strict=True))


def dippr9h_conductivity(mass_fractions, conductivities_W_m_K):
    """Vredeveld/DIPPR9H mass-fraction mixing law for nonaqueous liquids."""
    return sum(dippr9h_conductivity_terms(mass_fractions, conductivities_W_m_K)) ** -0.5


def dippr9i_conductivity(mole_fractions, molar_volumes_m3_mol, conductivities_W_m_K):
    """Li/DIPPR9I volume fractions and pairwise harmonic conductivities."""
    _vectors(mole_fractions, molar_volumes_m3_mol, conductivities_W_m_K)
    occupied = [x * v for x, v in zip(mole_fractions, molar_volumes_m3_mol, strict=True)]
    total = sum(occupied)
    phi = [v / total for v in occupied]
    return sum(
        a * b * 2 / (1 / ka + 1 / kb)
        for a, ka in zip(phi, conductivities_W_m_K, strict=True)
        for b, kb in zip(phi, conductivities_W_m_K, strict=True)
    )


def filippov_conductivity(mass_fractions, conductivities_W_m_K):
    """Binary mass rule with explicit input order; no sorting of components.

    The published signed (k2-k1) correction is order-dependent. The caller
    must supply the convention intended for its material; swapping labels
    alone does not describe the same empirical model.
    """
    _vectors(mass_fractions, conductivities_W_m_K)
    if len(mass_fractions) != 2:
        raise ValueError("Filippov requires exactly two components in explicit order")
    w1, w2 = mass_fractions
    k1, k2 = conductivities_W_m_K
    return _positive(w1 * k1 + w2 * k2 - 0.72 * w1 * w2 * (k2 - k1), "conductivity")


def wassiljewa_herning_zipperer_conductivity(
    mole_fractions, conductivities_W_m_K, molecular_weights_g_mol
):
    """Same sqrt(MW)-weighted mean as the existing Herning-Zipperer kernel."""
    _vectors(mole_fractions, conductivities_W_m_K, molecular_weights_g_mol)
    return herning_zipperer_viscosity(mole_fractions, conductivities_W_m_K, molecular_weights_g_mol)


def lindsay_bromley_conductivity(
    T,
    mole_fractions,
    conductivities_W_m_K,
    viscosities_Pa_s,
    boiling_temperatures_K,
    molecular_weights_g_mol,
):
    """Lindsay-Bromley gas mixture, using 1.5 Tb Sutherland constants.

    Hydrogen-specific alternative Sutherland constants are not implied by
    this published reference API. All supplied pure properties belong to T.
    """
    _positive(T, "temperature")
    _vectors(
        mole_fractions,
        conductivities_W_m_K,
        viscosities_Pa_s,
        boiling_temperatures_K,
        molecular_weights_g_mol,
    )
    s = [1.5 * tb for tb in boiling_temperatures_K]
    value = 0
    for i, (x, k) in enumerate(zip(mole_fractions, conductivities_W_m_K, strict=True)):
        denominator = 0
        for j, y in enumerate(mole_fractions):
            ratio = (
                viscosities_Pa_s[i]
                / viscosities_Pa_s[j]
                * (molecular_weights_g_mol[j] / molecular_weights_g_mol[i]) ** 0.75
                * (T + s[i])
                / (T + s[j])
            )
            interaction = 0.25 * (1 + ratio**0.5) ** 2 * (T + sqrt(s[i] * s[j])) / (T + s[i])
            denominator += y * interaction
        value += x * k / denominator
    return _positive(value, "conductivity")


def mixture_conductivity_ledger(
    mixture,
    component_conductivities_W_m_K,
    *,
    method,
    molar_volumes_m3_mol=None,
    viscosities_Pa_s=None,
    boiling_temperatures_K=None,
):
    """Use the existing phase/composition and return MixtureTransportLedger.

    Contributions retain supplied component inputs, not an invented additive
    decomposition of nonlinear mixing. DIPPR9H preserves its original ledger.
    """
    from chemworld.physchem.transport_properties import (
        MixtureTransportLedger,
        liquid_mixture_thermal_conductivity_dippr9h_ledger,
    )

    phases = {
        "DIPPR9H": "liquid",
        "DIPPR9I": "liquid",
        "Filippov": "liquid",
        "Lindsay_Bromley": "gas",
        "Wassiljewa_Herning_Zipperer": "gas",
    }
    if method not in phases or mixture.phase_label != phases[method]:
        raise ValueError("conductivity mixing method must match the mixture phase")
    supplied = {
        "molar_volumes_m3_mol": molar_volumes_m3_mol,
        "viscosities_Pa_s": viscosities_Pa_s,
        "boiling_temperatures_K": boiling_temperatures_K,
    }
    expected = (
        {"molar_volumes_m3_mol"}
        if method == "DIPPR9I"
        else {"viscosities_Pa_s", "boiling_temperatures_K"}
        if method == "Lindsay_Bromley"
        else set()
    )
    if {k for k, v in supplied.items() if v is not None} != expected:
        raise ValueError(f"{method} requires only these extra vectors: {sorted(expected)}")
    ids, x, w, mw = (
        mixture.component_ids,
        mixture.mole_fractions,
        mixture.mass_fractions,
        mixture.molecular_weights_g_mol,
    )
    ks = component_conductivities_W_m_K
    _vectors(x, ks, mw)
    _vectors(w)
    mean_mw = sum(z * m for z, m in zip(x, mw, strict=True))
    if any(
        not isclose(a, z * m / mean_mw, rel_tol=1e-9, abs_tol=1e-12)
        for a, z, m in zip(w, x, mw, strict=True)
    ):
        raise ValueError("mixture mass and mole fractions must describe the same composition")
    if method == "DIPPR9H":
        return liquid_mixture_thermal_conductivity_dippr9h_ledger(
            component_mass_fractions=dict(zip(ids, w, strict=True)),
            component_thermal_conductivities_W_m_K=dict(zip(ids, ks, strict=True)),
        )
    if method == "DIPPR9I":
        value = dippr9i_conductivity(x, molar_volumes_m3_mol, ks)
    elif method == "Filippov":
        value = filippov_conductivity(w, ks)
    elif method == "Lindsay_Bromley":
        value = lindsay_bromley_conductivity(
            mixture.temperature_K, x, ks, viscosities_Pa_s, boiling_temperatures_K, mw
        )
    else:
        value = wassiljewa_herning_zipperer_conductivity(x, ks, mw)
    contributions = {
        identity: {
            "mole_fraction": x[i],
            "mass_fraction": w[i],
            "molecular_weight_g_mol": mw[i],
            "thermal_conductivity_W_m_K": ks[i],
            "temperature_K": mixture.temperature_K,
            **{key: values[i] for key, values in supplied.items() if values is not None},
        }
        for i, identity in enumerate(ids)
    }
    warning = (
        "Filippov uses the supplied binary order and signed k2-k1 correction."
        if method == "Filippov"
        else "Empirical mixing rule; component errors propagate."
    )
    return MixtureTransportLedger(
        method + "_conductivity",
        mixture.phase_label,
        "mixture_thermal_conductivity",
        method,
        "W/(m*K)",
        value,
        contributions,
        warnings=(warning,),
    )
