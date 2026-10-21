"""Pure gas mixture viscosity kernels; shared by original transport ledgers.

Wilke (1950), doi:10.1063/1.1747673. The streamed and precomputed forms
evaluate the same interaction equation; no second mixture state is stored.
"""

from math import isclose, isfinite, sqrt

from chemworld.physchem.viscosity_kernels import _positive, viscosity_value


def _vectors(fractions, viscosities, weights=None):
    n = len(fractions)
    if not n or len(viscosities) != n or (weights is not None and len(weights) != n):
        raise ValueError("mixture vectors must be nonempty and aligned")
    for values, nonnegative in (
        (fractions, True),
        (viscosities, False),
        (() if weights is None else weights, False),
    ):
        if any(
            isinstance(v, bool)
            or not isinstance(v, int | float)
            or not isfinite(v)
            or (v < 0 if nonnegative else v <= 0)
            for v in values
        ):
            raise ValueError("fractions must be finite nonnegative; viscosity/MW finite positive")
    if not isclose(sum(fractions), 1, abs_tol=1e-10, rel_tol=0):
        raise ValueError("mole fractions must sum to one; no implicit normalization")


def _wilke_factors(mwi, mwj):
    base = 1 / sqrt(8 * (1 + mwi / mwj))
    quarter = (mwj / mwi) ** 0.25
    return base * quarter**2, base * 2 * quarter, base


def wilke_prefactors(molecular_weights_g_mol):
    """Three N by N matrices independent of viscosity (dimensionless)."""
    weights = molecular_weights_g_mol
    _vectors([1 / len(weights)] * len(weights) if weights else (), weights, weights)
    return tuple(
        tuple(tuple(_wilke_factors(a, b)[k] for b in weights) for a in weights) for k in range(3)
    )


def wilke_denominators(fractions, viscosities_Pa_s, molecular_weights_g_mol=None, *, factors=None):
    """One O(N²) interaction traversal, O(N) memory unless factors supplied."""
    _vectors(fractions, viscosities_Pa_s, molecular_weights_g_mol)
    n = len(fractions)
    if (factors is None) == (molecular_weights_g_mol is None):
        raise ValueError("supply exactly one of molecular weights and Wilke prefactors")
    if factors is not None and (
        len(factors) != 3
        or any(len(matrix) != n for matrix in factors)
        or any(len(row) != n for matrix in factors for row in matrix)
        or any(
            not isinstance(v, int | float) or isinstance(v, bool) or not isfinite(v) or v <= 0
            for matrix in factors
            for row in matrix
            for v in row
        )
    ):
        raise ValueError("Wilke prefactors must be three positive finite N by N matrices")
    denominators = []
    for i in range(n):
        denominator = 0
        for j in range(n):
            a, b, c = (
                tuple(matrix[i][j] for matrix in factors)
                if factors is not None
                else _wilke_factors(molecular_weights_g_mol[i], molecular_weights_g_mol[j])
            )
            ratio = viscosities_Pa_s[i] / viscosities_Pa_s[j]
            denominator += fractions[j] * (a * ratio + b * sqrt(ratio) + c)
        denominators.append(denominator)
    return tuple(denominators)


def wilke_viscosity(fractions, viscosities_Pa_s, molecular_weights_g_mol=None, *, factors=None):
    """Scalar/large/precomputed Wilke forms share the same denominator kernel."""
    denominators = wilke_denominators(
        fractions, viscosities_Pa_s, molecular_weights_g_mol, factors=factors
    )
    return sum(
        x * mu / d for x, mu, d in zip(fractions, viscosities_Pa_s, denominators, strict=True)
    )


def herning_zipperer_viscosity(fractions, viscosities_Pa_s, molecular_weights_g_mol):
    """Mole fractions weighted by sqrt(MW); zero fractions are retained."""
    _vectors(fractions, viscosities_Pa_s, molecular_weights_g_mol)
    weights = [x * sqrt(mw) for x, mw in zip(fractions, molecular_weights_g_mol, strict=True)]
    return sum(w * mu for w, mu in zip(weights, viscosities_Pa_s, strict=True)) / sum(weights)


def brokaw_viscosity(
    T, fractions, viscosities_Pa_s, molecular_weights_g_mol, *, polar_parameters, epsilon_k_K
):
    """Brokaw/DIPPR8D gas viscosity, Pa s; delta is dimensionless polarity.

    The reference API calls delta 'molecular_diameters', but it is NOT LJ
    sigma in angstrom. NASA TN D-4496 eq. 11 defines the polarity parameter.
    Pairs with both delta <= 0.1 use the nonpolar approximation Sij=1;
    all other pairs use the polar collision correction. No state is stored.
    """
    _positive(T, "temperature")
    _vectors(fractions, viscosities_Pa_s, molecular_weights_g_mol)
    if len(polar_parameters) != len(fractions) or len(epsilon_k_K) != len(fractions):
        raise ValueError("Brokaw polarity and energy vectors must match components")
    for value in polar_parameters:
        _positive(value, "Brokaw dimensionless polarity", zero=True)
    for value in epsilon_k_K:
        _positive(value, "epsilon/k")
    reduced = [T / value for value in epsilon_k_K]
    result = 0
    for i, (x, mu, mw) in enumerate(
        zip(fractions, viscosities_Pa_s, molecular_weights_g_mol, strict=True)
    ):
        denominator = 0
        for j, (y, muj, mwj) in enumerate(
            zip(fractions, viscosities_Pa_s, molecular_weights_g_mol, strict=True)
        ):
            ratio = mw / mwj
            m = (4 / ((1 + 1 / ratio) * (1 + ratio))) ** 0.25
            a = (
                m
                * ratio**-0.5
                * (
                    1
                    + (ratio - ratio**0.45)
                    / (2 * (1 + ratio) + (1 + ratio**0.45) * m**-0.5 / (1 + m))
                )
            )
            d1, d2 = polar_parameters[i], polar_parameters[j]
            s = (
                1
                if d1 <= 0.1 and d2 <= 0.1
                else (
                    (1 + sqrt(reduced[i] * reduced[j]) + d1 * d2 / 4)
                    / sqrt((1 + reduced[i] + d1 * d1 / 4) * (1 + reduced[j] + d2 * d2 / 4))
                )
            )
            denominator += y * sqrt(mu / muj) * s * a
        result += x * mu / denominator
    return _positive(result, "Brokaw viscosity")


def lorentz_bray_clarke_viscosity(
    T,
    molar_volume_m3_mol,
    fractions,
    *,
    molecular_weights_g_mol,
    critical_temperatures_K,
    critical_pressures_Pa,
    critical_volumes_m3_mol,
):
    """Common continuous LBC form, Pa s, from actual molar volume at T/P.

    Pressure acts through the caller's molar volume. There is no unused P
    argument or private density model. Pseudo-critical T/P/V and MW use
    mole-fraction means; pure Stiel-Thodos values use Herning-Zipperer.
    """
    _positive(T, "temperature")
    _positive(molar_volume_m3_mol, "actual molar volume")
    mw, tc, pc, vc = (
        molecular_weights_g_mol,
        critical_temperatures_K,
        critical_pressures_Pa,
        critical_volumes_m3_mol,
    )
    for values in (mw, tc, pc, vc):
        _vectors(fractions, values)

    def mean(values):
        return sum(x * v for x, v in zip(fractions, values, strict=True))

    pure = [
        viscosity_value("Stiel_Thodos", T, MW=m, Tc=t, Pc=p)
        for m, t, p in zip(mw, tc, pc, strict=True)
    ]
    low = herning_zipperer_viscosity(fractions, pure, mw)
    xi = mean(tc) ** (1 / 6) / (mean(mw) ** 0.5 * (mean(pc) / 101325) ** (2 / 3))
    density_reduced = mean(vc) / molar_volume_m3_mol
    poly = sum(
        a * density_reduced**i
        for i, a in enumerate((0.1023, 0.023364, 0.058533, -0.040758, 0.0093724))
    )
    return _positive(low + 0.001 * (poly**4 - 0.0001) / xi, "LBC viscosity")


def gas_mixture_viscosity_ledger(
    mixture, component_viscosities_Pa_s, *, method, polar_parameters=None, epsilon_k_K=None
):
    """Consume the original MixtureSpec; method selection never falls back.

    Contributions record the supplied per-component inputs, not an invented
    additive decomposition of a nonlinear mixing law. Wilke delegates its
    existing ledger, which also exposes its exact partial contributions.
    """
    from chemworld.physchem.transport_properties import (
        MixtureTransportLedger,
        wilke_gas_mixture_viscosity_ledger,
    )

    if mixture.phase_label != "gas":
        raise ValueError("gas mixture viscosity requires gas phase")
    ids, x, mw = mixture.component_ids, mixture.mole_fractions, mixture.molecular_weights_g_mol
    _vectors(x, component_viscosities_Pa_s, mw)
    if method != "Brokaw" and (polar_parameters is not None or epsilon_k_K is not None):
        raise ValueError("polarity/energy parameters apply only to Brokaw")
    if method == "Wilke":
        return wilke_gas_mixture_viscosity_ledger(
            component_mole_fractions=dict(zip(ids, x, strict=True)),
            component_viscosities_Pa_s=dict(zip(ids, component_viscosities_Pa_s, strict=True)),
            component_molecular_weights_g_mol=dict(zip(ids, mw, strict=True)),
        )
    if method == "Herning_Zipperer":
        value = herning_zipperer_viscosity(x, component_viscosities_Pa_s, mw)
    elif method == "Brokaw":
        if polar_parameters is None or epsilon_k_K is None:
            raise ValueError("Brokaw requires explicit polarity and energy parameters")
        value = brokaw_viscosity(
            mixture.temperature_K,
            x,
            component_viscosities_Pa_s,
            mw,
            polar_parameters=polar_parameters,
            epsilon_k_K=epsilon_k_K,
        )
    else:
        raise ValueError(f"unknown gas mixture viscosity method {method!r}")
    contributions = {
        k: {
            "mole_fraction": x[i],
            "viscosity_Pa_s": component_viscosities_Pa_s[i],
            "molecular_weight_g_mol": mw[i],
            "temperature_K": mixture.temperature_K,
        }
        for i, k in enumerate(ids)
    }
    if method == "Brokaw":
        for i, k in enumerate(ids):
            contributions[k].update(polar_parameter=polar_parameters[i], epsilon_k_K=epsilon_k_K[i])
    return MixtureTransportLedger(
        method + "_gas_viscosity",
        "gas",
        "mixture_viscosity",
        method,
        "Pa*s",
        value,
        contributions,
        warnings=("Dilute gas mixing rule; no pressure correction is implied.",),
    )


def dense_mixture_viscosity_ledger(
    mixture,
    molar_volume_m3_mol,
    *,
    critical_temperatures_K,
    critical_pressures_Pa,
    critical_volumes_m3_mol,
):
    """LBC result on the original ledger, using the volume at the mixture T/P.

    Components' critical inputs remain parameters. Operating Vm is provided
    by the caller's state closure; this method neither solves nor stores EOS.
    """
    from chemworld.physchem.transport_properties import MixtureTransportLedger

    if mixture.phase_label not in {"gas", "liquid"}:
        raise ValueError("LBC requires a single gas or liquid phase")
    value = lorentz_bray_clarke_viscosity(
        mixture.temperature_K,
        molar_volume_m3_mol,
        mixture.mole_fractions,
        molecular_weights_g_mol=mixture.molecular_weights_g_mol,
        critical_temperatures_K=critical_temperatures_K,
        critical_pressures_Pa=critical_pressures_Pa,
        critical_volumes_m3_mol=critical_volumes_m3_mol,
    )
    contributions = {
        k: {
            "mole_fraction": mixture.mole_fractions[i],
            "molecular_weight_g_mol": mixture.molecular_weights_g_mol[i],
            "critical_temperature_K": critical_temperatures_K[i],
            "critical_pressure_Pa": critical_pressures_Pa[i],
            "critical_volume_m3_mol": critical_volumes_m3_mol[i],
            "temperature_K": mixture.temperature_K,
            "pressure_Pa": mixture.pressure_Pa,
            "molar_volume_m3_mol": molar_volume_m3_mol,
        }
        for i, k in enumerate(mixture.component_ids)
    }
    return MixtureTransportLedger(
        "LBC_mixture_viscosity",
        mixture.phase_label,
        "mixture_viscosity",
        "Lorentz-Bray-Clarke",
        "Pa*s",
        value,
        contributions,
        warnings=("Pressure acts through the supplied volume at the mixture state.",),
    )
