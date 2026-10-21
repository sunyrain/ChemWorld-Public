"""Dielectric and refractive properties from published equations and data.

Water equations: IAPWS R8-97 and R9-97, including the original R8 constants.
Temperature is ITS-90 K; density kg/m3; vacuum wavelength m. These functions
do not infer density, pressure, phase stability, optical frequency or a fit.
Analytic partial derivatives use the existing nested FirstDerivative scalar.
"""

from math import isfinite, pi

from scipy.interpolate import interp1d

from chemworld.physchem.brix_data import BRIX_REFRACTIVE_INDICES
from chemworld.physchem.caloric_math import scalar_value
from chemworld.physchem.heat_capacity_kernels import _polynomial
from chemworld.physchem.property_reports import PropertyEvaluation
from chemworld.physchem.viscosity_kernels import _positive

VACUUM_PERMITTIVITY_1997 = 1 / (4e-7 * pi * 299792458**2)
AVOGADRO_SI = 6.02214076e23
_WATER_G_TERMS = (
    (0.978224486826, 1, 0.25),
    (-0.957771379375, 1, 1),
    (0.237511794148, 1, 2.5),
    (0.714692244396, 2, 1.5),
    (-0.298217036956, 3, 1.5),
    (-0.108863472196, 3, 2.5),
    (0.0949327488264, 4, 2),
    (-0.00980469816509, 5, 2),
    (0.000016516763497, 6, 5),
    (0.0000937359795772, 7, 0.5),
    (-0.00000000012317921872, 10, 10),
)


def _finite(value, label):
    v = scalar_value(value)
    if isinstance(v, bool) or not isinstance(v, int | float) or not isfinite(v):
        raise ValueError(f"{label} must be finite")
    return value


def relative_permittivity_crc(T, A, B=0, C=0, D=0):
    """CRC polynomial, dimensionless; the caller supplies its fitted domain."""
    _positive(T, "temperature")
    for p in (A, B, C, D):
        _finite(p, "permittivity coefficient")
    return _positive(_polynomial(T, (A, B, C, D)), "relative permittivity")


def water_permittivity_iapws(T, rho, *, constants="iapws1997"):
    """R8-97 at supplied mass density, retaining its 1997 physical constants.

    Enforces the mathematical real domain, not the empirical T/P or stable
    fluid region: 238..873 K with the pressure/metastability conditions in
    R8-97 section 4. Without an EOS, T/rho alone cannot certify that region.
    ``chemicals_1_5_2`` explicitly uses that version's published scientific
    precombined A/rho*g/T and B/rho constants. It does not reproduce the
    original release's verification table to its printed precision.
    """
    _positive(T, "temperature")
    _positive(rho, "mass density", zero=True)
    if constants not in {"iapws1997", "chemicals_1_5_2"}:
        raise ValueError("select an explicit water dielectric constant convention")
    if scalar_value(T) <= 228:
        raise ValueError("water permittivity requires T > 228 K")
    delta, tau = rho / 322, 647.096 / T
    g = 1 + sum(n * delta**i * tau**j for n, i, j in _WATER_G_TERMS)
    g += 0.00196096504426 * delta * (T / 228 - 1) ** -1.2
    _positive(g, "water correlation g factor")
    molar_density = rho / 0.018015268
    A = (
        6.0221367e23
        * 6.138e-30**2
        * molar_density
        * g
        / (VACUUM_PERMITTIVITY_1997 * 1.380658e-23 * T)
    )
    B = 6.0221367e23 * 1.636e-40 * molar_density / (3 * VACUUM_PERMITTIVITY_1997)
    if constants == "chemicals_1_5_2":
        A = 10.302249930663786 * rho * g / T
        B = 0.00020588442304500529 * rho
    if scalar_value(B) >= 1:
        raise ValueError("water dielectric root requires B < 1")
    discriminant = 9 + 2 * A + 18 * B + A * A + 10 * A * B + 9 * B * B
    return _positive((1 + A + 5 * B + discriminant**0.5) / (4 * (1 - B)), "relative permittivity")


def water_refractive_index_iapws(T, rho, wavelength_m=589.3e-9, *, extrapolate=False):
    """R9-97 refractive index relative to vacuum; explicit extrapolation only."""
    _positive(T, "temperature")
    _positive(rho, "mass density", zero=True)
    _positive(wavelength_m, "wavelength")
    if type(extrapolate) is not bool:
        raise ValueError("extrapolate must be boolean")
    if not extrapolate and not (
        261.15 <= scalar_value(T) <= 773.15
        and scalar_value(rho) <= 1060
        and 0.2e-6 <= scalar_value(wavelength_m) <= 1.1e-6
    ):
        raise ValueError("outside IAPWS refractive-index T/rho/wavelength range")
    delta, theta, w = rho / 1000, T / 273.15, wavelength_m / 0.589e-6
    if scalar_value(w * w) in (0.229202**2, 5.432937**2):
        raise ValueError("water refractive-index resonance pole")
    A = delta * (
        0.244257733
        + 0.00974634476 * delta
        - 0.00373234996 * theta
        + 0.000268678472 * w * w * theta
        + 0.00158920570 / (w * w)
        + 0.00245934259 / (w * w - 0.229202**2)
        + 0.900704920 / (w * w - 5.432937**2)
        - 0.0166626219 * delta * delta
    )
    if not -0.5 < scalar_value(A) < 1:
        raise ValueError("water refractive index has no positive finite real root")
    return _positive(((1 + 2 * A) / (1 - A)) ** 0.5, "refractive index")


def refractive_index_tde(T, Bs, Cs, wavelength_m=589.26e-9):
    """Two TDE polynomials; Bs/Cs in descending power order, vacuum lambda m."""
    _positive(T, "temperature")
    _positive(wavelength_m, "wavelength")
    if not len(Bs) or not len(Cs):
        raise ValueError("TDE needs both explicit coefficient sequences")
    for v in (*Bs, *Cs):
        _finite(v, "TDE coefficient")
    return _positive(
        _polynomial(T - 298.15, tuple(reversed(Bs)))
        + _polynomial(wavelength_m * 1e9 - 589.26, tuple(reversed(Cs))),
        "refractive index",
    )


def molar_refractivity(refractive_index, molar_volume_m3_mol):
    """Lorentz-Lorenz molar refractivity, m3/mol; positive real index.

    n<1 yields a signed effective response; no clipping to a molecular model.
    """
    _positive(refractive_index, "refractive index")
    _positive(molar_volume_m3_mol, "molar volume")
    return (refractive_index**2 - 1) / (refractive_index**2 + 2) * molar_volume_m3_mol


def refractive_index_from_molar_refractivity(refractivity_m3_mol, molar_volume_m3_mol):
    _finite(refractivity_m3_mol, "molar refractivity")
    _positive(molar_volume_m3_mol, "molar volume")
    if (
        not -0.5 * scalar_value(molar_volume_m3_mol)
        < scalar_value(refractivity_m3_mol)
        < (scalar_value(molar_volume_m3_mol))
    ):
        raise ValueError("molar refractivity requires -Vm/2 < Rm < Vm for a positive real index")
    return (
        (molar_volume_m3_mol + 2 * refractivity_m3_mol)
        / (molar_volume_m3_mol - refractivity_m3_mol)
    ) ** 0.5


def polarizability_volume(refractive_index, molar_volume_m3_mol):
    """Molecular polarizability volume in m3, not SI electric polarizability."""
    return 3 * molar_refractivity(refractive_index, molar_volume_m3_mol) / (4 * pi * AVOGADRO_SI)


_BRIX_TO_RI = interp1d(tuple(range(96)), BRIX_REFRACTIVE_INDICES, fill_value="extrapolate")
_RI_TO_BRIX = interp1d(BRIX_REFRACTIVE_INDICES, tuple(range(96)), fill_value="extrapolate")


def _brix_convert(value, inverse, extrapolate):
    _finite(value, "Brix/RI value")
    if type(extrapolate) is not bool:
        raise ValueError("extrapolate must be boolean")
    low, high = (BRIX_REFRACTIVE_INDICES[0], BRIX_REFRACTIVE_INDICES[-1]) if inverse else (0, 95)
    if not extrapolate and not low <= value <= high:
        raise ValueError("outside the Brix 0..95 reference table")
    result = float((_RI_TO_BRIX if inverse else _BRIX_TO_RI)(value))
    return (
        _finite(result, "Brix/RI conversion") if inverse else _positive(result, "refractive index")
    )


def brix_to_refractive_index(brix, *, extrapolate=False):
    """Reference table including 85..95 extension; not arbitrary-solution mass %."""
    return _brix_convert(brix, False, extrapolate)


def refractive_index_to_brix(refractive_index, *, extrapolate=False):
    _positive(refractive_index, "refractive index")
    return _brix_convert(refractive_index, True, extrapolate)


def brix_reading(index_evaluation, *, extrapolate=False):
    """Convert an existing RI observation; retain all conditions and warnings."""
    if index_evaluation.property_id != "refractive_index":
        raise ValueError("Brix conversion needs a refractive-index report")
    value = refractive_index_to_brix(
        index_evaluation.to("dimensionless").value, extrapolate=extrapolate
    )
    warnings = list(index_evaluation.warnings)
    warnings.append("Sucrose reference scale only; no temperature or composition correction")
    if not 0 <= value <= 85:
        warnings.append(
            "Outside official 0..85 Brix scale; extended table or explicit extrapolation"
        )
    return PropertyEvaluation(
        "brix",
        index_evaluation.correlation_id,
        "ICUMSA_1974_linear",
        value,
        "degBx",
        dict(index_evaluation.inputs),
        tuple(warnings),
    )


def parallel_plate_capacitance(permittivity, *, area_m2, separation_m):
    """Uniform dielectric plate geometry; F. No fringing or frequency model."""
    if permittivity.property_id != "relative_permittivity":
        raise ValueError("capacitance requires a relative-permittivity report")
    epsilon = _positive(permittivity.to("dimensionless").value, "relative permittivity")
    _positive(area_m2, "plate area")
    _positive(separation_m, "plate separation")
    value = _positive(VACUUM_PERMITTIVITY_1997 * epsilon * area_m2 / separation_m, "capacitance")
    return PropertyEvaluation(
        "capacitance",
        permittivity.correlation_id,
        "parallel_plates",
        value,
        "F",
        {**permittivity.inputs, "area_m2": area_m2, "separation_m": separation_m},
        permittivity.warnings,
    )
