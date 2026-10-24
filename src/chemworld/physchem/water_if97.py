"""IAPWS-IF97 basic equations in SI, with one energy and analytic derivatives.

Gibbs regions use (P, T); region 3 uses (rho, T). These homogeneous
evaluators are not a two-phase flash or an ice model. The shared region-4
curve is the only saturation-pressure authority. Reference: IAPWS R7-97(2012).
"""

from dataclasses import asdict, dataclass
from math import isclose, isfinite, sqrt

from chemworld.physchem.caloric_math import logarithm
from chemworld.physchem.second_order_pair import SecondOrderPair
from chemworld.physchem.water_if97_coefficients import (
    REGION1,
    REGION2,
    REGION2_IDEAL,
    REGION2_METASTABLE,
    REGION2_METASTABLE_IDEAL,
    REGION3,
    REGION3_LOG_COEFFICIENT,
    REGION5,
    REGION5_IDEAL,
)
from chemworld.physchem.water_saturation import (
    water_saturation_pressure,
    water_saturation_temperature,
)

WATER_GAS_CONSTANT_J_KG_K = 461.526
WATER_CRITICAL_TEMPERATURE_K = 647.096
WATER_CRITICAL_DENSITY_KG_M3 = 322.0


@dataclass(frozen=True)
class IF97State:
    region: int
    temperature_K: float
    pressure_Pa: float
    mass_density_kg_m3: float
    enthalpy_J_kg: float
    internal_energy_J_kg: float
    entropy_J_kg_K: float
    cp_J_kg_K: float
    cv_J_kg_K: float
    speed_of_sound_m_s: float
    gibbs_energy_J_kg: float
    isothermal_compressibility_Pa_inv: float
    thermal_expansion_K_inv: float
    stability_scope: str = "homogeneous_local_only"

    @property
    def specific_volume_m3_kg(self):
        return 1 / self.mass_density_kg_m3

    def to_dict(self):
        return {**asdict(self), "specific_volume_m3_kg": self.specific_volume_m3_kg}


def _positive(value, name):
    if isinstance(value, bool) or not isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite, positive and non-boolean")


def water_if97_b23_pressure(temperature_K):
    """Region 2/3 boundary, Pa (Eq. 5); accepts the common derivative seeds."""
    from chemworld.physchem.caloric_math import scalar_value

    value = scalar_value(temperature_K)
    _positive(value, "temperature_K")
    if not 623.15 <= value <= 863.15:
        raise ValueError("IF97 B23 requires 623.15 <= T <= 863.15 K")
    return 1e6 * (
        .34805185628969e3 - .11671859879975e1 * temperature_K
        + .10192970039326e-2 * temperature_K**2
    )


def water_if97_region(temperature_K, pressure_Pa):
    """Select the IF97 region; 4 means saturation needs a phase or quality."""
    T, p = temperature_K, pressure_Pa
    _positive(T, "temperature_K")
    _positive(p, "pressure_Pa")
    if not 273.15 <= T <= 2273.15 or p > (100e6 if T <= 1073.15 else 50e6):
        raise ValueError("state is outside the IF97 fluid T/P range")
    if T <= WATER_CRITICAL_TEMPERATURE_K:
        saturation = water_saturation_pressure(T)
        if isclose(p, saturation, rel_tol=2e-10, abs_tol=0):
            return 4
    if T <= 623.15:
        return 1 if p > saturation else 2
    if T <= 863.15:
        return 3 if p > water_if97_b23_pressure(T) else 2
    return 2 if T <= 1073.15 else 5


def _polynomial(x, y, terms):
    return sum(n * x**i * y**j for i, j, n in terms)


def _gibbs(T, p, region, metastable=False):
    """Specific Gibbs energy; all responses differentiate this expression."""
    if region == 1:
        gamma = _polynomial(7.1 - p / 16.53e6, 1386 / T - 1.222, REGION1)
    else:
        tau = (540 if region == 2 else 1000) / T
        pi = p / 1e6
        ideal, residual = (
            (REGION2_METASTABLE_IDEAL, REGION2_METASTABLE) if metastable else
            (REGION2_IDEAL, REGION2) if region == 2 else (REGION5_IDEAL, REGION5)
        )
        gamma = (
            logarithm(pi) + sum(n * tau**j for j, n in ideal)
            + _polynomial(pi, tau - .5 if region == 2 else tau, residual)
        )
    return WATER_GAS_CONSTANT_J_KG_K * T * gamma


def water_if97_liquid_gibbs(temperature_K, pressure_Pa):
    """Region-1 specific G, J/kg, preserving common analytic derivative seeds."""
    from chemworld.physchem.caloric_math import scalar_value

    if water_if97_region(scalar_value(temperature_K), scalar_value(pressure_Pa)) != 1:
        raise ValueError("aqueous solvent requires stable IF97 region-1 liquid water")
    return _gibbs(temperature_K, pressure_Pa, 1)


def water_if97_liquid_density(temperature_K, pressure_Pa):
    """Density from the pressure derivative of the same region-1 Gibbs equation."""
    water_if97_liquid_gibbs(temperature_K, pressure_Pa)
    x, y = 7.1 - pressure_Pa / 16.53e6, 1386 / temperature_K - 1.222
    volume = -WATER_GAS_CONSTANT_J_KG_K * temperature_K / 16.53e6 * sum(
        n * i * x**(i - 1) * y**j for i, j, n in REGION1 if i
    )
    return 1 / volume


def _gibbs_state(T, p, region, *, metastable=False):
    g = _gibbs(SecondOrderPair(T, dy=1), SecondOrderPair(p, dx=1), region, metastable)
    if not all(isfinite(v) for v in (g.value, g.dx, g.dy, g.dxx, g.dxy, g.dyy)):
        raise ValueError("IF97 response exceeds numerical range")
    v, s, cp = g.dx, -g.dy, -T * g.dyy
    if v <= 0 or g.dxx >= 0 or cp <= 0:
        raise ValueError("IF97 homogeneous state is locally unstable or singular")
    cv = cp + T * g.dxy**2 / g.dxx
    sound_squared = -v * v / (g.dxx - g.dxy**2 / g.dyy)
    if cv <= 0 or sound_squared <= 0:
        raise ValueError("IF97 heat capacity or acoustic response is unstable")
    h = g.value + T * s
    return IF97State(
        region, T, p, 1 / v, h, h - p * v, s, cp, cv, sqrt(sound_squared),
        g.value, -g.dxx / v, g.dxy / v,
        "explicit_metastable_vapor" if metastable else "homogeneous_local_only",
    )


def water_if97_gibbs_state(temperature_K, pressure_Pa, *, region=None, metastable=False):
    """Regions 1/2/5; an explicit region chooses a low-T saturation endpoint.

    Metastable steam is opt-in, limited to <=10 MPa and the published 5%
    equilibrium moisture enthalpy boundary. It never replaces a failed stable
    calculation. Region 3 requires the density evaluator or a separate root.
    """
    T, p = temperature_K, pressure_Pa
    selected = water_if97_region(T, p)
    if type(metastable) is not bool:
        raise ValueError("metastable must be a boolean")
    if region is not None and (type(region) is not int or region not in (1, 2, 5)):
        raise ValueError("Gibbs evaluation requires region 1, 2 or 5")
    if metastable:
        if region not in (None, 2) or not 611.657 <= p <= 10e6:
            raise ValueError("metastable vapor requires region 2 and 611.657 Pa <= P <= 10 MPa")
        Ts = water_saturation_temperature(p)
        if Ts < T:
            raise ValueError("metastable vapor must not exceed saturation temperature")
        result = _gibbs_state(T, p, 2, metastable=True)
        liquid, vapor = (_gibbs_state(Ts, p, k) for k in (1, 2))
        minimum_h = .05 * liquid.enthalpy_J_kg + .95 * vapor.enthalpy_J_kg
        if result.enthalpy_J_kg < minimum_h:
            raise ValueError("metastable vapor exceeds the IF97 5% moisture boundary")
        return result
    if selected == 4:
        if T > 623.15 or region not in (1, 2):
            raise ValueError("saturation requires an explicit phase; dense region needs rho")
    elif selected not in (1, 2, 5):
        raise ValueError("region 3 requires density evaluation or a pressure root")
    elif region is not None and region != selected:
        raise ValueError("requested Gibbs region is outside its declared range")
    return _gibbs_state(T, p, selected if region is None else region)


def _region3_energy(T, rho):
    density, temperature = SecondOrderPair(rho, dx=1), SecondOrderPair(T, dy=1)
    delta, tau = density / 322, WATER_CRITICAL_TEMPERATURE_K / temperature
    return WATER_GAS_CONSTANT_J_KG_K * temperature * (
        REGION3_LOG_COEFFICIENT * logarithm(delta) + _polynomial(delta, tau, REGION3)
    )


def water_if97_region3_state(temperature_K, mass_density_kg_m3):
    """Region-3 homogeneous T/rho equation, with local stability diagnostics.

    A subcritical locally stable density is not a global phase selection. No
    finite response is fabricated at the critical singularity or spinodal.
    """
    T, rho = temperature_K, mass_density_kg_m3
    _positive(T, "temperature_K")
    _positive(rho, "mass_density_kg_m3")
    if not 623.15 <= T <= 863.15:
        raise ValueError("IF97 region 3 requires 623.15 <= T <= 863.15 K")
    if abs(T - WATER_CRITICAL_TEMPERATURE_K) < 1e-10 and abs(rho - 322) < 1e-8:
        raise ValueError("IF97 critical response is singular")
    f = _region3_energy(T, rho)
    p = rho * rho * f.dx
    # Explicit relative tolerance for the polynomial's regional boundary.
    if not water_if97_b23_pressure(T) * (1 - 1e-10) <= p <= 100e6 * (1 + 1e-10):
        raise ValueError("density places the state outside IF97 region 3")
    dp_drho = 2 * rho * f.dx + rho * rho * f.dxx
    dp_dT = rho * rho * f.dxy
    cv = -T * f.dyy
    if dp_drho <= 0 or cv <= 0:
        raise ValueError("IF97 region-3 state is locally unstable or singular")
    cp = cv + T * dp_dT**2 / (rho * rho * dp_drho)
    s = -f.dy
    u = f.value + T * s
    h = u + p / rho
    return IF97State(
        3, T, p, rho, h, u, s, cp, cv,
        sqrt(dp_drho + T * dp_dT**2 / (rho * rho * cv)), h - T * s,
        1 / (rho * dp_drho), dp_dT / (rho * dp_drho),
    )
