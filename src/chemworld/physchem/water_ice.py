"""Homogeneous ice Ih from IAPWS R10-06(2009), Eq. 1 and Tables 2--4.

Copyright 2009 IAPWS; published coefficients reproduced with attribution.
https://iapws.org/technical-guidance/release/Ice-2009
The bounding rectangle is not a global phase classifier. Other ice phases,
liquid and vapor may be more stable. No transport or shear moduli are implied.
"""

from cmath import log as complex_log
from dataclasses import asdict, dataclass
from math import isfinite

from chemworld.physchem.second_order_pair import SecondOrderPair

ICE_TRIPLE_TEMPERATURE_K = 273.16
ICE_TRIPLE_PRESSURE_PA = 611.657
_NORMAL_PRESSURE_PA = 101325.0
_G0 = (-.632020233335886e6, .655022213658955, -.189369929326131e-7,
       .339746123271053e-14, -.556464869058991e-21)
_T1 = complex(.368017112855051e-1, .510878114959572e-1)
_T2 = complex(.337315741065416, .335449415919309)
_R1 = complex(.447050716285388e2, .656876847463481e2)
_R2 = (complex(-.725974574329220e2, -.781008427112870e2),
       complex(-.557107698030123e-4, .464578634580806e-4),
       complex(.234801409215913e-10, -.285651142904972e-10))
_ENTROPY_ZERO = {"iapws95": -.332733756492168e4, "absolute": .18913e3}


def _temperature_function(tau, coefficient):
    """The bracket in Eq. 1 and its first two tau derivatives.

    For |tau/t| < 1/8 the even-power series has <1e-27 truncation error
    relative to its first term (through power 36). This avoids subtracting
    its quadratic part from two logarithms at low T. It is the same function.
    """
    z = tau / coefficient
    z2 = z * z
    if abs(z) < .125:
        value = sum(2 * z**(2*k) / ((2*k-1)*(2*k)) for k in range(2, 19)) * coefficient
        first = sum(2 * z**(2*k-1) / (2*k-1) for k in range(2, 19))
    else:
        minus, plus = complex_log(coefficient - tau), complex_log(coefficient + tau)
        value = ((coefficient - tau) * minus + (coefficient + tau) * plus
                 - 2 * coefficient * complex_log(coefficient) - tau*tau/coefficient)
        first = plus - minus - 2*z
    second = 2 * z2 / (coefficient * (1-z2))
    return value, first, second


def ice_ih_gibbs_derivatives(temperature_K, pressure_Pa, *, entropy_reference="iapws95"):
    """Specific g and analytic Hessian in the original (x=p, y=T) container.

    Units: g J/kg; dx m³/kg; dy J/(kg K); dxx m³/(kg Pa);
    dxy m³/(kg K); dyy J/(kg K²). Supports the exact zero-K limit.
    """
    T, p = temperature_K, pressure_Pa
    if isinstance(T, bool) or not isfinite(T) or not 0 <= T <= ICE_TRIPLE_TEMPERATURE_K:
        raise ValueError("ice Ih requires 0 <= T <= 273.16 K")
    if isinstance(p, bool) or not isfinite(p) or not 0 <= p <= 210e6:
        raise ValueError("ice Ih requires 0 <= p <= 210 MPa")
    if entropy_reference not in _ENTROPY_ZERO:
        raise ValueError("entropy_reference must be iapws95 or absolute")
    pt, tt = ICE_TRIPLE_PRESSURE_PA, ICE_TRIPLE_TEMPERATURE_K
    reduced_pressure = (p - _NORMAL_PRESSURE_PA) / pt
    g0 = sum(n * reduced_pressure**k for k, n in enumerate(_G0))
    g0_p = sum(k * _G0[k] * reduced_pressure**(k-1) for k in range(1, 5)) / pt
    g0_pp = sum(k*(k-1)*_G0[k]*reduced_pressure**(k-2) for k in range(2, 5)) / pt**2
    r2 = sum(n * reduced_pressure**k for k, n in enumerate(_R2))
    r2_p = (_R2[1] + 2 * _R2[2] * reduced_pressure) / pt
    r2_pp = 2 * _R2[2] / pt**2
    f1, d1, dd1 = _temperature_function(T/tt, _T1)
    f2, d2, dd2 = _temperature_function(T/tt, _T2)
    s0 = _ENTROPY_ZERO[entropy_reference]
    return SecondOrderPair(
        g0 - s0*T + tt*(_R1*f1 + r2*f2).real,
        dx=g0_p + tt*(r2_p*f2).real,
        dy=-s0 + (_R1*d1 + r2*d2).real,
        dxx=g0_pp + tt*(r2_pp*f2).real,
        dxy=(r2_p*d2).real,
        dyy=(_R1*dd1 + r2*dd2).real/tt,
    )


@dataclass(frozen=True)
class IceIhState:
    temperature_K: float
    pressure_Pa: float
    mass_density_kg_m3: float
    gibbs_energy_J_kg: float
    helmholtz_energy_J_kg: float
    enthalpy_J_kg: float
    internal_energy_J_kg: float
    entropy_J_kg_K: float
    cp_J_kg_K: float
    cv_J_kg_K: float
    thermal_expansion_K_inv: float
    pressure_coefficient_Pa_K: float
    isothermal_compressibility_Pa_inv: float
    isentropic_compressibility_Pa_inv: float
    entropy_reference: str
    stability_scope: str = "homogeneous_ice_Ih_only"

    @property
    def specific_volume_m3_kg(self):
        return 1 / self.mass_density_kg_m3

    def to_dict(self):
        return {**asdict(self), "specific_volume_m3_kg": self.specific_volume_m3_kg}


def ice_ih_state(temperature_K, pressure_Pa, *, entropy_reference="iapws95"):
    """Mass-SI thermodynamics of the specified homogeneous ice-Ih branch.

    The default energy/entropy reference is the official IAPWS-95 choice.
    Absolute entropy changes g and f, never h/u or caloric/volume responses.
    At zero K, Cp=Cv=0 and kappa_s=kappa_T by the Debye limit.
    """
    T, p = temperature_K, pressure_Pa
    g = ice_ih_gibbs_derivatives(T, p, entropy_reference=entropy_reference)
    cp = -T*g.dyy
    cv = cp + T*g.dxy*g.dxy/g.dxx
    kappa = -g.dxx/g.dx
    kappa_s = kappa if g.dyy == 0 else kappa + g.dxy*g.dxy/(g.dx*g.dyy)
    if g.dx <= 0 or cp < 0 or cv < 0 or kappa <= 0 or kappa_s <= 0:
        raise ValueError("ice Ih homogeneous response is not locally stable")
    return IceIhState(
        T, p, 1/g.dx, g.value, g.value-p*g.dx, g.value-T*g.dy,
        g.value-T*g.dy-p*g.dx, -g.dy, cp, cv, g.dxy/g.dx, -g.dxy/g.dxx,
        kappa, kappa_s, entropy_reference,
    )
