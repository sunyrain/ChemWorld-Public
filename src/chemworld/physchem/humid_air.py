"""IAPWS G8-10 humid air: one mass-specific Helmholtz potential.

https://iapws.org/technical-guidance/release/SeaAir, Eqs. 1--11.
The homogeneous evaluator alone does not impose saturation or change material
inventory. Equilibrium with condensed water must use its chemical potential.
"""

from dataclasses import asdict, dataclass
from math import isclose, isfinite, sqrt

from chemworld.physchem.caloric_math import exponential, scalar_value
from chemworld.physchem.humid_air_parameters import (
    DRY_AIR_MOLAR_MASS_KG_MOL as MA,
)
from chemworld.physchem.humid_air_parameters import (
    WATER_MOLAR_MASS_KG_MOL as MW,
)
from chemworld.physchem.humid_air_parameters import iapws95_water, iapws_dry_air
from chemworld.physchem.residual_system import (
    ResidualDomainError,
    ResidualVariable,
    RootPolicy,
    solve_root,
)
from chemworld.physchem.second_order_pair import SecondOrderPair

HUMID_AIR_CROSS_GAS_CONSTANT_J_MOL_K = 8.314472
_PRESSURE_ROOT_TOLERANCE = 2e-12


def _validate(A, T, rho=None):
    a, t = scalar_value(A), scalar_value(T)
    if isinstance(a, bool) or not isfinite(a) or not 0 <= a <= 1:
        raise ValueError("dry_air_mass_fraction must be finite and in [0,1]")
    if isinstance(t, bool) or not isfinite(t) or not 193 <= t <= 473:
        raise ValueError("IAPWS G8 humid air requires 193 <= T <= 473 K")
    if rho is not None:
        r = scalar_value(rho)
        if isinstance(r, bool) or not isfinite(r) or r <= 0:
            raise ValueError("humid-air mass density must be finite and positive")


def humid_air_cross_virials(temperature_K):
    """BAW [m³/mol], CAAW and CAWW [m⁶/mol²], allowing native AD seeds."""
    _validate(.5, temperature_K)
    theta = temperature_K/100
    baw = 1e-6*sum(n*theta**t for n, t in (
        (.665687e2, -.237), (-.238834e3, -1.048), (-.176755e3, -3.183)))
    caaw = 1e-6*sum(n*theta**-i for i, n in enumerate(
        (.482737e-3, .105678e-2, -.656394e-2, .294442e-1, -.319317e-1)))
    caww = -1e-6*exponential(sum(n*theta**-i for i, n in enumerate(
        (-.10728876e2, .34780200e2, -.38338300e2, .33406000e2))))
    return baw, caaw, caww


def humid_air_cross_energy(dry_air_mass_fraction, temperature_K, mass_density_kg_m3):
    """Air/water cross contribution f_mix [J/kg], Eq. 8, on shared AD."""
    A, T, rho = dry_air_mass_fraction, temperature_K, mass_density_kg_m3
    _validate(A, T, rho)
    return _cross_energy(A, 1-A, T, rho)


def _cross_energy(A, V, T, rho):
    baw, caaw, caww = humid_air_cross_virials(T)
    return (2*A*V*rho*HUMID_AIR_CROSS_GAS_CONSTANT_J_MOL_K*T/(MA*MW)
            * (baw + .75*rho*(A/MA*caaw + V/MW*caww)))


def _pure_energy(fluid, T, rho):
    return (fluid.gas_constant_J_mol_K/fluid.molar_mass_kg_mol*T
            * fluid.alpha("total", rho/(fluid.molar_mass_kg_mol
                                        * fluid.reducing_molar_density_mol_m3),
                          fluid.reducing_temperature_K/T))


def humid_air_helmholtz(dry_air_mass_fraction, temperature_K, mass_density_kg_m3):
    """Specific f(A,T,rho), J/kg humid air. Pure limits are evaluated explicitly.

    Derivatives with respect to composition require 0<A<1: the chemical
    potential of an absent ideal-mixing species is not finite. T/rho
    derivatives at fixed pure composition remain well defined.
    """
    A, T, rho = dry_air_mass_fraction, temperature_K, mass_density_kg_m3
    _validate(A, T, rho)
    return _mixture_energy(A, 1-A, T, rho)


def _mixture_energy(A, V, T, rho):
    """One potential, preserving a caller's small component without subtraction.

    Internal callers supply complementary fractions and opposite AD seeds.
    The public A API still uses exactly V=1-A. Equilibrium instead seeds V,
    avoiding loss of trace water when A rounds to one.
    """
    if scalar_value(A) == 0 or scalar_value(V) == 0:
        if not isinstance(A, (float, int)) or not isinstance(V, (float, int)):
            raise ValueError("composition derivatives are singular at a pure endpoint")
        return _pure_energy(iapws_dry_air() if V == 0 else iapws95_water(), T, rho)
    return (A*_pure_energy(iapws_dry_air(), T, A*rho)
            + V*_pure_energy(iapws95_water(), T, V*rho)
            + _cross_energy(A, V, T, rho))


def humid_air_derivatives(dry_air_mass_fraction, temperature_K, mass_density_kg_m3,
                          *, contribution="total"):
    """All ten mass-basis value/gradient/Hessian entries, in A,T,rho coordinates."""
    A, T, rho = dry_air_mass_fraction, temperature_K, mass_density_kg_m3
    _validate(A, T, rho)
    if contribution not in ("total", "cross"):
        raise ValueError("contribution must be total or cross")
    function = humid_air_helmholtz if contribution == "total" else humid_air_cross_energy
    if contribution == "total" and not 0 < A < 1:
        raise ValueError("composition derivatives are singular at a pure endpoint")
    rt = function(A, SecondOrderPair(T, dy=1), SecondOrderPair(rho, dx=1))
    ar = function(SecondOrderPair(A, dx=1), T, SecondOrderPair(rho, dy=1))
    at = function(SecondOrderPair(A, dx=1), SecondOrderPair(T, dy=1), rho)
    return {"f": rt.value, "f_A": ar.dx, "f_T": rt.dy, "f_rho": rt.dx,
            "f_AA": ar.dxx, "f_AT": at.dxy, "f_Arho": ar.dxy,
            "f_TT": rt.dyy, "f_Trho": rt.dxy, "f_rhorho": rt.dxx}


@dataclass(frozen=True)
class HumidAirState:
    dry_air_mass_fraction: float
    water_mass_fraction: float
    temperature_K: float
    mass_density_kg_m3: float
    pressure_Pa: float
    helmholtz_energy_J_kg: float
    internal_energy_J_kg: float
    enthalpy_J_kg: float
    gibbs_energy_J_kg: float
    entropy_J_kg_K: float
    cv_J_kg_K: float
    cp_J_kg_K: float
    speed_of_sound_m_s: float
    isothermal_compressibility_Pa_inv: float
    thermal_expansion_K_inv: float
    dP_dT_rho_Pa_K: float
    dP_drho_T_Pa_m3_kg: float
    water_chemical_potential_J_kg: float | None
    air_chemical_potential_J_kg: float | None
    composition_stiffness_J_kg: float | None
    stability_scope: str = "homogeneous_local_only_saturation_not_enforced"

    @property
    def humidity_ratio_kg_water_kg_dry_air(self):
        a = self.dry_air_mass_fraction
        return self.water_mass_fraction/a if a else None

    @property
    def water_mole_fraction(self):
        a = self.dry_air_mass_fraction
        v = self.water_mass_fraction
        return (v/MW)/(a/MA+v/MW)

    @property
    def enthalpy_J_kg_dry_air(self):  # noqa: N802 - shared SI quantity spelling
        a = self.dry_air_mass_fraction
        return self.enthalpy_J_kg/a if a else None

    def to_dict(self):
        return {**asdict(self), "water_mole_fraction": self.water_mole_fraction,
                "humidity_ratio_kg_water_kg_dry_air": self.humidity_ratio_kg_water_kg_dry_air,
                "enthalpy_J_kg_dry_air": self.enthalpy_J_kg_dry_air}


def _thermal_derivatives(A, T, rho):
    return humid_air_helmholtz(A, SecondOrderPair(T, dy=1), SecondOrderPair(rho, dx=1))


def humid_air_state(dry_air_mass_fraction, temperature_K, mass_density_kg_m3):
    """Locally stable homogeneous gas properties; no global saturation claim."""
    A, T, rho = dry_air_mass_fraction, temperature_K, mass_density_kg_m3
    _validate(A, T, rho)
    return _state_from_fractions(A, 1-A, T, rho)


def _state_from_fractions(A, V, T, rho):
    f = _mixture_energy(A, V, SecondOrderPair(T, dy=1), SecondOrderPair(rho, dx=1))
    pressure = rho*rho*f.dx
    dpdt, dpdr = rho*rho*f.dxy, 2*rho*f.dx+rho*rho*f.dxx
    cv = -T*f.dyy
    # A density obtained from the shared 2e-12-relative pressure root can
    # recompute just above 5 MPa. Admit only that numerical residual band;
    # explicitly supplied TP pressures still have a strict upper bound.
    if (not isfinite(pressure) or pressure <= 0
            or (pressure > 5e6 and not isclose(pressure, 5e6,
                                               rel_tol=_PRESSURE_ROOT_TOLERANCE, abs_tol=0))):
        raise ValueError("humid-air pressure is outside 0 < p <= 5 MPa")
    if cv <= 0 or dpdr <= 0:
        raise ValueError("humid-air state is locally unstable")
    correction = T*dpdt*dpdt/(rho*rho)
    cp = cv+correction/dpdr
    gibbs = f.value+pressure/rho
    mu_water, mu_air, composition_stiffness = None, None, None
    if A > 0 and V > 0:
        ar = _mixture_energy(SecondOrderPair(A, dx=1), SecondOrderPair(V, dx=-1),
                             T, SecondOrderPair(rho, dy=1))
        mu_water, mu_air = gibbs-A*ar.dx, gibbs+V*ar.dx
        composition_stiffness = ar.dxx-rho*rho*ar.dxy*ar.dxy/dpdr
        if composition_stiffness <= 0:
            raise ValueError("humid-air composition is locally unstable")
    elif V == 0:
        mu_air = gibbs
    else:
        mu_water = gibbs
    state = HumidAirState(
        A, V, T, rho, pressure, f.value, f.value-T*f.dy, gibbs-T*f.dy, gibbs, -f.dy,
        cv, cp, sqrt(dpdr+correction/cv), 1/(rho*dpdr), dpdt/(rho*dpdr), dpdt, dpdr,
        mu_water, mu_air, composition_stiffness,
    )
    if any(not isfinite(v) for v in asdict(state).values() if isinstance(v, (int, float))):
        raise ValueError("nonfinite humid-air response")
    return state


def humid_air_tp(dry_air_mass_fraction, temperature_K, pressure_Pa):
    """Homogeneous gas pressure root with shared solver and local stability.

    Condensation is not inferred by this function. The ideal-density seed
    selects the connected gas branch; the solver rejects unstable trial states.
    """
    A, T, p = dry_air_mass_fraction, temperature_K, pressure_Pa
    _validate(A, T)
    return _tp_from_fractions(A, 1-A, T, p)


def humid_air_from_water_fraction(water_mass_fraction, temperature_K, pressure_Pa):
    """Same homogeneous TP kernel with a trace-water preserving coordinate."""
    V, T, p = water_mass_fraction, temperature_K, pressure_Pa
    _validate(V, T)
    return _tp_from_fractions(1-V, V, T, p)


def _tp_from_fractions(A, V, T, p):
    if isinstance(p, bool) or not isfinite(p) or not 0 < p <= 5e6:
        raise ValueError("humid-air pressure is outside 0 < p <= 5 MPa")
    ideal_density = p/((A*8.31451/MA+V*461.51805)*T)

    def trial(point):
        rho = float(point[0])
        f = _mixture_energy(A, V, SecondOrderPair(T, dy=1), SecondOrderPair(rho, dx=1))
        pressure = rho*rho*f.dx
        slope = 2*rho*f.dx+rho*rho*f.dxx
        if not isfinite(pressure) or pressure <= 0 or slope <= 0 or f.dyy >= 0:
            raise ResidualDomainError("humid-air gas root trial is locally unstable")
        return pressure, slope

    result = solve_root(
        lambda point: [trial(point)[0]-p], [ideal_density],
        jacobian=lambda point: [[trial(point)[1]]],
        variables=[ResidualVariable("rho", "kg/m3", ideal_density, "algebraic", 1e-30)],
        residual_scales=[p],
        policy=RootPolicy(tolerance=_PRESSURE_ROOT_TOLERANCE, maximum_iterations=80),
    )
    if not result.success:
        raise ValueError(f"humid-air gas pressure root failed: {result.failure_reason}")
    state = _state_from_fractions(A, V, T, float(result.value[0]))
    if abs(state.pressure_Pa-p) > 5e-11*p:
        raise ValueError("humid-air gas pressure residual exceeds acceptance")
    return state
