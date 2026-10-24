"""G8 gas/condensate equilibrium from caller-owned total masses and volume.

There is no persistent phase inventory. Condensate volume and latent energy
are included in the same U(T, m_air, m_water, volume) and its analytic response.
The explicit condensed surface excludes liquid/ice melting transitions.
"""

from dataclasses import asdict, dataclass
from functools import lru_cache
from math import exp, isfinite, log

from chemworld.physchem.humid_air import (
    HumidAirState,
    _cross_energy,
    _pure_energy,
    _state_from_fractions,
    _validate,
)
from chemworld.physchem.humid_air_equilibrium import (
    CondensedWaterState,
    _gas_trial,
    _water_mass_fraction,
    condensed_water_state,
)
from chemworld.physchem.humid_air_parameters import (
    DRY_AIR_MOLAR_MASS_KG_MOL as MA,
)
from chemworld.physchem.humid_air_parameters import iapws95_water, iapws_dry_air
from chemworld.physchem.residual_system import (
    ResidualDomainError,
    ResidualVariable,
    RootPolicy,
    linear_solve,
    solve_root,
)
from chemworld.physchem.second_order_pair import SecondOrderPair
from chemworld.physchem.water_ice_boundaries import ice_sublimation_pressure
from chemworld.physchem.water_saturation import water_saturation_pressure


@dataclass(frozen=True)
class HumidAirVolumeState:
    gas: HumidAirState
    condensate: CondensedWaterState | None
    dry_air_mass_kg: float
    total_water_mass_kg: float
    total_volume_m3: float
    gas_mass_kg: float
    condensed_water_mass_kg: float
    gas_volume_m3: float
    condensed_volume_m3: float
    internal_energy_J: float
    heat_capacity_J_K: float
    dair_internal_energy_J_kg: float
    dwater_internal_energy_J_kg: float
    chemical_potential_residual_J_kg: float | None
    volume_residual_m3: float
    condensed_surface: str
    derivative_scope: str = "selected_phase_branch_one_sided_at_condensate_disappearance"

    def to_dict(self):
        return {**asdict(self), "gas": self.gas.to_dict(),
                "condensate": self.condensate.to_dict() if self.condensate else None}


def _positive(value, name, *, zero=False):
    if (isinstance(value, bool) or not isfinite(value)
            or (value < 0 if zero else value <= 0)):
        raise ValueError(f"{name} must be finite and {'nonnegative' if zero else 'positive'}")


def _condensed(T, p, phase):
    # Only the existing gas pressure root's roundoff band may reach this
    # strict pure-condensate API. This is not an extended pressure domain.
    if 5e6 < p <= 5e6*(1+2e-12):
        p = 5e6
    return condensed_water_state(T, p, phase=phase)


def _dry_energy_response(T, rho):
    """Finite U derivatives at zero water, without evaluating log(0)."""
    t = SecondOrderPair(T, dx=1)
    f = _pure_energy(iapws_dry_air(), t, SecondOrderPair(rho, dy=1))
    u, ur = f.value-T*f.dx, f.dy-T*f.dxy
    water = iapws95_water()
    # Ideal water u has no density dependence, even as rho_water -> 0.
    fw = water.gas_constant_J_mol_K/water.molar_mass_kg_mol*t*water.alpha(
        "ideal", 1., water.reducing_temperature_K/t)
    cross = _cross_energy(SecondOrderPair(1., dx=-1), SecondOrderPair(0., dx=1),
                          SecondOrderPair(T, dy=1), rho)
    uw = -u-rho*ur+fw.value-T*fw.dx+cross.dx-T*cross.dxy
    return u, -T*f.dxx, uw, ur


def _homogeneous(gas, a, b, volume, phase, chemical):
    T, rho, w = gas.temperature_K, gas.mass_density_kg_m3, gas.water_mass_fraction
    if b == 0:
        u, ut, uw, ur = _dry_energy_response(T, rho)
    else:
        d = _gas_trial(w, T, rho)
        u, ut, uw, ur = (d[k] for k in ("u", "u_T", "u_V", "u_rho"))
    mass = a+b
    return HumidAirVolumeState(
        gas, None, a, b, volume, mass, 0., volume, 0., mass*u, mass*ut,
        u-w*uw+rho*ur, u+(1-w)*uw+rho*ur, chemical, 0., phase)


def humid_air_volume(temperature_K, dry_air_mass_kg, total_water_mass_kg, volume_m3,
                     *, condensed_phase):
    """Equilibrium and U response at fixed total volume, in SI mass units.

    Dry mass must be positive. At zero water the finite one-sided energy
    derivative is returned; the absent water chemical potential stays None.
    Explicit ice/liquid may be metastable, as labelled by its pure-water model.
    """
    T, a, b, volume = temperature_K, dry_air_mass_kg, total_water_mass_kg, volume_m3
    _validate(.5, T)
    _positive(a, "dry-air mass")
    _positive(b, "total water mass", zero=True)
    _positive(volume, "total volume")
    if condensed_phase not in ("liquid", "ice"):
        raise ValueError("fixed-volume humid air requires an explicit liquid or ice surface")
    lower, upper = (270., 473.) if condensed_phase == "liquid" else (193., 273.16)
    if not lower <= T <= upper:
        raise ValueError("temperature leaves the declared condensed-surface domain")
    # Validate before caching: Python cache keys equate True with 1.
    return _volume(T, a, b, volume, condensed_phase)


@lru_cache(maxsize=128)
def _volume(T, a, b, volume, phase):
    bulk_w = b/(a+b)
    homogeneous_error = None
    try:
        gas = _state_from_fractions(1-bulk_w, bulk_w, T, (a+b)/volume)
    except (ValueError, OverflowError) as exc:
        homogeneous_error = str(exc)
        gas = None
    if gas is not None:
        if b == 0:
            return _homogeneous(gas, a, b, volume, phase, None)
        condensed = _condensed(T, gas.pressure_Pa, phase)
        chemical = gas.water_chemical_potential_J_kg-condensed.gibbs_energy_J_kg
        if chemical <= 1e-5:
            return _homogeneous(gas, a, b, volume, phase, chemical)
    if b == 0:
        raise ValueError(f"dry-air volume state is outside the gas domain: {homogeneous_error}")

    # Ideal partial pressure and pure saturation correlations are seeds only.
    vapor_guess = (ice_sublimation_pressure(T) if phase == "ice"
                   else water_saturation_pressure(max(T, 273.15)))
    p0 = min(5e6, a*8.31451/MA*T/volume+vapor_guess)
    w0 = min(.999*bulk_w, _water_mass_fraction(min(.95, vapor_guess/p0)))
    c0 = b-a*w0/(1-w0)
    # A seed does not need a second, fully converged saturation solve at an
    # artificial pressure. Nominal condensate density only initializes Vgas;
    # every accepted trial uses the actual EOS volume and chemical potential.
    vg0 = volume-c0/(917. if phase == "ice" else 1000.)
    if vg0 <= 0:
        raise ValueError("no positive gas volume in the humid-air equilibrium seed")
    rho0 = a/(1-w0)/vg0

    @lru_cache(maxsize=8)
    def trial(logw, logrho):
        w, rho = exp(logw), exp(logrho)
        A, mass = 1-w, a/(1-w)
        c = b-a*w/A
        try:
            d = _gas_trial(w, T, rho)
            cond = _condensed(T, d["p"], phase)
        except (ValueError, OverflowError) as exc:
            raise ResidualDomainError(str(exc)) from exc
        v = 1/cond.mass_density_kg_m3
        vp = -cond.isothermal_compressibility_Pa_inv*v
        vt = cond.thermal_expansion_K_inv*v
        jac = [[d["mu_V"]-v*d["p_V"], d["mu_rho"]-v*d["p_rho"]],
               [a/A**2*(1/rho-v)+c*vp*d["p_V"], -mass/rho**2+c*vp*d["p_rho"]]]
        residual = [d["mu"]-cond.gibbs_energy_J_kg, mass/rho+c*v-volume]
        return w, rho, d, cond, mass, c, v, vp, vt, jac, residual

    def jacobian(point):
        w, rho, *_, jac, _ = trial(*point)
        return [[row[0]*w, row[1]*rho] for row in jac]

    root = solve_root(lambda point: trial(*point)[-1], [log(w0), log(rho0)],
        jacobian=jacobian,
        variables=[ResidualVariable("log_gas_water_fraction", "1", 1., "algebraic",
                                    -100., log(bulk_w)),
                   ResidualVariable("log_gas_density", "1", 1., "algebraic", -100., log(500.))],
        residual_scales=[1e5, volume],
        policy=RootPolicy(tolerance=1e-12, maximum_iterations=80))
    if not root.success:
        raise ValueError(f"humid-air volume equilibrium failed: {root.failure_reason}; "
                         f"homogeneous trial: {homogeneous_error}")
    w, rho, d, cond, mass, c, v, vp, vt, jac, residual = trial(*root.value)
    if c < 0 or abs(residual[0]) > 1e-5 or abs(residual[1]) > 2e-11*volume:
        raise ValueError("humid-air volume equilibrium exceeds mass/chemical/volume acceptance")
    gas = _state_from_fractions(1-w, w, T, rho)
    p, A = gas.pressure_Pa, 1-w
    uc = cond.enthalpy_J_kg-p*v
    uct = cond.cp_J_kg_K-p*vt
    ucp = -T*vt-p*vp
    # Partials before eliminating gas composition and density constraints.
    uw = a/A**2*(d["u"]-uc)+mass*d["u_V"]+c*ucp*d["p_V"]
    ur = mass*d["u_rho"]+c*ucp*d["p_rho"]
    ut = mass*d["u_T"]+c*(uct+ucp*d["p_T"])

    def response(first, second, explicit):
        dw, dr = map(float, linear_solve(jac, [-first, -second]))
        return explicit+uw*dw+ur*dr

    capacity = response(d["mu_T"]+cond.entropy_J_kg_K-v*d["p_T"],
                        c*(vt+vp*d["p_T"]), ut)
    ua = response(0., 1/(A*rho)-w*v/A, d["u"]/A-w*uc/A)
    ub = response(0., v, uc)
    if not all(isfinite(x) for x in (capacity, ua, ub)) or capacity <= 0:
        raise ValueError("humid-air equilibrium energy response is not finite/positive")
    return HumidAirVolumeState(gas, cond, a, b, volume, mass, c, mass/rho, c*v,
                              mass*d["u"]+c*uc, capacity, ua, ub,
                              residual[0], residual[1], phase)
