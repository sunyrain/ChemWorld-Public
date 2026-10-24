"""Common residual Helmholtz responses and roots for molecular EOS models.

Residual energies use the ideal gas at the same T,rho,x as reference. Absolute
caloric properties require an explicit ideal-gas reference from the caller.
Homogeneous mechanical stability is distinct from global mixture stability.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from math import exp, isfinite, log
from typing import Any, Protocol

from chemworld.physchem.caloric_math import partial_derivative
from chemworld.physchem.residual_system import ResidualVariable, RootPolicy, solve_root
from chemworld.physchem.second_order_pair import SecondOrderPair


class ResidualHelmholtzModel(Protocol):
    @property
    def component_ids(self) -> tuple[str, ...]: ...

    @property
    def gas_constant_J_mol_K(self) -> float: ...  # noqa: N802 - public SI convention

    def alphar(self, temperature: Any, density: Any, fractions: Sequence[Any]) -> Any: ...


@dataclass(frozen=True)
class MolecularEOSState:
    temperature_K: float
    molar_density_mol_m3: float
    mole_fractions: tuple[float, ...]
    pressure_Pa: float
    compressibility_factor: float
    residual_helmholtz_J_mol: float
    residual_internal_energy_J_mol: float
    residual_enthalpy_J_mol: float
    residual_entropy_J_mol_K: float
    residual_cv_J_mol_K: float
    residual_cp_J_mol_K: float
    dP_dT_rho_Pa_K: float
    dP_drho_T_Pa_m3_mol: float
    mechanically_stable: bool
    log_fugacity_coefficients: tuple[float, ...] | None

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "stability_scope": "homogeneous_mechanical_only",
                "energy_reference": "ideal_gas_at_same_T_rho_composition"}


def molecular_residual_chemical_potentials(
    model: ResidualHelmholtzModel, temperature_K: float, molar_density_mol_m3: float,
    mole_fractions: Sequence[float],
) -> tuple[float, ...]:
    """Residual chemical potentials in J/mol, including absent-component limits.

    Differentiate extensive residual A at fixed T,total V. These responses
    remain defined for positive-density homogeneous states with negative P,
    where a pressure-referenced fugacity coefficient is undefined.
    """
    model.alphar(temperature_K, molar_density_mol_m3, mole_fractions)
    amounts = dict(zip(model.component_ids, mole_fractions, strict=True))

    def extensive(values: Mapping[str, Any]) -> Any:
        total = sum(values.values())
        return total*model.alphar(temperature_K, total*molar_density_mol_m3,
                                 tuple(values[k]/total for k in model.component_ids))

    rt = model.gas_constant_J_mol_K*temperature_K
    return tuple(rt*partial_derivative(extensive, amounts, (k,)) for k in model.component_ids)


def molecular_eos_state(
    model: ResidualHelmholtzModel, temperature_K: float, molar_density_mol_m3: float,
    mole_fractions: Sequence[float], *, require_stable: bool = True,
) -> MolecularEOSState:
    t, rho, xs, R = temperature_K, molar_density_mol_m3, tuple(mole_fractions), (
        model.gas_constant_J_mol_K)
    if type(require_stable) is not bool:
        raise ValueError("require_stable must be boolean")
    if isinstance(rho, bool) or not isfinite(rho) or rho <= 0:
        raise ValueError("state density must be positive; raw alpha supports the vacuum limit")
    ar = SecondOrderPair(0)+model.alphar(
        SecondOrderPair(t, dx=1), SecondOrderPair(rho, dy=1), xs)
    z = 1+rho*ar.dy
    stiffness = 1+2*rho*ar.dy+rho*rho*ar.dyy
    thermal = z+t*rho*ar.dxy
    pressure = rho*R*t*z
    stable = pressure > 0 and stiffness > 0
    if require_stable and not stable:
        raise ValueError("homogeneous molecular EOS state is mechanically unstable")
    if stiffness == 0:
        raise ValueError("singular molecular EOS pressure response")
    u = -R*t*t*ar.dx
    cv = -R*(2*t*ar.dx+t*t*ar.dxx)
    lnphi = None
    if z > 0:
        lnphi = tuple(mu/(R*t)-log(z) for mu in molecular_residual_chemical_potentials(
            model, t, rho, xs))
    result = MolecularEOSState(
        t, rho, xs, pressure, z, R*t*ar.value, u, u+R*t*(z-1),
        -R*(ar.value+t*ar.dx), cv, cv+R*(thermal*thermal/stiffness-1),
        rho*R*thermal, R*t*stiffness, stable, lnphi)
    if any(not isfinite(v) for v in asdict(result).values() if isinstance(v, float)):
        raise ValueError("nonfinite molecular EOS response")
    # Source domains such as a pressure ceiling are known only after the
    # common state is assembled. Apply the model's optional validator here so
    # caloric, density-root and property consumers cannot bypass that boundary.
    validator = getattr(model, "validate_state", None)
    if validator is not None:
        validator(result)
    return result


def molecular_scaled_derivative(
    model: ResidualHelmholtzModel, temperature_K: float, molar_density_mol_m3: float,
    mole_fractions: Sequence[float], *, inverse_temperature_order: int = 0,
    density_order: int = 0,
) -> float:
    """beta^i rho^j d^(i+j)alpha_r/d beta^i d rho^j, beta=1/T, total order <=3."""
    i, j = inverse_temperature_order, density_order
    if any(type(k) is not int or k < 0 for k in (i, j)) or i+j > 3:
        raise ValueError("molecular derivative orders must be integers totaling zero to three")
    if isinstance(temperature_K, bool) or not isfinite(temperature_K) or temperature_K <= 0:
        raise ValueError("molecular derivative temperature must be finite and positive")
    if not i+j:
        return float(model.alphar(temperature_K, molar_density_mol_m3, mole_fractions))
    beta = 1/temperature_K
    return beta**i*molar_density_mol_m3**j*partial_derivative(
        lambda v: model.alphar(1/v["beta"], v["rho"], mole_fractions),
        {"beta": beta, "rho": molar_density_mol_m3}, ("beta",)*i+("rho",)*j)


def molecular_density_from_pressure(
    model: ResidualHelmholtzModel, temperature_K: float, pressure_Pa: float,
    mole_fractions: Sequence[float], *, initial_density_mol_m3: float,
    density_bounds_mol_m3: tuple[float, float],
) -> MolecularEOSState:
    """A declared local homogeneous root, using the common scaled Newton solver.

    Bounds and starting density select the basin; this is not global phase
    selection. Unstable roots fail instead of being labelled equilibrium.
    """
    low, high = density_bounds_mol_m3
    if (any(isinstance(v, bool) or not isfinite(v) or v <= 0 for v in (
        pressure_Pa, low, high, initial_density_mol_m3
    )) or not low < initial_density_mol_m3 < high):
        raise ValueError("positive pressure and an interior initial density are required")

    def state(log_density: Sequence[float]) -> MolecularEOSState:
        return molecular_eos_state(model, temperature_K, exp(log_density[0]),
                                   mole_fractions, require_stable=False)

    root = solve_root(
        lambda v: [state(v).pressure_Pa-pressure_Pa], [log(initial_density_mol_m3)],
        jacobian=lambda v: [[state(v).dP_drho_T_Pa_m3_mol*exp(v[0])]],
        variables=(ResidualVariable("log_density", "dimensionless", role="algebraic",
                                    lower=log(low), upper=log(high)),),
        residual_scales=(pressure_Pa,), policy=RootPolicy(tolerance=2e-12),
    )
    if not root.success:
        raise ValueError("molecular pressure-density solve failed: "+root.failure_reason)
    return molecular_eos_state(model, temperature_K, exp(root.value[0]), mole_fractions)
