"""Ideal Donnan/NP/Schloegl pore with reciprocal ion-water coupling.

Point-ion, unit convective coefficient model. Effective diffusivities and
hydraulic permeability are explicit material declarations. No independent
water-drag coefficient or current-efficiency multiplier is used.
"""

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from math import isfinite
from typing import Any, cast

import numpy as np
from scipy.optimize import brentq

from chemworld.physchem.multiion_membrane_properties import MultiIonMembraneProperties
from chemworld.physchem.property_reports import R_J_PER_MOL_K as R
from chemworld.physchem.residual_system import (
    ResidualDerivatives,
    ResidualEquation,
    ResidualSystem,
    ResidualVariable,
    RootPolicy,
    solve_steady_residual,
)

F = 96485.33212331001
DEFAULT_PORE_POLICY = RootPolicy(tolerance=1e-11)


def _bernoulli(value: Any) -> tuple[Any, Any]:
    """B(x)=x/(exp(x)-1) and derivative without zero/large-positive loss."""
    x = np.asarray(value, dtype=float)
    small = abs(x) < 1e-4
    positive = x > 50
    b, derivative = np.empty_like(x), np.empty_like(x)
    u = x[small]
    b[small] = 1 - u / 2 + u * u / 12 - u**4 / 720 + u**6 / 30240
    derivative[small] = -0.5 + u / 6 - u**3 / 180 + u**5 / 5040
    u = x[positive]
    exponential = np.exp(-u)
    b[positive] = u * exponential / (1 - exponential)
    derivative[positive] = exponential * (1 - u - exponential) / (1 - exponential) ** 2
    regular = ~(small | positive)
    u = x[regular]
    denominator = np.expm1(u)
    b[regular] = u / denominator
    derivative[regular] = (denominator - u * np.exp(u)) / denominator**2
    return b, derivative


@dataclass(frozen=True)
class IonExchangePoreSpec:
    fixed_charge_mol_m3: float
    thickness_m: float
    hydraulic_permeability_m_Pa_s: float
    effective_diffusivities_m2_s: tuple[tuple[str, float], ...]
    source_id: str
    segments: int = 16
    maximum_concentration_mol_m3: float = 2000.0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "effective_diffusivities_m2_s",
            tuple(tuple(v) for v in self.effective_diffusivities_m2_s),
        )
        ids = [r[0] for r in self.effective_diffusivities_m2_s]
        if (
            not self.source_id
            or len(ids) < 2
            or len(set(ids)) != len(ids)
            or not isfinite(self.fixed_charge_mol_m3)
            or any(
                not isfinite(v) or v <= 0
                for v in (
                    self.thickness_m,
                    self.hydraulic_permeability_m_Pa_s,
                    self.maximum_concentration_mol_m3,
                    *(r[1] for r in self.effective_diffusivities_m2_s),
                )
            )
            or type(self.segments) is not int
            or not 1 <= self.segments <= 128
        ):
            raise ValueError("ion-exchange pore requires sourced finite geometry and diffusivities")

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "effective_diffusivities_m2_s": [list(r) for r in self.effective_diffusivities_m2_s],
        }


def ion_exchange_partition(
    spec: IonExchangePoreSpec, charges: Mapping[str, int], concentrations: Mapping[str, float]
) -> dict[str, Any]:
    ids = tuple(k for k, _ in spec.effective_diffusivities_m2_s)
    if (
        set(ids) != set(charges)
        or set(ids) != set(concentrations)
        or any(
            not isfinite(c) or not 0 < c <= spec.maximum_concentration_mol_m3
            for c in concentrations.values()
        )
        or not min(charges.values()) < 0 < max(charges.values())
    ):
        raise ValueError(
            "ion-exchange pore requires all signed ions at positive declared concentrations"
        )
    c, z = np.array([concentrations[k] for k in ids]), np.array([charges[k] for k in ids])
    phi = brentq(
        lambda p: float(z @ (c * np.exp(-z * p)) + spec.fixed_charge_mol_m3),
        -100 / max(abs(z)),
        100 / max(abs(z)),
        xtol=1e-14,
    )
    pore = c * np.exp(-z * phi)
    return {
        "dimensionless_potential": phi,
        "pore_concentrations_mol_m3": dict(zip(ids, map(float, pore), strict=True)),
    }


class PoreResidualBlock:
    """Pure reusable residual block; no physical state or warm-start persistence."""

    def __init__(
        self,
        spec: IonExchangePoreSpec,
        properties: MultiIonMembraneProperties,
        left: Mapping[str, float],
        right: Mapping[str, float],
        temperature_K: float,
        pressure_Pa: float,
        *,
        pressure_right_Pa: float | None = None,
    ) -> None:
        self.spec, self.properties, self.t = spec, properties, temperature_K
        self.ids = tuple(k for k, _ in spec.effective_diffusivities_m2_s)
        if (
            self.ids != properties.ion_ids
            or properties.database.activity_model.family != "ideal"
            or spec.source_id not in {s.source_id for s in properties.database.sources}
        ):
            raise ValueError("pore requires the same sourced original ideal-ion database")
        self.n, self.m = len(self.ids), spec.segments
        self.z = np.array([properties.charges[k] for k in self.ids])
        self.diffusion = np.array([v for _, v in spec.effective_diffusivities_m2_s])
        self.boundaries = [
            ion_exchange_partition(spec, properties.charges, c) for c in (left, right)
        ]
        self.end_c = np.array(
            [[b["pore_concentrations_mol_m3"][k] for k in self.ids] for b in self.boundaries]
        )
        self.phi_left = self.boundaries[0]["dimensionless_potential"]
        self.phi_right_partition = self.boundaries[1]["dimensionless_potential"]
        self.dx = spec.thickness_m / self.m
        self.cscale = max(float(np.max(self.end_c)), abs(spec.fixed_charge_mol_m3), 1)
        self.flux_scales = self.diffusion * self.cscale / spec.thickness_m
        self.vscale = max(float(np.max(self.diffusion)) / spec.thickness_m, 1e-8)
        self.nc = (self.m - 1) * self.n
        self.nphi, self.nflux = self.m, self.n - 1
        self.width = self.nc + self.nphi + self.nflux + 1
        self.pore_pressures = []
        self.water_mu = []
        right_pressure = pressure_Pa if pressure_right_Pa is None else pressure_right_Pa
        for c, end, pressure in zip(
            (left, right), self.end_c, (pressure_Pa, right_pressure), strict=True
        ):
            values = []
            for comp in (c, dict(zip(self.ids, end, strict=True))):
                report = properties.database.chemical_potentials(
                    properties.amounts(1, comp), temperature_K=temperature_K, pressure_Pa=pressure
                )
                values.append(cast(float, report.chemical_potentials_J_mol[properties.water_id]))
            self.water_mu.append(values[0])
            self.pore_pressures.append(
                pressure + (values[0] - values[1]) / properties.water_volume_m3_mol
            )
        self.pressure_difference = self.pore_pressures[0] - self.pore_pressures[1]
        self.bulk_concentrations = (dict(left), dict(right))
        self.variables = tuple(
            ResidualVariable(
                f"log_c_{i}",
                "1",
                role="algebraic",
                lower=-35,
                upper=np.log(spec.maximum_concentration_mol_m3 * 100),
            )
            for i in range(self.nc)
        )
        self.variables += tuple(
            ResidualVariable(f"phi_{i}", "1", role="algebraic", lower=-100, upper=100)
            for i in range(self.nphi)
        )
        self.variables += tuple(
            ResidualVariable(f"flux_{i}", "1", role="algebraic") for i in range(self.nflux)
        )
        self.variables += (ResidualVariable("water_velocity", "1", role="algebraic"),)

    def initial(self, current_density: float) -> Any:
        cmean = np.mean(self.end_c, axis=0)
        conductivity = F * float(np.sum(self.z * self.z * self.diffusion * cmean))
        dphi = -current_density * self.spec.thickness_m / conductivity
        phi = np.linspace(self.phi_left, self.phi_right_partition + dphi, self.m + 1)
        c = np.linspace(self.end_c[0], self.end_c[1], self.m + 1)
        v = self.spec.hydraulic_permeability_m_Pa_s * (
            self.pressure_difference
            + R * self.t * self.spec.fixed_charge_mol_m3 * (phi[-1] - phi[0])
        )
        flux = (
            -self.diffusion * (self.end_c[1] - self.end_c[0]) / self.spec.thickness_m
            - self.z * self.diffusion * cmean * (phi[-1] - phi[0]) / self.spec.thickness_m
            + cmean * v
        )
        return np.r_[
            np.log(c[1:-1]).ravel(), phi[1:], flux[:-1] / self.flux_scales[:-1], v / self.vscale
        ]

    def decode(self, y: Any, current_density: float) -> tuple[Any, Any, Any, float]:
        c = np.vstack(
            (self.end_c[0], np.exp(y[: self.nc].reshape(self.m - 1, self.n)), self.end_c[1])
        )
        phi = np.r_[self.phi_left, y[self.nc : self.nc + self.nphi]]
        independent = y[self.nc + self.nphi : -1] * self.flux_scales[:-1]
        # Exact coordinate elimination of the current equation, not post-solve repair.
        flux = np.r_[independent, (current_density / F - self.z[:-1] @ independent) / self.z[-1]]
        return c, phi, flux, float(y[-1]) * self.vscale

    def residual(self, y: Any, current_density: float) -> Any:
        c, phi, flux, v = self.decode(y, current_density)
        # Exact NP segment flux for constant electric field and water velocity.
        # Exponential fitting resolves minority-ion layers without negative c.
        peclet = v * self.dx / self.diffusion - self.z * np.diff(phi)[:, None]
        backward, _ = _bernoulli(-peclet)
        forward, _ = _bernoulli(peclet)
        np_res = flux - self.diffusion / self.dx * (
            backward * c[:-1] - forward * c[1:]
        )
        water = v - self.spec.hydraulic_permeability_m_Pa_s * (
            self.pressure_difference
            + R * self.t * self.spec.fixed_charge_mol_m3 * (phi[-1] - phi[0])
        )
        return np.r_[
            (np_res / self.flux_scales).ravel(),
            (c[1:-1] @ self.z + self.spec.fixed_charge_mol_m3) / self.cscale,
            water / self.vscale,
        ]

    def jacobian(self, y: Any, current_density: float) -> tuple[Any, Any]:
        c, phi, _, v = self.decode(y, current_density)
        jac = np.zeros((self.width, self.width))
        dj = np.zeros(self.width)
        peclet = v * self.dx / self.diffusion - self.z * np.diff(phi)[:, None]
        backward, backward_derivative = _bernoulli(-peclet)
        forward, forward_derivative = _bernoulli(peclet)
        for segment in range(self.m):
            for ion in range(self.n):
                row = segment * self.n + ion
                scale = self.flux_scales[ion]
                gradient = self.diffusion[ion] / self.dx / scale
                pe_derivative = gradient * (
                    backward_derivative[segment, ion] * c[segment, ion]
                    + forward_derivative[segment, ion] * c[segment + 1, ion]
                )
                for node, sign in ((segment, -1), (segment + 1, 1)):
                    if 0 < node < self.m:
                        jac[row, (node - 1) * self.n + ion] = (
                            gradient
                            * sign
                            * (backward if sign < 0 else forward)[segment, ion]
                            * c[node, ion]
                        )
                    if node:
                        jac[row, self.nc + node - 1] = -sign * self.z[ion] * pe_derivative
                if ion < self.n - 1:
                    jac[row, self.nc + self.nphi + ion] = 1
                else:
                    jac[row, self.nc + self.nphi : -1] = (
                        -self.z[:-1] / self.z[-1] * self.flux_scales[:-1] / scale
                    )
                    dj[row] = 1 / (F * self.z[-1] * scale)
                jac[row, -1] = pe_derivative * self.vscale * self.dx / self.diffusion[ion]
        offset = self.m * self.n
        for node in range(1, self.m):
            jac[offset + node - 1, (node - 1) * self.n : node * self.n] = (
                self.z * c[node] / self.cscale
            )
        jac[-1, -1] = 1
        jac[-1, self.nc + self.nphi - 1] = (
            -self.spec.hydraulic_permeability_m_Pa_s
            * R
            * self.t
            * self.spec.fixed_charge_mol_m3
            / self.vscale
        )
        return jac, dj

    def readout(self, y: Any, current_density: float) -> dict[str, Any]:
        c, phi, flux, v = self.decode(y, current_density)
        voltage = R * self.t / F * (self.phi_right_partition - phi[-1])
        chemical = (
            R
            * self.t
            * np.array(
                [
                    np.log(self.bulk_concentrations[0][k] / self.bulk_concentrations[1][k])
                    for k in self.ids
                ]
            )
        )
        # Boundary irreversible power includes chemical, water and electrical ports.
        dissipation = float(
            flux @ chemical
            + v / self.properties.water_volume_m3_mol * (self.water_mu[0] - self.water_mu[1])
            + current_density * voltage
        )
        return {
            "ion_fluxes_mol_m2_s": dict(zip(self.ids, map(float, flux), strict=True)),
            "water_flux_m_s": v,
            "voltage_drop_V": voltage,
            "current_density_A_m2": current_density,
            "pore_concentrations_mol_m3": [
                dict(zip(self.ids, map(float, row), strict=True)) for row in c
            ],
            "pore_dimensionless_potential": phi.tolist(),
            "pore_interface_pressures_Pa": self.pore_pressures,
            "charge_residual_mol_m3": float(
                np.max(abs(c @ self.z + self.spec.fixed_charge_mol_m3))
            ),
            "dissipation_W_m2": dissipation,
        }


def ion_exchange_pore_point(
    spec: IonExchangePoreSpec,
    properties: MultiIonMembraneProperties,
    left: Mapping[str, float],
    right: Mapping[str, float],
    *,
    temperature_K: float,
    pressure_Pa: float,
    current_density_A_m2: float,
    pressure_right_Pa: float | None = None,
    policy: RootPolicy = DEFAULT_PORE_POLICY,
) -> dict[str, Any]:
    block = PoreResidualBlock(
        spec,
        properties,
        left,
        right,
        temperature_K,
        pressure_Pa,
        pressure_right_Pa=pressure_right_Pa,
    )
    j = current_density_A_m2
    system = ResidualSystem(
        "ideal_donnan_np_schloegl_pore_v1",
        block.variables,
        tuple(ResidualEquation(f"pore_{i}", "1") for i in range(block.width)),
        lambda t, y, rate, p: block.residual(y, j),
        ResidualDerivatives(
            "analytic",
            state=lambda t, y, rate, p: block.jacobian(y, j)[0],
            rate=lambda t, y, rate, p: np.zeros((block.width, block.width)),
        ),
    )
    solved = solve_steady_residual(system, block.initial(j), policy=policy)
    return {
        "status": "converged" if solved.root.success else "not_converged",
        "solver": {
            **solved.root.diagnostics,
            **solved.diagnostics,
            "failure_reason": solved.root.failure_reason,
        },
        **(block.readout(solved.root.value, j) if solved.root.success else {}),
    }
