"""Finite binary substitutional Gibbs equilibrium and caloric closure.

The common supporting line is checked against every stationary point of each
declared phase. A grid only initializes the tangent; it is not the returned
phase composition. Parameters are a finite native model, never a TDB backend.
"""

from dataclasses import asdict, dataclass
from itertools import pairwise
from math import isfinite, log

import numpy as np
from scipy.optimize import brentq, least_squares
from scipy.special import xlogy

from chemworld.physchem.reaction_network import R_J_PER_MOL_K as R


@dataclass(frozen=True)
class SolidPhaseSpec:
    name: str = "alpha"
    gA_J_mol: float = 0.0
    gB_J_mol: float = 0.0
    L0_J_mol: float = 10000.0
    L1_J_mol: float = 0.0

    def __post_init__(self) -> None:
        if not self.name or any(
            not isfinite(v) or abs(v) > 20000
            for v in (self.gA_J_mol, self.gB_J_mol, self.L0_J_mol, self.L1_J_mol)
        ):
            raise ValueError("finite solid phases require names and bounded synthetic energies")

    def reference_enthalpy(self, x):
        return (
            x * self.gA_J_mol
            + (1 - x) * self.gB_J_mol
            + x * (1 - x) * (self.L0_J_mol + self.L1_J_mol * (2 * x - 1))
        )

    def gibbs(self, x, temperature):
        return self.reference_enthalpy(x) + R * temperature * (xlogy(x, x) + xlogy(1 - x, 1 - x))

    def slope(self, x, temperature):
        return (
            self.gA_J_mol
            - self.gB_J_mol
            + R * temperature * log(x / (1 - x))
            + self.L0_J_mol * (1 - 2 * x)
            + self.L1_J_mol * (-6 * x * x + 6 * x - 1)
        )

    def stationary_points(self, slope, temperature):
        # g'' multiplied by x(1-x) is cubic. Its real roots bracket every
        # monotone piece of g', so no metastable minimum can be missed by a grid.
        coefficients = (
            12 * self.L1_J_mol,
            2 * self.L0_J_mol - 18 * self.L1_J_mol,
            -2 * self.L0_J_mol + 6 * self.L1_J_mol,
            R * temperature,
        )
        roots = np.roots(np.trim_zeros(coefficients, "f"))
        cuts = [
            1e-14,
            *(float(r.real) for r in roots if abs(r.imag) < 1e-10 and 1e-14 < r.real < 1 - 1e-14),
            1 - 1e-14,
        ]
        cuts.sort()
        candidates = [0.0, 1.0]

        def f(x):
            return self.slope(x, temperature) - slope

        for lo, hi in pairwise(cuts):
            if f(lo) * f(hi) <= 0:
                candidates.append(brentq(f, lo, hi, xtol=1e-14))
        return candidates


@dataclass(frozen=True)
class SolidComposition:
    phase_name: str
    x_A: float
    fraction: float


@dataclass(frozen=True)
class SolidEquilibrium:
    temperature_K: float
    amounts_mol: dict[str, float]
    phases: tuple[SolidComposition, ...]
    gibbs_J: float
    enthalpy_J: float
    tangent_residual_J_mol: float
    support_deficit_J_mol: float


@dataclass(frozen=True)
class SolidSolutionSpec:
    phases: tuple[SolidPhaseSpec, ...] = (SolidPhaseSpec(),)
    reference_temperature_K: float = 300.0
    molar_heat_capacity_J_mol_K: float = 30.0
    support_heat_capacity_J_K: float = 0.2
    minimum_temperature_K: float = 290.0
    maximum_temperature_K: float = 850.0
    molar_volume_L_mol: float = 0.01
    pressure_Pa: float = 101325.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "phases", tuple(self.phases))
        if not 1 <= len(self.phases) <= 2 or len({p.name for p in self.phases}) != len(self.phases):
            raise ValueError("one or two uniquely named binary solid phases are supported")
        if any(
            not isfinite(v) or v <= 0
            for v in (
                self.reference_temperature_K,
                self.molar_heat_capacity_J_mol_K,
                self.minimum_temperature_K,
                self.maximum_temperature_K,
                self.molar_volume_L_mol,
                self.pressure_Pa,
            )
        ):
            raise ValueError("solid caloric and geometric parameters must be finite and positive")
        if (
            not isfinite(self.support_heat_capacity_J_K)
            or self.support_heat_capacity_J_K < 0
            or not 290 <= self.minimum_temperature_K < self.maximum_temperature_K <= 850
        ):
            raise ValueError("solid model requires nonnegative support and a 290-850 K domain")

    @property
    def properties(self):
        return self

    @property
    def component_ids(self):
        return ("A", "B")

    def to_dict(self):
        return {**asdict(self), "phases": [asdict(p) for p in self.phases]}

    @classmethod
    def from_dict(cls, payload):
        return cls(**{**payload, "phases": tuple(SolidPhaseSpec(**p) for p in payload["phases"])})

    def reference_gibbs(self, temperature):
        return self.molar_heat_capacity_J_mol_K * (
            temperature
            - self.reference_temperature_K
            - temperature * log(temperature / self.reference_temperature_K)
        )

    def phase_enthalpy(self, phase, x, temperature):
        return phase.reference_enthalpy(x) + self.molar_heat_capacity_J_mol_K * (
            temperature - self.reference_temperature_K
        )

    def _support_deficit(self, temperature, slope, intercept):
        return max(
            0.0,
            max(
                intercept - (p.gibbs(x, temperature) - slope * x)
                for p in self.phases
                for x in p.stationary_points(slope, temperature)
            ),
        )

    def at_tp(self, amounts, temperature):
        if (
            set(amounts) != {"A", "B"}
            or any(not isfinite(v) or v < 0 for v in amounts.values())
            or not self.minimum_temperature_K <= temperature <= self.maximum_temperature_K
        ):
            raise ValueError(
                "solid equilibrium needs finite A/B stock within its temperature domain"
            )
        total = sum(amounts.values())
        if total == 0:
            return SolidEquilibrium(
                temperature,
                dict(amounts),
                (),
                0.0,
                self.support_heat_capacity_J_K * (temperature - self.reference_temperature_K),
                0.0,
                0.0,
            )
        z = amounts["A"] / total
        uniform = min(self.phases, key=lambda p: p.gibbs(z, temperature))
        tolerance = 1e-7 * R * temperature
        deficit = 0.0
        if 0 < z < 1:
            slope = uniform.slope(z, temperature)
            deficit = self._support_deficit(
                temperature, slope, uniform.gibbs(z, temperature) - slope * z
            )
        residual = 0.0
        if z in (0, 1) or deficit <= tolerance:
            domains = (SolidComposition(uniform.name, z, 1.0),)
        else:
            grid = np.unique(np.r_[np.linspace(0, 1, 257), z])
            energies = np.array([p.gibbs(grid, temperature) for p in self.phases])
            winners = np.argmin(energies, axis=0)
            hull = []
            for i, x in enumerate(grid):
                point = (float(x), float(energies[winners[i], i]), int(winners[i]))
                while len(hull) > 1:
                    a, b = hull[-2:]
                    if (b[1] - a[1]) * (point[0] - b[0]) < (point[1] - b[1]) * (b[0] - a[0]):
                        break
                    hull.pop()
                hull.append(point)
            left, right = next((a, b) for a, b in pairwise(hull) if a[0] <= z <= b[0])
            p, q = self.phases[left[2]], self.phases[right[2]]

            def equations(x):
                u, v = x
                dp, dq = p.slope(u, temperature), q.slope(v, temperature)
                return np.array(
                    (dp - dq, p.gibbs(u, temperature) - u * dp - q.gibbs(v, temperature) + v * dq)
                ) / (R * temperature)

            solution = least_squares(
                equations,
                [max(left[0], 2e-14), min(right[0], 1 - 2e-14)],
                bounds=([1e-14, z], [z, 1 - 1e-14]),
                xtol=1e-13,
                ftol=1e-13,
                gtol=1e-13,
                max_nfev=200,
            )
            u, v = map(float, solution.x)
            residual = float(max(abs(equations(solution.x)))) * R * temperature
            slope = p.slope(u, temperature)
            deficit = self._support_deficit(temperature, slope, p.gibbs(u, temperature) - u * slope)
            if not solution.success or residual > tolerance or deficit > tolerance or v - u < 1e-12:
                raise RuntimeError("solid common tangent did not satisfy global stability")
            fraction = (v - z) / (v - u)
            domains = (
                SolidComposition(p.name, u, fraction),
                SolidComposition(q.name, v, 1 - fraction),
            )
        by_name = {p.name: p for p in self.phases}
        gibbs = total * sum(
            c.fraction
            * (by_name[c.phase_name].gibbs(c.x_A, temperature) + self.reference_gibbs(temperature))
            for c in domains
        )
        enthalpy = total * sum(
            c.fraction * self.phase_enthalpy(by_name[c.phase_name], c.x_A, temperature)
            for c in domains
        )
        enthalpy += self.support_heat_capacity_J_K * (temperature - self.reference_temperature_K)
        return SolidEquilibrium(
            temperature, dict(amounts), domains, float(gibbs), float(enthalpy), residual, deficit
        )

    def at_ph(self, index, amounts, enthalpy):
        if not isfinite(enthalpy):
            raise ValueError("solid H must be finite")
        if sum(amounts.values()) == 0 and self.support_heat_capacity_J_K == 0:
            raise ValueError("an empty unsupported solid has no caloric temperature")
        low, high = self.minimum_temperature_K, self.maximum_temperature_K
        hlow, hhigh = (self.at_tp(amounts, t).enthalpy_J for t in (low, high))
        if not hlow - 1e-9 <= enthalpy <= hhigh + 1e-9:
            raise ValueError("solid enthalpy leaves its declared domain")
        target = min(max(enthalpy, hlow), hhigh)
        temperature = brentq(
            lambda t: self.at_tp(amounts, t).enthalpy_J - target, low, high, xtol=1e-9
        )
        return self.at_tp(amounts, temperature)


def ideal_two_phase_spec(**kwargs):
    return SolidSolutionSpec(
        phases=(SolidPhaseSpec("alpha", 0, 5000, 0), SolidPhaseSpec("beta", 5000, 0, 0)), **kwargs
    )
