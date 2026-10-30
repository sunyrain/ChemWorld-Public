"""Axisymmetric opposed-flow steady queries on the original shared root.

G=rho*u, V=v/r and Lambda=(1/r)*dp/dr. This finite-volume similarity
problem has two material inlets and a pressure eigenvalue, no flame phase.
It does not advance a world, clock or reservoir. Transport is native; no
Cantera backend is called. Radiation, Soret and non-ideal EOS are excluded.
Equations: https://cantera.org/3.2/reference/onedim/governing-equations.html
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from functools import partial
from math import fsum, isfinite, sqrt
from traceback import format_exc
from types import MappingProxyType
from typing import Any

import numpy as np

from chemworld.physchem.finite_volume import DiffusionMesh
from chemworld.physchem.gas_polynomial_transport import (
    GasTransportPolynomialSpec,
    _bound_polynomial_response,
)
from chemworld.physchem.gas_premixed_flow import _face_mole_gradients, _fractions, _positive
from chemworld.physchem.gas_transport import (
    GasTransportSpec,
    gas_transport_from_dict,
    gas_transport_response,
)
from chemworld.physchem.gas_transport_fluxes import gas_transport_fluxes
from chemworld.physchem.reaction_network import R_J_PER_MOL_K as R
from chemworld.physchem.reaction_network import ReactionNetworkSpec
from chemworld.physchem.residual_system import (
    DEFAULT_ROOT_POLICY,
    ResidualDerivatives,
    ResidualDomainError,
    ResidualEquation,
    ResidualSystem,
    ResidualVariable,
    RootPolicy,
    solve_steady_residual,
)

MODEL = "opposed-gas-flow-v1"
_INITIAL_FIELDS = {
    "amounts_mol",
    "temperatures_K",
    "spread_rates_per_s",
    "face_mass_fluxes_kg_m2_s",
    "radial_pressure_gradient_Pa_m2",
    "left_inlet_trace_mole_fractions",
    "right_inlet_trace_mole_fractions",
}


def _finite(value, name):
    if isinstance(value, bool) or not isinstance(value, int | float) or not isfinite(value):
        raise ValueError(f"{name} must be explicitly finite numeric")
    return float(value)


@dataclass(frozen=True)
class GasOpposedFlowSpec:
    mesh: DiffusionMesh
    transport: GasTransportSpec | GasTransportPolynomialSpec
    pressure_Pa: float
    left_inlet_temperature_K: float
    right_inlet_temperature_K: float
    left_inlet_mole_fractions: Mapping[str, float]
    right_inlet_mole_fractions: Mapping[str, float]
    left_mass_flux_kg_m2_s: float
    right_mass_flux_kg_m2_s: float
    closure_species_id: str
    temperature_bounds_K: tuple[float, float]
    left_spread_rate_per_s: float = 0.0
    right_spread_rate_per_s: float = 0.0
    flux_model: str = "mixture-averaged"
    maximum_mach: float = 0.05

    def __post_init__(self):
        if not isinstance(self.mesh, DiffusionMesh) or self.mesh.geometry != "planar":
            raise ValueError("opposed flow requires the original planar DiffusionMesh")
        if self.mesh.cells < 2:
            raise ValueError("opposed flow needs at least two control volumes")
        if not isinstance(self.transport, GasTransportSpec | GasTransportPolynomialSpec):
            raise ValueError("declare the original native gas transport")
        if self.transport.thermal_diffusion != "off":
            raise ValueError("opposed-flow first slice requires explicit Soret off")
        ids = (
            self.transport.species_ids
            if isinstance(self.transport, GasTransportPolynomialSpec)
            else tuple(p.component_id for p in self.transport.parameters)
        )
        for name in ("left_inlet_mole_fractions", "right_inlet_mole_fractions"):
            object.__setattr__(self, name, MappingProxyType(_fractions(getattr(self, name), ids)))
        if self.closure_species_id not in ids:
            raise ValueError("closure species must be an original species")
        for name in (
            "pressure_Pa",
            "left_inlet_temperature_K",
            "right_inlet_temperature_K",
            "left_mass_flux_kg_m2_s",
            "right_mass_flux_kg_m2_s",
        ):
            _positive(getattr(self, name), name)
        if self.pressure_Pa > self.transport.maximum_pressure_Pa:
            raise ValueError("pressure exceeds the explicit transport domain")
        if (
            not isinstance(self.temperature_bounds_K, tuple | list)
            or len(self.temperature_bounds_K) != 2
        ):
            raise ValueError("temperature bounds require exactly two endpoints")
        lo, hi = (_positive(t, "temperature bound") for t in self.temperature_bounds_K)
        if not lo < hi or any(
            not lo <= t <= hi
            for t in (self.left_inlet_temperature_K, self.right_inlet_temperature_K)
        ):
            raise ValueError("both inlet temperatures must be within the declared domain")
        object.__setattr__(self, "temperature_bounds_K", (lo, hi))
        _finite(self.left_spread_rate_per_s, "left spread rate")
        _finite(self.right_spread_rate_per_s, "right spread rate")
        if self.flux_model not in {"mixture-averaged", "stefan-maxwell"}:
            raise ValueError("unsupported whole-mixture diffusion model")
        if _positive(self.maximum_mach, "maximum Mach") > 0.3:
            raise ValueError("declare an axial-centerline low-Mach bound <=0.3")

    def to_dict(self):
        return {
            "model": MODEL,
            "mesh": {
                "geometry": "planar",
                "faces_m": list(self.mesh.faces_m),
                "area_m2": self.mesh.area_m2,
            },
            "transport": self.transport.to_dict(),
            **{
                name: dict(getattr(self, name))
                for name in ("left_inlet_mole_fractions", "right_inlet_mole_fractions")
            },
            "temperature_bounds_K": list(self.temperature_bounds_K),
            **{
                name: getattr(self, name)
                for name in (
                    "pressure_Pa",
                    "left_inlet_temperature_K",
                    "right_inlet_temperature_K",
                    "left_mass_flux_kg_m2_s",
                    "right_mass_flux_kg_m2_s",
                    "closure_species_id",
                    "left_spread_rate_per_s",
                    "right_spread_rate_per_s",
                    "flux_model",
                    "maximum_mach",
                )
            },
        }

    @classmethod
    def from_dict(cls, value):
        fields = set(cls.__dataclass_fields__) | {"model"}
        if not isinstance(value, Mapping) or set(value) != fields or value["model"] != MODEL:
            raise ValueError("unknown opposed-flow codec or fields")
        if not isinstance(value["mesh"], Mapping) or set(value["mesh"]) != {
            "geometry",
            "faces_m",
            "area_m2",
        }:
            raise ValueError("unknown opposed-flow mesh codec fields")
        data = {k: v for k, v in value.items() if k not in {"model", "mesh", "transport"}}
        return cls(
            mesh=DiffusionMesh(**value["mesh"]),
            transport=gas_transport_from_dict(value["transport"]),
            **data,
        )


def opposed_flow_initial_profile(
    spec: GasOpposedFlowSpec,
    network: ReactionNetworkSpec,
    *,
    temperatures_K: Sequence[float],
    mole_fractions: Sequence[Mapping[str, float]],
    spread_rates_per_s: Sequence[float] | None = None,
    face_mass_fluxes_kg_m2_s: Sequence[float] | None = None,
    radial_pressure_gradient_Pa_m2: float = 0.0,
    left_inlet_trace_mole_fractions: Mapping[str, float] | None = None,
    right_inlet_trace_mole_fractions: Mapping[str, float] | None = None,
):
    """Caller-owned EOS guess only; no inventory, phase or pressure forcing."""
    if not isinstance(spec, GasOpposedFlowSpec) or not isinstance(network, ReactionNetworkSpec):
        raise ValueError("initial guess requires the original spec and network")
    if len(temperatures_K) != spec.mesh.cells or len(mole_fractions) != spec.mesh.cells:
        raise ValueError("initial profiles must cover every control volume")
    amounts, temperatures = [], []
    for t, x, volume in zip(temperatures_K, mole_fractions, spec.mesh.cell_volumes_m3, strict=True):
        t = _positive(t, "initial temperature")
        if not spec.temperature_bounds_K[0] <= t <= spec.temperature_bounds_K[1]:
            raise ValueError("initial temperature outside declared domain")
        x = _fractions(x, network.species_ids)
        total = spec.pressure_Pa * float(volume) / (R * t)
        amounts.append({s: total * x[s] for s in network.species_ids})
        temperatures.append(t)
    v = (
        np.interp(
            spec.mesh.centers_m,
            (0, spec.mesh.faces_m[-1]),
            (spec.left_spread_rate_per_s, spec.right_spread_rate_per_s),
        ).tolist()
        if spread_rates_per_s is None
        else list(spread_rates_per_s)
    )
    g = (
        np.interp(
            spec.mesh.faces_m,
            (0, spec.mesh.faces_m[-1]),
            (spec.left_mass_flux_kg_m2_s, -spec.right_mass_flux_kg_m2_s),
        ).tolist()
        if face_mass_fluxes_kg_m2_s is None
        else list(face_mass_fluxes_kg_m2_s)
    )
    if len(v) != spec.mesh.cells or len(g) != spec.mesh.cells + 1:
        raise ValueError("V/G guesses must cover the actual cells/faces")
    return {
        "amounts_mol": amounts,
        "temperatures_K": temperatures,
        "spread_rates_per_s": [_finite(value, "spread rate") for value in v],
        "face_mass_fluxes_kg_m2_s": [_finite(value, "face mass flux") for value in g],
        "radial_pressure_gradient_Pa_m2": _finite(radial_pressure_gradient_Pa_m2, "Lambda"),
        "left_inlet_trace_mole_fractions": _fractions(
            spec.left_inlet_mole_fractions
            if left_inlet_trace_mole_fractions is None
            else left_inlet_trace_mole_fractions,
            network.species_ids,
        ),
        "right_inlet_trace_mole_fractions": _fractions(
            spec.right_inlet_mole_fractions
            if right_inlet_trace_mole_fractions is None
            else right_inlet_trace_mole_fractions,
            network.species_ids,
        ),
    }


class _OpposedResidual:
    def __init__(
        self,
        spec,
        network,
        initial,
        maximum_evaluations,
        progress,
        *,
        thermo_memo_max_entries=None,
    ):
        self.spec = spec = GasOpposedFlowSpec.from_dict(deepcopy(spec.to_dict()))
        self.network = network = ReactionNetworkSpec.from_dict(deepcopy(network.to_dict()))
        self.ids, self.count, self.size = network.species_ids, spec.mesh.cells, len(network.species)
        self.active = [s for s in self.ids if s != spec.closure_species_id]
        self.calls, self.maximum_evaluations, self.progress = 0, maximum_evaluations, progress
        if network.stochastic is not None or any(
            s.phase != "gas" or s.charge for s in network.species
        ):
            raise ValueError("opposed flow requires the original deterministic neutral gas network")
        if set(network.imported_gas_thermochemistry) != set(self.ids):
            raise ValueError("all original thermochemistry is required")
        lo, hi = spec.temperature_bounds_K
        if any(
            lo < m.min_temperature_K or hi > m.max_temperature_K
            for m in network.imported_gas_thermochemistry.values()
        ):
            raise ValueError("temperature interval exceeds original thermochemistry")
        self.thermo_memo = None
        self.thermo_views = None
        if thermo_memo_max_entries is not None:
            if not isinstance(spec.transport, GasTransportPolynomialSpec):
                raise ValueError("query thermo memo currently requires polynomial transport")
            from chemworld.physchem.gas_query_thermo import GasQueryThermoMemo

            self.thermo_memo = GasQueryThermoMemo(
                network.imported_gas_thermochemistry,
                maximum_states=thermo_memo_max_entries,
            )
            self.thermo_views = self.thermo_memo.views
        self.bound_response = None
        if isinstance(spec.transport, GasTransportPolynomialSpec):
            spec.transport.validate_network(network)
            if (
                lo < spec.transport.fit_temperature_range_K[0]
                or hi > spec.transport.fit_temperature_range_K[1]
            ):
                raise ValueError("temperature interval exceeds original transport fit")
            self.bound_response = partial(
                _bound_polynomial_response,
                spec.transport,
                self.ids,
                MappingProxyType(network.imported_gas_thermochemistry)
                if self.thermo_views is None
                else self.thermo_views,
            )
        left = self.response(spec.left_inlet_temperature_K, spec.left_inlet_mole_fractions)
        right = self.response(spec.right_inlet_temperature_K, spec.right_inlet_mole_fractions)
        self.masses = dict(left.molecular_weights_kg_mol)
        self.feed_y = (dict(left.mass_fractions), dict(right.mass_fractions))
        for point in (left, right):
            network.amount_derivatives(
                {
                    s: point.molar_density_mol_m3
                    * float(spec.mesh.cell_volumes_m3[0])
                    * point.mole_fractions[s]
                    for s in self.ids
                },
                volume_L=float(spec.mesh.cell_volumes_m3[0]) * 1000,
                temperature_K=point.temperature_K,
                **({"species_thermo": self.thermo_views} if self.thermo_views is not None else {}),
            )
        if not isinstance(initial, Mapping) or set(initial) != _INITIAL_FIELDS:
            raise ValueError("opposed initial profile has missing or unknown fields")
        if (
            any(
                len(initial[name]) != self.count
                for name in ("amounts_mol", "temperatures_K", "spread_rates_per_s")
            )
            or len(initial["face_mass_fluxes_kg_m2_s"]) != self.count + 1
        ):
            raise ValueError("opposed initial profile does not match the actual mesh")
        amounts = []
        for row in initial["amounts_mol"]:
            if not isinstance(row, Mapping) or set(row) != set(self.ids):
                raise ValueError("initial stocks must cover all original species")
            values = [_finite(row[s], "initial stock") for s in self.ids]
            if min(values) < 0 or fsum(values) <= 0:
                raise ValueError("initial stocks must be nonnegative with a positive cell total")
            amounts.extend(values)
        t = [_positive(t, "initial temperature") for t in initial["temperatures_K"]]
        if any(not lo <= value <= hi for value in t):
            raise ValueError("initial temperature outside declared domain")
        v = [_finite(v, "initial V") for v in initial["spread_rates_per_s"]]
        g = [_finite(g, "initial G") for g in initial["face_mass_fluxes_kg_m2_s"]]
        lam = _finite(initial["radial_pressure_gradient_Pa_m2"], "initial Lambda")
        traces = [
            _fractions(initial[side + "_inlet_trace_mole_fractions"], self.ids)
            for side in ("left", "right")
        ]
        self.n_stop = self.count * self.size
        self.t_start, self.v_start = self.n_stop, self.n_stop + self.count
        self.g_start = self.n_stop + 2 * self.count
        self.lambda_col = self.g_start + self.count + 1
        self.trace_start = self.lambda_col + 1
        self.initial = np.asarray(
            [
                *amounts,
                *t,
                *v,
                *g,
                lam,
                *(traces[0][s] for s in self.ids),
                *(traces[1][s] for s in self.ids),
            ]
        )
        tref = 0.5 * (left.temperature_K + right.temperature_K)
        gref = max(spec.left_mass_flux_kg_m2_s, spec.right_mass_flux_kg_m2_s)
        rho = 0.5 * (left.density_kg_m3 + right.density_kg_m3)
        vref = max(
            abs(spec.left_spread_rate_per_s),
            abs(spec.right_spread_rate_per_s),
            gref / (rho * spec.mesh.faces_m[-1]),
        )
        lamref = rho * vref**2
        mass_scale = gref * spec.mesh.area_m2
        heat_scale = mass_scale * max(
            max(
                abs(p.partial_enthalpies_J_mol[s]),
                (
                    network.imported_gas_thermochemistry
                    if self.thermo_views is None
                    else self.thermo_views
                )[s]
                .evaluate(p.temperature_K)
                .cp_J_mol_K
                * p.temperature_K,
            )
            / self.masses[s]
            for p in (left, right)
            for s in self.ids
        )
        momentum_scale = max(
            mass_scale * vref,
            max(left.viscosity_Pa_s, right.viscosity_Pa_s)
            * vref
            * spec.mesh.area_m2
            / spec.mesh.faces_m[-1],
        )
        for value in (tref, gref, vref, lamref, mass_scale, heat_scale, momentum_scale):
            _positive(value, "opposed residual scale")
        self.variables, self.equations = [], []
        for i in range(self.count):
            scale = spec.pressure_Pa * float(spec.mesh.cell_volumes_m3[i]) / (R * tref)
            self.variables.extend(
                ResidualVariable(f"n:{i}:{s}", "mol", scale, role="algebraic", lower=0)
                for s in self.ids
            )
            self.equations.extend(
                ResidualEquation(
                    f"species:{i}:{s}", "mol/s", mass_scale / min(self.masses.values())
                )
                for s in self.active
            )
            self.equations.append(ResidualEquation(f"EOS:{i}", "Pa", spec.pressure_Pa))
        self.variables.extend(
            ResidualVariable(f"T:{i}", "K", tref, role="algebraic", lower=lo, upper=hi)
            for i in range(self.count)
        )
        self.variables.extend(
            ResidualVariable(f"V:{i}", "1/s", vref, role="algebraic") for i in range(self.count)
        )
        self.variables.extend(
            ResidualVariable(f"G:{i}", "kg/(m^2*s)", gref, role="algebraic")
            for i in range(self.count + 1)
        )
        self.variables.append(ResidualVariable("Lambda", "Pa/m^2", lamref, role="algebraic"))
        for side in ("left", "right"):
            self.variables.extend(
                ResidualVariable(f"trace:{side}:{s}", "1", 1, role="algebraic", lower=0, upper=1)
                for s in self.ids
            )
        self.equations.extend(
            ResidualEquation(f"energy:{i}", "W", heat_scale) for i in range(self.count)
        )
        self.equations.extend(
            ResidualEquation(f"momentum:{i}", "kg/s^2", momentum_scale) for i in range(self.count)
        )
        self.equations.extend(
            ResidualEquation(f"continuity:{i}", "kg/s", mass_scale) for i in range(self.count)
        )
        for side in ("left", "right"):
            self.equations.extend(
                ResidualEquation(
                    f"feed:{side}:{s}", "mol/s", mass_scale / min(self.masses.values())
                )
                for s in self.active
            )
        self.equations.extend(
            ResidualEquation(f"trace-sum:{side}", "1", 1) for side in ("left", "right")
        )
        self.equations.extend(
            ResidualEquation(f"G-boundary:{side}", "kg/s", mass_scale) for side in ("left", "right")
        )

    def response(self, t, x):
        try:
            if self.bound_response is not None:
                return self.bound_response(
                    temperature_K=float(t), pressure_Pa=self.spec.pressure_Pa, mole_fractions=x
                )
            return gas_transport_response(
                self.spec.transport,
                self.network,
                temperature_K=float(t),
                pressure_Pa=self.spec.pressure_Pa,
                mole_fractions=x,
            )
        except (ValueError, ArithmeticError) as exc:
            raise ResidualDomainError("opposed transport trial: " + str(exc)) from exc

    def decode(self, vector):
        value = np.asarray(vector)
        if value.shape != self.initial.shape or not np.all(np.isfinite(value)):
            raise ResidualDomainError(
                "opposed trial has invalid raw shape or nonfinite coordinates"
            )
        n = value[: self.n_stop].reshape((self.count, self.size))
        t = value[self.t_start : self.v_start]
        v = value[self.v_start : self.g_start]
        g = value[self.g_start : self.lambda_col]
        lam = float(value[self.lambda_col])
        z = value[self.trace_start :].reshape((2, self.size))
        totals, ztotals = [fsum(row) for row in n], [fsum(row) for row in z]
        if np.any(n < 0) or np.any(z < 0) or min([*totals, *ztotals]) <= 0:
            raise ResidualDomainError("opposed trial contains negative or empty material")
        if np.any(t < self.spec.temperature_bounds_K[0]) or np.any(
            t > self.spec.temperature_bounds_K[1]
        ):
            raise ResidualDomainError("opposed trial temperature outside declared domain")
        x = [
            dict(zip(self.ids, map(float, row / total), strict=True))
            for row, total in zip(n, totals, strict=True)
        ]
        traces = [
            dict(zip(self.ids, map(float, row / total), strict=True))
            for row, total in zip(z, ztotals, strict=True)
        ]
        return n, t, v, g, lam, x, traces, ztotals

    def state_sparsity(self):
        columns = [set() for _ in self.variables]
        boundary = self.n_stop + 3 * self.count

        def cell_rows(i):
            return {
                *range(i * self.size, (i + 1) * self.size),
                self.n_stop + i,
                self.n_stop + self.count + i,
                self.n_stop + 2 * self.count + i,
            }

        def feed_rows(side):
            return set(
                range(boundary + side * (self.size - 1), boundary + (side + 1) * (self.size - 1))
            )

        for i in range(self.count):
            rows = set().union(
                *(cell_rows(j) for j in range(max(0, i - 1), min(self.count, i + 2)))
            )
            if i == 0:
                rows |= feed_rows(0)
            if i == self.count - 1:
                rows |= feed_rows(1)
            for column in [
                *range(i * self.size, (i + 1) * self.size),
                self.t_start + i,
                self.v_start + i,
            ]:
                columns[column] |= rows
        for face in range(self.count + 1):
            for i in (face - 1, face):
                if 0 <= i < self.count:
                    columns[self.g_start + face] |= cell_rows(i)
            if face in (0, self.count):
                side = int(face == self.count)
                columns[self.g_start + face] |= feed_rows(side)
                columns[self.g_start + face].add(boundary + 2 * self.size + side)
        columns[self.lambda_col] = set(
            range(self.n_stop + self.count, self.n_stop + 2 * self.count)
        )
        for side, i in ((0, 0), (1, self.count - 1)):
            rows = cell_rows(i) | feed_rows(side) | {boundary + 2 * (self.size - 1) + side}
            for column in range(
                self.trace_start + side * self.size, self.trace_start + (side + 1) * self.size
            ):
                columns[column] |= rows
        return tuple(tuple(sorted(rows)) for rows in columns)

    def evaluate(self, vector, *, retain_details=True):
        if type(retain_details) is not bool:
            raise ValueError("retain_details must be explicitly boolean")
        self.calls += 1
        if self.calls > self.maximum_evaluations:
            raise ResidualDomainError("opposed residual evaluation budget exceeded")
        if self.progress is not None:
            self.progress(
                {
                    "event": "opposed-residual",
                    "evaluations": self.calls,
                    "maximum_evaluations": self.maximum_evaluations,
                }
            )
        try:
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                return self._evaluate(vector, retain_details=retain_details)
        except ResidualDomainError:
            raise
        except (ValueError, ArithmeticError) as exc:
            raise ResidualDomainError("opposed finite trial domain: " + str(exc)) from exc

    def _evaluate(self, vector, *, retain_details=True):
        n, t, v, g, lam, x, traces, ztotals = self.decode(vector)
        spec, ids, area = self.spec, self.ids, self.spec.mesh.area_m2
        points = [self.response(float(ti), row) for ti, row in zip(t, x, strict=True)]
        ends = [
            self.response(spec.left_inlet_temperature_K, traces[0]),
            self.response(spec.right_inlet_temperature_K, traces[1]),
        ]
        all_points = [ends[0], *points, ends[1]]
        coords = [0.0, *spec.mesh.centers_m, spec.mesh.faces_m[-1]]
        spread = [spec.left_spread_rate_per_s, *v, spec.right_spread_rate_per_s]
        fs, fh, fm, parts, states, receipts = [], [], [], [], [], []
        for face in range(self.count + 1):
            left, right = all_points[face : face + 2]
            distance = float(coords[face + 1] - coords[face])
            weight = (
                0.5
                if face in (0, self.count)
                else float((spec.mesh.faces_m[face] - coords[face]) / distance)
            )
            mean_y = {
                s: (1 - weight) * left.mass_fractions[s] + weight * right.mass_fractions[s]
                for s in ids
            }
            molar = {s: mean_y[s] / self.masses[s] for s in ids}
            total = fsum(molar.values())
            mid = self.response(
                (1 - weight) * left.temperature_K + weight * right.temperature_K,
                {s: value / total for s, value in molar.items()},
            )
            gradient, receipt = _face_mole_gradients(
                left.mole_fractions, right.mole_fractions, distance, spec.closure_species_id
            )
            if retain_details:
                receipts.append(receipt)
            flux = gas_transport_fluxes(
                mid,
                mole_fraction_gradients_per_m=gradient,
                temperature_gradient_K_m=(right.temperature_K - left.temperature_K) / distance,
                model=spec.flux_model,
                frame="mass",
                thermal_diffusion="off",
            )
            advective = left if g[face] >= 0 else right
            vad = spread[face] if g[face] >= 0 else spread[face + 1]
            diffusion_h = flux.diffusion_enthalpy_flux_W_m2
            if face in (0, self.count):
                boundary = ends[int(face == self.count)]
                # Both ends are true Dirichlet-T/V inlets, never outlet extrapolations.
                advective = self.response(boundary.temperature_K, advective.mole_fractions)
                vad = spread[0] if face == 0 else spread[-1]
                diffusion_h = fsum(
                    boundary.partial_enthalpies_J_mol[s] * flux.molar_fluxes_mol_m2_s[s]
                    for s in ids
                )
            fs.append(
                {
                    s: area
                    * (
                        float(g[face]) * advective.mass_fractions[s] / self.masses[s]
                        + flux.molar_fluxes_mol_m2_s[s]
                    )
                    for s in ids
                }
            )
            bulk_h = fsum(
                advective.mass_fractions[s] * advective.partial_enthalpies_J_mol[s] / self.masses[s]
                for s in ids
            )
            fh.append(area * (float(g[face]) * bulk_h + flux.fourier_heat_flux_W_m2 + diffusion_h))
            fm.append(
                area
                * (
                    float(g[face]) * float(vad)
                    - mid.viscosity_Pa_s * float(spread[face + 1] - spread[face]) / distance
                )
            )
            if retain_details:
                part = flux.to_dict()
                part.update(
                    diffusion_enthalpy_flux_W_m2=diffusion_h,
                    total_heat_flux_W_m2=flux.fourier_heat_flux_W_m2 + diffusion_h,
                )
                parts.append(part)
                states.append(
                    {
                        "mid_temperature_K": mid.temperature_K,
                        "mid_mole_fractions": dict(mid.mole_fractions),
                        "advective_temperature_K": advective.temperature_K,
                        "advective_mole_fractions": dict(advective.mole_fractions),
                        "viscosity_Pa_s": mid.viscosity_Pa_s,
                        "advective_spread_rate_per_s": float(vad),
                    }
                )
        source, rad_species, rad_mass, rad_h = [], [], [], []
        sr, eos, heat, momentum, continuity, chem_mass = [], [], [], [], [], []
        cell_mass = [fsum(float(row[j]) * self.masses[s] for j, s in enumerate(ids)) for row in n]
        if retain_details:
            elements = sorted(
                {element for species in self.network.species for element in species.composition}
            )
            composition = {s.species_id: s.composition for s in self.network.species}
        rad_elements = []
        for i in range(self.count):
            row = dict(zip(ids, map(float, n[i]), strict=True))
            try:
                omega = self.network.amount_derivatives(
                    row,
                    volume_L=float(spec.mesh.cell_volumes_m3[i]) * 1000,
                    temperature_K=float(t[i]),
                    **(
                        {"species_thermo": self.thermo_views}
                        if self.thermo_views is not None
                        else {}
                    ),
                )
            except (ValueError, ArithmeticError) as exc:
                raise ResidualDomainError("opposed chemistry trial: " + str(exc)) from exc
            if any(not isfinite(value) for value in omega.values()):
                raise ResidualDomainError("opposed trial chemistry produced nonfinite rates")
            if retain_details:
                source.append(omega)
            radial = {s: 2 * float(v[i]) * row[s] for s in ids}
            if retain_details:
                rad_species.append(radial)
            rad_mass.append(2 * float(v[i]) * cell_mass[i])
            rad_h.append(
                2 * float(v[i]) * fsum(row[s] * points[i].partial_enthalpies_J_mol[s] for s in ids)
            )
            if retain_details:
                rad_elements.append(
                    {e: fsum(radial[s] * composition[s].get(e, 0) for s in ids) for e in elements}
                )
            sr.append({s: fs[i + 1][s] - fs[i][s] + radial[s] - omega[s] for s in ids})
            eos.append(
                fsum(row.values()) * R * float(t[i]) / float(spec.mesh.cell_volumes_m3[i])
                - spec.pressure_Pa
            )
            heat.append(fh[i + 1] - fh[i] + rad_h[i])
            momentum.append(
                fm[i + 1]
                - fm[i]
                + 3 * float(v[i]) ** 2 * cell_mass[i]
                + lam * float(spec.mesh.cell_volumes_m3[i])
            )
            continuity.append(area * float(g[i + 1] - g[i]) + rad_mass[i])
            if retain_details:
                chem_mass.append(fsum(omega[s] * self.masses[s] for s in ids))
        signed_feed = (spec.left_mass_flux_kg_m2_s, -spec.right_mass_flux_kg_m2_s)
        boundary_species = [
            {
                s: fs[face][s] - area * signed_feed[side] * self.feed_y[side][s] / self.masses[s]
                for s in ids
            }
            for side, face in ((0, 0), (1, self.count))
        ]
        g_boundary = [area * (float(g[0]) - signed_feed[0]), area * (float(g[-1]) - signed_feed[1])]
        residual = [
            value for i in range(self.count) for value in [*(sr[i][s] for s in self.active), eos[i]]
        ]
        residual.extend([*heat, *momentum, *continuity])
        residual.extend(boundary_species[side][s] for side in (0, 1) for s in self.active)
        residual.extend([ztotals[0] - 1, ztotals[1] - 1, *g_boundary])
        if not retain_details:
            values = np.asarray(residual)
            if not np.all(np.isfinite(values)):
                raise ResidualDomainError("opposed residual exceeded its finite numeric domain")
            return values, None
        face_mass = [fsum(row[s] * self.masses[s] for s in ids) for row in fs]
        element_balance = {
            e: fsum((fs[-1][s] - fs[0][s]) * composition[s].get(e, 0) for s in ids)
            + fsum(row[e] for row in rad_elements)
            for e in elements
        }
        chemical_elements = [
            {e: fsum(row[s] * composition[s].get(e, 0) for s in ids) for e in elements}
            for row in source
        ]
        global_chemical_elements = {e: fsum(row[e] for row in chemical_elements) for e in elements}
        mass_balance = face_mass[-1] - face_mass[0] + fsum(rad_mass)
        details = {
            "amounts_mol": [dict(zip(ids, map(float, row), strict=True)) for row in n],
            "temperatures_K": list(map(float, t)),
            "spread_rates_per_s": list(map(float, v)),
            "face_mass_fluxes_kg_m2_s": list(map(float, g)),
            "radial_pressure_gradient_Pa_m2": lam,
            "left_inlet_trace_mole_fractions": traces[0],
            "right_inlet_trace_mole_fractions": traces[1],
            "inlet_trace_coordinate_sums": ztotals,
            "cell_density_kg_m3": [
                m / float(vol) for m, vol in zip(cell_mass, spec.mesh.cell_volumes_m3, strict=True)
            ],
            "axial_face_species_flows_mol_s": fs,
            "axial_face_enthalpy_flows_W": fh,
            "axial_face_momentum_flows_kg_s2": fm,
            "axial_face_mass_flows_kg_s": face_mass,
            "radial_species_outflows_mol_s": rad_species,
            "radial_mass_outflows_kg_s": rad_mass,
            "radial_enthalpy_outflows_W": rad_h,
            "radial_element_outflows_mol_atoms_s": rad_elements,
            "radial_convective_momentum_terms_kg_s2": [
                3 * float(vi) ** 2 * m for vi, m in zip(v, cell_mass, strict=True)
            ],
            "pressure_momentum_terms_kg_s2": [
                lam * float(vol) for vol in spec.mesh.cell_volumes_m3
            ],
            "chemical_sources_mol_s": source,
            "chemical_mass_residuals_kg_s": chem_mass,
            "global_chemical_mass_residual_kg_s": fsum(chem_mass),
            "chemical_element_residuals_mol_atoms_s": chemical_elements,
            "global_chemical_element_residuals_mol_atoms_s": global_chemical_elements,
            "species_balance_residuals_mol_s": sr,
            "EOS_residuals_Pa": eos,
            "energy_balance_residuals_W": heat,
            "momentum_balance_residuals_kg_s2": momentum,
            "continuity_residuals_kg_s": continuity,
            "inlet_species_residuals_mol_s": boundary_species,
            "G_boundary_residuals_kg_s": g_boundary,
            "global_element_balance_residuals_mol_atoms_s": element_balance,
            "global_element_pde_balance_residuals_mol_atoms_s": {
                e: element_balance[e] - global_chemical_elements[e] for e in elements
            },
            "global_mass_balance_residual_kg_s": mass_balance,
            "global_mass_pde_balance_residual_kg_s": mass_balance - fsum(chem_mass),
            "global_enthalpy_balance_residual_W": fh[-1] - fh[0] + fsum(rad_h),
            "face_diffusive_fluxes": parts,
            "face_transport_and_advective_states": states,
            "composition_gradient_roundoff_receipts": receipts,
        }
        values = np.asarray(residual)
        if not np.all(np.isfinite(values)):
            raise ResidualDomainError("opposed residual exceeded its finite numeric domain")
        return values, details


def _opposed_fresh_checks(
    assembly,
    profile,
    *,
    root_policy,
    species_tolerance_mol_s,
    energy_tolerance_W,
    pressure_tolerance_Pa,
    mass_tolerance_kg_s,
    momentum_tolerance_kg_s2,
    element_tolerance_mol_s,
):
    """Original complete accepted-candidate checks, without any root call.

    Caller owns full ``assembly.evaluate`` and its original exception boundary.
    This extraction preserves the original Mach arithmetic, check order and
    profile mutation; it neither repairs a candidate nor changes inventories.
    """
    spec, network = assembly.spec, assembly.network
    mach = []
    states = []
    for i, (t, n) in enumerate(zip(profile["temperatures_K"], profile["amounts_mol"], strict=True)):
        x = {s: n[s] / fsum(n.values()) for s in assembly.ids}
        states.append(
            (
                t,
                x,
                0.5
                * (
                    profile["face_mass_fluxes_kg_m2_s"][i]
                    + profile["face_mass_fluxes_kg_m2_s"][i + 1]
                ),
                profile["cell_density_kg_m3"][i],
            )
        )
    for face, entry in enumerate(profile["face_transport_and_advective_states"]):
        point = assembly.response(entry["mid_temperature_K"], entry["mid_mole_fractions"])
        if face in (0, assembly.count):
            side = "left" if face == 0 else "right"
            t = getattr(spec, side + "_inlet_temperature_K")
            x = profile[side + "_inlet_trace_mole_fractions"]
            point = assembly.response(t, x)
        states.append(
            (
                point.temperature_K,
                point.mole_fractions,
                profile["face_mass_fluxes_kg_m2_s"][face],
                point.density_kg_m3,
            )
        )
    for t, x, g, rho in states:
        cp = fsum(
            x[s]
            * (
                network.imported_gas_thermochemistry
                if assembly.thermo_views is None
                else assembly.thermo_views
            )[s]
            .evaluate(t)
            .cp_J_mol_K
            for s in assembly.ids
        )
        if cp <= R:
            raise ValueError("opposed ideal-gas sound speed requires positive Cv")
        mean_mass = fsum(x[s] * assembly.masses[s] for s in assembly.ids)
        mach.append(abs(g) / rho / sqrt(cp / (cp - R) * R * t / mean_mass))
    profile["axial_centerline_Mach_cells_and_faces"] = mach
    checks = {
        "all_species_including_omitted_balance": max(
            abs(v) for row in profile["species_balance_residuals_mol_s"] for v in row.values()
        )
        <= species_tolerance_mol_s,
        "both_incoming_all_species": max(
            abs(v) for row in profile["inlet_species_residuals_mol_s"] for v in row.values()
        )
        <= species_tolerance_mol_s,
        "EOS": max(map(abs, profile["EOS_residuals_Pa"])) <= pressure_tolerance_Pa,
        "both_trace_normalizations": max(abs(v - 1) for v in profile["inlet_trace_coordinate_sums"])
        <= root_policy.tolerance,
        "energy": max(map(abs, profile["energy_balance_residuals_W"])) <= energy_tolerance_W,
        "radial_momentum": max(map(abs, profile["momentum_balance_residuals_kg_s2"]))
        <= momentum_tolerance_kg_s2,
        "continuity": max(map(abs, profile["continuity_residuals_kg_s"])) <= mass_tolerance_kg_s,
        "both_G_boundary_values_and_signs": max(map(abs, profile["G_boundary_residuals_kg_s"]))
        <= mass_tolerance_kg_s
        and profile["face_mass_fluxes_kg_m2_s"][0] > 0
        and profile["face_mass_fluxes_kg_m2_s"][-1] < 0,
        "chemical_mass": max(map(abs, profile["chemical_mass_residuals_kg_s"]))
        <= mass_tolerance_kg_s,
        "global_mass": abs(profile["global_mass_balance_residual_kg_s"]) <= mass_tolerance_kg_s,
        "global_mass_pde": abs(profile["global_mass_pde_balance_residual_kg_s"])
        <= mass_tolerance_kg_s,
        "global_chemical_mass": abs(profile["global_chemical_mass_residual_kg_s"])
        <= mass_tolerance_kg_s,
        "global_elements": max(
            map(abs, profile["global_element_balance_residuals_mol_atoms_s"].values())
        )
        <= element_tolerance_mol_s,
        "global_elements_pde": max(
            map(abs, profile["global_element_pde_balance_residuals_mol_atoms_s"].values())
        )
        <= element_tolerance_mol_s,
        "chemical_elements": max(
            abs(value)
            for row in profile["chemical_element_residuals_mol_atoms_s"]
            for value in row.values()
        )
        <= element_tolerance_mol_s,
        "global_chemical_elements": max(
            map(abs, profile["global_chemical_element_residuals_mol_atoms_s"].values())
        )
        <= element_tolerance_mol_s,
        "global_enthalpy": abs(profile["global_enthalpy_balance_residual_W"]) <= energy_tolerance_W,
        "axial_centerline_low_Mach": max(mach) <= spec.maximum_mach,
    }
    return checks


def solve_opposed_gas_flow(
    spec: GasOpposedFlowSpec,
    network: ReactionNetworkSpec,
    initial_profile: Mapping[str, Any],
    *,
    root_policy: RootPolicy = DEFAULT_ROOT_POLICY,
    derivative_step: float = 6e-6,
    maximum_unknowns: int = 512,
    maximum_evaluations: int = 100000,
    species_tolerance_mol_s: float = 1e-9,
    energy_tolerance_W: float = 1e-6,
    pressure_tolerance_Pa: float = 1e-4,
    mass_tolerance_kg_s: float = 1e-10,
    momentum_tolerance_kg_s2: float = 1e-8,
    element_tolerance_mol_s: float = 1e-9,
    jacobian_strategy: str = "dense",
    progress: Any = None,
    thermo_memo_max_entries: int | None = None,
    residual_only_trials: bool = False,
):
    """Fresh-check every original species and radial ledger; no physical fallback.

    Explicit opt-ins only reduce numerical work. A positive memo cap shares
    private NASA7 views across polynomial transport, both-feed preflight, cell
    chemistry, Cp scales and final Mach. No network cached property is replaced.
    Residual-only trials omit diagnostic serialization and ledgers outside F;
    the accepted fresh evaluator still constructs the complete original profile
    and all eighteen physical checks. None/False retain the original path.
    """
    if not isinstance(spec, GasOpposedFlowSpec) or not isinstance(network, ReactionNetworkSpec):
        raise ValueError("opposed query requires its original spec and network")
    if not isinstance(root_policy, RootPolicy):
        raise ValueError("declare the shared RootPolicy")
    if type(residual_only_trials) is not bool:
        raise ValueError("residual_only_trials must be explicitly boolean")
    if jacobian_strategy not in {"dense", "sparse-colored"}:
        raise ValueError("declare dense or sparse-colored numerical Jacobian")
    for value in (
        derivative_step,
        species_tolerance_mol_s,
        energy_tolerance_W,
        pressure_tolerance_Pa,
        mass_tolerance_kg_s,
        momentum_tolerance_kg_s2,
        element_tolerance_mol_s,
    ):
        _positive(value, "opposed query tolerance")
    if any(type(v) is not int or v < 1 for v in (maximum_unknowns, maximum_evaluations)):
        raise ValueError("opposed numerical budgets must be positive integers")
    assembly = _OpposedResidual(
        spec,
        network,
        initial_profile,
        maximum_evaluations,
        progress,
        thermo_memo_max_entries=thermo_memo_max_entries,
    )
    if len(assembly.variables) > maximum_unknowns:
        raise ValueError("opposed unknown count exceeds its explicit budget")
    system = ResidualSystem(
        MODEL,
        tuple(assembly.variables),
        tuple(assembly.equations),
        (lambda time, y, rate, parameters: assembly.evaluate(y, retain_details=False)[0])
        if residual_only_trials
        else (lambda time, y, rate, parameters: assembly.evaluate(y)[0]),
        ResidualDerivatives(
            "finite_difference",
            relative_step=derivative_step,
            state_sparsity=assembly.state_sparsity()
            if jacobian_strategy == "sparse-colored"
            else None,
        ),
    )
    report = {
        "schema": "opposed-gas-flow-query-v1",
        "success": False,
        "failure_reason": "",
        "profile": None,
        "spec": assembly.spec.to_dict(),
        "initial_value": assembly.initial.tolist(),
        "root_value": None,
        "unknowns": len(assembly.variables),
        "residual_evaluations": 0,
        "jacobian_strategy": jacobian_strategy,
        "numerical_derivatives": "shared finite_difference",
        "live_operations_executed": 0,
        "experimental_elapsed_s": 0.0,
        "scope": "local axisymmetric ideal-gas similarity BVP; axial-centerline Mach only; "
        "no automatic flame/extinction identification, Soret or mesh qualification",
        "boundary_policy": "two fixed Tin and signed total incoming species fluxes; "
        "given V at both inlets; Lambda eigenvalue; no phase",
    }

    def finish():
        # Numeric cache accounting includes the final fresh checks, never world work.
        if assembly.thermo_memo is not None:
            report["thermo_memo_statistics"] = assembly.thermo_memo.statistics()
        if residual_only_trials:
            report["residual_only_trials"] = True
        return report

    try:
        solved = solve_steady_residual(
            system, assembly.initial, policy=root_policy, progress=progress
        )
    except Exception as exc:
        report.update(
            failure_reason="root_exception",
            residual_evaluations=assembly.calls,
            root_exception={
                "type": f"{type(exc).__module__}.{type(exc).__qualname__}",
                "message": str(exc),
                "traceback": format_exc(),
            },
        )
        return finish()
    report.update(
        failure_reason=solved.root.failure_reason,
        root_diagnostics=solved.root.diagnostics,
        derivative_diagnostics=solved.diagnostics,
        residual_evaluations=assembly.calls,
        root_value=None if solved.root.value is None else solved.root.value.tolist(),
    )
    if not solved.root.success:
        return finish()
    try:
        _, profile = assembly.evaluate(solved.root.value)
    except ResidualDomainError as exc:
        report.update(
            failure_reason="fresh_check_failed: " + str(exc), residual_evaluations=assembly.calls
        )
        return finish()
    checks = _opposed_fresh_checks(
        assembly,
        profile,
        root_policy=root_policy,
        species_tolerance_mol_s=species_tolerance_mol_s,
        energy_tolerance_W=energy_tolerance_W,
        pressure_tolerance_Pa=pressure_tolerance_Pa,
        mass_tolerance_kg_s=mass_tolerance_kg_s,
        momentum_tolerance_kg_s2=momentum_tolerance_kg_s2,
        element_tolerance_mol_s=element_tolerance_mol_s,
    )
    report.update(
        success=all(checks.values()),
        physical_checks=checks,
        failure_reason="" if all(checks.values()) else "fresh_physical_balance_failed",
        residual_evaluations=assembly.calls,
    )
    report["profile" if report["success"] else "candidate_profile"] = profile
    return finish()
