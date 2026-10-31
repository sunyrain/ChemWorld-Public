"""Native steady query: single inlet to a reactive wall.

Original native FV transport and shared root only. No runtime/world ownership,
no material/source qualification, dynamic wall energy, radiation or Soret.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, replace
from functools import partial
from math import fsum, isfinite, sqrt
from traceback import format_exc
from types import MappingProxyType
from typing import Any

import numpy as np

from chemworld.physchem.catalysis_network import (
    CatalyticMechanismSpec,
    catalytic_inventory_key,
    evaluate_catalytic_rates,
    make_catalytic_slot,
)
from chemworld.physchem.catalysis_steady_chart import build_catalytic_steady_chart
from chemworld.physchem.coupled_system import CoupledStateView
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
from chemworld.physchem.specs import ComponentSpec
from chemworld.physchem.surface_import_rates import ImportedSurfaceReaction

MODEL = "impinging-surface-gas-flow-v1"
_INITIAL_FIELDS = {
    "amounts_mol",
    "temperatures_K",
    "spread_rates_per_s",
    "face_mass_fluxes_kg_m2_s",
    "radial_pressure_gradient_Pa_m2",
    "inlet_trace_mole_fractions",
    "wall_trace_mole_fractions",
    "surface_amounts_mol",
}


def _finite(value, name):
    if isinstance(value, bool) or not isinstance(value, int | float) or not isfinite(value):
        raise ValueError(f"{name} must be finite explicit numeric")
    return float(value)


@dataclass(frozen=True)
class GasImpingingSurfaceSpec:
    mesh: DiffusionMesh
    transport: GasTransportSpec | GasTransportPolynomialSpec
    surface: CatalyticMechanismSpec
    pressure_Pa: float
    inlet_temperature_K: float
    wall_temperature_K: float
    inlet_mole_fractions: Mapping[str, float]
    inlet_mass_flux_kg_m2_s: float
    closure_species_id: str
    temperature_bounds_K: tuple[float, float]
    inlet_spread_rate_per_s: float = 0.0
    flux_model: str = "mixture-averaged"
    maximum_mach: float = 0.05

    def __post_init__(self) -> None:
        if (
            not isinstance(self.mesh, DiffusionMesh)
            or self.mesh.geometry != "planar"
            or self.mesh.cells < 2
        ):
            raise ValueError("declare the original planar FV mesh with >=2 cells")
        if not isinstance(self.transport, GasTransportSpec | GasTransportPolynomialSpec):
            raise ValueError("declare native transport")
        if self.transport.thermal_diffusion != "off":
            raise ValueError("surface prototype requires Soret off")
        s = self.surface
        if (
            not isinstance(s, CatalyticMechanismSpec)
            or not s.site_species_complete
            or len(s.site_types) != 1
            or s.reservoir_species
            or set(s.site_densities_mol_m2) != set(s.site_types)
            or any(any(row) for row in s.interaction_matrix_J_mol)
            or any(not isinstance(r, ImportedSurfaceReaction) for r in s.reactions)
            or any(s.species_by_id[k].surface_declaration is None for k in s.adsorbates)
        ):
            raise ValueError(
                "declare one complete imported ideal surface; no reservoirs/interactions"
            )
        ids = (
            self.transport.species_ids
            if isinstance(self.transport, GasTransportPolynomialSpec)
            else tuple(p.component_id for p in self.transport.parameters)
        )
        if set(s.gas_species) != set(ids) or self.closure_species_id not in ids:
            raise ValueError("surface/transport must declare all gas species")
        if any(v.species.charge or v.species.phase not in {"gas", "surface"} for v in s.species):
            raise ValueError("neutral gas/surface only")
        object.__setattr__(
            self,
            "inlet_mole_fractions",
            MappingProxyType(_fractions(self.inlet_mole_fractions, ids)),
        )
        for name in (
            "pressure_Pa",
            "inlet_temperature_K",
            "wall_temperature_K",
            "inlet_mass_flux_kg_m2_s",
        ):
            _positive(getattr(self, name), name)
        if self.pressure_Pa > self.transport.maximum_pressure_Pa:
            raise ValueError("pressure exceeds transport domain")
        if (
            not isinstance(self.temperature_bounds_K, tuple | list)
            or len(self.temperature_bounds_K) != 2
        ):
            raise ValueError("declare exactly two temperature endpoints")
        lo, hi = (_positive(t, "temperature bound") for t in self.temperature_bounds_K)
        if not lo < hi or any(
            not lo <= t <= hi for t in (self.inlet_temperature_K, self.wall_temperature_K)
        ):
            raise ValueError("Tin/Tw outside declared domain")
        if not s.minimum_temperature_K <= self.wall_temperature_K <= s.maximum_temperature_K:
            raise ValueError("Tw outside surface mechanism domain")
        object.__setattr__(self, "temperature_bounds_K", (lo, hi))
        _finite(self.inlet_spread_rate_per_s, "inlet V")
        if self.flux_model not in {"mixture-averaged", "stefan-maxwell"}:
            raise ValueError("unsupported whole-mixture flux")
        if _positive(self.maximum_mach, "maximum Mach") > 0.3:
            raise ValueError("declare low axial-centerline Mach")

    def to_dict(self):
        return {
            "model": MODEL,
            "mesh": {
                "geometry": "planar",
                "faces_m": list(self.mesh.faces_m),
                "area_m2": self.mesh.area_m2,
            },
            "transport": self.transport.to_dict(),
            "surface": self.surface.to_dict(),
            "inlet_mole_fractions": dict(self.inlet_mole_fractions),
            "temperature_bounds_K": list(self.temperature_bounds_K),
            **{
                name: getattr(self, name)
                for name in (
                    "pressure_Pa",
                    "inlet_temperature_K",
                    "wall_temperature_K",
                    "inlet_mass_flux_kg_m2_s",
                    "closure_species_id",
                    "inlet_spread_rate_per_s",
                    "flux_model",
                    "maximum_mach",
                )
            },
        }

    @classmethod
    def from_dict(cls, value):
        if (
            not isinstance(value, Mapping)
            or set(value) != set(cls.__dataclass_fields__) | {"model"}
            or value["model"] != MODEL
        ):
            raise ValueError("unknown impinging-surface codec fields/version")
        if set(value["mesh"]) != {"geometry", "faces_m", "area_m2"}:
            raise ValueError("unknown mesh codec fields")
        data = {
            k: v for k, v in value.items() if k not in {"model", "mesh", "transport", "surface"}
        }
        return cls(
            DiffusionMesh(**value["mesh"]),
            gas_transport_from_dict(value["transport"]),
            CatalyticMechanismSpec.from_dict(value["surface"]),
            **data,
        )


def impinging_surface_initial_profile(
    spec,
    network,
    *,
    temperatures_K,
    mole_fractions,
    surface_coverages,
    spread_rates_per_s=None,
    face_mass_fluxes_kg_m2_s=None,
    radial_pressure_gradient_Pa_m2=0.0,
    inlet_trace_mole_fractions=None,
    wall_trace_mole_fractions=None,
):
    """Caller-owned numerical EOS guess, including actual original surface stock."""
    if not isinstance(spec, GasImpingingSurfaceSpec) or not isinstance(
        network, ReactionNetworkSpec
    ):
        raise ValueError("declare original gas spec/network")
    if len(temperatures_K) != spec.mesh.cells or len(mole_fractions) != spec.mesh.cells:
        raise ValueError("guess must cover every original cell")
    capacities = {k: v * spec.mesh.area_m2 for k, v in spec.surface.site_densities_mol_m2.items()}
    chart = build_catalytic_steady_chart(spec.surface, capacities, surface_coverages)
    amounts = []
    for t, x, volume in zip(temperatures_K, mole_fractions, spec.mesh.cell_volumes_m3, strict=True):
        if (
            not spec.temperature_bounds_K[0]
            <= _positive(t, "T guess")
            <= spec.temperature_bounds_K[1]
        ):
            raise ValueError("T guess outside explicit bounds")
        x = _fractions(x, network.species_ids)
        total = spec.pressure_Pa * float(volume) / (R * t)
        amounts.append({k: total * x[k] for k in network.species_ids})
    v = (
        np.interp(
            spec.mesh.centers_m, (0, spec.mesh.faces_m[-1]), (spec.inlet_spread_rate_per_s, 0.0)
        ).tolist()
        if spread_rates_per_s is None
        else list(spread_rates_per_s)
    )
    g = (
        np.interp(
            spec.mesh.faces_m, (0, spec.mesh.faces_m[-1]), (spec.inlet_mass_flux_kg_m2_s, 0.0)
        ).tolist()
        if face_mass_fluxes_kg_m2_s is None
        else list(face_mass_fluxes_kg_m2_s)
    )
    if len(v) != spec.mesh.cells or len(g) != spec.mesh.cells + 1:
        raise ValueError("V/G guesses must cover original mesh")
    return {
        "amounts_mol": amounts,
        "temperatures_K": list(temperatures_K),
        "spread_rates_per_s": [_finite(z, "V") for z in v],
        "face_mass_fluxes_kg_m2_s": [_finite(z, "G") for z in g],
        "radial_pressure_gradient_Pa_m2": _finite(radial_pressure_gradient_Pa_m2, "Lambda"),
        "inlet_trace_mole_fractions": _fractions(
            spec.inlet_mole_fractions
            if inlet_trace_mole_fractions is None
            else inlet_trace_mole_fractions,
            network.species_ids,
        ),
        "wall_trace_mole_fractions": _fractions(
            mole_fractions[-1] if wall_trace_mole_fractions is None else wall_trace_mole_fractions,
            network.species_ids,
        ),
        "surface_amounts_mol": {
            k: surface_coverages[k] * chart.capacities_mol[i] for i, k in enumerate(chart.species)
        },
    }


class _ImpingingResidual:
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
        self.spec = spec = GasImpingingSurfaceSpec.from_dict(deepcopy(spec.to_dict()))
        self.network = network = ReactionNetworkSpec.from_dict(deepcopy(network.to_dict()))
        self.ids, self.count, self.size = network.species_ids, spec.mesh.cells, len(network.species)
        self.active = [s for s in self.ids if s != spec.closure_species_id]
        self.calls, self.maximum_evaluations, self.progress = 0, maximum_evaluations, progress
        if network.stochastic is not None or any(
            s.phase != "gas" or s.charge for s in network.species
        ):
            raise ValueError(
                "impinging flow requires the original deterministic neutral gas network"
            )
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
        if not isinstance(initial, Mapping) or set(initial) != _INITIAL_FIELDS:
            raise ValueError("impinging initial profile has missing or unknown fields")
        self.surface = spec.surface
        self.surface_masses = {
            k: ComponentSpec(k, value.species.formula).molecular_weight_g_mol / 1000
            for k, value in self.surface.species_by_id.items()
        }
        for name in self.ids:
            original = self.surface.species_by_id[name]
            if (
                network.species[network.species_ids.index(name)] != original.species
                or network.imported_gas_thermochemistry[name].to_dict()
                != original.gas_model.to_dict()
            ):
                raise ValueError(
                    "surface requires identical original gas identities/thermochemistry"
                )
        self.capacities = {
            k: v * spec.mesh.area_m2 for k, v in self.surface.site_densities_mol_m2.items()
        }
        scales = self.surface.coverage_amount_scales_mol(self.capacities)
        if set(initial["surface_amounts_mol"]) != set(self.surface.adsorbates):
            raise ValueError("initial surface stock must cover all original configurations")
        surface_amounts = {
            k: _finite(initial["surface_amounts_mol"][k], "surface stock") for k in scales
        }
        theta = {k: surface_amounts[k] / scales[k] for k in scales}
        self.chart = build_catalytic_steady_chart(self.surface, self.capacities, theta)
        self.surface_slot = make_catalytic_slot(
            "query-wall",
            "steady-query",
            self.surface,
            capacities_mol=self.capacities,
            initial_amounts_mol=surface_amounts,
            fixed=True,
        )
        inlet_reference = self.response(spec.inlet_temperature_K, spec.inlet_mole_fractions)
        wall_reference = self.response(
            spec.wall_temperature_K, initial["wall_trace_mole_fractions"]
        )
        self.masses = dict(inlet_reference.molecular_weights_kg_mol)
        self.feed_y = dict(inlet_reference.mass_fractions)
        for point in (inlet_reference, wall_reference):
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
            raise ValueError("impinging initial profile has missing or unknown fields")
        if (
            any(
                len(initial[name]) != self.count
                for name in ("amounts_mol", "temperatures_K", "spread_rates_per_s")
            )
            or len(initial["face_mass_fluxes_kg_m2_s"]) != self.count + 1
        ):
            raise ValueError("impinging initial profile does not match the actual mesh")
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
            _fractions(initial[side + "_trace_mole_fractions"], self.ids)
            for side in ("inlet", "wall")
        ]
        self.n_stop = self.count * self.size
        self.t_start, self.v_start = self.n_stop, self.n_stop + self.count
        self.g_start = self.n_stop + 2 * self.count
        self.lambda_col = self.g_start + self.count + 1
        self.trace_start = self.lambda_col + 1
        self.q_start = self.trace_start + 2 * self.size
        self.initial = np.asarray(
            [
                *amounts,
                *t,
                *v,
                *g,
                lam,
                *(traces[0][s] for s in self.ids),
                *(traces[1][s] for s in self.ids),
                *np.zeros(self.chart.rank),
            ]
        )
        tref = 0.5 * (inlet_reference.temperature_K + wall_reference.temperature_K)
        gref = max(spec.inlet_mass_flux_kg_m2_s, 0.0)
        rho = 0.5 * (inlet_reference.density_kg_m3 + wall_reference.density_kg_m3)
        vref = max(
            abs(spec.inlet_spread_rate_per_s),
            abs(0.0),
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
            for p in (inlet_reference, wall_reference)
            for s in self.ids
        )
        momentum_scale = max(
            mass_scale * vref,
            max(inlet_reference.viscosity_Pa_s, wall_reference.viscosity_Pa_s)
            * vref
            * spec.mesh.area_m2
            / spec.mesh.faces_m[-1],
        )
        for value in (tref, gref, vref, lamref, mass_scale, heat_scale, momentum_scale):
            _positive(value, "impinging residual scale")
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
        for side in ("inlet", "wall"):
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
        for side in ("inlet", "wall"):
            self.equations.extend(
                ResidualEquation(
                    f"feed:{side}:{s}", "mol/s", mass_scale / min(self.masses.values())
                )
                for s in self.active
            )
        self.equations.extend(
            ResidualEquation(f"trace-sum:{side}", "1", 1) for side in ("inlet", "wall")
        )
        self.equations.extend(
            ResidualEquation(f"G-boundary:{side}", "kg/s", mass_scale) for side in ("inlet", "wall")
        )
        self.variables.extend(
            ResidualVariable(f"surface-q:{i}", "1", 1, role="algebraic")
            for i in range(self.chart.rank)
        )
        self.equations.extend(
            ResidualEquation(
                f"surface-steady:{i}",
                "1/s",
                mass_scale / min(self.masses.values()) / min(self.chart.capacities_mol),
            )
            for i in range(self.chart.rank)
        )

    def wall_rates(self, coverages, wall_x, *, probe_volume_L=1.0):
        """Original mol/s kernel on a transient view; area is already in site capacity."""
        probe_volume_L = _positive(probe_volume_L, "probe volume")
        amounts = {
            k: float(coverages[i]) * self.chart.capacities_mol[i]
            for i, k in enumerate(self.chart.species)
        }
        current = replace(self.surface_slot, values={**self.surface_slot.values, **amounts})
        gas = {
            k: self.spec.pressure_Pa
            * (probe_volume_L / 1000)
            * wall_x[k]
            / (R * self.spec.wall_temperature_K)
            for k in self.ids
        }
        gas.update({catalytic_inventory_key(current, k): value for k, value in amounts.items()})
        view = CoupledStateView(gas, self.spec.wall_temperature_K, probe_volume_L)
        rates = evaluate_catalytic_rates(view, current, self.surface, gas_volume_L=probe_volume_L)
        if any(not isfinite(v) for v in rates.amount_rates_mol_s.values()):
            raise ResidualDomainError("surface kernel returned nonfinite original rates")
        return rates, current

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
            raise ResidualDomainError("impinging transport trial: " + str(exc)) from exc

    def decode(self, vector):
        value = np.asarray(vector)
        if value.shape != self.initial.shape or not np.all(np.isfinite(value)):
            raise ResidualDomainError(
                "impinging trial has invalid raw shape or nonfinite coordinates"
            )
        n = value[: self.n_stop].reshape((self.count, self.size))
        t = value[self.t_start : self.v_start]
        v = value[self.v_start : self.g_start]
        g = value[self.g_start : self.lambda_col]
        lam = float(value[self.lambda_col])
        z = value[self.trace_start : self.q_start].reshape((2, self.size))
        theta = self.chart.reconstruct(value[self.q_start :])
        if np.any(theta < 0):
            raise ResidualDomainError("surface affine chart reconstructed negative original stock")
        totals, ztotals = [fsum(row) for row in n], [fsum(row) for row in z]
        if np.any(n < 0) or np.any(z < 0) or min([*totals, *ztotals]) <= 0:
            raise ResidualDomainError("impinging trial contains negative or empty material")
        if np.any(t < self.spec.temperature_bounds_K[0]) or np.any(
            t > self.spec.temperature_bounds_K[1]
        ):
            raise ResidualDomainError("impinging trial temperature outside declared domain")
        x = [
            dict(zip(self.ids, map(float, row / total), strict=True))
            for row, total in zip(n, totals, strict=True)
        ]
        traces = [
            dict(zip(self.ids, map(float, row / total), strict=True))
            for row, total in zip(z, ztotals, strict=True)
        ]
        return n, t, v, g, lam, x, traces, ztotals, theta

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
        surface_rows = set(range(len(self.equations) - self.chart.rank, len(self.equations)))
        wall_rows = feed_rows(1)
        for column in range(self.q_start, len(self.variables)):
            columns[column] |= surface_rows | wall_rows
        for column in range(self.trace_start + self.size, self.q_start):
            columns[column] |= surface_rows
        return tuple(tuple(sorted(rows)) for rows in columns)

    def evaluate(self, vector, *, retain_details=True):
        if type(retain_details) is not bool:
            raise ValueError("retain_details must be explicitly boolean")
        self.calls += 1
        if self.calls > self.maximum_evaluations:
            raise ResidualDomainError("impinging residual evaluation budget exceeded")
        if self.progress is not None:
            self.progress(
                {
                    "event": "impinging-residual",
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
            raise ResidualDomainError("impinging finite trial domain: " + str(exc)) from exc

    def _evaluate(self, vector, *, retain_details=True):
        n, t, v, g, lam, x, traces, ztotals, theta = self.decode(vector)
        spec, ids, area = self.spec, self.ids, self.spec.mesh.area_m2
        points = [self.response(float(ti), row) for ti, row in zip(t, x, strict=True)]
        ends = [
            self.response(spec.inlet_temperature_K, traces[0]),
            self.response(spec.wall_temperature_K, traces[1]),
        ]
        all_points = [ends[0], *points, ends[1]]
        coords = [0.0, *spec.mesh.centers_m, spec.mesh.faces_m[-1]]
        spread = [spec.inlet_spread_rate_per_s, *v, 0.0]
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
                # The inlet and reactive wall each use their actual fixed T/V boundary values.
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
                    species_thermo=self.thermo_views,
                )
            except (ValueError, ArithmeticError) as exc:
                raise ResidualDomainError("impinging chemistry trial: " + str(exc)) from exc
            if any(not isfinite(value) for value in omega.values()):
                raise ResidualDomainError("impinging trial chemistry produced nonfinite rates")
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
        wall_rates, wall_slot = self.wall_rates(theta, traces[1])
        rgas = {s: wall_rates.amount_rates_mol_s[s] for s in ids}
        rsurface = {
            s: wall_rates.amount_rates_mol_s[catalytic_inventory_key(wall_slot, s)]
            for s in self.chart.species
        }
        coverage_rates = np.asarray(
            [rsurface[s] / self.chart.capacities_mol[i] for i, s in enumerate(self.chart.species)]
        )
        surface_residual = self.chart.reduce_rates(coverage_rates)
        boundary_species = [
            {
                s: fs[0][s] - area * spec.inlet_mass_flux_kg_m2_s * self.feed_y[s] / self.masses[s]
                for s in ids
            },
            {s: fs[-1][s] + rgas[s] for s in ids},
        ]
        g_boundary = [area * (float(g[0]) - spec.inlet_mass_flux_kg_m2_s), area * float(g[-1])]
        residual = [
            value for i in range(self.count) for value in [*(sr[i][s] for s in self.active), eos[i]]
        ]
        residual.extend([*heat, *momentum, *continuity])
        residual.extend(boundary_species[side][s] for side in (0, 1) for s in self.active)
        residual.extend([ztotals[0] - 1, ztotals[1] - 1, *g_boundary])
        residual.extend(surface_residual)
        if not retain_details:
            values = np.asarray(residual)
            if not np.all(np.isfinite(values)):
                raise ResidualDomainError("impinging residual exceeded its finite numeric domain")
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
        surface_theta = dict(zip(self.chart.species, map(float, theta), strict=True))
        surface_amounts = {
            s: float(theta[i]) * self.chart.capacities_mol[i]
            for i, s in enumerate(self.chart.species)
        }
        original_theta = dict(zip(self.chart.species, self.chart.initial_coverages, strict=True))
        surface_site_residuals = {
            site: fsum(
                surface_amounts[s] * self.surface.site_occupancies[s]
                for s in self.chart.species
                if self.surface.adsorbate_sites[s] == site
            )
            - cap
            for site, cap in self.capacities.items()
        }
        surface_elements = sorted(
            {e for species in self.surface.species for e in species.species.composition}
        )
        all_wall_rates = {**rgas, **rsurface}
        wall_chemical_elements = {
            e: fsum(
                value * self.surface.species_by_id[s].species.composition.get(e, 0)
                for s, value in all_wall_rates.items()
            )
            for e in surface_elements
        }
        details = {
            "surface_coverages": surface_theta,
            "surface_amounts_mol": surface_amounts,
            "surface_site_capacities_mol": dict(self.capacities),
            "surface_site_residuals_mol_sites": surface_site_residuals,
            "surface_original_conserved_totals_mol": self.chart.totals(original_theta).tolist(),
            "surface_actual_conserved_totals_mol": self.chart.totals(surface_theta).tolist(),
            "wall_chemical_element_residuals_mol_atoms_s": wall_chemical_elements,
            "wall_chemical_mass_residual_kg_s": fsum(
                value * self.surface_masses[s] for s, value in all_wall_rates.items()
            ),
            "surface_rates_mol_s": rsurface,
            "surface_coverage_rates_per_s": coverage_rates.tolist(),
            "surface_reduced_residual_per_s": surface_residual.tolist(),
            "surface_chart": self.chart.to_dict(),
            "surface_gas_rates_mol_s": rgas,
            "surface_gas_rates_mol_m2_s": {s: rgas[s] / area for s in ids},
            "surface_forward_rates_mol_s": dict(wall_rates.forward_rates_mol_s),
            "surface_reverse_rates_mol_s": dict(wall_rates.reverse_rates_mol_s),
            "wall_external_compensation_W": -fh[-1],
            "surface_energy_policy": "one formation-enthalpy Fh; no second reaction heat source",
            "surface_probe": {"volume_L": 1.0, "role": "transient rate view; no world/clock"},
            "amounts_mol": [dict(zip(ids, map(float, row), strict=True)) for row in n],
            "temperatures_K": list(map(float, t)),
            "spread_rates_per_s": list(map(float, v)),
            "face_mass_fluxes_kg_m2_s": list(map(float, g)),
            "radial_pressure_gradient_Pa_m2": lam,
            "inlet_trace_mole_fractions": traces[0],
            "wall_trace_mole_fractions": traces[1],
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
            raise ResidualDomainError("impinging residual exceeded its finite numeric domain")
        return values, details


def solve_impinging_surface_gas_flow(
    spec: GasImpingingSurfaceSpec,
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
    surface_tolerance_per_s: float = 1e-8,
    jacobian_strategy: str = "dense",
    progress: Any = None,
    thermo_memo_max_entries: int | None = None,
    residual_only_trials: bool = False,
):
    """Fresh-check every original species and radial ledger; no physical fallback.

    Explicit opt-ins only reduce numerical work. A positive memo cap shares
    private NASA7 views across polynomial transport, inlet/wall preflight, cell
    chemistry, Cp scales and final Mach. No network cached property is replaced.
    Residual-only trials omit diagnostic serialization and ledgers outside F;
    the accepted fresh evaluator still constructs the complete original profile
    and all twenty-five physical checks. None/False retain the full-detail path.
    """
    if not isinstance(spec, GasImpingingSurfaceSpec) or not isinstance(
        network, ReactionNetworkSpec
    ):
        raise ValueError("impinging query requires its original spec and network")
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
        surface_tolerance_per_s,
    ):
        _positive(value, "impinging query tolerance")
    if any(type(v) is not int or v < 1 for v in (maximum_unknowns, maximum_evaluations)):
        raise ValueError("impinging numerical budgets must be positive integers")
    assembly = _ImpingingResidual(
        spec,
        network,
        initial_profile,
        maximum_evaluations,
        progress,
        thermo_memo_max_entries=thermo_memo_max_entries,
    )
    if len(assembly.variables) > maximum_unknowns:
        raise ValueError("impinging unknown count exceeds its explicit budget")
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
        "schema": "impinging-gas-flow-query-v1",
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
        "scope": "steady ideal single-surface query; no world/clock, "
        "dynamic wall, deposition, Soret, material/source or mesh qualification",
        "boundary_policy": "one Tin/total incoming flux; reactive wall Tw, Gwall=0/Vwall=0; "
        "full original q-chart; Lambda eigenvalue; no artificial phase",
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
    except Exception as exc:
        report.update(
            failure_reason="fresh_check_failed: " + str(exc),
            residual_evaluations=assembly.calls,
            fresh_exception={
                "type": f"{type(exc).__module__}.{type(exc).__qualname__}",
                "message": str(exc),
                "traceback": format_exc(),
            },
        )
        return finish()
    spec, network = assembly.spec, assembly.network
    try:
        mach = []
        states = []
        for i, (t, n) in enumerate(
            zip(profile["temperatures_K"], profile["amounts_mol"], strict=True)
        ):
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
                side = "inlet" if face == 0 else "wall"
                t = getattr(spec, side + "_temperature_K")
                x = profile[side + "_trace_mole_fractions"]
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
                raise ValueError("impinging ideal-gas sound speed requires positive Cv")
            mean_mass = fsum(x[s] * assembly.masses[s] for s in assembly.ids)
            mach.append(abs(g) / rho / sqrt(cp / (cp - R) * R * t / mean_mass))
        profile["axial_centerline_Mach_cells_and_faces"] = mach
        checks = {
            "all_species_including_omitted_balance": max(
                abs(v) for row in profile["species_balance_residuals_mol_s"] for v in row.values()
            )
            <= species_tolerance_mol_s,
            "inlet_and_reactive_wall_all_species": max(
                abs(v) for row in profile["inlet_species_residuals_mol_s"] for v in row.values()
            )
            <= species_tolerance_mol_s,
            "EOS": max(map(abs, profile["EOS_residuals_Pa"])) <= pressure_tolerance_Pa,
            "both_trace_normalizations": max(
                abs(v - 1) for v in profile["inlet_trace_coordinate_sums"]
            )
            <= root_policy.tolerance,
            "energy": max(map(abs, profile["energy_balance_residuals_W"])) <= energy_tolerance_W,
            "radial_momentum": max(map(abs, profile["momentum_balance_residuals_kg_s2"]))
            <= momentum_tolerance_kg_s2,
            "continuity": max(map(abs, profile["continuity_residuals_kg_s"]))
            <= mass_tolerance_kg_s,
            "inlet_and_impermeable_wall_G": max(map(abs, profile["G_boundary_residuals_kg_s"]))
            <= mass_tolerance_kg_s
            and profile["face_mass_fluxes_kg_m2_s"][0] > 0
            and abs(spec.mesh.area_m2 * profile["face_mass_fluxes_kg_m2_s"][-1])
            <= mass_tolerance_kg_s,
            "all_original_surface_rates": max(map(abs, profile["surface_rates_mol_s"].values()))
            <= species_tolerance_mol_s,
            "all_original_surface_coverage_rates": max(
                map(abs, profile["surface_coverage_rates_per_s"])
            )
            <= surface_tolerance_per_s,
            "all_original_surface_pools": assembly.chart.same_class(profile["surface_coverages"]),
            "surface_nonnegative_stock": min(profile["surface_amounts_mol"].values()) >= 0,
            "all_original_sites": all(
                abs(value) <= 1e-10 * assembly.capacities[k]
                for k, value in profile["surface_site_residuals_mol_sites"].items()
            ),
            "wall_chemical_elements": max(
                map(abs, profile["wall_chemical_element_residuals_mol_atoms_s"].values())
            )
            <= element_tolerance_mol_s,
            "wall_chemical_mass": abs(profile["wall_chemical_mass_residual_kg_s"])
            <= mass_tolerance_kg_s,
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
            "global_enthalpy": abs(profile["global_enthalpy_balance_residual_W"])
            <= energy_tolerance_W,
            "axial_centerline_low_Mach": max(mach) <= spec.maximum_mach,
        }
        checks = {name: bool(value) for name, value in checks.items()}
        report.update(
            success=all(checks.values()),
            physical_checks=checks,
            failure_reason="" if all(checks.values()) else "fresh_physical_balance_failed",
            residual_evaluations=assembly.calls,
        )
        report["profile" if report["success"] else "candidate_profile"] = profile
    except Exception as exc:
        report.update(
            success=False,
            failure_reason="fresh_check_exception",
            candidate_profile=profile,
            fresh_exception={
                "type": f"{type(exc).__module__}.{type(exc).__qualname__}",
                "message": str(exc),
                "traceback": format_exc(),
            },
            residual_evaluations=assembly.calls,
        )
    return finish()
