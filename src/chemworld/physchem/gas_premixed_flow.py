"""Conservative planar premixed boundary-value queries on the shared residual solver.

This is a pure analysis problem: finite-volume n/T profiles do not advance any
world, consume reservoirs or create an experimental clock. Transport and rates
come from the original network. Fixed-flow solutions and a free-flow mass-flux
eigenvalue have different explicit declarations; a converged cold branch is
never automatically called a flame.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, replace
from functools import partial
from math import fsum, isfinite, sqrt
from types import MappingProxyType
from typing import Any

import numpy as np

from chemworld.physchem.finite_volume import DiffusionMesh
from chemworld.physchem.gas_polynomial_transport import (
    GasTransportPolynomialSpec,
    _bound_polynomial_response,
)
from chemworld.physchem.gas_soret_transport import midpoint_temperature_log_gradient
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

MODEL = "premixed-gas-flow-v1"


def _face_mole_gradients(left, right, distance, closure):
    gradients = {s: (right[s] - left[s]) / distance for s in left}
    raw_closure = gradients[closure]
    gradients[closure] = -fsum(v for s, v in gradients.items() if s != closure)
    return gradients, {
        "raw_sum_per_m": fsum([*(v for s, v in gradients.items() if s != closure), raw_closure]),
        "closure_correction_per_m": gradients[closure] - raw_closure,
    }


def _positive(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{name} must be a positive finite number")
    result = float(value)
    if not isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be a positive finite number")
    return result


def _fractions(value: Mapping[str, float], ids: Sequence[str]) -> dict[str, float]:
    if not isinstance(value, Mapping) or set(value) != set(ids):
        raise ValueError("mole fractions must cover exactly the original species")
    if any(isinstance(v, bool) or not isinstance(v, int | float) for v in value.values()):
        raise ValueError("mole fractions must be finite nonnegative numbers")
    row = {s: float(value[s]) for s in ids}
    if any(not isfinite(v) or v < 0 for v in row.values()) or abs(fsum(row.values()) - 1) > 1e-12:
        raise ValueError("mole fractions must be nonnegative and sum to one")
    return row


@dataclass(frozen=True)
class GasFlamePhaseSpec:
    """Temperature at a physical location, unchanged by subsequent mesh refinement."""

    position_m: float
    temperature_K: float

    def __post_init__(self) -> None:
        _positive(self.position_m, "phase position")
        _positive(self.temperature_K, "phase temperature")


@dataclass(frozen=True)
class GasPremixedFlowSpec:
    mesh: DiffusionMesh
    transport: GasTransportSpec | GasTransportPolynomialSpec
    pressure_Pa: float
    inlet_temperature_K: float
    inlet_mole_fractions: Mapping[str, float]
    closure_species_id: str
    temperature_bounds_K: tuple[float, float]
    energy_equation: bool
    mass_flux_kg_m2_s: float | None
    phase: GasFlamePhaseSpec | None = None
    flux_model: str = "mixture-averaged"
    maximum_mach: float = 0.05
    advection_scheme: str = "upwind"

    def __post_init__(self) -> None:
        if not isinstance(self.mesh, DiffusionMesh) or self.mesh.geometry != "planar":
            raise ValueError("premixed flow requires the original planar DiffusionMesh")
        if self.mesh.cells < 2:
            raise ValueError("premixed flow needs at least two control volumes")
        if not isinstance(self.transport, GasTransportSpec | GasTransportPolynomialSpec):
            raise ValueError("declare a supported original gas transport model")
        ids = (
            self.transport.species_ids
            if isinstance(self.transport, GasTransportPolynomialSpec)
            else tuple(p.component_id for p in self.transport.parameters)
        )
        object.__setattr__(
            self,
            "inlet_mole_fractions",
            MappingProxyType(_fractions(self.inlet_mole_fractions, ids)),
        )
        if self.closure_species_id not in ids:
            raise ValueError("closure species must identify an original species")
        _positive(self.pressure_Pa, "pressure")
        if self.pressure_Pa > self.transport.maximum_pressure_Pa:
            raise ValueError("premixed pressure exceeds the explicit transport domain")
        _positive(self.inlet_temperature_K, "inlet temperature")
        if (
            not isinstance(self.temperature_bounds_K, list | tuple)
            or len(self.temperature_bounds_K) != 2
        ):
            raise ValueError("temperature bounds need exactly two endpoints")
        lo, hi = (_positive(v, "temperature bound") for v in self.temperature_bounds_K)
        if lo >= hi or not lo <= self.inlet_temperature_K <= hi:
            raise ValueError("temperature interval must contain the inlet temperature")
        object.__setattr__(self, "temperature_bounds_K", (lo, hi))
        if type(self.energy_equation) is not bool:
            raise ValueError("energy_equation must be explicitly boolean")
        if self.flux_model not in {"mixture-averaged", "stefan-maxwell"}:
            raise ValueError("unsupported whole-mixture flux model")
        if self.transport.thermal_diffusion != "off" and self.flux_model != "mixture-averaged":
            raise ValueError("mixture Soret requires mixture-averaged premixed diffusion")
        if self.advection_scheme not in {"upwind", "linear-upwind"}:
            raise ValueError("unsupported explicit premixed advection scheme")
        if _positive(self.maximum_mach, "maximum Mach") > 0.3:
            raise ValueError("constant-pressure flow requires a declared low-Mach bound <=0.3")
        if self.mass_flux_kg_m2_s is None:
            if not self.energy_equation or not isinstance(self.phase, GasFlamePhaseSpec):
                raise ValueError("free flow needs an energy equation and explicit phase condition")
            if not (
                self.mesh.centers_m[0] < self.phase.position_m < self.mesh.centers_m[-1]
                and lo < self.phase.temperature_K < hi
            ):
                raise ValueError("phase condition must be inside the mesh and thermal domain")
        else:
            _positive(self.mass_flux_kg_m2_s, "mass flux")
            if self.phase is not None:
                raise ValueError("fixed mass flux and free-flow phase condition are exclusive")

    def to_dict(self):
        return {
            "model": MODEL,
            "mesh": {
                "geometry": "planar",
                "faces_m": list(self.mesh.faces_m),
                "area_m2": self.mesh.area_m2,
            },
            "transport": self.transport.to_dict(),
            "pressure_Pa": self.pressure_Pa,
            "inlet_temperature_K": self.inlet_temperature_K,
            "inlet_mole_fractions": dict(self.inlet_mole_fractions),
            "closure_species_id": self.closure_species_id,
            "temperature_bounds_K": list(self.temperature_bounds_K),
            "energy_equation": self.energy_equation,
            "mass_flux_kg_m2_s": self.mass_flux_kg_m2_s,
            "phase": None
            if self.phase is None
            else {"position_m": self.phase.position_m, "temperature_K": self.phase.temperature_K},
            "flux_model": self.flux_model,
            "maximum_mach": self.maximum_mach,
            "advection_scheme": self.advection_scheme,
        }

    @classmethod
    def from_dict(cls, value):
        expected = {
            "model",
            "mesh",
            "transport",
            "pressure_Pa",
            "inlet_temperature_K",
            "inlet_mole_fractions",
            "closure_species_id",
            "temperature_bounds_K",
            "energy_equation",
            "mass_flux_kg_m2_s",
            "phase",
            "flux_model",
            "maximum_mach",
            "advection_scheme",
        }
        if not isinstance(value, Mapping) or set(value) != expected or value["model"] != MODEL:
            raise ValueError("unknown premixed-flow codec or fields")
        data = {k: v for k, v in value.items() if k not in {"model", "mesh", "transport", "phase"}}
        return cls(
            mesh=DiffusionMesh(**value["mesh"]),
            transport=gas_transport_from_dict(value["transport"]),
            phase=None if value["phase"] is None else GasFlamePhaseSpec(**value["phase"]),
            **data,
        )


def premixed_flow_initial_profile(
    spec: GasPremixedFlowSpec,
    network: ReactionNetworkSpec,
    *,
    temperatures_K: Sequence[float],
    mole_fractions: Sequence[Mapping[str, float]],
    inlet_trace_mole_fractions: Mapping[str, float] | None = None,
    mass_flux_kg_m2_s: float | None = None,
) -> dict[str, Any]:
    """Construct real EOS amounts as an explicit caller-owned initial guess."""
    if len(temperatures_K) != spec.mesh.cells or len(mole_fractions) != spec.mesh.cells:
        raise ValueError("initial profiles must cover every original control volume")
    amounts = []
    for temperature, fractions, volume in zip(
        temperatures_K, mole_fractions, spec.mesh.cell_volumes_m3, strict=True
    ):
        t = _positive(temperature, "profile temperature")
        if not spec.temperature_bounds_K[0] <= t <= spec.temperature_bounds_K[1]:
            raise ValueError("profile temperature outside declared domain")
        x = _fractions(fractions, network.species_ids)
        total = spec.pressure_Pa * float(volume) / (R * t)
        amounts.append({s: total * x[s] for s in network.species_ids})
    trace = (
        spec.inlet_mole_fractions
        if inlet_trace_mole_fractions is None
        else inlet_trace_mole_fractions
    )
    mass = spec.mass_flux_kg_m2_s if mass_flux_kg_m2_s is None else mass_flux_kg_m2_s
    _positive(mass, "initial mass flux")
    if spec.mass_flux_kg_m2_s is not None and mass != spec.mass_flux_kg_m2_s:
        raise ValueError("initial mass flux differs from the fixed-flow declaration")
    return {
        "amounts_mol": amounts,
        "temperatures_K": list(temperatures_K),
        "inlet_trace_mole_fractions": _fractions(trace, network.species_ids),
        "mass_flux_kg_m2_s": mass,
    }


class _PremixedResidual:
    """One local residual evaluation; no alternate physical state or integrator."""

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
        # Query-local material ownership: progress callbacks cannot change the
        # chemistry while transport silently retains an older cached binding.
        network = ReactionNetworkSpec.from_dict(deepcopy(network.to_dict()))
        self.spec, self.network = spec, network
        self.ids = network.species_ids
        self.thermo_memo = None
        self.thermo_views = None
        if thermo_memo_max_entries is not None:
            # Explicit opt-in scope: the existing private polynomial binding.
            # Do not replace a network cached_property to intercept other consumers.
            if not isinstance(spec.transport, GasTransportPolynomialSpec):
                raise ValueError("query thermo memo currently requires polynomial transport")
            from chemworld.physchem.gas_query_thermo import GasQueryThermoMemo

            self.thermo_memo = GasQueryThermoMemo(
                network.imported_gas_thermochemistry,
                maximum_states=thermo_memo_max_entries,
            )
            self.thermo_views = self.thermo_memo.views
        self.bound_transport_response = None
        if isinstance(spec.transport, GasTransportPolynomialSpec):
            spec.transport.validate_network(network)
            self.bound_transport_response = partial(
                _bound_polynomial_response,
                spec.transport,
                self.ids,
                MappingProxyType(network.imported_gas_thermochemistry)
                if self.thermo_views is None
                else self.thermo_views,
            )
        self.active = [s for s in self.ids if s != spec.closure_species_id]
        self.count, self.size = spec.mesh.cells, len(self.ids)
        self.calls, self.maximum_evaluations, self.progress = 0, maximum_evaluations, progress
        self.nominal_mass_flux = _positive(initial["mass_flux_kg_m2_s"], "initial mass flux")
        if set(initial) != {
            "amounts_mol",
            "temperatures_K",
            "inlet_trace_mole_fractions",
            "mass_flux_kg_m2_s",
        }:
            raise ValueError("initial profile has missing or unknown fields")
        if network.stochastic is not None or any(
            s.phase != "gas" or s.charge for s in network.species
        ):
            raise ValueError("premixed flow requires a deterministic neutral gas network")
        if any(network.element_balance_residuals().values()) or network.charge_balance_residuals():
            raise ValueError("premixed reaction stoichiometry must conserve atoms and charge")
        if set(network.imported_gas_thermochemistry) != set(self.ids):
            raise ValueError("premixed flow requires all original gas thermochemistry")
        lo, hi = spec.temperature_bounds_K
        for model in network.imported_gas_thermochemistry.values():
            if lo < model.min_temperature_K or hi > model.max_temperature_K:
                raise ValueError("thermal bounds exceed original thermochemistry")
        if isinstance(spec.transport, GasTransportPolynomialSpec) and (
            lo < spec.transport.fit_temperature_range_K[0]
            or hi > spec.transport.fit_temperature_range_K[1]
        ):
            raise ValueError("thermal bounds exceed source-fit transport domain")
        inlet = self.response(spec.inlet_temperature_K, spec.inlet_mole_fractions)
        self.masses = dict(inlet.molecular_weights_kg_mol)
        self.inlet_y = dict(inlet.mass_fractions)
        # Resolve invalid rate declarations before nonlinear trial errors are
        # translated to the shared domain-error protocol.
        network.amount_derivatives(
            {
                s: inlet.molar_density_mol_m3
                * spec.mesh.cell_volumes_m3[0]
                * spec.inlet_mole_fractions[s]
                for s in self.ids
            },
            volume_L=float(spec.mesh.cell_volumes_m3[0]) * 1000,
            temperature_K=spec.inlet_temperature_K,
            **({"species_thermo": self.thermo_views} if self.thermo_views is not None else {}),
        )
        self.amount_scales = (
            spec.pressure_Pa * spec.mesh.cell_volumes_m3 / (R * spec.inlet_temperature_K)
        )
        self.flow_scale = self.nominal_mass_flux * spec.mesh.area_m2 / min(self.masses.values())
        _positive(self.flow_scale, "species residual scale")
        self.heat_scale = max(
            1.0,
            self.nominal_mass_flux
            * spec.mesh.area_m2
            * max(
                max(
                    abs(inlet.partial_enthalpies_J_mol[s] / self.masses[s]),
                    (
                        network.imported_gas_thermochemistry
                        if self.thermo_views is None
                        else self.thermo_views
                    )[s]
                    .evaluate(spec.inlet_temperature_K)
                    .cp_J_mol_K
                    * spec.inlet_temperature_K
                    / self.masses[s],
                )
                for s in self.ids
            ),
        )
        self.variables, self.equations = [], []
        for cell in range(self.count):
            self.variables.extend(
                ResidualVariable(
                    f"n:{cell}:{s}",
                    "mol",
                    float(self.amount_scales[cell]),
                    role="algebraic",
                    lower=0.0,
                )
                for s in self.ids
            )
            self.equations.extend(
                ResidualEquation(f"species:{cell}:{s}", "mol/s", self.flow_scale)
                for s in self.active
            )
            self.equations.append(ResidualEquation(f"EOS:{cell}", "Pa", spec.pressure_Pa))
        self.variables.extend(
            ResidualVariable(
                f"T:{cell}", "K", spec.inlet_temperature_K, role="algebraic", lower=lo, upper=hi
            )
            for cell in range(self.count)
        )
        self.equations.extend(
            ResidualEquation(
                f"energy:{cell}" if spec.energy_equation else f"fixed-T:{cell}",
                "W" if spec.energy_equation else "K",
                self.heat_scale if spec.energy_equation else spec.inlet_temperature_K,
            )
            for cell in range(self.count)
        )
        self.variables.extend(
            ResidualVariable(f"inlet-trace:{s}", "1", 1.0, role="algebraic", lower=0.0, upper=1.0)
            for s in self.ids
        )
        self.equations.extend(
            ResidualEquation("inlet-flux:" + s, "mol/s", self.flow_scale) for s in self.active
        )
        self.equations.append(ResidualEquation("inlet-trace-sum", "1", 1.0))
        if spec.mass_flux_kg_m2_s is None:
            self.variables.append(
                ResidualVariable(
                    "mass-flux", "kg/(m^2*s)", self.nominal_mass_flux, role="algebraic", lower=0.0
                )
            )
            self.equations.append(
                ResidualEquation("phase-temperature", "K", spec.inlet_temperature_K)
            )
        if (
            len(initial["amounts_mol"]) != self.count
            or len(initial["temperatures_K"]) != self.count
        ):
            raise ValueError("initial profile must cover every original cell")
        amounts = []
        for row in initial["amounts_mol"]:
            if (
                not isinstance(row, Mapping)
                or set(row) != set(self.ids)
                or any(
                    isinstance(v, bool)
                    or not isinstance(v, int | float)
                    or not isfinite(v)
                    or v < 0
                    for v in row.values()
                )
            ):
                raise ValueError("initial original amounts must be finite nonnegative and complete")
            amounts.extend(float(row[s]) for s in self.ids)
        for value in initial["temperatures_K"]:
            _positive(value, "profile temperature")
        trace = _fractions(initial["inlet_trace_mole_fractions"], self.ids)
        self.initial = np.asarray(
            [
                *amounts,
                *initial["temperatures_K"],
                *(trace[s] for s in self.ids),
                *([self.nominal_mass_flux] if spec.mass_flux_kg_m2_s is None else []),
            ]
        )
        if spec.mass_flux_kg_m2_s is not None and self.nominal_mass_flux != spec.mass_flux_kg_m2_s:
            raise ValueError("initial mass flux differs from fixed-flow declaration")

    def response(self, t, x):
        try:
            if self.bound_transport_response is not None:
                return self.bound_transport_response(
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
            raise ResidualDomainError("premixed transport trial: " + str(exc)) from exc

    def state_sparsity(self):
        """Conservative physical stencil, including trace and physical phase rows."""
        columns = [set() for _ in self.variables]
        count, size = self.count, self.size
        trace_start = count * (size + 1)

        def cell_columns(cell):
            return [*range(cell * size, (cell + 1) * size), count * size + cell]

        all_species_rows = []
        for cell in range(count):
            species_rows = list(range(cell * size, (cell + 1) * size - 1))
            all_species_rows.extend(species_rows)
            energy_row = count * size + cell
            dependent_rows = species_rows + ([energy_row] if self.spec.energy_equation else [])
            reach = 2 if self.spec.advection_scheme == "linear-upwind" else 1
            for neighbor in range(max(0, cell - reach), min(count, cell + 2)):
                for column in cell_columns(neighbor):
                    columns[column].update(dependent_rows)
            for column in cell_columns(cell):
                columns[column].add((cell + 1) * size - 1)  # Original EOS.
            if not self.spec.energy_equation:
                columns[count * size + cell].add(energy_row)
        inlet_rows = list(range(trace_start, trace_start + size - 1))
        for column in cell_columns(0):
            columns[column].update(inlet_rows)
        for column in range(trace_start, trace_start + size):
            columns[column].update(inlet_rows)
            columns[column].add(trace_start + size - 1)
            columns[column].update(range(size - 1))
            if self.spec.energy_equation:
                columns[column].add(count * size)
        if self.spec.mass_flux_kg_m2_s is None:
            columns[-1].update([*all_species_rows, *inlet_rows])
            if self.spec.energy_equation:
                columns[-1].update(range(count * size, count * (size + 1)))
            right = int(np.searchsorted(self.spec.mesh.centers_m, self.spec.phase.position_m))
            for cell in {max(0, min(count - 1, right - 1)), min(count - 1, right)}:
                columns[count * size + cell].add(trace_start + size)
        return tuple(tuple(sorted(rows)) for rows in columns)

    def decode(self, vector):
        stop = self.count * self.size
        amounts = np.asarray(vector[:stop]).reshape((self.count, self.size))
        temperatures = np.asarray(vector[stop : stop + self.count])
        trace = np.asarray(vector[stop + self.count : stop + self.count + self.size])
        mass = (
            float(vector[-1])
            if self.spec.mass_flux_kg_m2_s is None
            else self.spec.mass_flux_kg_m2_s
        )
        totals = np.asarray([fsum(row) for row in amounts])
        trace_sum = fsum(trace)
        if (
            np.any(amounts < 0)
            or np.any(totals <= 0)
            or np.any(trace < 0)
            or trace_sum <= 0
            or mass <= 0
        ):
            raise ResidualDomainError(
                "premixed trial has negative/empty material or nonpositive flow"
            )
        x = [
            dict(zip(self.ids, map(float, row / total), strict=True))
            for row, total in zip(amounts, totals, strict=True)
        ]
        # z/sum(z) is only the trial coordinate map. Its independent normalization
        # residual must pass before accepting any inlet trace.
        trace_x = dict(zip(self.ids, map(float, trace / trace_sum), strict=True))
        return amounts, temperatures, x, trace_x, trace_sum, mass

    def evaluate(self, vector, *, retain_details=True):
        if type(retain_details) is not bool:
            raise ValueError("retain_details must be explicitly boolean")
        self.calls += 1
        if self.calls > self.maximum_evaluations:
            raise ResidualDomainError("premixed residual evaluation budget exceeded")
        if self.progress is not None:
            self.progress(
                {
                    "event": "premixed-residual",
                    "evaluations": self.calls,
                    "maximum_evaluations": self.maximum_evaluations,
                }
            )
        amounts, temps, x, trace, trace_sum, mass = self.decode(vector)
        spec, ids = self.spec, self.ids
        area = spec.mesh.area_m2
        points = [self.response(float(t), row) for t, row in zip(temps, x, strict=True)]
        boundary = self.response(spec.inlet_temperature_K, trace)
        face_species, face_heat, face_parts, face_states = [], [], [], []
        gradient_receipts = []
        for face in range(self.count):
            left = boundary if face == 0 else points[face - 1]
            right = points[face]
            distance = (
                float(spec.mesh.centers_m[0])
                if face == 0
                else float(spec.mesh.centers_m[face] - spec.mesh.centers_m[face - 1])
            )
            weight = (
                0.5
                if face == 0
                else float((spec.mesh.faces_m[face] - spec.mesh.centers_m[face - 1]) / distance)
            )
            mean_y = {
                s: (1 - weight) * left.mass_fractions[s] + weight * right.mass_fractions[s]
                for s in ids
            }
            molar = {s: mean_y[s] / self.masses[s] for s in ids}
            total = fsum(molar.values())
            mid = self.response(
                (1 - weight) * left.temperature_K + weight * right.temperature_K,
                {s: v / total for s, v in molar.items()},
            )
            # The independent composition coordinates satisfy sum(X)=1.
            # Differentiate that identity instead of amplifying subtraction
            # roundoff by 1/dx in a nearly uniform fine-grid face. This changes
            # no composition, inventory, constitutive law or public tolerance.
            gradients, receipt = _face_mole_gradients(
                left.mole_fractions, right.mole_fractions, distance, spec.closure_species_id
            )
            if retain_details:
                gradient_receipts.append(receipt)
            flux = gas_transport_fluxes(
                mid,
                mole_fraction_gradients_per_m=gradients,
                temperature_gradient_K_m=(right.temperature_K - left.temperature_K) / distance,
                model=spec.flux_model,
                frame="mass",
                thermal_diffusion=spec.transport.thermal_diffusion,
                **(
                    {
                        "temperature_log_gradient_per_m": midpoint_temperature_log_gradient(
                            left.temperature_K, right.temperature_K, distance
                        )
                    }
                    if spec.transport.thermal_diffusion != "off"
                    else {}
                ),
            )
            if face == 0:
                # The constitutive coefficients approximate the half-cell
                # resistance at its midpoint. Material crosses the actual
                # boundary at Tin: both convective and diffusive h must use Tin.
                diffusion_h = fsum(
                    boundary.partial_enthalpies_J_mol[s] * flux.molar_fluxes_mol_m2_s[s]
                    for s in ids
                )
                flux = replace(
                    flux,
                    diffusion_enthalpy_flux_W_m2=diffusion_h,
                    total_heat_flux_W_m2=flux.fourier_heat_flux_W_m2 + diffusion_h,
                )
            advective = left
            if face > 0 and spec.advection_scheme == "linear-upwind":
                cell = face - 1
                if cell:
                    previous, following = points[cell - 1], left
                    gap = float(spec.mesh.centers_m[cell] - spec.mesh.centers_m[cell - 1])
                else:
                    previous, following = left, right
                    gap = distance
                reach = float(spec.mesh.faces_m[face] - spec.mesh.centers_m[cell])
                y_face = {
                    s: left.mass_fractions[s]
                    + reach * (following.mass_fractions[s] - previous.mass_fractions[s]) / gap
                    for s in ids
                }
                if any(v < 0 or v > 1 for v in y_face.values()):
                    raise ResidualDomainError(
                        "linear-upwind trial reconstructs an inadmissible composition"
                    )
                n_face = {s: y_face[s] / self.masses[s] for s in ids}
                total_face = fsum(n_face.values())
                t_face = (
                    left.temperature_K
                    + reach * (following.temperature_K - previous.temperature_K) / gap
                )
                advective = self.response(t_face, {s: v / total_face for s, v in n_face.items()})
            face_species.append(
                {
                    s: area
                    * (
                        mass * advective.mass_fractions[s] / self.masses[s]
                        + flux.molar_fluxes_mol_m2_s[s]
                    )
                    for s in ids
                }
            )
            bulk_h = fsum(
                advective.mass_fractions[s] * advective.partial_enthalpies_J_mol[s] / self.masses[s]
                for s in ids
            )
            face_heat.append(area * (mass * bulk_h + flux.total_heat_flux_W_m2))
            if retain_details:
                face_parts.append(flux.to_dict())
                face_states.extend(
                    {
                        "temperature_K": point.temperature_K,
                        "mole_fractions": dict(point.mole_fractions),
                    }
                    for point in (mid, advective)
                )
        last = points[-1]
        face_species.append({s: area * mass * last.mass_fractions[s] / self.masses[s] for s in ids})
        face_heat.append(
            area
            * mass
            * fsum(
                last.mass_fractions[s] * last.partial_enthalpies_J_mol[s] / self.masses[s]
                for s in ids
            )
        )
        source, species_residuals, eos, heat = [], [], [], []
        for cell in range(self.count):
            row = dict(zip(ids, map(float, amounts[cell]), strict=True))
            try:
                rates = self.network.amount_derivatives(
                    row,
                    volume_L=float(spec.mesh.cell_volumes_m3[cell]) * 1000,
                    temperature_K=float(temps[cell]),
                    **(
                        {"species_thermo": self.thermo_views}
                        if self.thermo_views is not None
                        else {}
                    ),
                )
            except (ValueError, ArithmeticError) as exc:
                raise ResidualDomainError("premixed chemistry trial: " + str(exc)) from exc
            if any(not isfinite(v) for v in rates.values()):
                raise ResidualDomainError("premixed chemistry trial has nonfinite rates")
            source.append(rates)
            species_residuals.append(
                {s: face_species[cell + 1][s] - face_species[cell][s] - rates[s] for s in ids}
            )
            eos.append(
                fsum(row.values()) * R * temps[cell] / spec.mesh.cell_volumes_m3[cell]
                - spec.pressure_Pa
            )
            heat.append(face_heat[cell + 1] - face_heat[cell])
        residual = [
            v
            for cell in range(self.count)
            for v in [*(species_residuals[cell][s] for s in self.active), eos[cell]]
        ]
        residual.extend(heat if spec.energy_equation else temps - spec.inlet_temperature_K)
        residual.extend(
            face_species[0][s] - area * mass * self.inlet_y[s] / self.masses[s] for s in self.active
        )
        residual.append(trace_sum - 1)
        if spec.mass_flux_kg_m2_s is None:
            residual.append(
                float(np.interp(spec.phase.position_m, spec.mesh.centers_m, temps))
                - spec.phase.temperature_K
            )
        if not retain_details:
            return np.asarray(residual), None
        details = {
            "amounts_mol": [dict(zip(ids, map(float, row), strict=True)) for row in amounts],
            "temperatures_K": list(map(float, temps)),
            "inlet_trace_mole_fractions": trace,
            "inlet_trace_coordinate_sum": trace_sum,
            "mass_flux_kg_m2_s": mass,
            "face_species_flows_mol_s": face_species,
            "face_enthalpy_flows_W": face_heat,
            "interior_diffusive_fluxes": face_parts,
            "composition_gradient_roundoff_receipts": gradient_receipts,
            "face_transport_and_advective_states": face_states,
            "chemical_sources_mol_s": source,
            "species_balance_residuals_mol_s": species_residuals,
            "EOS_residuals_Pa": list(map(float, eos)),
            "energy_balance_residuals_W": heat,
            "face_mass_flows_kg_s": [
                fsum(self.masses[s] * row[s] for s in ids) for row in face_species
            ],
            "chemical_mass_residuals_kg_s": [
                fsum(self.masses[s] * row[s] for s in ids) for row in source
            ],
        }
        return np.asarray(residual), details


def solve_premixed_gas_flow(
    spec: GasPremixedFlowSpec,
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
    progress: Any = None,
    jacobian_strategy: str = "dense",
    thermo_memo_max_entries: int | None = None,
    residual_only_trials: bool = False,
) -> dict[str, Any]:
    """Return a fresh-checked local BVP root; no time/stock mutation or fallback.

    Isothermal mode substitutes T=Tin for every energy row and reports required
    bath power separately. In adiabatic mode total enthalpy flux is conserved;
    inlet Fourier/diffusion enthalpy are results, not a second forced boundary.

    Numerical work reductions require explicit opt-in. A positive
    thermo_memo_max_entries opts NASA7/polynomial-transport queries into a private
    bounded exact-float-temperature memo; no network model or cached property is
    replaced. residual_only_trials omits diagnostic serialization only inside the
    shared residual callback. The accepted root always gets the original complete
    fresh profile and every original physical check. Defaults retain the old path.
    """
    if not isinstance(spec, GasPremixedFlowSpec) or not isinstance(network, ReactionNetworkSpec):
        raise ValueError("premixed query requires explicit spec and original network")
    if not isinstance(root_policy, RootPolicy):
        raise ValueError("premixed query requires the shared RootPolicy")
    if type(residual_only_trials) is not bool:
        raise ValueError("residual_only_trials must be explicitly boolean")
    if jacobian_strategy not in {"dense", "sparse-colored"}:
        raise ValueError("declare dense or sparse-colored numerical Jacobian")
    for value, name in (
        (derivative_step, "derivative step"),
        (species_tolerance_mol_s, "species tolerance"),
        (energy_tolerance_W, "energy tolerance"),
        (pressure_tolerance_Pa, "pressure tolerance"),
        (mass_tolerance_kg_s, "mass tolerance"),
    ):
        _positive(value, name)
    if any(type(v) is not int or v < 1 for v in (maximum_unknowns, maximum_evaluations)):
        raise ValueError("premixed numerical budgets must be positive integers")
    assembly = _PremixedResidual(
        spec,
        network,
        initial_profile,
        maximum_evaluations,
        progress,
        thermo_memo_max_entries=thermo_memo_max_entries,
    )
    if len(assembly.variables) > maximum_unknowns:
        raise ValueError("premixed unknown count exceeds its explicit budget")
    system = ResidualSystem(
        MODEL,
        tuple(assembly.variables),
        tuple(assembly.equations),
        (lambda t, y, rate, p: assembly.evaluate(y, retain_details=False)[0])
        if residual_only_trials
        else (lambda t, y, rate, p: assembly.evaluate(y)[0]),
        ResidualDerivatives(
            "finite_difference",
            relative_step=derivative_step,
            state_sparsity=assembly.state_sparsity()
            if jacobian_strategy == "sparse-colored"
            else None,
        ),
    )
    solved = solve_steady_residual(system, assembly.initial, policy=root_policy, progress=progress)
    report = {
        "schema": "premixed-gas-flow-query-v1",
        "success": False,
        "failure_reason": solved.root.failure_reason,
        "root_diagnostics": solved.root.diagnostics,
        "derivative_diagnostics": solved.diagnostics,
        "residual_evaluations": assembly.calls,
        "unknowns": len(assembly.variables),
        "numerical_derivatives": "shared finite_difference",
        "jacobian_strategy": jacobian_strategy,
        "live_operations_executed": 0,
        "experimental_elapsed_s": 0.0,
        "boundary_policy": (
            "fixed Tin and total incoming species flux; zero outlet gradients; "
            "inlet heat is an output"
        ),
        "scope": (
            "local planar premixed BVP; no automatic flame identification, "
            "mesh qualification, Soret or opposed flow"
            if spec.transport.thermal_diffusion == "off"
            else "local planar mixture-Soret premixed BVP; no automatic flame identification, "
            "independent mesh qualification or opposed flow"
        ),
        "profile": None,
    }

    def finish():
        # Numerical cache accounting only; no experimental work or elapsed clock.
        if assembly.thermo_memo is not None:
            report["thermo_memo_statistics"] = assembly.thermo_memo.statistics()
        if residual_only_trials:
            report["residual_only_trials"] = True
        return report

    if not solved.root.success:
        return finish()
    try:
        _, profile = assembly.evaluate(solved.root.value)
    except ResidualDomainError as exc:
        report.update(
            failure_reason="fresh_check_failed: " + str(exc),
            residual_evaluations=assembly.calls,
        )
        return finish()
    species = max(
        abs(v) for row in profile["species_balance_residuals_mol_s"] for v in row.values()
    )
    energy = max(map(abs, profile["energy_balance_residuals_W"]))
    pressure = max(map(abs, profile["EOS_residuals_Pa"]))
    trace = abs(profile["inlet_trace_coordinate_sum"] - 1)
    incoming = profile["face_species_flows_mol_s"][0]
    mass = profile["mass_flux_kg_m2_s"]
    inlet = max(
        abs(incoming[s] - spec.mesh.area_m2 * mass * assembly.inlet_y[s] / assembly.masses[s])
        for s in assembly.ids
    )
    mass_error = max(abs(v - spec.mesh.area_m2 * mass) for v in profile["face_mass_flows_kg_s"])
    source_mass_error = max(map(abs, profile["chemical_mass_residuals_kg_s"]))
    mach = []
    for t, row in [
        (spec.inlet_temperature_K, profile["inlet_trace_mole_fractions"]),
        *zip(profile["temperatures_K"], profile["amounts_mol"], strict=True),
        *(
            (row["temperature_K"], row["mole_fractions"])
            for row in profile["face_transport_and_advective_states"]
        ),
    ]:
        total = fsum(row.values())
        x = {s: row[s] / total for s in assembly.ids}
        point = assembly.response(t, x)
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
            raise ValueError("premixed ideal-gas sound speed needs positive Cv")
        mean_mass = fsum(x[s] * assembly.masses[s] for s in assembly.ids)
        mach.append((mass / point.density_kg_m3) / sqrt(cp / (cp - R) * R * t / mean_mass))
    profile["Mach_inlet_trace_cells_and_faces"] = mach
    checks = {
        "all_species_including_omitted_balance": species <= species_tolerance_mol_s,
        "all_incoming_species_including_omitted": inlet <= species_tolerance_mol_s,
        "EOS": pressure <= pressure_tolerance_Pa,
        "inlet_trace_normalization": trace <= root_policy.tolerance,
        "energy": not spec.energy_equation or energy <= energy_tolerance_W,
        "constant_face_mass_flow": mass_error <= mass_tolerance_kg_s,
        "chemical_mass_conservation": source_mass_error <= mass_tolerance_kg_s,
        "low_Mach": max(mach) <= spec.maximum_mach,
    }
    if not spec.energy_equation:
        profile["required_bath_heat_W_by_cell"] = profile["energy_balance_residuals_W"]
    report.update(
        success=all(checks.values()),
        failure_reason="" if all(checks.values()) else "fresh_physical_balance_failed",
        residual_evaluations=assembly.calls,
        physical_checks=checks,
        maximum_species_residual_mol_s=species,
        maximum_energy_residual_W=energy,
        maximum_EOS_residual_Pa=pressure,
        profile=profile,
    )
    return finish()
