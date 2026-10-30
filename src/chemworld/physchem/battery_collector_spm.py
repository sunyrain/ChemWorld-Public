"""Isothermal one-plus-one-dimensional SPM with explicit current collectors.

Each collector cell owns one spherical particle.  The particle inventories and
the two collector potentials are coupled in one residual system; there is no
per-cell time integrator or prescribed current split.  This is intentionally a
small, source-faithful model: electrolyte, heat, SEI, plating, hysteresis and
multiple particle phases are outside this family.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import cached_property
from math import asinh, hypot, isfinite, sqrt
from typing import Any, NotRequired, Self, TypedDict, cast

import numpy as np
from numpy.typing import NDArray
from scipy import sparse

from chemworld.physchem.battery_current_collector import CollectorGeometrySpec
from chemworld.physchem.battery_models import (
    BATTERY_ALGEBRAIC,
    BATTERY_COORDINATES,
    BatteryThermodynamics,
    F,
    ParticleSpec,
    R,
    particle_surface_fraction,
)
from chemworld.physchem.coupled_system import (
    CoupledAdvanceResult,
    CoupledAlgebraic,
    CoupledCoordinate,
    CoupledEvent,
    CoupledProcess,
    CoupledState,
    CoupledStateView,
    CoupledSystem,
    InternalTransfer,
    ProcessContribution,
)
from chemworld.physchem.finite_volume import ConservativeDiffusion
from chemworld.physchem.reactor_shared import ScalarControlProgram
from chemworld.physchem.residual_system import (
    ResidualDerivatives,
    ResidualEquation,
    ResidualNumericalError,
    ResidualSystem,
    ResidualVariable,
    RootPolicy,
    solve_steady_residual,
)

COLLECTOR_SPM_FAMILY = "collector-spm-isothermal-v1"


class CollectorSPMFields(TypedDict):
    """Derived per-cell fields and the shared circuit's algebraic values."""

    current_A: float
    voltage_V: float
    reaction_current_A: NDArray[np.float64]
    negative_reaction_current_A: NDArray[np.float64]
    positive_reaction_current_A: NDArray[np.float64]
    through_current_A: NDArray[np.float64]
    surface_fraction: NDArray[np.float64]
    average_fraction: NDArray[np.float64]
    kinetic_overpotential_V: NDArray[np.float64]
    collector_potential_negative_V: NDArray[np.float64]
    collector_potential_positive_V: NDArray[np.float64]
    negative_collector_face_current_A: NDArray[np.float64]
    positive_collector_face_current_A: NDArray[np.float64]
    local_voltage_V: NDArray[np.float64]
    source_boundary_voltage_V: float
    electrolyte_potential_V: NDArray[np.float64]
    bv_residual_V: NDArray[np.float64]
    algebraic_values: dict[str, float]
    closure_diagnostic: NotRequired[dict[str, Any]]


@dataclass(frozen=True)
class CollectorSPMReadout:
    voltage_V: float
    current_A: float
    negative_surface_fraction: float
    positive_surface_fraction: float
    negative_average_fraction: float
    positive_average_fraction: float
    negative_overpotential_V: float
    positive_overpotential_V: float
    negative_reaction_current_A: tuple[float, ...]
    positive_reaction_current_A: tuple[float, ...]
    negative_collector_potential_V: tuple[float, ...]
    positive_collector_potential_V: tuple[float, ...]
    local_voltage_V: tuple[float, ...]
    source_boundary_voltage_V: float

    def to_dict(self) -> dict[str, Any]:
        """Return the bounded diagnostic payload used by JSON consumers.

        Reaction currents and collector potentials are derived readout data;
        they are deliberately serialized as lists and never folded into the
        persistent device state.
        """
        return {
            "model_family": COLLECTOR_SPM_FAMILY,
            "voltage_V": float(self.voltage_V),
            "current_A": float(self.current_A),
            "negative_surface_fraction": float(self.negative_surface_fraction),
            "positive_surface_fraction": float(self.positive_surface_fraction),
            "negative_average_fraction": float(self.negative_average_fraction),
            "positive_average_fraction": float(self.positive_average_fraction),
            "negative_overpotential_V": float(self.negative_overpotential_V),
            "positive_overpotential_V": float(self.positive_overpotential_V),
            "negative_reaction_current_A": list(self.negative_reaction_current_A),
            "positive_reaction_current_A": list(self.positive_reaction_current_A),
            "negative_collector_potential_V": list(self.negative_collector_potential_V),
            "positive_collector_potential_V": list(self.positive_collector_potential_V),
            "local_voltage_V": list(self.local_voltage_V),
            "source_boundary_voltage_V": float(self.source_boundary_voltage_V),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Self:
        """Decode a readout payload without creating a second device state."""
        payload = dict(value)
        if payload.pop("model_family", COLLECTOR_SPM_FAMILY) != COLLECTOR_SPM_FAMILY:
            raise ValueError("unknown collector SPM readout family")
        for name in (
            "negative_reaction_current_A",
            "positive_reaction_current_A",
            "negative_collector_potential_V",
            "positive_collector_potential_V",
            "local_voltage_V",
        ):
            if name not in payload:
                raise ValueError(f"collector SPM readout is missing {name}")
            payload[name] = tuple(float(item) for item in payload[name])
        return cls(**payload)


@dataclass(frozen=True)
class BatteryCollectorSPMSpec:
    """Fixed-electrolyte single-particle collector model.

    ``initial_*_fractions`` are per-collector-cell stoichiometries.  A scalar
    is broadcast and a tuple preserves deliberately nonuniform initial SOC.
    """

    negative: ParticleSpec = field(default_factory=lambda: ParticleSpec(cells=24))
    positive: ParticleSpec = field(default_factory=lambda: ParticleSpec(cells=24))
    initial_negative_fractions: tuple[float, ...] | float = 0.7
    initial_positive_fractions: tuple[float, ...] | float = 0.3
    height_m: float = 1e-3
    width_m: float = 1e-2
    negative_collector_thickness_m: float = 1e-5
    positive_collector_thickness_m: float = 1e-5
    negative_collector_conductivity_S_m: float = 5.96e7
    positive_collector_conductivity_S_m: float = 3.5e7
    collector_cells: int = 8
    negative_tab: str = "left"
    negative_ocv_intercept_V: float = 0.2
    negative_ocv_slope_V: float = -0.1
    positive_ocv_intercept_V: float = 4.2
    positive_ocv_slope_V: float = -0.8
    temperature_K: float = 298.15
    surface_margin: float = 0.01
    minimum_voltage_V: float = 2.5
    maximum_voltage_V: float = 4.5
    maximum_current_A: float = 2.0
    series_resistance_ohm: float = 0.0
    provenance: str = "synthetic-collector-spm-v1"
    numerical_form: str = "residual"
    thermal: object | None = None
    electrolyte: object | None = None
    sei: object | None = None
    plating: object | None = None

    def __post_init__(self) -> None:
        negative: object = self.negative
        positive: object = self.positive
        if isinstance(negative, dict):
            object.__setattr__(self, "negative", ParticleSpec(**negative))
        if isinstance(positive, dict):
            object.__setattr__(self, "positive", ParticleSpec(**positive))
        if self.numerical_form != "residual":
            raise ValueError("collector SPM requires the common residual formulation")
        if self.thermal is not None or self.electrolyte is not None or self.sei is not None:
            raise ValueError("collector SPM is fixed-T/fixed-electrolyte without side chemistry")
        if self.plating is not None:
            raise ValueError("collector SPM does not support plating")
        if not isfinite(self.temperature_K) or self.temperature_K <= 0:
            raise ValueError("temperature must be positive and finite")
        if not 0 < self.surface_margin < 0.5:
            raise ValueError("surface margin must be strictly between zero and one half")
        if not isfinite(self.series_resistance_ohm) or self.series_resistance_ohm < 0:
            raise ValueError("series resistance must be finite and nonnegative")
        if self.minimum_voltage_V >= self.maximum_voltage_V:
            raise ValueError("voltage bounds are invalid")
        if not isfinite(self.maximum_current_A) or self.maximum_current_A <= 0:
            raise ValueError("maximum current must be positive and finite")
        for value in (
            self.negative_ocv_intercept_V,
            self.negative_ocv_slope_V,
            self.positive_ocv_intercept_V,
            self.positive_ocv_slope_V,
            self.minimum_voltage_V,
            self.maximum_voltage_V,
        ):
            if not isfinite(value):
                raise ValueError("OCV and voltage bounds must be finite")
        geometry = self.geometry
        for name, values in (
            ("initial_negative_fractions", self.initial_negative_fractions),
            ("initial_positive_fractions", self.initial_positive_fractions),
        ):
            if np.isscalar(values):
                values = tuple(float(cast(float, values)) for _ in range(geometry.cells))
            else:
                values = tuple(float(value) for value in cast(tuple[float, ...], values))
            if len(values) != geometry.cells or any(
                not self.surface_margin < value < 1 - self.surface_margin for value in values
            ):
                raise ValueError(f"{name} must contain one valid fraction per collector cell")
            object.__setattr__(self, name, values)

    @property
    def model_family(self) -> str:
        return COLLECTOR_SPM_FAMILY

    @property
    def model_id(self) -> str:
        return COLLECTOR_SPM_FAMILY

    @cached_property
    def geometry(self) -> CollectorGeometrySpec:
        return CollectorGeometrySpec(
            self.height_m,
            self.width_m,
            self.negative_collector_thickness_m,
            self.positive_collector_thickness_m,
            self.negative_collector_conductivity_S_m,
            self.positive_collector_conductivity_S_m,
            self.collector_cells,
            self.negative_tab,
        )

    @property
    def volume_l(self) -> float:
        return 1000 * (
            self.height_m
            * self.width_m
            * (self.negative_collector_thickness_m + self.positive_collector_thickness_m)
            + self.negative.active_volume_m3
            + self.positive.active_volume_m3
        )

    @cached_property
    def particle_bins(self) -> tuple[tuple[str, int, ParticleSpec, tuple[str, ...], str], ...]:
        bins = []
        for name, particle in (("negative", self.negative), ("positive", self.positive)):
            local = ParticleSpec(
                radius_m=particle.radius_m,
                diffusivity_m2_s=particle.diffusivity_m2_s,
                maximum_concentration_mol_m3=particle.maximum_concentration_mol_m3,
                active_volume_m3=particle.active_volume_m3 / self.collector_cells,
                cells=particle.cells,
                mesh_exponent=particle.mesh_exponent,
                exchange_current_scale_A_m2=particle.exchange_current_scale_A_m2,
            )
            for i in range(self.collector_cells):
                identity = f"{name}:z:{i}"
                bins.append((name, i, local, local.keys(identity), identity))
        return tuple(bins)

    @property
    def inventory_species(self) -> dict[str, str]:
        return {key: "Li" for _, _, _, keys, _ in self.particle_bins for key in keys}

    @property
    def inventory_keys(self) -> tuple[str, ...]:
        return tuple(self.inventory_species)

    @property
    def coordinate_keys(self) -> tuple[str, ...]:
        return BATTERY_COORDINATES

    @property
    def hysteresis_keys(self) -> tuple[()]:
        return ()

    @property
    def reaction_keys(self) -> tuple[str, ...]:
        return tuple(f"collector:z:{i}:reaction_A" for i in range(self.collector_cells - 1))

    @property
    def algebraic_names(self) -> tuple[str, ...]:
        return (*self.reaction_keys, *BATTERY_ALGEBRAIC)

    @property
    def usable_charge_c(self) -> float:
        negative = sum(
            p.capacity_mol * f
            for f, (_, _, p, _, _) in zip(
                cast(tuple[float, ...], self.initial_negative_fractions),
                self.particle_bins[: self.collector_cells],
                strict=True,
            )
        )
        positive = sum(
            p.capacity_mol * (1 - f)
            for f, (_, _, p, _, _) in zip(
                cast(tuple[float, ...], self.initial_positive_fractions),
                self.particle_bins[self.collector_cells :],
                strict=True,
            )
        )
        return F * min(negative, positive)

    @cached_property
    def electrical_maps(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        # The collector map is affine in q and tab voltage; keep this property
        # explicit for frequency/diagnostic consumers rather than hide it in a
        # cell solver.
        maps = self.geometry.negative_green, self.geometry.positive_green
        for matrix in maps:
            matrix.setflags(write=False)
        return maps

    def ocv(
        self, electrode: str, theta: float | np.floating[Any], *, derivative: bool = False,
    ) -> float | np.floating[Any]:
        if electrode == "negative":
            intercept, slope = self.negative_ocv_intercept_V, self.negative_ocv_slope_V
        elif electrode == "positive":
            intercept, slope = self.positive_ocv_intercept_V, self.positive_ocv_slope_V
        else:
            raise ValueError("unknown electrode")
        return slope if derivative else intercept + slope * theta

    def initial_state(self) -> CoupledState:
        amounts: dict[str, float] = {}
        for name, fractions in (
            ("negative", self.initial_negative_fractions),
            ("positive", self.initial_positive_fractions),
        ):
            for fraction, (_, _, particle, keys, _) in zip(
                cast(tuple[float, ...], fractions),
                [item for item in self.particle_bins if item[0] == name],
                strict=True,
            ):
                amounts.update(
                    zip(
                        keys,
                        particle.mesh.cell_volumes_m3
                        * particle.maximum_concentration_mol_m3
                        * fraction,
                        strict=True,
                    )
                )
        return CoupledState(
            amounts,
            self.temperature_K,
            self.volume_l,
            coordinates=dict.fromkeys(BATTERY_COORDINATES, 0.0),
        )

    def to_dict(self) -> dict[str, Any]:
        def particle_dict(particle: ParticleSpec) -> dict[str, float]:
            return {
                "radius_m": particle.radius_m,
                "diffusivity_m2_s": particle.diffusivity_m2_s,
                "maximum_concentration_mol_m3": particle.maximum_concentration_mol_m3,
                "active_volume_m3": particle.active_volume_m3,
                "cells": particle.cells,
                "mesh_exponent": particle.mesh_exponent,
                "exchange_current_scale_A_m2": particle.exchange_current_scale_A_m2,
            }

        return {
            "model_family": self.model_family,
            "negative": particle_dict(self.negative),
            "positive": particle_dict(self.positive),
            "initial_negative_fractions": list(
                cast(tuple[float, ...], self.initial_negative_fractions)),
            "initial_positive_fractions": list(
                cast(tuple[float, ...], self.initial_positive_fractions)),
            "height_m": self.height_m,
            "width_m": self.width_m,
            "negative_collector_thickness_m": self.negative_collector_thickness_m,
            "positive_collector_thickness_m": self.positive_collector_thickness_m,
            "negative_collector_conductivity_S_m": self.negative_collector_conductivity_S_m,
            "positive_collector_conductivity_S_m": self.positive_collector_conductivity_S_m,
            "collector_cells": self.collector_cells,
            "negative_tab": self.negative_tab,
            "negative_ocv_intercept_V": self.negative_ocv_intercept_V,
            "negative_ocv_slope_V": self.negative_ocv_slope_V,
            "positive_ocv_intercept_V": self.positive_ocv_intercept_V,
            "positive_ocv_slope_V": self.positive_ocv_slope_V,
            "temperature_K": self.temperature_K,
            "surface_margin": self.surface_margin,
            "minimum_voltage_V": self.minimum_voltage_V,
            "maximum_voltage_V": self.maximum_voltage_V,
            "maximum_current_A": self.maximum_current_A,
            "series_resistance_ohm": self.series_resistance_ohm,
            "provenance": self.provenance,
            "numerical_form": self.numerical_form,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Self:
        value = dict(value)
        value.pop("model_family", None)
        return cls(**value)


def _kinetics(
    spec: BatteryCollectorSPMSpec, particle: ParticleSpec, theta: float, current: float,
    temperature: float,
) -> tuple[float, float, float, float]:
    """BV and its derivatives; only residual trial kinetics use an extension.

    The actual surface fraction is never changed. Accepted operation is stopped
    at its declared surface margin by the same coupled event solver.
    """
    trial = float(np.clip(theta, spec.surface_margin / 2, 1 - spec.surface_margin / 2))
    j0 = particle.exchange_current_scale_A_m2 * sqrt(trial * (1 - trial))
    denominator = 2 * particle.area_m2 * j0
    scale = 2 * R * temperature / F
    eta = scale * asinh(current / denominator)
    d_current = scale / hypot(denominator, current)
    d_theta = (
        -d_current * current * (1 - 2 * trial) / (2 * trial * (1 - trial))
        if trial == theta
        else 0.0
    )
    return eta, d_current, d_theta, eta / temperature


def _fractions(
    spec: BatteryCollectorSPMSpec, state: CoupledStateView, values: Mapping[str, float],
) -> CollectorSPMFields:
    current, voltage = (values[key] for key in BATTERY_ALGEBRAIC)
    qn, qp = _reaction_currents(spec, values)
    surfaces, averages, eta_values, ocv_values = [], [], [], []
    for q, (name, _, particle, keys, identity) in zip((*qn, *qp), spec.particle_bins, strict=True):
        theta = particle_surface_fraction(particle, identity, state, q / F)
        eta_values.append(_kinetics(spec, particle, theta, q, state.temperature_K)[0])
        surfaces.append(theta)
        averages.append(sum(state.amounts_mol[key] for key in keys) / particle.capacity_mol)
        ocv_values.append(spec.ocv(name, theta))
    n = spec.collector_cells
    negative, positive = spec.geometry.potentials(
        qn, qp, voltage + current * spec.series_resistance_ohm
    )
    faces_n, faces_p = spec.geometry.face_currents(qn, qp)
    ocv, eta = np.asarray(ocv_values), np.asarray(eta_values)
    local = positive - negative
    # No lateral electrolyte current is declared. Each through-plane cell has
    # its own derived electrolyte offset and the same q in its two electrodes.
    phi_e = negative - ocv[:n] - eta[:n]
    residual = local - (ocv[n:] - ocv[:n] + eta[n:] - eta[:n])
    return {
        "current_A": float(current),
        "voltage_V": float(voltage),
        "reaction_current_A": np.r_[qn, qp],
        "negative_reaction_current_A": qn,
        "positive_reaction_current_A": qp,
        "through_current_A": qn,
        "surface_fraction": np.asarray(surfaces),
        "average_fraction": np.asarray(averages),
        "kinetic_overpotential_V": np.asarray(eta),
        "collector_potential_negative_V": negative,
        "collector_potential_positive_V": positive,
        "negative_collector_face_current_A": faces_n,
        "positive_collector_face_current_A": faces_p,
        "local_voltage_V": local,
        "source_boundary_voltage_V": spec.geometry.boundary_voltage_v(positive),
        "electrolyte_potential_V": phi_e,
        "bv_residual_V": residual,
        "algebraic_values": dict(values),
    }


def _reaction_currents(
    spec: BatteryCollectorSPMSpec, values: Mapping[str, float],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    free = np.asarray([values[key] for key in spec.reaction_keys], dtype=float)
    q = np.r_[free, values[BATTERY_ALGEBRAIC[0]] - np.sum(free)]
    return q, -q


def _reaction_map(
    spec: BatteryCollectorSPMSpec, identities: Sequence[str],
) -> NDArray[np.float64]:
    """Derivative of the unique local through-current in arbitrary packed order."""
    index = {key: i for i, key in enumerate(identities)}
    matrix = np.zeros((spec.collector_cells, len(identities)))
    for row, key in enumerate(spec.reaction_keys):
        matrix[row, index[key]] = 1.0
        matrix[-1, index[key]] = -1.0
    matrix[-1, index[BATTERY_ALGEBRAIC[0]]] = 1.0
    return matrix


def _control_residual(
    spec: BatteryCollectorSPMSpec, state: CoupledStateView, program: ScalarControlProgram,
) -> float:
    current, voltage = (state.algebraic_values[key] for key in BATTERY_ALGEBRAIC)
    target = program.value_at(state.time_s)
    if program.unit == "A":
        return (current - target) / spec.maximum_current_A
    requested = current + spec.maximum_current_A * (voltage - target)
    return (
        current - np.clip(requested, -spec.maximum_current_A, spec.maximum_current_A)
    ) / spec.maximum_current_A


def _closure_jacobian(
    spec: BatteryCollectorSPMSpec, state: CoupledStateView, program: ScalarControlProgram,
    identities: Sequence[str],
) -> sparse.csc_matrix:
    """Full derivative of local BV/collector constraints, including surface flux."""
    index = {key: i for i, key in enumerate(identities)}
    n = spec.collector_cells
    qmap = _reaction_map(spec, identities)
    qn, qp = _reaction_currents(spec, state.algebraic_values)
    jac = np.zeros((n + 1, len(identities)))
    gn, gp = spec.electrical_maps
    jac[:n] = (gn + gp) @ qmap
    ik, vk = (index[key] for key in BATTERY_ALGEBRAIC)
    jac[:n, ik] += spec.series_resistance_ohm
    jac[:n, vk] += 1.0
    for current, (name, z, particle, keys, identity) in zip(
        (*qn, *qp), spec.particle_bins, strict=True
    ):
        sign = 1.0 if name == "negative" else -1.0
        theta = particle_surface_fraction(particle, identity, state, current / F)
        _, eta_i, eta_theta, eta_t = _kinetics(spec, particle, theta, current, state.temperature_K)
        slope = spec.ocv(name, theta, derivative=True) + eta_theta
        flux_factor = particle.mesh.spherical_surface_distance_m / (
            F * particle.area_m2 * particle.diffusivity_m2_s * particle.maximum_concentration_mol_m3
        )
        # Residual has +(Un+eta_n) and -(Up+eta_p), while qp=-q.
        jac[z] += (eta_i - slope * flux_factor) * qmap[z]
        if keys[-1] in index:
            jac[z, index[keys[-1]]] += (
                sign
                * slope
                / (particle.mesh.cell_volumes_m3[-1] * particle.maximum_concentration_mol_m3)
            )
        if "temperature" in index:
            jac[z, index["temperature"]] += sign * eta_t
    current, voltage = (state.algebraic_values[key] for key in BATTERY_ALGEBRAIC)
    requested = current + spec.maximum_current_A * (voltage - program.value_at(state.time_s))
    if program.unit == "V" and abs(requested) < spec.maximum_current_A:
        jac[-1, vk] = -1.0
    else:
        jac[-1, ik] = 1 / spec.maximum_current_A
    return sparse.csc_matrix(jac)


def collector_spm_algebraic(
    spec: BatteryCollectorSPMSpec, program: ScalarControlProgram,
) -> CoupledAlgebraic:
    if program.unit not in {"A", "V"}:
        raise ValueError("collector SPM control uses A or V")
    for _, value in program.setpoints:
        if program.unit == "A" and abs(value) > spec.maximum_current_A:
            raise ValueError("collector current setpoint exceeds declared source")
        if program.unit == "V" and not spec.minimum_voltage_V <= value <= spec.maximum_voltage_V:
            raise ValueError("collector voltage setpoint is outside declared domain")
    n = spec.collector_cells
    variables = (
        *(
            ResidualVariable(key, "A", spec.maximum_current_A, "algebraic")
            for key in spec.reaction_keys
        ),
        ResidualVariable(
            BATTERY_ALGEBRAIC[0],
            "A",
            spec.maximum_current_A,
            "algebraic",
            -spec.maximum_current_A,
            spec.maximum_current_A,
        ),
        ResidualVariable(BATTERY_ALGEBRAIC[1], "V", 1.0, "algebraic"),
    )
    equations = (
        *(ResidualEquation(f"collector:bv:{i}", "V") for i in range(n)),
        ResidualEquation("battery:control", "1"),
    )

    def initial_values(state: CoupledStateView) -> dict[str, float]:
        if program.unit == "V":
            # A CV Newton step can request a current far beyond the source
            # limit. Projecting only the terminal current then leaves the
            # free local currents at that unconstrained step; the eliminated
            # final cell must compensate, spoiling the electrical initial
            # guess. Find an admissible active branch using this same CC
            # residual and solver, without clipping any local current.
            for current in (spec.maximum_current_A, -spec.maximum_current_A):
                try:
                    candidate = collector_spm_fields(spec, state, current_A=current)
                except ResidualNumericalError:
                    # An optional endpoint seed must not make an otherwise
                    # solvable interior CV state unavailable. The original CV
                    # consistency solve remains authoritative below.
                    continue
                values = candidate["algebraic_values"]
                candidate_state = CoupledStateView(
                    state.amounts_mol,
                    state.temperature_K,
                    state.volume_L,
                    state.time_s,
                    coordinates=state.coordinates,
                    algebraic_values=values,
                )
                # Check all BV equations and the ORIGINAL CV complementarity,
                # rather than selecting a branch from an OCV approximation.
                # This is only a seed: the outer coupled CV solve validates
                # the same equations again, including the differential rates.
                if np.linalg.norm(residual(candidate_state), ord=np.inf) <= 1e-11:
                    return dict(values)
        current = program.value_at(state.time_s) if program.unit == "A" else 0.0
        means = {
            name: np.mean(
                [
                    sum(state.amounts_mol[k] for k in keys) / particle.capacity_mol
                    for n0, _, particle, keys, _ in spec.particle_bins
                    if n0 == name
                ]
            )
            for name in ("negative", "positive")
        }
        qn = np.full(n - 1, current / n)
        voltage = spec.ocv("positive", means["positive"]) - spec.ocv("negative", means["negative"])
        return dict(
            zip(
                spec.algebraic_names,
                [*qn, current, voltage],
                strict=True,
            )
        )

    def residual(state: CoupledStateView) -> NDArray[np.float64]:
        values = state.algebraic_values
        f = _fractions(spec, state, values)
        return np.r_[f["bv_residual_V"], _control_residual(spec, state, program)]

    return CoupledAlgebraic(variables, equations, initial_values, residual)


def collector_spm_fields(
    spec: BatteryCollectorSPMSpec, state: CoupledStateView, *, current_A: float | None = None,
    program: ScalarControlProgram | None = None,
) -> CollectorSPMFields:
    if (current_A is None) == (program is None):
        raise ValueError("supply either a current or a control programme")
    if current_A is not None:
        if not isfinite(current_A) or abs(current_A) > spec.maximum_current_A:
            raise ValueError("current is outside collector source bounds")
        program = ScalarControlProgram(((0.0, current_A),), "A")
    names = spec.algebraic_names
    # Exactly one input was supplied, and a current has already become a program.
    program = cast(ScalarControlProgram, program)
    closure = collector_spm_algebraic(spec, program)

    def view(y: NDArray[np.float64]) -> CoupledStateView:
        return CoupledStateView(
            state.amounts_mol,
            state.temperature_K,
            state.volume_L,
            state.time_s,
            coordinates=state.coordinates,
            algebraic_values=dict(zip(names, y, strict=True)),
        )

    residual = ResidualSystem(
        "battery:collector-spm-closure",
        closure.variables,
        closure.equations,
        lambda _t, y, _rate, _p: closure.residual(view(y)),
        ResidualDerivatives(
            "analytic",
            state=lambda _t, y, _rate, _p: _closure_jacobian(spec, view(y), program, names),
            rate=lambda *_: sparse.csc_matrix((len(names), len(names))),
        ),
    )
    guess = closure.initial_values(state)
    answer = solve_steady_residual(
        residual, [guess[key] for key in names], policy=RootPolicy(tolerance=1e-11)
    )
    if not answer.root.success:
        raise ResidualNumericalError(
            "collector SPM electrical closure:" + answer.root.failure_reason
        )
    values = dict(zip(names, answer.root.value, strict=True))
    fields = _fractions(spec, state, values)
    return {**fields, "closure_diagnostic": answer.root.diagnostics}


def collector_spm_readout(
    spec: BatteryCollectorSPMSpec, state: CoupledStateView, current_A: float,
) -> CollectorSPMReadout:
    fields = collector_spm_fields(spec, state, current_A=current_A)
    n = spec.collector_cells
    return CollectorSPMReadout(
        fields["voltage_V"],
        fields["current_A"],
        float(np.mean(fields["surface_fraction"][:n])),
        float(np.mean(fields["surface_fraction"][n:])),
        float(np.mean(fields["average_fraction"][:n])),
        float(np.mean(fields["average_fraction"][n:])),
        float(np.mean(fields["kinetic_overpotential_V"][:n])),
        float(np.mean(fields["kinetic_overpotential_V"][n:])),
        tuple(fields["negative_reaction_current_A"]),
        tuple(fields["positive_reaction_current_A"]),
        tuple(fields["collector_potential_negative_V"]),
        tuple(fields["collector_potential_positive_V"]),
        tuple(fields["local_voltage_V"]),
        fields["source_boundary_voltage_V"],
    )


def collector_spm_control_mode(
    spec: BatteryCollectorSPMSpec, state: CoupledStateView, program: ScalarControlProgram,
) -> tuple[str, float]:
    target = program.value_at(state.time_s)
    if program.unit == "A":
        return "constant_current", target
    if program.unit != "V":
        raise ValueError("collector SPM control uses A or V")
    fields = collector_spm_fields(spec, state, program=program)
    current = fields["current_A"]
    if abs(current) >= spec.maximum_current_A * (1 - 1e-8):
        return ("negative_current_limit" if current < 0 else "positive_current_limit"), current
    return "constant_voltage", target


def collector_spm_control_current(
    spec: BatteryCollectorSPMSpec, state: CoupledStateView, program: ScalarControlProgram,
) -> float:
    if BATTERY_ALGEBRAIC[0] in state.algebraic_values:
        return state.algebraic_values[BATTERY_ALGEBRAIC[0]]
    if program.unit == "A":
        target = program.value_at(state.time_s)
        if abs(target) > spec.maximum_current_A:
            raise ValueError("current is outside collector source capability")
        return target
    return float(collector_spm_fields(spec, state, program=program)["current_A"])


def collector_spm_system(
    spec: BatteryCollectorSPMSpec, program: ScalarControlProgram, *,
    thermal_device: object | None = None, control_events: Sequence[CoupledEvent] = (),
) -> CoupledSystem:
    if thermal_device is not None:
        raise ValueError("collector SPM is isothermal and cannot accept a thermal device")
    closure = collector_spm_algebraic(spec, program)
    processes = [
        ConservativeDiffusion(p.mesh, p.diffusivity_m2_s)
        .couple(keys, process_id=f"battery:collector:{identity}", owner_id="battery")
        .process
        for _, _, p, keys, identity in spec.particle_bins
    ]

    def fields(state: CoupledStateView) -> CollectorSPMFields:
        if all(key in state.algebraic_values for key in spec.algebraic_names):
            return _fractions(spec, state, state.algebraic_values)
        return collector_spm_fields(spec, state, program=program)

    def source(state: CoupledStateView) -> ProcessContribution:
        f = fields(state)
        n = spec.collector_cells
        transfers = []
        for z, q in enumerate(f["through_current_A"]):
            negative = spec.particle_bins[z][3][-1]
            positive = spec.particle_bins[n + z][3][-1]
            origin, destination = (negative, positive) if q >= 0 else (positive, negative)
            transfers.append(
                InternalTransfer(
                    f"battery:collector:Li:z:{z}", origin, destination, abs(float(q)) / F
                )
            )
        return ProcessContribution(
            internal_transfers=tuple(transfers),
            coordinate_rates={
                "discharged_charge_C": f["current_A"],
                "delivered_work_J": f["current_A"] * f["voltage_V"],
            },
        )

    def validate(state: CoupledStateView) -> None:
        if (
            abs(state.temperature_K - spec.temperature_K) > 1e-8
            or abs(state.volume_L - spec.volume_l) > 1e-12
        ):
            raise ValueError("collector SPM requires its declared isothermal volume")
        for _, _, particle, keys, _ in spec.particle_bins:
            if any(
                state.amounts_mol[key] > volume * particle.maximum_concentration_mol_m3 * (1 + 1e-8)
                for key, volume in zip(keys, particle.mesh.cell_volumes_m3, strict=True)
            ):
                raise ValueError("collector particle inventory exceeds finite capacity")

    processes.append(
        CoupledProcess("battery:collector-spm-intercalation", source, validate_state=validate)
    )

    def surface_limit(state: CoupledStateView, row: int, lower: bool) -> float:
        f = fields(state)
        q, theta = f["reaction_current_A"][row], f["surface_fraction"][row]
        # A stopped endpoint may rest or accept inward current. These gates
        # only select a stopping test; they never clip the actual transfer.
        if lower:
            return theta - spec.surface_margin if q > 0 else 1.0
        return 1 - spec.surface_margin - theta if q < 0 else 1.0

    events = [
        CoupledEvent(
            "battery:collector:minimum-voltage",
            lambda s: fields(s)["voltage_V"] - spec.minimum_voltage_V,
        ),
        CoupledEvent(
            "battery:collector:maximum-voltage",
            lambda s: spec.maximum_voltage_V - fields(s)["voltage_V"],
        ),
    ]
    def surface_event(row: int, lower: bool) -> Callable[[CoupledStateView], float]:
        def event(state: CoupledStateView) -> float:
            return surface_limit(state, row, lower)

        return event

    for row, (name, z, _particle, _keys, _identity) in enumerate(spec.particle_bins):
        for lower, label in ((True, "minimum"), (False, "maximum")):
            events.append(
                CoupledEvent(
                    f"battery:collector:{name}:z:{z}:{label}-surface",
                    surface_event(row, lower),
                )
            )

    diffusion_blocks = tuple(
        (keys, ConservativeDiffusion(p.mesh, p.diffusivity_m2_s).inventory_jacobian_s_inv)
        for _, _, p, keys, _ in spec.particle_bins
    )

    def jacobian(state: CoupledStateView, identities: Sequence[str]) -> sparse.csc_matrix:
        index = {key: i for i, key in enumerate(identities)}
        jac = sparse.lil_matrix((len(identities), len(identities)))
        for keys, block in diffusion_blocks:
            columns = [index[key] for key in keys]
            jac[np.ix_(columns, columns)] = block
        qmap = _reaction_map(spec, identities)
        for name, z, _particle, keys, _identity in spec.particle_bins:
            sign = -1.0 if name == "negative" else 1.0
            jac[index[keys[-1]]] += sign * qmap[z] / F
        ik, vk = (index[key] for key in BATTERY_ALGEBRAIC)
        current, voltage = (state.algebraic_values[key] for key in BATTERY_ALGEBRAIC)
        jac[index["discharged_charge_C"], ik] = 1.0
        jac[index["delivered_work_J"], ik] = voltage
        jac[index["delivered_work_J"], vk] = current
        rows = [index[key] for key in spec.algebraic_names]
        jac[rows, :] = -_closure_jacobian(spec, state, program, identities)
        return jac.tocsc()

    scales = {k: p.capacity_mol / p.cells for _, _, p, keys, _ in spec.particle_bins for k in keys}
    nkeys = [k for n, _, _, keys, _ in spec.particle_bins if n == "negative" for k in keys]
    pkeys = [k for n, _, _, keys, _ in spec.particle_bins if n == "positive" for k in keys]
    return CoupledSystem(
        BatteryThermodynamics(spec),
        tuple(processes),
        isothermal=True,
        events=(*events, *control_events),
        inventory_scales_mol=scales,
        coordinates=(
            CoupledCoordinate("discharged_charge_C", "C", spec.usable_charge_c),
            CoupledCoordinate("delivered_work_J", "J", spec.usable_charge_c * 4),
        ),
        conservation_laws={
            "lithium_mol": dict.fromkeys(spec.inventory_keys, 1),
            "negative_faradaic_C": {**dict.fromkeys(nkeys, F), "discharged_charge_C": 1},
            "positive_faradaic_C": {**dict.fromkeys(pkeys, F), "discharged_charge_C": -1},
            **{
                f"local_lithium:z:{z}": dict.fromkeys(
                    (*spec.particle_bins[z][3], *spec.particle_bins[spec.collector_cells + z][3]),
                    1.0,
                )
                for z in range(spec.collector_cells)
            },
        },
        numerical_form="residual",
        algebraic=closure,
        residual_rhs_jacobian=jacobian,
    )


def advance_collector_spm(
    spec: BatteryCollectorSPMSpec,
    state: CoupledState,
    program: ScalarControlProgram,
    duration_s: float,
    *,
    evaluation_times_s: Sequence[float] | None = None,
    thermal_device: object | None = None,
    control_events: Sequence[CoupledEvent] = (),
) -> CoupledAdvanceResult:
    result = collector_spm_system(
        spec, program, thermal_device=thermal_device, control_events=control_events
    ).advance(state, duration_s, evaluation_times_s=evaluation_times_s)
    return result


def default_collector_spm_spec(
    *, cells: int = 6, collector_cells: int = 4,
) -> BatteryCollectorSPMSpec:
    """Small deterministic fixture for examples and focused integration tests."""
    particle = ParticleSpec(cells=cells, active_volume_m3=5e-7)
    return BatteryCollectorSPMSpec(
        negative=particle,
        positive=particle,
        collector_cells=collector_cells,
        maximum_current_A=0.5,
        provenance="synthetic-collector-spm-fixture-v1",
    )


__all__ = [
    "COLLECTOR_SPM_FAMILY",
    "BatteryCollectorSPMSpec",
    "CollectorSPMReadout",
    "advance_collector_spm",
    "collector_spm_algebraic",
    "collector_spm_control_current",
    "collector_spm_control_mode",
    "collector_spm_fields",
    "collector_spm_readout",
    "collector_spm_system",
    "default_collector_spm_spec",
]
