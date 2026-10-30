"""Finite-metal, two-domain DFN half cell on the original coupled residual.

Positive current strips the fixed-geometry metal and lithiates the spatial
positive working electrode. Salt has a nonzero metal-side diffusive boundary
flux. No zero-thickness negative porous electrode, moving foil, private solver,
thermal closure, or side-reaction model is introduced.
"""

from dataclasses import asdict, dataclass, field, replace
from functools import cached_property, lru_cache
from itertools import pairwise
from math import isclose, isfinite
from numbers import Integral, Real

import numpy as np
from scipy import sparse

from chemworld.data.scientific_inputs import ScientificTable, _keys, _text
from chemworld.foundation.units import convert_value
from chemworld.physchem.battery_composite import _monotone
from chemworld.physchem.battery_electrolyte import ELECTROLYTE_SPECIES
from chemworld.physchem.battery_half_cell import (
    METAL_CONVENTION,
    METAL_KEY,
    LithiumMetalCounterElectrodeSpec,
)
from chemworld.physchem.battery_materials import ElectrodeOCV
from chemworld.physchem.battery_models import (
    BATTERY_ALGEBRAIC,
    BATTERY_COORDINATES,
    BatteryThermodynamics,
    F,
    ParticleSpec,
    R,
    particle_surface_fraction,
)
from chemworld.physchem.battery_porous_electrical import integral_centres
from chemworld.physchem.coupled_system import (
    CoupledAlgebraic,
    CoupledCoordinate,
    CoupledEvent,
    CoupledProcess,
    CoupledState,
    CoupledStateView,
    CoupledSystem,
    ProcessContribution,
)
from chemworld.physchem.finite_volume import ConservativeDiffusion, DiffusionMesh
from chemworld.physchem.reactor_shared import ScalarControlProgram
from chemworld.physchem.residual_system import (
    ResidualDerivatives,
    ResidualDomainError,
    ResidualEquation,
    ResidualNumericalError,
    ResidualSystem,
    ResidualVariable,
    RootPolicy,
    solve_steady_residual,
)

HALF_CELL_DFN_FAMILY = "half-cell-dfn-isothermal-v1"
ELECTROLYTE_VERSION = "half-cell-electrolyte-isothermal-v1"
METAL_KINETICS_VERSION = "metal-reference-salt-power-law-v1"
PHI_E = "half-cell-dfn:metal-electrolyte-potential_V"
ELECTRICAL_CLOSURE_TOLERANCE = 1e-12


def _finite(value, name):
    if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value):
        raise ValueError(f"{name} must be a finite real number")
    return float(value)


def _ocv_trial_monotone_certified(curve):
    """Analytic sufficient proof on every real FV surface trial, not sampled slopes.

    An affine polynomial with nonpositive slope and individually nonincreasing
    exp/tanh terms is nonincreasing on the whole real line. A declared endpoint
    clip preserves order and its positive rational barrier has negative slope.
    For other clipped expressions, outward intervals cover the entire finite
    clip domain; the remaining real line is the declared constant extension.
    Unproved expressions remain available under CC but cannot select a CV root.
    """
    polynomial = curve.polynomial_V

    def nonpositive_product(a, k):
        # Sign tests avoid certifying a tiny positive a*k rounded down to zero.
        return a == 0 or k == 0 or (a < 0) != (k < 0)

    termwise = (
        not any(polynomial[2:])
        and (len(polynomial) < 2 or polynomial[1] <= 0)
        and all(
            nonpositive_product(a, k) for a, k, _ in (*curve.exponential_terms, *curve.tanh_terms)
        )
    )
    if termwise:
        return True
    if curve.endpoint_policy is None:
        return False
    policy = curve.endpoint_policy
    knots = np.linspace(policy.stoichiometry_clip, 1 - policy.stoichiometry_clip, 65)
    try:
        return all(
            curve.enclosure(lo, hi, derivative=True, trial=True).upper <= 0
            for lo, hi in pairwise(knots)
        )
    except (ValueError, ArithmeticError):
        return False


@dataclass(frozen=True)
class BatteryHalfCellElectrolyteSpec:
    """Two real porous domains, with constant binary-electrolyte transport."""

    area_m2: float = 0.02
    lengths_m: tuple[float, float] = (2.5e-5, 1e-4)
    porosities: tuple[float, float] = (0.5, 0.4)
    bruggeman: tuple[float, float] = (1.5, 1.5)
    cells: tuple[int, int] = (24, 24)
    diffusivity_m2_s: float = 2e-10
    conductivity_S_m: float = 1.0
    cation_transference: float = 0.4
    thermodynamic_factor: float = 1.0
    initial_concentration_mol_m3: float = 1000.0
    minimum_concentration_mol_m3: float = 1.0
    positive_solid_conductivity_S_m: float = 100.0
    model_version: str = ELECTROLYTE_VERSION

    def __post_init__(self) -> None:
        if self.model_version != ELECTROLYTE_VERSION:
            raise ValueError("unknown half-cell electrolyte version")
        for name in ("lengths_m", "porosities", "bruggeman", "cells"):
            values = tuple(getattr(self, name))
            if len(values) != 2 or any(
                isinstance(v, bool) or not isinstance(v, Real) or not isfinite(v) or v <= 0
                for v in values
            ):
                raise ValueError(f"{name} needs two positive finite entries")
            object.__setattr__(self, name, values)
        if any(not isinstance(v, Integral) or v < 2 for v in self.cells):
            raise ValueError("each half-cell salt domain needs at least two integer cells")
        if any(v >= 1 for v in self.porosities):
            raise ValueError("half-cell porosities must be strictly inside (0,1)")
        for name in (
            "area_m2",
            "diffusivity_m2_s",
            "conductivity_S_m",
            "thermodynamic_factor",
            "initial_concentration_mol_m3",
            "minimum_concentration_mol_m3",
            "positive_solid_conductivity_S_m",
        ):
            value = _finite(getattr(self, name), name)
            if value <= 0:
                raise ValueError(f"{name} must be positive")
            object.__setattr__(self, name, value)
        transference = _finite(self.cation_transference, "cation_transference")
        if not 0 <= transference < 1:
            raise ValueError("cation transference must be in [0,1)")
        object.__setattr__(self, "cation_transference", transference)
        if self.minimum_concentration_mol_m3 >= self.initial_concentration_mol_m3:
            raise ValueError("initial salt must exceed the declared positive cutoff")
        effective = np.r_[
            self.transport_efficiencies * self.diffusivity_m2_s,
            self.transport_efficiencies * self.conductivity_S_m,
        ]
        if any(not isfinite(v) or v <= 0 for v in effective):
            raise ValueError("effective half-cell transport must be positive and representable")

    @cached_property
    def transport_efficiencies(self):
        values = np.power(self.porosities, self.bruggeman)
        values.flags.writeable = False
        return values

    @cached_property
    def mesh(self):
        faces = [0.0]
        for length, count in zip(self.lengths_m, self.cells, strict=True):
            faces.extend(faces[-1] + np.linspace(0, length, count + 1)[1:])
        return DiffusionMesh("planar", tuple(faces), area_m2=self.area_m2)

    @cached_property
    def region_slices(self):
        ns, np_ = self.cells
        return {"separator": slice(0, ns), "positive": slice(ns, ns + np_)}

    @cached_property
    def keys(self):
        return tuple(f"electrolyte:salt:{i}" for i in range(sum(self.cells)))

    @cached_property
    def diffusion(self):
        return ConservativeDiffusion(
            self.mesh,
            tuple(np.repeat(self.transport_efficiencies * self.diffusivity_m2_s, self.cells)),
            storage_fractions=tuple(np.repeat(self.porosities, self.cells)),
        )

    def concentrations(self, state):
        return np.array([state.amounts_mol[k] for k in self.keys]) / (
            self.diffusion.storage_volumes_m3
        )

    @cached_property
    def boundary_maps(self):
        """Exact quadratic cell-average reconstruction with prescribed gradients."""
        size = len(self.keys)
        matrix = np.zeros((2, size))
        widths = np.diff(self.mesh.faces_m)
        for row, indices in enumerate(((0, 1), (size - 1, size - 2))):
            h0, h1 = widths[list(indices)]
            weight = h0**2 / ((h0 + h1) * (2 * h0 + h1))
            matrix[row, indices[0]] = 1 + weight
            matrix[row, indices[1]] = -weight
        h0, h1 = widths[:2]
        distance = h0 * (h0 + h1) / (2 * (2 * h0 + h1))
        # c_boundary = reconstruction - distance * dce/dx at the left.
        current_coefficient = (
            distance
            * (1 - self.cation_transference)
            / (self.area_m2 * F * self.diffusivity_m2_s * self.transport_efficiencies[0])
        )
        matrix.flags.writeable = False
        return matrix, float(current_coefficient)

    def boundary_concentrations(self, state, current_A):
        current = _finite(current_A, "boundary current")
        matrix, coefficient = self.boundary_maps
        values = matrix @ self.concentrations(state)
        values[0] += coefficient * current
        return values

    def minimum_resolved_concentration(self, state, current_A):
        return float(
            min(
                np.min(self.concentrations(state)),
                np.min(self.boundary_concentrations(state, current_A)),
            )
        )

    def initial_amounts(self):
        return dict(
            zip(
                self.keys,
                self.diffusion.storage_volumes_m3 * self.initial_concentration_mol_m3,
                strict=True,
            )
        )

    def to_dict(self):
        return {
            key: list(value) if isinstance(value, tuple) else value
            for key, value in asdict(self).items()
        }

    @classmethod
    def from_dict(cls, value):
        if set(value) != set(cls.__dataclass_fields__):
            raise ValueError("unknown or missing half-cell electrolyte fields")
        return cls(**value)


@dataclass(frozen=True)
class LithiumMetalSaltKineticsSpec:
    """Sourced salt exponent; j0 and its reference ce,T remain in the metal table."""

    parameters: ScientificTable
    parameter_row_id: str
    model_version: str = METAL_KINETICS_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.parameters, ScientificTable) or self.parameters.kind != "parameters":
            raise ValueError("metal salt kinetics require a sourced parameter table")
        _text(self.parameter_row_id, "parameter_row_id")
        rows = {row.row_id: row for row in self.parameters.rows}
        if self.model_version != METAL_KINETICS_VERSION or self.parameter_row_id not in rows:
            raise ValueError("unknown metal salt kinetics version or source row")
        cells = rows[self.parameter_row_id].values
        if set(cells) != {"salt_exponent"}:
            raise ValueError("metal salt kinetics require exactly the explicit salt exponent")
        cell = cells["salt_exponent"]
        if cell.quality != "valid" or cell.value is None:
            raise ValueError("metal salt exponent must be a valid input")
        exponent = convert_value(cell.value, cell.unit, "dimensionless")
        if not isfinite(exponent) or exponent < 0:
            raise ValueError("metal salt exponent must be finite and nonnegative")

    @cached_property
    def salt_exponent(self):
        row = next(row for row in self.parameters.rows if row.row_id == self.parameter_row_id)
        value = row.values["salt_exponent"]
        return convert_value(value.value, value.unit, "dimensionless")

    def to_dict(self):
        return {
            "parameters": self.parameters.to_dict(),
            "parameter_row_id": self.parameter_row_id,
            "model_version": self.model_version,
        }

    @classmethod
    def from_dict(cls, value):
        _keys(value, ("parameters", "parameter_row_id", "model_version"))
        return cls(
            ScientificTable.from_dict(value["parameters"]),
            value["parameter_row_id"],
            value["model_version"],
        )


@dataclass(frozen=True)
class BatteryHalfCellDFNSpec:
    metal: LithiumMetalCounterElectrodeSpec
    positive: ParticleSpec
    positive_ocv: ElectrodeOCV
    metal_kinetics: LithiumMetalSaltKineticsSpec
    electrolyte: BatteryHalfCellElectrolyteSpec = field(
        default_factory=BatteryHalfCellElectrolyteSpec
    )
    working_reference_concentration_mol_m3: float = 1000.0
    initial_positive_fraction: float = 0.3
    temperature_K: float = 298.15
    surface_margin: float = 0.01
    minimum_voltage_V: float = 2.5
    maximum_voltage_V: float = 4.5
    maximum_current_A: float = 2.0
    provenance: str = "declared-fixed-geometry-half-cell-DFN-v1"

    def __post_init__(self) -> None:
        constructors = {
            "metal": LithiumMetalCounterElectrodeSpec.from_dict,
            "positive": lambda value: ParticleSpec(**value),
            "positive_ocv": ElectrodeOCV.from_dict,
            "metal_kinetics": LithiumMetalSaltKineticsSpec.from_dict,
            "electrolyte": BatteryHalfCellElectrolyteSpec.from_dict,
        }
        for name, constructor in constructors.items():
            if isinstance(getattr(self, name), dict):
                object.__setattr__(self, name, constructor(getattr(self, name)))
        if (
            not isinstance(self.metal, LithiumMetalCounterElectrodeSpec)
            or not isinstance(self.positive, ParticleSpec)
            or not isinstance(self.positive_ocv, ElectrodeOCV)
            or not isinstance(self.metal_kinetics, LithiumMetalSaltKineticsSpec)
            or not isinstance(self.electrolyte, BatteryHalfCellElectrolyteSpec)
        ):
            raise ValueError("DFN half cell needs explicit metal, working particle, OCV and salt")
        for name in (
            "working_reference_concentration_mol_m3",
            "initial_positive_fraction",
            "temperature_K",
            "surface_margin",
            "minimum_voltage_V",
            "maximum_voltage_V",
            "maximum_current_A",
        ):
            object.__setattr__(self, name, _finite(getattr(self, name), name))
        if (
            self.working_reference_concentration_mol_m3 <= 0
            or self.temperature_K <= 0
            or not 0 < self.surface_margin < 0.5
            or self.maximum_current_A <= 0
            or self.minimum_voltage_V >= self.maximum_voltage_V
            or not self.surface_margin < self.initial_positive_fraction < 1 - self.surface_margin
        ):
            raise ValueError("invalid DFN half-cell operating domain")
        _text(self.provenance, "provenance")
        curve = self.positive_ocv
        if (
            "DFN" not in curve.supported_models
            or curve.stoichiometry_domain[0] > self.surface_margin
            or curve.stoichiometry_domain[1] < 1 - self.surface_margin
            or not _monotone(curve, self.surface_margin, 1 - self.surface_margin)
        ):
            raise ValueError("working OCV must cover the DFN surfaces and be nonincreasing")
        metal = self.metal.native_parameters
        if not isclose(
            metal["reference_temperature_K"], self.temperature_K, rel_tol=0, abs_tol=1e-8
        ):
            raise ValueError("metal salt kinetics require their declared reference temperature")
        if not isclose(metal["area_m2"], self.electrolyte.area_m2, rel_tol=1e-12, abs_tol=0):
            raise ValueError("planar metal and the two salt domains must have the same area")
        allowed = (
            self.electrolyte.area_m2
            * self.electrolyte.lengths_m[1]
            * (1 - self.electrolyte.porosities[1])
        )
        if self.positive.active_volume_m3 > allowed and not isclose(
            self.positive.active_volume_m3, allowed, rel_tol=8 * np.finfo(float).eps, abs_tol=0
        ):
            raise ValueError("working active volume exceeds its non-pore electrode volume")
        if any(
            not isfinite(v) or v <= 0
            for v in (self.usable_charge_c, self.volume_l, self.positive.area_m2)
        ):
            raise ValueError("derived DFN half-cell scales must be positive and finite")

    @property
    def model_family(self):
        return HALF_CELL_DFN_FAMILY

    model_id = model_family

    @property
    def numerical_form(self) -> str:
        return "residual"

    @property
    def thermal(self) -> None:
        return None

    sei = plating = negative_ocp = positive_ocp = thermal

    @property
    def series_resistance_ohm(self) -> float:
        return 0.0

    @property
    def has_directional_ocp(self) -> bool:
        return False

    @property
    def hysteresis_keys(self):
        return ()

    @property
    def coordinate_keys(self):
        return BATTERY_COORDINATES

    @property
    def volume_l(self):
        return (
            1000
            * self.electrolyte.area_m2
            * (self.metal.native_parameters["thickness_m"] + sum(self.electrolyte.lengths_m))
        )

    @property
    def initial_stoichiometric_discharge_bound_c(self):
        return F * min(
            self.metal.initial_amount_mol,
            self.positive.capacity_mol * (1 - self.surface_margin - self.initial_positive_fraction),
        )

    @property
    def usable_charge_c(self):
        """Initial positive-direction stoichiometric bound, used as a solver scale."""
        return self.initial_stoichiometric_discharge_bound_c

    def current_at_c_rate(self, rate):
        """Explicit initial-bound C rate, not an empirical nominal capacity rating."""
        return _finite(rate, "initial-bound C-rate") * self.usable_charge_c / 3600

    @cached_property
    def cv_monotone_certified(self):
        """Conservative full-trial passivity proof; CC retains general beta >= 0.

        For cb=cbase+kI with cbase>0 and 0<=beta<=1, the metal BV derivative
        contains 1-beta*kI/cb>0 wherever cb is not floored. In the floored
        extension its j0 is constant. The concentration term -alpha*ln(cb)
        is nonincreasing with I. Positive resistances and the monotone working
        branches therefore give a unique bounded-source CV closure. The state
        conditions are enforced separately at every CV residual evaluation.
        """
        return self.metal_kinetics.salt_exponent <= 1 and _ocv_trial_monotone_certified(
            self.positive_ocv
        )

    @cached_property
    def particle_bins(self):
        widths = np.diff(self.electrolyte.mesh.faces_m)[self.electrolyte.region_slices["positive"]]
        bins = []
        for i, width in enumerate(widths):
            particle = replace(
                self.positive,
                active_volume_m3=(
                    self.positive.active_volume_m3 * width / self.electrolyte.lengths_m[1]
                ),
            )
            identity = f"positive:x:{i}"
            bins.append(("positive", i, particle, particle.keys(identity), identity))
        return tuple(bins)

    @cached_property
    def inventory_species(self):
        return {
            METAL_KEY: "Li",
            **{key: "Li" for _, _, _, keys, _ in self.particle_bins for key in keys},
            **dict.fromkeys(self.electrolyte.keys, ELECTROLYTE_SPECIES),
        }

    @property
    def inventory_keys(self):
        return tuple(self.inventory_species)

    @cached_property
    def reaction_keys(self):
        return tuple(
            f"half-cell-dfn:positive:reaction:{i}_A" for i in range(self.electrolyte.cells[1] - 1)
        )

    @cached_property
    def algebraic_names(self):
        return (*self.reaction_keys, PHI_E, *BATTERY_ALGEBRAIC)

    @cached_property
    def reaction_map(self):
        matrix = np.zeros((self.electrolyte.cells[1], len(self.algebraic_names)))
        for i in range(len(self.reaction_keys)):
            matrix[i, i], matrix[-1, i] = 1, -1
        matrix[-1, -2] = -1
        matrix.flags.writeable = False
        return matrix

    @cached_property
    def electrical_maps(self):
        columns = []
        for column in np.eye(len(self.algebraic_names)).T:
            _, _, phi_e, phi_s, _, _ = _linear_electrical(self, column)
            columns.append(phi_s - phi_e[self.electrolyte.region_slices["positive"]])
        result = np.asarray(columns).T
        result.flags.writeable = False
        return self.reaction_map, result

    def initial_coordinates(self):
        return dict.fromkeys(self.coordinate_keys, 0.0)

    def initial_state(self):
        amounts = {METAL_KEY: self.metal.initial_amount_mol, **self.electrolyte.initial_amounts()}
        for _, _, particle, keys, _ in self.particle_bins:
            amounts.update(
                zip(
                    keys,
                    particle.mesh.cell_volumes_m3
                    * particle.maximum_concentration_mol_m3
                    * self.initial_positive_fraction,
                    strict=True,
                )
            )
        return CoupledState(
            amounts, self.temperature_K, self.volume_l, coordinates=self.initial_coordinates()
        )

    def to_dict(self):
        return {
            "model_family": self.model_family,
            "metal": self.metal.to_dict(),
            "positive": asdict(self.positive),
            "positive_ocv": self.positive_ocv.to_dict(),
            "metal_kinetics": self.metal_kinetics.to_dict(),
            "electrolyte": self.electrolyte.to_dict(),
            **{
                key: getattr(self, key)
                for key in (
                    "working_reference_concentration_mol_m3",
                    "initial_positive_fraction",
                    "temperature_K",
                    "surface_margin",
                    "minimum_voltage_V",
                    "maximum_voltage_V",
                    "maximum_current_A",
                    "provenance",
                )
            },
        }

    @classmethod
    def from_dict(cls, value):
        if set(value) != {"model_family", *cls.__dataclass_fields__}:
            raise ValueError("unknown or missing DFN half-cell fields")
        data = dict(value)
        if data.pop("model_family") != HALF_CELL_DFN_FAMILY:
            raise ValueError("unsupported DFN half-cell family")
        return cls(**data)


@dataclass(frozen=True)
class HalfCellDFNReadout:
    voltage_V: float
    current_A: float
    working_surface_fraction: float
    working_average_fraction: float
    metal_amount_mol: float
    metal_overpotential_V: float
    working_overpotential_V: float
    metal_ohmic_V: float
    metal_resistance_ohm: float
    concentration_overpotential_V: float
    electrolyte_ohmic_V: float
    solid_ohmic_V: float

    def __post_init__(self) -> None:
        """Keep public measurements finite and JSON-safe.

        The DFN readout is derived data, so it must never become a second
        device state.  Rejecting non-finite values at this boundary keeps
        runtime meters, snapshots and external JSON consumers deterministic.
        """
        for name in (
            "voltage_V",
            "current_A",
            "working_surface_fraction",
            "working_average_fraction",
            "metal_amount_mol",
            "metal_overpotential_V",
            "working_overpotential_V",
            "metal_ohmic_V",
            "metal_resistance_ohm",
            "concentration_overpotential_V",
            "electrolyte_ohmic_V",
            "solid_ohmic_V",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value):
                raise ValueError(f"DFN half-cell readout {name} must be finite")

    def to_dict(self):
        """Return the stable JSON measurement contract for this model."""
        return {
            "model_family": HALF_CELL_DFN_FAMILY,
            **{
                name: float(getattr(self, name))
                for name in (
                    "voltage_V",
                    "current_A",
                    "working_surface_fraction",
                    "working_average_fraction",
                    "metal_amount_mol",
                    "metal_overpotential_V",
                    "working_overpotential_V",
                    "metal_ohmic_V",
                    "metal_resistance_ohm",
                    "concentration_overpotential_V",
                    "electrolyte_ohmic_V",
                    "solid_ohmic_V",
                )
            },
        }

    @classmethod
    def from_dict(cls, value):
        """Restore a readout while rejecting another model or stray fields."""
        if not isinstance(value, dict):
            raise ValueError("DFN half-cell readout must be an object")
        payload = dict(value)
        if payload.pop("model_family", HALF_CELL_DFN_FAMILY) != HALF_CELL_DFN_FAMILY:
            raise ValueError("unknown DFN half-cell readout family")
        names = {
            "voltage_V",
            "current_A",
            "working_surface_fraction",
            "working_average_fraction",
            "metal_amount_mol",
            "metal_overpotential_V",
            "working_overpotential_V",
            "metal_ohmic_V",
            "metal_resistance_ohm",
            "concentration_overpotential_V",
            "electrolyte_ohmic_V",
            "solid_ohmic_V",
        }
        if set(payload) != names:
            raise ValueError("unknown or missing DFN half-cell readout fields")
        return cls(**payload)


def _linear_electrical(spec, vector):
    """Two-domain geometry map using the original exact linear-field integral."""
    e = spec.electrolyte
    current, voltage, offset = vector[-2], vector[-1], vector[-3]
    selected = np.asarray(vector[: len(spec.reaction_keys)])
    q = np.r_[selected, -current - sum(selected)]
    ns, _ = e.cells
    ionic = np.r_[np.full(ns + 1, current), current + np.cumsum(q)] / e.area_m2
    widths = np.diff(e.mesh.faces_m)
    conductivity = np.repeat(e.conductivity_S_m * e.transport_efficiencies, e.cells)
    resistance, _ = integral_centres(ionic, widths, conductivity)
    phi_e = offset - resistance
    solid_faces = current / e.area_m2 - ionic[ns:]
    integral, full = integral_centres(solid_faces, widths[ns:], e.positive_solid_conductivity_S_m)
    return q, ionic, phi_e, voltage + full - integral, solid_faces, resistance


def _fields(spec, state, values):
    e = spec.electrolyte
    vector = np.array([values[key] for key in spec.algebraic_names])
    current, voltage, offset = vector[-2], vector[-1], vector[-3]
    q, ionic, phi_e, phi_s, solid_faces, resistance = _linear_electrical(spec, vector)
    c = e.concentrations(state)
    boundaries = e.boundary_concentrations(state, current)
    # Smooth finite trials only; signed concentrations still drive stopping and validation.
    trial_c = np.maximum(c, e.minimum_concentration_mol_m3 / 2)
    cb = max(boundaries[0], e.minimum_concentration_mol_m3 / 2)
    thermal = 2 * R * state.temperature_K / F
    alpha = thermal * (1 - e.cation_transference) * e.thermodynamic_factor
    concentration = alpha * np.log(trial_c / cb)
    phi_e += concentration
    ns, _ = e.cells
    surfaces, averages, eta, exchange, ocv, areas = [], [], [], [], [], []
    for i, (_, _, particle, keys, identity) in enumerate(spec.particle_bins):
        theta = particle_surface_fraction(particle, identity, state, q[i] / F)
        trial = np.clip(theta, spec.surface_margin / 2, 1 - spec.surface_margin / 2)
        j0 = particle.exchange_current_scale_A_m2 * np.sqrt(
            trial_c[ns + i] / spec.working_reference_concentration_mol_m3 * trial * (1 - trial)
        )
        surfaces.append(theta)
        averages.append(sum(state.amounts_mol[key] for key in keys) / particle.capacity_mol)
        eta.append(thermal * np.arcsinh(q[i] / (2 * particle.area_m2 * j0)))
        exchange.append(j0)
        ocv.append(spec.positive_ocv.derivatives(theta, trial=True)[0])
        areas.append(particle.area_m2)
    metal = spec.metal.native_parameters
    metal_j0 = (
        metal["exchange_current_density_A_m2"]
        * (cb / metal["electrolyte_concentration_mol_m3"]) ** spec.metal_kinetics.salt_exponent
    )
    metal_eta = thermal * np.arcsinh(current / (2 * metal["area_m2"] * metal_j0))
    positive = e.region_slices["positive"]
    weights = e.mesh.cell_volumes_m3[positive]
    return {
        "current_A": float(current),
        "voltage_V": float(voltage),
        "reaction_current_A": q,
        "interfacial_current_A_m2": q / np.asarray(areas),
        "electrolyte_concentration_mol_m3": c,
        "electrolyte_boundary_concentration_mol_m3": boundaries,
        "metal_boundary_concentration_mol_m3": float(boundaries[0]),
        "electrolyte_potential_V": phi_e,
        "metal_electrolyte_potential_V": float(offset),
        "electrolyte_face_current_A_m2": ionic,
        "positive_solid_face_current_A_m2": solid_faces,
        "solid_potential_V": phi_s,
        "surface_fraction": np.asarray(surfaces),
        "average_fraction": np.asarray(averages),
        "kinetic_overpotential_V": np.asarray(eta),
        "exchange_current_density_A_m2": np.asarray(exchange),
        "open_circuit_potential_V": np.asarray(ocv),
        "metal_exchange_current_density_A_m2": float(metal_j0),
        "metal_amount_mol": state.amounts_mol[METAL_KEY],
        "metal_concentration_mol_m3": 1 / metal["partial_molar_volume_m3_mol"],
        "metal_overpotential_V": float(metal_eta),
        "metal_ohmic_V": float(current * spec.metal.resistance_ohm),
        "metal_boundary_residual_V": float(
            offset + current * spec.metal.resistance_ohm + metal_eta
        ),
        "bv_residual_V": phi_s - phi_e[positive] - np.asarray(ocv) - np.asarray(eta),
        "concentration_overpotential_V": float(
            np.average(concentration[positive], weights=weights)
        ),
        "electrolyte_ohmic_V": -float(np.average(resistance[positive], weights=weights)),
        "solid_ohmic_V": float(voltage - np.average(phi_s, weights=weights)),
    }


def _validate_program(spec, program):
    if not isinstance(spec, BatteryHalfCellDFNSpec):
        raise ValueError("DFN half cell requires its explicit specification")
    if (
        not isinstance(program, ScalarControlProgram)
        or program.unit not in {"A", "V"}
        or any(
            abs(v) > spec.maximum_current_A
            if program.unit == "A"
            else not spec.minimum_voltage_V <= v <= spec.maximum_voltage_V
            for _, v in program.setpoints
        )
    ):
        raise ValueError("DFN half cell requires an in-domain current or voltage programme")
    if program.unit == "V" and not spec.cv_monotone_certified:
        raise ValueError("DFN half-cell CV requires an analytic monotonicity certificate")


def _validate_cv_state(spec, state, program):
    if program.unit != "V":
        return
    base = spec.electrolyte.boundary_concentrations(state, 0.0)[0]
    if not isfinite(base) or base <= 0:
        raise ResidualDomainError("DFN half-cell CV requires a positive reconstructed salt base")
    for _, _, particle, keys, _ in spec.particle_bins:
        theta_bulk = state.amounts_mol[keys[-1]] / (
            particle.mesh.cell_volumes_m3[-1] * particle.maximum_concentration_mol_m3
        )
        if not 0 <= theta_bulk <= 1:
            raise ResidualDomainError("DFN half-cell CV requires last-cell bulk fractions in [0,1]")
    # For each working branch theta=theta_bulk-d*q. In the unclipped kinetic
    # interval, d[q/sqrt(theta*(1-theta))]/dq has the positive numerator
    # theta*(1-theta_bulk)+theta_bulk*(1-theta). Outside it j0 is constant.
    # Combined with the global OCV certificate, the branch is passive for all q,
    # including opposed local currents and unrestricted Newton surface trials.


def half_cell_dfn_algebraic(spec, program):
    _validate_program(spec, program)
    variables = (
        *(
            ResidualVariable(k, "A", spec.maximum_current_A, "algebraic")
            for k in spec.reaction_keys
        ),
        ResidualVariable(PHI_E, "V", 1, "algebraic"),
        ResidualVariable(BATTERY_ALGEBRAIC[0], "A", spec.maximum_current_A, "algebraic"),
        ResidualVariable(BATTERY_ALGEBRAIC[1], "V", 1, "algebraic"),
    )
    equations = (
        *(
            ResidualEquation(f"half-cell-dfn:bv:positive:{i}", "V")
            for i in range(spec.electrolyte.cells[1])
        ),
        ResidualEquation("half-cell-dfn:metal-boundary", "V"),
        ResidualEquation("half-cell-dfn:control", "1"),
    )

    def initial(state):
        _validate_cv_state(spec, state, program)
        current = program.value_at(state.time_s) if program.unit == "A" else 0.0
        q = np.full(len(spec.reaction_keys), -current / spec.electrolyte.cells[1])
        theta = (
            sum(state.amounts_mol[k] for _, _, _, keys, _ in spec.particle_bins for k in keys)
            / spec.positive.capacity_mol
        )
        guess = dict(
            zip(
                spec.algebraic_names,
                [
                    *q,
                    -current * spec.metal.resistance_ohm,
                    current,
                    spec.positive_ocv.derivatives(theta, trial=True)[0],
                ],
                strict=True,
            )
        )
        fields = _fields(spec, state, guess)
        guess[PHI_E] -= fields["metal_overpotential_V"]
        return guess

    def residual(state):
        _validate_cv_state(spec, state, program)
        fields = _fields(spec, state, state.algebraic_values)
        current, voltage = fields["current_A"], fields["voltage_V"]
        target = program.value_at(state.time_s)
        if program.unit == "A":
            control = (current - target) / spec.maximum_current_A
        else:
            requested = current + spec.maximum_current_A * (voltage - target)
            control = (
                current - np.clip(requested, -spec.maximum_current_A, spec.maximum_current_A)
            ) / spec.maximum_current_A
        return np.r_[fields["bv_residual_V"], fields["metal_boundary_residual_V"], control]

    return CoupledAlgebraic(variables, equations, initial, residual)


def _closure_jacobian(spec, state, program, identities):
    index = {key: i for i, key in enumerate(identities)}
    e = spec.electrolyte
    na = len(spec.algebraic_names)
    algebraic_columns = [index[key] for key in spec.algebraic_names]
    qmap, electrical = spec.electrical_maps
    jac = np.zeros((na, len(identities)))
    np_ = e.cells[1]
    jac[:np_, algebraic_columns] = electrical
    fields = _fields(spec, state, state.algebraic_values)
    thermal = 2 * R * state.temperature_K / F
    alpha = thermal * (1 - e.cation_transference) * e.thermodynamic_factor
    cb_raw = fields["metal_boundary_concentration_mol_m3"]
    cb = max(cb_raw, e.minimum_concentration_mol_m3 / 2)
    boundary_gradient = np.zeros(len(identities))
    if cb_raw == cb:
        boundary_map, current_coefficient = e.boundary_maps
        for j, key in enumerate(e.keys):
            if key in index:
                boundary_gradient[index[key]] = (
                    boundary_map[0, j] / e.diffusion.storage_volumes_m3[j]
                )
        boundary_gradient[index[BATTERY_ALGEBRAIC[0]]] = current_coefficient
    for i, (_, _, particle, keys, _) in enumerate(spec.particle_bins):
        cx = e.cells[0] + i
        c = fields["electrolyte_concentration_mol_m3"][cx]
        trial_c = max(c, e.minimum_concentration_mol_m3 / 2)
        theta = fields["surface_fraction"][i]
        trial = np.clip(theta, spec.surface_margin / 2, 1 - spec.surface_margin / 2)
        j0 = fields["exchange_current_density_A_m2"][i]
        q = fields["reaction_current_A"][i]
        z = q / (2 * particle.area_m2 * j0)
        factor = thermal / np.sqrt(1 + z * z)
        eta_q = factor / (2 * particle.area_m2 * j0)
        eta_theta = (
            -factor * z * (1 - 2 * trial) / (2 * trial * (1 - trial)) if theta == trial else 0.0
        )
        slope = spec.positive_ocv.derivatives(theta, trial=True)[1]
        theta_q = -particle.mesh.spherical_surface_distance_m / (
            F * particle.area_m2 * particle.diffusivity_m2_s * particle.maximum_concentration_mol_m3
        )
        jac[i, algebraic_columns] -= ((slope + eta_theta) * theta_q + eta_q) * qmap[i]
        if keys[-1] in index:
            jac[i, index[keys[-1]]] -= (slope + eta_theta) / (
                particle.mesh.cell_volumes_m3[-1] * particle.maximum_concentration_mol_m3
            )
        jac[i] += alpha / cb * boundary_gradient
        if c == trial_c and e.keys[cx] in index:
            jac[i, index[e.keys[cx]]] += (-alpha / c + factor * z / (2 * c)) / (
                e.diffusion.storage_volumes_m3[cx]
            )
        if "temperature" in index:
            jac[i, index["temperature"]] += (
                -alpha * np.log(trial_c / cb) - fields["kinetic_overpotential_V"][i]
            ) / state.temperature_K
    current = fields["current_A"]
    metal_j0 = fields["metal_exchange_current_density_A_m2"]
    z = current / (2 * spec.metal.native_parameters["area_m2"] * metal_j0)
    factor = thermal / np.sqrt(1 + z * z)
    jac[np_] = -factor * z * spec.metal_kinetics.salt_exponent / cb * boundary_gradient
    jac[np_, index[PHI_E]] += 1
    jac[np_, index[BATTERY_ALGEBRAIC[0]]] += spec.metal.resistance_ohm + factor / (
        2 * spec.metal.native_parameters["area_m2"] * metal_j0
    )
    if "temperature" in index:
        jac[np_, index["temperature"]] += fields["metal_overpotential_V"] / state.temperature_K
    requested = current + spec.maximum_current_A * (
        fields["voltage_V"] - program.value_at(state.time_s)
    )
    if program.unit == "A" or abs(requested) >= spec.maximum_current_A:
        jac[-1, index[BATTERY_ALGEBRAIC[0]]] = 1 / spec.maximum_current_A
    else:
        jac[-1, index[BATTERY_ALGEBRAIC[1]]] = -1
    return sparse.csc_matrix(jac)


def half_cell_dfn_fields(spec, state, *, current_A=None, program=None):
    if isinstance(state, CoupledState):
        # Persistent device state must obey the physical declaration before a
        # public readout. Newton views retain their mathematical trial freedom.
        validate_half_cell_dfn_state(spec, state)
    if (current_A is None) == (program is None):
        raise ValueError("supply either a current or a control programme")
    if current_A is not None:
        current_A = _finite(current_A, "current")
        if abs(current_A) > spec.maximum_current_A:
            raise ValueError("current is outside the DFN half-cell source capability")
        program = ScalarControlProgram(((0, current_A),), "A")
    closure = half_cell_dfn_algebraic(spec, program)
    _validate_cv_state(spec, state, program)
    names = spec.algebraic_names
    if all(key in state.algebraic_values for key in names) and (
        current_A is None or state.algebraic_values[BATTERY_ALGEBRAIC[0]] == current_A
    ):
        return _fields(spec, state, state.algebraic_values)

    def view(y):
        return CoupledStateView(
            state.amounts_mol,
            state.temperature_K,
            state.volume_L,
            state.time_s,
            coordinates=state.coordinates,
            algebraic_values=dict(zip(names, y, strict=True)),
        )

    system = ResidualSystem(
        "half-cell-dfn:electrical-closure",
        closure.variables,
        closure.equations,
        lambda t, y, rate, params: closure.residual(view(y)),
        ResidualDerivatives(
            "hybrid",
            state=lambda t, y, rate, params: _closure_jacobian(spec, view(y), program, names),
        ),
    )
    guess = closure.initial_values(state)
    answer = solve_steady_residual(
        system,
        [guess[name] for name in names],
        policy=RootPolicy(tolerance=ELECTRICAL_CLOSURE_TOLERANCE),
    )
    if not answer.root.success:
        raise ResidualNumericalError(
            "DFN half-cell electrical closure:" + answer.root.failure_reason
        )
    fields = _fields(spec, state, dict(zip(names, answer.root.value, strict=True)))
    if abs(fields["current_A"]) > spec.maximum_current_A * (1 + 1e-10):
        raise ResidualNumericalError("DFN half-cell circuit exceeds its actual source bound")
    return {**fields, "closure_diagnostic": answer.root.diagnostics}


def half_cell_dfn_readout(spec, state, current_A):
    fields = half_cell_dfn_fields(spec, state, current_A=current_A)
    weights = np.array([p.active_volume_m3 for _, _, p, _, _ in spec.particle_bins])
    return HalfCellDFNReadout(
        fields["voltage_V"],
        fields["current_A"],
        float(np.average(fields["surface_fraction"], weights=weights)),
        float(np.average(fields["average_fraction"], weights=weights)),
        state.amounts_mol[METAL_KEY],
        fields["metal_overpotential_V"],
        float(np.average(fields["kinetic_overpotential_V"], weights=weights)),
        fields["metal_ohmic_V"],
        spec.metal.resistance_ohm,
        fields["concentration_overpotential_V"],
        fields["electrolyte_ohmic_V"],
        fields["solid_ohmic_V"],
    )


def half_cell_dfn_control_current(spec, state, program):
    return float(half_cell_dfn_fields(spec, state, program=program)["current_A"])


def half_cell_dfn_control_mode(spec, state, program):
    _validate_program(spec, program)
    if program.unit == "A":
        return "constant_current", program.value_at(state.time_s)
    fields = half_cell_dfn_fields(spec, state, program=program)
    target = program.value_at(state.time_s)
    if fields["voltage_V"] > target + 1e-9:
        return "positive_current_limit", spec.maximum_current_A
    if fields["voltage_V"] < target - 1e-9:
        return "negative_current_limit", -spec.maximum_current_A
    return "constant_voltage", target


def validate_half_cell_dfn_state(spec, state):
    if set(state.amounts_mol) != set(spec.inventory_keys) or set(state.coordinates) != set(
        spec.coordinate_keys
    ):
        raise ValueError("DFN half cell requires its unique metal/working/salt and Q/W layout")
    if (
        abs(state.temperature_K - spec.temperature_K) > 1e-8
        or abs(state.volume_L - spec.volume_l) > 1e-12
    ):
        raise ValueError("DFN half cell requires its declared isothermal fixed volume")
    for _, _, particle, keys, _ in spec.particle_bins:
        if any(
            not 0
            <= state.amounts_mol[k]
            <= volume * particle.maximum_concentration_mol_m3 * (1 + 1e-8)
            for k, volume in zip(keys, particle.mesh.cell_volumes_m3, strict=True)
        ):
            raise ValueError("working particle inventory exceeds its finite capacity")
    metal = state.amounts_mol[METAL_KEY]
    charge = state.coordinates["discharged_charge_C"]
    working = sum(state.amounts_mol[k] for _, _, _, keys, _ in spec.particle_bins for k in keys)
    if (
        not isfinite(metal)
        or metal < 0
        or not isclose(
            F * metal + charge, F * spec.metal.initial_amount_mol, abs_tol=1e-8, rel_tol=1e-8
        )
        or not isclose(
            F * working - charge,
            F * spec.positive.capacity_mol * spec.initial_positive_fraction,
            abs_tol=1e-8,
            rel_tol=1e-8,
        )
    ):
        raise ValueError("DFN half-cell metal/working inventory violates its Faraday balance")
    if not isclose(
        sum(state.amounts_mol[k] for k in spec.electrolyte.keys),
        sum(spec.electrolyte.initial_amounts().values()),
        abs_tol=1e-12,
        rel_tol=1e-8,
    ):
        raise ValueError("DFN half-cell salt inventory violates its closed balance")


def half_cell_dfn_system(spec, program, *, thermal_device=None, control_events=()):
    _validate_program(spec, program)
    if thermal_device is not None:
        raise ValueError("DFN half-cell thermal and chemical-energy closure is unsupported")
    e = spec.electrolyte
    processes = [
        ConservativeDiffusion(p.mesh, p.diffusivity_m2_s)
        .couple(keys, process_id=f"half-cell-dfn:{identity}:diffusion", owner_id="battery")
        .process
        for _, _, p, keys, identity in spec.particle_bins
    ]
    processes.append(
        e.diffusion.couple(
            e.keys, process_id="half-cell-dfn:salt-diffusion", owner_id="battery"
        ).process
    )

    def fields(state):
        return half_cell_dfn_fields(spec, state, program=program)

    def source(state):
        f = fields(state)
        current, q = f["current_A"], f["reaction_current_A"]
        amounts = {
            METAL_KEY: -current / F,
            **{
                keys[-1]: -value / F
                for value, (_, _, _, keys, _) in zip(q, spec.particle_bins, strict=True)
            },
        }
        salt = np.r_[np.zeros(e.cells[0]), q] * (1 - e.cation_transference) / F
        salt[0] += (1 - e.cation_transference) * current / F
        amounts.update(zip(e.keys, salt, strict=True))
        return ProcessContribution(
            amount_rates_mol_s=amounts,
            coordinate_rates={
                "discharged_charge_C": current,
                "delivered_work_J": current * f["voltage_V"],
            },
        )

    def validate(state):
        validate_half_cell_dfn_state(spec, state)
        current = fields(state)["current_A"]
        if e.minimum_resolved_concentration(state, current) < e.minimum_concentration_mol_m3 * (
            1 - 1e-7
        ):
            raise ValueError("DFN half-cell salt boundary/profile is outside its positive domain")

    def metal_distance(state):
        # CC has an exact specified direction. A roundoff-positive Newton
        # residual at declared I=0 must not turn a legal empty-metal rest
        # into stripping. Only the direction of the stopping test changes;
        # the solved current, salt flux and inventory equations stay intact.
        current = (
            program.value_at(state.time_s) if program.unit == "A" else fields(state)["current_A"]
        )
        if program.unit == "V" and current > 0 and state.amounts_mol[METAL_KEY] <= 0:
            # At an empty interface, only the independently closed I=0 port
            # can establish numerical rest. A tiny current alone is not a
            # certificate. Use the existing voltage residual tolerance, and
            # change no current, salt source or metal inventory derivative.
            zero = half_cell_dfn_fields(spec, state, current_A=0.0)
            if abs(zero["voltage_V"] - program.value_at(state.time_s)) <= (
                ELECTRICAL_CLOSURE_TOLERANCE
            ):
                return spec.metal.initial_amount_mol
        return state.amounts_mol[METAL_KEY] if current > 0 else spec.metal.initial_amount_mol

    processes.append(
        CoupledProcess("half-cell-dfn:metal-working-salt", source, validate_state=validate)
    )
    events = (
        CoupledEvent(
            "half-cell-dfn:metal:depleted",
            metal_distance,
        ),
        CoupledEvent(
            "half-cell-dfn:working:depleted",
            lambda s: float(np.min(fields(s)["surface_fraction"])) - spec.surface_margin,
        ),
        CoupledEvent(
            "half-cell-dfn:working:saturated",
            lambda s: 1 - spec.surface_margin - float(np.max(fields(s)["surface_fraction"])),
        ),
        CoupledEvent(
            "half-cell-dfn:electrolyte:depleted",
            lambda s: (
                e.minimum_resolved_concentration(s, fields(s)["current_A"])
                - e.minimum_concentration_mol_m3
            ),
        ),
        CoupledEvent(
            "half-cell-dfn:minimum-voltage",
            lambda s: fields(s)["voltage_V"] - spec.minimum_voltage_V,
        ),
        CoupledEvent(
            "half-cell-dfn:maximum-voltage",
            lambda s: spec.maximum_voltage_V - fields(s)["voltage_V"],
        ),
    )
    pkeys = [key for _, _, _, keys, _ in spec.particle_bins for key in keys]
    scales = {
        METAL_KEY: spec.metal.initial_amount_mol,
        **e.initial_amounts(),
        **{
            key: p.capacity_mol / p.cells for _, _, p, keys, _ in spec.particle_bins for key in keys
        },
    }

    @lru_cache(maxsize=8)
    def linear_jacobian(identities):
        index = {key: i for i, key in enumerate(identities)}
        rows, columns, data = [], [], []

        def add(row_ids, column_ids, matrix):
            coo = sparse.coo_matrix(matrix)
            rows.extend(np.asarray(row_ids)[coo.row])
            columns.extend(np.asarray(column_ids)[coo.col])
            data.extend(coo.data)

        acols = [index[key] for key in spec.algebraic_names]
        for i, (_, _, particle, keys, _) in enumerate(spec.particle_bins):
            cols = [index[key] for key in keys]
            add(
                cols,
                cols,
                ConservativeDiffusion(
                    particle.mesh, particle.diffusivity_m2_s
                ).inventory_jacobian_s_inv,
            )
            add([index[keys[-1]]], acols, [-spec.reaction_map[i] / F])
        salt_cols = [index[key] for key in e.keys]
        add(salt_cols, salt_cols, e.diffusion.inventory_jacobian_s_inv)
        salt_map = np.vstack((np.zeros((e.cells[0], len(acols))), spec.reaction_map))
        salt_map[0, -2] += 1
        add(salt_cols, acols, salt_map * (1 - e.cation_transference) / F)
        add([index[METAL_KEY]], [index[BATTERY_ALGEBRAIC[0]]], [[-1 / F]])
        add([index["discharged_charge_C"]], [index[BATTERY_ALGEBRAIC[0]]], [[1]])
        return sparse.csc_matrix((data, (rows, columns)), shape=(len(identities), len(identities)))

    def jacobian(state, identities):
        index = {key: i for i, key in enumerate(identities)}
        arows = np.array([index[key] for key in spec.algebraic_names])
        closure = -_closure_jacobian(spec, state, program, identities).tocoo()
        ci, cv = (index[key] for key in BATTERY_ALGEBRAIC)
        work = index["delivered_work_J"]
        dynamic = sparse.csc_matrix(
            (
                np.r_[
                    closure.data,
                    state.algebraic_values[BATTERY_ALGEBRAIC[1]],
                    state.algebraic_values[BATTERY_ALGEBRAIC[0]],
                ],
                (np.r_[arows[closure.row], work, work], np.r_[closure.col, ci, cv]),
            ),
            shape=(len(identities), len(identities)),
        )
        return linear_jacobian(tuple(identities)) + dynamic

    return CoupledSystem(
        BatteryThermodynamics(spec),
        tuple(processes),
        isothermal=True,
        numerical_form="residual",
        events=(*events, *control_events),
        coordinates=(
            CoupledCoordinate("discharged_charge_C", "C", spec.usable_charge_c),
            CoupledCoordinate("delivered_work_J", "J", spec.usable_charge_c * 4),
        ),
        inventory_scales_mol=scales,
        conservation_laws={
            "lithium_mol": dict.fromkeys(spec.inventory_keys, 1),
            "solid_lithium_mol": dict.fromkeys((METAL_KEY, *pkeys), 1),
            "electrolyte_salt_mol": dict.fromkeys(e.keys, 1),
            "metal_faradaic_C": {METAL_KEY: F, "discharged_charge_C": 1},
            "working_faradaic_C": {**dict.fromkeys(pkeys, F), "discharged_charge_C": -1},
        },
        algebraic=half_cell_dfn_algebraic(spec, program),
        residual_rhs_jacobian=jacobian,
    )


def advance_half_cell_dfn(
    spec,
    state,
    program,
    duration_s,
    *,
    evaluation_times_s=None,
    thermal_device=None,
    control_events=(),
):
    result = half_cell_dfn_system(
        spec, program, thermal_device=thermal_device, control_events=control_events
    ).advance(state, duration_s, evaluation_times_s=evaluation_times_s)
    return replace(
        result,
        solver_diagnostic={
            **result.solver_diagnostic,
            "battery_control_mode": half_cell_dfn_control_mode(spec, result.final_state, program)[
                0
            ],
            "metal_geometry_scope": METAL_CONVENTION,
            "energy_scope": "isothermal; no DFN half-cell chemical-energy closure",
        },
    )


__all__ = [
    "BatteryHalfCellDFNSpec",
    "BatteryHalfCellElectrolyteSpec",
    "HalfCellDFNReadout",
    "LithiumMetalSaltKineticsSpec",
    "advance_half_cell_dfn",
    "half_cell_dfn_algebraic",
    "half_cell_dfn_control_current",
    "half_cell_dfn_control_mode",
    "half_cell_dfn_fields",
    "half_cell_dfn_readout",
    "half_cell_dfn_system",
]
