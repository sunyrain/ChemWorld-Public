"""Finite lithium metal / positive-particle SPM on the common residual solver.

Positive current dissolves metal and inserts lithium into the working particle.
Metal area, thickness, conductivity and exchange density stay fixed: this is a
finite-reservoir, fixed-interface approximation, not a moving metal boundary.
The port law follows the positive-working PyBaMM 25.4.2 SPM convention. The
separate stationary impedance API consumes this original residual. No dynamic
electrolyte, thermal, side-reaction or equilibrium-energy model is implied.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from functools import cached_property
from itertools import pairwise
from math import asinh, isclose, isfinite, sqrt
from types import MappingProxyType
from typing import Any, Self, cast

import numpy as np
from numpy.typing import NDArray
from scipy import sparse

from chemworld.data.scientific_inputs import ScientificTable, _keys, _text
from chemworld.foundation.units import convert_value
from chemworld.physchem.battery_materials import ElectrodeOCV
from chemworld.physchem.battery_models import (
    BATTERY_ALGEBRAIC,
    BATTERY_COORDINATES,
    ParticleSpec,
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
from chemworld.physchem.electrochemistry import FARADAY_C_PER_MOL as F
from chemworld.physchem.electrochemistry import R_J_PER_MOL_K as R
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

METAL_KEY = "metal:Li"
HALF_CELL_FAMILY = "half-cell-spm-isothermal-v1"
METAL_CONVENTION = "fixed-geometry-finite-lithium-v1"
METAL_UNITS = {
    "area_m2": "m^2",
    "thickness_m": "m",
    "conductivity_S_m": "S/m",
    "partial_molar_volume_m3_mol": "m^3/mol",
    "exchange_current_density_A_m2": "A/m^2",
    "reference_temperature_K": "K",
    "electrolyte_concentration_mol_m3": "mol/m^3",
}


@dataclass(frozen=True)
class LithiumMetalCounterElectrodeSpec:
    parameters: ScientificTable
    parameter_row_id: str
    convention: str = METAL_CONVENTION
    _native: Mapping[str, float] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.parameters, ScientificTable) or self.parameters.kind != "parameters":
            raise ValueError("lithium metal requires a sourced parameter table")
        _text(self.parameter_row_id, "parameter_row_id")
        rows = {row.row_id: row for row in self.parameters.rows}
        if self.parameter_row_id not in rows or self.convention != METAL_CONVENTION:
            raise ValueError("lithium metal requires an existing row and fixed-geometry convention")
        cells = rows[self.parameter_row_id].values
        if cells.keys() != METAL_UNITS.keys():
            raise ValueError("lithium metal parameters must exactly match its declared equation")
        if any(v.quality != "valid" or v.value is None for v in cells.values()):
            raise ValueError("lithium metal parameters must be valid finite measurements or inputs")
        native = {k: convert_value(cast(float, cells[k].value), cells[k].unit, u)
                  for k, u in METAL_UNITS.items()}
        if any(not isfinite(v) or v <= 0 for v in native.values()):
            raise ValueError("lithium metal parameters must be positive and finite")
        object.__setattr__(self, "_native", MappingProxyType(native))
        if any(not isfinite(v) or v <= 0 for v in
               (self.volume_m3, self.initial_amount_mol, self.resistance_ohm)):
            raise ValueError("derived lithium metal geometry/inventory/resistance is nonfinite")

    @property
    def native_parameters(self) -> Mapping[str, float]:
        return self._native

    @property
    def volume_m3(self) -> float:
        return self._native["area_m2"] * self._native["thickness_m"]

    @property
    def initial_amount_mol(self) -> float:
        return self.volume_m3 / self._native["partial_molar_volume_m3_mol"]

    @property
    def resistance_ohm(self) -> float:
        return self._native["thickness_m"] / (
            self._native["conductivity_S_m"] * self._native["area_m2"])

    def to_dict(self) -> dict[str, Any]:
        return {"parameters": self.parameters.to_dict(), "parameter_row_id": self.parameter_row_id,
                "convention": self.convention}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Self:
        _keys(value, ("parameters", "parameter_row_id", "convention"))
        return cls(ScientificTable.from_dict(value["parameters"]), value["parameter_row_id"],
                   value["convention"])


@dataclass(frozen=True)
class BatteryHalfCellSPMSpec:
    metal: LithiumMetalCounterElectrodeSpec
    positive: ParticleSpec
    positive_ocv: ElectrodeOCV
    initial_positive_fraction: float = 0.3
    temperature_K: float = 298.15
    electrolyte_concentration_mol_m3: float = 1000.0
    surface_margin: float = 0.01
    minimum_voltage_V: float = 2.5
    maximum_voltage_V: float = 4.5
    maximum_current_A: float = 2.0
    provenance: str = "declared-fixed-geometry-half-cell-v1"

    def __post_init__(self) -> None:
        if isinstance(self.metal, dict):
            object.__setattr__(self, "metal",
                               LithiumMetalCounterElectrodeSpec.from_dict(self.metal))
        positive: object = self.positive
        if isinstance(positive, dict):
            object.__setattr__(self, "positive", ParticleSpec(**positive))
        if isinstance(self.positive_ocv, dict):
            object.__setattr__(self, "positive_ocv", ElectrodeOCV.from_dict(self.positive_ocv))
        if (not isinstance(self.metal, LithiumMetalCounterElectrodeSpec)
                or not isinstance(self.positive, ParticleSpec)
                or not isinstance(self.positive_ocv, ElectrodeOCV)
                or "SPM" not in self.positive_ocv.supported_models):
            raise ValueError(
                "half cell needs explicit metal, positive particle and sourced SPM OCV")
        values = (self.initial_positive_fraction, self.temperature_K,
                  self.electrolyte_concentration_mol_m3, self.surface_margin,
                  self.minimum_voltage_V, self.maximum_voltage_V, self.maximum_current_A)
        if (any(isinstance(v, bool) or not isfinite(v) for v in values)
                or not 0 < self.surface_margin < 0.5
                or not self.surface_margin < self.initial_positive_fraction < 1-self.surface_margin
                or self.temperature_K <= 0 or self.electrolyte_concentration_mol_m3 <= 0
                or self.maximum_current_A <= 0 or self.minimum_voltage_V >= self.maximum_voltage_V):
            raise ValueError("invalid half-cell operating domain")
        _text(self.provenance, "provenance")
        if (self.positive_ocv.stoichiometry_domain[0] > self.surface_margin
                or self.positive_ocv.stoichiometry_domain[1] < 1-self.surface_margin):
            raise ValueError("working OCV source must cover the real stopping surfaces")
        for key, expected in (("reference_temperature_K", self.temperature_K),
                              ("electrolyte_concentration_mol_m3",
                               self.electrolyte_concentration_mol_m3)):
            if not isclose(self.metal.native_parameters[key], expected, abs_tol=1e-8, rel_tol=0):
                raise ValueError(
                    "metal kinetics require their declared temperature and salt condition")

    @property
    def model_family(self) -> str:
        return HALF_CELL_FAMILY

    @property
    def model_id(self) -> str:
        return HALF_CELL_FAMILY

    @property
    def numerical_form(self) -> str:
        return "residual"

    @property
    def thermal(self) -> None:
        return None

    @property
    def sei(self) -> None:
        return None

    @property
    def electrolyte(self) -> None:
        return None

    @property
    def has_directional_ocp(self) -> bool:
        return False

    @property
    def hysteresis_keys(self) -> tuple[()]:
        return ()

    @property
    def coordinate_keys(self) -> tuple[str, ...]:
        return BATTERY_COORDINATES

    @property
    def inventory_keys(self) -> tuple[str, ...]:
        return (METAL_KEY, *self.positive.keys("positive"))

    @property
    def inventory_species(self) -> dict[str, str]:
        return dict.fromkeys(self.inventory_keys, "Li")

    @property
    def volume_l(self) -> float:
        return (self.metal.volume_m3 + self.positive.active_volume_m3) * 1000

    @property
    def usable_charge_c(self) -> float:
        return F * min(self.metal.initial_amount_mol, self.positive.capacity_mol * (
            1-self.surface_margin-self.initial_positive_fraction))

    def current_at_c_rate(self, rate: float) -> float:
        if not isfinite(rate):
            raise ValueError("half-cell C rate must be finite")
        return rate * self.usable_charge_c / 3600

    def initial_coordinates(self) -> dict[str, float]:
        return dict.fromkeys(self.coordinate_keys, 0.0)

    def initial_state(self) -> CoupledState:
        n = {METAL_KEY: self.metal.initial_amount_mol,
             **dict(zip(self.positive.keys("positive"), self.positive.mesh.cell_volumes_m3
                        * self.positive.maximum_concentration_mol_m3
                        * self.initial_positive_fraction, strict=True))}
        return CoupledState(n, self.temperature_K, self.volume_l,
                            coordinates=self.initial_coordinates())

    def to_dict(self) -> dict[str, Any]:
        return {"model_family": self.model_family, "metal": self.metal.to_dict(),
                "positive": asdict(self.positive), "positive_ocv": self.positive_ocv.to_dict(),
                **{key: getattr(self, key) for key in (
                    "initial_positive_fraction", "temperature_K",
                    "electrolyte_concentration_mol_m3",
                    "surface_margin", "minimum_voltage_V", "maximum_voltage_V",
                    "maximum_current_A", "provenance")}}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Self:
        data = dict(value)
        if data.pop("model_family", None) != HALF_CELL_FAMILY:
            raise ValueError("unsupported half-cell family")
        return cls(**data)

    @cached_property
    def cv_monotone_certified(self) -> bool:
        # Nonpositive dU/dtheta plus passive BV gives a unique current at fixed
        # inventories. Include the full current-limit trial extrapolation, not
        # just the accepted source domain. Accepted states still stop at its bounds.
        curve = self.positive_ocv
        if (len(curve.polynomial_V) <= 1 and not curve.exponential_terms
                and not curve.tanh_terms):
            return True
        p = self.positive
        width = self.maximum_current_A * p.mesh.spherical_surface_distance_m / (
            F*p.area_m2*p.diffusivity_m2_s*p.maximum_concentration_mol_m3)
        points = np.linspace(-width, 1+width, 65)
        return all(curve.enclosure(a, b, derivative=True, trial=True).upper <= 0
                   for a, b in pairwise(points))


@dataclass(frozen=True)
class HalfCellReadout:
    voltage_V: float
    current_A: float
    working_surface_fraction: float
    working_average_fraction: float
    metal_amount_mol: float
    metal_overpotential_V: float
    working_overpotential_V: float
    metal_ohmic_V: float
    metal_resistance_ohm: float

    def __post_init__(self) -> None:
        """Keep the diagnostic codec finite so strict JSON remains possible."""
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
        ):
            try:
                finite = isfinite(float(getattr(self, name)))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"half-cell readout {name} must be finite") from exc
            if not finite:
                raise ValueError(f"half-cell readout {name} must be finite")

    def to_dict(self) -> dict[str, str | float]:
        """Return a JSON-safe diagnostic payload without persistent state."""
        return {
            "model_family": HALF_CELL_FAMILY,
            "voltage_V": float(self.voltage_V),
            "current_A": float(self.current_A),
            "working_surface_fraction": float(self.working_surface_fraction),
            "working_average_fraction": float(self.working_average_fraction),
            "metal_amount_mol": float(self.metal_amount_mol),
            "metal_overpotential_V": float(self.metal_overpotential_V),
            "working_overpotential_V": float(self.working_overpotential_V),
            "metal_ohmic_V": float(self.metal_ohmic_V),
            "metal_resistance_ohm": float(self.metal_resistance_ohm),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Self:
        """Decode a readout payload, rejecting a different battery family."""
        payload = dict(value)
        if payload.pop("model_family", HALF_CELL_FAMILY) != HALF_CELL_FAMILY:
            raise ValueError("unknown half-cell readout family")
        return cls(**{key: float(item) for key, item in payload.items()})


def _half_cell_voltage(
    spec: BatteryHalfCellSPMSpec, state: CoupledStateView, current: float,
) -> tuple[HalfCellReadout, tuple[float, float, float]]:
    """Port and its actual partial derivatives, including the surface FV closure."""
    if not isfinite(current) or abs(current) > spec.maximum_current_A:
        raise ValueError("current is outside half-cell source capability")
    p, metal = spec.positive, spec.metal.native_parameters
    theta = particle_surface_fraction(p, "positive", state, -current/F)
    # Same smooth finite trial extension as the existing particle kernel; the
    # real theta drives events. Neither accepted inventory nor theta is projected.
    trial = float(np.clip(theta, spec.surface_margin/2, 1-spec.surface_margin/2))
    j0w = p.exchange_current_scale_A_m2 * sqrt(trial*(1-trial))
    zw = -current/(2*p.area_m2*j0w)
    zm = current/(2*metal["area_m2"]*metal["exchange_current_density_A_m2"])
    thermal = 2*R*state.temperature_K/F
    ew, em = thermal*asinh(zw), thermal*asinh(zm)
    u, slope = spec.positive_ocv.derivatives(theta, trial=True)
    ew_theta = (-thermal*zw/sqrt(1+zw*zw)*(1-2*trial)/(2*trial*(1-trial))
                if theta == trial else 0.0)
    theta_i = p.mesh.spherical_surface_distance_m / (
        F*p.area_m2*p.diffusivity_m2_s*p.maximum_concentration_mol_m3)
    resistance = spec.metal.resistance_ohm
    vi = (slope+ew_theta)*theta_i - thermal/(2*p.area_m2*j0w*sqrt(1+zw*zw))
    vi -= thermal/(2*metal["area_m2"]*metal["exchange_current_density_A_m2"]
                   * sqrt(1+zm*zm)) + resistance
    vn = (slope+ew_theta)/(p.mesh.cell_volumes_m3[-1]*p.maximum_concentration_mol_m3)
    vt = (ew-em)/state.temperature_K
    reading = HalfCellReadout(float(u+ew-em-current*resistance), current, theta,
        sum(state.amounts_mol[k] for k in p.keys("positive"))/p.capacity_mol,
        state.amounts_mol[METAL_KEY], em, ew, current*resistance, resistance)
    return reading, (float(vi), float(vn), float(vt))


def half_cell_readout(
    spec: BatteryHalfCellSPMSpec, state: CoupledStateView, current_A: float,
) -> HalfCellReadout:
    return _half_cell_voltage(spec, state, current_A)[0]


def half_cell_control_mode(
    spec: BatteryHalfCellSPMSpec, state: CoupledStateView, program: ScalarControlProgram,
) -> tuple[str, float]:
    target = program.value_at(state.time_s)
    if program.unit == "A":
        return "constant_current", target
    if program.unit != "V" or not spec.cv_monotone_certified:
        raise ValueError("half-cell CV requires a certified monotone working OCV")
    limit = spec.maximum_current_A
    if half_cell_readout(spec, state, -limit).voltage_V < target:
        return "negative_current_limit", -limit
    if half_cell_readout(spec, state, limit).voltage_V > target:
        return "positive_current_limit", limit
    return "constant_voltage", target


def half_cell_algebraic(
    spec: BatteryHalfCellSPMSpec, program: ScalarControlProgram,
) -> CoupledAlgebraic:
    if program.unit == "V" and not spec.cv_monotone_certified:
        raise ValueError("half-cell CV requires a certified monotone working OCV")
    ik, vk = BATTERY_ALGEBRAIC
    variables = (ResidualVariable(ik, "A", spec.maximum_current_A, "algebraic",
                                  -spec.maximum_current_A, spec.maximum_current_A),
                 ResidualVariable(vk, "V", 1.0, "algebraic"))
    equations = (ResidualEquation("half-cell:control", "1"),
                 ResidualEquation("half-cell:terminal-voltage", "V"))

    def initial(state: CoupledStateView) -> dict[str, float]:
        mode, value = half_cell_control_mode(spec, state, program)
        current = 0.0 if mode == "constant_voltage" else value
        return {ik: current, vk: half_cell_readout(spec, state, current).voltage_V}

    def residual(state: CoupledStateView) -> list[float]:
        current, voltage = (state.algebraic_values[k] for k in BATTERY_ALGEBRAIC)
        mode, target = half_cell_control_mode(spec, state, program)
        control = voltage-target if mode == "constant_voltage" else (
            (current-target)/spec.maximum_current_A)
        return [control, voltage-half_cell_readout(spec, state, current).voltage_V]

    return CoupledAlgebraic(variables, equations, initial, residual)


def half_cell_control_current(
    spec: BatteryHalfCellSPMSpec, state: CoupledStateView, program: ScalarControlProgram,
) -> float:
    if BATTERY_ALGEBRAIC[0] in state.algebraic_values:
        return state.algebraic_values[BATTERY_ALGEBRAIC[0]]
    mode, target = half_cell_control_mode(spec, state, program)
    if mode != "constant_voltage":
        if abs(target) > spec.maximum_current_A:
            raise ValueError("current is outside half-cell source capability")
        return target
    closure = half_cell_algebraic(spec, program)
    guess = closure.initial_values(state)

    def residual(
        t: float, y: NDArray[np.float64], rate: NDArray[np.float64], params: NDArray[np.float64],
    ) -> Sequence[float] | NDArray[np.float64]:
        view = CoupledStateView(state.amounts_mol, state.temperature_K, state.volume_L,
            state.time_s, coordinates=state.coordinates,
            algebraic_values=dict(zip(BATTERY_ALGEBRAIC, y, strict=True)))
        return closure.residual(view)

    def jacobian(
        t: float, y: NDArray[np.float64], rate: NDArray[np.float64], params: NDArray[np.float64],
    ) -> list[list[float]]:
        return [[0.0, 1.0], [-_half_cell_voltage(spec, state, y[0])[1][0], 1.0]]

    system = ResidualSystem("half-cell:electrical-closure", closure.variables, closure.equations,
        residual, ResidualDerivatives("hybrid", state=jacobian))
    answer = solve_steady_residual(system, [guess[k] for k in BATTERY_ALGEBRAIC],
                                   policy=RootPolicy(tolerance=1e-12))
    if not answer.root.success:
        raise ResidualNumericalError("half-cell control:"+answer.root.failure_reason)
    return float(answer.root.value[0])


def validate_half_cell_state(spec: BatteryHalfCellSPMSpec, state: CoupledStateView) -> None:
    if (set(state.amounts_mol) != set(spec.inventory_keys)
            or set(state.coordinates) != set(spec.coordinate_keys)):
        raise ValueError("half-cell state must use its unique metal/working inventory and Q/W")
    if (abs(state.temperature_K-spec.temperature_K) > 1e-8
            or abs(state.volume_L-spec.volume_l) > 1e-12):
        raise ValueError("half cell requires its declared isothermal fixed volume")
    p = spec.positive
    for key, volume in zip(p.keys("positive"), p.mesh.cell_volumes_m3, strict=True):
        if not 0 <= state.amounts_mol[key] <= volume*p.maximum_concentration_mol_m3*(1+1e-8):
            raise ValueError("working particle inventory is outside its finite capacity")
    metal = state.amounts_mol[METAL_KEY]
    if not isfinite(metal) or metal < 0:
        raise ValueError("finite lithium metal cannot have negative inventory")
    charge = state.coordinates["discharged_charge_C"]
    working = sum(state.amounts_mol[k] for k in p.keys("positive"))
    if (not isclose(F*metal+charge, F*spec.metal.initial_amount_mol, abs_tol=1e-8, rel_tol=1e-8)
            or not isclose(F*working-charge, F*p.capacity_mol*spec.initial_positive_fraction,
                           abs_tol=1e-8, rel_tol=1e-8)):
        raise ValueError(
            "half-cell inventory and passed charge violate the declared Faraday balance")


class _HalfCellThermodynamics:
    def __init__(self, spec: BatteryHalfCellSPMSpec) -> None:
        self.inventory_keys = spec.inventory_keys

    def heat_capacity_j_k(self, state: CoupledStateView) -> float:
        return 1.0

    def internal_energy_j(self, state: CoupledStateView) -> None:
        return None

    def chemical_energy_rate_w(
        self, state: CoupledStateView, rates: Mapping[str, float],
        contributions: Sequence[ProcessContribution],
    ) -> float:
        return 0.0


def _half_cell_rhs_jacobian(
    spec: BatteryHalfCellSPMSpec, state: CoupledStateView, program: ScalarControlProgram,
    identities: Sequence[str],
) -> sparse.csc_matrix:
    index = {key: i for i, key in enumerate(identities)}
    jac = np.zeros((len(identities), len(identities)))
    keys = spec.positive.keys("positive")
    cols = [index[k] for k in keys]
    jac[np.ix_(cols, cols)] = ConservativeDiffusion(
        spec.positive.mesh, spec.positive.diffusivity_m2_s).inventory_jacobian_s_inv.toarray()
    ik, vk = BATTERY_ALGEBRAIC
    current = state.algebraic_values[ik]
    reading, (vi, vn, vt) = _half_cell_voltage(spec, state, current)
    jac[index[METAL_KEY], index[ik]] = -1/F
    jac[index[keys[-1]], index[ik]] = 1/F
    dv = np.zeros(len(identities))
    dv[index[ik]], dv[index[keys[-1]]] = vi, vn
    if "temperature" in index:
        dv[index["temperature"]] = vt
    jac[index["discharged_charge_C"], index[ik]] = 1.0
    jac[index["delivered_work_J"]] = current*dv
    jac[index["delivered_work_J"], index[ik]] += reading.voltage_V
    mode, _ = half_cell_control_mode(spec, state, program)
    if mode == "constant_voltage":
        jac[index[ik], index[vk]] = -1.0
    else:
        jac[index[ik], index[ik]] = -1/spec.maximum_current_A
    jac[index[vk]] = dv
    jac[index[vk], index[vk]] -= 1.0
    return sparse.csc_matrix(jac)


def half_cell_system(
    spec: BatteryHalfCellSPMSpec, program: ScalarControlProgram, *,
    thermal_device: object | None = None, control_events: Sequence[CoupledEvent] = (),
) -> CoupledSystem:
    if not isinstance(spec, BatteryHalfCellSPMSpec):
        raise ValueError("half-cell advance requires its explicit specification")
    if thermal_device is not None:
        raise ValueError("half-cell thermal and chemical-energy closure is not implemented")
    if not isinstance(program, ScalarControlProgram) or program.unit not in {"A", "V"} or any(
        (abs(value) > spec.maximum_current_A if program.unit == "A"
         else not spec.minimum_voltage_V <= value <= spec.maximum_voltage_V)
        for _, value in program.setpoints
    ):
        raise ValueError("half cell requires an in-domain current or voltage programme")
    p, keys = spec.positive, spec.positive.keys("positive")
    diffusion = ConservativeDiffusion(p.mesh, p.diffusivity_m2_s).couple(
        keys, process_id="half-cell:working-diffusion", owner_id="positive").process

    def transfer(state: CoupledStateView) -> ProcessContribution:
        current = half_cell_control_current(spec, state, program)
        read = half_cell_readout(spec, state, current)
        source, target = (METAL_KEY, keys[-1]) if current >= 0 else (keys[-1], METAL_KEY)
        return ProcessContribution(internal_transfers=(
            InternalTransfer("half-cell:Li", source, target, abs(current)/F),),
            coordinate_rates={"discharged_charge_C": current,
                              "delivered_work_J": current*read.voltage_V})

    def current(state: CoupledStateView) -> float:
        return half_cell_control_current(spec, state, program)

    def surface(state: CoupledStateView) -> float:
        return particle_surface_fraction(p, "positive", state, -current(state)/F)

    events = [
        # An empty fixed interface can rest or plate in reverse. This switches
        # only the *stopping test*, never the current, Li balance or metal stock.
        CoupledEvent("half-cell:metal:depleted", lambda s: s.amounts_mol[METAL_KEY]
                     if current(s) > 0 else spec.metal.initial_amount_mol),
        CoupledEvent("half-cell:working:depleted", lambda s: surface(s)-spec.surface_margin),
        CoupledEvent("half-cell:working:saturated", lambda s: 1-spec.surface_margin-surface(s)),
        CoupledEvent("half-cell:minimum-voltage", lambda s:
                     half_cell_readout(spec, s, current(s)).voltage_V-spec.minimum_voltage_V),
        CoupledEvent("half-cell:maximum-voltage", lambda s:
                     spec.maximum_voltage_V-half_cell_readout(spec, s, current(s)).voltage_V),
    ]
    if program.unit == "V":
        def voltage_event(value: float) -> Callable[[CoupledStateView], float]:
            def event(state: CoupledStateView) -> float:
                return (half_cell_readout(spec, state, value).voltage_V
                        - program.value_at(state.time_s))

            return event

        events.extend(CoupledEvent(f"half-cell:cv:{label}-limit", voltage_event(value),
            direction=0, terminal=False) for label, value in (
                ("negative", -spec.maximum_current_A), ("positive", spec.maximum_current_A)))
    return CoupledSystem(
        _HalfCellThermodynamics(spec),
        (diffusion, CoupledProcess("half-cell:metal-working-exchange", transfer,
                                  validate_state=lambda s: validate_half_cell_state(spec, s))),
        isothermal=True, numerical_form="residual", events=(*events, *control_events),
        coordinates=(CoupledCoordinate("discharged_charge_C", "C", spec.usable_charge_c),
                     CoupledCoordinate("delivered_work_J", "J", spec.usable_charge_c*4)),
        inventory_scales_mol={METAL_KEY: spec.metal.initial_amount_mol,
                             **dict.fromkeys(keys, p.capacity_mol/p.cells)},
        conservation_laws={
            "lithium_mol": dict.fromkeys(spec.inventory_keys, 1.0),
            "metal_faradaic_C": {METAL_KEY: F, "discharged_charge_C": 1.0},
            "working_faradaic_C": {**dict.fromkeys(keys, F), "discharged_charge_C": -1.0}},
        algebraic=half_cell_algebraic(spec, program),
        residual_rhs_jacobian=lambda s, identities:
            _half_cell_rhs_jacobian(spec, s, program, identities),
    )


def advance_half_cell(
    spec: BatteryHalfCellSPMSpec, state: CoupledState, program: ScalarControlProgram,
    duration_s: float, *, evaluation_times_s: Sequence[float] | None = None,
    thermal_device: object | None = None, control_events: Sequence[CoupledEvent] = (),
) -> CoupledAdvanceResult:
    result = half_cell_system(spec, program, thermal_device=thermal_device,
        control_events=control_events).advance(state, duration_s,
                                               evaluation_times_s=evaluation_times_s)
    return replace(result, solver_diagnostic={**result.solver_diagnostic,
        "battery_control_mode": half_cell_control_mode(spec, result.final_state, program)[0],
        "metal_geometry_scope": METAL_CONVENTION,
        "energy_scope": "isothermal; no half-cell chemical-energy closure"})


__all__ = ["BatteryHalfCellSPMSpec", "HalfCellReadout", "LithiumMetalCounterElectrodeSpec",
           "advance_half_cell", "half_cell_algebraic", "half_cell_control_current",
           "half_cell_control_mode", "half_cell_readout", "half_cell_system"]
