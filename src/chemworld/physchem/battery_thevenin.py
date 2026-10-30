"""Sourced empirical Thevenin RC/thermal equations on the common residual kernel.

Equation source: PyBaMM 25.4.2 equivalent_circuit_elements (BSD-3-Clause).
Positive current discharges; v_k is the positive voltage drop, not the upstream
negative eta_k. Actual cell/jig temperatures are device-owned coordinates. The
external fixed temperature is a formal reference frame, not either body.

The empirical heat law does not identify chemical species or a total chemical
energy function. Global thermodynamic energy closure is explicitly undefined;
charge and the declared two-node heat equation have separate exact balances.
Optional spatial SOC diffusion uses the official PR 5869 voltage/heat correction.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from math import isfinite
from types import MappingProxyType
from typing import Any, NotRequired, Self, TypedDict, cast

import numpy as np

from chemworld.data.scientific_inputs import ScientificTable, _keys, _sequence, _text
from chemworld.foundation.units import convert_value
from chemworld.physchem.battery_thevenin_control import (
    CONTROLLER_CURRENT,
    DYNAMIC_MODES,
    TheveninControlProgram,
    TheveninControlSpec,
    isolate_control_current,
    thevenin_control_program,
)
from chemworld.physchem.battery_thevenin_diffusion import TheveninDiffusionSpec
from chemworld.physchem.coupled_system import (
    CoupledAlgebraic,
    CoupledCoordinate,
    CoupledEvent,
    CoupledProcess,
    CoupledState,
    CoupledStateView,
    CoupledSystem,
    KineticOnlyThermodynamics,
    ProcessContribution,
)
from chemworld.physchem.reactor_shared import ScalarControlProgram
from chemworld.physchem.residual_system import (
    ResidualDomainError,
    ResidualEquation,
    ResidualVariable,
)
from chemworld.physchem.tabulated_functions import TabulatedScalar

THEVENIN_ID = "thevenin-empirical-rc-thermal-v1"
SOC = "thevenin:soc"
CELL_T = "thevenin:cell_temperature_K"
JIG_T = "thevenin:jig_temperature_K"
CHARGE = "discharged_charge_C"
WORK = "delivered_work_J"
THROUGHPUT_Q = "thevenin:throughput_charge_C"
THROUGHPUT_E = "thevenin:throughput_work_J"
MODEL_HEAT = "thevenin:empirical_heat_J"
AMBIENT_LOSS = "thevenin:ambient_loss_J"
CURRENT, VOLTAGE = "thevenin:current_A", "thevenin:voltage_V"

CONSTANT_UNITS = {
    "capacity_C": "C", "maximum_current_A": "A",
    "minimum_voltage_V": "V", "maximum_voltage_V": "V",
    "initial_soc": "dimensionless", "minimum_soc": "dimensionless",
    "maximum_soc": "dimensionless", "initial_cell_temperature_K": "K",
    "initial_jig_temperature_K": "K", "minimum_temperature_K": "K",
    "maximum_temperature_K": "K", "cell_heat_capacity_J_K": "J/K",
    "jig_heat_capacity_J_K": "J/K", "cell_jig_conductance_W_K": "W/K",
    "jig_ambient_conductance_W_K": "W/K", "reference_frame_temperature_K": "K",
    "housing_volume_L": "L",
}
R_INPUTS = {"soc": "dimensionless", "temperature_K": "K", "current_A": "A"}


def rc_key(index: int) -> str:
    return f"thevenin:polarization_{index}_V"


def _law(law: TabulatedScalar, values: Mapping[str, float]) -> float:
    return law.value({name: values[name] for name, _, _ in law.axes}, trial=True)


def _check_law(
    law: object, unit: str, allowed_inputs: Mapping[str, str],
    domains: Mapping[str, tuple[float, float]],
) -> None:
    if not isinstance(law, TabulatedScalar) or law.output_unit != unit:
        raise ValueError(f"Thevenin function requires its explicit native output unit {unit}")
    for name, _, input_unit in law.axes:
        if name not in allowed_inputs or input_unit != allowed_inputs[name]:
            raise ValueError("Thevenin function input identity/unit is incompatible")
        lower, upper = law.domain[name]
        if lower > domains[name][0] or upper < domains[name][1]:
            raise ValueError("Thevenin data must cover the declared operating domain")


@dataclass(frozen=True)
class TheveninSpec:
    parameters: ScientificTable
    parameter_row_id: str
    ocv: TabulatedScalar
    series_resistance: TabulatedScalar
    rc_elements: tuple[tuple[TabulatedScalar, TabulatedScalar], ...]
    entropic_change: TabulatedScalar
    ambient_program: ScalarControlProgram
    diffusion: TheveninDiffusionSpec | None = None
    control: TheveninControlSpec | None = None
    _native: Mapping[str, float] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.parameters, ScientificTable) or self.parameters.kind != "parameters":
            raise ValueError("Thevenin constants require a sourced parameter table")
        _text(self.parameter_row_id, "parameter_row_id")
        rows = {row.row_id: row for row in self.parameters.rows}
        if self.parameter_row_id not in rows:
            raise ValueError("Thevenin parameter row does not exist")
        elements = tuple(tuple(_sequence(pair, "RC pair"))
                         for pair in _sequence(self.rc_elements, "rc_elements"))
        if len(elements) > 64 or any(len(pair) != 2 for pair in elements):
            raise ValueError("Thevenin supports zero through 64 explicit R/C pairs")
        required = {**CONSTANT_UNITS,
                    **{f"initial_rc_{i}_V": "V" for i in range(len(elements))}}
        row = rows[self.parameter_row_id]
        if row.values.keys() != required.keys():
            raise ValueError("Thevenin constants must match the selected RC model exactly")
        values = {key: convert_value(cast(float, row.values[key].value), row.values[key].unit, unit)
                  for key, unit in required.items()}
        if any(not isfinite(value) for value in values.values()):
            raise ValueError("converted Thevenin parameters must remain finite")
        positive = ("capacity_C", "maximum_current_A", "cell_heat_capacity_J_K",
                    "jig_heat_capacity_J_K", "reference_frame_temperature_K", "housing_volume_L")
        if any(values[key] <= 0 for key in positive):
            raise ValueError("Thevenin capacity, limits, thermal masses and frame must be positive")
        if any(values[key] < 0 for key in ("cell_jig_conductance_W_K",
                                           "jig_ambient_conductance_W_K")):
            raise ValueError("Thevenin thermal conductances cannot be negative")
        if not 0 < values["minimum_voltage_V"] < values["maximum_voltage_V"]:
            raise ValueError("Thevenin voltage limits must be positive and increasing")
        if not (0 <= values["minimum_soc"] < values["initial_soc"]
                < values["maximum_soc"] <= 1):
            raise ValueError("Thevenin initial SOC must lie inside its physical stopping bounds")
        lo, hi = values["minimum_temperature_K"], values["maximum_temperature_K"]
        if not (0 < lo < values["initial_cell_temperature_K"] < hi
                and lo < values["initial_jig_temperature_K"] < hi):
            raise ValueError("Thevenin body temperatures must lie inside positive stopping bounds")
        if (not isinstance(self.ambient_program, ScalarControlProgram)
                or self.ambient_program.unit != "K"
                or any(value <= 0 for _, value in self.ambient_program.setpoints)):
            raise ValueError(
                "Thevenin ambient boundary requires an explicit positive kelvin program")
        if self.control is not None and not isinstance(self.control, TheveninControlSpec):
            raise ValueError("Thevenin controller must have an explicit sourced specification")
        domains = {"soc": (values["minimum_soc"], values["maximum_soc"]),
                   "temperature_K": (lo, hi),
                   "current_A": (-values["maximum_current_A"], values["maximum_current_A"])}
        _check_law(self.ocv, "V", {"soc": "dimensionless"}, domains)
        if self.diffusion is not None:
            if not isinstance(self.diffusion, TheveninDiffusionSpec):
                raise ValueError("Thevenin diffusion must have an explicit sourced specification")
            if self.ocv.axes and np.any(np.diff(self.ocv._values) < 0):
                raise ValueError("passive SOC diffusion requires nondecreasing OCV with SOC")
        _check_law(self.series_resistance, "Ohm", R_INPUTS, domains)
        if self.series_resistance.value_range[0] < 0:
            raise ValueError("Thevenin series resistance cannot be negative")
        for resistance, capacitance in elements:
            _check_law(resistance, "Ohm", R_INPUTS, domains)
            _check_law(capacitance, "F", R_INPUTS, domains)
            if min(resistance.value_range[0], capacitance.value_range[0]) <= 0:
                raise ValueError("RC elements require strictly positive R and C over their data")
        domains["ocv"] = self.ocv.value_range
        _check_law(self.entropic_change, "V/K", {"ocv": "V", "temperature_K": "K"}, domains)
        object.__setattr__(self, "rc_elements", elements)
        object.__setattr__(self, "_native", MappingProxyType(values))

    @property
    def native_parameters(self) -> Mapping[str, float]:
        return self._native

    @property
    def coordinate_declarations(self) -> tuple[CoupledCoordinate, ...]:
        p = self._native
        # Event surfaces retain the exact authored stopping bounds. Dense-output
        # root evaluation can land a few floating-point units to either side;
        # numerical validity allows roundoff, never clips the accepted state or
        # permits advancement past a stopping event. This is not a data domain
        # extension or a change to any comparison/qualification tolerance.
        soc_guard = 32*np.spacing(1.)
        temperature_guard = 32*np.spacing(max(
            p["maximum_temperature_K"], p["reference_frame_temperature_K"]))
        work_scale = p["capacity_C"]*p["maximum_voltage_V"]
        heat_scale = (p["cell_heat_capacity_J_K"]+p["jig_heat_capacity_J_K"])*(
            p["maximum_temperature_K"]-p["minimum_temperature_K"])
        return (
            CoupledCoordinate(SOC, "dimensionless", 1., p["minimum_soc"]-soc_guard,
                              p["maximum_soc"]+soc_guard),
            *(CoupledCoordinate(rc_key(i), "V", p["maximum_voltage_V"])
              for i in range(len(self.rc_elements))),
            *(self.diffusion.coordinate_declarations if self.diffusion is not None else ()),
            *(self.control.coordinate_declarations if self.control is not None else ()),
            *(CoupledCoordinate(key, "K", p["reference_frame_temperature_K"],
                                p["minimum_temperature_K"]-temperature_guard,
                                p["maximum_temperature_K"]+temperature_guard)
              for key in (CELL_T, JIG_T)),
            CoupledCoordinate(CHARGE, "C", p["capacity_C"]),
            CoupledCoordinate(WORK, "J", work_scale),
            CoupledCoordinate(THROUGHPUT_Q, "C", p["capacity_C"], 0.),
            CoupledCoordinate(THROUGHPUT_E, "J", work_scale, 0.),
            CoupledCoordinate(MODEL_HEAT, "J", heat_scale),
            CoupledCoordinate(AMBIENT_LOSS, "J", heat_scale),
        )

    def initial_state(self) -> CoupledState:
        p = self._native
        coordinates = dict.fromkeys((item.name for item in self.coordinate_declarations), 0.)
        coordinates.update({SOC: p["initial_soc"], CELL_T: p["initial_cell_temperature_K"],
                            JIG_T: p["initial_jig_temperature_K"]})
        coordinates.update({rc_key(i): p[f"initial_rc_{i}_V"]
                            for i in range(len(self.rc_elements))})
        if self.control is not None:
            coordinates[CONTROLLER_CURRENT] = self.control.native_parameters["initial_current_A"]
        return CoupledState({}, p["reference_frame_temperature_K"], p["housing_volume_L"],
                            coordinates=coordinates)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
                "model_family": THEVENIN_ID, "parameters": self.parameters.to_dict(),
                "parameter_row_id": self.parameter_row_id, "ocv": self.ocv.to_dict(),
                "series_resistance": self.series_resistance.to_dict(),
                "rc_elements": [{"resistance": r.to_dict(), "capacitance": c.to_dict()}
                                for r, c in self.rc_elements],
                "entropic_change": self.entropic_change.to_dict(),
                "ambient_program": self.ambient_program.to_dict()}
        if self.diffusion is not None:
            payload["diffusion"] = self.diffusion.to_dict()
        if self.control is not None:
            payload["control"] = self.control.to_dict()
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Self:
        _keys(payload, (
            "model_family", "parameters", "parameter_row_id", "ocv", "series_resistance",
            "rc_elements", "entropic_change", "ambient_program"), optional=("diffusion", "control"))
        if payload["model_family"] != THEVENIN_ID:
            raise ValueError("unknown Thevenin model identity")
        pairs = []
        for pair in _sequence(payload["rc_elements"], "rc_elements"):
            _keys(pair, ("resistance", "capacitance"))
            pairs.append((TabulatedScalar.from_dict(pair["resistance"]),
                          TabulatedScalar.from_dict(pair["capacitance"])))
        _keys(payload["ambient_program"], ("setpoints", "unit", "mode"))
        return cls(ScientificTable.from_dict(payload["parameters"]), payload["parameter_row_id"],
                   TabulatedScalar.from_dict(payload["ocv"]),
                   TabulatedScalar.from_dict(payload["series_resistance"]), tuple(pairs),
                   TabulatedScalar.from_dict(payload["entropic_change"]),
                   ScalarControlProgram(**payload["ambient_program"]),
                   TheveninDiffusionSpec.from_dict(payload["diffusion"])
                   if "diffusion" in payload else None,
                   TheveninControlSpec.from_dict(payload["control"])
                   if "control" in payload else None)


class TheveninFields(TypedDict):
    current_A: float
    voltage_V: float
    ocv_V: float
    r0_ohm: float
    resistance_ohm: tuple[float, ...]
    capacitance_F: tuple[float, ...]
    polarization_V: tuple[float, ...]
    soc: float
    distributed_soc: tuple[float, ...]
    surface_soc: float
    diffusion_overpotential_V: float
    diffusion_heat_W: float
    cell_temperature_K: float
    jig_temperature_K: float
    ambient_temperature_K: float
    empirical_heat_W: float
    cell_to_jig_W: float
    ambient_loss_W: float
    controller_current_A: NotRequired[float]


def _fields(spec: TheveninSpec, state: CoupledStateView, current: float) -> TheveninFields:
    p, x = spec._native, state.coordinates
    values = {"soc": x[SOC], "temperature_K": x[CELL_T], "current_A": current}
    ocv = _law(spec.ocv, values)
    values["ocv"] = ocv
    r0 = _law(spec.series_resistance, values)
    resistance = tuple(_law(r, values) for r, _ in spec.rc_elements)
    capacitance = tuple(_law(c, values) for _, c in spec.rc_elements)
    if r0 < 0 or any(value <= 0 for value in (*resistance, *capacitance)):
        raise ResidualDomainError("numerical trial leaves the positive R/C continuation")
    polarization = tuple(x[rc_key(i)] for i in range(len(spec.rc_elements)))
    profile = (x[SOC],) if spec.diffusion is None else spec.diffusion.profile(x[SOC], x)
    surface = x[SOC] if spec.diffusion is None else spec.diffusion.surface_soc(x[SOC], x)
    # Official PyBaMM PR 5869: eta_D = U(surface)-U(mean), Q_D = -I*eta_D.
    # After current reversal Q_D may be negative as a stored gradient relaxes.
    diffusion_overpotential = _law(spec.ocv, {"soc": surface})-ocv
    diffusion_heat = -current*diffusion_overpotential
    voltage = ocv+diffusion_overpotential-current*r0-sum(polarization)
    empirical_heat = (current**2*r0+current*sum(polarization)+diffusion_heat
                      - current*x[CELL_T]*_law(spec.entropic_change, values))
    ambient = spec.ambient_program.value_at(state.time_s)
    return {"current_A": current, "voltage_V": voltage, "ocv_V": ocv, "r0_ohm": r0,
            "resistance_ohm": resistance, "capacitance_F": capacitance,
            "polarization_V": polarization, "soc": x[SOC],
            "distributed_soc": tuple(float(v) for v in profile), "surface_soc": surface,
            "diffusion_overpotential_V": diffusion_overpotential,
            "diffusion_heat_W": diffusion_heat,
            "cell_temperature_K": x[CELL_T], "jig_temperature_K": x[JIG_T],
            "ambient_temperature_K": ambient, "empirical_heat_W": empirical_heat,
            "cell_to_jig_W": p["cell_jig_conductance_W_K"]*(x[CELL_T]-x[JIG_T]),
            "ambient_loss_W": p["jig_ambient_conductance_W_K"]*(x[JIG_T]-ambient)}


def thevenin_system(
    spec: TheveninSpec,
    program: TheveninControlProgram | ScalarControlProgram | Mapping[str, Any],
    *, control_events: Sequence[CoupledEvent] = (),
) -> CoupledSystem:
    """Build named actual port/controller equations on the common residual."""
    program = thevenin_control_program(program)
    mode = program.control_mode
    p = spec._native
    limit, voltage_scale = p["maximum_current_A"], p["maximum_voltage_V"]
    if any((program.unit == "A" and abs(value) > limit)
           or (program.unit == "V" and not
               p["minimum_voltage_V"] <= value <= p["maximum_voltage_V"])
           for _, value in program.setpoints):
        raise ValueError("Thevenin target exceeds the physical device bounds")
    if program.current_limit is not None and any(abs(value) > limit
                                               for _, value in program.current_limit.setpoints):
        raise ValueError("CCCV current limit exceeds the physical source capability")
    if program.current_branch_A is not None and not (
            -limit <= program.current_branch_A[0] < program.current_branch_A[1] <= limit):
        raise ValueError("declared current branch exceeds the physical source capability")
    if mode in DYNAMIC_MODES and spec.control is None:
        raise ValueError("dynamic mode requires an installed persistent controller state")
    if mode == "voltage" and not spec.series_resistance.positive_coordinate_product_derivative(
        "current_A"
    ):
        raise ValueError("CV requires a certified nonsingular monotone instantaneous I*R0 law")

    def saturated_current(state: CoupledStateView, target: float) -> float | None:
        # The certified positive derivative of I*R0 makes V(I) strictly
        # decreasing at a fixed physical state, including current-dependent R.
        # Endpoint voltages therefore identify the active source constraint
        # without solving another equation or changing the finite-source law.
        if _fields(spec, state, limit)["voltage_V"] >= target:
            return limit
        if _fields(spec, state, -limit)["voltage_V"] <= target:
            return -limit
        return None

    cached_key: tuple[float, tuple[tuple[str, float], ...]] | None = None
    cached_current: float | None = None

    def requested_current(state: CoupledStateView) -> float | None:
        nonlocal cached_key, cached_current
        target = program.value_at(state.time_s)
        if mode == "current":
            return target
        if mode in DYNAMIC_MODES:
            q = state.coordinates[CONTROLLER_CURRENT]
            return max(q, cast(ScalarControlProgram, program.current_limit).value_at(state.time_s)
                       ) if mode == "cccv" else q
        if mode == "voltage":
            return saturated_current(state, target)
        # Algebraic power/load roots use the shared certified isolator. This
        # cache only reuses the same fixed physical state while consistency
        # perturbs I/V; it is not a second persistent state or a time solver.
        key = (state.time_s, tuple(state.coordinates.items()))
        if key != cached_key:
            inputs = {"soc": state.coordinates[SOC], "temperature_K": state.coordinates[CELL_T]}
            branch = (cast(tuple[float, float], program.current_branch_A)
                      if mode == "power" else (-limit, limit))
            cached_current = isolate_control_current(spec.series_resistance, inputs,
                _fields(spec, state, 0.)["voltage_V"], target, branch, mode, limit)
            cached_key = key
        return cached_current

    def guess(state: CoupledStateView) -> dict[str, float]:
        requested = requested_current(state)
        current = 0. if requested is None else requested
        return {CURRENT: current, VOLTAGE: _fields(spec, state, current)["voltage_V"]}

    def residual(state: CoupledStateView) -> list[float]:
        current, voltage = (state.algebraic_values[key] for key in (CURRENT, VOLTAGE))
        actual = _fields(spec, state, current)
        target = program.value_at(state.time_s)
        requested = requested_current(state)
        control = ((voltage-target)/voltage_scale if requested is None
                   else (current-requested)/limit)
        return [control, voltage-actual["voltage_V"]]

    algebraic = CoupledAlgebraic(
        # Resolve the exact active CC/CV branch with the common Newton solver.
        # Differencing a clipped projection across a source-switching corner
        # gives a spurious I derivative and can stall consistency. Branches
        # agree at their shared physical root; no current/state is clipped.
        (ResidualVariable(CURRENT, "A", limit, "algebraic"),
         ResidualVariable(VOLTAGE, "V", voltage_scale, "algebraic")),
        (ResidualEquation("thevenin:control", "1"),
         ResidualEquation("thevenin:terminal", "V", voltage_scale)), guess, residual)

    def evaluate(state: CoupledStateView) -> ProcessContribution:
        current, voltage = (state.algebraic_values[key] for key in (CURRENT, VOLTAGE))
        if mode in DYNAMIC_MODES:
            # The controller already declares I=q or I=max(q,Icc). Substitute
            # that exact algebraic relation into its differential equations.
            # Otherwise a stale consistency guess can evaluate V/I near zero
            # even though the actual controller state lies safely in its domain.
            # The shared terminal/control residual still enforces identical I/V.
            current = cast(float, requested_current(state))
            voltage = _fields(spec, state, current)["voltage_V"]
        f = _fields(spec, state, current)
        rates = {
            SOC: -current/p["capacity_C"], CHARGE: current, WORK: current*voltage,
            THROUGHPUT_Q: abs(current), THROUGHPUT_E: abs(current*voltage),
            CELL_T: (f["empirical_heat_W"]-f["cell_to_jig_W"])/p["cell_heat_capacity_J_K"],
            JIG_T: (f["cell_to_jig_W"]-f["ambient_loss_W"])/p["jig_heat_capacity_J_K"],
            MODEL_HEAT: f["empirical_heat_W"], AMBIENT_LOSS: f["ambient_loss_W"],
        }
        rates.update({rc_key(i): current/c-state.coordinates[rc_key(i)]/(r*c)
                      for i, (r, c) in enumerate(zip(f["resistance_ohm"], f["capacitance_F"],
                                                   strict=True))})
        if spec.diffusion is not None:
            rates.update(spec.diffusion.deviation_rates(
                state.coordinates, current, p["capacity_C"]))
        if spec.control is not None:
            rates[CONTROLLER_CURRENT] = spec.control.rate(
                mode, state.coordinates[CONTROLLER_CURRENT], current, voltage,
                program.value_at(state.time_s))
        return ProcessContribution(coordinate_rates=rates)

    def validate(state: CoupledStateView) -> None:
        if (state.temperature_K != p["reference_frame_temperature_K"]
                or state.volume_L != p["housing_volume_L"]):
            raise ValueError("Thevenin formal reference frame/housing is not a thermal cell state")
        if spec.diffusion is not None:
            spec.diffusion.validate_profile(state.coordinates[SOC], state.coordinates,
                                            p["minimum_soc"], p["maximum_soc"])
        if mode in DYNAMIC_MODES:
            guard = 32*np.spacing(max(1., limit))
            if abs(cast(float, requested_current(state))) > limit+guard:
                raise ValueError("dynamic controller actual current exceeds the source domain")
        if mode == "differential_resistance":
            low, high = cast(tuple[float, float], program.current_branch_A)
            guard = 32*np.spacing(max(1., abs(low), abs(high)))
            if not low-guard <= state.coordinates[CONTROLLER_CURRENT] <= high+guard:
                raise ValueError(
                    "differential-resistance state is outside its declared current branch")

    events = [CoupledEvent("thevenin:minimum-soc", lambda s: s.coordinates[SOC]-p["minimum_soc"]),
              CoupledEvent("thevenin:maximum-soc", lambda s: p["maximum_soc"]-s.coordinates[SOC]),
              CoupledEvent("thevenin:minimum-voltage",
                           lambda s: s.algebraic_values[VOLTAGE]-p["minimum_voltage_V"]),
              CoupledEvent("thevenin:maximum-voltage",
                           lambda s: p["maximum_voltage_V"]-s.algebraic_values[VOLTAGE])]
    if mode in DYNAMIC_MODES:
        events.extend((
            CoupledEvent("thevenin:minimum-source-current",
                         lambda s: s.algebraic_values[CURRENT]+limit),
            CoupledEvent("thevenin:maximum-source-current",
                         lambda s: limit-s.algebraic_values[CURRENT]),
        ))
    if mode == "differential_resistance":
        low, high = cast(tuple[float, float], program.current_branch_A)
        events.extend((
            CoupledEvent("thevenin:minimum-control-branch",
                         lambda s: s.coordinates[CONTROLLER_CURRENT]-low),
            CoupledEvent("thevenin:maximum-control-branch",
                         lambda s: high-s.coordinates[CONTROLLER_CURRENT]),
        ))
    if spec.diffusion is not None:
        diffusion = spec.diffusion
        events.extend((
            CoupledEvent("thevenin:minimum-surface-soc", lambda s:
                diffusion.surface_soc(s.coordinates[SOC], s.coordinates)-p["minimum_soc"]),
            CoupledEvent("thevenin:maximum-surface-soc", lambda s:
                p["maximum_soc"]-diffusion.surface_soc(s.coordinates[SOC], s.coordinates)),
            CoupledEvent("thevenin:minimum-internal-soc", lambda s:
                min(diffusion.profile(s.coordinates[SOC], s.coordinates))-p["minimum_soc"]),
            CoupledEvent("thevenin:maximum-internal-soc", lambda s:
                p["maximum_soc"]-max(diffusion.profile(s.coordinates[SOC], s.coordinates))),
        ))
    for key in (CELL_T, JIG_T):
        def minimum_temperature(state: CoupledStateView, key: str = key) -> float:
            return state.coordinates[key]-p["minimum_temperature_K"]

        def maximum_temperature(state: CoupledStateView, key: str = key) -> float:
            return p["maximum_temperature_K"]-state.coordinates[key]

        events.extend((
            CoupledEvent(f"{key}:minimum", minimum_temperature),
            CoupledEvent(f"{key}:maximum", maximum_temperature),
        ))
    return CoupledSystem(
        KineticOnlyThermodynamics(),
        (CoupledProcess("thevenin:device", evaluate, validate_state=validate),),
        isothermal=True, numerical_form="residual", algebraic=algebraic,
        coordinates=spec.coordinate_declarations, events=(*events, *control_events),
        conservation_laws={
            "circuit_charge_C": {SOC: p["capacity_C"], CHARGE: 1},
            "empirical_node_heat_J": {CELL_T: p["cell_heat_capacity_J_K"],
                                      JIG_T: p["jig_heat_capacity_J_K"],
                                      MODEL_HEAT: -1, AMBIENT_LOSS: 1},
        })


def thevenin_fields(
    spec: TheveninSpec, state: CoupledState,
    program: TheveninControlProgram | ScalarControlProgram | Mapping[str, Any],
) -> TheveninFields:
    """Read the same consistent terminal, without persisting an I/V truth copy."""
    result = thevenin_system(spec, program).advance(state, 0.)
    algebraic = result.solver_diagnostic["algebraic_values"]
    view = CoupledStateView(state.amounts_mol, state.temperature_K, state.volume_L, state.time_s,
                            coordinates=state.coordinates, algebraic_values=algebraic)
    values = _fields(spec, view, algebraic[CURRENT])
    values["voltage_V"] = algebraic[VOLTAGE]
    if spec.control is not None:
        values["controller_current_A"] = state.coordinates[CONTROLLER_CURRENT]
    return values
