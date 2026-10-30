"""Two finite electrode lithium stores with independent OCVs and series R.

The isothermal SplitOCVR equations follow PyBaMM 25.4.2, without particle,
electrolyte, kinetic or thermal states. Explicit finite-source CV extends the
source's prescribed-current model. Every trajectory uses the common coupled
residual; a closed algebraic current expression supplies its consistent guess.
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from math import isclose, isfinite
from numbers import Real
from typing import Any, Literal, Self, TypeAlias, TypedDict, cast

import numpy as np
from numpy.typing import NDArray
from scipy import sparse

from chemworld.data.scientific_inputs import InputSource, _text
from chemworld.physchem.battery_materials import ElectrodeOCV
from chemworld.physchem.battery_models import BATTERY_ALGEBRAIC, BATTERY_COORDINATES, F
from chemworld.physchem.coupled_system import (
    CoupledAdvanceResult,
    CoupledAlgebraic,
    CoupledCoordinate,
    CoupledEvent,
    CoupledProcess,
    CoupledState,
    CoupledStateView,
    CoupledSystem,
    ProcessContribution,
)
from chemworld.physchem.data_conflicts import DataSourceProvenance
from chemworld.physchem.reactor_shared import ScalarControlProgram
from chemworld.physchem.residual_system import ResidualEquation, ResidualVariable

SPLIT_OCVR_FAMILY = "split-ocvr-isothermal-v1"
SPLIT_OCVR_KEYS = ("negative:Li", "positive:Li")
_ELECTRODES = ("negative", "positive")
ControlMode: TypeAlias = Literal[
    "constant_current", "constant_voltage", "positive_current_limit", "negative_current_limit",
]


class SplitOCVRFields(TypedDict):
    negative_average_fraction: float
    positive_average_fraction: float
    negative_ocv_V: float
    positive_ocv_V: float
    open_circuit_voltage_V: float
    current_A: float
    voltage_V: float
    ohmic_drop_V: float
    control_mode: ControlMode | Literal["specified_current"]
    discharged_charge_C: float
    delivered_work_J: float


def _default_ocv(electrode: str) -> ElectrodeOCV:
    source = InputSource(
        DataSourceProvenance(
            "chemworld:synthetic-split-ocvr",
            0,
            "Authored linear electrode OCV; not a measured material fit",
            "1",
            "MIT (ChemWorld-authored expression)",
        ),
        "synthetic",
    )
    return ElectrodeOCV(
        "synthetic-split-" + electrode,
        source,
        (0.2, -0.1) if electrode == "negative" else (4.2, -0.8),
        stoichiometry_domain=(0.0, 1.0),
        supported_models=("SplitOCVR",),
    )


def _finite(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value):
        raise ValueError(f"SplitOCVR {name} must be finite")


@dataclass(frozen=True)
class BatterySplitOCVRSpec:
    negative_capacity_C: float = 100.0
    positive_capacity_C: float = 120.0
    negative_ocv: ElectrodeOCV = field(default_factory=lambda: _default_ocv("negative"))
    positive_ocv: ElectrodeOCV = field(default_factory=lambda: _default_ocv("positive"))
    resistance_ohm: float = 0.1
    initial_negative_fraction: float = 0.7
    initial_positive_fraction: float = 0.3
    temperature_K: float = 298.15
    housing_volume_L: float = 0.01
    negative_minimum_fraction: float = 0.0001
    negative_maximum_fraction: float = 0.999
    positive_minimum_fraction: float = 0.0001
    positive_maximum_fraction: float = 0.999
    minimum_voltage_V: float = 2.5
    maximum_voltage_V: float = 4.5
    maximum_current_A: float = 2.0
    numerical_form: str = "residual"
    provenance: str = "synthetic SplitOCVR; independent capacities and linear OCVs"

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            if name not in {"negative_ocv", "positive_ocv", "numerical_form", "provenance"}:
                _finite(getattr(self, name), name)
        if (
            any(
                getattr(self, k) <= 0
                for k in (
                    "negative_capacity_C",
                    "positive_capacity_C",
                    "temperature_K",
                    "housing_volume_L",
                    "maximum_current_A",
                )
            )
            or self.resistance_ohm < 0
            or self.minimum_voltage_V >= self.maximum_voltage_V
            or self.numerical_form != "residual"
        ):
            raise ValueError("invalid SplitOCVR operating domain")
        for electrode in _ELECTRODES:
            low, high = self.fraction_bounds(electrode)
            theta = getattr(self, "initial_" + electrode + "_fraction")
            if not 0 <= low < high <= 1 or not low <= theta <= high:
                raise ValueError("SplitOCVR fractions must lie within their independent bounds")
            curve = getattr(self, electrode + "_ocv")
            if isinstance(curve, dict):
                curve = ElectrodeOCV.from_dict(curve)
                object.__setattr__(self, electrode + "_ocv", curve)
            if not isinstance(curve, ElectrodeOCV) or "SplitOCVR" not in curve.supported_models:
                raise ValueError("OCV must explicitly support SplitOCVR")
            if not curve.stoichiometry_domain[0] <= low < high <= curve.stoichiometry_domain[1]:
                raise ValueError("OCV domain must cover the SplitOCVR operating interval")
            # Derived inventories and scaling must remain representable.
            capacity = getattr(self, electrode + "_capacity_C") / F
            if not isfinite(capacity) or capacity <= 0:
                raise ValueError("SplitOCVR capacity is not representable in mol")
        if not isfinite(self.usable_charge_c) or self.usable_charge_c <= 0:
            raise ValueError("SplitOCVR working-window capacity must be finite and positive")
        if not isfinite(self.maximum_current_A * self.resistance_ohm):
            raise ValueError("SplitOCVR source voltage drop is not representable")
        _text(self.provenance, "SplitOCVR provenance")

    @property
    def model_family(self) -> str:
        return SPLIT_OCVR_FAMILY

    @property
    def model_id(self) -> str:
        return SPLIT_OCVR_FAMILY

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
        return SPLIT_OCVR_KEYS

    @property
    def inventory_species(self) -> dict[str, str]:
        return dict.fromkeys(self.inventory_keys, "Li")

    @property
    def volume_l(self) -> float:
        return self.housing_volume_L

    @property
    def series_resistance_ohm(self) -> float:
        return self.resistance_ohm

    @property
    def supported_control_units(self) -> tuple[str, ...]:
        return ("A",) if self.resistance_ohm == 0 else ("A", "V")

    @property
    def usable_charge_c(self) -> float:
        """Nominal common working-window capacity, independent of current SOC."""
        return min(
            getattr(self, e + "_capacity_C")
            * (getattr(self, e + "_maximum_fraction") - getattr(self, e + "_minimum_fraction"))
            for e in _ELECTRODES
        )

    def fraction_bounds(self, electrode: str) -> tuple[float, float]:
        if electrode not in _ELECTRODES:
            raise ValueError("unknown SplitOCVR electrode")
        return (
            getattr(self, electrode + "_minimum_fraction"),
            getattr(self, electrode + "_maximum_fraction"),
        )

    def current_at_c_rate(self, rate: float) -> float:
        _finite(rate, "C rate")
        current = rate * self.usable_charge_c / 3600
        _finite(current, "C-rate current")
        return current

    def initial_coordinates(self) -> dict[str, float]:
        return dict.fromkeys(self.coordinate_keys, 0.0)

    def initial_state(self) -> CoupledState:
        return CoupledState(
            {
                e + ":Li": getattr(self, e + "_capacity_C")
                / F
                * getattr(self, "initial_" + e + "_fraction")
                for e in _ELECTRODES
            },
            self.temperature_K,
            self.volume_l,
            coordinates=self.initial_coordinates(),
        )

    def ocv(self, electrode: str, theta: float, *, derivative: bool = False) -> float:
        if electrode not in _ELECTRODES:
            raise ValueError("unknown SplitOCVR electrode")
        curve = getattr(self, electrode + "_ocv")
        # Trial extrapolation is explicit: accepted inventories are never clipped.
        return (
            curve.derivatives(theta, trial=True)[1]
            if derivative
            else curve.value(theta, trial=True)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_family": self.model_family,
            **{
                k: v.to_dict() if isinstance(v := getattr(self, k), ElectrodeOCV) else v
                for k in self.__dataclass_fields__
            },
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Self:
        value = dict(value)
        if value.pop("model_family", None) != SPLIT_OCVR_FAMILY:
            raise ValueError("unsupported SplitOCVR family")
        return cls(**value)


def _material_fields(spec: BatterySplitOCVRSpec, state: CoupledStateView) -> dict[str, float]:
    result = {}
    for e in _ELECTRODES:
        theta = state.amounts_mol[e + ":Li"] * F / getattr(spec, e + "_capacity_C")
        result[e + "_average_fraction"] = theta
        result[e + "_ocv_V"] = spec.ocv(e, theta)
    result["open_circuit_voltage_V"] = result["positive_ocv_V"] - result["negative_ocv_V"]
    return result


def validate_split_ocvr_state(spec: BatterySplitOCVRSpec, state: CoupledStateView) -> None:
    if set(state.amounts_mol) != set(spec.inventory_keys) or set(state.coordinates) != set(
        spec.coordinate_keys
    ):
        raise ValueError("SplitOCVR requires exactly two Li inventories and Q/W coordinates")
    if not isclose(state.temperature_K, spec.temperature_K, rel_tol=0, abs_tol=1e-8) or not isclose(
        state.volume_L, spec.volume_l, rel_tol=1e-12, abs_tol=1e-14
    ):
        raise ValueError("SplitOCVR requires its fixed temperature and housing volume")
    for value in (*state.amounts_mol.values(), *state.coordinates.values()):
        _finite(value, "accepted state")
    q = state.coordinates[BATTERY_COORDINATES[0]]
    for e, sign in zip(_ELECTRODES, (-1, 1), strict=True):
        capacity = getattr(spec, e + "_capacity_C")
        n = state.amounts_mol[e + ":Li"]
        theta = n * F / capacity
        low, high = spec.fraction_bounds(e)
        if n < 0 or not low - 1e-10 <= theta <= high + 1e-10:
            raise ValueError("SplitOCVR accepted fraction lies outside its declared boundary")
        if not isclose(
            capacity * (theta - getattr(spec, "initial_" + e + "_fraction")),
            sign * q,
            rel_tol=1e-8,
            abs_tol=max(capacity, 1) * 1e-10,
        ):
            raise ValueError("SplitOCVR true Li and Q violate Faraday balance")


def _check_program(spec: BatterySplitOCVRSpec, program: ScalarControlProgram) -> None:
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
        raise ValueError("SplitOCVR requires an in-domain CC or finite-source CV program")
    if program.unit == "V" and spec.resistance_ohm == 0:
        raise ValueError("SplitOCVR R=0 CV is unreachable or has nonunique current")


def split_ocvr_control_mode(
    spec: BatterySplitOCVRSpec, state: CoupledStateView, program: ScalarControlProgram,
) -> tuple[ControlMode, float]:
    _check_program(spec, program)
    target = program.value_at(state.time_s)
    if program.unit == "A":
        return "constant_current", target
    delta = _material_fields(spec, state)["open_circuit_voltage_V"] - target
    bound = spec.resistance_ohm * spec.maximum_current_A
    if delta >= bound:
        return "positive_current_limit", spec.maximum_current_A
    if delta <= -bound:
        return "negative_current_limit", -spec.maximum_current_A
    return "constant_voltage", target


def _controlled_current(
    spec: BatterySplitOCVRSpec, state: CoupledStateView, program: ScalarControlProgram,
) -> float:
    mode, target = split_ocvr_control_mode(spec, state, program)
    return (
        (_material_fields(spec, state)["open_circuit_voltage_V"] - target) / spec.resistance_ohm
        if mode == "constant_voltage"
        else target
    )


def split_ocvr_algebraic(
    spec: BatterySplitOCVRSpec, program: ScalarControlProgram,
) -> CoupledAlgebraic:
    _check_program(spec, program)
    ik, vk = BATTERY_ALGEBRAIC

    def initial(state: CoupledStateView) -> dict[str, float]:
        current = _controlled_current(spec, state, program)
        u = _material_fields(spec, state)["open_circuit_voltage_V"]
        return {ik: current, vk: u - spec.resistance_ohm * current}

    def residual(state: CoupledStateView) -> NDArray[np.float64]:
        current, voltage = (state.algebraic_values[k] for k in BATTERY_ALGEBRAIC)
        u = _material_fields(spec, state)["open_circuit_voltage_V"]
        mode, target = split_ocvr_control_mode(spec, state, program)
        return np.array(
            [
                voltage - u + spec.resistance_ohm * current,
                voltage - target
                if mode == "constant_voltage"
                else (current - target) / spec.maximum_current_A,
            ]
        )

    return CoupledAlgebraic(
        (
            ResidualVariable(ik, "A", spec.maximum_current_A, "algebraic"),
            ResidualVariable(vk, "V", 1.0, "algebraic"),
        ),
        (
            ResidualEquation("split-ocvr:terminal-voltage", "V"),
            ResidualEquation("battery:control", "dimensionless"),
        ),
        initial,
        residual,
    )


def _closure_jacobian(
    spec: BatterySplitOCVRSpec, state: CoupledStateView, program: ScalarControlProgram,
    identities: Sequence[str],
) -> sparse.csc_matrix:
    index = {k: i for i, k in enumerate(identities)}
    ik, vk = BATTERY_ALGEBRAIC
    jac = np.zeros((2, len(identities)))
    f = _material_fields(spec, state)
    for e, sign in zip(_ELECTRODES, (1, -1), strict=True):
        key = e + ":Li"
        if key in index:
            jac[0, index[key]] = (
                sign
                * spec.ocv(e, f[e + "_average_fraction"], derivative=True)
                * F
                / getattr(spec, e + "_capacity_C")
            )
    jac[0, index[ik]], jac[0, index[vk]] = spec.resistance_ohm, 1.0
    mode, _ = split_ocvr_control_mode(spec, state, program)
    jac[1, index[vk] if mode == "constant_voltage" else index[ik]] = (
        1.0 if mode == "constant_voltage" else 1 / spec.maximum_current_A
    )
    return sparse.csc_matrix(jac)


def split_ocvr_fields(
    spec: BatterySplitOCVRSpec, state: CoupledStateView, *, current_A: float | None = None,
    program: ScalarControlProgram | None = None,
) -> SplitOCVRFields:
    if (current_A is None) == (program is None):
        raise ValueError("supply exactly one SplitOCVR current or control program")
    if isinstance(state, CoupledState):
        validate_split_ocvr_state(spec, state)
    f = _material_fields(spec, state)
    mode: ControlMode | Literal["specified_current"]
    if current_A is not None:
        _finite(current_A, "current")
        if abs(current_A) > spec.maximum_current_A:
            raise ValueError("SplitOCVR current exceeds the finite source")
        current, mode = current_A, "specified_current"
        voltage = f["open_circuit_voltage_V"] - spec.resistance_ohm * current
    else:
        program = cast(ScalarControlProgram, program)
        mode, _ = split_ocvr_control_mode(spec, state, program)
        # A saved CoupledState may carry a prior stage's algebraic solution.
        # Only live residual views consume their trial algebraic values.
        if not isinstance(state, CoupledState) and all(
            k in state.algebraic_values for k in BATTERY_ALGEBRAIC
        ):
            current, voltage = (state.algebraic_values[k] for k in BATTERY_ALGEBRAIC)
        else:
            current = _controlled_current(spec, state, program)
            voltage = f["open_circuit_voltage_V"] - spec.resistance_ohm * current
    return cast(SplitOCVRFields, {
        **f,
        "current_A": current,
        "voltage_V": voltage,
        "ohmic_drop_V": spec.resistance_ohm * current,
        "control_mode": mode,
        **{k: state.coordinates[k] for k in BATTERY_COORDINATES},
    })


@dataclass(frozen=True)
class SplitOCVRReadout:
    voltage_V: float
    current_A: float
    negative_average_fraction: float
    positive_average_fraction: float
    negative_ocv_V: float
    positive_ocv_V: float
    ohmic_drop_V: float
    discharged_charge_C: float
    delivered_work_J: float


def split_ocvr_readout(
    spec: BatterySplitOCVRSpec, state: CoupledStateView, current_A: float,
) -> SplitOCVRReadout:
    f = split_ocvr_fields(spec, state, current_A=current_A)
    return SplitOCVRReadout(
        voltage_V=f["voltage_V"], current_A=f["current_A"],
        negative_average_fraction=f["negative_average_fraction"],
        positive_average_fraction=f["positive_average_fraction"],
        negative_ocv_V=f["negative_ocv_V"], positive_ocv_V=f["positive_ocv_V"],
        ohmic_drop_V=f["ohmic_drop_V"], discharged_charge_C=f["discharged_charge_C"],
        delivered_work_J=f["delivered_work_J"],
    )


def split_ocvr_control_current(
    spec: BatterySplitOCVRSpec, state: CoupledStateView, program: ScalarControlProgram,
) -> float:
    return split_ocvr_fields(spec, state, program=program)["current_A"]


class _SplitOCVRThermodynamics:
    inventory_keys = SPLIT_OCVR_KEYS

    def heat_capacity_j_k(self, state: CoupledStateView) -> float:
        return 1.0

    def internal_energy_j(self, state: CoupledStateView) -> None:
        return None

    def chemical_energy_rate_w(
        self, state: CoupledStateView, rates: Mapping[str, float],
        contributions: Sequence[ProcessContribution],
    ) -> float:
        return 0.0


def _program_slope(program: ScalarControlProgram, time: float) -> float:
    if program.mode == "linear":
        for (t0, v0), (t1, v1) in zip(program.setpoints[:-1], program.setpoints[1:], strict=True):
            if t0 <= time < t1:
                return (v1 - v0) / (t1 - t0)
    return 0.0


def split_ocvr_system(
    spec: BatterySplitOCVRSpec, program: ScalarControlProgram, *,
    thermal_device: object | None = None, control_events: Sequence[CoupledEvent] = (),
) -> CoupledSystem:
    if thermal_device is not None:
        raise ValueError("SplitOCVR is isothermal; external thermal coupling is unsupported")
    closure = split_ocvr_algebraic(spec, program)

    def fields(state: CoupledStateView) -> SplitOCVRFields:
        return split_ocvr_fields(spec, state, program=program)

    def evaluate(state: CoupledStateView) -> ProcessContribution:
        f = fields(state)
        current, voltage = f["current_A"], f["voltage_V"]
        return ProcessContribution(
            amount_rates_mol_s=dict(zip(SPLIT_OCVR_KEYS, (-current / F, current / F), strict=True)),
            coordinate_rates=dict(
                zip(BATTERY_COORDINATES, (current, current * voltage), strict=True)
            ),
        )

    def fraction_event(
        electrode: str, sign: int, bound: float, direction: int,
    ) -> Callable[[CoupledStateView], float]:
        def event(state: CoupledStateView) -> float:
            return (
                direction * (
                    cast(float, cast(Mapping[str, object], fields(state))[
                        electrode + "_average_fraction"]
                    ) - bound
                )
                if direction * sign * fields(state)["current_A"] < 0
                else 1.0
            )

        return event

    events = []
    for e, sign in zip(_ELECTRODES, (-1, 1), strict=True):
        for name, bound, direction in (
            ("minimum", spec.fraction_bounds(e)[0], 1),
            ("maximum", spec.fraction_bounds(e)[1], -1),
        ):
            events.append(
                CoupledEvent(
                    f"split-ocvr:{e}:{name}-fraction",
                    fraction_event(e, sign, bound, direction),
                )
            )

    def voltage_event(state: CoupledStateView, bound: float, direction: int) -> float:
        f = fields(state)
        numeric_fields: Mapping[str, object] = f
        mode, _ = split_ocvr_control_mode(spec, state, program)
        derivative = (
            _program_slope(program, state.time_s)
            if mode == "constant_voltage"
            else f["current_A"]
            * sum(
                spec.ocv(e, cast(float, numeric_fields[e + "_average_fraction"]), derivative=True)
                / getattr(spec, e + "_capacity_C")
                for e in _ELECTRODES
            )
            - (
                spec.resistance_ohm * _program_slope(program, state.time_s)
                if mode == "constant_current"
                else 0.0
            )
        )
        distance = direction * (f["voltage_V"] - bound)
        # Permit inward continuation of a located boundary, not recovery from
        # an already inadmissible initial voltage. This tolerance only admits
        # event-root roundoff; the actual voltage/inventory is never changed.
        return 1.0 if distance >= -1e-10 and direction * derivative > 0 else distance

    events.extend(
        (
            CoupledEvent(
                "battery:minimum-voltage", lambda s: voltage_event(s, spec.minimum_voltage_V, 1)
            ),
            CoupledEvent(
                "battery:maximum-voltage", lambda s: voltage_event(s, spec.maximum_voltage_V, -1)
            ),
        )
    )

    def jacobian(state: CoupledStateView, identities: Sequence[str]) -> sparse.csc_matrix:
        index = {k: i for i, k in enumerate(identities)}
        ik, vk = BATTERY_ALGEBRAIC
        jac = np.zeros((len(identities), len(identities)))
        jac[index[SPLIT_OCVR_KEYS[0]], index[ik]] = -1 / F
        jac[index[SPLIT_OCVR_KEYS[1]], index[ik]] = 1 / F
        jac[index[BATTERY_COORDINATES[0]], index[ik]] = 1.0
        jac[index[BATTERY_COORDINATES[1]], index[ik]] = state.algebraic_values[vk]
        jac[index[BATTERY_COORDINATES[1]], index[vk]] = state.algebraic_values[ik]
        jac[[index[k] for k in BATTERY_ALGEBRAIC], :] = -_closure_jacobian(
            spec, state, program, identities
        ).toarray()
        return sparse.csc_matrix(jac)

    return CoupledSystem(
        _SplitOCVRThermodynamics(),
        (
            CoupledProcess(
                "split-ocvr:finite-electrodes",
                evaluate,
                validate_state=lambda s: validate_split_ocvr_state(spec, s),
            ),
        ),
        isothermal=True,
        numerical_form="residual",
        algebraic=closure,
        residual_rhs_jacobian=jacobian,
        events=(*events, *control_events),
        inventory_scales_mol={e + ":Li": getattr(spec, e + "_capacity_C") / F for e in _ELECTRODES},
        coordinates=(
            CoupledCoordinate(BATTERY_COORDINATES[0], "C", spec.usable_charge_c),
            CoupledCoordinate(BATTERY_COORDINATES[1], "J", spec.usable_charge_c),
        ),
        conservation_laws={
            "lithium": dict.fromkeys(SPLIT_OCVR_KEYS, 1.0),
            "faradaic:negative": {SPLIT_OCVR_KEYS[0]: F, BATTERY_COORDINATES[0]: 1.0},
            "faradaic:positive": {SPLIT_OCVR_KEYS[1]: F, BATTERY_COORDINATES[0]: -1.0},
        },
    )


def advance_split_ocvr(
    spec: BatterySplitOCVRSpec,
    state: CoupledState,
    program: ScalarControlProgram,
    duration_s: float,
    *,
    evaluation_times_s: Sequence[float] | None = None,
    thermal_device: object | None = None,
    control_events: Sequence[CoupledEvent] = (),
) -> CoupledAdvanceResult:
    result = split_ocvr_system(
        spec, program, thermal_device=thermal_device, control_events=control_events
    ).advance(state, duration_s, evaluation_times_s=evaluation_times_s)
    return replace(
        result,
        solver_diagnostic={
            **result.solver_diagnostic,
            "battery_control_mode": split_ocvr_control_mode(spec, result.final_state, program)[0],
            "energy_scope": "isothermal SplitOCVR; terminal electrical work only; no heat closure",
        },
    )
