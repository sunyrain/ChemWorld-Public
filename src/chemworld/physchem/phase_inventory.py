"""Common finite-compartment n/H advance with algebraic phase closure.

Columns and concentrators supply their own equipment ports and stop laws.
This numerical strategy owns no session or persisted inventory. Outgoing
boundary integrals are measurements, separate from retained compartment n/H.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import Field, dataclass, field, replace
from math import isfinite
from typing import Any, ClassVar, Generic, Literal, Protocol, TypeAlias, TypeVar, cast, overload

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import spmatrix

from chemworld.physchem.solver_backend import ODESolverPolicy, solve_ode


class PhaseInventoryState(Protocol):
    """Caller-owned dataclass inventory; optional work/outlet ports are separate."""

    __dataclass_fields__: ClassVar[dict[str, Field[Any]]]

    @property
    def amounts_mol(self) -> tuple[dict[str, float], ...]: ...

    @property
    def enthalpies_J(self) -> tuple[float, ...]: ...  # noqa: N802 - existing SI field

    @property
    def time_s(self) -> float: ...

    @property
    def heater_energy_J(self) -> float: ...  # noqa: N802 - existing SI field

    @property
    def condenser_energy_J(self) -> float: ...  # noqa: N802 - existing SI field


class _OutletInventory(Protocol):
    @property
    def outlet_amounts_mol(self) -> Mapping[str, float]: ...

    @property
    def outlet_enthalpy_J(self) -> float: ...  # noqa: N802 - existing SI field


class _WorkInventory(Protocol):
    @property
    def shaft_import_energy_J(self) -> float: ...  # noqa: N802 - existing SI field

    @property
    def shaft_export_energy_J(self) -> float: ...  # noqa: N802 - existing SI field


class _InventoryComponents(Protocol):
    @property
    def component_ids(self) -> Sequence[str]: ...


class PhaseInventorySpec(Protocol):
    @property
    def properties(self) -> _InventoryComponents: ...


_State = TypeVar("_State", bound=PhaseInventoryState)
_Phase = TypeVar("_Phase")
_Phase_co = TypeVar("_Phase_co", covariant=True)
_Vector: TypeAlias = NDArray[np.float64]


class _InventoryClosure(Protocol[_Phase_co]):
    def at_ph(self, index: int, amounts: Mapping[str, float], enthalpy: float) -> _Phase_co: ...


@dataclass(frozen=True)
class PhaseInventoryRates:
    amounts_mol_s: _Vector
    enthalpy_W: _Vector
    heater_W: float
    condenser_W: float
    streams: tuple[dict[str, Any], ...]
    outlet_amounts_mol_s: dict[str, float] = field(default_factory=dict, kw_only=True)
    outlet_enthalpy_W: float = field(default=0.0, kw_only=True)
    shaft_import_W: float = field(default=0.0, kw_only=True)
    shaft_export_W: float = field(default=0.0, kw_only=True)


@dataclass(frozen=True)
class PhaseInventoryAdvance(Generic[_State]):
    initial_state: _State
    final_state: _State
    trajectory: tuple[_State, ...]
    stopping_event: str | None
    solver_diagnostic: dict[str, Any]
    component_residuals_mol: dict[str, float]
    enthalpy_residual_J: float

    @property
    def actual_duration_s(self) -> float:
        return self.final_state.time_s - self.initial_state.time_s


@dataclass(frozen=True)
class _PhaseInventoryView:
    amounts_mol: tuple[dict[str, float], ...]
    enthalpies_J: tuple[float, ...]
    time_s: float


PhaseInventoryStop: TypeAlias = Callable[[_PhaseInventoryView, tuple[_Phase | None, ...]], float]
PhaseInventoryRate: TypeAlias = Callable[
    [_PhaseInventoryView, tuple[_Phase | None, ...]], PhaseInventoryRates,
]
_ResultFactory: TypeAlias = Callable[
    [_State, _State, tuple[_State, ...], str | None, dict[str, Any], dict[str, float], float],
    PhaseInventoryAdvance[_State],
]


class _EventCallback(Protocol):
    terminal: bool
    direction: int

    def __call__(self, time: float, scaled: _Vector) -> float: ...


def integrate_phase_inventory(
    spec: PhaseInventorySpec,
    state: _State,
    *,
    duration_s: float,
    rates: PhaseInventoryRate[_Phase],
    closure_indices: Iterable[int],
    stop_functions: Sequence[tuple[str, PhaseInventoryStop[_Phase]]],
    enthalpy_bounds: Callable[[int, Mapping[str, float]], tuple[float, float]] | None,
    result_type: _ResultFactory[_State] = PhaseInventoryAdvance,
    evaluation_times_s: Sequence[float] | NDArray[np.floating[Any]] | None = None,
    max_step_s: float | None = None,
    rtol: float = 1e-8,
    progress: Callable[[float, int], None] | None = None,
    jac_sparsity: NDArray[Any] | spmatrix | None = None,
    solver_id: str = "finite_phase_inventory_bdf_v1",
    open_outlet: bool = False,
    balance_scale_mol: float | None = None,
    diagnostic_counters: dict[str, int] | None = None,
    operating_limit_error: type[Exception] = ValueError,
    initial_boundary: Literal["reject", "stop"] = "reject",
    constant_amounts: bool = False,
    include_shaft_work: bool = False,
    reuse_cell_closures: bool = False,
) -> PhaseInventoryAdvance[_State]:
    """One solve for retained n/H, utility heat and explicit outlet integrals.

    ``rates`` contributes a single candidate derivative. PH is algebraic;
    utility and outlet counters cannot alter the constitutive candidate state.
    ``reuse_cell_closures`` requires a deterministic, side-effect-free n/H
    closure and bounds provider for the duration of this solve. Stateful search
    providers retain the default complete evaluation sequence.
    Explicit shaft accounting adds separate nonnegative import/export integrals
    to the same solve. Its state must expose shaft_import_energy_J and
    shaft_export_energy_J; the default numerical coordinates remain unchanged.
    A model may select a genuine zero-duration initial stop or reject a request
    that starts at an already reached operating limit.
    """
    ids, count = spec.properties.component_ids, len(state.amounts_mol)
    width = len(ids)
    if (
        not isfinite(duration_s)
        or duration_s < 0
        or initial_boundary not in {"reject", "stop"}
        or any(set(n) != set(ids) for n in state.amounts_mol)
        or len(state.enthalpies_J) != count
        or type(constant_amounts) is not bool
        or type(include_shaft_work) is not bool
        or type(reuse_cell_closures) is not bool
        or (constant_amounts and open_outlet)
    ):
        raise ValueError("phase integration needs matching finite inventories, H and duration")
    if include_shaft_work and any(
        not hasattr(state, name) or isinstance(getattr(state, name), bool)
        or not isfinite(getattr(state, name)) or getattr(state, name) < 0
        for name in ("shaft_import_energy_J", "shaft_export_energy_J")
    ):
        raise ValueError("shaft accounting requires owned nonnegative import/export integrals")
    initial_n = np.array([[n[k] for k in ids] for n in state.amounts_mol])
    # These strategies have internal ports and optional outlets, with no
    # material inlet. A species absent everywhere is an invariant zero, not
    # a physical direction for finite-difference Jacobian perturbations.
    absent = np.sum(initial_n, axis=0) == 0
    # In a closed heat-only slice n_dot is identically zero. Eliminate these
    # coordinates from the numerical solve: even a Jacobian trial must not
    # move a line compound off its fixed stoichiometry. The accepted inventory
    # remains the original physical stock, never a projected/repaired trial.
    n_end = 0 if constant_amounts else count*width
    h_end = n_end+count
    utility_count = 4 if include_shaft_work else 2
    q_end = h_end + utility_count
    scales = np.r_[
        np.empty(0) if constant_amounts else np.maximum(initial_n, 1e-3).ravel(),
        np.maximum(np.abs(state.enthalpies_J), 10),
        np.full(utility_count, 100.0),
    ]
    initial = np.r_[np.empty(0) if constant_amounts else initial_n.ravel(),
                    state.enthalpies_J, np.zeros(utility_count)]
    if open_outlet:
        scales = np.r_[scales, np.maximum(np.sum(initial_n, axis=0), 1e-3), 100.0]
        initial = np.r_[initial, np.zeros(width + 1)]
    counters = diagnostic_counters if diagnostic_counters is not None else {}
    counters.setdefault("trial_domain_extensions", 0)
    minimum_trial_counters = [0.0] * utility_count
    corrections_J: dict[str, float] = {}
    corrections_mol: dict[str, float] = {}
    cache: tuple[
        tuple[float, tuple[float, ...]],
        tuple[_PhaseInventoryView, tuple[_Phase | None, ...]],
    ] | None = None
    # A finite-difference trial usually changes only one compartment. Each
    # constitutive closure depends on that cell's n/H, not on the global
    # clock or utility integrals. Keep one exact point per cell for this solve
    # only; rates and event functions still see every candidate and time.
    cell_bounds: list[tuple[tuple[float, ...], tuple[float, float]] | None] = [None] * count
    cell_flashes: list[tuple[tuple[tuple[float, ...], float], _Phase] | None] = [None] * count
    evaluations = [0]

    @overload
    def unpack(vector: _Vector, *, accepted: Literal[True], time: float = 0) -> _State: ...

    @overload
    def unpack(
        vector: _Vector, *, accepted: Literal[False] = False, time: float = 0,
    ) -> _PhaseInventoryView: ...

    def unpack(
        vector: _Vector, *, accepted: bool = False, time: float = 0,
    ) -> _State | _PhaseInventoryView:
        raw = initial_n if constant_amounts else vector[:n_end].reshape((count, width))
        if accepted and np.min(raw) < -1e-12:
            raise RuntimeError("accepted phase state has a negative inventory")
        amounts = tuple(
            dict(zip(ids, map(float, np.where(absent, 0, np.maximum(row, 0))), strict=True))
            for row in raw
        )
        enthalpies = tuple(map(float, vector[n_end:h_end]))
        if not accepted:
            for j in range(utility_count):
                minimum_trial_counters[j] = min(minimum_trial_counters[j], float(vector[h_end + j]))
            return _PhaseInventoryView(amounts, enthalpies, state.time_s + time)
        for i, j in zip(*np.nonzero(raw < 0), strict=True):
            corrections_mol[f"cell:{i}:{ids[j]}@{time}"] = float(-raw[i, j])
        increments = [float(vector[h_end + j]) for j in range(utility_count)]
        for j, value in enumerate(increments):
            if value < -1e-8:
                raise RuntimeError("accepted phase utility integral is negative")
            if value < 0:
                corrections_J[f"utility:{j}@{time}"] = -value
                increments[j] = 0.0
        outlet: dict[str, Any] = {}
        if open_outlet:
            raw_out = vector[q_end : q_end + width]
            if np.min(raw_out) < -1e-12:
                raise RuntimeError("accepted outlet amount integral is negative")
            for j, value in enumerate(raw_out):
                if value < 0:
                    corrections_mol[f"outlet:{ids[j]}@{time}"] = float(-value)
            outlet_state = cast(_OutletInventory, state)
            previous = outlet_state.outlet_amounts_mol
            outlet = {
                "outlet_amounts_mol": {
                    key: previous.get(key, 0) + max(float(raw_out[j]), 0.0)
                    for j, key in enumerate(ids)
                },
                "outlet_enthalpy_J": outlet_state.outlet_enthalpy_J + float(vector[-1]),
            }
        shaft = {} if not include_shaft_work else {
            "shaft_import_energy_J": cast(_WorkInventory, state).shaft_import_energy_J
            + increments[2],
            "shaft_export_energy_J": cast(_WorkInventory, state).shaft_export_energy_J
            + increments[3],
        }
        return replace(
            state,
            amounts_mol=amounts,
            enthalpies_J=enthalpies,
            time_s=state.time_s + time,
            heater_energy_J=state.heater_energy_J + increments[0],
            condenser_energy_J=state.condenser_energy_J + increments[1],
            **shaft,
            **outlet,
        )

    def evaluate(
        scaled: _Vector, time: float,
    ) -> tuple[_PhaseInventoryView, tuple[_Phase | None, ...]]:
        nonlocal cache
        vector = scaled * scales
        candidate = unpack(vector, time=time)
        key = (time, tuple(scaled[:h_end]))
        if cache is not None and cache[0] == key:
            return cache[1]
        flashes: list[_Phase | None] = [None] * count
        for i in closure_indices:
            n, h = candidate.amounts_mol[i], candidate.enthalpies_J[i]
            n_key = tuple(n[k] for k in ids)
            bounds = cell_bounds[i]
            if not reuse_cell_closures or bounds is None or bounds[0] != n_key:
                bound_function = cast(
                    Callable[[int, Mapping[str, float]], tuple[float, float]], enthalpy_bounds,
                )
                bounds = (n_key, bound_function(i, n))
                cell_bounds[i] = bounds
            low, high = bounds[1]
            if not low <= h <= high:
                counters["trial_domain_extensions"] += 1
            bounded_h = min(max(h, low), high)
            key_i = (n_key, bounded_h)
            cached = cell_flashes[i]
            if not reuse_cell_closures or cached is None or cached[0] != key_i:
                # Only declared closure cells require at_ph. Zero-closure models
                # supply their own isothermal rates and have no phase evaluator.
                closure = cast(_InventoryClosure[_Phase], spec)
                cached = (key_i, closure.at_ph(i, n, bounded_h))
                cell_flashes[i] = cached
            flashes[i] = cached[1]
        cache = (key, (candidate, tuple(flashes)))
        return cache[1]

    def rhs(time: float, scaled: _Vector) -> _Vector:
        candidate, flashes = evaluate(scaled, time)
        contribution = rates(candidate, flashes)
        if constant_amounts and np.any(contribution.amounts_mol_s != 0):
            raise ValueError("constant-inventory heat integration requires zero matter rates")
        if np.any(contribution.amounts_mol_s[:, absent] != 0):
            raise ValueError("closed-input phase contribution creates an absent component")
        derivative = np.r_[
            np.empty(0) if constant_amounts else contribution.amounts_mol_s.ravel(),
            contribution.enthalpy_W,
            contribution.heater_W,
            contribution.condenser_W,
        ]
        work = (contribution.shaft_import_W, contribution.shaft_export_W)
        if any(isinstance(value, bool) or not isfinite(value) or value < 0 for value in work):
            raise ValueError("shaft powers require finite nonnegative import/export values")
        if include_shaft_work:
            derivative = np.r_[derivative, work]
        elif any(work):
            raise ValueError("shaft power requires explicitly enabled shaft accounting")
        if open_outlet:
            outgoing = contribution.outlet_amounts_mol_s
            if set(outgoing) - set(ids) or any(not isfinite(v) or v < 0 for v in outgoing.values()):
                raise ValueError("outlet ports require finite nonnegative component rates")
            derivative = np.r_[
                derivative, [outgoing.get(key, 0) for key in ids], contribution.outlet_enthalpy_W
            ]
        elif contribution.outlet_amounts_mol_s or contribution.outlet_enthalpy_W:
            raise ValueError("outlet contribution needs a declared open boundary")
        if not np.all(np.isfinite(derivative)):
            raise RuntimeError("phase contribution contains a nonfinite rate")
        evaluations[0] += 1
        if progress is not None:
            progress(time, evaluations[0])
        return derivative / scales

    names: list[str] = []
    events: list[_EventCallback] = []
    for name, function in stop_functions:

        def root(
            time: float, scaled: _Vector, value: PhaseInventoryStop[_Phase] = function,
        ) -> float:
            return value(*evaluate(scaled, time))

        callback = cast(_EventCallback, root)
        callback.terminal, callback.direction = True, -1
        names.append(name)
        events.append(callback)
    for name, event_function in zip(names, events, strict=True):
        if event_function(0, initial / scales) <= 0:
            if initial_boundary == "reject":
                raise operating_limit_error(f"phase system starts at or outside {name}")
            return result_type(
                state,
                state,
                (state,),
                name,
                {"initial_boundary": True},
                dict.fromkeys(ids, 0.0),
                0.0,
            )
    if duration_s == 0:
        return result_type(
            state,
            state,
            (state,),
            "zero_duration",
            {"initial_boundary": True},
            dict.fromkeys(ids, 0.0),
            0.0,
        )
    report = solve_ode(
        rhs,
        initial / scales,
        time_span_s=(0, duration_s),
        evaluation_times_s=evaluation_times_s,
        events=events,
        jac_sparsity=jac_sparsity,
        policy=ODESolverPolicy(
            solver_id, "BDF", rtol, 1e-11, max_step_s=max_step_s, event_names=tuple(names)
        ),
    )
    report.raise_for_failure("finite phase inventory")
    trajectory = [
        unpack(y * scales, accepted=True, time=float(t))
        for t, y in zip(report.t, report.y.T, strict=True)
    ]
    stopping = None
    if report.terminal_point is not None:
        time, vector, index = report.terminal_point
        final = unpack(vector * scales, accepted=True, time=time)
        stopping = names[index]
        if not trajectory or trajectory[-1].time_s != final.time_s:
            trajectory.append(final)
    if not trajectory:
        raise RuntimeError("phase solver returned no accepted state")
    if trajectory[0].time_s != state.time_s:
        trajectory.insert(0, state)
    final = trajectory[-1]
    outlet_delta = (
        dict.fromkeys(ids, 0.0)
        if not open_outlet
        else {
            key: cast(_OutletInventory, final).outlet_amounts_mol.get(key, 0)
            - cast(_OutletInventory, state).outlet_amounts_mol.get(key, 0)
            for key in ids
        }
    )
    residuals = {
        key: sum(n[key] for n in final.amounts_mol)
        - sum(n[key] for n in state.amounts_mol)
        + outlet_delta[key]
        for key in ids
    }
    energy_residual = (
        sum(final.enthalpies_J)
        - sum(state.enthalpies_J)
        - final.heater_energy_J
        + state.heater_energy_J
        + final.condenser_energy_J
        - state.condenser_energy_J
        + (cast(_OutletInventory, final).outlet_enthalpy_J
           - cast(_OutletInventory, state).outlet_enthalpy_J if open_outlet else 0)
    )
    if include_shaft_work:
        final_work, initial_work = cast(_WorkInventory, final), cast(_WorkInventory, state)
        energy_residual += (
            -(final_work.shaft_import_energy_J - initial_work.shaft_import_energy_J)
            + final_work.shaft_export_energy_J - initial_work.shaft_export_energy_J
        )
    material_scale = (
        sum(map(sum, (n.values() for n in state.amounts_mol)))
        if balance_scale_mol is None
        else balance_scale_mol
    )
    energy_scale = max(1, abs(sum(state.enthalpies_J)),
                       final.heater_energy_J - state.heater_energy_J)
    if include_shaft_work:
        energy_scale = max(
            energy_scale,
            final_work.shaft_import_energy_J - initial_work.shaft_import_energy_J,
            final_work.shaft_export_energy_J - initial_work.shaft_export_energy_J,
            final.condenser_energy_J - state.condenser_energy_J,
        )
    if max(map(abs, residuals.values())) > 1e-8 * max(material_scale, 1e-8) or abs(
        energy_residual
    ) > 1e-6 * energy_scale:
        raise RuntimeError("phase advance failed independent component/H conservation")
    return result_type(
        state,
        final,
        tuple(trajectory),
        stopping,
        {
            **report.diagnostic.to_dict(),
            "constant_amounts": constant_amounts,
            **counters,
            "invariant_zero_components": [key for j, key in enumerate(ids) if absent[j]],
            "minimum_trial_utility_integrals_J": minimum_trial_counters,
            "roundoff_resource_corrections_J": corrections_J,
            "roundoff_inventory_corrections_mol": corrections_mol,
            **({"shaft_accounting": "separate_import_export"} if include_shaft_work else {}),
        },
        residuals,
        float(energy_residual),
    )
