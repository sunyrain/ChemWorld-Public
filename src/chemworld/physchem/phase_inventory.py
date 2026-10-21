"""Common finite-compartment n/H advance with algebraic phase closure.

Columns and concentrators supply their own equipment ports and stop laws.
This numerical strategy owns no session or persisted inventory. Outgoing
boundary integrals are measurements, separate from retained compartment n/H.
"""

from dataclasses import dataclass, field, replace
from math import isfinite
from typing import Any

import numpy as np

from chemworld.physchem.solver_backend import ODESolverPolicy, solve_ode


@dataclass(frozen=True)
class PhaseInventoryRates:
    amounts_mol_s: np.ndarray
    enthalpy_W: np.ndarray
    heater_W: float
    condenser_W: float
    streams: tuple[dict, ...]
    outlet_amounts_mol_s: dict[str, float] = field(default_factory=dict, kw_only=True)
    outlet_enthalpy_W: float = field(default=0.0, kw_only=True)


@dataclass(frozen=True)
class PhaseInventoryAdvance:
    initial_state: Any
    final_state: Any
    trajectory: tuple
    stopping_event: str | None
    solver_diagnostic: dict
    component_residuals_mol: dict[str, float]
    enthalpy_residual_J: float

    @property
    def actual_duration_s(self):
        return self.final_state.time_s - self.initial_state.time_s


@dataclass(frozen=True)
class _PhaseInventoryView:
    amounts_mol: tuple[dict[str, float], ...]
    enthalpies_J: tuple[float, ...]
    time_s: float


def integrate_phase_inventory(
    spec,
    state,
    *,
    duration_s,
    rates,
    closure_indices,
    stop_functions,
    enthalpy_bounds,
    result_type=PhaseInventoryAdvance,
    evaluation_times_s=None,
    max_step_s=None,
    rtol=1e-8,
    progress=None,
    jac_sparsity=None,
    solver_id="finite_phase_inventory_bdf_v1",
    open_outlet=False,
    balance_scale_mol=None,
    diagnostic_counters=None,
    operating_limit_error=ValueError,
    initial_boundary="reject",
    constant_amounts=False,
):
    """One solve for retained n/H, utility heat and explicit outlet integrals.

    ``rates`` contributes a single candidate derivative. PH is algebraic;
    utility and outlet counters cannot alter the constitutive candidate state.
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
        or (constant_amounts and open_outlet)
    ):
        raise ValueError("phase integration needs matching finite inventories, H and duration")
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
    q_end = h_end + 2
    scales = np.r_[
        np.empty(0) if constant_amounts else np.maximum(initial_n, 1e-3).ravel(),
        np.maximum(np.abs(state.enthalpies_J), 10),
        100.0,
        100.0,
    ]
    initial = np.r_[np.empty(0) if constant_amounts else initial_n.ravel(),
                    state.enthalpies_J, 0.0, 0.0]
    if open_outlet:
        scales = np.r_[scales, np.maximum(np.sum(initial_n, axis=0), 1e-3), 100.0]
        initial = np.r_[initial, np.zeros(width + 1)]
    counters = diagnostic_counters if diagnostic_counters is not None else {}
    counters.setdefault("trial_domain_extensions", 0)
    minimum_trial_counters = [0.0, 0.0]
    corrections_J, corrections_mol = {}, {}
    cache = [None, None]
    evaluations = [0]

    def unpack(vector, *, accepted=False, time=0):
        raw = initial_n if constant_amounts else vector[:n_end].reshape((count, width))
        if accepted and np.min(raw) < -1e-12:
            raise RuntimeError("accepted phase state has a negative inventory")
        amounts = tuple(
            dict(zip(ids, map(float, np.where(absent, 0, np.maximum(row, 0))), strict=True))
            for row in raw
        )
        enthalpies = tuple(map(float, vector[n_end:h_end]))
        if not accepted:
            for j in (0, 1):
                minimum_trial_counters[j] = min(minimum_trial_counters[j], float(vector[h_end + j]))
            return _PhaseInventoryView(amounts, enthalpies, state.time_s + time)
        for i, j in zip(*np.nonzero(raw < 0), strict=True):
            corrections_mol[f"cell:{i}:{ids[j]}@{time}"] = float(-raw[i, j])
        increments = [float(vector[h_end]), float(vector[h_end + 1])]
        for j, value in enumerate(increments):
            if value < -1e-8:
                raise RuntimeError("accepted phase utility integral is negative")
            if value < 0:
                corrections_J[f"utility:{j}@{time}"] = -value
                increments[j] = 0.0
        outlet = {}
        if open_outlet:
            raw_out = vector[q_end : q_end + width]
            if np.min(raw_out) < -1e-12:
                raise RuntimeError("accepted outlet amount integral is negative")
            for j, value in enumerate(raw_out):
                if value < 0:
                    corrections_mol[f"outlet:{ids[j]}@{time}"] = float(-value)
            previous = state.outlet_amounts_mol
            outlet = {
                "outlet_amounts_mol": {
                    key: previous.get(key, 0) + max(float(raw_out[j]), 0.0)
                    for j, key in enumerate(ids)
                },
                "outlet_enthalpy_J": state.outlet_enthalpy_J + float(vector[-1]),
            }
        return replace(
            state,
            amounts_mol=amounts,
            enthalpies_J=enthalpies,
            time_s=state.time_s + time,
            heater_energy_J=state.heater_energy_J + increments[0],
            condenser_energy_J=state.condenser_energy_J + increments[1],
            **outlet,
        )

    def evaluate(scaled, time):
        vector = scaled * scales
        candidate = unpack(vector, time=time)
        key = (time, tuple(scaled[:h_end]))
        if cache[0] == key:
            return cache[1]
        flashes = [None] * count
        for i in closure_indices:
            n, h = candidate.amounts_mol[i], candidate.enthalpies_J[i]
            low, high = enthalpy_bounds(i, n)
            if not low <= h <= high:
                counters["trial_domain_extensions"] += 1
            flashes[i] = spec.at_ph(i, n, min(max(h, low), high))
        cache[:] = [key, (candidate, tuple(flashes))]
        return cache[1]

    def rhs(time, scaled):
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

    names, events = [], []
    for name, function in stop_functions:

        def root(time, scaled, value=function):
            return value(*evaluate(scaled, time))

        root.terminal, root.direction = True, -1
        names.append(name)
        events.append(root)
    for name, function in zip(names, events, strict=True):
        if function(0, initial / scales) <= 0:
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
            key: final.outlet_amounts_mol.get(key, 0) - state.outlet_amounts_mol.get(key, 0)
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
        + (final.outlet_enthalpy_J - state.outlet_enthalpy_J if open_outlet else 0)
    )
    material_scale = (
        sum(map(sum, (n.values() for n in state.amounts_mol)))
        if balance_scale_mol is None
        else balance_scale_mol
    )
    if max(map(abs, residuals.values())) > 1e-8 * max(material_scale, 1e-8) or abs(
        energy_residual
    ) > 1e-6 * max(1, abs(sum(state.enthalpies_J)), final.heater_energy_J - state.heater_energy_J):
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
        },
        residuals,
        float(energy_residual),
    )
