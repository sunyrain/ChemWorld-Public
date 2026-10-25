"""Finite objective/constraint design on original read-only SBML candidates.

This is an optimization consumer, not parameter estimation. The objective and
constraints name existing declared readouts, including source assignment-rule
parameters. No response data, surrogate, penalty physics or new expression DSL
is introduced. Candidate controls never rerun source initialization.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from math import isclose, isfinite
from types import MappingProxyType
from typing import Any

from chemworld.foundation.units import convert_value
from chemworld.physchem.biochemical_analysis import analyze_biochemical_steady_state
from chemworld.physchem.biochemical_analysis_specs import BiochemicalAnalysisPolicy
from chemworld.physchem.bounded_design import (
    BoundedDesignPolicy,
    DesignConstraint,
    DesignEvaluationError,
    DesignPoint,
    DesignVariable,
    solve_bounded_design,
)
from chemworld.physchem.coupled_system import CoupledState
from chemworld.physchem.sbml_sensitivity import check_sensitivity_scope
from chemworld.physchem.sbml_system import (
    SBMLSystemSpec,
    _parameter_signature,
    parameter_observable,
)


@dataclass(frozen=True)
class SBMLDesignControl:
    variable: DesignVariable
    source_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.variable, DesignVariable) or not isinstance(self.source_id, str):
            raise ValueError("SBML design requires a shared DesignVariable and source parameter")
        if not self.source_id or self.source_id.strip() != self.source_id:
            raise ValueError("source parameter identity must be explicit")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SBMLDesignMetric:
    metric_id: str
    channel_id: str
    sample_offset_s: float = 0.0

    def __post_init__(self) -> None:
        if any(
            not isinstance(v, str) or not v or v.strip() != v
            for v in (self.metric_id, self.channel_id)
        ):
            raise ValueError("design metric needs explicit metric and original channel identities")
        if (
            isinstance(self.sample_offset_s, bool)
            or not isfinite(self.sample_offset_s)
            or (self.sample_offset_s < 0)
        ):
            raise ValueError("metric sample offset must be finite nonnegative seconds")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SBMLDesignSpec:
    controls: tuple[SBMLDesignControl, ...]
    metrics: tuple[SBMLDesignMetric, ...]
    objective_metric: str
    maximize: bool = False
    mode: str = "trajectory"
    constraints: tuple[DesignConstraint, ...] = ()
    starts: tuple[Mapping[str, float], ...] = ()
    policy: BoundedDesignPolicy = field(default_factory=BoundedDesignPolicy)
    steady_policy: BiochemicalAnalysisPolicy = field(default_factory=BiochemicalAnalysisPolicy)

    def __post_init__(self) -> None:
        for name in ("controls", "metrics", "constraints"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        object.__setattr__(self, "starts", tuple(MappingProxyType(dict(s)) for s in self.starts))
        if not self.controls or any(not isinstance(c, SBMLDesignControl) for c in self.controls):
            raise ValueError("design requires explicit original SBML controls")
        if len({c.source_id for c in self.controls}) != len(self.controls) or len(
            {c.variable.parameter_id for c in self.controls}
        ) != len(self.controls):
            raise ValueError("design control and source identities must be unique")
        if not self.metrics or any(not isinstance(m, SBMLDesignMetric) for m in self.metrics):
            raise ValueError("design requires explicit named readout metrics")
        names = {m.metric_id for m in self.metrics}
        if len(names) != len(self.metrics) or self.objective_metric not in names:
            raise ValueError("distinct metrics must contain the declared objective")
        if self.mode not in {"trajectory", "steady"} or type(self.maximize) is not bool:
            raise ValueError("choose trajectory/steady design and an explicit objective direction")
        if self.mode == "steady" and any(m.sample_offset_s != 0 for m in self.metrics):
            raise ValueError("steady metrics have no transient sampling offset")
        if not isinstance(self.policy, BoundedDesignPolicy) or not isinstance(
            self.steady_policy, BiochemicalAnalysisPolicy
        ):
            raise ValueError("design uses the original bounded and steady solver policies")
        if any(
            not isinstance(c, DesignConstraint) or c.metric not in names for c in self.constraints
        ):
            raise ValueError("constraints must name declared design metrics")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "native-sbml-design-v1",
            "controls": [c.to_dict() for c in self.controls],
            "metrics": [m.to_dict() for m in self.metrics],
            "objective_metric": self.objective_metric,
            "maximize": self.maximize,
            "mode": self.mode,
            "constraints": [asdict(c) for c in self.constraints],
            "starts": [dict(s) for s in self.starts],
            "policy": asdict(self.policy),
            "steady_policy": self.steady_policy.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> SBMLDesignSpec:
        values = dict(payload)
        if values.pop("schema", None) != "native-sbml-design-v1":
            raise ValueError("unknown native SBML design declaration")
        values["controls"] = tuple(
            SBMLDesignControl(DesignVariable(**c["variable"]), c["source_id"])
            for c in values["controls"]
        )
        values["metrics"] = tuple(SBMLDesignMetric(**m) for m in values["metrics"])
        values["constraints"] = tuple(DesignConstraint(**c) for c in values["constraints"])
        values["policy"] = BoundedDesignPolicy(**values["policy"])
        values["steady_policy"] = BiochemicalAnalysisPolicy(**values["steady_policy"])
        return cls(**values)


class _SBMLDesignEvaluation:
    def __init__(
        self,
        model: SBMLSystemSpec,
        current_values: Mapping[str, float],
        spec: SBMLDesignSpec,
        start_time_s: float,
    ) -> None:
        check_sensitivity_scope(model, tuple(c.source_id for c in spec.controls))
        if isinstance(start_time_s, bool) or not isfinite(start_time_s) or start_time_s < 0:
            raise ValueError("design current time must be finite nonnegative seconds")
        if set(current_values) != set(model.frame.initial_values):
            raise ValueError("design requires the original complete current coordinates")
        self.model, self.spec, self.start_time_s = model, spec, start_time_s
        self.initial = dict(current_values)
        model.frame.validate_values(self.initial, start_time_s)
        factors = {model.model.declaration.get("conversionFactor")} | {
            s.get("conversionFactor") for s in model.frame._species.values()
        }
        if factors & {c.source_id for c in spec.controls}:
            raise ValueError("design controls cannot change stoichiometric conversionFactor")
        self.control_units = {
            c.source_id: parameter_observable(model.model, c.source_id).unit for c in spec.controls
        }
        for control in spec.controls:
            convert_value(1.0, control.variable.unit, self.control_units[control.source_id])
        channels = {o.channel_id: o.unit for o in model.observables}
        if any(m.channel_id not in channels for m in spec.metrics):
            raise ValueError("design metrics must consume existing declared SBML readout channels")
        self.metric_units = {m.metric_id: channels[m.channel_id] for m in spec.metrics}
        if any(c.unit != self.metric_units[c.metric] for c in spec.constraints):
            raise ValueError("constraint unit must match its original native readout unit")
        if not isfinite(start_time_s + max(m.sample_offset_s for m in spec.metrics)):
            raise ValueError("design sample times exceed the finite laboratory time domain")
        self.system = model.continuous_system() if spec.mode == "trajectory" else None

    def source_values(self, values: Mapping[str, float]) -> dict[str, float]:
        if set(values) != {c.variable.parameter_id for c in self.spec.controls}:
            raise ValueError("candidate must name exactly the design parameter whitelist")
        result = {}
        for control in self.spec.controls:
            number = values[control.variable.parameter_id]
            control.variable.encode(number)
            result[control.source_id] = (
                convert_value(number, control.variable.unit, self.control_units[control.source_id])
                / _parameter_signature(self.model.model, control.source_id).si_factor
            )
        return result

    def __call__(self, values: dict[str, float]) -> DesignPoint:
        diagnostics: dict[str, Any] = {
            "mode": self.spec.mode,
            "sample_records": [],
            "continuous_solves": [],
            "steady_analysis": None,
        }
        try:
            source = self.source_values(values)
            diagnostics["source_controls"] = source
            controlled = self.model.frame.controlled_values(self.initial, self.start_time_s, source)
            diagnostics["initial_coordinates_after_control"] = dict(controlled)
            observations = {}
            if self.spec.mode == "steady":
                result = analyze_biochemical_steady_state(
                    self.model.frame,
                    controlled,
                    time_s=self.start_time_s,
                    policy=self.spec.steady_policy,
                    include_control=False,
                    include_lna=False,
                )
                diagnostics["steady_analysis"] = result.to_dict()
                if not result.success:
                    raise DesignEvaluationError(
                        "steady design candidate did not converge", diagnostics
                    )
                candidate = {
                    **controlled,
                    **{
                        "N:" + name: amount
                        for name, amount in (result.steady_amounts_mol or {}).items()
                    },
                }
                readouts = self.model.readout(candidate, self.start_time_s)
                observations = {
                    m.metric_id: float(readouts[m.channel_id]) for m in self.spec.metrics
                }
                diagnostics["sample_records"].append(
                    {
                        "kind": "steady",
                        "coordinates": candidate,
                        "readouts": readouts,
                    }
                )
            else:
                _, volumes = self.model.frame.quantities(controlled, self.start_time_s)
                candidate_state = CoupledState(
                    {},
                    298.15,
                    next(iter(volumes.values())),
                    time_s=self.start_time_s,
                    coordinates=controlled,
                )
                previous = 0.0
                for offset in sorted({m.sample_offset_s for m in self.spec.metrics}):
                    if offset > previous:
                        assert self.system is not None
                        advanced = self.system.advance(candidate_state, offset - previous)
                        diagnostics["continuous_solves"].append(dict(advanced.solver_diagnostic))
                        if advanced.outcome != "success" or not isclose(
                            advanced.actual_duration_s,
                            offset - previous,
                            rel_tol=1e-12,
                            abs_tol=1e-12,
                        ):
                            raise DesignEvaluationError(
                                "trajectory design sample was not reached", diagnostics
                            )
                        candidate_state = advanced.final_state
                    readouts = self.model.readout(
                        candidate_state.coordinates, candidate_state.time_s
                    )
                    diagnostics["sample_records"].append(
                        {
                            "sample_offset_s": offset,
                            "time_s": candidate_state.time_s,
                            "coordinates": dict(candidate_state.coordinates),
                            "readouts": readouts,
                        }
                    )
                    observations.update(
                        {
                            m.metric_id: float(readouts[m.channel_id])
                            for m in self.spec.metrics
                            if m.sample_offset_s == offset
                        }
                    )
                    previous = offset
            objective = observations[self.spec.objective_metric]
            diagnostics["objective_value_original_units"] = objective
            return DesignPoint(
                -objective if self.spec.maximize else objective, observations, diagnostics
            )
        except (ValueError, ArithmeticError) as exc:
            if isinstance(exc, DesignEvaluationError):
                raise
            raise DesignEvaluationError(str(exc), diagnostics) from exc


def optimize_sbml_design(
    model: SBMLSystemSpec,
    current_values: Mapping[str, float],
    spec: SBMLDesignSpec,
    *,
    start_time_s: float = 0.0,
    progress: Any = None,
) -> dict[str, Any]:
    """Optimize declared controls from fixed current inventories, without committing.

    Steady candidates all use the same supplied state as their local root seed
    and conserved-total class. Root stability and failures remain in every trial;
    neither global branch selection nor a global optimum is claimed.
    """
    if not isinstance(spec, SBMLDesignSpec):
        raise ValueError("SBML optimization requires an explicit native design declaration")
    evaluate = _SBMLDesignEvaluation(model, current_values, spec, start_time_s)
    result = solve_bounded_design(
        tuple(c.variable for c in spec.controls),
        spec.constraints,
        evaluate,
        starts=spec.starts,
        policy=spec.policy,
        progress=progress,
    )
    return {
        "schema": "native-sbml-design-result-v1",
        "design": spec.to_dict(),
        "model_id": model.model.declaration.get("id", "anonymous"),
        "source": model.model.source.to_dict(),
        "start_time_s": start_time_s,
        "initial_coordinates": dict(current_values),
        "metric_units": evaluate.metric_units,
        "objective": {
            "metric": spec.objective_metric,
            "maximize": spec.maximize,
            "unit": evaluate.metric_units[spec.objective_metric],
            "value": None
            if result.point is None
            else result.point.observations[spec.objective_metric],
        },
        "optimization": result.to_dict(),
        "recommended_source_parameters": None
        if result.values is None
        else evaluate.source_values(result.values),
        "world_advanced": False,
        "live_resources_charged": False,
        "initialization": "hold_supplied_initialized_coordinates_then_apply_source_controls",
        "steady_root_selection": "same_current_state_seed_per_candidate; stability recorded",
        "boundary": (
            "Finite-start local objective/constraint optimization, not parameter estimation or "
            "a global optimum. Hypothetical roots/trajectories never replace actual world state. "
            "Formal chemical composition and energy remain undeclared."
        ),
    }
