"""Frozen generated-composition design for the first-paper qualification."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from chemworld.world.composition_coverage import (
    CompositionCoverageSuite,
    CompositionCoverageTarget,
    ContinuousCoverageAxis,
    DiscreteCoverageAxis,
    OrderedWorkflowTemplate,
    generate_world_composition_coverage,
)
from chemworld.world.process_time_budget import derive_process_time_budget_policy

QUALIFICATION_DESIGN_VERSION = "first-paper-composition-qualification-design-v3"
EXPECTED_PATTERN_CASE_COUNTS = {
    "phase-observation": 6,
    "reaction-thermal-observation": 6,
    "phase-separation-observation": 6,
    "reaction-crystallization-observation": 6,
    "reaction-distillation-observation": 8,
    "reaction-continuous-flow-observation": 6,
    "reaction-electrochemistry-observation": 7,
    "reaction-phase-separation-observation": 7,
}
EXPECTED_GENERATED_CASE_COUNT = sum(EXPECTED_PATTERN_CASE_COUNTS.values())
UNSEEN_PATTERN_ID = "reaction-distillation-observation"

_IMPLICIT_PROCESS_TIME_ALLOWANCE_S = {
    "quench": 120.0,
    "collect_fraction": 60.0,
    "filter_crystals": 480.0,
    "separate_phase": 60.0,
    "transfer": 60.0,
}

_ADDITIONAL_PROCESS_REPEATS = {
    "phase-observation": {},
    "reaction-thermal-observation": {"heat": 1},
    "phase-separation-observation": {"mix": 1, "settle": 1},
    "reaction-crystallization-observation": {
        "heat": 1,
        "cool_crystallize": 1,
    },
    "reaction-distillation-observation": {
        "heat": 1,
        "evaporate": 1,
        "distill": 1,
        "collect_fraction": 1,
    },
    "reaction-continuous-flow-observation": {"run_flow": 1},
    "reaction-electrochemistry-observation": {"electrolyze": 1},
    "reaction-phase-separation-observation": {
        "heat": 1,
        "mix": 1,
        "settle": 1,
        "concentrate": 1,
        "transfer": 1,
    },
}


def _component(kind: str, *, role: str | None = None) -> dict[str, Any]:
    return {"kind": kind, "role": role or kind, "parameters": {}}


def _base_request(
    pattern_id: str,
    component_kinds: Sequence[str],
    *,
    budget: int,
    resources: dict[str, float | int],
    operations: Sequence[str] | None = None,
    instruments: Sequence[str] | None = None,
) -> dict[str, Any]:
    task: dict[str, Any] = {
        "budget": budget,
        "resources": {"operation_budget": budget, **resources},
        "description": f"Frozen qualification suite for {pattern_id}.",
        "tags": ["first-paper-qualification", "coverage-generated"],
    }
    if operations is not None:
        task["operations"] = list(operations)
    if instruments is not None:
        task["instruments"] = list(instruments)
    return {
        "schema_version": "chemworld-world-composition-0.1",
        "composition_id": f"qualification-{pattern_id}",
        "world_split": "public-test",
        "components": [_component(kind) for kind in component_kinds],
        "task": task,
    }


def _axis(
    axis_id: str,
    binding: str | tuple[str, ...],
    left: Any,
    right: Any,
) -> DiscreteCoverageAxis:
    bindings = (binding,) if isinstance(binding, str) else binding
    return DiscreteCoverageAxis(
        axis_id=axis_id,
        bindings=bindings,
        values=(left, right),
    )


def _continuous(
    axis_id: str,
    lower: float,
    upper: float,
    unit: str,
) -> ContinuousCoverageAxis:
    return ContinuousCoverageAxis(
        axis_id=axis_id,
        lower=lower,
        upper=upper,
        unit=unit,
    )


def _placeholder(axis_id: str) -> dict[str, str]:
    return {"coverage_axis": axis_id}


def _attach_process_time_policy(
    request: dict[str, Any],
    *,
    pattern_id: str,
    workflows: Sequence[OrderedWorkflowTemplate],
    continuous_axes: Sequence[ContinuousCoverageAxis] = (),
) -> None:
    resources = request["task"]["resources"]
    additional_repeats = dict(_ADDITIONAL_PROCESS_REPEATS[pattern_id])
    declared_instrument_uses = resources.get("instrument_uses")
    required_measure_count = max(
        (
            sum(
                1
                for action in workflow.actions
                if action.get("operation") == "measure"
            )
            for workflow in workflows
        ),
        default=0,
    )
    if declared_instrument_uses is not None:
        if (
            isinstance(declared_instrument_uses, bool)
            or not isinstance(declared_instrument_uses, int)
            or declared_instrument_uses < required_measure_count
        ):
            raise ValueError(
                "task instrument_uses must cover every measurement in the frozen workflow"
            )
        additional_repeats["measure"] = (
            declared_instrument_uses - required_measure_count
        )
    policy = derive_process_time_budget_policy(
        pattern_id=pattern_id,
        workflows=workflows,
        continuous_axes=continuous_axes,
        additional_repeat_limits=additional_repeats,
        implicit_operation_allowance_s=_IMPLICIT_PROCESS_TIME_ALLOWANCE_S,
    )
    resources["time_s"] = policy.process_time_limit_s
    resources["process_time_policy"] = policy.to_dict()


def _reaction_charge(*, include_catalyst: bool = False) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = [
        {"operation": "add_solvent", "volume_L": 0.025, "solvent": 1},
        {"operation": "add_reagent", "amount_mol": 0.010},
    ]
    if include_catalyst:
        actions.append(
            {
                "operation": "add_catalyst",
                "catalyst_amount_mol": 0.0002,
                "catalyst": 1,
            }
        )
    return actions


def _heat_action() -> dict[str, Any]:
    return {
        "operation": "heat",
        "target_temperature_K": _placeholder("reaction_temperature_K"),
        "duration_s": _placeholder("reaction_duration_s"),
        "stirring_speed_rpm": 650.0,
    }


def _close_actions() -> list[dict[str, Any]]:
    return [
        {"operation": "terminate"},
        {"operation": "measure", "instrument": "final_assay"},
    ]


def _phase_observation_suite() -> CompositionCoverageSuite:
    pattern_id = "phase-observation"
    request = _base_request(
        pattern_id,
        ("phase", "observation"),
        budget=5,
        resources={
            "sample_volume_L": 0.001,
            "instrument_uses": 3,
            "final_assays": 1,
        },
        operations=("add_solvent", "add_reagent", "terminate", "measure"),
    )
    axes = (
        _axis(
            "phase_profile",
            "components.phase.parameters.phases",
            ["aqueous"],
            ["aqueous", "organic"],
        ),
        _axis(
            "instrument_profile",
            (
                "components.observation.parameters.instruments",
                "task.instruments",
            ),
            ["ph_meter", "final_assay"],
            ["ph_meter", "uvvis", "final_assay"],
        ),
        _axis(
            "objective",
            "task.objective",
            "balanced",
            "yield",
        ),
    )
    workflows = (
        OrderedWorkflowTemplate(
            workflow_id="charge-characterize-then-final-assay",
            actions=(
                {"operation": "add_solvent", "volume_L": 0.025, "solvent": 0},
                {"operation": "add_reagent", "amount_mol": 0.010},
                {"operation": "measure", "instrument": "ph_meter"},
                *_close_actions(),
            ),
        ),
    )
    _attach_process_time_policy(
        request,
        pattern_id=pattern_id,
        workflows=workflows,
    )
    return generate_world_composition_coverage(
        request,
        suite_id="qualification-phase-observation",
        discrete_axes=axes,
        workflows=workflows,
        target=CompositionCoverageTarget(continuous_samples=4, seed=101),
    )


def _reaction_thermal_suite() -> CompositionCoverageSuite:
    pattern_id = "reaction-thermal-observation"
    request = _base_request(
        pattern_id,
        ("reaction", "thermal", "observation"),
        budget=8,
        resources={
            "sample_volume_L": 0.001,
            "time_s": 3600.0,
            "instrument_uses": 3,
            "final_assays": 1,
        },
    )
    axes = (
        _axis(
            "reaction_family",
            "components.reaction.parameters.family",
            "declared-family-a",
            "declared-family-b",
        ),
        _axis(
            "thermal_range",
            "components.thermal.parameters.temperature_range_K",
            [340.0, 400.0],
            [345.0, 405.0],
        ),
        _axis(
            "instrument_profile",
            (
                "components.observation.parameters.instruments",
                "task.instruments",
            ),
            ["hplc", "final_assay"],
            ["hplc", "uvvis", "final_assay"],
        ),
    )
    continuous = (
        _continuous("reaction_temperature_K", 350.0, 390.0, "K"),
        _continuous("reaction_duration_s", 600.0, 1800.0, "s"),
    )
    direct = [*_reaction_charge(), _heat_action(), *_close_actions()]
    measured = [
        *_reaction_charge(),
        _heat_action(),
        {"operation": "measure", "instrument": "hplc"},
        *_close_actions(),
    ]
    workflows = (
        OrderedWorkflowTemplate("direct-final-assay", tuple(direct)),
        OrderedWorkflowTemplate("process-measurement", tuple(measured)),
    )
    _attach_process_time_policy(
        request,
        pattern_id=pattern_id,
        workflows=workflows,
        continuous_axes=continuous,
    )
    return generate_world_composition_coverage(
        request,
        suite_id="qualification-reaction-thermal-observation",
        discrete_axes=axes,
        continuous_axes=continuous,
        workflows=workflows,
        target=CompositionCoverageTarget(continuous_samples=4, seed=102),
    )


def _phase_separation_suite() -> CompositionCoverageSuite:
    pattern_id = "phase-separation-observation"
    request = _base_request(
        pattern_id,
        ("phase", "separation", "observation"),
        budget=12,
        resources={
            "sample_volume_L": 0.001,
            "time_s": 3600.0,
            "instrument_uses": 3,
            "final_assays": 1,
        },
    )
    axes = (
        _axis(
            "phase_profile",
            "components.phase.parameters.phases",
            ["aqueous", "organic"],
            ["reactor_liquid", "aqueous", "organic"],
        ),
        _axis(
            "instrument_profile",
            (
                "components.observation.parameters.instruments",
                "task.instruments",
            ),
            ["hplc", "final_assay"],
            ["hplc", "gc", "final_assay"],
        ),
        _axis("objective", "task.objective", "balanced", "yield"),
    )
    continuous = (
        _continuous("phase_volume_L", 0.010, 0.020, "L"),
        _continuous("extractant_volume_L", 0.010, 0.025, "L"),
        _continuous("mix_duration_s", 60.0, 300.0, "s"),
        _continuous("settle_duration_s", 120.0, 600.0, "s"),
    )
    prefix: list[dict[str, Any]] = [
        {"operation": "add_solvent", "volume_L": 0.020, "solvent": 1},
        {
            "operation": "add_phase",
            "phase": "aqueous",
            "volume_L": _placeholder("phase_volume_L"),
        },
        {
            "operation": "add_extractant",
            "extractant": 1,
            "volume_L": _placeholder("extractant_volume_L"),
        },
        {
            "operation": "mix",
            "duration_s": _placeholder("mix_duration_s"),
            "stirring_speed_rpm": 650.0,
        },
        {"operation": "settle", "duration_s": _placeholder("settle_duration_s")},
    ]
    before = [
        *prefix,
        {"operation": "measure", "instrument": "hplc"},
        {"operation": "separate_phase", "target_phase": "organic"},
        *_close_actions(),
    ]
    after = [
        *prefix,
        {"operation": "separate_phase", "target_phase": "organic"},
        {"operation": "measure", "instrument": "hplc"},
        *_close_actions(),
    ]
    workflows = (
        OrderedWorkflowTemplate("measure-before-separation", tuple(before)),
        OrderedWorkflowTemplate("measure-after-separation", tuple(after)),
    )
    _attach_process_time_policy(
        request,
        pattern_id=pattern_id,
        workflows=workflows,
        continuous_axes=continuous,
    )
    return generate_world_composition_coverage(
        request,
        suite_id="qualification-phase-separation-observation",
        discrete_axes=axes,
        continuous_axes=continuous,
        workflows=workflows,
        target=CompositionCoverageTarget(continuous_samples=6, seed=103),
    )


def _crystallization_suite() -> CompositionCoverageSuite:
    pattern_id = "reaction-crystallization-observation"
    request = _base_request(
        pattern_id,
        ("reaction", "thermal", "crystallization", "observation"),
        budget=14,
        resources={
            "sample_volume_L": 0.001,
            "time_s": 14_400.0,
            "instrument_uses": 4,
            "final_assays": 1,
        },
        instruments=("hplc", "final_assay"),
    )
    axes = (
        _axis(
            "reaction_family",
            "components.reaction.parameters.family",
            "declared-family-a",
            "declared-family-b",
        ),
        _axis(
            "thermal_range",
            "components.thermal.parameters.temperature_range_K",
            [340.0, 400.0],
            [345.0, 405.0],
        ),
        _axis(
            "seed_mass_range",
            "components.crystallization.parameters.seed_mass_range_g",
            [0.001, 0.012],
            [0.002, 0.015],
        ),
    )
    continuous = (
        _continuous("reaction_temperature_K", 350.0, 390.0, "K"),
        _continuous("reaction_duration_s", 600.0, 1800.0, "s"),
        _continuous("seed_mass_g", 0.002, 0.010, "g"),
        _continuous("cooling_temperature_K", 275.0, 305.0, "K"),
        _continuous("cooling_duration_s", 900.0, 3600.0, "s"),
    )
    crystallize: list[dict[str, Any]] = [
        {"operation": "measure", "instrument": "hplc"},
        {"operation": "seed_crystals", "seed_mass_g": _placeholder("seed_mass_g")},
        {
            "operation": "cool_crystallize",
            "target_temperature_K": _placeholder("cooling_temperature_K"),
            "duration_s": _placeholder("cooling_duration_s"),
        },
        {"operation": "measure", "instrument": "hplc"},
        {"operation": "filter_crystals"},
    ]
    direct = [
        *_reaction_charge(include_catalyst=True),
        _heat_action(),
        {"operation": "quench"},
        *crystallize,
        *_close_actions(),
    ]
    delayed = [
        *_reaction_charge(include_catalyst=True),
        {"operation": "wait", "duration_s": 120.0, "stirring_speed_rpm": 500.0},
        _heat_action(),
        {"operation": "quench"},
        *crystallize,
        *_close_actions(),
    ]
    workflows = (
        OrderedWorkflowTemplate("direct-crystallization", tuple(direct)),
        OrderedWorkflowTemplate("wait-before-heat", tuple(delayed)),
    )
    _attach_process_time_policy(
        request,
        pattern_id=pattern_id,
        workflows=workflows,
        continuous_axes=continuous,
    )
    return generate_world_composition_coverage(
        request,
        suite_id="qualification-reaction-crystallization-observation",
        discrete_axes=axes,
        continuous_axes=continuous,
        workflows=workflows,
        target=CompositionCoverageTarget(continuous_samples=4, seed=104),
    )


def _distillation_suite() -> CompositionCoverageSuite:
    pattern_id = "reaction-distillation-observation"
    request = _base_request(
        pattern_id,
        ("reaction", "thermal", "distillation", "observation"),
        budget=16,
        resources={
            "sample_volume_L": 0.001,
            "time_s": 14_400.0,
            "instrument_uses": 4,
            "final_assays": 1,
        },
    )
    axes = (
        _axis(
            "reaction_family",
            "components.reaction.parameters.family",
            "declared-family-a",
            "declared-family-b",
        ),
        _axis(
            "fraction_count",
            "components.distillation.parameters.fraction_count",
            2,
            4,
        ),
        _axis(
            "instrument_profile",
            (
                "components.observation.parameters.instruments",
                "task.instruments",
            ),
            ["hplc", "gc", "final_assay"],
            ["hplc", "gc", "uvvis", "final_assay"],
        ),
        _axis(
            "thermal_temperature_range_K",
            "components.thermal.parameters.temperature_range_K",
            [340.0, 400.0],
            [345.0, 405.0],
        ),
        _axis(
            "distillation_temperature_range_K",
            "components.distillation.parameters.temperature_range_K",
            [315.0, 400.0],
            [320.0, 405.0],
        ),
        _axis(
            "reflux_ratio_range",
            "components.distillation.parameters.reflux_ratio_range",
            [0.5, 3.5],
            [0.8, 4.0],
        ),
    )
    continuous = (
        _continuous("reaction_temperature_K", 350.0, 390.0, "K"),
        _continuous("reaction_duration_s", 600.0, 1800.0, "s"),
        _continuous("evaporation_temperature_K", 325.0, 345.0, "K"),
        _continuous("evaporation_duration_s", 300.0, 900.0, "s"),
        _continuous("distillation_temperature_K", 350.0, 390.0, "K"),
        _continuous("distillation_duration_s", 900.0, 2400.0, "s"),
        _continuous("reflux_ratio", 1.0, 3.0, "dimensionless"),
        _continuous("transfer_fraction", 0.65, 0.95, "dimensionless"),
    )
    downstream = [
        {
            "operation": "evaporate",
            "target_temperature_K": _placeholder("evaporation_temperature_K"),
            "duration_s": _placeholder("evaporation_duration_s"),
        },
        {
            "operation": "distill",
            "target_temperature_K": _placeholder("distillation_temperature_K"),
            "duration_s": _placeholder("distillation_duration_s"),
            "reflux_ratio": _placeholder("reflux_ratio"),
        },
        {
            "operation": "collect_fraction",
            "transfer_fraction": _placeholder("transfer_fraction"),
        },
    ]
    workflow_a = [
        *_reaction_charge(include_catalyst=True),
        _heat_action(),
        {"operation": "quench"},
        {"operation": "measure", "instrument": "hplc"},
        *downstream,
        {"operation": "measure", "instrument": "gc"},
        *_close_actions(),
    ]
    workflow_b = [
        *_reaction_charge(),
        _heat_action(),
        {"operation": "quench"},
        *downstream,
        {"operation": "measure", "instrument": "hplc"},
        *_close_actions(),
    ]
    workflows = (
        OrderedWorkflowTemplate("hplc-before-gc-after", tuple(workflow_a)),
        OrderedWorkflowTemplate("hplc-after-fraction", tuple(workflow_b)),
    )
    _attach_process_time_policy(
        request,
        pattern_id=pattern_id,
        workflows=workflows,
        continuous_axes=continuous,
    )
    return generate_world_composition_coverage(
        request,
        suite_id="qualification-reaction-distillation-observation",
        discrete_axes=axes,
        continuous_axes=continuous,
        workflows=workflows,
        target=CompositionCoverageTarget(continuous_samples=8, seed=105),
    )


def _flow_suite() -> CompositionCoverageSuite:
    pattern_id = "reaction-continuous-flow-observation"
    request = _base_request(
        pattern_id,
        ("reaction", "thermal", "continuous_flow", "observation"),
        budget=10,
        resources={
            "sample_volume_L": 0.001,
            "time_s": 7200.0,
            "instrument_uses": 3,
            "final_assays": 1,
        },
    )
    axes = (
        _axis(
            "reaction_family",
            "components.reaction.parameters.family",
            "declared-family-a",
            "declared-family-b",
        ),
        _axis(
            "flow_rate_range",
            "components.continuous_flow.parameters.flow_rate_range_mL_min",
            [0.2, 6.0],
            [0.5, 8.0],
        ),
        _axis(
            "residence_time_range",
            "components.continuous_flow.parameters.residence_time_range_s",
            [30.0, 900.0],
            [60.0, 1200.0],
        ),
        _axis(
            "instrument_profile",
            (
                "components.observation.parameters.instruments",
                "task.instruments",
            ),
            ["uvvis", "final_assay"],
            ["hplc", "uvvis", "final_assay"],
        ),
    )
    continuous = (
        _continuous("flow_rate_mL_min", 0.5, 5.0, "mL/min"),
        _continuous("residence_time_s", 60.0, 600.0, "s"),
        _continuous("flow_temperature_K", 330.0, 390.0, "K"),
        _continuous("flow_duration_s", 1200.0, 3600.0, "s"),
    )
    flow = [
        {
            "operation": "set_flow_rate",
            "flow_rate_mL_min": _placeholder("flow_rate_mL_min"),
            "residence_time_s": _placeholder("residence_time_s"),
        },
        {
            "operation": "run_flow",
            "target_temperature_K": _placeholder("flow_temperature_K"),
            "duration_s": _placeholder("flow_duration_s"),
        },
    ]
    workflow_a = [
        *_reaction_charge(include_catalyst=True),
        *flow,
        {"operation": "measure", "instrument": "uvvis"},
        *_close_actions(),
    ]
    workflow_b = [
        *_reaction_charge(),
        {"operation": "measure", "instrument": "uvvis"},
        *flow,
        *_close_actions(),
    ]
    workflows = (
        OrderedWorkflowTemplate("measure-after-flow", tuple(workflow_a)),
        OrderedWorkflowTemplate("measure-before-flow", tuple(workflow_b)),
    )
    _attach_process_time_policy(
        request,
        pattern_id=pattern_id,
        workflows=workflows,
        continuous_axes=continuous,
    )
    return generate_world_composition_coverage(
        request,
        suite_id="qualification-reaction-continuous-flow-observation",
        discrete_axes=axes,
        continuous_axes=continuous,
        workflows=workflows,
        target=CompositionCoverageTarget(continuous_samples=6, seed=106),
    )


def _electrochemistry_suite() -> CompositionCoverageSuite:
    pattern_id = "reaction-electrochemistry-observation"
    request = _base_request(
        pattern_id,
        ("reaction", "electrochemistry", "observation"),
        budget=14,
        resources={
            "sample_volume_L": 0.001,
            "time_s": 7200.0,
            "instrument_uses": 4,
            "final_assays": 1,
        },
    )
    axes = (
        _axis(
            "reaction_family",
            "components.reaction.parameters.family",
            "declared-family-a",
            "declared-family-b",
        ),
        _axis(
            "potential_range",
            "components.electrochemistry.parameters.potential_range_V",
            [0.4, 2.0],
            [0.5, 2.2],
        ),
        _axis(
            "current_range",
            "components.electrochemistry.parameters.current_range_mA",
            [20.0, 180.0],
            [25.0, 200.0],
        ),
        _axis(
            "instrument_profile",
            (
                "components.observation.parameters.instruments",
                "task.instruments",
            ),
            ["ph_meter", "uvvis", "final_assay"],
            ["uvvis", "ph_meter", "final_assay"],
        ),
    )
    continuous = (
        _continuous("potential_V", 0.5, 1.8, "V"),
        _continuous("current_mA", 25.0, 150.0, "mA"),
        _continuous("electrolysis_duration_s", 300.0, 1800.0, "s"),
    )

    def first_electrical_cycle() -> list[dict[str, Any]]:
        return [
            {
                "operation": "set_potential",
                "potential_V": _placeholder("potential_V"),
                "current_mA": _placeholder("current_mA"),
                "electrolyte_profile": 1,
            },
            {
                "operation": "electrolyze",
                "duration_s": _placeholder("electrolysis_duration_s"),
            },
        ]

    second_electrical_cycle: list[dict[str, Any]] = [
        {
            "operation": "set_potential",
            "potential_V": _placeholder("potential_V"),
            "current_mA": 175.0,
            "electrolyte_profile": 1,
        },
        {
            "operation": "electrolyze",
            "duration_s": _placeholder("electrolysis_duration_s"),
        },
    ]

    workflow_a = [
        *_reaction_charge(),
        *first_electrical_cycle(),
        {"operation": "measure", "instrument": "ph_meter"},
        {"operation": "measure", "instrument": "uvvis"},
        *second_electrical_cycle,
        {"operation": "measure", "instrument": "uvvis"},
        *_close_actions(),
    ]
    workflow_b = [
        *_reaction_charge(),
        *first_electrical_cycle(),
        {"operation": "measure", "instrument": "uvvis"},
        {"operation": "measure", "instrument": "ph_meter"},
        *second_electrical_cycle,
        {"operation": "measure", "instrument": "uvvis"},
        *_close_actions(),
    ]
    workflows = (
        OrderedWorkflowTemplate("ph-then-uvvis", tuple(workflow_a)),
        OrderedWorkflowTemplate("uvvis-then-ph", tuple(workflow_b)),
    )
    _attach_process_time_policy(
        request,
        pattern_id=pattern_id,
        workflows=workflows,
        continuous_axes=continuous,
    )
    return generate_world_composition_coverage(
        request,
        suite_id="qualification-reaction-electrochemistry-observation",
        discrete_axes=axes,
        continuous_axes=continuous,
        workflows=workflows,
        target=CompositionCoverageTarget(continuous_samples=6, seed=107),
    )


def _purification_suite() -> CompositionCoverageSuite:
    pattern_id = "reaction-phase-separation-observation"
    request = _base_request(
        pattern_id,
        ("reaction", "thermal", "phase", "separation", "observation"),
        budget=20,
        resources={
            "sample_volume_L": 0.001,
            "time_s": 3600.0,
            "instrument_uses": 3,
            "final_assays": 1,
        },
    )
    axes = (
        _axis(
            "reaction_family",
            "components.reaction.parameters.family",
            "declared-family-a",
            "declared-family-b",
        ),
        _axis(
            "phase_profile",
            "components.phase.parameters.phases",
            ["reactor_liquid", "aqueous", "organic"],
            ["aqueous", "organic", "solid"],
        ),
        _axis(
            "instrument_profile",
            (
                "components.observation.parameters.instruments",
                "task.instruments",
            ),
            ["hplc", "final_assay"],
            ["hplc", "gc", "final_assay"],
        ),
        _axis("objective", "task.objective", "balanced", "yield"),
    )
    continuous = (
        _continuous("reaction_temperature_K", 350.0, 390.0, "K"),
        _continuous("reaction_duration_s", 600.0, 1800.0, "s"),
        _continuous("phase_volume_L", 0.010, 0.020, "L"),
        _continuous("extractant_volume_L", 0.010, 0.025, "L"),
        _continuous("mix_duration_s", 60.0, 300.0, "s"),
        _continuous("settle_duration_s", 120.0, 600.0, "s"),
        _continuous("wash_volume_L", 0.003, 0.010, "L"),
        _continuous("concentrate_duration_s", 300.0, 900.0, "s"),
        _continuous("transfer_fraction", 0.65, 0.95, "dimensionless"),
    )
    downstream: list[dict[str, Any]] = [
        {
            "operation": "add_phase",
            "phase": "aqueous",
            "volume_L": _placeholder("phase_volume_L"),
        },
        {
            "operation": "add_extractant",
            "extractant": 1,
            "volume_L": _placeholder("extractant_volume_L"),
        },
        {
            "operation": "mix",
            "duration_s": _placeholder("mix_duration_s"),
            "stirring_speed_rpm": 650.0,
        },
        {"operation": "settle", "duration_s": _placeholder("settle_duration_s")},
        {"operation": "separate_phase", "target_phase": "organic"},
        {"operation": "wash", "wash_volume_L": _placeholder("wash_volume_L")},
        {"operation": "dry"},
        {
            "operation": "concentrate",
            "duration_s": _placeholder("concentrate_duration_s"),
        },
        {
            "operation": "transfer",
            "transfer_fraction": _placeholder("transfer_fraction"),
        },
    ]
    workflow_a = [
        *_reaction_charge(include_catalyst=True),
        _heat_action(),
        {"operation": "quench"},
        {"operation": "measure", "instrument": "hplc"},
        *downstream,
        {"operation": "measure", "instrument": "hplc"},
        *_close_actions(),
    ]
    workflow_b = [
        *_reaction_charge(),
        _heat_action(),
        {"operation": "quench"},
        *downstream,
        {"operation": "measure", "instrument": "hplc"},
        *_close_actions(),
    ]
    workflows = (
        OrderedWorkflowTemplate("measure-before-and-after", tuple(workflow_a)),
        OrderedWorkflowTemplate("measure-after-purification", tuple(workflow_b)),
    )
    _attach_process_time_policy(
        request,
        pattern_id=pattern_id,
        workflows=workflows,
        continuous_axes=continuous,
    )
    return generate_world_composition_coverage(
        request,
        suite_id="qualification-reaction-phase-separation-observation",
        discrete_axes=axes,
        continuous_axes=continuous,
        workflows=workflows,
        target=CompositionCoverageTarget(continuous_samples=6, seed=108),
    )


def build_generated_suites() -> tuple[CompositionCoverageSuite, ...]:
    suites = (
        _phase_observation_suite(),
        _reaction_thermal_suite(),
        _phase_separation_suite(),
        _crystallization_suite(),
        _distillation_suite(),
        _flow_suite(),
        _electrochemistry_suite(),
        _purification_suite(),
    )
    observed_counts = {
        str(suite.cases[0].compiled.compatibility.pattern): len(suite.cases)
        for suite in suites
    }
    if observed_counts != EXPECTED_PATTERN_CASE_COUNTS:
        raise RuntimeError(
            "frozen generated-composition denominators drifted: "
            f"expected {EXPECTED_PATTERN_CASE_COUNTS}, observed {observed_counts}"
        )
    return suites


__all__ = [
    "EXPECTED_GENERATED_CASE_COUNT",
    "EXPECTED_PATTERN_CASE_COUNTS",
    "QUALIFICATION_DESIGN_VERSION",
    "UNSEEN_PATTERN_ID",
    "build_generated_suites",
]
