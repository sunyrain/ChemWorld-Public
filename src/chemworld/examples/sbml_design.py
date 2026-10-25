"""Optimize declared SBML controls, then use normal set/wait/measure actions."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from chemworld.data.sbml import SBMLModel
from chemworld.data.scientific_inputs import InputSource
from chemworld.examples.sbml_calibration import pool_model
from chemworld.physchem.bounded_design import (
    BoundedDesignPolicy,
    DesignConstraint,
    DesignVariable,
)
from chemworld.physchem.data_conflicts import DataSourceProvenance
from chemworld.physchem.sbml_design import (
    SBMLDesignControl,
    SBMLDesignMetric,
    SBMLDesignSpec,
)
from chemworld.physchem.sbml_system import SBMLObservable
from chemworld.runtime import ChemWorldRuntime
from chemworld.runtime.sbml_design import analyze_sbml_design_world
from chemworld.world.sbml_world import sbml_world


def design_model(*, scaled: bool = False, initial_assignment: bool = False) -> SBMLModel:
    """A reversible finite formal pool and a source-defined combined control cost."""
    original = pool_model(
        kind="reversible",
        initial_a=0.8,
        initial_b=0.2,
        scaled=scaled,
        initial_assignment=initial_assignment,
    )
    xml = original.to_xml().replace(
        "</listOfParameters>",
        '<parameter id="total_rate" value="0" units="rate" constant="false"/>'
        '</listOfParameters><listOfRules><assignmentRule variable="total_rate">'
        '<math xmlns="http://www.w3.org/1998/Math/MathML"><apply><plus/>'
        "<ci>k1</ci><ci>k2</ci></apply></math></assignmentRule></listOfRules>",
    )
    return SBMLModel(
        xml,
        InputSource(
            DataSourceProvenance(
                "chemworld:sbml-design52",
                0,
                "Authored analytic reversible pools and objective controls",
                "1",
                "CC0-1.0",
            ),
            "synthetic",
        ),
    )


def design_spec(name: str = "steady_max", *, log_coordinate: bool = False) -> SBMLDesignSpec:
    """Physical k1/k2 coordinates are s^-1 regardless of source SBML time units."""
    names = {"steady_max", "steady_min", "trajectory_max", "trajectory_min", "steady_budget"}
    if name not in names:
        raise ValueError("unknown fixed SBML design example")
    trajectory = name.startswith("trajectory")
    mode = "trajectory" if trajectory else "steady"
    offset = 2.0 if trajectory else 0.0
    controls = [
        SBMLDesignControl(
            DesignVariable(
                "forward_rate", "Hz", (0.05, 1.0), 0.3, "log" if log_coordinate else "linear"
            ),
            "k1",
        )
    ]
    starts: tuple[dict[str, float], ...] = ({"forward_rate": 0.3}, {"forward_rate": 0.75})
    constraints = []
    if name.endswith("min"):
        constraints.append(
            DesignConstraint("product", "mol", 1.0, 1e-8, lower=0.6 if trajectory else 0.75)
        )
    elif name == "steady_budget":
        controls.append(
            SBMLDesignControl(DesignVariable("reverse_rate", "Hz", (0.1, 0.6), 0.2), "k2")
        )
        starts = (
            {"forward_rate": 0.3, "reverse_rate": 0.2},
            {"forward_rate": 0.55, "reverse_rate": 0.15},
        )
        constraints.append(DesignConstraint("total_control", "Hz", 1.0, 1e-8, upper=0.8))
    else:
        constraints.append(DesignConstraint("control", "Hz", 1.0, 1e-8, upper=0.65))
    metrics = (
        SBMLDesignMetric("product", "amount_B", offset),
        SBMLDesignMetric("reactant", "amount_A", offset),
        SBMLDesignMetric("control", "parameter_k1", 0.0),
        SBMLDesignMetric("total_control", "parameter_total_rate", 0.0),
        SBMLDesignMetric("forward_flux", "reaction_rate_forward", offset),
        SBMLDesignMetric("product_concentration", "concentration_B", offset),
        SBMLDesignMetric("volume", "volume_V", 0.0),
    )
    return SBMLDesignSpec(
        tuple(controls),
        metrics,
        "control" if name.endswith("min") else "product",
        maximize=not name.endswith("min"),
        mode=mode,
        constraints=tuple(constraints),
        starts=starts,
        policy=BoundedDesignPolicy(
            maximum_evaluations=500,
            maximum_iterations=80,
            objective_scale=1.0,
            tolerance=1e-10,
            relative_difference_step=1e-5,
        ),
    )


def design_world(*, scaled: bool = False, initial_assignment: bool = False) -> Any:
    return sbml_world(
        design_model(scaled=scaled, initial_assignment=initial_assignment),
        observables=(
            SBMLObservable("amount", "A"),
            SBMLObservable("amount", "B"),
            SBMLObservable("concentration", "B"),
            SBMLObservable("volume", "V"),
            SBMLObservable("parameter", "k1", "Hz"),
            SBMLObservable("parameter", "total_rate", "Hz"),
            SBMLObservable("reaction_rate", "forward"),
        ),
        controlled_parameters=("k1", "k2"),
        readout_relative_sd=0.0,
        budget=100,
        rtol=1e-10,
        atol=1e-13,
    )


def example_report() -> dict[str, Any]:
    composition = design_world()
    runtime = ChemWorldRuntime(composition=composition)
    try:
        runtime.reset(seed=52)
        before = runtime.snapshot().to_dict()
        specification = replace(design_spec("trajectory_min"), starts=({"forward_rate": 0.3},))
        report = analyze_sbml_design_world(runtime._state, composition, specification)
        unchanged = before == runtime.snapshot().to_dict()
        replies = []
        if report["recommended_source_parameters"] is not None:
            for action in (
                {
                    "operation": "set_model_parameters",
                    "values": report["recommended_source_parameters"],
                },
                {"operation": "wait", "duration_s": 2.0},
                {"operation": "measure", "instrument": "formal_quantity_meter"},
            ):
                outcome = runtime.execute(action)
                replies.append(
                    {"action": action, "transaction_status": outcome.info["transaction_status"]}
                )
        return {
            "analysis": report,
            "world_unchanged_by_analysis": unchanged,
            "actual_actions": replies,
            "final_snapshot": runtime.snapshot().to_dict(),
        }
    finally:
        runtime.close()
