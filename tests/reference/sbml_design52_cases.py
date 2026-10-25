"""Fixed authored source models and optimization declarations for parent52."""

from dataclasses import replace

from chemworld.data.sbml import SBMLModel
from chemworld.examples.sbml_design import design_model, design_spec, design_world
from chemworld.physchem.bounded_design import DesignConstraint
from chemworld.physchem.sbml_system import SBMLSystemSpec

OPTIMIZATION_CASES = (
    "steady_max",
    "steady_min",
    "steady_budget",
    "trajectory_max",
    "trajectory_min",
    "scaled_steady_budget",
    "scaled_trajectory_min",
    "log_trajectory_min",
)
BOUNDARY_CASES = ("infeasible", "evaluation_budget", "physical_failure", "steady_seed_failure")


def inputs(name):
    scaled = name.startswith("scaled_")
    coordinate_log = name.startswith("log_")
    base = name.removeprefix("scaled_").removeprefix("log_")
    if name in BOUNDARY_CASES:
        base = "steady_max"
    composition = design_world(scaled=scaled)
    model = composition.physics.models_of("sbml")["formal_workspace"]
    specification = design_spec(base, log_coordinate=coordinate_log)
    if name == "infeasible":
        specification = replace(
            specification, constraints=(DesignConstraint("product", "mol", 1.0, 1e-8, lower=1.1),)
        )
    elif name == "evaluation_budget":
        specification = replace(
            specification, policy=replace(specification.policy, maximum_evaluations=1)
        )
    elif name == "physical_failure":
        original = design_model()
        xml = (
            original.to_xml()
            .replace(
                "</listOfParameters>",
                '<parameter id="k_limit" value="0.4" units="rate" '
                'constant="true"/></listOfParameters>',
            )
            .replace(
                "</model>",
                "<listOfConstraints><constraint><math "
                'xmlns="http://www.w3.org/1998/Math/MathML"><apply><lt/><ci>k1</ci>'
                "<ci>k_limit</ci></apply></math></constraint></listOfConstraints></model>",
            )
        )
        model = replace(model, model=SBMLModel(xml, original.source))
    elif name == "steady_seed_failure":
        # An autonomous open birth process has no steady root. The source rate
        # is positive; the optimizer must keep its failed original root solve.
        original = design_model()
        # Keep amount dimensions through a fixed boundary pool A, while B is
        # produced at k1*A and never removed. There is no finite B steady state.
        xml = original.to_xml().replace(
            'hasOnlySubstanceUnits="true" boundaryCondition="false" constant="false"',
            'hasOnlySubstanceUnits="true" boundaryCondition="true" constant="false"',
            1,
        )
        start = xml.index('<reaction id="backward"')
        end = xml.index("</reaction>", start) + len("</reaction>")
        xml = xml[:start] + xml[end:]
        observables = tuple(o for o in model.observables if o.source_id != "backward")
        model = replace(model, model=SBMLModel(xml, original.source), observables=observables)
    assert isinstance(model, SBMLSystemSpec)
    return model, dict(model.frame.initial_values), specification
