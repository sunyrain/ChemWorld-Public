"""Fixed 52F coverage and original source declarations, not reference calculations."""

from chemworld.data.sbml import SBMLModel
from chemworld.examples.sbml_calibration import pool_model
from chemworld.examples.sbml_continuation import branch_options, fold_model
from chemworld.physchem.residual_system import RootPolicy
from chemworld.physchem.sbml_system import SBMLObservable, SBMLSystemSpec

CASES = (
    {"name": "scalar"},
    {"name": "scalar-source-units", "scaled": True},
    {"name": "scalar-chain-volume", "assignment_chain": True, "volume_L": 2.3},
    {"name": "scalar-metric", "metric": True},
    {"name": "coupled", "coupled": True},
    {"name": "coupled-source-units", "coupled": True, "scaled": True},
    {"name": "coupled-chain-volume", "coupled": True, "assignment_chain": True, "volume_L": 3.7},
    {"name": "closed-current-total", "closed": True},
    {"name": "closed-source-units", "closed": True, "scaled": True},
    {"name": "closed-metric-held-control", "closed": True, "metric": True},
    {"name": "reverse-two-folds", "reverse": True},
    {"name": "current-boundary-pool", "held": True},
    {"name": "parameter-domain-stop", "failure": "parameter"},
    {"name": "evaluation-budget-stop", "failure": "evaluation"},
    {"name": "seed-iteration-stop", "failure": "seed"},
    {"name": "branch-point-stop", "failure": "rank", "closed": True},
    {"name": "world-current-control", "world": True},
    {
        "name": "world-coupled-source-chain",
        "world": True,
        "coupled": True,
        "scaled": True,
        "assignment_chain": True,
        "volume_L": 2.3,
    },
)


def build(case):
    scaled, coupled, closed = (case.get(k, False) for k in ("scaled", "coupled", "closed"))
    if closed:
        source = pool_model(kind="reversible", initial_b=0.2, scaled=scaled)
        seconds = 60 if scaled else 1
        options = {
            "parameter": "k1",
            "amount_scales_mol": {"A": 1, "B": 1},
            "parameter_scale": seconds,
            "parameter_bounds": [0.05 * seconds, 5 * seconds],
            "maximum_points": 20,
        }
        if case.get("failure") == "rank":
            source = pool_model(kind="reversible", initial_a=0.5, initial_b=0.5)
            source = SBMLModel(source.raw_text.replace("<ci>k2</ci>", "<ci>k1</ci>"), source.source)
            options["parameter_bounds"] = [0, 5]
        observations = (
            SBMLObservable("amount", "A"),
            SBMLObservable("concentration", "B"),
            SBMLObservable("reaction_rate", "forward"),
        )
        # Source pool reaction identifiers are deliberately resolved from the source.
        observations = (
            *observations[:2],
            SBMLObservable(
                "reaction_rate", source.declaration.items("listOfReactions")[0].get("id")
            ),
        )
        spec = SBMLSystemSpec(source, observables=observations, controlled_parameters=("k1", "k2"))
        current = dict(spec.frame.initial_values)
        current["N:A"], current["N:B"] = 0.9, 0.4
        if case.get("metric"):
            options["amount_scales_mol"] = {"A": 0.7, "B": 1.4}
            current["P:k2"] = 0.6 * seconds
        if case.get("failure") == "rank":
            current.update({"N:A": 0.5, "N:B": 0.5, "P:k1": 0})
    else:
        options = branch_options(scaled=scaled, coupled=coupled)
        source_options = {
            k: v
            for k, v in case.items()
            if k in {"scaled", "coupled", "assignment_chain", "volume_L"}
        }
        if case.get("reverse"):
            source_options.update(initial_x_mol=2.2, initial_parameter=1.628)
            options.update(initial_direction=-1, maximum_points=60)
        if case.get("failure") == "seed":
            source_options["initial_x_mol"] = 4.0
            options["root_policy"] = RootPolicy(maximum_iterations=1)
        source = fold_model(**source_options)
        observations = (
            SBMLObservable("amount", "X"),
            SBMLObservable("concentration", "X"),
            SBMLObservable("reaction_rate", "feed"),
        )
        spec = SBMLSystemSpec(source, observables=observations, controlled_parameters=("mu", "k"))
        current = dict(spec.frame.initial_values)
        if case.get("metric"):
            options.update(amount_scales_mol={"X": 0.7}, parameter_scale=2.0)
        if case.get("held"):
            current["N:H"] = 1.5
        if case.get("failure") == "parameter":
            options.update(
                parameter_bounds=[0.05, 0.25],
                maximum_points=20,
                maximum_retries=3,
                minimum_step=0.00625,
            )
        if case.get("failure") == "evaluation":
            options["maximum_evaluations"] = 2
    return spec, current, options
