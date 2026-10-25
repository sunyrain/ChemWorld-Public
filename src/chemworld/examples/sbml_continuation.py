"""Follow a bistable formal reaction network from an actual native-world state."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from chemworld.data.sbml import SBMLModel
from chemworld.data.scientific_inputs import InputSource
from chemworld.physchem.data_conflicts import DataSourceProvenance
from chemworld.physchem.sbml_system import SBMLObservable
from chemworld.runtime import ChemWorldRuntime
from chemworld.runtime.sbml_continuation import analyze_sbml_steady_branch_world
from chemworld.world.sbml_world import sbml_world


def fold_model(
    *,
    scaled: bool = False,
    coupled: bool = False,
    assignment_chain: bool = False,
    volume_L: float = 1.0,
    initial_x_mol: float = 0.1,
    initial_parameter: float = 0.221,
) -> SBMLModel:
    """Authored positive channels: x' = mu + 3x² - 2.5x - x³, optional y'=.7x-.4y.

    Amounts are mol and time is seconds in this description. The optional source
    document uses mmol/min. Held H catalyzes the feed; it is an original boundary
    pool and is not silently reset by hypothetical parameter changes.
    """
    amount, seconds = (0.001, 60.0) if scaled else (1.0, 1.0)
    exponent = -3 if scaled else 0

    def unit(name: str, powers: tuple[tuple[str, int, int, float], ...]) -> str:
        return (
            f'<unitDefinition id="{name}"><listOfUnits>'
            + "".join(
                f'<unit kind="{base}" exponent="{power}" scale="{scale}" multiplier="{factor}"/>'
                for base, power, scale, factor in powers
            )
            + "</listOfUnits></unitDefinition>"
        )

    units = "".join(
        (
            unit("amount", (("mole", 1, exponent, 1.0),)),
            unit("clock", (("second", 1, 0, seconds),)),
            unit("incoming", (("mole", 1, exponent, 1.0), ("second", -1, 0, seconds))),
            unit("first_order", (("second", -1, 0, seconds),)),
            unit("second_order", (("mole", -1, exponent, 1.0), ("second", -1, 0, seconds))),
            unit("third_order", (("mole", -2, exponent, 1.0), ("second", -1, 0, seconds))),
        )
    )
    amounts = {"X": initial_x_mol, "H": 2.0, **({"Y": 0.175} if coupled else {})}
    species = "".join(
        f'<species id="{name}" compartment="V" initialAmount="{value / amount}" '
        f'hasOnlySubstanceUnits="true" boundaryCondition="{str(name == "H").lower()}" '
        'constant="false"/>'
        for name, value in amounts.items()
    )
    params = (
        ("mu", initial_parameter * seconds / amount, "incoming"),
        ("a", 3 * seconds * amount, "second_order"),
        ("b", 2.5 * seconds, "first_order"),
        ("c", seconds * amount * amount, "third_order"),
        ("H_ref", 2 / amount, "amount"),
        ("k", 0.7 * seconds, "first_order"),
        ("gamma", 0.4 * seconds, "first_order"),
    )
    parameters = "".join(
        f'<parameter id="{name}" value="{value}" units="{units_name}" constant="true"/>'
        for name, value, units_name in params
    )
    rules = ""
    if assignment_chain:
        parameters += (
            '<parameter id="twice_mu" units="incoming" constant="false"/>'
            '<parameter id="effective_mu" units="incoming" constant="false"/>'
        )
        rules = (
            '<listOfRules><assignmentRule variable="twice_mu">'
            '<math xmlns="http://www.w3.org/1998/Math/MathML"><apply><times/>'
            "<cn>2</cn><ci>mu</ci></apply></math></assignmentRule>"
            '<assignmentRule variable="effective_mu">'
            '<math xmlns="http://www.w3.org/1998/Math/MathML"><apply><divide/>'
            "<ci>twice_mu</ci><cn>2</cn></apply></math></assignmentRule></listOfRules>"
        )
    control = "effective_mu" if assignment_chain else "mu"
    channels = [
        (
            "feed",
            {"H": 1},
            {"H": 1, "X": 1},
            f"<apply><divide/><apply><times/><ci>{control}</ci><ci>H</ci></apply>"
            "<ci>H_ref</ci></apply>",
        ),
        (
            "autocatalysis",
            {"X": 2},
            {"X": 3},
            "<apply><times/><ci>a</ci><ci>X</ci><ci>X</ci></apply>",
        ),
        ("loss", {"X": 1}, {}, "<apply><times/><ci>b</ci><ci>X</ci></apply>"),
        (
            "cubic_loss",
            {"X": 3},
            {"X": 2},
            "<apply><times/><ci>c</ci><ci>X</ci><ci>X</ci><ci>X</ci></apply>",
        ),
    ]
    if coupled:
        channels.extend(
            (
                (
                    "output",
                    {"X": 1},
                    {"X": 1, "Y": 1},
                    "<apply><times/><ci>k</ci><ci>X</ci></apply>",
                ),
                ("output_loss", {"Y": 1}, {}, "<apply><times/><ci>gamma</ci><ci>Y</ci></apply>"),
            )
        )

    def side(tag: str, items: dict[str, int]) -> str:
        if not items:
            return ""
        return (
            f"<listOf{tag}>"
            + "".join(
                f'<speciesReference species="{s}" stoichiometry="{v}" constant="true"/>'
                for s, v in items.items()
            )
            + f"</listOf{tag}>"
        )

    reactions = "".join(
        f'<reaction id="{name}" reversible="false">{side("Reactants", left)}'
        f'{side("Products", right)}<kineticLaw><math xmlns="http://www.w3.org/1998/Math/MathML">'
        f"{law}</math></kineticLaw></reaction>"
        for name, left, right, law in channels
    )
    xml = (
        '<sbml xmlns="http://www.sbml.org/sbml/level3/version2/core" level="3" version="2">'
        '<model id="fold_network" timeUnits="clock" substanceUnits="amount" extentUnits="amount" '
        f'volumeUnits="litre"><listOfUnitDefinitions>{units}</listOfUnitDefinitions>'
        f'<listOfCompartments><compartment id="V" size="{volume_L}" units="litre" '
        'spatialDimensions="3" constant="true"/></listOfCompartments>'
        f"<listOfSpecies>{species}</listOfSpecies><listOfParameters>{parameters}</listOfParameters>"
        f"{rules}<listOfReactions>{reactions}</listOfReactions></model></sbml>"
    )
    source = InputSource(
        DataSourceProvenance(
            "chemworld:sbml-continuation52",
            0,
            "Authored analytic bistable formal network; no empirical composition or energy",
            "1",
            "CC0-1.0",
        ),
        "synthetic",
    )
    return SBMLModel(xml, source)


def branch_options(*, scaled: bool = False, coupled: bool = False) -> dict[str, Any]:
    multiplier = 60000.0 if scaled else 1.0
    return {
        "parameter": "mu",
        "amount_scales_mol": {"X": 1.0, **({"Y": 2.0} if coupled else {})},
        "parameter_scale": multiplier,
        "parameter_bounds": [0.05 * multiplier, 5 * multiplier],
        "maximum_points": 80,
        "step_size": 0.05,
    }


def example_report() -> dict[str, Any]:
    source = fold_model(coupled=True, assignment_chain=True)
    observables = (SBMLObservable("amount", "X"), SBMLObservable("concentration", "Y"))
    composition = sbml_world(source, controlled_parameters=("mu", "k"), observables=observables)
    runtime, restored = ChemWorldRuntime(composition=composition), None
    try:
        runtime.reset(seed=52)
        transactions = []
        for action in (
            {"operation": "set_model_parameters", "values": {"mu": 0.23, "k": 0.8}},
            {"operation": "wait", "duration_s": 0.1},
        ):
            result = runtime.execute(action)
            transactions.append(result.info["transaction_status"])
        before = runtime._state.to_dict()
        options = branch_options(coupled=True)
        report = analyze_sbml_steady_branch_world(runtime._state, composition, **options)
        snapshot = json.loads(json.dumps(runtime.snapshot().to_dict()))
        restored = ChemWorldRuntime.from_snapshot(snapshot)
        replay = analyze_sbml_steady_branch_world(restored._state, composition, **options)
        return {
            "analysis": report,
            "transactions": transactions,
            "world_unchanged": before == runtime._state.to_dict(),
            "json_replay_exact": report == replay,
            "snapshot": snapshot,
            "options": options,
        }
    finally:
        runtime.close()
        if restored:
            restored.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = example_report()
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(
        json.dumps(
            {
                "status": result["analysis"]["status"],
                "points": result["analysis"]["completed_points"],
                "fold_brackets": len(result["analysis"]["fold_brackets"]),
                "world_unchanged": result["world_unchanged"],
                "json_replay_exact": result["json_replay_exact"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
