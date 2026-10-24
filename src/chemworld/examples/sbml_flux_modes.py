"""Read original SBML flux modes before and after a native conversion control."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from chemworld.data.sbml import SBMLModel
from chemworld.data.scientific_inputs import InputSource
from chemworld.physchem.data_conflicts import DataSourceProvenance
from chemworld.runtime import ChemWorldRuntime
from chemworld.runtime.sbml_flux_modes import analyze_sbml_flux_modes
from chemworld.world.sbml_world import sbml_world


def flux_model(
    matrix: Any = ((1, 1, -1),),
    *,
    irreversible: Any = (True, True, True),
    external_matrix: Any = (),
    controlled_row: int | None = 0,
    scaled: bool = False,
    zero_rates: bool = False,
) -> SBMLModel:
    """Small original-source formal network; no invented molecular compositions."""
    names = tuple(f"I{i}" for i in range(len(matrix)))
    external = tuple(f"E{i}" for i in range(len(external_matrix)))
    rows = (*matrix, *external_matrix)
    species = "".join(
        f'<species id="{name}" compartment="V" initialAmount="2" '
        f'hasOnlySubstanceUnits="true" boundaryCondition="{str(name in external).lower()}" '
        f'constant="{str(name in external).lower()}"'
        + (' substanceUnits="small_amount"' if scaled and name == "I0" else "")
        + (
            ' conversionFactor="cf"'
            if controlled_row is not None and name == f"I{controlled_row}"
            else ""
        )
        + "/>"
        for name in (*names, *external)
    )
    parameters = '<parameter id="cf" value="1" units="dimensionless" constant="true"/>'
    reactions = []
    for column, flag in enumerate(irreversible):
        parameters += (
            f'<parameter id="q{column}" value="{0 if zero_rates else 0.1}" '
            'units="extent_rate" constant="true"/>'
        )
        sides = []
        for tag, direction in (("Reactants", -1), ("Products", 1)):
            references = "".join(
                f'<speciesReference species="{name}" stoichiometry="{abs(float(row[column]))}" '
                'constant="true"/>'
                for name, row in zip((*names, *external), rows, strict=True)
                if direction * row[column] > 0
            )
            if references:
                sides.append(f"<listOf{tag}>{references}</listOf{tag}>")
        reactions.append(
            f'<reaction id="r{column}" reversible="{str(not flag).lower()}">'
            + "".join(sides)
            + '<kineticLaw><math xmlns="http://www.w3.org/1998/Math/MathML">'
            f"<ci>q{column}</ci></math></kineticLaw></reaction>"
        )
    units = (
        '<unitDefinition id="extent_rate"><listOfUnits>'
        '<unit kind="mole" exponent="1" scale="0" multiplier="1"/>'
        '<unit kind="second" exponent="-1" scale="0" multiplier="1"/>'
        "</listOfUnits></unitDefinition>"
        '<unitDefinition id="small_amount"><listOfUnits>'
        '<unit kind="mole" exponent="1" scale="-3" multiplier="1"/>'
        "</listOfUnits></unitDefinition>"
    )
    text = (
        '<sbml xmlns="http://www.sbml.org/sbml/level3/version2/core" level="3" version="2">'
        '<model id="flux_modes" timeUnits="second" substanceUnits="mole" '
        'extentUnits="mole" volumeUnits="litre">'
        f"<listOfUnitDefinitions>{units}</listOfUnitDefinitions>"
        '<listOfCompartments><compartment id="V" spatialDimensions="3" size="1" '
        'units="litre" constant="true"/></listOfCompartments>'
        f"<listOfSpecies>{species}</listOfSpecies><listOfParameters>{parameters}</listOfParameters>"
        "<listOfReactions>" + "".join(reactions) + "</listOfReactions></model></sbml>"
    )
    source = InputSource(
        DataSourceProvenance(
            "chemworld:elementary-flux-modes52",
            0,
            "Authored formal reaction-balance example",
            "1",
            "CC0-1.0",
        ),
        "synthetic",
    )
    return SBMLModel(text, source)


def run_example() -> dict[str, Any]:
    composition = sbml_world(flux_model(), controlled_parameters=("cf",), budget=100)
    model = composition.physics.models_of("sbml")["formal_workspace"]
    runtime = ChemWorldRuntime(composition=composition)
    try:
        runtime.reset(seed=0)
        wait = runtime.execute({"operation": "wait", "duration_s": 1.0})
        before_snapshot = runtime.snapshot().to_dict()
        before = analyze_sbml_flux_modes(runtime._state, model)
        if runtime.snapshot().to_dict() != before_snapshot:
            raise ValueError("flux-mode analysis changed the original runtime")
        control = runtime.execute({"operation": "set_model_parameters", "values": {"cf": 0.0}})
        after_snapshot = runtime.snapshot().to_dict()
        after = analyze_sbml_flux_modes(runtime._state, model)
        if runtime.snapshot().to_dict() != after_snapshot:
            raise ValueError("flux-mode analysis changed the controlled runtime")
        return {
            "source_xml": model.model.to_xml(),
            "before": before,
            "after": after,
            "operations": [wait.info, control.info],
            "before_snapshot": before_snapshot,
            "after_snapshot": after_snapshot,
            "snapshots_unchanged_by_analysis": True,
        }
    finally:
        runtime.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("use a fresh output file")
    report = run_example()
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(
        json.dumps(
            {"before": len(report["before"]["modes"]), "after": len(report["after"]["modes"])}
        )
    )


if __name__ == "__main__":
    main()
