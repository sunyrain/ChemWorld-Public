"""Known-sigma profiling of an authored affine SBML rate observed publicly."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from chemworld.data.calibration import CalibrationReadout, NativeCalibrationExperiment
from chemworld.data.sbml import SBMLModel
from chemworld.data.sbml_calibration import SBMLCalibration, SBMLCalibrationParameter
from chemworld.data.sbml_profile import GAUSSIAN_KNOWN_SIGMA, sbml_profile_likelihood
from chemworld.data.scientific_inputs import (
    InputValue,
    QuantityColumn,
    ScientificRow,
    ScientificTable,
)
from chemworld.examples.sbml_calibration import source
from chemworld.physchem.sbml_system import SBMLObservable
from chemworld.world.sbml_world import sbml_world

TIMES = (0.0, 0.3, 0.8, 1.4, 2.2)
OBSERVATIONS = (1.01, 1.23, 1.52, 1.94, 2.47)
SIGMAS = (0.04, 0.06, 0.05, 0.08, 0.05)
A_GRID = (0.7, 0.85, 1.0, 1.15, 1.3)
B_GRID = (0.4, 0.55, 0.7, 0.85, 1.0)


def affine_inputs(
    *, scaled: bool = False, coordinate: str = "linear", b_upper: float = 2.0
) -> tuple[Any, Any, Any]:
    """Fixed synthetic data; composition and energy remain explicitly unspecified."""
    scale, seconds = (-3, 60.0) if scaled else (0, 1.0)
    amount = 10.0**scale

    def unit(name: str, components: str) -> str:
        return (
            f'<unitDefinition id="{name}"><listOfUnits>{components}</listOfUnits></unitDefinition>'
        )

    mole = f'<unit kind="mole" exponent="1" scale="{scale}" multiplier="1"/>'
    inverse_time = f'<unit kind="second" exponent="-1" scale="0" multiplier="{seconds}"/>'
    units = unit("amount", mole) + unit("rate", inverse_time) + unit("flux", mole + inverse_time)
    units += unit("clock", f'<unit kind="second" exponent="1" scale="0" multiplier="{seconds}"/>')
    species = "".join(
        f'<species id="{name}" compartment="V" initialAmount="0" hasOnlySubstanceUnits="true" '
        'boundaryCondition="false" constant="false"/>'
        for name in ("X", "Y")
    )
    parameters_xml = "".join(
        f'<parameter id="{name}" value="{value}" units="{units_id}" constant="true"/>'
        for name, value, units_id in (
            ("c", seconds / amount, "flux"),
            ("a", 0.3 * seconds / amount, "flux"),
            ("b", 0.2 * seconds, "rate"),
        )
    )

    def reaction(name: str, target: str, expression: str) -> str:
        return (
            f'<reaction id="{name}" reversible="false"><listOfProducts>'
            f'<speciesReference species="{target}" stoichiometry="1" constant="true"/>'
            '</listOfProducts><kineticLaw><math xmlns="http://www.w3.org/1998/Math/MathML">'
            f"{expression}</math></kineticLaw></reaction>"
        )

    reactions = reaction("growth", "X", "<ci>c</ci>") + reaction(
        "signal", "Y", "<apply><plus/><ci>a</ci><apply><times/><ci>b</ci><ci>X</ci></apply></apply>"
    )
    xml = (
        '<sbml xmlns="http://www.sbml.org/sbml/level3/version2/core" level="3" version="2">'
        '<model id="affine" timeUnits="clock" substanceUnits="amount" extentUnits="amount" '
        f'volumeUnits="litre"><listOfUnitDefinitions>{units}</listOfUnitDefinitions>'
        '<listOfCompartments><compartment id="V" spatialDimensions="3" size="1" '
        'units="litre" constant="true"/></listOfCompartments>'
        f"<listOfSpecies>{species}</listOfSpecies><listOfParameters>{parameters_xml}</listOfParameters>"
        f"<listOfReactions>{reactions}</listOfReactions></model></sbml>"
    )
    model = SBMLModel(xml, source())
    observable = SBMLObservable("reaction_rate", "signal")
    world = sbml_world(
        model,
        observables=(observable,),
        controlled_parameters=("a", "b"),
        readout_relative_sd=0.0,
        rtol=2e-10,
        atol=1e-12,
        budget=100,
    )
    parameters = tuple(
        SBMLCalibrationParameter(
            QuantityColumn(name, unit_name, "parameter", (0.01, upper), "authored affine control"),
            name,
            initial,
            coordinate,
        )
        for name, unit_name, upper, initial in (("a", "mol/s", 3.0, 0.3), ("b", "Hz", b_upper, 0.2))
    )
    columns = (
        QuantityColumn("time", "s", "condition", (0, 10), "fixed sampling schedule"),
        QuantityColumn(observable.channel_id, "mol/s", "observation", (-1000, 1000), "public rate"),
    )
    rows, experiments = [], []
    for split in ("train", "validation"):
        actions: list[dict[str, Any]] = [
            {"operation": "set_model_parameters", "values": {"a": 0.3, "b": 0.2}}
        ]
        readouts, previous = [], 0.0
        for index, (time_s, observed, sigma) in enumerate(
            zip(TIMES, OBSERVATIONS, SIGMAS, strict=True)
        ):
            if time_s > previous:
                actions.append({"operation": "wait", "duration_s": time_s - previous})
            actions.append({"operation": "measure", "instrument": "formal_quantity_meter"})
            row_id = f"{split}-{index}"
            cells = {
                "time": InputValue(time_s, "s", 0.0, "fixed schedule"),
                observable.channel_id: InputValue(
                    observed, "mol/s", sigma, "known synthetic sigma"
                ),
            }
            rows.append(
                ScientificRow(row_id, source().provenance.source_id, split, row_id, split, cells)
            )
            readouts.append(
                CalibrationReadout(
                    row_id, observable.channel_id, len(actions) - 1, observable.channel_id
                )
            )
            previous = time_s
        experiments.append(
            NativeCalibrationExperiment(split, world, tuple(actions), tuple(readouts))
        )
    return (
        ScientificTable("affine-profile52", "measurements", columns, (source(),), tuple(rows)),
        parameters,
        tuple(experiments),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("use a fresh output filename")
    result = sbml_profile_likelihood(
        SBMLCalibration(*affine_inputs()),
        "a",
        A_GRID,
        noise_model=GAUSSIAN_KNOWN_SIGMA,
        progress=lambda record: print(json.dumps(record), flush=True),
    )
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(
        json.dumps(
            {key: result[key] for key in ("status", "counts", "crossing_brackets", "resources")}
        )
    )


if __name__ == "__main__":
    main()
