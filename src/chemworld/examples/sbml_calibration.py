"""Fit declared formal-model controls from public native time-series histories."""

from __future__ import annotations

import argparse
import json
from math import exp
from pathlib import Path
from typing import Any

from chemworld.data.calibration import CalibrationReadout, NativeCalibrationExperiment
from chemworld.data.sbml import SBMLModel
from chemworld.data.sbml_calibration import SBMLCalibration, SBMLCalibrationParameter
from chemworld.data.scientific_inputs import (
    InputSource,
    InputValue,
    QuantityColumn,
    ScientificRow,
    ScientificTable,
)
from chemworld.physchem.data_conflicts import DataSourceProvenance
from chemworld.physchem.sbml_system import SBMLObservable
from chemworld.world.sbml_world import sbml_world

TIMES = (0.0, 0.2, 0.6, 1.2, 2.5, 5.0)


def source() -> InputSource:
    return InputSource(
        DataSourceProvenance(
            "chemworld:sbml-calibration52",
            0,
            "Authored analytic formal pools; synthetic data, undeclared chemical composition",
            "1",
            "CC0-1.0",
        ),
        "synthetic",
    )


def pool_model(
    *,
    kind: str = "decay",
    initial_a: float = 0.8,
    initial_b: float = 0.0,
    volume_L: float = 2.0,
    scaled: bool = False,
    initial_assignment: bool = False,
) -> SBMLModel:
    """Original SBML with explicit source units; all rate constants are controls."""
    if kind not in {"decay", "reversible", "confounded"}:
        raise ValueError("unknown authored pool model")
    amount, volume, seconds = (0.001, 0.001, 60.0) if scaled else (1.0, 1.0, 1.0)
    scale = -3 if scaled else 0
    units = (
        f'<unitDefinition id="amount"><listOfUnits><unit kind="mole" exponent="1" '
        f'scale="{scale}" multiplier="1"/></listOfUnits></unitDefinition>'
        f'<unitDefinition id="volume"><listOfUnits><unit kind="litre" exponent="1" '
        f'scale="{scale}" multiplier="1"/></listOfUnits></unitDefinition>'
        f'<unitDefinition id="clock"><listOfUnits><unit kind="second" exponent="1" '
        f'scale="0" multiplier="{seconds}"/></listOfUnits></unitDefinition>'
        f'<unitDefinition id="rate"><listOfUnits><unit kind="second" exponent="-1" '
        f'scale="0" multiplier="{seconds}"/></listOfUnits></unitDefinition>'
    )
    if initial_assignment:
        units += (
            '<unitDefinition id="amount_time"><listOfUnits>'
            f'<unit kind="mole" exponent="1" scale="{scale}" multiplier="1"/>'
            f'<unit kind="second" exponent="1" scale="0" multiplier="{seconds}"/>'
            "</listOfUnits></unitDefinition>"
        )
    parameters = (
        f'<parameter id="k1" value="{0.3 * seconds}" units="rate" constant="true"/>'
        f'<parameter id="k2" value="{0.2 * seconds}" units="rate" constant="true"/>'
    )
    if initial_assignment:
        parameters += (
            f'<parameter id="initial_factor" value="{initial_a / amount / (0.3 * seconds)}" '
            'units="amount_time" constant="true"/>'
        )
    initial = (
        '<listOfInitialAssignments><initialAssignment symbol="A">'
        '<math xmlns="http://www.w3.org/1998/Math/MathML"><apply><times/>'
        "<ci>initial_factor</ci><ci>k1</ci></apply></math>"
        "</initialAssignment></listOfInitialAssignments>"
        if initial_assignment
        else ""
    )
    # Amount-only species make the independent closed forms transparent. Native
    # concentration readouts still use the explicitly declared compartment volume.
    species = "".join(
        f'<species id="{name}" compartment="V" initialAmount="{value / amount}" '
        'hasOnlySubstanceUnits="true" boundaryCondition="false" constant="false"/>'
        for name, value in (("A", initial_a), ("B", initial_b))
    )

    def reaction(name: str, left: str, right: str, rate: str) -> str:
        return (
            f'<reaction id="{name}" reversible="false"><listOfReactants>'
            f'<speciesReference species="{left}" stoichiometry="1" constant="true"/>'
            "</listOfReactants><listOfProducts>"
            f'<speciesReference species="{right}" stoichiometry="1" constant="true"/>'
            '</listOfProducts><kineticLaw><math xmlns="http://www.w3.org/1998/Math/MathML">'
            f"<apply><times/>{rate}<ci>{left}</ci></apply></math></kineticLaw></reaction>"
        )

    kinetics = reaction(
        "forward",
        "A",
        "B",
        ("<apply><plus/><ci>k1</ci><ci>k2</ci></apply>" if kind == "confounded" else "<ci>k1</ci>"),
    )
    if kind == "reversible":
        kinetics += reaction("backward", "B", "A", "<ci>k2</ci>")
    xml = (
        '<sbml xmlns="http://www.sbml.org/sbml/level3/version2/core" level="3" version="2">'
        f'<model id="{kind}" timeUnits="clock" substanceUnits="amount" extentUnits="amount" '
        f'volumeUnits="volume"><listOfUnitDefinitions>{units}</listOfUnitDefinitions>'
        '<listOfCompartments><compartment id="V" spatialDimensions="3" '
        f'size="{volume_L / volume}" units="volume" constant="true"/></listOfCompartments>'
        f"<listOfSpecies>{species}</listOfSpecies><listOfParameters>{parameters}</listOfParameters>"
        f"{initial}<listOfReactions>{kinetics}</listOfReactions></model></sbml>"
    )
    return SBMLModel(xml, source())


def fixed_inputs(
    *,
    kind: str = "decay",
    scaled: bool = False,
    boundary: bool = False,
    initial_assignment: bool = False,
) -> tuple[Any, Any, Any]:
    """Noiseless closed-form data with separately declared observation uncertainty."""
    truth = {"k1": 0.55, "k2": 0.23}
    names = ("k1",) if kind == "decay" else ("k1", "k2")
    parameters = tuple(
        SBMLCalibrationParameter(
            QuantityColumn(
                name,
                "Hz",
                "parameter",
                (0.04, 0.4 if boundary else 1.5),
                "constant rate control after source initialization",
            ),
            name,
            0.3 if name == "k1" else 0.2,
        )
        for name in names
    )
    observables = (
        (SBMLObservable("concentration", "A"), SBMLObservable("reaction_rate", "forward"))
        if kind == "reversible"
        else (SBMLObservable("amount", "A"),)
    )
    columns = (
        QuantityColumn("time", "s", "condition", (0, 10), "elapsed native laboratory time"),
        *(
            QuantityColumn(
                o.channel_id, o.unit, "observation", (-10, 10), "declared formal quantity"
            )
            for o in observables
        ),
    )
    rows, experiments = [], []
    for group_index, initial_a in enumerate((0.8, 0.45, 1.1)):
        initial_b = 0.2 if kind == "reversible" else 0.0
        model = pool_model(
            kind=kind,
            initial_a=initial_a,
            initial_b=initial_b,
            scaled=scaled,
            initial_assignment=initial_assignment,
        )
        world = sbml_world(
            model,
            observables=observables,
            controlled_parameters=names,
            readout_relative_sd=0.0,
            rtol=2e-10,
            atol=1e-12,
            budget=100,
        )
        group = f"pools-{group_index}"
        split = "train" if group_index < 2 else "validation"
        actions: list[dict[str, Any]] = [
            {"operation": "set_model_parameters", "values": dict.fromkeys(names, 0.3)}
        ]
        readouts, previous = [], 0.0
        for index, time_s in enumerate(TIMES):
            if time_s > previous:
                actions.append({"operation": "wait", "duration_s": time_s - previous})
            actions.append({"operation": "measure", "instrument": "formal_quantity_meter"})
            row_id = f"{group}-{index}"
            k1, k2 = truth["k1"], truth["k2"]
            if kind == "reversible":
                equilibrium = (initial_a + initial_b) * k2 / (k1 + k2)
                value = equilibrium + (initial_a - equilibrium) * exp(-(k1 + k2) * time_s)
            else:
                value = initial_a * exp(-(k1 + (k2 if kind == "confounded" else 0)) * time_s)
            quantities = {
                "amount_A": value,
                "concentration_A": value / 2.0,
                "reaction_rate_forward": k1 * value,
            }
            cells = {"time": InputValue(time_s, "s", 0.0, "Exact fixed acquisition schedule")}
            for observable in observables:
                cells[observable.channel_id] = InputValue(
                    quantities[observable.channel_id],
                    observable.unit,
                    0.002,
                    "Explicit synthetic standard uncertainty; no empirical measurement claim",
                )
                readouts.append(
                    CalibrationReadout(
                        row_id, observable.channel_id, len(actions) - 1, observable.channel_id
                    )
                )
            rows.append(
                ScientificRow(row_id, source().provenance.source_id, group, row_id, split, cells)
            )
            previous = time_s
        experiments.append(
            NativeCalibrationExperiment(group, world, tuple(actions), tuple(readouts))
        )
    table = ScientificTable(
        "sbml-control-calibration52-" + kind, "measurements", columns, (source(),), tuple(rows)
    )
    return table, parameters, tuple(experiments)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("use a fresh output filename")
    table, parameters, experiments = fixed_inputs(kind="reversible")
    report = SBMLCalibration(
        table, parameters, experiments, progress=lambda p: print(json.dumps(p), flush=True)
    ).fit()
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({key: report.get(key) for key in ("status", "estimate", "resources")}))
    if report["status"] != "fitted":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
