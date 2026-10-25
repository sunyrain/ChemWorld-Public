"""Fixed input recipes for 50D entry adapters; no numerical execution."""

import hashlib
import json
from pathlib import Path

from sedml_experiment_cases import model_xml

TIMES = (0.0, 0.13, 0.47, 0.83, 1.17, 1.63, 2.13, 2.37, 2.63, 3.11, 4.0)
PENDING = frozenset((*range(1, 10), 11, 17, 18))


def author_source():
    from chemworld.data.scientific_inputs import InputSource
    from chemworld.physchem.data_conflicts import DataSourceProvenance

    return InputSource(
        DataSourceProvenance(
            "native-entry-sbml-50d",
            0,
            "Fixed original 18 source models and declared control/failure fixtures",
            "1",
            "CC0-1.0",
        ),
        "synthetic",
    )


def course_actions(end, steps):
    actions = []
    for index in range(steps + 1):
        if index:
            actions.append({"operation": "wait", "duration_s": end / steps})
        actions.append({"operation": "measure", "instrument": "formal_quantity_meter"})
    return actions


def control_actions():
    actions = []
    for value in (0.1, 0.2):
        actions.append({"operation": "set_model_parameters", "values": {"k": value}})
        actions.extend(course_actions(1.0, 2))
        actions.append({"operation": "measure", "instrument": "formal_quantity_meter"})
        actions.append({"operation": "set_model_parameters", "values": {"k": value * 1.5}})
        actions.extend(course_actions(2.0, 4))
    return [*actions, {"operation": "terminate"}]


def recipes(original_directory, output):
    """Bind the original saved inputs, never regenerate the eighteen reference models."""
    from chemworld.data.sbml import SBMLModel
    from chemworld.physchem.sbml_system import SBMLObservable, parameter_observable
    from chemworld.world.sbml_world import sbml_world

    original_directory = Path(original_directory)
    report = json.loads((original_directory / "report.json").read_text(encoding="utf-8"))
    models = []
    for index in range(1, 19):
        name = f"trajectory-{index:02}.xml"
        original_path = original_directory / ("ode-" + name)
        data = original_path.read_bytes()
        xml_text = original_path.read_text(encoding="utf-8")
        expected = next(
            unit for unit in report["units"] if unit["form"] == "ode" and unit["input"] == name
        )
        # The original runner hashes text before Windows newline translation.
        if hashlib.sha256(xml_text.encode("utf-8")).hexdigest() != expected["sha256"]:
            raise ValueError("original eighteen-source model binding differs")
        (output / (f"B{index:02}-original.xml")).write_bytes(data)
        scale = 60.0 if index == 2 else 1.0
        actions = []
        for node, target in enumerate(TIMES):
            if node:
                actions.append(
                    {"operation": "wait", "duration_s": (target - TIMES[node - 1]) * scale}
                )
            actions.append({"operation": "measure", "instrument": "formal_quantity_meter"})
        actions.append({"operation": "terminate"})
        models.append(
            {
                "id": f"B{index:02}",
                "xml": xml_text,
                "original_bytes_sha256": hashlib.sha256(data).hexdigest(),
                "scale": scale,
                "actions": actions,
                "checkpoint": 15,
                "failure": False,
                "pending": True if index in PENDING else False if index == 10 else None,
            }
        )
    xml = model_xml("D")
    models.append(
        {
            "id": "parameter-control",
            "xml": xml,
            "scale": 1.0,
            "actions": control_actions(),
            "checkpoint": 17,
            "failure": False,
            "pending": None,
        }
    )
    invalid = [
        {"operation": "measure", "instrument": "formal_quantity_meter"},
        {"operation": "set_model_parameters", "values": {"k": 0.9, "missing": 1}},
        {"operation": "measure", "instrument": "formal_quantity_meter"},
        {"operation": "terminate"},
    ]
    models.append(
        {
            "id": "invalid-batch",
            "xml": xml,
            "scale": 1.0,
            "actions": invalid,
            "checkpoint": 2,
            "failure": True,
            "pending": None,
        }
    )
    constraint = (
        '<listOfConstraints><constraint><math xmlns="http://www.w3.org/1998/Math/MathML">'
        "<apply><gt/><ci>A</ci><cn>0.8</cn></apply></math></constraint></listOfConstraints>"
    )
    failed_wait = [invalid[0], {"operation": "wait", "duration_s": 1.5}, *invalid[2:]]
    models.append(
        {
            "id": "constraint-rollback",
            "xml": xml.replace("</model>", constraint + "</model>"),
            "scale": 1.0,
            "actions": failed_wait,
            "checkpoint": 2,
            "failure": True,
            "pending": None,
        }
    )
    prepared = []
    for form in ("ode", "residual"):
        for case in models:
            model = SBMLModel(case["xml"], author_source())
            observations = tuple(
                SBMLObservable(kind, key)
                for kind, names in (
                    ("amount", ("A", "B")),
                    ("concentration", ("A", "B")),
                    ("volume", ("c1", "c2")),
                )
                for key in names
            )
            controls = () if case["id"].startswith("B") else ("k",)
            if controls:
                observations += (
                    parameter_observable(model, "k"),
                    SBMLObservable("reaction_rate", "J"),
                )
            world = sbml_world(
                model,
                budget=80,
                numerical_form=form,
                solver_method="BDF",
                maximum_step_s=0.02 * case["scale"],
                rtol=1e-10,
                atol=1e-12,
                readout_relative_sd=0.02,
                observables=observations,
                controlled_parameters=controls,
            )
            record = {key: value for key, value in case.items() if key != "xml"}
            record.update(
                form=form,
                composition=world.to_dict(),
                source_sha256=hashlib.sha256(case["xml"].encode("utf-8")).hexdigest(),
            )
            path = output / (form + "-" + case["id"] + ".json")
            path.write_text(json.dumps(record, indent=2, allow_nan=False), encoding="utf-8")
            (output / (case["id"] + ".xml")).write_text(case["xml"], encoding="utf-8")
            prepared.append(path)
    return prepared
