"""Original S02 protocols and standard archives for installed experiment execution."""

import argparse
import json
from copy import deepcopy
from pathlib import Path
from xml.etree import ElementTree as ET

from integer51h_distribution_cases import units
from sedml51j_cases import protocol

from chemworld.data.combine_archive import COMBINE_URI, ArchiveEntry, CombineArchive
from chemworld.data.sedml_declarations import SEDMLDocument
from chemworld.world.sedml_protocol import SEDMLProtocol


def integer_event_source(parsed):
    """Keep original chemistry and give the original source a real pending queue."""
    from chemworld.data.sbml import SBMLModel
    from chemworld.data.scientific_inputs import InputSource
    from chemworld.physchem.data_conflicts import DataSourceProvenance

    core = "{http://www.sbml.org/sbml/level3/version2/core}"
    math = "{http://www.w3.org/1998/Math/MathML}"
    tree = ET.fromstring(parsed.to_xml())
    model = tree.find(core + "model")
    definitions = model.find(core + "listOfUnitDefinitions")
    definition = ET.SubElement(definitions, core + "unitDefinition", id="flux_units")
    terms = ET.SubElement(definition, core + "listOfUnits")
    for kind, exponent in (("item", "1"), ("second", "-1")):
        ET.SubElement(terms, core + "unit", kind=kind, exponent=exponent,
                      scale="0", multiplier="1")
    parameters = model.find(core + "listOfParameters")
    coefficient = float(next(p.get("value") for p in parameters if p.get("id") == "k0"))
    for identity, unit in (("marker", "item"), ("memory", "item"), ("flux", "flux_units")):
        ET.SubElement(parameters, core + "parameter", id=identity, value="0",
                      units=unit, constant="false")
    law = model.find(".//" + core + "kineticLaw")
    original_math = law.find(math + "math")
    rules = ET.Element(core + "listOfRules")
    rule = ET.SubElement(rules, core + "assignmentRule", variable="flux")
    rule.append(deepcopy(original_math))
    law.remove(original_math)
    alias = ET.SubElement(law, math + "math")
    ET.SubElement(alias, math + "ci").text = "flux"
    # Source schema order: rules and constraints precede reactions, events follow them.
    model.insert(list(model).index(model.find(core + "listOfReactions")), rules)
    constraints = ET.Element(core + "listOfConstraints")
    constraints.append(ET.fromstring(
        '<constraint xmlns="' + core[1:-1] + '"><math xmlns="' + math[1:-1] + '">'
        '<apply><eq/><apply><plus/><ci>A</ci><ci>B</ci></apply>'
        '<cn xmlns:s="' + core[1:-1] + '" s:units="item">200</cn></apply>'
        '</math></constraint>'
    ))
    model.insert(list(model).index(model.find(core + "listOfReactions")), constraints)
    time = '<csymbol definitionURL="http://www.sbml.org/sbml/symbols/time">time</csymbol>'
    events = ET.SubElement(model, core + "listOfEvents")
    cases = (
        ("early_capture", time, "0.2", "second", "marker", "<ci>A</ci>", 0.5, 3),
        ("long_capture", time, "0.4", "second", "memory", "<ci>B</ci>", 4, 2),
        ("control_edge", "<ci>k0</ci>", repr(coefficient * 1.25), "k_order_2", "marker",
         '<apply><plus/><ci>marker</ci><cn s:units="item">1</cn></apply>', None, 1),
    )
    for identity, left, right, unit, target, value, delay, priority in cases:
        xml = ('<event xmlns="' + core[1:-1] + '" xmlns:s="' + core[1:-1]
               + '" id="' + identity + '" useValuesFromTriggerTime="true">'
               '<trigger initialValue="false" persistent="true"><math xmlns="'
               + math[1:-1] + '"><apply><geq/>' + left + '<cn s:units="' + unit + '">'
               + right + '</cn></apply></math></trigger><priority><math xmlns="'
               + math[1:-1] + '"><cn>' + str(priority) + '</cn></math></priority>')
        if delay is not None:
            xml += ('<delay><math xmlns="' + math[1:-1] + '"><cn s:units="second">'
                    + str(delay) + '</cn></math></delay>')
        xml += ('<listOfEventAssignments><eventAssignment variable="' + target
                + '"><math xmlns="' + math[1:-1] + '">' + value + '</math>'
                '</eventAssignment></listOfEventAssignments></event>')
        events.append(ET.fromstring(xml))
    source = InputSource(DataSourceProvenance(
        "sbml51k-installed", 0, "Original S02 chemistry with source rules and delayed events",
        "1", "CC0-1.0"), "synthetic")
    return SBMLModel(ET.tostring(tree, encoding="unicode"), source,
                     items_per_mole=parsed.items_per_mole)


def event_reports(xml):
    """Observe original parameter coordinates with their normal physical units."""
    generators, datasets = [], []
    for identity in ("marker", "memory"):
        generators.append(
            '<dataGenerator id="dg_' + identity + '"><listOfVariables><variable id="obs_'
            + identity + '" taskReference="scan" target="/s:sbml/s:model/s:listOfParameters/'
            's:parameter[@id=\'' + identity + '\']"/></listOfVariables>'
            '<math xmlns="http://www.w3.org/1998/Math/MathML"><ci>obs_' + identity
            + '</ci></math></dataGenerator>'
        )
        datasets.append('<dataSet id="column_' + identity + '" label="' + identity
                        + '" dataReference="dg_' + identity + '"/>')
    return xml.replace('</listOfDataGenerators>', ''.join(generators) + '</listOfDataGenerators>')\
        .replace('</listOfDataSets>', ''.join(datasets) + '</listOfDataSets>')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--integer-events", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    for method in ("direct", "next", "coarse"):
        original = protocol(units()[1], "item_amount", method)
        parsed = original.models["m"]
        if args.integer_events:
            parsed = integer_event_source(parsed)
        tree = ET.fromstring(parsed.to_xml())
        ns = "{http://www.sbml.org/sbml/level3/version2/core}"
        coefficient = float(
            next(n.get("value") for n in tree.iter(ns + "parameter") if n.get("id") == "k0")
        )
        values = "".join(
            "<value>" + repr(coefficient * scale) + "</value>" for scale in (0.5, 1.0, 1.5)
        )
        for reset in (True, False):
            identity = method + "-" + str(reset).lower()
            repeat = (
                '<repeatedTask id="scan" range="rates" resetModel="'
                + str(reset).lower()
                + '" concatenate="false"><listOfRanges><vectorRange id="rates">'
                + values
                + '</vectorRange></listOfRanges><listOfChanges><setValue modelReference="m" '
                'range="rates" target="/s:sbml/s:model/s:listOfParameters/'
                's:parameter[@id=\'k0\']/@value"><math xmlns="http://www.w3.org/1998/Math/MathML">'
                "<ci>rates</ci></math></setValue></listOfChanges><listOfSubTasks>"
                '<subTask task="course"/></listOfSubTasks></repeatedTask>'
            )
            xml = original.document.to_xml().replace("</listOfTasks>", repeat + "</listOfTasks>")
            xml = xml.replace('taskReference="course"', 'taskReference="scan"')
            if args.integer_events:
                xml = event_reports(xml)
            plan = SEDMLProtocol(
                SEDMLDocument(xml, original.document.source),
                {"model.xml": parsed},
                readout_relative_sd=0.02,
            )
            (args.output / (identity + ".json")).write_text(
                json.dumps(plan.to_dict()), encoding="utf-8"
            )
            archive = CombineArchive(
                (
                    ArchiveEntry("model.xml", COMBINE_URI + "sbml", parsed.to_xml().encode()),
                    ArchiveEntry("experiment.sedml", COMBINE_URI + "sed-ml", xml.encode(), True),
                ),
                parsed.source,
            )
            (args.output / (identity + ".omex")).write_bytes(archive.to_bytes())


if __name__ == "__main__":
    main()
