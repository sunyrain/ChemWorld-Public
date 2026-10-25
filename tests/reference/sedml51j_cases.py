"""Fixed standard declarations over the unmodified fourteen qualified source models."""

from xml.sax.saxutils import escape

from sbml51i_science_cases import AVOGADRO, source_job

KISAO = {
    "direct": "KISAO:0000029",
    "next": "KISAO:0000027",
    "coarse": "KISAO:0000084",
    "fine": "KISAO:0000084",
}


def experiment(unit, strategy, seed=17):
    values = [("KISAO:0000488", seed), ("KISAO:0000415", 100000)]
    if strategy in {"coarse", "fine"}:
        values += [
            ("KISAO:0000467", 0.1 if strategy == "coarse" else 0.005),
            ("KISAO:0000228", 0.1 if strategy == "coarse" else 0.01),
            ("KISAO:0000230", 0.01),
            ("KISAO:0000249", 10),
        ]
    parameters = "".join(
        '<algorithmParameter kisaoID="' + k + '" value="' + str(v) + '"/>' for k, v in values
    )
    generators, datasets = [], []
    for name in sorted(unit["initial"]):
        target = "/s:sbml/s:model/s:listOfSpecies/s:species[@id='" + escape(name) + "']"
        generators.append(
            '<dataGenerator id="dg_' + name + '"><listOfVariables><variable '
            'id="obs_' + name + '" taskReference="course" target="' + target + '" '
            'symbol="KISAO:0000836"/></listOfVariables><math '
            'xmlns="http://www.w3.org/1998/Math/MathML"><ci>obs_' + name + "</ci></math>"
            "</dataGenerator>"
        )
        datasets.append(
            '<dataSet id="column_'
            + name
            + '" label="'
            + name
            + '" dataReference="dg_'
            + name
            + '"/>'
        )
    return (
        '<sedML xmlns="http://sed-ml.org/sed-ml/level1/version4" level="1" version="4" '
        'xmlns:s="http://www.sbml.org/sbml/level3/version2/core"><listOfModels><model id="m" '
        'language="urn:sedml:language:sbml" source="model.xml"/></listOfModels>'
        '<listOfSimulations><uniformTimeCourse id="simulation" initialTime="0" '
        'outputStartTime="0" outputEndTime="' + str(unit["duration_s"]) + '" numberOfSteps="24">'
        '<algorithm kisaoID="'
        + KISAO[strategy]
        + '"><listOfAlgorithmParameters>'
        + parameters
        + "</listOfAlgorithmParameters></algorithm></uniformTimeCourse></listOfSimulations>"
        '<listOfTasks><task id="course" modelReference="m" simulationReference="simulation"/>'
        "</listOfTasks><listOfDataGenerators>"
        + "".join(generators)
        + '</listOfDataGenerators><listOfOutputs><report id="report"><listOfDataSets>'
        + "".join(datasets)
        + "</listOfDataSets></report></listOfOutputs></sedML>"
    )


def protocol(unit, encoding, strategy, seed=17):
    from chemworld.data.sbml import SBMLModel
    from chemworld.data.scientific_inputs import InputSource
    from chemworld.data.sedml_declarations import SEDMLDocument
    from chemworld.physchem.data_conflicts import DataSourceProvenance
    from chemworld.world.sedml_protocol import SEDMLProtocol

    provenance = InputSource(
        DataSourceProvenance(
            "sedml51j-wiring",
            0,
            "Original fourteen elementary sources through standard experiment declarations",
            "1",
            "CC0-1.0",
        ),
        "synthetic",
    )
    parsed = SBMLModel(
        source_job(unit, encoding, strategy)["sbml"], provenance, items_per_mole=AVOGADRO
    )
    return SEDMLProtocol(
        SEDMLDocument(experiment(unit, strategy, seed), provenance), {"model.xml": parsed}
    )
