"""Optional COPASI direct-method adapter for our finite synthetic SBML fixture.

Reference-only dependency. No COPASI implementation is imported by the runtime.
"""

from xml.etree.ElementTree import Element, SubElement, tostring

from chemworld.foundation.discrete_inventory import AVOGADRO


def fixture_sbml(network, counts):
    root = Element("sbml", xmlns="http://www.sbml.org/sbml/level2/version4", level="2", version="4")
    model = SubElement(root, "model", id="finite_enzyme")
    compartments = SubElement(model, "listOfCompartments")
    SubElement(
        compartments,
        "compartment",
        id="cell",
        size=str(network.stochastic["volume_L"]),
        spatialDimensions="3",
        units="litre",
        constant="true",
    )
    species = SubElement(model, "listOfSpecies")
    for name in network.species_ids:
        SubElement(
            species,
            "species",
            id=name,
            compartment="cell",
            initialAmount=str(counts.get(name, 0) / AVOGADRO),
            substanceUnits="mole",
            hasOnlySubstanceUnits="false",
            boundaryCondition="false",
            constant="false",
        )
    reactions = SubElement(model, "listOfReactions")
    for channel in network.reactions:
        reaction = SubElement(reactions, "reaction", id=channel.reaction_id, reversible="false")
        for tag, terms in (
            ("listOfReactants", channel.reactants),
            ("listOfProducts", channel.products),
        ):
            group = SubElement(reaction, tag)
            for key, n in terms.items():
                SubElement(group, "speciesReference", species=key, stoichiometry=str(n))
        law = SubElement(reaction, "kineticLaw")
        math = SubElement(law, "math", xmlns="http://www.w3.org/1998/Math/MathML")
        apply = SubElement(math, "apply")
        SubElement(apply, "times")
        SubElement(apply, "ci").text = "cell"
        SubElement(apply, "ci").text = "k"
        for key, n in channel.reactants.items():
            if n != 1:
                raise ValueError("reference fixture uses only distinct first-order reactants")
            SubElement(apply, "ci").text = key
        params = SubElement(law, "listOfParameters")
        SubElement(params, "parameter", id="k", value=str(channel.rate_law.parameters["k"]))
    return tostring(root, encoding="unicode")


class CopasiReference:
    def __init__(self, network, counts):
        import COPASI

        self.api = COPASI
        self.dm = COPASI.CRootContainer.addDatamodel()
        if not self.dm.importSBMLFromString(fixture_sbml(network, counts)):
            raise RuntimeError(COPASI.CCopasiMessage.getAllMessageText())
        model = self.dm.getModel()
        model.setAvogadro(AVOGADRO, COPASI.CCore.Framework_Concentration)
        self.task = self.dm.getTask("Time-Course")
        self.task.setMethodType(COPASI.CTaskEnum.Method_directMethod)
        problem = self.task.getProblem()
        problem.setStepNumber(1)
        problem.setTimeSeriesRequested(True)
        problem.setAutomaticStepSize(False)
        problem.setOutputEvent(False)
        self.task.getMethod().getParameter("Use Random Seed").setValue(True)
        self.names = tuple(network.species_ids)
        self.initial = {key: counts.get(key, 0) for key in self.names}
        self.version = COPASI.CVersion.VERSION.getVersion()

    def run(self, seed, duration):
        self.task.getProblem().setDuration(duration)
        self.task.getMethod().getParameter("Random Seed").setValue(seed)
        if not self.task.process(True):
            raise RuntimeError(self.api.CCopasiMessage.getAllMessageText())
        series = self.task.getTimeSeries()
        indices = {str(series.getTitle(i)): i for i in range(series.getNumVariables())}
        result = {}
        for key in self.names:
            i = indices[key]
            initial = series.getData(0, i)
            if abs(initial - self.initial[key]) > 1e-7:
                raise ValueError(f"COPASI initial unit/count mismatch: {key}={initial}")
            value = series.getData(series.getRecordedSteps() - 1, i)
            if abs(value - round(value)) > 1e-7:
                raise ValueError("noninteger reference result")
            result[key] = round(value)
        return result

    def close(self):
        self.api.CRootContainer.removeDatamodel(self.dm)
