"""Fixed twelve self-authored complete models/experiments; CC0-1.0 inputs only."""

CORE = "http://www.sbml.org/sbml/level3/version2/core"
SEDML = "http://sed-ml.org/sed-ml/level1/version4"
MATH = "http://www.w3.org/1998/Math/MathML"
PARAMETER = "/s:sbml/s:model/s:listOfParameters/s:parameter[@id='k']"


def math_xml(expression):
    return '<math xmlns="'+MATH+'">'+expression+'</math>'


def unit(identity, terms):
    return ('<unitDefinition id="'+identity+'"><listOfUnits>'+''.join(
        f'<unit kind="{kind}" exponent="{exponent}" scale="{scale}" multiplier="{multiplier}"/>'
        for kind, exponent, scale, multiplier in terms)+'</listOfUnits></unitDefinition>')


def model_xml(kind, *, scaled=False):
    amount, volume, time = ("mmol", "mL", "minute") if scaled else ("mole", "litre", "second")
    scale, seconds = (-3, 60) if scaled else (0, 1)
    units = [unit("per_time", (("second", -1, 0, seconds),)),
        unit("molar", (("mole", 1, scale, 1), ("litre", -1, scale, 1))),
        unit("molar_rate", (("mole", 1, scale, 1), ("litre", -1, scale, 1),
                            ("second", -1, 0, seconds)))]
    if scaled:
        units += [unit("mmol", (("mole", 1, -3, 1),)), unit("mL", (("litre", 1, -3, 1),)),
                  unit("minute", (("second", 1, 0, 60),))]
    numerator = '<apply><times/><ci>c1</ci><ci>k</ci><ci>A</ci></apply>'
    expression = numerator
    extra = ""
    if kind == "M":
        expression = ('<apply><divide/>'+numerator
                      +'<apply><plus/><ci>Km</ci><ci>A</ci></apply></apply>')
        extra = '<parameter id="Km" value="0.4" units="molar" constant="true"/>'
    elif kind == "F":
        expression = ('<apply><divide/>'+numerator+'<apply><plus/><cn>1</cn><apply><divide/>'
            '<ci>B</ci><apply><times/><ci>c2</ci><ci>Ki</ci></apply></apply></apply></apply>')
        extra = '<parameter id="Ki" value="0.3" units="molar" constant="true"/>'
    c1, c2, k = ("1000", "500", "12") if scaled else ("1", "0.5", "0.2")
    k_unit = "molar_rate" if kind == "M" else "per_time"
    return (f'<sbml xmlns="{CORE}" level="3" version="2"><model id="source_{kind}" '
        f'timeUnits="{time}" substanceUnits="{amount}" extentUnits="{amount}" '
        f'volumeUnits="{volume}">'
        '<notes><body xmlns="http://www.w3.org/1999/xhtml"><p>Self-authored CC0 formal kinetics; '
        'chemical compositions and energies are unspecified.</p></body></notes>'
        '<listOfUnitDefinitions>'+''.join(units)+'</listOfUnitDefinitions><listOfCompartments>'
        f'<compartment id="c1" size="{c1}" units="{volume}" spatialDimensions="3" constant="true"/>'
        f'<compartment id="c2" size="{c2}" units="{volume}" spatialDimensions="3" constant="true"/>'
        '</listOfCompartments><listOfSpecies>'
        f'<species id="A" compartment="c1" initialConcentration="1" substanceUnits="{amount}" '
        'hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>'
        f'<species id="B" compartment="c2" initialAmount="0" substanceUnits="{amount}" '
        'hasOnlySubstanceUnits="true" boundaryCondition="false" constant="false"/>'
        f'</listOfSpecies><listOfParameters><parameter id="k" value="{k}" units="{k_unit}" '
        'constant="true"/>'+extra+'</listOfParameters><listOfReactions><reaction id="J" '
        'reversible="false"><listOfReactants><speciesReference species="A" stoichiometry="1" '
        'constant="true"/></listOfReactants><listOfProducts><speciesReference species="B" '
        'stoichiometry="1" constant="true"/></listOfProducts><kineticLaw>'+math_xml(expression)
        +'</kineticLaw></reaction></listOfReactions></model></sbml>')


def simulation(identity="simulation", *, initial="0", start="0", end="3", steps=6):
    return (f'<uniformTimeCourse id="{identity}" initialTime="{initial}" outputStartTime="{start}" '
        f'outputEndTime="{end}" numberOfSteps="{steps}"><algorithm kisaoID="KISAO:0000088">'
        '<listOfAlgorithmParameters><algorithmParameter kisaoID="KISAO:0000209" value="1e-10"/>'
        '<algorithmParameter kisaoID="KISAO:0000211" value="1e-12"/>'
        '<algorithmParameter kisaoID="KISAO:0000467" value="0.02"/>'
        '</listOfAlgorithmParameters></algorithm></uniformTimeCourse>')


def vector(identity, values):
    return '<vectorRange id="'+identity+'">'+''.join('<value>'+str(value)+'</value>'
        for value in values)+'</vectorRange>'


def change_from_range(identity="k_grid"):
    return (f'<setValue modelReference="m" range="{identity}" target="{PARAMETER}/@value">'
        +math_xml('<ci>'+identity+'</ci>')+'</setValue>')


def repeat(identity, ranges, children, *, reset, concatenate=False, changes="", master="k_grid"):
    return (f'<repeatedTask id="{identity}" range="{master}" resetModel="{str(reset).lower()}" '
        f'concatenate="{str(concatenate).lower()}"><listOfRanges>'+ranges+'</listOfRanges>'
        +('<listOfChanges>'+changes+'</listOfChanges>' if changes else '')
        +'<listOfSubTasks>'+children+'</listOfSubTasks></repeatedTask>')


def experiment_xml(case):
    index = int(case["id"][1:])
    source = '<model id="m" language="urn:sedml:language:sbml" source="../models/model.xml"/>'
    if index == 5:
        variables = (f'<listOfVariables><variable id="base_in" modelReference="base" '
            f'target="{PARAMETER}/@value"/><variable id="current_in" modelReference="m" '
            f'target="{PARAMETER}/@value"/></listOfVariables>')
        source = ('<model id="base" language="urn:sedml:language:sbml" '
            'source="../models/model.xml"/><model id="m" language="urn:sedml:language:sbml" '
            'source="#base"><listOfChanges><changeAttribute target="'+PARAMETER+'/@value" '
            'newValue="0.3"/><computeChange target="'+PARAMETER+'/@value">'+variables
            +math_xml('<apply><plus/><ci>base_in</ci><ci>current_in</ci></apply>')
            +'</computeChange></listOfChanges></model>')
    simulations = (simulation(end="0.05") if index == 4 else
        simulation(initial="5", start="5.5", end="7.5", steps=4) if index == 6 else simulation())
    tasks = '<task id="course" modelReference="m" simulationReference="simulation"/>'
    root_task = "course"
    if 7 <= index <= 10:
        if index in (7, 8):
            start, end, spacing = ("0.1", "0.3", "linear") if index == 7 else ("0.05", "0.2", "log")
            ranges = (f'<uniformRange id="k_grid" start="{start}" end="{end}" '
                       f'numberOfSteps="2" type="{spacing}"/>')
        elif index == 9:
            ranges = vector("k_grid", ("0.1", "0.2", "0.4"))
        else:
            ranges = (vector("base_grid", ("0.05", "0.1", "0.2"))
                +'<functionalRange id="k_grid" range="base_grid">'
                +math_xml('<apply><times/><cn>2</cn><ci>base_grid</ci></apply>')+'</functionalRange>')
        tasks += repeat("scan", ranges, '<subTask task="course" order="1"/>',
            reset=index in (7, 9), changes=change_from_range())
        root_task = "scan"
    elif index == 11:
        tasks += repeat("inner", vector("inner_grid", (1, 2, 3)),
            '<subTask task="course" order="1"/>', master="inner_grid", reset=False)
        tasks += repeat("scan", vector("k_grid", ("0.1", "0.2")),
            '<subTask task="inner" order="1"/>', reset=True, changes=change_from_range())
        root_task = "scan"
    elif index == 12:
        simulations = (simulation("fast_sim", end="1", steps=2)
                       +simulation("slow_sim", end="2", steps=4))
        tasks = ('<task id="fast" modelReference="m" simulationReference="fast_sim"/>'
                 '<task id="slow" modelReference="m" simulationReference="slow_sim"/>')
        change = (f'<setValue modelReference="m" target="{PARAMETER}/@value"><listOfVariables>'
            f'<variable id="sub_k" modelReference="m" target="{PARAMETER}"/></listOfVariables>'
            +math_xml('<apply><times/><cn>1.5</cn><ci>sub_k</ci></apply>')+'</setValue>')
        children = ('<subTask task="slow" order="2"><listOfChanges>'+change
            +'</listOfChanges></subTask><subTask task="fast" order="1"/>')
        tasks += repeat("scan", vector("k_grid", ("0.1", "0.2")), children,
                         reset=False, concatenate=True, changes=change_from_range())
        root_task = "scan"
    generators, datasets = "", ""
    targets = (("t", 'symbol="urn:sedml:symbol:time"'),
        ("a", 'target="/s:sbml/s:model/s:listOfSpecies/s:species[@id=\'A\']"'),
        ("b", 'target="/s:sbml/s:model/s:listOfSpecies/s:species[@id=\'B\']"'),
        ("j", 'target="/s:sbml/s:model/s:listOfReactions/s:reaction[@id=\'J\']"'),
        ("v", 'target="/s:sbml/s:model/s:listOfCompartments/s:compartment[@id=\'c1\']"'))
    for label, target in targets:
        generators += (f'<dataGenerator id="dg_{label}"><listOfVariables>'
            f'<variable id="obs_{label}" '
            f'taskReference="{root_task}" modelReference="m" {target}/></listOfVariables>'
            +math_xml('<ci>obs_'+label+'</ci>')+'</dataGenerator>')
        datasets += f'<dataSet id="column_{label}" label="{label}" dataReference="dg_{label}"/>'
    return (f'<sedML xmlns="{SEDML}" xmlns:s="{CORE}" level="1" version="4"><listOfModels>'
        +source+'</listOfModels><listOfSimulations>'+simulations+'</listOfSimulations><listOfTasks>'
        +tasks+'</listOfTasks><listOfDataGenerators>'+generators+'</listOfDataGenerators>'
        '<listOfOutputs><report id="report"><listOfDataSets>'+datasets
        +'</listOfDataSets></report></listOfOutputs></sedML>')


def cases():
    shapes = [(7,)]*5+[(5,)]+[(3, 7)]*4+[(2, 3, 7), (16,)]
    operations = [13]*5+[10, 39, 42, 39, 42, 78, 34]
    kinds = ("D", "M", "F", "D", "D", "D", "D", "D", "M", "F", "D", "D")
    for index, (kind, shape, count) in enumerate(zip(kinds, shapes, operations, strict=True), 1):
        record = {"id": f"C{index:02}", "kind": kind, "shape": shape,
                  "expected_operations": count, "scaled": index == 4}
        record["model_xml"] = model_xml(kind, scaled=index == 4)
        record["sedml_xml"] = experiment_xml(record)
        yield record
