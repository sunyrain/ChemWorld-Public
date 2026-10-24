"""Exactly twelve prospective units, each with ten fixed comparison fields."""

from fractions import Fraction

STRUCTURAL = (
    {
        "name": "irreversible_branch",
        "matrix": ((1, -1, -1),),
        "irreversible": (True,) * 3,
        "vectors": ((1, 1, 0), (1, 0, 1)),
        "bidirectional": (False, False),
        "rank": 1,
        "lineality": 0,
        "total": 6,
    },
    {
        "name": "single_reversible_no_pseudocycle",
        "matrix": ((-1,), (1,)),
        "irreversible": (False,),
        "vectors": (),
        "bidirectional": (),
        "rank": 1,
        "lineality": 0,
        "total": 0,
    },
    {
        "name": "all_reversible_circuits",
        "matrix": ((1, 1, -1),),
        "irreversible": (False,) * 3,
        "vectors": ((1, -1, 0), (1, 0, 1), (0, 1, 1)),
        "bidirectional": (True,) * 3,
        "rank": 1,
        "lineality": 2,
        "total": 6,
    },
    {
        "name": "pointed_mixed_nonextreme_efm",
        "matrix": ((1, 1, -1),),
        "irreversible": (False, True, True),
        "vectors": ((-1, 1, 0), (1, 0, 1), (0, 1, 1)),
        "bidirectional": (False,) * 3,
        "rank": 1,
        "lineality": 0,
        "total": 6,
    },
    {
        "name": "genuine_internal_cycle",
        "matrix": ((-1, 0, 1), (1, -1, 0), (0, 1, -1)),
        "irreversible": (True,) * 3,
        "vectors": ((1, 1, 1),),
        "bidirectional": (False,),
        "rank": 2,
        "lineality": 0,
        "total": 7,
    },
    {
        "name": "external_exchange",
        "matrix": ((1, -1),),
        "external": ((-1, 0), (0, 1)),
        "irreversible": (True, True),
        "vectors": ((1, 1),),
        "bidirectional": (False,),
        "rank": 1,
        "lineality": 0,
        "total": 3,
    },
    {
        "name": "conservation_redundant_rows",
        "matrix": ((-1, 1, 0), (1, -1, 0), (2, -2, 0)),
        "irreversible": (True,) * 3,
        "vectors": ((0, 0, 1), (1, 1, 0)),
        "bidirectional": (False, False),
        "rank": 1,
        "lineality": 0,
        "total": 6,
    },
    {
        "name": "fractional_source_units",
        "matrix": ((Fraction(1, 2), -1, -Fraction(1, 4)),),
        "irreversible": (True,) * 3,
        "vectors": ((2, 1, 0), (1, 0, 2)),
        "bidirectional": (False, False),
        "rank": 1,
        "lineality": 0,
        "total": 6,
        "scaled": True,
    },
    {
        "name": "near_dependent_exact_full_rank",
        "matrix": ((1.0, 1.0), (1.0, 1.0 + 2.0**-40)),
        "irreversible": (False, False),
        "vectors": (),
        "bidirectional": (),
        "rank": 2,
        "lineality": 0,
        "total": 0,
    },
)

STRUCTURAL_FIELDS = (
    "status",
    "complete",
    "mode_set",
    "mode_count",
    "rank",
    "lineality",
    "total_supports",
    "evaluated_supports",
    "independent_minimal_balance",
    "direction_exchange_and_native_balance",
)
SPECIAL = (
    (
        "native_current_control",
        (
            "before_complete",
            "after_complete",
            "before_modes",
            "after_modes",
            "actual_current_control",
            "actual_row_factor",
            "same_current_clock",
            "analysis_immutable",
            "operations_committed",
            "kinetic_zero_does_not_delete_modes",
        ),
    ),
    (
        "budget_incomplete",
        (
            "status",
            "complete",
            "no_modes_proven",
            "modes",
            "visited",
            "total",
            "reason",
            "complete_empty_distinction",
            "arithmetic_status",
            "arithmetic_reason",
        ),
    ),
    (
        "scope_rejection",
        (
            "rate_rule",
            "assigned_species",
            "event",
            "time_dependent",
            "boolean_coefficient",
            "nonfinite_coefficient",
            "duplicate_reaction",
            "dimension_mismatch",
            "overlapping_pools",
            "invalid_budget",
        ),
    ),
)


def source_scope_model(defect):
    """Valid original declarations outside the reaction-only analysis profile."""
    from xml.etree import ElementTree as ET

    from chemworld.data.sbml import SBMLModel
    from chemworld.examples.sbml_flux_modes import flux_model

    model = flux_model()
    tree = ET.fromstring(model.to_xml())
    core = "{http://www.sbml.org/sbml/level3/version2/core}"
    math = "{http://www.w3.org/1998/Math/MathML}"
    root = tree.find(core + "model")
    if defect == "rate_rule":
        parameter = next(p for p in root.find(core + "listOfParameters") if p.get("id") == "q0")
        parameter.set("constant", "false")
        rules = ET.SubElement(root, core + "listOfRules")
        rule = ET.SubElement(rules, core + "rateRule", variable="q0")
        ET.SubElement(ET.SubElement(rule, math + "math"), math + "cn").text = "0"
    elif defect == "assigned_species":
        ET.SubElement(
            root.find(core + "listOfSpecies"),
            core + "species",
            id="U",
            compartment="V",
            initialAmount="2",
            hasOnlySubstanceUnits="true",
            boundaryCondition="false",
            constant="false",
        )
        rules = ET.SubElement(root, core + "listOfRules")
        rule = ET.SubElement(rules, core + "assignmentRule", variable="U")
        ET.SubElement(ET.SubElement(rule, math + "math"), math + "ci").text = "I0"
    elif defect == "event":
        events = ET.SubElement(root, core + "listOfEvents")
        event = ET.SubElement(events, core + "event", id="e", useValuesFromTriggerTime="true")
        trigger = ET.SubElement(event, core + "trigger", initialValue="false", persistent="true")
        ET.SubElement(ET.SubElement(trigger, math + "math"), math + "true")
    elif defect == "time_dependent":
        expression = next(tree.iter(math + "math"))
        expression.remove(expression[0])
        product = ET.SubElement(expression, math + "apply")
        ET.SubElement(product, math + "times")
        ET.SubElement(product, math + "ci").text = "q0"
        ET.SubElement(
            product,
            math + "csymbol",
            encoding="text",
            definitionURL="http://www.sbml.org/sbml/symbols/time",
        ).text = "time"
    else:
        raise ValueError("unknown scope fixture")
    return SBMLModel(ET.tostring(tree, encoding="unicode"), model.source)
