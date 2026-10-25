"""Fixed 24-unit passive RED coverage using sourced synthetic 48C inputs."""

from dataclasses import replace

from chemworld.physchem.reverse_electrodialysis import ReverseElectrodialysisSpec
from reference.electrodialysis48_fixtures import amounts
from reference.electrodialysis48_fixtures import spec as base_spec

CASES = (
    *(
        {
            "name": f"point-{ions}ion-{direction}-{resistance}ohm",
            "kind": "point",
            "ions": ions,
            "reverse": direction == "reverse",
            "resistance": resistance,
        }
        for ions in (3, 4)
        for direction in ("forward", "reverse")
        for resistance in (30.0, 300.0)
    ),
    {"name": "open-circuit", "kind": "point", "ions": 3, "resistance": None},
    {"name": "short-circuit", "kind": "point", "ions": 4, "resistance": 0.0},
    {"name": "equal-liquids", "kind": "point", "ions": 4, "resistance": 100.0, "equal": True},
    {"name": "binary-analytic-open", "kind": "analytic", "ions": 2, "resistance": None},
    {"name": "grid-three-ion", "kind": "grid", "ions": 3, "resistance": 100.0},
    {"name": "grid-four-ion", "kind": "grid", "ions": 4, "resistance": 100.0},
    {"name": "finite-three-constant", "kind": "finite", "ions": 3, "resistance": 100.0},
    {
        "name": "finite-three-ramp",
        "kind": "finite",
        "ions": 3,
        "resistance": 300.0,
        "ramp": (30.0, 2.0),
    },
    {"name": "finite-four-constant", "kind": "finite", "ions": 4, "resistance": 100.0},
    {
        "name": "finite-four-ramp",
        "kind": "finite",
        "ions": 4,
        "resistance": 300.0,
        "ramp": (30.0, 2.0),
    },
    {"name": "world-shared-three-ion", "kind": "world", "ions": 3, "shared": True},
    {"name": "world-local-four-ion", "kind": "world", "ions": 4, "shared": False},
    {"name": "electrical-load-bath-stops", "kind": "limits", "ions": 3},
    {"name": "water-stop", "kind": "finite", "ions": 3, "resistance": 100.0, "stop": "water"},
    {"name": "capacity-stop", "kind": "finite", "ions": 3, "resistance": 100.0, "stop": "capacity"},
    {"name": "ion-domain-initial-stop", "kind": "ion-stop", "ions": 3},
)


def build(case):
    pair = base_spec(case["ions"])
    ns = (amounts(pair, dilution=0.5), amounts(pair, dilution=1.0))
    if case.get("reverse"):
        ns = (ns[1], ns[0])
    if case.get("equal"):
        ns = (dict(ns[0]), dict(ns[0]))
    if case["kind"] == "analytic":
        diff = tuple((key, 1e-10) for key in pair.properties.ion_ids)
        pair = replace(
            pair,
            aem=replace(
                pair.aem, hydraulic_permeability_m_Pa_s=1e-30, effective_diffusivities_m2_s=diff
            ),
            cem=replace(
                pair.cem, hydraulic_permeability_m_Pa_s=1e-30, effective_diffusivities_m2_s=diff
            ),
        )
    if case.get("stop") == "water":
        pair = replace(pair, minimum_water_mol=1 - 2e-8)
    elif case.get("stop") == "capacity":
        # Exact authored initial 3-ion volume .018000855 L plus a finite 4e-10 L margin.
        pair = replace(pair, capacities_L=(0.0180008554, 0.04))
    if case["kind"] == "ion-stop":
        pair = replace(pair, minimum_ion_concentration_mol_m3=3.0)
    return ReverseElectrodialysisSpec(pair, 1e6, 1.0, 1.0), ns
