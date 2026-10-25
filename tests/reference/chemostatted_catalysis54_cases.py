"""37-unit coverage frozen in EXPERIMENT.md after installed canary 05."""

from dataclasses import replace

from chemworld.examples.chemostatted_catalysis import example_boundary, example_mechanism
from chemworld.physchem.catalysis_network import make_catalytic_slot
from chemworld.physchem.chemostatted_catalysis import pack_chemostatted_state
from reference.chemostatted_catalysis54_reference import equilibrium_potential


def fixed_cases():
    cases = [
        {
            "unit": f"{kind}-{temperature}-{pH}-{potential}",
            "kind": kind,
            "temperature": temperature,
            "pH": pH,
            "potential": potential,
            "duration": 20.0,
            "mode": "physical",
            "event": None,
        }
        for kind in ("volmer", "her", "hydroxylation")
        for temperature in (288.15, 318.15)
        for pH in (2.0, 4.0)
        for potential in (-0.25, -0.05)
    ]
    cases.extend(
        {
            "unit": f"nernst-{t}-{pH}",
            "kind": "volmer",
            "temperature": t,
            "pH": pH,
            "potential": equilibrium_potential("volmer", t, pH),
            "duration": 0,
            "mode": "equilibrium",
            "event": None,
        }
        for t in (288.15, 318.15)
        for pH in (1.0, 3.0, 5.0)
    )
    cases.extend(
        {
            "unit": name,
            "kind": kind,
            "temperature": 298.15,
            "pH": 2.0,
            "potential": e,
            "duration": 100.0,
            "mode": "physical",
            "event": event,
        }
        for name, kind, e, event in (
            ("pressure-stop", "her", -0.25, "gas_pressure_limit"),
            ("current-stop", "her", -0.25, "electrode_current_limit"),
            ("heating-stop", "volmer", -0.05, "bath_heating_limit"),
            ("cooling-stop", "volmer", -0.3, "bath_cooling_limit"),
        )
    )
    cases.extend(
        {
            "unit": "world-" + kind,
            "kind": kind,
            "temperature": 298.15,
            "pH": 2.0,
            "potential": -0.25,
            "duration": 20.0,
            "mode": "world",
            "event": None,
        }
        for kind in ("volmer", "her", "hydroxylation")
    )
    return cases


def physical_case(case):
    spec = example_mechanism(case["kind"])
    boundary = example_boundary(pH=case["pH"], temperature=case["temperature"])
    if case["event"] == "gas_pressure_limit":
        boundary = replace(
            boundary,
            maximum_pressure_Pa=1e-6 * 8.31446261815324 * case["temperature"] / 1e-5 + 0.05,
        )
    elif case["event"] == "electrode_current_limit":
        boundary = replace(boundary, maximum_abs_current_A=1e-7)
    elif case["event"] == "bath_heating_limit":
        boundary = replace(boundary, bath=replace(boundary.bath, maximum_power_W=1e-9))
    elif case["event"] == "bath_cooling_limit":
        boundary = replace(boundary, bath=replace(boundary.bath, maximum_cooling_power_W=1e-9))
    theta = 0.5 if case["mode"] == "equilibrium" else 0.1
    ads = (
        {"H_ads": theta * 1e-6}
        if case["kind"] != "hydroxylation"
        else {"oxygen_bound": (1 - theta) * 1e-6, "hydroxyl_bound": theta * 1e-6}
    )
    slot = make_catalytic_slot(
        "catalyst", "reactor", spec, capacities_mol={"metal": 1e-6}, initial_amounts_mol=ads
    )
    return spec, boundary, slot, pack_chemostatted_state(spec, slot, boundary, {"hydrogen": 1e-6})
