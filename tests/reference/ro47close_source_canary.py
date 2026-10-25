"""Actual WaterTAP 1.8 original 1D constructions, before native implementation."""

import argparse
import json
import os
import sys
import time
import traceback
from importlib.metadata import version
from pathlib import Path

if reference_path := os.environ.get("CHEMWORLD_WATERTAP_REFERENCE"):
    sys.path.insert(0, str(Path(reference_path).resolve()))


def build_source(
    mode,
    *,
    nfe=3,
    neutral=False,
    temperature=298.15,
    pressure=5e6,
    area=10.0,
    length=5.0,
    feed_scale=1.0,
    salinity=0.035,
    molecular_weights=None,
    source_domain="default",
):
    from idaes.core import FlowsheetBlock
    from pyomo.environ import ConcreteModel
    from watertap.property_models.multicomp_aq_sol_prop_pack import (
        MCASParameterBlock,
    )
    from watertap.property_models.seawater_prop_pack import SeawaterParameterBlock
    from watertap.unit_models.reverse_osmosis_1D import (
        ConcentrationPolarizationType,
        MassTransferCoefficient,
        PressureChangeType,
        ReverseOsmosis1D,
    )
    from watertap.unit_models.reverse_osmosis_base import ModuleType

    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)
    if mode == "mcas":
        ions = ("Na_+", "Cl_-", "Ca_2+", "SO4_2-") + (("urea",) if neutral else ())
        options = {"mw_data": molecular_weights} if molecular_weights else {}
        if neutral:
            options["charge"] = {"Na_+": 1, "Cl_-": -1, "Ca_2+": 2, "SO4_2-": -2, "urea": 0}
        m.fs.properties = MCASParameterBlock(
            solute_list=list(ions),
            diffusivity_data={("Liq", k): 1.5e-9 for k in ions},
            material_flow_basis="mass",
            **options,
        )
        # Exact authored neutral salt feed. The source's mixed-permeate
        # electroneutrality equation determines B_Cl, as in its own example.
        molar = {"H2O": 50.0, "Na_+": 0.25, "Cl_-": 0.25, "Ca_2+": 0.01, "SO4_2-": 0.01}
        if neutral:
            molar["urea"] = 0.02
        molar.update({key: value * feed_scale for key, value in molar.items() if key != "H2O"})
    else:
        m.fs.properties = SeawaterParameterBlock()
    u = m.fs.unit = ReverseOsmosis1D(
        property_package=m.fs.properties,
        has_pressure_change=True,
        concentration_polarization_type=ConcentrationPolarizationType.calculated,
        mass_transfer_coefficient=MassTransferCoefficient.calculated,
        pressure_change_type=PressureChangeType.calculated,
        module_type=ModuleType.flat_sheet,
        finite_elements=nfe,
    )
    if mode == "mcas":
        from pyomo.environ import value

        flows = {k: v * value(m.fs.properties.mw_comp[k]) for k, v in molar.items()}
    else:
        flows = {"H2O": 1 - salinity, "TDS": salinity}
    for key, rate in flows.items():
        u.inlet.flow_mass_phase_comp[0, "Liq", key].fix(rate)
    u.inlet.pressure[0].fix(pressure)
    u.inlet.temperature[0].fix(temperature)
    u.A_comp.fix(4.2e-12)
    u.B_comp.fix(3.5e-8)
    if mode == "mcas":
        u.B_comp[0, "Cl_-"].unfix()
    u.permeate.pressure[0].fix(101325.0)
    u.feed_side.channel_height.fix(0.001)
    u.feed_side.spacer_porosity.fix(0.9)
    u.area.fix(area)
    u.length.fix(length)
    for block in (
        *u.feed_side.properties.values(),
        *(u.feed_side.properties_interface[0, x] for x in u.difference_elements),
        *(u.permeate_side[0, x] for x in u.difference_elements),
        *u.mixed_permeate.values(),
    ):
        _ = block.enth_mass_phase
    if source_domain not in {"default", "forward-pressure-and-nonnegative-solute"}:
        raise ValueError("unknown explicit source reference domain")
    bounds = {
        "deltaP": {"original": [-1e6, 0.0], "selected": [-1e6, 0.0]},
        "solute_flux": {"original": [1e-8, 1e-3], "selected": [1e-8, 1e-3]},
        "water_flux": {"original": [1e-4, 3e-2], "selected": [1e-4, 3e-2]},
        "water_recovery": {"original": [1e-2, 1 - 1e-6], "selected": [1e-2, 1 - 1e-6]},
        "solute_recovery": {"original": [1e-5, 1 - 1e-6], "selected": [1e-5, 1 - 1e-6]},
    }
    if source_domain == "forward-pressure-and-nonnegative-solute":
        u.deltaP[0].setlb(101325.0 - pressure)
        bounds["deltaP"]["selected"] = [101325.0 - pressure, 0.0]
        bounds["solute_flux"]["selected"] = [0.0, 1e-3]
        bounds["water_recovery"]["selected"] = [0.0, 1 - 1e-6]
        bounds["solute_recovery"]["selected"] = [0.0, 1 - 1e-6]
        for variable in u.recovery_mass_phase_comp.values():
            variable.setlb(0.0)
        for (_time, _x, _phase, component), variable in u.flux_mass_phase_comp.items():
            if component in m.fs.properties.solute_set:
                variable.setlb(0.0)
    m._chemworld_reference_configuration = {"source_domain": source_domain, "bounds": bounds}
    return m


def run(output):
    from idaes.core.util.model_statistics import degrees_of_freedom
    from pyomo.contrib.pynumero.asl import AmplInterface
    from pyomo.environ import Constraint, SolverFactory, Var

    output.mkdir(parents=True, exist_ok=False)
    available = SolverFactory("ipopt").available(exception_flag=False)
    metadata = {
        "versions": {p: version(p) for p in ("watertap", "idaes-pse", "pyomo")},
        "ipopt_available": available,
        "pynumero_asl_available": AmplInterface.available(),
        "scope": "source construction and solver availability, not scientific qualification",
    }
    rows = []
    for mode in ("mcas", "seawater"):
        start = time.monotonic()
        row = {"mode": mode}
        print(json.dumps({"stage": "construct-source", "mode": mode}), flush=True)
        try:
            model = build_source(mode)
            row.update(
                constructed=True,
                degrees_of_freedom=degrees_of_freedom(model),
                variables=len(list(model.component_data_objects(Var))),
                equalities=len(list(model.component_data_objects(Constraint, active=True))),
                has_mixed_permeate_charge_constraint=hasattr(
                    model.fs.unit, "eq_electroneutrality_permeate"
                ),
            )
            if available:
                result = SolverFactory("ipopt").solve(model)
                row.update(
                    solve_attempted=True, termination=str(result.solver.termination_condition)
                )
            else:
                row.update(
                    solve_attempted=False,
                    termination="unavailable-ipopt-and-ASL",
                    numerical_comparison_passed=False,
                )
        except Exception as exc:
            row.update(constructed=False, error=repr(exc), traceback=traceback.format_exc())
        row["elapsed_s"] = time.monotonic() - start
        rows.append(row)
        print(json.dumps(row), flush=True)
    (output / "summary.json").write_text(
        json.dumps({**metadata, "rows": rows}, indent=2), encoding="utf8"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
