"""Fixed 36-unit explicit selection through original caloric and process consumers."""

import argparse
import json
import math
import time
import traceback
from itertools import product
from pathlib import Path

import mpmath as mp
from tests.reference.run_critical_process_block import compare, gas, liquid
from tests.reference.run_virial_pure_block import independent

from chemworld.data.scientific_inputs import InputSource
from chemworld.physchem import (
    ComponentPropertyPackage,
    ComponentSpec,
    PropertyCorrelation,
    PropertySelectionPolicy,
    virial_correlation,
)
from chemworld.physchem.data_conflicts import DataSourceProvenance
from chemworld.physchem.property_inputs import PropertyInputCard
from chemworld.physchem.property_reports import CaloricReference
from chemworld.physchem.specs import ComponentUncertainty


def card(kind, label):
    component = ComponentSpec("example", "C7H8" if kind == "gas" else "C2H6O")
    source = InputSource(
        DataSourceProvenance(label, 0, "Fixed synthetic selection model", "1", "synthetic"),
        "synthetic",
    )
    if kind == "gas":
        curve = virial_correlation(
            "Abbott",
            parameters={"Tc": 500 if label == "a" else 600, "Pc": 4e6, "omega": 0.2},
            temperature_range_K=(650, 1000),
            correlation_id=label,
            source_note=source.provenance.citation,
        )
        reference = None
    else:
        prop = "liquid_density" if kind == "liquid" else "liquid_heat_capacity"
        c = (
            {"rho_ref": 800 if label == "a" else 1000, "T_ref": 300, "alpha": 0}
            if kind == "liquid"
            else {"a": 50 if label == "a" else 100}
        )
        curve = PropertyCorrelation(
            label,
            "linear_liquid_density" if kind == "liquid" else "cp_polynomial",
            c,
            {"temperature": "K"},
            "kg/m^3" if kind == "liquid" else "J/(mol*K)",
            prop,
            {"temperature": (250, 400)},
            source.provenance.citation,
            {"phase": "liquid"},
        )
        reference = (
            CaloricReference(phase="liquid", temperature_K=280, entropy_J_mol_K=0)
            if kind == "caloric"
            else None
        )
    return PropertyInputCard(
        ComponentPropertyPackage(component, (curve,)),
        (source,),
        {
            label: ComponentUncertainty(
                label,
                unit=curve.output_unit,
                source_id=label,
                coverage="unspecified",
                note="Synthetic selection input",
            )
        },
        reference,
        "Explicit synthetic reference, no material prediction claim",
    )


def unit(kind, mode, order, temperature):
    policy = (
        PropertySelectionPolicy()
        if mode == "declared_order"
        else PropertySelectionPolicy("correlation", "b")
        if mode == "correlation"
        else PropertySelectionPolicy("source_priority", source_priority=("b", "a"))
    )
    selected = order[0] if mode == "declared_order" else "b"
    inputs = tuple(card(kind, key) for key in order)
    prop = inputs[0].package.correlations[0].property_id
    combined = PropertyInputCard.combine(
        inputs, reference_note="Fixed selection process", selection_policies={prop: policy}
    )
    restored = PropertyInputCard.from_json(json.dumps(combined.to_dict()))
    assert restored.to_dict() == combined.to_dict()
    report = restored.package.selection_report(prop, temperature_K=temperature)
    assert len(report.candidates) == 2 and report.selected_id == selected
    if kind == "liquid":
        volume = (
            restored.package.component.molecular_weight_g_mol
            / 1000
            / (800 if selected == "a" else 1000)
        )
        result = liquid([restored], temperature, lambda t, x: volume)
    elif kind == "gas":
        parameters = {"Tc": 500 if selected == "a" else 600, "Pc": 4e6, "omega": 0.2}
        result = gas(restored, temperature, lambda t: independent("Abbott", t, parameters))
    else:
        cp = 50 if selected == "a" else 100
        state = restored.evaluate_state(
            temperature_K=temperature,
            pressure_Pa=101325,
            phase="liquid",
            quantities=("cp", "enthalpy", "entropy"),
        )
        result = {
            "comparisons": {
                name: compare(state.properties[name].value, value)
                for name, value in {
                    "cp": cp,
                    "enthalpy": cp * (temperature - 280),
                    "entropy": cp * math.log(temperature / 280),
                }.items()
            },
            "state_report": state.to_dict(),
        }
    return {"selection": report.to_dict(), "input_card": combined.to_dict(), **result}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    mp.mp.dps = 60
    rows, started = [], time.perf_counter()
    cases = [
        (kind, mode, order, t)
        for kind, mode, order in product(
            ("liquid", "caloric", "gas"),
            ("declared_order", "correlation", "source_priority"),
            (("a", "b"), ("b", "a")),
        )
        for t in ((700, 900) if kind == "gas" else (290, 320))
    ]
    assert len(cases) == 36
    for i, (kind, mode, order, t) in enumerate(cases, 1):
        row = {
            "unit": f"{kind}/{mode}/{''.join(order)}/{t}",
            "inputs": {"kind": kind, "mode": mode, "order": order, "temperature_K": t},
        }
        try:
            row.update(metrics=unit(kind, mode, order, t), status="passed")
        except Exception:
            row.update(status="failed", failure=traceback.format_exc())
        rows.append(row)
        elapsed = time.perf_counter() - started
        print(
            f"{i}/36 {row['unit']}: {row['status']} {i / elapsed:.2f}/s "
            f"ETA {elapsed * (36 - i) / i:.1f}s",
            flush=True,
        )
    summary = {
        "task": "NATIVE-CONV-22G1",
        "mode": "development",
        "block": "selection-process",
        "total": 36,
        "passed": sum(r["status"] == "passed" for r in rows),
        "failed": sum(r["status"] == "failed" for r in rows),
        "seconds": time.perf_counter() - started,
        "results": rows,
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf8"
    )
    print(json.dumps({k: v for k, v in summary.items() if k != "results"}))
    for row in rows:
        if row["status"] == "failed":
            print(row["unit"], row["failure"].splitlines()[-1])
    return bool(summary["failed"])


if __name__ == "__main__":
    raise SystemExit(main())
