"""Fixed pump/drive/source-catalog development block; fluids is reference only."""

import argparse
import json
import time
from itertools import pairwise, product
from math import isclose
from pathlib import Path

import fluids
import fluids.pump as ref

from chemworld.data.scientific_inputs import (
    InputSource,
    InputValue,
    QuantityColumn,
    ScientificRow,
    ScientificTable,
)
from chemworld.physchem.data_conflicts import DataSourceProvenance
from chemworld.physchem.pump_engineering import (
    DriveEfficiencyMap,
    ElectricalSupplyCatalog,
    ElectricalSupplyRecord,
    MotorEfficiencyCatalog,
    MotorPartLoadCatalog,
    MotorSizeCatalog,
    PumpDriveSpec,
    corripio_motor_efficiency,
    corripio_pump_efficiency,
    ideal_motor_current,
    motor_synchronous_speed,
    pump_specific_diameter,
    pump_specific_speed,
)
from chemworld.physchem.tabulated_functions import TabulatedScalar
from chemworld.physchem.transport import FluidState, PipeSpec, PumpSpec, pipe_pressure_drop


def source():
    return InputSource(
        DataSourceProvenance(
            "local-pump",
            0,
            "Installed fluids 1.3.1 pump references",
            "1.3.1",
            "Local source observations; redistribution not asserted",
        ),
        "literature",
    )


def table(name, units, grouped):
    rows = []
    for group, values in grouped.items():
        for index, record in enumerate(values):
            rows.append(
                ScientificRow(
                    f"{group}:{index}",
                    "local-pump",
                    group,
                    f"{group}:{index}",
                    "parameters",
                    {
                        k: InputValue(float(v), units[k], None, "Published value")
                        for k, v in record.items()
                    },
                )
            )
    return ScientificTable(
        name,
        "parameters",
        tuple(
            QuantityColumn(k, unit, "parameter", (-1e9, 1e9), "Source parameter")
            for k, unit in units.items()
        ),
        (source(),),
        tuple(rows),
    )


def catalogs():
    sizes = MotorSizeCatalog(
        table(
            "sizes",
            {"power": "W", "nominal_hp": "dimensionless"},
            {
                "NEMA": [
                    {"power": p, "nominal_hp": h}
                    for p, h in zip(ref.nema_sizes, ref.nema_sizes_hp, strict=True)
                ]
            },
        )
    )
    curves = {}
    for grade, closed, poles in product(("minimum", "high"), (False, True), (2, 4, 6, 8)):
        if grade == "high" and poles == 8:
            continue
        stem = "min" if grade == "minimum" else "high"
        powers = getattr(ref, f"nema_{stem}_P")
        housing = "closed" if closed else "open"
        efficiencies = getattr(ref, f"nema_{stem}_full_{housing}_{poles}p")
        curves[f"{grade}:{housing}:{poles}"] = [
            {"power": p * ref.hp, "efficiency": e}
            for p, e in zip(powers, efficiencies, strict=True)
        ]
    motors = MotorEfficiencyCatalog(
        table(
            "motors",
            {"power": "W", "efficiency": "dimensionless"},
            curves,
        )
    )
    vfd_table = table(
        "vfd",
        {
            "power": "W",
            "load": "dimensionless",
            "efficiency": "dimensionless",
        },
        {
            "PWM": [
                {"power": p * ref.hp, "load": load, "efficiency": ref.VFD_efficiencies[i][j]}
                for i, p in enumerate(ref.VFD_efficiency_powers)
                for j, load in enumerate(ref.VFD_efficiency_loads)
            ]
        },
    )
    vfd = DriveEfficiencyMap(
        TabulatedScalar(
            vfd_table,
            (("power_W", "power", "W"), ("load", "load", "dimensionless")),
            "efficiency",
            "dimensionless",
        )
    )
    underload = MotorPartLoadCatalog(
        table(
            "underload",
            {
                "upper_power": "W",
                "unity_load": "dimensionless",
                **{f"c{i}": "dimensionless" for i in range(5)},
            },
            {
                "load": [
                    dict(
                        upper_power=limit * ref.hp,
                        unity_load=threshold,
                        **{f"c{i}": value for i, value in enumerate(coefficients)},
                    )
                    for limit, threshold, coefficients in zip(
                        (1, 5, 10, 25, 60, 100),
                        ref._efficiency_ones,
                        ref._efficiency_lists,
                        strict=True,
                    )
                ]
            },
        )
    )
    records = []
    for sector, values in (
        ("residential", ref.residential_power),
        ("industrial", ref.industrial_power),
    ):
        for code, value in values.items():
            voltages = value.voltage if isinstance(value.voltage, tuple) else (value.voltage,)
            records.append(
                ElectricalSupplyRecord(
                    code,
                    value.country,
                    sector,
                    value.freq,
                    voltages,
                    None if value.plugs is None else tuple(value.plugs),
                    source(),
                )
            )
    supplies = ElectricalSupplyCatalog(
        tuple(records),
        tuple(ref.electrical_plug_types),
        tuple(ref.residential_power_frequencies),
        tuple(ref.voltages_1_phase_residential),
        tuple(ref.voltages_3_phase),
        source(),
    )
    return sizes, motors, vfd, underload, supplies


def inputs():
    rows = []

    def add(family, args):
        rows.append({"family": family, "args": args})

    for q in (50, 75, 100, 250, 500, 1000, 2500, 5000):
        add("pump_efficiency", [q / 15850.323])
    for p in (0.25, 1, 3, 10, 50, 100, 250, 1000):
        add("motor_efficiency", [p * ref.hp])
    for args in product((0.001, 0.01, 0.1), (1, 10, 100), (600, 1800), (0.05, 0.3)):
        add("specific", args)
    for args in product((50, 60), (2, 4, 6, 8)):
        add("synchronous", args)
    for args in product((100, 1000, 10000), (120, 230, 400), (1, 3), (0.7, 1)):
        add("current", args)
    for value in (
        *ref.nema_sizes,
        *((a + b) / 2 for a, b in pairwise(ref.nema_sizes)),
        0,
        2 * ref.nema_sizes[-1],
    ):
        add("size", [value])
    for grade, closed, poles in product((False, True), (False, True), (2, 4, 6, 8)):
        if grade and poles == 8:
            continue
        powers = [p * ref.hp for p in (ref.nema_high_P if grade else ref.nema_min_P)]
        for p in (
            *powers,
            *((a + b) / 2 for a, b in pairwise(powers)),
            powers[0] / 2,
            powers[-1] * 2,
        ):
            add("CSA", [p, closed, poles, grade])
    for power, load in product(ref.VFD_efficiency_powers, ref.VFD_efficiency_loads):
        add("VFD", [power * ref.hp, load])
    for (p0, p1), (l0, l1) in product(
        pairwise(ref.VFD_efficiency_powers), pairwise(ref.VFD_efficiency_loads)
    ):
        add("VFD", [(p0 + p1) / 2 * ref.hp, (l0 + l1) / 2])
    for p, load in ((1, 0.5), (800, 0.5), (30, 0.001), (30, 1.5)):
        add("VFD", [p * ref.hp, load])
    for p, load in product((0.5, 3, 8, 20, 50, 80), (0, 0.1, 0.25, 0.5, 0.7, 0.95, 1)):
        add("part_load", [p * ref.hp, load])
    for sector, catalog in (
        ("residential", ref.residential_power),
        ("industrial", ref.industrial_power),
    ):
        for code in catalog:
            add("supply", [sector, code])
    add("supply_metadata", [])
    for viscosity, maximum in product((0.001, 0.1), (0.002, 0.02)):
        add("drive", [viscosity, maximum])
    assert len(rows) == 1475
    return rows


def evaluate(row, objects):
    sizes, motors, vfd, underload, supplies = objects
    family, a = row["family"], row["args"]
    if family == "pump_efficiency":
        return [(corripio_pump_efficiency(*a), ref.Corripio_pump_efficiency(*a))]
    if family == "motor_efficiency":
        return [(corripio_motor_efficiency(*a), ref.Corripio_motor_efficiency(*a))]
    if family == "specific":
        q, h, n, d = a
        return [
            (pump_specific_speed(q, h, n), ref.specific_speed(q, h, n)),
            (pump_specific_diameter(q, h, d), ref.specific_diameter(q, h, d)),
        ]
    if family == "synchronous":
        return [(motor_synchronous_speed(*a), ref.speed_synchronous(f=a[0], poles=a[1]))]
    if family == "current":
        p, v, phases, pf = a
        return [
            (
                ideal_motor_current(p, v, phases=phases, power_factor=pf),
                ref.current_ideal(p, v, phase=phases, PF=pf),
            )
        ]
    if family == "size":
        values = []
        for fn in (
            lambda: sizes.select(a[0]).values["power"].to("W").value,
            lambda: ref.motor_round_size(a[0]),
        ):
            try:
                values.append(fn())
            except ValueError:
                values.append("rejected")
        return [tuple(values)]
    if family == "CSA":
        p, closed, poles, high = a
        return [
            (
                motors.efficiency(
                    p, closed=closed, poles=poles, high_efficiency=high, outside="clip", digits=4
                ).efficiency,
                ref.CSA_motor_efficiency(p, closed, poles, high),
            )
        ]
    if family == "VFD":
        return [
            (
                vfd.evaluate({"power_W": a[0], "load": a[1]}, outside="clip", digits=4).efficiency,
                ref.VFD_efficiency(*a),
            )
        ]
    if family == "part_load":
        return [(underload.factor(*a), ref.motor_efficiency_underloaded(*a))]
    if family == "supply":
        sector, code = a
        original = (ref.residential_power if sector == "residential" else ref.industrial_power)[
            code
        ]
        record = supplies.lookup(sector, code)
        voltages = original.voltage if isinstance(original.voltage, tuple) else (original.voltage,)
        restored = ElectricalSupplyRecord.from_dict(json.loads(json.dumps(record.to_dict())))
        return [
            (
                record == restored
                and record.voltages_V == voltages
                and record.country == original.country
                and record.frequency_Hz == original.freq
                and record.plugs == (None if original.plugs is None else tuple(original.plugs)),
                True,
            )
        ]
    if family == "supply_metadata":
        restored = ElectricalSupplyCatalog.from_dict(json.loads(json.dumps(supplies.to_dict())))
        return [
            (
                restored == supplies
                and supplies.plug_types == tuple(ref.electrical_plug_types)
                and supplies.residential_frequencies_Hz == tuple(ref.residential_power_frequencies)
                and supplies.residential_voltages_V == tuple(ref.voltages_1_phase_residential)
                and supplies.industrial_voltages_V == tuple(ref.voltages_3_phase),
                True,
            )
        ]
    if family == "drive":
        mu, maximum = a
        fluid, pipe = FluidState(1000, mu), PipeSpec(1, 0.005)
        drive = PumpDriveSpec(PumpSpec(1e5, 0.1, 0.7), maximum, 0.9, 0.95)
        result = drive.operating_point(
            1e-5,
            lambda q: (
                pipe_pressure_drop(pipe, fluid, volumetric_flow_m3_s=q).pressure_drop_total_Pa
            ),
        )
        q, dp = result["actual_flow_m3_s"], result["pressure_rise_Pa"]
        return [
            (result["shaft_power_W"], q * dp / 0.7),
            (result["electrical_power_W"], q * dp / (0.7 * 0.9 * 0.95)),
            (
                result["hydraulic_power_W"] + result["total_external_loss_W"],
                result["electrical_power_W"],
            ),
            (result["shaft_power_W"] <= maximum * (1 + 1e-10), True),
        ]
    raise AssertionError(family)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    results = []
    objects = catalogs()
    rows = inputs()
    count = passed = 0
    for index, row in enumerate(rows):
        record = {"input": row, "comparisons": []}
        try:
            for native, reference in evaluate(row, objects):
                equal = (
                    native == reference
                    if isinstance(native, (str, bool))
                    else isclose(native, reference, rel_tol=2e-12, abs_tol=1e-12)
                )
                record["comparisons"].append(
                    {"native": native, "reference": reference, "passed": equal}
                )
                count += 1
                passed += equal
        except Exception as error:
            record["exception"] = f"{type(error).__name__}: {error}"
        results.append(record)
        if (index + 1) % 250 == 0 or index + 1 == len(rows):
            elapsed = time.perf_counter() - started
            eta = elapsed * (len(rows) - index - 1) / (index + 1)
            print(
                f"pump {index + 1}/{len(rows)}, {passed}/{count} checks, "
                f"{(index + 1) / elapsed:.1f} inputs/s, ETA {eta:.1f}s",
                flush=True,
            )
    failures = [
        r for r in results if "exception" in r or not all(c["passed"] for c in r["comparisons"])
    ]
    report = {
        "reference_version": fluids.__version__,
        "mode": "development",
        "inputs": len(rows),
        "expected_comparisons": 1523,
        "completed_comparisons": count,
        "passed": passed,
        "failed": count - passed,
        "exceptions": sum("exception" in r for r in results),
        "failures": failures,
        "elapsed_s": time.perf_counter() - started,
        "records": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in ("records", "failures")}))
    return 0 if not failures and count == 1523 else 1


if __name__ == "__main__":
    raise SystemExit(main())
