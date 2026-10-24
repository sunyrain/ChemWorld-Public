"""Fixed independent 30-unit development block for catalytic surface analysis."""

import argparse
import json
import math
import time
from dataclasses import replace
from itertools import product
from pathlib import Path

from catalysis54_fixtures import conditions, mechanism
from catalysis54_reference import apparent_energy, cycle, finite_batch, frumkin

from chemworld.physchem.catalysis_analysis import (
    catalytic_rate_control,
    catalytic_steady_states,
    map_catalytic_descriptors,
    solve_catalytic_steady_state,
    with_catalytic_descriptors,
)
from chemworld.physchem.catalysis_network import CatalyticMechanismSpec
from chemworld.physchem.residual_system import RootPolicy
from chemworld.runtime import ChemWorldRuntime
from chemworld.world.descriptor_catalysis import descriptor_catalysis_world


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    units, records, outputs = [], [], {}

    def compare(name, actual, expected, *, relative=2e-6, absolute=1e-9):
        ok = (
            actual == expected
            if isinstance(expected, bool)
            else math.isfinite(actual)
            and math.isclose(actual, expected, rel_tol=relative, abs_tol=absolute)
        )
        records.append(
            {
                "unit": units[-1]["name"],
                "quantity": name,
                "actual": actual,
                "expected": expected,
                "relative_tolerance": relative,
                "absolute_tolerance": absolute,
                "passed": bool(ok),
            }
        )

    def unit(name, function):
        before = len(records)
        row = {"name": name, "error": None}
        units.append(row)
        try:
            outputs[name] = function()
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
        selected = records[before:]
        row.update(
            comparisons_total=len(selected),
            comparisons_passed=sum(r["passed"] for r in selected),
            failures=[r for r in selected if not r["passed"]],
        )
        row["passed"] = row["error"] is None and not row["failures"]
        elapsed = time.monotonic() - started
        print(
            f"units={len(units)}/30 elapsed={elapsed:.2f}s "
            f"throughput={len(units) / max(elapsed, 0.001):.2f}/s "
            f"ETA={elapsed / len(units) * (30 - len(units)):.2f}s {name} passed={row['passed']}",
            flush=True,
        )

    def steady(t, binding):
        model = with_catalytic_descriptors(mechanism(), {"binding": binding, "strain": 0})
        boundary = conditions(temperature=t)
        result = solve_catalytic_steady_state(model, boundary, {"bound": 0.2}, target_species="B")
        ref = cycle(model, boundary)
        compare("converged", result["status"] == "converged", True)
        compare(
            "coverage", result["coverages"]["bound"], ref["coverage"], relative=0, absolute=2e-7
        )
        compare("turnover", result["turnover_rate_s_inv"], ref["turnover"])
        compare("jacobian", result["jacobian_s_inv"][0][0], ref["eigenvalue"])
        compare("stable", result["stability"] == "stable", True)
        return result

    for t, binding in product((320, 350, 400), (-3000, 3000)):
        unit(f"steady:{t}:{binding}", lambda t=t, b=binding: steady(t, b))

    def control(binding):
        model = with_catalytic_descriptors(mechanism(), {"binding": binding, "strain": 0})
        boundary = conditions()
        result = catalytic_rate_control(model, boundary, {"bound": 0.2}, target_species="B")
        ref = cycle(model, boundary)
        compare("available", result["status"] == "available", True)
        for reaction in ("adsorption", "conversion"):
            compare(
                "drc:" + reaction,
                result["rate_control"][reaction]["value"],
                ref["drc_" + reaction],
                relative=0,
                absolute=2e-5,
            )
        compare(
            "drc_sum", sum(r["value"] for r in result["rate_control"].values()), 1.0, absolute=2e-5
        )
        for gas in ("A", "B"):
            compare(
                "order:" + gas,
                result["reaction_orders"][gas]["value"],
                ref["order_" + gas],
                relative=0,
                absolute=2e-5,
            )
        compare(
            "apparent_energy",
            result["apparent_activation_energy"]["value"],
            apparent_energy(model, boundary),
            relative=2e-4,
            absolute=0.1,
        )
        return result

    for binding in (-3000, 0, 3000):
        unit(f"control:{binding}", lambda b=binding: control(b))

    multi, boundary_multi = mechanism(multistable=True), conditions(multistable=True)
    ref_multi = frumkin(multi, boundary_multi)

    def multiroot(index, seed):
        result = solve_catalytic_steady_state(
            multi, boundary_multi, {"bound": seed}, target_species="A"
        )
        compare("converged", result["status"] == "converged", True)
        compare(
            "coverage",
            result["coverages"]["bound"],
            ref_multi[index]["coverage"],
            relative=0,
            absolute=2e-7,
        )
        compare("eigenvalue", result["jacobian_s_inv"][0][0], ref_multi[index]["eigenvalue"])
        compare("stability", result["stability"] == ("unstable" if index == 1 else "stable"), True)
        return result

    for index, seed in enumerate((0.01, 0.5, 0.99)):
        unit(f"frumkin:{seed}", lambda i=index, s=seed: multiroot(i, s))

    points = [{"binding": b, "strain": s} for b, s in product((-6000, 0, 6000), (-2000, 0, 2000))]
    mapped = map_catalytic_descriptors(
        mechanism(), conditions(), points, [{"bound": 0.2}], target_species="B"
    )
    (output / "descriptor-map.json").write_text(json.dumps(mapped, indent=2))

    def grid(index):
        row = mapped["points"][index]
        compare("converged", row["status"] == "converged", True)
        ref = cycle(with_catalytic_descriptors(mechanism(), points[index]), conditions())
        compare(
            "coverage",
            row["roots"][0]["coverages"]["bound"],
            ref["coverage"],
            relative=0,
            absolute=2e-7,
        )
        compare("turnover", row["roots"][0]["turnover_rate_s_inv"], ref["turnover"])
        return row

    for index in range(9):
        unit(f"grid:{index}", lambda i=index: grid(i))

    def hole():
        result = map_catalytic_descriptors(
            mechanism(),
            conditions(),
            [
                {"binding": 0, "strain": 0},
                {"binding": 7000, "strain": 0},
                {"binding": 3000, "strain": 0},
            ],
            [{"bound": 0.2}],
            target_species="B",
        )
        compare("count", result["points_total"], 3)
        compare("valid_count", result["points_converged"], 2)
        compare("invalid_retained", result["points"][1]["status"] == "rejected", True)
        compare("no_fill", result["points"][1]["roots"] == [], True)
        return result

    unit("retained_invalid_map_point", hole)

    def limited():
        result = solve_catalytic_steady_state(
            multi,
            boundary_multi,
            {"bound": 0.2},
            target_species="A",
            policy=RootPolicy(maximum_iterations=1),
        )
        compare("nonconvergence", result["status"] == "not_converged", True)
        compare("no_fake_root", result["coverages"] is None, True)
        return result

    unit("retained_unconverged_root", limited)

    def high():
        model = mechanism()
        boundary = replace(conditions(), gas_partial_pressures_Pa={"A": 1e7, "B": 1000.0})
        result = solve_catalytic_steady_state(model, boundary, {"bound": 0.98}, target_species="B")
        ref = cycle(model, boundary)
        compare("high_coverage", result["coverages"]["bound"] > 0.95, True)
        compare(
            "coverage", result["coverages"]["bound"], ref["coverage"], relative=0, absolute=2e-7
        )
        compare("turnover", result["turnover_rate_s_inv"], ref["turnover"])
        return result

    unit("high_coverage", high)

    def world(binding):
        model = with_catalytic_descriptors(mechanism(), {"binding": binding, "strain": 0})
        boundary = conditions(capacity=1e-8)
        steady_state = solve_catalytic_steady_state(
            model, boundary, {"bound": 0.2}, target_species="B"
        )
        world = descriptor_catalysis_world(
            model,
            steady_state["gas_amounts_mol"],
            capacities_mol=boundary.capacities_mol,
            initial_adsorbates_mol=steady_state["adsorbate_amounts_mol"],
            volume_L=1,
            temperature_K=350,
            support_heat_capacity_J_K=5,
            stop_temperature_K=440,
            maximum_pressure_Pa=2e5,
        )
        native, resumed = ChemWorldRuntime(composition=world), None
        try:
            native.reset(seed=54)
            first = native.execute({"operation": "wait", "duration_s": 2})
            compare("first_commit", first.info["transaction_status"] == "committed", True)
            snapshot = json.loads(json.dumps(native.snapshot().to_dict()))
            (output / f"world-{binding}.json").write_text(json.dumps(world.to_dict(), indent=2))
            (output / f"snapshot-{binding}.json").write_text(json.dumps(snapshot, indent=2))
            resumed = ChemWorldRuntime.from_snapshot(snapshot)
            for runtime in (native, resumed):
                second = runtime.execute({"operation": "wait", "duration_s": 3})
                compare("continued_commit", second.info["transaction_status"] == "committed", True)
            compare("exact_snapshot", native._state.to_dict() == resumed._state.to_dict(), True)
            ref = finite_batch(
                model,
                boundary,
                steady_state["gas_amounts_mol"],
                steady_state["adsorbate_amounts_mol"]["bound"],
                5,
            )
            actual = {
                **native._state.species_amounts,
                "bound": native._state.state_slots.slots[0].values["bound"],
            }
            for key in ref:
                compare("finite:" + key, actual[key], ref[key], relative=0, absolute=2e-9)
            compare(
                "element_balance",
                sum(actual.values()),
                sum(steady_state["gas_amounts_mol"].values())
                + steady_state["adsorbate_amounts_mol"]["bound"],
                relative=0,
                absolute=1e-10,
            )
            compare(
                "sites",
                native._state.state_slots.slots[0].values["capacity:metal"],
                1e-8,
                relative=0,
                absolute=1e-10,
            )
            compare("gas_really_consumed", actual["A"] < steady_state["gas_amounts_mol"]["A"], True)
            return {"steady": steady_state, "final": actual, "independent_final": ref}
        finally:
            native.close()
            if resumed is not None:
                resumed.close()

    for binding in (-3000, 3000):
        unit(f"world:{binding}", lambda b=binding: world(b))

    def unavailable(reverse):
        result = catalytic_rate_control(
            mechanism(),
            conditions(temperature=350 if reverse else 300),
            {"bound": 0.2},
            target_species="A" if reverse else "B",
        )
        if reverse:
            compare("no_reverse_log", result["status"] == "unavailable", True)
        else:
            compare("partial", result["status"] == "partial", True)
            compare(
                "outside_stencil",
                result["apparent_activation_energy"]["status"] == "unavailable",
                True,
            )
            compare(
                "two_trials_retained", len(result["apparent_activation_energy"]["solutions"]), 2
            )
        return result

    unit("reverse_target_unavailable", lambda: unavailable(True))
    unit("temperature_stencil_unavailable", lambda: unavailable(False))

    def codec():
        model = with_catalytic_descriptors(mechanism(), {"binding": 1000, "strain": 500})
        restored = CatalyticMechanismSpec.from_dict(json.loads(json.dumps(model.to_dict())))
        a = solve_catalytic_steady_state(model, conditions(), {"bound": 0.2}, target_species="B")
        b = solve_catalytic_steady_state(restored, conditions(), {"bound": 0.2}, target_species="B")
        compare("exact_model_result", a == b, True)
        return b

    unit("codec_equivalence", codec)

    def seeds():
        result = catalytic_steady_states(
            multi,
            boundary_multi,
            [{"bound": s} for s in (0.01, 0.5, 0.99, 0.2)],
            target_species="A",
        )
        compare("attempts", len(result["attempts"]), 4)
        compare("roots", result["distinct_roots"], 3)
        for i, row in enumerate(result["roots"]):
            compare(
                "root:" + str(i),
                row["coverages"]["bound"],
                ref_multi[i]["coverage"],
                relative=0,
                absolute=2e-7,
            )
        return result

    unit("multiseed_enumeration", seeds)

    summary = {
        "status": "passed" if all(u["passed"] for u in units) and len(units) == 30 else "failed",
        "units_total": len(units),
        "units_passed": sum(u["passed"] for u in units),
        "comparisons_total": len(records),
        "comparisons_passed": sum(r["passed"] for r in records),
        "units": units,
        "failures": [u for u in units if not u["passed"]],
        "elapsed_seconds": time.monotonic() - started,
        "evidence_status": "development; synthetic models; independent mathematical reference",
    }
    for name, value in (("records", records), ("outputs", outputs), ("summary", summary)):
        (output / f"{name}.json").write_text(json.dumps(value, indent=2))
    print(
        json.dumps({k: v for k, v in summary.items() if k not in ("units", "failures")}), flush=True
    )
    return 0 if summary["status"] == "passed" else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    raise SystemExit(run(parser.parse_args().output))
