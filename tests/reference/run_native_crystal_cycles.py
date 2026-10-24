"""Fixed native cold/warm/cold cycles and temporal-birth refinement evidence."""

import importlib.util
import json
import sys
import time
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

from chemworld.campaign_resources import CampaignResourceCard
from chemworld.examples.native_crystal_research import (
    synthetic_crystal_model,
    synthetic_crystal_thermodynamics,
)
from chemworld.foundation import equipment_settings
from chemworld.physchem.crystal_thermal_kinetics import advance_crystal_population
from chemworld.physchem.elements import molecular_weight
from chemworld.runtime import ChemWorldRuntime
from chemworld.runtime.crystal_thermal import crystal_inventory_energy, phase_material_energy
from chemworld.world.crystal_world import crystal_world_spec


def refinement(report, save):
    source = Path(__file__).parents[1] / "test_crystal_thermal_kinetics.py"
    module = importlib.util.spec_from_file_location("crystal_fixtures", source)
    fixture = importlib.util.module_from_spec(module)
    module.loader.exec_module(fixture)
    attempt = {"started_utc": datetime.now(UTC).isoformat(), "interval_denominator": 3, "rows": []}
    report.setdefault("birth_refinement_attempts", []).append(attempt)
    start = time.monotonic()
    save()
    for interval in (30.0, 15.0, 7.5):
        print(f"Birth refinement interval {interval} s", flush=True)
        row = {"interval_s": interval}
        try:
            model, thermal, initial = fixture.crystal_fixture(
                diameter=100e-6, saturation=2, nucleation=2e7
            )
            model = replace(model, birth_interval_s=interval)
            result = advance_crystal_population(
                model,
                initial,
                thermal,
                600,
                isothermal=True,
                progress=lambda r: print(r, flush=True),
            )
            final = result.final_state
            birth, seed_birth = fixture.quantities(model, initial)
            current, seed = fixture.quantities(model, final)
            total_count = sum(c.particle_count for c in final.cohorts)
            solid = sum(c.target_mol for c in final.cohorts)
            mean = sum(c.particle_count * c.diameter_m for c in final.cohorts) / total_count
            error = max(abs(current[k] - birth[k]) / birth[k] for k in birth)
            seed_error = abs(seed - seed_birth) / seed_birth
            energy_error = abs(result.energy_balance_residual_J) / max(
                abs(result.boundary_heat_J), 20
            )
            row.update(
                passed=bool(error <= 1e-8 and seed_error <= 1e-8 and energy_error <= 1e-6),
                solid_target_mol=solid,
                number_mean_diameter_m=mean,
                particle_count=total_count,
                normalized_component_error=error,
                normalized_seed_error=seed_error,
                normalized_energy_error=energy_error,
            )
        except Exception as exc:
            row.update(passed=False, error_type=type(exc).__name__, error=str(exc))
        attempt["rows"].append(row)
        save()
    try:
        assert all(row["passed"] for row in attempt["rows"])
        differences = {}
        for key in ("solid_target_mol", "number_mean_diameter_m"):
            a, b, c = (row[key] for row in attempt["rows"])
            differences[key] = {
                "coarse_difference": abs(b - a),
                "fine_difference": abs(c - b),
                "fine_relative_difference": abs(c - b) / abs(c),
            }
            assert abs(c - b) <= abs(b - a)
            assert abs(c - b) / abs(c) <= 0.02
        attempt.update(passed=True, differences=differences)
    except Exception as exc:
        attempt.update(passed=False, error_type=type(exc).__name__, error=str(exc))
    attempt.update(elapsed_s=time.monotonic() - start, finished_utc=datetime.now(UTC).isoformat())
    save()
    print({k: v for k, v in attempt.items() if k != "rows"}, flush=True)
    if not attempt["passed"]:
        raise SystemExit(1)


def cycles(report, save):
    attempt = {
        "started_utc": datetime.now(UTC).isoformat(),
        "cycle_denominator": 2,
        "actions": [],
        "cycles": [],
    }
    report.setdefault("native_cycle_attempts", []).append(attempt)
    save()
    start = time.monotonic()
    session, replay = None, None
    try:
        mw = molecular_weight({"C": 6, "H": 12, "O": 1}) / 1000
        model = synthetic_crystal_model(mw)
        world = crystal_world_spec(
            model, synthetic_crystal_thermodynamics(), {"P": "C6H12O", "I": "C3H8O"}
        )
        thermal = world.physics.models_of('thermal')["reactor"]
        card = CampaignResourceCard(
            card_id="native-crystal-two-cycles",
            operation_attempt_limit=120,
            vessel_start_limit=1,
            final_assay_limit=0,
            nonfinal_instrument_use_limit=4,
            stock_limits={"solvent_L": 0.02, "seed_g": 0.001},
        )
        session = ChemWorldRuntime(composition=world, campaign_resource_card=card)
        session.reset(seed=23)
        historical = {}
        source_energy, seed_input = 0.0, 0.0

        def quantities(state):
            total = dict(state.species_amounts)
            settings = equipment_settings(state.equipment, "crystallizer")
            seed = settings.get("seed_target_mol", 0) + settings.get("dissolved_seed_target_mol", 0)
            volume = state.volume_L
            energy = crystal_inventory_energy(state, model, thermal)
            for sample in state.samples.samples.values():
                if sample.retired:
                    continue
                for key, n in sample.species_amounts_mol.items():
                    total[key] += n
                seed += sample.seed_target_mol
                volume += sample.volume_L
                energy += sum(
                    phase_material_energy(phase, sample.temperature_K, model, thermal)
                    for phase in sample.contents.phases.phases.values()
                )
            return total, seed, energy, volume

        birth, seed_birth, energy_birth, volume_birth = quantities(session._state)

        def check_state():
            current, seed, energy, volume = quantities(session._state)
            expected = {**birth, "P": birth["P"] + seed_input}
            errors = {key: abs(current[key] - expected[key]) / expected[key] for key in expected}
            seed_error = abs(seed - seed_birth - seed_input) / (seed_birth + seed_input)
            utility = session._state.ledger.energy_jacket_J - session._state.ledger.heat_loss_J
            energy_error = abs(energy - energy_birth - utility - source_energy) / max(
                abs(energy - energy_birth), abs(utility), abs(source_energy), 1
            )
            for name, old in historical.items():
                assert session._state.samples.samples[name].to_dict() == old
            historical.update(
                {name: sample.to_dict() for name, sample in session._state.samples.samples.items()}
            )
            assert max(errors.values()) <= 1e-8, errors
            assert seed_error <= 1e-8, seed_error
            assert energy_error <= 1e-6, energy_error
            assert (
                abs(volume - volume_birth - session._state.solvent_accounting.added.volume_L)
                <= 1e-12
            )
            return {
                "normalized_component_errors": errors,
                "normalized_seed_error": seed_error,
                "normalized_energy_error": energy_error,
                "global_internal_energy_J": energy,
                "global_liquid_carrier_L": volume,
            }

        def execute(**action):
            nonlocal source_energy, seed_input
            print(
                f"Native crystal action {len(attempt['actions']) + 1}/20: {action['operation']}",
                flush=True,
            )
            before = session._state
            for current in (session, replay):
                if current is None:
                    continue
                result = current.execute(action)
                if current is session:
                    row = {
                        "action": action,
                        "transaction_status": result.info["transaction_status"],
                        "actual_duration_s": result.info.get("actual_duration_s"),
                        "time_s": current._state.ledger.time_s,
                        "error": result.info.get("error_message"),
                        "diagnostics": result.info.get("outcome_diagnostics"),
                    }
                    attempt["actions"].append(row)
                    save()
                assert result.info["transaction_status"] == "committed", row
            if action["operation"] == "seed_crystals":
                amount = action["seed_mass_g"] / 1000 / mw
                seed_input += amount
                source_energy += amount * thermal.thermodynamics.species_energy_j_mol(
                    model.solid_property_keys["P"], before.temperature_K
                )
            elif action["operation"] == "resuspend_crystals":
                source_energy += session._state.metadata["last_crystal_material_packet"][
                    "inlet_carrier_energy_J"
                ]
            attempt["actions"][-1].update(check_state())
            save()

        execute(operation="seed_crystals", seed_mass_g=0.001)
        for cycle in (1, 2):
            print(f"Native crystal cycle {cycle}/2", flush=True)
            execute(operation="cool_crystallize", target_temperature_K=278.15, duration_s=1200)
            execute(operation="measure", instrument="phase_state")
            execute(operation="filter_crystals")
            execute(operation="resuspend_crystals", solvent=0, volume_L=0.01)
            loaded = session._state.phases.phases["solid"].species_amounts_mol["P"]
            execute(
                operation="heat", target_temperature_K=350, duration_s=600, stirring_speed_rpm=600
            )
            execute(operation="wait", duration_s=60, stirring_speed_rpm=600)
            warm = session._state.phases.phases["solid"].species_amounts_mol["P"]
            fraction = warm / loaded if loaded else None
            execute(operation="pause_control")
            execute(operation="cool_crystallize", target_temperature_K=278.15, duration_s=1200)
            execute(operation="measure", instrument="phase_state")
            attempt["cycles"].append(
                {
                    "cycle": cycle,
                    "passed": True,
                    "warm_remaining_solid_fraction": fraction,
                    "warm_goal_met": bool(fraction is not None and fraction <= 0.5),
                    **check_state(),
                }
            )
            if cycle == 1:
                replay = ChemWorldRuntime.from_snapshot(
                    json.loads(json.dumps(session.snapshot().to_dict()))
                )
                attempt["snapshot_checkpoint_after_cycle"] = 1
            save()
        execute(operation="terminate")
        assert session._state.to_dict() == replay._state.to_dict()
        assert (
            session._campaign_resource_ledger.snapshot()
            == replay._campaign_resource_ledger.snapshot()
        )
        state = session._state
        assert state.ledger.time_s == 7084
        assert abs(state.volume_L - 0.010928) <= 1e-12
        assert len(state.samples.samples) == 2
        assert abs(sum(s.volume_L for s in state.samples.samples.values()) - 0.029072) <= 1e-12
        attempt.update(
            passed=True,
            exact_snapshot_continuation=True,
            final_time_s=state.ledger.time_s,
            unmet_warm_goals=sum(not c["warm_goal_met"] for c in attempt["cycles"]),
            final_state=state.to_dict(),
            resource_ledger=session._campaign_resource_ledger.snapshot(),
        )
    except Exception as exc:
        attempt.update(passed=False, error_type=type(exc).__name__, error=str(exc))
    finally:
        if session is not None:
            session.close()
        if replay is not None:
            replay.close()
    attempt.update(elapsed_s=time.monotonic() - start, finished_utc=datetime.now(UTC).isoformat())
    save()
    print(
        {
            k: attempt.get(k)
            for k in ("passed", "error", "elapsed_s", "final_time_s", "unmet_warm_goals")
        },
        flush=True,
    )
    if not attempt["passed"]:
        raise SystemExit(1)


def main():
    path = Path(sys.argv[1])
    report = json.loads(path.read_text(encoding="utf-8"))

    def save():
        path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    if sys.argv[2] == "refinement":
        return refinement(report, save)
    return cycles(report, save)


if __name__ == "__main__":
    main()
