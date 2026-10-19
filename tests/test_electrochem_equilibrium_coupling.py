from __future__ import annotations

from collections.abc import Callable
from typing import Any

import gymnasium as gym
import pytest

import chemworld.physchem.electrochemistry as electrochemistry_kernel
import chemworld.runtime.electrochemical_services as runtime_electrochemistry
from chemworld.foundation import equipment_settings, upsert_equipment_record
from chemworld.physchem.electrochem_transport import (
    DiffusionLayerSpec,
    diffusion_layer_current_response,
)
from chemworld.physchem.electrochemistry import (
    ElectrodeReactionSpec,
    run_electrolysis,
)
from chemworld.physchem.equilibrium import (
    davies_aqueous_activity_coefficient,
    weak_acid_davies_activity_ratio,
)
from chemworld.physchem.equilibrium_chemistry import (
    SolubilityProductSpec,
    solve_aqueous_electrolyte_equilibrium,
)
from chemworld.world.electrochemistry import ElectrochemistryModuleSpec
from chemworld.world.instruments import chemworld_instruments


def _configured_env(
    *,
    seed: int = 0,
    electrolyte_profile: int = 1,
    settings: dict[str, float] | None = None,
) -> gym.Env:
    env = gym.make("ChemWorld", task_id="electrochemical-conversion", seed=seed)
    env.reset(seed=seed)
    env.step({"operation": "add_solvent", "volume_L": 0.026, "solvent": 0})
    env.step({"operation": "add_reagent", "amount_mol": 0.010})
    action: dict[str, Any] = {
        "operation": "set_potential",
        "potential_V": 1.15,
        "current_mA": 75.0,
        "electrolyte_profile": electrolyte_profile,
    }
    _observation, _reward, _terminated, _truncated, info = env.step(action)
    assert info["transaction_status"] == "committed", info
    if settings:
        state = env.unwrapped._state
        configured = equipment_settings(
            state.equipment,
            "electrochemical_cell",
        )
        env.unwrapped._state = state.replace(
            equipment=upsert_equipment_record(
                state.equipment,
                equipment_id="electrochemical_cell",
                equipment_type="electrochemical_cell",
                attached_vessel_id=state.vessel_id,
                status="configured",
                settings={**configured, **settings},
            )
        )
    return env


def _physical_snapshot(state: Any) -> dict[str, Any]:
    return {
        "species_amounts": dict(state.species_amounts),
        "volume_L": state.volume_L,
        "temperature_K": state.temperature_K,
        "pressure_Pa": state.pressure_Pa,
        "phase": state.phase,
        "equipment": state.equipment.to_dict(),
        "time_s": state.ledger.time_s,
        "energy_jacket_J": state.ledger.energy_jacket_J,
        "heat_reaction_J": state.ledger.heat_reaction_J,
        "process_metrics": dict(state.process.metrics),
    }


def test_set_potential_is_configuration_only() -> None:
    env = gym.make("ChemWorld", task_id="electrochemical-conversion", seed=0)
    try:
        env.reset(seed=0)
        env.step({"operation": "add_solvent", "volume_L": 0.026, "solvent": 0})
        env.step({"operation": "add_reagent", "amount_mol": 0.010})
        before = env.unwrapped._state
        _obs, _reward, _terminated, _truncated, info = env.step(
            {
                "operation": "set_potential",
                "potential_V": 1.15,
                "current_mA": 75.0,
                "electrolyte_profile": 1,
            }
        )
        after = env.unwrapped._state

        assert info["transaction_status"] == "committed"
        assert after.species_amounts == before.species_amounts
        assert after.ledger.time_s == before.ledger.time_s
        assert after.ledger.energy_jacket_J == before.ledger.energy_jacket_J
        settings = equipment_settings(after.equipment, "electrochemical_cell")
        assert settings["potential_V"] == pytest.approx(1.15)
        assert "runtime_model_ids" not in settings
    finally:
        env.close()


def test_runtime_calls_every_coupled_kernel_and_closes_ledgers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {
        "nernst": 0,
        "butler_volmer": 0,
        "electrolysis": 0,
        "transport": 0,
        "double_layer": 0,
        "aqueous_equilibrium": 0,
    }

    def wrap(module: Any, name: str, counter: str) -> None:
        original: Callable[..., Any] = getattr(module, name)

        def counted(*args: Any, **kwargs: Any) -> Any:
            calls[counter] += 1
            return original(*args, **kwargs)

        monkeypatch.setattr(module, name, counted)

    wrap(electrochemistry_kernel, "nernst_potential", "nernst")
    wrap(electrochemistry_kernel, "butler_volmer_current", "butler_volmer")
    wrap(runtime_electrochemistry, "run_electrolysis", "electrolysis")
    wrap(
        runtime_electrochemistry,
        "diffusion_layer_current_response",
        "transport",
    )
    wrap(
        runtime_electrochemistry,
        "simulate_double_layer_current_step",
        "double_layer",
    )
    wrap(
        runtime_electrochemistry,
        "solve_aqueous_electrolyte_equilibrium",
        "aqueous_equilibrium",
    )

    env = _configured_env()
    try:
        before = env.unwrapped._state
        species_view = env.unwrapped.runtime.domain_services.species_view
        reactant = species_view.reactant_species(before)
        product = species_view.primary_target_species
        impurity = species_view.primary_impurity_species
        _obs, _reward, _terminated, _truncated, info = env.step(
            {"operation": "electrolyze", "duration_s": 1800.0}
        )
        after = env.unwrapped._state

        assert info["transaction_status"] == "committed", info
        assert all(count > 0 for count in calls.values()), calls
        assert calls["electrolysis"] == 2
        metrics = after.process.metrics
        assert abs(metrics["charge_balance_residual_C"]) < 1.0e-9
        assert abs(metrics["material_balance_residual_mol"]) < 1.0e-12
        assert abs(metrics["energy_balance_residual_J"]) < 1.0e-8
        assert metrics["signed_terminal_work_J"] == pytest.approx(
            metrics["signed_interfacial_work_J"]
            + metrics["ohmic_loss_J"]
            + metrics["energy_balance_residual_J"]
        )
        assert abs(metrics["electrolyte_charge_balance_error_eq"]) < 1.0e-9
        assert metrics["electrolyte_material_balance_error_mol"] < 1.0e-9
        assert metrics["charge_C"] == pytest.approx(
            metrics["faradaic_charge_C"]
            + metrics["capacitive_charge_C"]
            + metrics["side_reaction_charge_C"]
        )
        reactant_loss = before.species_amounts[reactant] - after.species_amounts[reactant]
        product_gain = after.species_amounts[product] - before.species_amounts[product]
        impurity_gain = after.species_amounts[impurity] - before.species_amounts[impurity]
        assert reactant_loss == pytest.approx(product_gain + impurity_gain)
        assert after.ledger.energy_jacket_J - before.ledger.energy_jacket_J == pytest.approx(
            metrics["electrical_work_J"]
        )
        settings = equipment_settings(after.equipment, "electrochemical_cell")
        assert set(settings["runtime_model_ids"]) == {
            "nernst_butler_volmer_faradaic_v1",
            "diffusion_layer_limiting_current_v1",
            "randles_double_layer_transient_v1",
            "aqueous_acid_base_ph_observation",
        }
        assert settings["transport_diagnostic"]["charge_balance_residual_C"] == pytest.approx(0.0)
        assert settings["double_layer_diagnostic"]["charge_balance_residual_C"] == pytest.approx(
            0.0
        )
        assert settings["aqueous_equilibrium_diagnostic"]["converged"] is True
        assert species_view.mechanism.manifest.validation_report.passed
        assert equipment_settings(after.equipment, "batch_reactor")
        final_assay = chemworld_instruments()["final_assay"]
        assert {"electrochemical_selectivity", "energy_efficiency"}.issubset(
            final_assay.observable_keys
        )
        assert env.unwrapped.constitution.check_state(after).passed
    finally:
        env.close()


def test_transport_and_electrolyte_perturbations_are_identifiable() -> None:
    fast = _configured_env(
        seed=4,
        settings={
            "diffusivity_m2_s": 1.0e-8,
            "diffusion_layer_thickness_m": 1.0e-6,
            "supporting_electrolyte_mol": 0.0,
            "precipitating_salt_mol": 0.0,
        },
    )
    limited = _configured_env(
        seed=4,
        settings={
            "diffusivity_m2_s": 1.0e-12,
            "diffusion_layer_thickness_m": 1.0e-2,
            "supporting_electrolyte_mol": 0.20 * 0.026,
            "precipitating_salt_mol": 0.0,
        },
    )
    try:
        fast.step({"operation": "electrolyze", "duration_s": 3600.0})
        limited.step({"operation": "electrolyze", "duration_s": 3600.0})
        fast_metrics = fast.unwrapped._state.process.metrics
        limited_metrics = limited.unwrapped._state.process.metrics

        assert (
            fast_metrics["transport_current_efficiency"]
            > (limited_metrics["transport_current_efficiency"])
        )
        assert fast_metrics["faradaic_charge_C"] > limited_metrics["faradaic_charge_C"]
        assert fast_metrics["limiting_current_A"] > limited_metrics["limiting_current_A"]
        assert (
            fast_metrics["electrolyte_ionic_strength_mol_kg"]
            < (limited_metrics["electrolyte_ionic_strength_mol_kg"])
        )
        assert (
            fast_metrics["redox_activity_coefficient"]
            > (limited_metrics["redox_activity_coefficient"])
        )
        assert fast_metrics["equilibrium_potential_V"] != pytest.approx(
            limited_metrics["equilibrium_potential_V"]
        )
    finally:
        fast.close()
        limited.close()


@pytest.mark.parametrize(
    "settings",
    [
        {"electrolyte_conductivity_S_m": 0.0},
        {"diffusion_layer_thickness_m": 0.1},
        {"equilibrium_max_activity_iterations": 1.5},
        {"voltage_window_V": 0.5},
    ],
)
def test_invalid_cell_configuration_fails_closed(settings: dict[str, float]) -> None:
    env = gym.make("ChemWorld", task_id="electrochemical-conversion", seed=1)
    try:
        env.reset(seed=1)
        env.step({"operation": "add_solvent", "volume_L": 0.026, "solvent": 0})
        env.step({"operation": "add_reagent", "amount_mol": 0.010})
        before = _physical_snapshot(env.unwrapped._state)
        action = {
            "operation": "set_potential",
            "potential_V": 1.15,
            "current_mA": 75.0,
            "electrolyte_profile": 1,
            **settings,
        }
        _obs, _reward, _terminated, _truncated, info = env.step(action)

        assert info["transaction_status"] != "committed"
        assert _physical_snapshot(env.unwrapped._state) == before
    finally:
        env.close()


@pytest.mark.parametrize(
    "settings",
    [
        {"equilibrium_max_activity_iterations": 1.0},
        {"equilibrium_max_precipitation_passes": 1.0},
        {"supporting_electrolyte_mol": 0.50 * 0.026},
    ],
)
def test_infeasible_or_nonconverged_electrolyte_rolls_back(settings: dict[str, float]) -> None:
    env = _configured_env(seed=2, settings=settings)
    try:
        before = _physical_snapshot(env.unwrapped._state)
        _obs, _reward, _terminated, _truncated, info = env.step(
            {"operation": "electrolyze", "duration_s": 60.0}
        )

        assert info["transaction_status"] != "committed"
        assert _physical_snapshot(env.unwrapped._state) == before
    finally:
        env.close()


def test_public_aqueous_electrolyte_profiles_activate_distinct_coupled_physics() -> None:
    environments = [
        _configured_env(seed=4, electrolyte_profile=profile)
        for profile in range(4)
    ]
    try:
        snapshots: list[dict[str, float]] = []
        for env in environments:
            _observation, _reward, _terminated, _truncated, info = env.step(
                {"operation": "electrolyze", "duration_s": 1800.0}
            )
            assert info["transaction_status"] == "committed", info
            snapshots.append(dict(env.unwrapped._state.process.metrics))

        low_support, high_support, acidic_transport, precipitation_prone = snapshots
        assert high_support["transport_current_efficiency"] > low_support[
            "transport_current_efficiency"
        ]
        assert high_support["ohmic_efficiency"] > low_support["ohmic_efficiency"]
        assert acidic_transport["electrolyte_pH"] < low_support["electrolyte_pH"]
        assert precipitation_prone["electrolyte_precipitated_mol"] > max(
            snapshot["electrolyte_precipitated_mol"] for snapshot in snapshots[:3]
        )
        assert len(
            {round(snapshot["redox_activity_coefficient"], 8) for snapshot in snapshots}
        ) >= 3
    finally:
        for env in environments:
            env.close()


def test_electrochemical_ph_meter_reads_the_coupled_electrolyte_state() -> None:
    env = _configured_env(seed=4, electrolyte_profile=3)
    try:
        env.step({"operation": "electrolyze", "duration_s": 1800.0})
        metrics = env.unwrapped._state.process.metrics
        _observation, _reward, _terminated, _truncated, info = env.step(
            {"operation": "measure", "instrument": "ph_meter"}
        )
        estimates = info["processed_estimate"]

        assert estimates["pH_normalized"] == pytest.approx(
            metrics["electrolyte_pH"] / 14.0,
            abs=0.02,
        )
        assert estimates["precipitation_signal"] > 0.95
        assert metrics["electrolyte_acid_dissociation_fraction"] > 0.0
    finally:
        env.close()


def test_temperature_domain_violation_rolls_back_electrolysis() -> None:
    env = _configured_env(seed=3)
    try:
        state = env.unwrapped._state
        env.unwrapped._state = state.replace(temperature_K=400.0)
        before = _physical_snapshot(env.unwrapped._state)
        _obs, _reward, _terminated, _truncated, info = env.step(
            {"operation": "electrolyze", "duration_s": 60.0}
        )

        assert info["transaction_status"] != "committed"
        assert _physical_snapshot(env.unwrapped._state) == before
    finally:
        env.close()


def test_aqueous_equilibrium_ph_ksp_and_applicability_contracts() -> None:
    result = solve_aqueous_electrolyte_equilibrium(
        acid_total_mol=0.005,
        volume_L=1.0,
        pka=4.76,
        temperature_K=298.15,
        supporting_electrolyte_mol=0.02,
        precipitating_salt_mol=0.002,
        solubility_product=SolubilityProductSpec("Salt(s)", "Salt+", "Salt-", ksp=1.0e-8),
    )
    assert result.converged
    assert 2.0 < result.acid_base.pH < 5.0
    assert result.precipitation.total_precipitated_mol > 0.0
    assert result.material_balance_error_mol < 1.0e-9
    assert result.charge_balance_error_eq < 1.0e-9
    assert result.activity_coefficient_ratio == pytest.approx(
        weak_acid_davies_activity_ratio(
            ionic_strength_mol_kg=result.acid_base.ionic_strength_mol_kg,
            temperature_K=298.15,
        ),
        rel=1.0e-8,
    )
    with pytest.raises(ValueError, match="ionic-strength applicability"):
        davies_aqueous_activity_coefficient(
            ionic_strength_mol_kg=0.6,
            charge=1.0,
        )
    with pytest.raises(ValueError, match="temperature applicability"):
        davies_aqueous_activity_coefficient(
            ionic_strength_mol_kg=0.1,
            charge=1.0,
            temperature_K=400.0,
        )
    with pytest.raises(RuntimeError, match="did not converge"):
        solve_aqueous_electrolyte_equilibrium(
            acid_total_mol=0.005,
            volume_L=1.0,
            pka=4.76,
            supporting_electrolyte_mol=0.1,
            max_activity_iterations=1,
        )


def test_charge_limited_electrolysis_and_transport_close_exactly() -> None:
    spec = ElectrodeReactionSpec(
        reaction_id="A_to_P",
        electrons_transferred=2.0,
        standard_potential_V=1.0,
        reaction_quotient_exponents={"P": 1.0, "A": -1.0},
        exchange_current_density_A_m2=20.0,
        electrode_area_m2=0.01,
    )
    result = run_electrolysis(
        spec,
        electrode_potential_V=1.3,
        duration_s=100.0,
        activities={"A": 1.0, "P": 1.0},
        available_substrate_mol=1.0,
        applied_current_A=0.1,
        useful_charge_limit_C=4.0,
        capacitive_charge_C=1.0,
    )
    assert result.faradaic_charge_C <= 4.0
    assert result.charge_C == pytest.approx(
        result.faradaic_charge_C + result.capacitive_charge_C + result.side_reaction_charge_C
    )
    assert result.charge_balance_residual_C == pytest.approx(0.0)

    transport = diffusion_layer_current_response(
        DiffusionLayerSpec(
            model_id="test",
            electrons_transferred=2.0,
            electrode_area_m2=0.01,
            diffusivity_m2_s=1.0e-9,
            diffusion_layer_thickness_m=1.0e-4,
            electrolyte_volume_m3=1.0e-4,
            provenance_id="test",
        ),
        bulk_concentration_mol_m3=10.0,
        applied_current_A=1.0,
        duration_s=10.0,
    )
    assert transport.charge_balance_residual_C == pytest.approx(0.0)
    assert transport.material_balance_residual_mol == pytest.approx(0.0, abs=1.0e-15)


def test_world_module_declares_runtime_coupling_laws() -> None:
    payload = ElectrochemistryModuleSpec().to_dict()
    assert payload["version"] == "0.3"
    assert {
        "diffusion_layer_limiting_current",
        "randles_double_layer_transient",
        "aqueous_activity_charge_balance",
        "solubility_product_hooks",
    }.issubset(payload["laws"])
