from __future__ import annotations

import json

import numpy as np
import pytest

from chemworld.physchem.concentration_adapter_manifest import (
    OWNED_PATHS,
    VacuumConcentrationProvider,
    vacuum_concentration_adapter_manifest,
    vacuum_concentration_provider_contract,
)
from chemworld.physchem.concentration_units import (
    CONCENTRATION_MODEL_ID,
    IDAES_COMMIT,
    IDAES_FLASH_PATH,
    ConcentrationComponentSpec,
    VacuumConcentrationRequest,
    VacuumConcentratorSpec,
    binary_rayleigh_residual,
    simulate_vacuum_concentration,
    vacuum_concentration_model_card,
)
from chemworld.physchem.maturity import ModelAdapterManifest, validate_model_card


def _component(
    component_id: str,
    *,
    vapor_pressure_Pa: float,
    latent_heat_J_mol: float = 10_000.0,
    heat_capacity_J_mol_K: float = 100.0,
    molar_volume_L_mol: float = 0.10,
    activity_coefficient: float = 1.0,
    evaluation_temperature_K: float = 330.0,
    thermal_limit_K: float = 380.0,
) -> ConcentrationComponentSpec:
    return ConcentrationComponentSpec(
        component_id=component_id,
        vapor_pressure_Pa=vapor_pressure_Pa,
        activity_coefficient=activity_coefficient,
        latent_heat_J_mol=latent_heat_J_mol,
        liquid_heat_capacity_J_mol_K=heat_capacity_J_mol_K,
        liquid_molar_volume_L_mol=molar_volume_L_mol,
        evaluation_temperature_K=evaluation_temperature_K,
        thermal_limit_K=thermal_limit_K,
        provenance_id=f"declared-{component_id}-operating-profile",
    )


def _equipment(
    *,
    max_volume_L: float = 2.0,
    minimum_volume_L: float = 0.05,
    max_power_W: float = 2_000.0,
    max_rate_mol_s: float = 1.0,
    condenser_recovery: float = 0.90,
    minimum_pressure_Pa: float = 5_000.0,
    maximum_temperature_K: float = 390.0,
    maximum_duration_s: float = 20_000.0,
) -> VacuumConcentratorSpec:
    return VacuumConcentratorSpec(
        equipment_id="bounded-vacuum-concentrator",
        max_working_volume_L=max_volume_L,
        minimum_residual_volume_L=minimum_volume_L,
        max_heater_power_W=max_power_W,
        max_evaporation_rate_mol_s=max_rate_mol_s,
        condenser_recovery_fraction=condenser_recovery,
        minimum_pressure_Pa=minimum_pressure_Pa,
        maximum_temperature_K=maximum_temperature_K,
        maximum_duration_s=maximum_duration_s,
        provenance_id="declared-vacuum-concentrator-card",
    )


def _request(
    *,
    feed: dict[str, float] | None = None,
    specs: dict[str, ConcentrationComponentSpec] | None = None,
    target_id: str = "product",
    solvent_ids: tuple[str, ...] = ("solvent",),
    initial_temperature_K: float = 330.0,
    operating_temperature_K: float = 330.0,
    pressure_Pa: float = 50_000.0,
    duration_s: float = 100.0,
    power_W: float = 100.0,
    equipment: VacuumConcentratorSpec | None = None,
    endpoint: float = 0.20,
    minimum_target_recovery: float = 0.95,
) -> VacuumConcentrationRequest:
    resolved_feed = feed or {"solvent": 10.0, "product": 1.0}
    resolved_specs = specs or {
        "solvent": _component(
            "solvent",
            vapor_pressure_Pa=100_000.0,
            evaluation_temperature_K=operating_temperature_K,
        ),
        "product": _component(
            "product",
            vapor_pressure_Pa=0.0,
            latent_heat_J_mol=20_000.0,
            heat_capacity_J_mol_K=150.0,
            molar_volume_L_mol=0.05,
            evaluation_temperature_K=operating_temperature_K,
        ),
    }
    return VacuumConcentrationRequest(
        feed_amounts_mol=resolved_feed,
        component_specs=resolved_specs,
        target_component_id=target_id,
        solvent_component_ids=solvent_ids,
        initial_temperature_K=initial_temperature_K,
        operating_temperature_K=operating_temperature_K,
        pressure_Pa=pressure_Pa,
        duration_s=duration_s,
        heater_power_W=power_W,
        equipment=equipment or _equipment(),
        target_solvent_remaining_fraction=endpoint,
        minimum_target_recovery=minimum_target_recovery,
    )


def _assert_closed(result) -> None:
    assert result.material_balance_error_mol <= 1.0e-8
    assert result.volume_balance_error_L <= 1.0e-8
    assert result.energy_balance_error_J <= 1.0e-8
    assert max(result.component_balance_error_mol.values(), default=0.0) <= 1.0e-8
    assert result.heat_duty_J <= result.available_heater_energy_J + 1.0e-6


def test_single_volatile_solvent_matches_exact_energy_limit() -> None:
    result = simulate_vacuum_concentration(
        _request(endpoint=0.0, minimum_target_recovery=1.0)
    )
    expected_evaporation = 100.0 * 100.0 / 10_000.0
    assert result.evaporated_amounts_mol["solvent"] == pytest.approx(
        expected_evaporation,
        abs=1.0e-9,
    )
    assert result.evaporated_amounts_mol["product"] == pytest.approx(0.0)
    assert result.latent_energy_J == pytest.approx(10_000.0)
    assert result.heat_duty_J == pytest.approx(10_000.0)
    assert result.target_recovery == pytest.approx(1.0)
    _assert_closed(result)


def test_binary_differential_trajectory_satisfies_closed_form_rayleigh_identity() -> None:
    feed = {"light": 0.6, "heavy": 0.4}
    specs = {
        "light": _component("light", vapor_pressure_Pa=200_000.0),
        "heavy": _component("heavy", vapor_pressure_Pa=100_000.0),
    }
    result = simulate_vacuum_concentration(
        _request(
            feed=feed,
            specs=specs,
            target_id="heavy",
            solvent_ids=("light",),
            pressure_Pa=20_000.0,
            duration_s=10.0,
            power_W=100.0,
            endpoint=0.0,
            minimum_target_recovery=0.0,
            equipment=_equipment(max_volume_L=1.0, minimum_volume_L=0.0),
        )
    )
    initial_total = sum(feed.values())
    final_total = sum(result.liquid_amounts_mol.values())
    residual = binary_rayleigh_residual(
        initial_total_mol=initial_total,
        final_total_mol=final_total,
        initial_light_fraction=feed["light"] / initial_total,
        final_light_fraction=result.liquid_amounts_mol["light"] / final_total,
        relative_volatility=2.0,
    )
    assert final_total == pytest.approx(0.9, abs=2.0e-8)
    assert residual <= 2.0e-7
    _assert_closed(result)


def test_vacuum_pressure_controls_whether_boiling_is_possible() -> None:
    vacuum = simulate_vacuum_concentration(_request(pressure_Pa=50_000.0))
    high_pressure = simulate_vacuum_concentration(_request(pressure_Pa=120_000.0))
    assert vacuum.evaporated_amounts_mol["solvent"] > 0.0
    assert high_pressure.evaporated_amounts_mol["solvent"] == pytest.approx(0.0)
    assert high_pressure.termination_reason == "equilibrium_pressure"
    assert any("bubble pressure" in warning for warning in high_pressure.warnings)
    _assert_closed(vacuum)
    _assert_closed(high_pressure)


def test_incomplete_sensible_heating_consumes_energy_without_evaporation() -> None:
    result = simulate_vacuum_concentration(
        _request(
            initial_temperature_K=300.0,
            operating_temperature_K=350.0,
            duration_s=10.0,
            power_W=10.0,
        )
    )
    expected_capacity = 10.0 * 100.0 + 1.0 * 150.0
    assert result.final_temperature_K == pytest.approx(300.0 + 100.0 / expected_capacity)
    assert result.sensible_energy_J == pytest.approx(100.0)
    assert result.latent_energy_J == pytest.approx(0.0)
    assert result.termination_reason == "heating_incomplete"
    _assert_closed(result)


def test_solvent_endpoint_stops_operation_early() -> None:
    result = simulate_vacuum_concentration(
        _request(
            duration_s=200.0,
            power_W=1_000.0,
            endpoint=0.5,
            minimum_target_recovery=1.0,
        )
    )
    assert result.endpoint_met is True
    assert result.solvent_remaining_fraction == pytest.approx(0.5, abs=2.0e-8)
    assert result.termination_reason == "solvent_endpoint"
    assert result.elapsed_time_s < result.requested_duration_s
    _assert_closed(result)


def test_volatile_target_recovery_limit_stops_before_hidden_loss() -> None:
    specs = {
        "solvent": _component("solvent", vapor_pressure_Pa=100_000.0),
        "product": _component("product", vapor_pressure_Pa=50_000.0),
    }
    result = simulate_vacuum_concentration(
        _request(
            specs=specs,
            pressure_Pa=10_000.0,
            duration_s=1_000.0,
            power_W=1_000.0,
            endpoint=0.05,
            minimum_target_recovery=0.99,
        )
    )
    assert result.target_recovery == pytest.approx(0.99, abs=2.0e-8)
    assert result.target_recovery_constraint_met is True
    assert result.termination_reason == "target_recovery_limit"
    assert result.endpoint_met is False
    assert result.evaporated_amounts_mol["product"] > 0.0
    assert any("volatile target loss" in warning for warning in result.warnings)
    _assert_closed(result)


def test_equipment_minimum_volume_is_a_hard_stop_event() -> None:
    result = simulate_vacuum_concentration(
        _request(
            duration_s=1_000.0,
            power_W=1_000.0,
            endpoint=0.0,
            minimum_target_recovery=1.0,
            equipment=_equipment(minimum_volume_L=0.55),
        )
    )
    assert result.final_equivalent_liquid_volume_L == pytest.approx(0.55, abs=2.0e-8)
    assert result.termination_reason == "minimum_residual_volume"
    assert any("minimum residual" in warning for warning in result.warnings)
    _assert_closed(result)


def test_equipment_rate_limit_caps_evaporation_below_power_limit() -> None:
    result = simulate_vacuum_concentration(
        _request(
            duration_s=100.0,
            power_W=1_000.0,
            endpoint=0.0,
            minimum_target_recovery=1.0,
            equipment=_equipment(max_rate_mol_s=0.001),
        )
    )
    assert result.evaporated_amounts_mol["solvent"] == pytest.approx(0.1, abs=1.0e-9)
    assert result.average_evaporation_rate_mol_s == pytest.approx(0.001)
    assert result.heater_energy_utilization < 0.02
    _assert_closed(result)


def test_condenser_recovery_splits_every_vapor_component_without_loss() -> None:
    result = simulate_vacuum_concentration(
        _request(
            endpoint=0.0,
            minimum_target_recovery=1.0,
            equipment=_equipment(condenser_recovery=0.75),
        )
    )
    for key, evaporated in result.evaporated_amounts_mol.items():
        assert result.condensate_amounts_mol[key] == pytest.approx(0.75 * evaporated)
        assert result.vent_amounts_mol[key] == pytest.approx(0.25 * evaporated)
    assert any("vent-loss" in warning for warning in result.warnings)
    _assert_closed(result)


def test_zero_power_and_already_met_endpoint_are_explicit_noop_limits() -> None:
    zero_power = simulate_vacuum_concentration(_request(power_W=0.0))
    already_met = simulate_vacuum_concentration(_request(endpoint=1.0))
    assert zero_power.termination_reason == "zero_heater_power"
    assert zero_power.heat_duty_J == 0.0
    assert already_met.termination_reason == "solvent_endpoint"
    assert already_met.elapsed_time_s == 0.0
    assert already_met.endpoint_met is True
    _assert_closed(zero_power)
    _assert_closed(already_met)


def test_zero_equipment_evaporation_capacity_is_an_explicit_limit() -> None:
    result = simulate_vacuum_concentration(
        _request(
            endpoint=0.0,
            minimum_target_recovery=1.0,
            equipment=_equipment(max_rate_mol_s=0.0),
        )
    )
    assert result.termination_reason == "zero_evaporation_capacity"
    assert sum(result.evaporated_amounts_mol.values()) == pytest.approx(0.0)
    assert any("capacity is zero" in warning for warning in result.warnings)
    _assert_closed(result)


def test_near_thermal_limit_is_visible_without_claiming_degradation_kinetics() -> None:
    specs = {
        "solvent": _component(
            "solvent",
            vapor_pressure_Pa=100_000.0,
            evaluation_temperature_K=376.0,
            thermal_limit_K=380.0,
        ),
        "product": _component(
            "product",
            vapor_pressure_Pa=0.0,
            evaluation_temperature_K=376.0,
            thermal_limit_K=380.0,
        ),
    }
    result = simulate_vacuum_concentration(
        _request(
            specs=specs,
            initial_temperature_K=376.0,
            operating_temperature_K=376.0,
        )
    )
    assert result.minimum_thermal_margin_K == pytest.approx(4.0)
    assert any("within 5 K" in warning for warning in result.warnings)
    _assert_closed(result)


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (
            lambda: _component("x", vapor_pressure_Pa=-1.0),
            "vapor_pressure_Pa",
        ),
        (lambda: _request(pressure_Pa=1_000.0), "vacuum limit"),
        (
            lambda: _request(operating_temperature_K=400.0),
            "equipment maximum",
        ),
        (
            lambda: _request(
                specs={
                    "solvent": _component(
                        "solvent",
                        vapor_pressure_Pa=100_000.0,
                        evaluation_temperature_K=350.0,
                        thermal_limit_K=340.0,
                    ),
                    "product": _component(
                        "product",
                        vapor_pressure_Pa=0.0,
                        evaluation_temperature_K=350.0,
                        thermal_limit_K=340.0,
                    ),
                },
                initial_temperature_K=350.0,
                operating_temperature_K=350.0,
            ),
            "component thermal limits",
        ),
        (lambda: _request(duration_s=30_000.0), "duration_s exceeds"),
        (lambda: _request(power_W=3_000.0), "heater_power_W exceeds"),
        (
            lambda: _request(equipment=_equipment(max_volume_L=0.5)),
            "exceeds equipment capacity",
        ),
        (
            lambda: _request(solvent_ids=("product",)),
            "cannot also be a solvent",
        ),
        (
            lambda: _request(specs={"solvent": _component("solvent", vapor_pressure_Pa=1.0)}),
            "exactly match",
        ),
        (
            lambda: _request(
                specs={
                    "solvent": _component("solvent", vapor_pressure_Pa=100_000.0),
                    "product": _component(
                        "product",
                        vapor_pressure_Pa=0.0,
                        evaluation_temperature_K=331.0,
                    ),
                }
            ),
            "evaluated at operating_temperature_K",
        ),
    ],
)
def test_invalid_concentration_domains_fail_explicitly(factory, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        factory()


def test_deterministic_domain_sweep_preserves_all_ledgers_and_limits() -> None:
    rng = np.random.default_rng(20260711)
    for _ in range(30):
        feed = {
            "solvent": float(rng.uniform(2.0, 10.0)),
            "cosolvent": float(rng.uniform(0.5, 4.0)),
            "product": float(rng.uniform(0.2, 2.0)),
        }
        specs = {
            "solvent": _component(
                "solvent",
                vapor_pressure_Pa=float(rng.uniform(60_000.0, 180_000.0)),
                latent_heat_J_mol=float(rng.uniform(8_000.0, 30_000.0)),
                activity_coefficient=float(rng.uniform(0.7, 1.5)),
            ),
            "cosolvent": _component(
                "cosolvent",
                vapor_pressure_Pa=float(rng.uniform(10_000.0, 100_000.0)),
                latent_heat_J_mol=float(rng.uniform(10_000.0, 40_000.0)),
                activity_coefficient=float(rng.uniform(0.7, 1.5)),
            ),
            "product": _component(
                "product",
                vapor_pressure_Pa=float(rng.uniform(0.0, 20_000.0)),
                latent_heat_J_mol=float(rng.uniform(20_000.0, 60_000.0)),
                molar_volume_L_mol=0.05,
            ),
        }
        initial_volume = sum(
            feed[key] * specs[key].liquid_molar_volume_L_mol for key in feed
        )
        result = simulate_vacuum_concentration(
            _request(
                feed=feed,
                specs=specs,
                solvent_ids=("solvent", "cosolvent"),
                pressure_Pa=float(rng.uniform(10_000.0, 70_000.0)),
                duration_s=float(rng.uniform(0.0, 500.0)),
                power_W=float(rng.uniform(0.0, 1_000.0)),
                endpoint=float(rng.uniform(0.1, 0.9)),
                minimum_target_recovery=float(rng.uniform(0.8, 1.0)),
                equipment=_equipment(
                    max_volume_L=initial_volume + 0.5,
                    minimum_volume_L=0.0,
                    condenser_recovery=float(rng.uniform(0.0, 1.0)),
                    max_rate_mol_s=float(rng.uniform(0.001, 0.2)),
                ),
            )
        )
        assert result.target_recovery_constraint_met
        assert result.final_equivalent_liquid_volume_L >= -1.0e-10
        _assert_closed(result)


def test_provider_uses_wf00_contract_for_success_and_failure() -> None:
    provider = VacuumConcentrationProvider()
    success = provider.evaluate({"request": _request()})
    assert success.success is True
    assert success.outputs["concentration_result"]["model_id"] == CONCENTRATION_MODEL_ID
    assert success.diagnostics["material_balance_error_mol"] <= 1.0e-8
    assert success.diagnostics["termination_reason"]

    failure = provider.evaluate({"request": {}})
    assert failure.success is False
    assert failure.failure_reason == "request must be a VacuumConcentrationRequest"
    assert failure.outputs == {}


def test_model_card_and_adapter_are_bounded_hash_verified_evidence() -> None:
    card = vacuum_concentration_model_card()
    assert validate_model_card(card) == []
    assert card.model_id == CONCENTRATION_MODEL_ID
    assert any(IDAES_COMMIT in reference for reference in card.reference_reading)
    assert any(IDAES_FLASH_PATH in reference for reference in card.reference_reading)
    assert any("not evaporator scale-up" in note for note in card.model_limit_notes)

    contract = vacuum_concentration_provider_contract()
    manifest = vacuum_concentration_adapter_manifest()
    assert manifest.provider_contract == contract
    assert manifest.owned_paths == OWNED_PATHS
    assert manifest.replaces_model_ids == ()
    payload = manifest.to_dict()
    assert ModelAdapterManifest.from_dict(payload).to_dict() == payload
    payload["adapter_version"] = "tampered"
    with pytest.raises(ValueError, match="hash mismatch"):
        ModelAdapterManifest.from_dict(payload)


def test_concentration_result_serialization_is_deterministic() -> None:
    request = _request(endpoint=0.0, minimum_target_recovery=1.0)
    left = json.dumps(simulate_vacuum_concentration(request).to_dict(), sort_keys=True)
    right = json.dumps(simulate_vacuum_concentration(request).to_dict(), sort_keys=True)
    assert left == right
