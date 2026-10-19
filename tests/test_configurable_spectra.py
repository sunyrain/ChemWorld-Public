from __future__ import annotations

import numpy as np
import pytest

from chemworld.agent_interface import action_schema
from chemworld.envs.chemworld_env import ChemWorldEnv
from chemworld.physchem.configurable_spectra import (
    CHANNELS,
    acquire_spectrum,
    calibration,
    design_matrix,
    fit_packet,
)
from chemworld.world.spectral_contract import (
    acquisition_cost,
    acquisition_seconds,
    default_spectral_settings,
)


def request():
    return {
        "schema_version": "chemworld-world-composition-0.1",
        "composition_id": "spectral-research",
        "components": [
            {"kind": "reaction"},
            {"kind": "thermal"},
            {
                "kind": "observation",
                "parameters": {"instruments": ["nmr", "ir", "ms", "final_assay"]},
            },
        ],
        "task": {"budget": 40, "instruments": ["nmr", "ir", "ms", "final_assay"]},
    }


def configuration(instrument, extended=False):
    return {
        "operation": "configure_instrument",
        "instrument": instrument,
        **(
            {"scan_count": 32.0, "resolution_factor": 2.0, "dilution_factor": 4.0}
            if extended
            else default_spectral_settings()
        ),
    }


def actions(instrument="nmr", extended=False):
    return [
        {"operation": "add_solvent", "volume_L": 0.03, "solvent": 0},
        {"operation": "add_reagent", "amount_mol": 0.015},
        {
            "operation": "heat",
            "duration_s": 600,
            "target_temperature_K": 365,
            "stirring_speed_rpm": 600,
        },
        configuration(instrument, extended),
        {"operation": "measure", "instrument": instrument},
        {"operation": "terminate"},
        {"operation": "measure", "instrument": "final_assay"},
    ]


def make_env(seed=0):
    env = ChemWorldEnv(composition=request(), seed=seed)
    env.reset(seed=seed)
    return env


def step(env, action):
    result = env.step(action)
    info = result[-1]
    assert info["transaction_status"] == "committed", (
        action,
        info.get("rollback_reason"),
        {k: v for k, v in info.get("preconditions", {}).items() if not v},
    )
    return result


@pytest.mark.parametrize("instrument", ["nmr", "ir", "ms"])
@pytest.mark.parametrize("extended", [False, True])
@pytest.mark.parametrize("seed", [0, 1])
def test_lifecycle(instrument, extended, seed):
    env = make_env(seed)
    for action in actions(instrument, extended):
        before = env._state
        _, _, terminal, truncated, info = step(env, action)
        if action == {"operation": "measure", "instrument": instrument}:
            packet = info["raw_signal"]
            assert len(packet["axis"]["values"]) == len(packet["raw_signal"]["values"])
            assert np.all(np.isfinite(packet["raw_signal"]["values"]))
            assert packet["processed_estimates"] == fit_packet(packet)
            view = env.observation_view("tool_json")
            summary = view["lab_report"]["spectra_summary"]
            assert summary["has_spectral_packet"]
            assert summary["sample_count"] == len(packet["axis"]["values"])
            assert view["spectral_fit"] == packet["processed_estimates"]
            assert summary["spectral_fit"] == view["spectral_fit"]
            assert view["raw_signal"] == packet
            assert view["processed_estimate"] == info["processed_estimate"]
            assert "sampled points" in view["lab_report"]["text"]
            assert "standard errors" in view["lab_report"]["text"]
            config = packet["settings"]
            assert env._state.ledger.analysis_time_s == pytest.approx(
                acquisition_seconds(instrument, config)
            )
            assert env._state.ledger.cost - before.ledger.cost == pytest.approx(
                acquisition_cost(instrument, config)
            )
            assert env._state.ledger.time_s == before.ledger.time_s
            assert env._state.temperature_K == before.temperature_K
    assert terminal and not truncated


@pytest.mark.parametrize("instrument", ["nmr", "ir", "ms"])
def test_zero_noise_recovery_blank_saturation_and_determinism(instrument):
    config = default_spectral_settings()
    concentrations = (0.15, 0.2, 0.1)
    packet = acquire_spectrum(
        instrument, concentrations, config, np.random.default_rng(0), noise_multiplier=0
    )
    assert list(packet["processed_estimates"]["concentration_mol_L"].values()) == pytest.approx(
        concentrations, abs=1e-8
    )
    assert packet == acquire_spectrum(
        instrument, concentrations, config, np.random.default_rng(0), noise_multiplier=0
    )
    blank = acquire_spectrum(
        instrument, (0.0, 0.0, 0.0), config, np.random.default_rng(0), noise_multiplier=0
    )
    assert set(blank["processed_estimates"]["missingness"].values()) == {"below_lod"}
    assert all(
        value is None for value in blank["processed_estimates"]["concentration_mol_L"].values()
    )
    overloaded = acquire_spectrum(
        instrument, (100.0, 100.0, 100.0), config, np.random.default_rng(0)
    )
    assert set(overloaded["processed_estimates"]["missingness"].values()) == {"saturation"}


@pytest.mark.parametrize("instrument", ["nmr", "ir", "ms"])
def test_resolution_preserves_integral_and_scan_noise(instrument):
    config = default_spectral_settings()
    axis, narrow = calibration(instrument, config)
    _, broad = calibration(instrument, config | {"resolution_factor": 4})
    a, b = design_matrix(axis, narrow), design_matrix(axis, broad)
    assert np.max(b[:, 0]) < np.max(a[:, 0])
    assert np.trapezoid(a, axis, axis=0) == pytest.approx(np.trapezoid(b, axis, axis=0), rel=0.04)
    first = acquire_spectrum(instrument, (0.1, 0.2, 0.1), config, np.random.default_rng(1))
    more = acquire_spectrum(
        instrument, (0.1, 0.2, 0.1), config | {"scan_count": 32}, np.random.default_rng(1)
    )
    assert more["noise"]["standard_deviation"] * 2 == first["noise"]["standard_deviation"]


def test_public_schema_and_gym_roundtrip():
    env = make_env()
    action = configuration("nmr")
    assert set(action) - {"operation"} == {
        f["field"] for f in action_schema(env, "configure_instrument")["fields"]
    }
    decoded = env.action_codec.decode_vector(env.action_codec.encode_vector(action))
    assert decoded == action
    step(env, decoded)


@pytest.mark.parametrize(
    "updates",
    [
        {"scan_count": 0.5},
        {"scan_count": 1.5},
        {"scan_count": True},
        {"resolution_factor": 0},
        {"dilution_factor": float("nan")},
        {"instrument": "hplc"},
    ],
)
def test_invalid_configuration_preserves_equipment(updates):
    env = make_env()
    before = env._state
    assert env.step(configuration("nmr") | updates)[-1]["transaction_status"] != "committed"
    assert env._state.equipment == before.equipment


@pytest.mark.parametrize("instrument", ["nmr", "ir", "ms"])
def test_fixed_64_replicates_match_declared_fit_uncertainty(instrument):
    truth = np.asarray((0.15, 0.2, 0.1))
    errors = []
    for seed in range(64):
        packet = acquire_spectrum(
            instrument, tuple(truth), default_spectral_settings(), np.random.default_rng(seed)
        )
        fit = packet["processed_estimates"]
        errors.append(
            [fit["concentration_mol_L"][name] - truth[i] for i, name in enumerate(CHANNELS)]
        )
    predicted = np.asarray(list(fit["standard_error_mol_L"].values()))
    assert np.all(np.abs(np.mean(errors, axis=0)) <= 4 * predicted / np.sqrt(64))
    ratio = np.std(errors, axis=0, ddof=1) / predicted
    assert np.all((ratio >= 0.65) & (ratio <= 1.35))


def test_spectral_measurement_resource_rejection_and_public_boundary():
    from chemworld.campaign_resources import CampaignResourceCard
    from chemworld.physchem.spectroscopy_adapter_manifest import FORBIDDEN_PACKET_KEYS

    card = CampaignResourceCard(
        card_id="spectral-resource",
        operation_attempt_limit=40,
        vessel_start_limit=1,
        final_assay_limit=1,
        nonfinal_instrument_use_limit=1,
        stock_limits={"solvent_L": 0.1, "reagent_mol": 0.1},
    )
    env = ChemWorldEnv(composition=request(), campaign_resource_card=card)
    env.reset()
    info = None
    for action in actions()[:5]:
        info = step(env, action)[-1]

    def keys(value):
        if isinstance(value, dict):
            return set(value) | set().union(*(keys(v) for v in value.values()))
        if isinstance(value, list):
            return set().union(*(keys(v) for v in value))
        return set()

    assert not (keys(info["raw_signal"]) & FORBIDDEN_PACKET_KEYS)
    before = env._state
    assert (
        env.step({"operation": "measure", "instrument": "nmr"})[-1]["transaction_status"]
        == "campaign_resource_rejected"
    )
    assert env._state.ledger.analysis_time_s == before.ledger.analysis_time_s
    assert env._state.volume_L == before.volume_L
