"""Conformance probes must stay inside the action's declared discrete domain."""

import gymnasium as gym
import pytest

import chemworld  # noqa: F401
from chemworld.eval.runtime_domain_affordance_audit import _candidate_actions


def test_spectral_candidates_use_integer_scans_and_execute():
    with gym.make("ChemWorld", task_id="reaction-to-distillation", seed=0) as env:
        env.reset(seed=0)
        base = env.unwrapped
        schema = base.action_schema("configure_instrument")
        candidates = dict(_candidate_actions(schema))
        assert {action["scan_count"] for action in candidates.values()} == {1, 64, 128}
        for action in candidates.values():
            assert type(action["scan_count"]) is int
            validation = base.validate_action(action)
            assert validation["valid"], validation
            assert env.step(action)[4]["transaction_status"] == "committed"
        fractional = {**candidates["midpoint"], "scan_count": 64.5}
        invalid = base.validate_action(fractional)
        assert not invalid["valid"]
        assert "payload_spectral_configuration_valid" in invalid["invalid_reasons"]


@pytest.mark.parametrize("inclusive,expected", [(True, {1, 2}), (False, {2})])
def test_integer_boundaries_respect_exclusive_lower_limit(inclusive, expected):
    schema = {"operation": "example", "fields": [{
        "field": "count", "type": "integer", "bounds": {"low": 1., "high": 2.9},
        "lower_bound_inclusive": inclusive,
    }]}
    candidates = _candidate_actions(schema)
    assert {action["count"] for _, action in candidates} == expected
    assert all(type(action["count"]) is int for _, action in candidates)


def test_empty_integer_interval_is_an_invalid_schema_not_an_invalid_action():
    schema = {"operation": "example", "fields": [{
        "field": "count", "type": "integer", "bounds": {"low": 1.2, "high": 1.8},
    }]}
    with pytest.raises(ValueError, match="No integer value"):
        _candidate_actions(schema)
