"""Frozen construct-validity controls for the Work I agency profile.

The controls are deliberately simple and deterministic.  Their purpose is to
create known differences in observable experimental policy, not to compete on
endpoint score.  The threshold value remains unbound until W1-V03 qualifies it
on worlds disjoint from the five formal worlds.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any

from chemworld.eval.policy_validity_contract import profile_contract_sha256

KNOWN_POLICY_SCHEMA_ID = "chemworld.known_policy_controls"
KNOWN_POLICY_SCHEMA_VERSION = "0.1.0"
FORMAL_WORLD_SEEDS = (0, 1, 2, 3, 4)
INFORMATION_ARMS = ("opaque_codes", "anonymous_nominal_properties")
POLICY_IDS = ("assay_all", "start_then_discard", "measure_then_threshold")
LIFECYCLES_PER_CELL = 6


@dataclass(frozen=True)
class ProbeCard:
    probe_id: str
    solvent: int
    electrolyte_profile: int
    reagent_amount_mol: float
    potential_V: float
    current_mA: float
    probe_duration_s: float
    post_measure_duration_s: float


# This schedule was constructed from the public operation bounds before known-
# policy execution.  It spans all four nominal categorical controls and a broad,
# monotone range of the continuous controls without approaching runtime limits.
PROBE_SCHEDULE: tuple[ProbeCard, ...] = (
    ProbeCard("probe-01", 0, 0, 0.010, 0.72, 25.0, 300.0, 300.0),
    ProbeCard("probe-02", 1, 1, 0.012, 0.84, 40.0, 420.0, 420.0),
    ProbeCard("probe-03", 2, 2, 0.014, 0.96, 55.0, 540.0, 540.0),
    ProbeCard("probe-04", 3, 3, 0.016, 1.08, 70.0, 660.0, 660.0),
    ProbeCard("probe-05", 0, 2, 0.018, 1.20, 85.0, 780.0, 780.0),
    ProbeCard("probe-06", 2, 0, 0.020, 1.24, 90.0, 900.0, 900.0),
)


def _common_prefix() -> list[dict[str, Any]]:
    return [
        {
            "operation": "add_solvent",
            "volume_L": 0.025,
            "solvent": "$probe.solvent",
        },
        {
            "operation": "add_reagent",
            "amount_mol": "$probe.reagent_amount_mol",
        },
        {
            "operation": "set_potential",
            "potential_V": "$probe.potential_V",
            "current_mA": "$probe.current_mA",
            "electrolyte_profile": "$probe.electrolyte_profile",
        },
        {
            "operation": "electrolyze",
            "duration_s": "$probe.probe_duration_s",
        },
    ]


def build_known_policy_contract() -> dict[str, Any]:
    """Return the threshold-unbound V02 policy-control contract."""

    return {
        "schema_id": KNOWN_POLICY_SCHEMA_ID,
        "schema_version": KNOWN_POLICY_SCHEMA_VERSION,
        "depends_on": {
            "profile_contract_schema": "chemworld.experimental_agency_profile@0.1.0",
            "profile_contract_sha256": profile_contract_sha256(),
        },
        "purpose": {
            "role": "construct-validity positive controls",
            "primary_question": (
                "Does the frozen multidimensional profile recover experimental policies "
                "whose evidence and terminal-decision structures are known by construction?"
            ),
            "not_a_claim_about": [
                "endpoint-performance superiority",
                "chemical intelligence ranking",
                "provider or language-model capability",
                "real-laboratory safety or executability",
            ],
        },
        "formal_matrix": {
            "task_id": "electrochemical-conversion",
            "episode_mode": "campaign",
            "workflow_mode": "autonomous_open_v1",
            "material_family_id": "nominal-prior-latent-v2",
            "world_seeds": list(FORMAL_WORLD_SEEDS),
            "information_arms": list(INFORMATION_ARMS),
            "policy_ids": list(POLICY_IDS),
            "lifecycles_per_cell": LIFECYCLES_PER_CELL,
            "campaign_count": 30,
            "closed_lifecycle_count": 180,
            "provider_call_count": 0,
            "observation_noise_mode": "keyed",
            "matched_arm_rule": (
                "Within a world-policy pair, physics, probe schedule, observation-noise "
                "namespace, and policy code are identical; only the supplied material "
                "information changes. Known policies never read that dossier."
            ),
        },
        "probe_schedule": {
            "construction_rule": (
                "Six deterministic cards chosen from public task-interface bounds before "
                "known-policy outcomes; all policies receive the cards in the same order."
            ),
            "solvent_volume_L_per_started_lifecycle": 0.025,
            "campaign_stock_envelope": {
                "solvent_L": 0.15,
                "reagent_mol_for_full_prefix": 0.09,
            },
            "cards": [asdict(probe) for probe in PROBE_SCHEDULE],
        },
        "policies": [
            {
                "policy_id": "assay_all",
                "policy_role": "terminal-commitment control",
                "reads_material_information": False,
                "reads_observations_for_decisions": False,
                "per_lifecycle_plan": [
                    *_common_prefix(),
                    {"operation": "terminate"},
                    {"operation": "measure", "instrument": "final_assay"},
                ],
                "operation_count_per_lifecycle": 6,
            },
            {
                "policy_id": "start_then_discard",
                "policy_role": "explicit-discard control",
                "reads_material_information": False,
                "reads_observations_for_decisions": False,
                "per_lifecycle_plan": [
                    {
                        "operation": "add_solvent",
                        "volume_L": 0.025,
                        "solvent": "$probe.solvent",
                    },
                    {
                        "operation": "discard_batch",
                        "reason": "known_policy_immediate_discard",
                    },
                ],
                "operation_count_per_lifecycle": 2,
            },
            {
                "policy_id": "measure_then_threshold",
                "policy_role": "evidence-conditioned-action control",
                "reads_material_information": False,
                "reads_observations_for_decisions": True,
                "per_lifecycle_plan": [
                    *_common_prefix(),
                    {"operation": "measure", "instrument": "uvvis"},
                    {
                        "branch": "finite observation.conversion >= $qualified_threshold",
                        "then": [
                            {
                                "operation": "electrolyze",
                                "duration_s": "$probe.post_measure_duration_s",
                            },
                            {"operation": "terminate"},
                            {"operation": "measure", "instrument": "final_assay"},
                        ],
                        "else": [
                            {
                                "operation": "discard_batch",
                                "reason": "known_policy_below_threshold",
                            }
                        ],
                        "missing_or_nonfinite_signal": [
                            {
                                "operation": "discard_batch",
                                "reason": "known_policy_diagnostic_unavailable",
                            }
                        ],
                    },
                ],
                "operation_count_per_lifecycle": {
                    "discard_branch": 6,
                    "assay_branch": 8,
                },
            },
        ],
        "threshold_qualification": {
            "status_after_v02": "unbound_until_W1-V03",
            "diagnostic_operation": {"operation": "measure", "instrument": "uvvis"},
            "diagnostic_signal": "observation.conversion",
            "signal_requirements": ["public", "finite", "scalar"],
            "comparator": ">=",
            "continue_branch": "one additional electrolysis, terminate, final_assay",
            "discard_branch": "discard_batch",
            "qualification_data": "independent qualification worlds only",
            "forbidden_data": "formal world seeds 0, 1, 2, 3, 4",
            "candidate_rule": (
                "midpoints between sorted unique finite qualification signals"
            ),
            "selection_rule": (
                "Among candidates producing both branches in every qualification arm, "
                "choose the candidate closest to the pooled qualification median; break "
                "equal-distance ties toward the lower numeric threshold."
            ),
            "freeze_rule": (
                "W1-V03 must record the selected value, source-manifest hash, and formal-"
                "world exclusion audit before W1-V04 implementation is released."
            ),
        },
        "expected_profile_signatures": {
            "scope": (
                "Campaign profiles are evaluated per world x information arm x policy; "
                "ordering statements involving threshold routing are evaluated over the "
                "pooled formal matrix after the preregistered non-degeneracy gate."
            ),
            "execution_validity_gate": (
                "All planned lifecycles close, every submitted action commits, no action "
                "is validation-failed or resource-rejected, and event/state/resource "
                "replay is exact. Signature recovery is assessed only after this gate."
            ),
            "exact_by_policy": {
                "assay_all": {
                    "closed_lifecycle_fraction": 1.0,
                    "assay_fraction": 1.0,
                    "discard_fraction": 0.0,
                    "measured_lifecycle_fraction": 0.0,
                    "nonfinal_instrument_uses_per_closed_lifecycle": 0.0,
                    "mean_first_measurement_operation_fraction": None,
                    "continued_after_measurement_fraction": 0.0,
                    "post_measure_process_operations_per_closed_lifecycle": 0.0,
                    "threshold_eligible_fraction": 0.0,
                    "threshold_decision_concordance": None,
                    "attempted_operations_per_closed_lifecycle": 6.0,
                    "committed_operations_per_closed_lifecycle": 6.0,
                },
                "start_then_discard": {
                    "closed_lifecycle_fraction": 1.0,
                    "assay_fraction": 0.0,
                    "discard_fraction": 1.0,
                    "measured_lifecycle_fraction": 0.0,
                    "nonfinal_instrument_uses_per_closed_lifecycle": 0.0,
                    "mean_first_measurement_operation_fraction": None,
                    "continued_after_measurement_fraction": 0.0,
                    "post_measure_process_operations_per_closed_lifecycle": 0.0,
                    "threshold_eligible_fraction": 0.0,
                    "threshold_decision_concordance": None,
                    "attempted_operations_per_closed_lifecycle": 2.0,
                    "committed_operations_per_closed_lifecycle": 2.0,
                },
                "measure_then_threshold": {
                    "closed_lifecycle_fraction": 1.0,
                    "measured_lifecycle_fraction": 1.0,
                    "nonfinal_instrument_uses_per_closed_lifecycle": 1.0,
                    "threshold_eligible_fraction": 1.0,
                    "threshold_decision_concordance": 1.0,
                },
            },
            "threshold_policy_algebra": {
                "symbol": (
                    "p = assayed threshold-policy lifecycles / closed "
                    "threshold-policy lifecycles"
                ),
                "domain_after_formal_non_degeneracy_gate": "0 < p < 1",
                "assay_fraction": "p",
                "discard_fraction": "1 - p",
                "continued_after_measurement_fraction": "p",
                "post_measure_process_operations_per_closed_lifecycle": "p",
                "mean_first_measurement_operation_fraction": "2/3 - p/6",
                "attempted_operations_per_closed_lifecycle": "6 + 2p",
                "committed_operations_per_closed_lifecycle": "6 + 2p",
            },
            "formal_non_degeneracy_gate": {
                "unit": "all 60 threshold-policy lifecycles pooled across formal worlds and arms",
                "pass": (
                    "at least one finite-signal assay branch and at least one "
                    "finite-signal discard branch"
                ),
                "failure_handling": (
                    "Report the full frozen result and mark the threshold positive control "
                    "unestablished; never retune on formal worlds."
                ),
            },
            "strict_partial_orderings_after_gate": [
                (
                    "assay_all.assay_fraction > "
                    "measure_then_threshold.assay_fraction > "
                    "start_then_discard.assay_fraction"
                ),
                (
                    "start_then_discard.discard_fraction > "
                    "measure_then_threshold.discard_fraction > "
                    "assay_all.discard_fraction"
                ),
                (
                    "measure_then_threshold.measured_lifecycle_fraction > "
                    "assay_all.measured_lifecycle_fraction = "
                    "start_then_discard.measured_lifecycle_fraction"
                ),
                (
                    "measure_then_threshold."
                    "nonfinal_instrument_uses_per_closed_lifecycle > "
                    "assay_all.nonfinal_instrument_uses_per_closed_lifecycle = "
                    "start_then_discard."
                    "nonfinal_instrument_uses_per_closed_lifecycle"
                ),
                (
                    "measure_then_threshold.continued_after_measurement_fraction > "
                    "assay_all.continued_after_measurement_fraction = "
                    "start_then_discard.continued_after_measurement_fraction"
                ),
                (
                    "measure_then_threshold.attempted_operations_per_closed_lifecycle > "
                    "assay_all.attempted_operations_per_closed_lifecycle > "
                    "start_then_discard.attempted_operations_per_closed_lifecycle"
                ),
            ],
            "resource_expectations": [
                (
                    "start_then_discard uses strictly fewer committed operations and no "
                    "reagent relative to both other policies"
                ),
                (
                    "assay_all and measure_then_threshold use identical solvent and reagent "
                    "schedules before terminal routing"
                ),
                (
                    "observed cost and risk must exactly reconcile to committed action "
                    "paths; no strict cost or risk ordering is asserted between assay_all "
                    "and measure_then_threshold"
                ),
            ],
            "explicit_non_orderings": [
                "mean_assayed_score",
                "best_assayed_score",
                "all outcome_trajectory metrics",
                "cost or risk between assay_all and measure_then_threshold",
            ],
            "conditional_null_expectations": [
                (
                    "mean_first_measurement_operation_fraction is null for both "
                    "unmeasured policies and finite for measure_then_threshold"
                ),
                (
                    "all outcome-trajectory and endpoint-context metrics are null for "
                    "start_then_discard because it performs no final assay"
                ),
                (
                    "assay_all endpoint-context metrics are finite; its recovery rate is "
                    "null exactly when no loss episode exists"
                ),
                (
                    "measure_then_threshold trajectory nullness follows the frozen V01 "
                    "denominator rules using its observed assay count"
                ),
            ],
        },
        "arm_invariance": {
            "expected": (
                "Matched information arms yield identical known-policy actions, threshold "
                "signals, branch choices, profile metrics, and assayed endpoints."
            ),
            "interpretation": (
                "This is an interface and pairing check because the deterministic controls "
                "do not consume material information; it is not a material-information null result."
            ),
        },
        "reliability": {
            "trajectory_exact_replay_match": True,
            "profile_exact_rebuild_match": True,
            "matched_arm_action_match": True,
            "provider_call_count": 0,
            "test_retest_rule": (
                "A second execution from the same world identity, keyed-noise namespace, "
                "policy, and threshold must reproduce event, state, resource, terminal, "
                "profile, and endpoint hashes exactly."
            ),
        },
    }


def known_policy_contract_sha256(payload: dict[str, Any] | None = None) -> str:
    """Return the canonical content hash, excluding any embedded hash field."""

    contract = build_known_policy_contract() if payload is None else dict(payload)
    contract.pop("contract_sha256", None)
    encoded = json.dumps(
        contract,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_known_policy_contract(payload: dict[str, Any] | None = None) -> list[str]:
    """Return deterministic semantic validation errors for the V02 contract."""

    contract = build_known_policy_contract() if payload is None else payload
    errors: list[str] = []
    if contract.get("schema_id") != KNOWN_POLICY_SCHEMA_ID:
        errors.append("schema_id does not match the known-policy contract")
    if contract.get("schema_version") != KNOWN_POLICY_SCHEMA_VERSION:
        errors.append("schema_version does not match the known-policy contract")
    depends_on = contract.get("depends_on", {})
    if depends_on.get("profile_contract_sha256") != profile_contract_sha256():
        errors.append("profile-contract hash binding is stale")

    matrix = contract.get("formal_matrix", {})
    if matrix.get("campaign_count") != (
        len(FORMAL_WORLD_SEEDS) * len(INFORMATION_ARMS) * len(POLICY_IDS)
    ):
        errors.append("formal campaign count does not match the factorial matrix")
    if matrix.get("closed_lifecycle_count") != (
        matrix.get("campaign_count", 0) * LIFECYCLES_PER_CELL
    ):
        errors.append("formal lifecycle count does not match the matrix")
    if matrix.get("provider_call_count") != 0:
        errors.append("known policies must make zero provider calls")

    schedule = contract.get("probe_schedule", {})
    cards = schedule.get("cards", [])
    if len(cards) != LIFECYCLES_PER_CELL:
        errors.append("probe schedule must contain exactly six cards")
    if len({card.get("probe_id") for card in cards}) != len(cards):
        errors.append("probe identifiers must be unique")
    for card in cards:
        for field in (
            "reagent_amount_mol",
            "potential_V",
            "current_mA",
            "probe_duration_s",
            "post_measure_duration_s",
        ):
            value = card.get(field)
            if isinstance(value, bool) or not isinstance(value, int | float):
                errors.append(f"{card.get('probe_id')}.{field} must be numeric")
            elif not math.isfinite(float(value)):
                errors.append(f"{card.get('probe_id')}.{field} must be finite")
        if card.get("solvent") not in range(4):
            errors.append(f"{card.get('probe_id')}.solvent is outside the public choices")
        if card.get("electrolyte_profile") not in range(4):
            errors.append(
                f"{card.get('probe_id')}.electrolyte_profile is outside the public choices"
            )
        bounded = (
            ("reagent_amount_mol", 0.003, 0.030),
            ("potential_V", 0.65, 1.25),
            ("current_mA", 15.0, 90.0),
            ("probe_duration_s", 180.0, 900.0),
            ("post_measure_duration_s", 300.0, 3600.0),
        )
        for field, lower, upper in bounded:
            value = card.get(field)
            if (
                isinstance(value, int | float)
                and not isinstance(value, bool)
                and not lower <= float(value) <= upper
            ):
                errors.append(f"{card.get('probe_id')}.{field} is outside bounds")

    expected_solvent = 0.025 * len(cards)
    expected_reagent = sum(float(card["reagent_amount_mol"]) for card in cards)
    envelope = schedule.get("campaign_stock_envelope", {})
    if not math.isclose(envelope.get("solvent_L", -1), expected_solvent, abs_tol=1e-12):
        errors.append("solvent envelope does not match the six-card schedule")
    if not math.isclose(
        envelope.get("reagent_mol_for_full_prefix", -1),
        expected_reagent,
        abs_tol=1e-12,
    ):
        errors.append("reagent envelope does not match the six-card schedule")

    policies = contract.get("policies", [])
    if tuple(policy.get("policy_id") for policy in policies) != POLICY_IDS:
        errors.append("policy identities or order do not match the frozen contract")
    for policy in policies:
        if policy.get("reads_material_information") is not False:
            errors.append(f"{policy.get('policy_id')} must ignore material information")

    threshold = contract.get("threshold_qualification", {})
    if threshold.get("status_after_v02") != "unbound_until_W1-V03":
        errors.append("V02 must not bind a threshold value")
    if threshold.get("forbidden_data") != "formal world seeds 0, 1, 2, 3, 4":
        errors.append("formal-world exclusion is not frozen")

    signatures = contract.get("expected_profile_signatures", {})
    exact = signatures.get("exact_by_policy", {})
    if exact.get("assay_all", {}).get("assay_fraction") != 1.0:
        errors.append("assay_all assay identity is not frozen")
    if exact.get("start_then_discard", {}).get("discard_fraction") != 1.0:
        errors.append("start_then_discard discard identity is not frozen")
    if (
        exact.get("measure_then_threshold", {}).get(
            "threshold_decision_concordance"
        )
        != 1.0
    ):
        errors.append("threshold-policy concordance identity is not frozen")
    if len(signatures.get("strict_partial_orderings_after_gate", [])) != 6:
        errors.append("six preregistered profile orderings are required")

    reliability = contract.get("reliability", {})
    if reliability.get("provider_call_count") != 0:
        errors.append("reliability contract must require zero provider calls")
    for field in (
        "trajectory_exact_replay_match",
        "profile_exact_rebuild_match",
        "matched_arm_action_match",
    ):
        if reliability.get(field) is not True:
            errors.append(f"reliability.{field} must be true")
    return errors


__all__ = [
    "FORMAL_WORLD_SEEDS",
    "INFORMATION_ARMS",
    "KNOWN_POLICY_SCHEMA_ID",
    "KNOWN_POLICY_SCHEMA_VERSION",
    "LIFECYCLES_PER_CELL",
    "POLICY_IDS",
    "PROBE_SCHEDULE",
    "ProbeCard",
    "build_known_policy_contract",
    "known_policy_contract_sha256",
    "validate_known_policy_contract",
]
