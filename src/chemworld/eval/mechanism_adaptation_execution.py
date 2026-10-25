"""Executable orchestration for the v0.2 mechanism-adaptation benchmark.

The module has two deliberately separate jobs:

* produce Gate A from environment observations without making provider calls; and
* expand and execute the frozen paired campaign matrix with durable per-row artifacts.

Gate A uses only numeric values available in the public observation/reward stream.  It
does not train or update an evaluated Agent, and it does not expose hidden candidate
IDs to an Agent.  The Gaussian oracle is an evaluator-side diagnostic used to decide
whether the environment contains enough information at the frozen budget.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from collections.abc import Callable, Collection, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, cast

import numpy as np

from chemworld.agents.mechanism_adaptation_live_llm import (
    CandidateLabelMode,
    MechanismAdaptationLiveLLMAgent,
    MechanismCandidateSpec,
)
from chemworld.agents.stateful_scientific import (
    StatefulScientificMechanismAgent,
)
from chemworld.agents.task_recipes import (
    DIAGNOSTIC_RECIPE_DESIGN_V1,
    diagnostic_task_recipe_vectors,
    task_recipe_event_count,
    task_recipe_from_unit_vector,
)
from chemworld.envs.chemworld_env import ChemWorldEnv
from chemworld.eval.flagship_diagnostics import (
    ContinuingPublicViewAgent,
    FeedbackCondition,
    run_two_phase_campaign,
)
from chemworld.eval.mechanism_adaptation import (
    PRIMARY_CHANGE_TRACK_ID,
    GaussianMechanismOracle,
    build_paired_campaign_matrix,
    change_detection_summary,
    evaluation_track,
    identifiability_certificate,
    load_mechanism_adaptation_protocol,
    reference_sufficiency_certificate,
    validate_mechanism_adaptation_protocol,
    wilson_interval,
)
from chemworld.eval.mechanism_design_audit import (
    audit_mechanism_design,
    declared_relational_action_groups,
    relational_coverage_witness,
)
from chemworld.eval.mechanism_gate_decision import (
    ONLINE_ATTAINABILITY_CERTIFICATE_VERSION,
    gate_a_certificate_decision,
    gate_a_execution_contract_binding,
)
from chemworld.eval.mechanism_preregistration import (
    validate_mechanism_preregistration,
)
from chemworld.eval.mechanism_relation_graph import (
    validate_diagnostic_relation_graph,
)
from chemworld.eval.provenance import (
    canonical_json_sha256 as canonical_sha256,
)
from chemworld.eval.trial_store import (
    ConfirmatoryTrialKey,
    ConfirmatoryTrialStore,
    execute_jobs_resumable,
)
from chemworld.physchem.mechanism_library import configuration_root
from chemworld.providers.deepseek import DeepSeekClient
from chemworld.tasks import get_task

DEFAULT_PROTOCOL_PATH = (
    configuration_root() / "benchmark/mechanism_adaptation_v0.3.0_rc28.json"
)
DEFAULT_GATE_A_PLAN_PATH = (
    configuration_root()
    / "benchmark/mechanism_adaptation_gate_a_v0.3.0_rc28.json"
)
DEFAULT_LLM_METHODS_PATH = (
    configuration_root() / "methods/llm_v0.4/llm_methods_rc25.json"
)
EXECUTION_SCHEMA_VERSION = "chemworld-mechanism-adaptation-execution-0.2"
GATE_A_REPORT_VERSION = "chemworld-mechanism-adaptation-gate-a-report-0.3.0"
PUBLIC_EXPERIMENT_FEATURE_ENCODING_VERSION = (
    "chemworld-public-experiment-finite-first-last-range-0.1"
)
PUBLIC_EXPERIMENT_FEATURE_BLOCK_SIZE = 4

_CRITICAL_INSTRUMENTS = {
    "reaction-to-crystallization": "hplc",
    "electrochemical-conversion": "uvvis",
}


def load_json_object(path: str | Path) -> dict[str, Any]:
    """Load a JSON object and reject ambiguous non-object roots."""

    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    if payload.get("schema_version") == "chemworld-json-composition-0.1":
        extends = payload.get("extends")
        expected_base_sha = str(payload.get("base_sha256", "")).lower()
        overrides = payload.get("overrides")
        remove_paths = payload.get("remove_paths", [])
        if (
            not isinstance(extends, str)
            or not extends
            or not isinstance(overrides, Mapping)
            or not isinstance(remove_paths, list)
        ):
            raise ValueError("invalid JSON composition")
        base_path = (source.parent / extends).resolve()
        if source.parent.resolve() not in base_path.parents:
            raise ValueError("JSON composition base must stay in its source directory")
        observed_base_sha = hashlib.sha256(base_path.read_bytes()).hexdigest()
        if observed_base_sha != expected_base_sha:
            raise ValueError(
                "JSON composition base hash mismatch: "
                f"expected {expected_base_sha}, observed {observed_base_sha}"
            )
        resolved = _deep_merge_json(load_json_object(base_path), overrides)
        for raw_path in remove_paths:
            if not isinstance(raw_path, str) or not raw_path:
                raise ValueError("JSON composition remove_paths must be dotted strings")
            _remove_json_path(resolved, raw_path)
        resolved["plan_composition"] = {
            "schema_version": payload["schema_version"],
            "source": source.name,
            "extends": base_path.name,
            "base_sha256": observed_base_sha,
            "remove_paths": list(remove_paths),
        }
        return resolved
    return payload


def _deep_merge_json(
    base: Mapping[str, Any],
    overrides: Mapping[str, Any],
) -> dict[str, Any]:
    result = dict(base)
    for key, value in overrides.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[str(key)] = _deep_merge_json(
                cast(Mapping[str, Any], result[str(key)]),
                value,
            )
        else:
            result[str(key)] = value
    return result


def _remove_json_path(payload: dict[str, Any], dotted_path: str) -> None:
    parts = dotted_path.split(".")
    target = payload
    for part in parts[:-1]:
        value = target.get(part)
        if not isinstance(value, dict):
            raise ValueError(f"JSON composition removal path missing: {dotted_path}")
        target = value
    if parts[-1] not in target:
        raise ValueError(f"JSON composition removal path missing: {dotted_path}")
    del target[parts[-1]]


def load_protocol_object(path: str | Path) -> dict[str, Any]:
    """Load a resolved, hash-pinned mechanism-adaptation protocol."""

    return load_mechanism_adaptation_protocol(path)


def build_action_library(
    task_id: str,
    *,
    action_count: int,
    seed: int,
    design_id: str = DIAGNOSTIC_RECIPE_DESIGN_V1,
) -> dict[str, np.ndarray]:
    """Return a deterministic bounded set of public complete-recipe designs."""

    if action_count < 3:
        raise ValueError("Gate A requires at least three public actions per task")
    task_info = get_task(task_id).to_dict()
    task_seed = _stable_seed(seed, task_id)
    rng = np.random.default_rng(task_seed)
    vectors = diagnostic_task_recipe_vectors(
        task_info,
        action_count=action_count,
        rng=rng,
        design_id=design_id,
    )
    return {
        f"design-{index:02d}": np.asarray(vector, dtype=float)
        for index, vector in enumerate(vectors)
    }


def _canonical_policy_cycle(
    *,
    canonical_action_ids: Sequence[str],
    ranked_action_ids: Sequence[str],
    action_count: int,
) -> list[str]:
    """Return an information-selected action set in frozen library order.

    Predictive-fit information determines which actions enter the policy, but
    Monte Carlo ranking noise must not also determine their cyclic execution
    order.  The latter changes which references exist when a hidden change
    occurs and is therefore part of the protocol, not an incidental fit
    outcome.
    """

    canonical = [str(item) for item in canonical_action_ids]
    ranked = [str(item) for item in ranked_action_ids]
    if action_count <= 0 or action_count > len(canonical):
        raise ValueError("policy action count must be within the action library")
    if len(set(canonical)) != len(canonical):
        raise ValueError("canonical action ids must be unique")
    unknown = sorted(set(ranked) - set(canonical))
    if unknown:
        raise ValueError(f"ranked policy actions are outside the action library: {unknown}")
    ranked_unique = list(dict.fromkeys(ranked))
    if len(ranked_unique) < action_count:
        raise ValueError("ranked policy actions do not cover the requested action count")
    selected = set(ranked_unique[:action_count])
    return [action_id for action_id in canonical if action_id in selected]


def _validate_online_policy_action_count(
    *,
    action_count: int,
    gate_budget: int,
) -> None:
    """Allow a fixed phase-invariant action set to cycle over a longer horizon."""

    if action_count <= 0 or action_count > gate_budget:
        raise ValueError(
            "online policy action coverage must be positive and no greater "
            "than its controlling budget"
        )


def encode_public_experiment_trace(
    trace: Sequence[tuple[Mapping[str, Any], float]],
) -> list[float]:
    """Summarize one public experiment without pseudo-replicating state snapshots.

    The environment returns a fixed public observation after every operation.  Most
    state fields persist across several later operations, so concatenating every
    snapshot makes one physical signal appear as many independent Gaussian features.
    For each public scalar this encoder emits one fixed four-value block:

    ``ever_finite, first_finite, last_finite, finite_range``.

    The same block is emitted for the public reward sequence.  No task names, hidden
    truth, operation labels, evaluator-only outcomes, or hand-selected metrics enter
    the representation.
    """

    if not trace:
        raise ValueError("a public experiment trace cannot be empty")
    observations = [observation for observation, _reward in trace]
    keys = sorted(observations[0])
    if any(sorted(observation) != keys for observation in observations[1:]):
        raise ValueError("public observation keys changed within one experiment")

    features: list[float] = []
    for key in keys:
        arrays = [
            np.asarray(observation[key], dtype=float).reshape(-1) for observation in observations
        ]
        shapes = {array.shape for array in arrays}
        if len(shapes) != 1:
            raise ValueError(
                f"public observation field {key!r} changed shape within one experiment"
            )
        stacked = np.stack(arrays, axis=0)
        for column in stacked.T:
            features.extend(_finite_series_summary(column))

    rewards = np.asarray([float(reward) for _observation, reward in trace])
    features.extend(_finite_series_summary(rewards))
    return features


def _finite_series_summary(values: np.ndarray) -> list[float]:
    finite_values = np.asarray(values, dtype=float).reshape(-1)
    finite_values = finite_values[np.isfinite(finite_values)]
    if finite_values.size == 0:
        return [0.0, 0.0, 0.0, 0.0]
    return [
        1.0,
        float(finite_values[0]),
        float(finite_values[-1]),
        float(np.max(finite_values) - np.min(finite_values)),
    ]


class PublicCampaignObservationSession:
    """Execute complete public recipes in one nuisance-consistent campaign world."""

    def __init__(
        self,
        *,
        task_id: str,
        seed: int,
        interventions: Sequence[Mapping[str, Any]],
        action_library: Mapping[str, np.ndarray],
        experiment_horizon: int,
        observation_seed: int | None = None,
        observation_noise_mode: str = "keyed",
        observation_noise_namespace: str = "mechanism-adaptation-confirmatory",
    ) -> None:
        if experiment_horizon <= 0:
            raise ValueError("experiment_horizon must be positive")
        self.task_id = task_id
        self.task_info = get_task(task_id).to_dict()
        self.action_library = action_library
        per_experiment = task_recipe_event_count(self.task_info)
        self.environment = ChemWorldEnv(
            task_id=task_id,
            seed=int(seed),
            episode_mode_override="campaign",
            budget_override=(per_experiment + 1) * int(experiment_horizon),
            observation_seed_override=observation_seed,
            observation_noise_mode=observation_noise_mode,
            observation_noise_namespace=observation_noise_namespace,
            world_interventions=tuple(dict(item) for item in interventions),
        )
        self.environment.reset(seed=int(seed))
        self.observation_noise_audit: list[dict[str, Any]] = []

    def observe(self, action_id: str) -> list[float]:
        try:
            vector = self.action_library[action_id]
        except KeyError as error:
            raise ValueError(f"unknown public action ID: {action_id}") from error
        recipe = task_recipe_from_unit_vector(self.task_info, vector)
        trace: list[tuple[Mapping[str, Any], float]] = []
        operation_noise: list[dict[str, Any]] = []
        experiment_ended = False
        for action in recipe["steps"]:
            observation, reward, _terminated, _truncated, info = self.environment.step(action)
            trace.append((observation, float(reward)))
            operation_noise.append(
                self.environment.observation_noise_provenance()
            )
            experiment_ended = bool(info.get("experiment_ended"))
        if not experiment_ended or info.get("transaction_status") != "committed":
            raise RuntimeError("compiled Gate A recipe did not produce a committed experiment")
        self.observation_noise_audit.append(
            {
                "experiment_index": len(self.observation_noise_audit),
                "action_id": action_id,
                "operation_noise": operation_noise,
                "operation_noise_bundle_sha256": canonical_sha256(
                    operation_noise
                ),
            }
        )
        return encode_public_experiment_trace(trace)

    def close(self) -> None:
        self.environment.close()

    def __enter__(self) -> PublicCampaignObservationSession:
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()


def _sample_experiment_job(job: Mapping[str, Any]) -> list[float]:
    action_id = str(job["action_id"])
    action_library = {action_id: np.asarray(job["action_vector"], dtype=float)}
    with PublicCampaignObservationSession(
        task_id=str(job["task_id"]),
        seed=int(job["seed"]),
        interventions=job["interventions"],
        action_library=action_library,
        experiment_horizon=1,
        observation_seed=(
            None if job.get("observation_seed") is None else int(job["observation_seed"])
        ),
    ) as session:
        return session.observe(action_id)


def _sample_paired_contrast_job(job: Mapping[str, Any]) -> list[float]:
    """Return post-minus-pre public traces for one matched recipe batch."""

    action_ids = [str(item) for item in job["action_ids"]]
    raw_feature_indices = job.get("feature_indices")
    feature_indices = (
        None
        if raw_feature_indices is None
        else {
            str(action_id): np.asarray(indices, dtype=int)
            for action_id, indices in cast(Mapping[str, Sequence[int]], raw_feature_indices).items()
        }
    )
    if feature_indices is not None and not set(action_ids).issubset(feature_indices):
        raise ValueError("paired contrast feature selection must cover every batch action")
    action_library = {
        str(key): np.asarray(value, dtype=float) for key, value in job["action_library"].items()
    }
    with PublicCampaignObservationSession(
        task_id=str(job["task_id"]),
        seed=int(job["world_seed"]),
        interventions=(),
        action_library=action_library,
        experiment_horizon=len(action_ids),
        observation_seed=int(job["pre_observation_seed"]),
    ) as pre_session:
        pre = [np.asarray(pre_session.observe(action_id), dtype=float) for action_id in action_ids]
    with PublicCampaignObservationSession(
        task_id=str(job["task_id"]),
        seed=int(job["world_seed"]),
        interventions=job["interventions"],
        action_library=action_library,
        experiment_horizon=len(action_ids),
        observation_seed=int(job["post_observation_seed"]),
    ) as post_session:
        post = [
            np.asarray(post_session.observe(action_id), dtype=float) for action_id in action_ids
        ]
    contrasts: list[np.ndarray] = []
    for action_id, pre_values, post_values in zip(action_ids, pre, post, strict=True):
        contrast = post_values - pre_values
        if feature_indices is not None:
            try:
                indices = feature_indices[action_id]
            except KeyError as error:
                raise ValueError(
                    "paired contrast feature selection must cover every batch action"
                ) from error
            if (
                indices.ndim != 1
                or indices.size == 0
                or np.any(indices < 0)
                or np.any(indices >= contrast.size)
            ):
                raise ValueError("paired contrast selected feature indices are invalid")
            contrast = contrast[indices]
        contrasts.append(contrast)
    return np.concatenate(contrasts).tolist()


def _online_evidence_channel(encoding: str, action_id: str) -> str:
    if encoding not in {
        "transition_contrast",
        "stable_contrast",
        "absolute_trace",
    }:
        raise ValueError(f"unsupported online evidence encoding: {encoding}")
    return f"{encoding}\u241f{action_id}"


def _online_relational_evidence_channel(
    encoding: str,
    action_id: str,
    reference_action_id: str,
) -> str:
    """Name one ordered cross-action contrast without task-specific semantics."""

    if encoding not in {
        "relational_transition_contrast",
        "relational_stable_contrast",
    }:
        raise ValueError(f"unsupported online relational evidence encoding: {encoding}")
    if action_id == reference_action_id:
        raise ValueError("relational evidence requires two distinct public actions")
    return f"{encoding}\u241f{action_id}\u241f{reference_action_id}"


def _online_hypothesis_evidence_channel(
    hypothesis: tuple[str, int | None],
    action_id: str,
    *,
    reference_action_id: str | None = None,
    reference_experiment_index: int | None,
    current_experiment_index: int,
) -> tuple[str, str]:
    """Select the predictive channel implied by one latent change-point state."""

    candidate_id, change_after_experiment = hypothesis
    if reference_experiment_index is None:
        encoding = "absolute_trace"
        channel = _online_evidence_channel(encoding, action_id)
    elif (
        candidate_id != "no_change"
        and change_after_experiment is not None
        and reference_experiment_index <= change_after_experiment < current_experiment_index
    ):
        if reference_action_id is not None and reference_action_id != action_id:
            channel = _online_relational_evidence_channel(
                "relational_transition_contrast",
                action_id,
                reference_action_id,
            )
        else:
            channel = _online_evidence_channel("transition_contrast", action_id)
    else:
        if reference_action_id is not None and reference_action_id != action_id:
            channel = _online_relational_evidence_channel(
                "relational_stable_contrast",
                action_id,
                reference_action_id,
            )
        else:
            channel = _online_evidence_channel("stable_contrast", action_id)
    return candidate_id, channel


def _advance_online_change_point_hypotheses(
    posterior: Mapping[tuple[str, int | None], float],
    *,
    change_after_experiment: int,
    allowed_change_times: Sequence[int],
    candidate_ids: Sequence[str],
    hazard: float | Mapping[str, float],
) -> dict[tuple[str, int | None], float]:
    """Expand only protocol-allowed, absorbing latent change-point states."""

    updated = {
        hypothesis: float(probability)
        for hypothesis, probability in posterior.items()
        if float(probability) > 0.0
    }
    if change_after_experiment not in set(allowed_change_times):
        return updated
    point_hazard = (
        float(hazard[str(change_after_experiment)])
        if isinstance(hazard, Mapping)
        else float(hazard)
    )
    if not 0.0 <= point_hazard <= 1.0:
        raise ValueError("online change-point hazard must be in [0, 1]")
    no_change = ("no_change", None)
    no_change_probability = updated.get(no_change, 0.0)
    updated[no_change] = (1.0 - point_hazard) * no_change_probability
    changed_ids = [candidate_id for candidate_id in candidate_ids if candidate_id != "no_change"]
    if not changed_ids:
        raise ValueError("online hidden-change filtering requires changed states")
    for candidate_id in changed_ids:
        hypothesis = (candidate_id, change_after_experiment)
        updated[hypothesis] = (
            updated.get(hypothesis, 0.0)
            + point_hazard * no_change_probability / len(changed_ids)
        )
    total = sum(updated.values())
    if total <= 0.0:
        raise ValueError("online change-point hypothesis mass vanished")
    return {
        hypothesis: probability / total
        for hypothesis, probability in updated.items()
        if probability > 0.0
    }


def _online_family_posterior(
    hypothesis_posterior: Mapping[tuple[str, int | None], float],
    candidate_ids: Sequence[str],
) -> dict[str, float]:
    result = dict.fromkeys(candidate_ids, 0.0)
    for (candidate_id, _change_time), probability in hypothesis_posterior.items():
        result[candidate_id] += float(probability)
    total = sum(result.values())
    if total <= 0.0:
        raise ValueError("online family posterior mass vanished")
    return {candidate_id: probability / total for candidate_id, probability in result.items()}


def _online_predictive_lookup(
    payload: Mapping[str, Mapping[str, Any]],
) -> dict[tuple[str, str], tuple[np.ndarray, np.ndarray]]:
    predictives: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
    for item in payload.values():
        candidate_id = str(item["candidate_id"])
        action_id = str(item["action_id"])
        mean = np.asarray(item["mean"], dtype=float)
        variance = np.asarray(item["variance"], dtype=float)
        if mean.ndim != 1 or mean.shape != variance.shape:
            raise ValueError("online predictive dimensions are inconsistent")
        predictives[(candidate_id, action_id)] = (
            mean,
            np.maximum(variance, 1e-12),
        )
    return predictives


def _relational_evidence_samples(
    *,
    candidate_ids: Sequence[str],
    action_ids: Sequence[str],
    transition_samples: Mapping[tuple[str, str], Sequence[Sequence[float]]],
    stable_samples: Mapping[tuple[str, str], Sequence[Sequence[float]]],
    absolute_samples: Mapping[tuple[str, str], Sequence[Sequence[float]]],
) -> dict[tuple[str, str], list[list[float]]]:
    """Derive ordered cross-action contrasts from nuisance-aligned fit samples.

    For a current action ``a`` and an earlier reference action ``b``, the
    transition channel models ``post(a) - pre(b)`` while the stable channel
    models ``post(a) - post_reference(b)``.  The component arrays are reconstructed
    from the already sampled absolute and same-action contrasts, so enabling
    relational evidence does not create a second fit namespace or inspect held-out
    outcomes.
    """

    result: dict[tuple[str, str], list[list[float]]] = {}
    for candidate_id in candidate_ids:
        for action_id in action_ids:
            current_absolute = absolute_samples[(candidate_id, action_id)]
            for reference_action_id in action_ids:
                if reference_action_id == action_id:
                    continue
                reference_absolute = absolute_samples[(candidate_id, reference_action_id)]
                reference_transition = transition_samples[
                    (candidate_id, reference_action_id)
                ]
                reference_stable = stable_samples[(candidate_id, reference_action_id)]
                sample_count = len(current_absolute)
                if (
                    len(reference_absolute) != sample_count
                    or len(reference_transition) != sample_count
                    or len(reference_stable) != sample_count
                ):
                    raise ValueError("relational fit samples are not nuisance-aligned")
                transition_rows: list[list[float]] = []
                stable_rows: list[list[float]] = []
                for current, reference_post, delta, stable_delta in zip(
                    current_absolute,
                    reference_absolute,
                    reference_transition,
                    reference_stable,
                    strict=True,
                ):
                    current_values = np.asarray(current, dtype=float)
                    reference_post_values = np.asarray(reference_post, dtype=float)
                    transition_values = np.asarray(delta, dtype=float)
                    stable_values = np.asarray(stable_delta, dtype=float)
                    if not (
                        current_values.shape
                        == reference_post_values.shape
                        == transition_values.shape
                        == stable_values.shape
                    ):
                        raise ValueError("relational fit sample dimensions are inconsistent")
                    reference_pre = reference_post_values - transition_values
                    reference_stable_post = reference_post_values - stable_values
                    transition_rows.append((current_values - reference_pre).tolist())
                    stable_rows.append(
                        (current_values - reference_stable_post).tolist()
                    )
                result[
                    (
                        candidate_id,
                        _online_relational_evidence_channel(
                            "relational_transition_contrast",
                            action_id,
                            reference_action_id,
                        ),
                    )
                ] = transition_rows
                result[
                    (
                        candidate_id,
                        _online_relational_evidence_channel(
                            "relational_stable_contrast",
                            action_id,
                            reference_action_id,
                        ),
                    )
                ] = stable_rows
    return result


def _declared_relational_action_groups(
    *,
    task_id: str,
    contract: Mapping[str, Any],
    action_library: Mapping[str, np.ndarray],
) -> dict[str, tuple[tuple[str, ...], ...]]:
    """Resolve every declared diagnostic relation to public action groups."""

    task_info = get_task(task_id).to_dict()
    recipes = {
        action_id: task_recipe_from_unit_vector(task_info, vector)
        for action_id, vector in action_library.items()
    }
    try:
        return declared_relational_action_groups(
            recipes=recipes,
            contract=contract,
        )
    except ValueError as exc:
        raise ValueError(
            f"invalid declared relational action groups for {task_id}: {exc}"
        ) from exc


def _relation_channel_information(
    evidence_information: Mapping[str, float],
    *,
    action_id: str,
    reference_action_id: str,
) -> float:
    """Use the weaker fit-only channel as a robust unknown-change-time score."""

    transition = float(
        evidence_information[
            _online_relational_evidence_channel(
                "relational_transition_contrast",
                action_id,
                reference_action_id,
            )
        ]
    )
    stable = float(
        evidence_information[
            _online_relational_evidence_channel(
                "relational_stable_contrast",
                action_id,
                reference_action_id,
            )
        ]
    )
    return min(transition, stable)


def _select_relational_coverage_actions(
    *,
    canonical_action_ids: Sequence[str],
    declaration_groups: Mapping[str, Sequence[Sequence[str]]],
    evidence_information: Mapping[str, float],
    action_count: int,
) -> tuple[list[str], dict[str, Any]]:
    """Select one fit-ranked public relation group per declared intervention."""

    canonical = [str(item) for item in canonical_action_ids]
    canonical_position = {
        action_id: index for index, action_id in enumerate(canonical)
    }
    if not declaration_groups:
        return [], {
            "required": False,
            "declarations": {},
            "selected_groups": {},
            "selected_action_ids": [],
        }
    declaration_ids = sorted(declaration_groups)
    choices = [tuple(tuple(str(item) for item in group) for group in declaration_groups[key])
               for key in declaration_ids]
    candidates: list[
        tuple[float, int, tuple[str, ...], tuple[tuple[str, ...], ...]]
    ] = []
    for selected_groups in itertools.product(*choices):
        selected_set = {
            action_id for group in selected_groups for action_id in group
        }
        if len(selected_set) > action_count:
            continue
        score = 0.0
        for group in selected_groups:
            unknown_actions = sorted(set(group) - set(canonical))
            if unknown_actions:
                raise ValueError(
                    "relational action group contains unknown actions: "
                    + ", ".join(unknown_actions)
                )
            ordered_scores = [
                _relation_channel_information(
                    evidence_information,
                    action_id=action_id,
                    reference_action_id=reference_action_id,
                )
                for action_id, reference_action_id in itertools.permutations(group, 2)
                if canonical_position[reference_action_id]
                < canonical_position[action_id]
            ]
            if not ordered_scores:
                raise ValueError(
                    "a relational action group requires at least two distinct actions "
                    "with a realizable canonical cold-start direction"
                )
            score += max(ordered_scores)
        ordered_actions = tuple(item for item in canonical if item in selected_set)
        candidates.append(
            (-score, len(selected_set), ordered_actions, selected_groups)
        )
    if not candidates:
        raise ValueError(
            "online policy action count cannot cover every declared relational intervention"
        )
    _negative_score, _size, selected_actions, selected_groups = min(candidates)
    return list(selected_actions), {
        "required": True,
        "declarations": {
            key: [list(group) for group in declaration_groups[key]]
            for key in declaration_ids
        },
        "selected_groups": {
            key: list(group)
            for key, group in zip(declaration_ids, selected_groups, strict=True)
        },
        "selected_action_ids": list(selected_actions),
        "selection_rule": (
            "one_group_per_declared_diagnostic_relation_maximizing_the_minimum_of_"
            "transition_and_stable_fit_only_information_in_a_canonical_cold_start_"
            "direction_then_minimizing_action_union"
        ),
    }


def _relationally_covered_batch_ids(
    *,
    batches: Mapping[str, Sequence[str]],
    declaration_groups: Mapping[str, Sequence[Sequence[str]]],
) -> list[str]:
    """Return controlled batches that close one public relation per declaration."""

    eligible: list[str] = []
    for batch_id, raw_action_ids in batches.items():
        action_ids = {str(item) for item in raw_action_ids}
        covers_all = all(
            any(
                {str(item) for item in group}.issubset(action_ids)
                for group in groups
            )
            for groups in declaration_groups.values()
        )
        if covers_all:
            eligible.append(str(batch_id))
    return sorted(eligible)


def _relational_reference_priorities(
    *,
    action_ids: Sequence[str],
    evidence_information: Mapping[str, float],
) -> dict[str, list[str]]:
    """Pre-rank ordered relation references using fit-only information."""

    result: dict[str, list[str]] = {}
    for action_id in action_ids:
        result[str(action_id)] = sorted(
            (str(item) for item in action_ids if item != action_id),
            key=lambda reference_action_id: (
                -_relation_channel_information(
                    evidence_information,
                    action_id=str(action_id),
                    reference_action_id=reference_action_id,
                ),
                reference_action_id,
            ),
        )
    return result


def _declared_relational_reference_actions(
    *,
    action_ids: Sequence[str],
    selected_groups: Mapping[str, Sequence[str]],
    evidence_information: Mapping[str, float],
) -> dict[str, list[str]]:
    """Prioritize exact declared intervention partners within the policy set."""

    policy_actions = {str(item) for item in action_ids}
    by_action: dict[str, set[str]] = {
        str(action_id): set() for action_id in action_ids
    }
    for raw_group in selected_groups.values():
        group = [str(item) for item in raw_group]
        if not set(group).issubset(policy_actions):
            raise ValueError("selected relational group is outside the policy action set")
        for action_id, reference_action_id in itertools.permutations(group, 2):
            by_action[action_id].add(reference_action_id)
    return {
        action_id: sorted(
            references,
            key=lambda reference_action_id: (
                -_relation_channel_information(
                    evidence_information,
                    action_id=action_id,
                    reference_action_id=reference_action_id,
                ),
                reference_action_id,
            ),
        )
        for action_id, references in by_action.items()
    }


def _select_online_reference_action(
    *,
    action_id: str,
    available_reference_action_ids: Collection[str],
    relational_evidence_enabled: bool,
    declared_relational_references: Sequence[str],
    fit_ranked_relational_references: Sequence[str],
) -> tuple[str | None, str]:
    """Choose one non-duplicated evidence reference for the current experiment."""

    available = {str(item) for item in available_reference_action_ids}
    if action_id in available:
        return action_id, "same_action_temporal"
    if relational_evidence_enabled:
        for reference_action_id in declared_relational_references:
            if reference_action_id in available:
                return str(reference_action_id), "declared_relational"
        for reference_action_id in fit_ranked_relational_references:
            if reference_action_id in available:
                return str(reference_action_id), "fit_ranked_relational"
    return None, "absolute"


def _select_discriminative_feature_blocks(
    samples: Mapping[tuple[str, str], Sequence[Sequence[float]]],
    *,
    candidate_ids: Sequence[str],
    evidence_action_ids: Sequence[str],
    block_size: int,
    block_count: int,
    variance_floor: float,
) -> tuple[dict[str, tuple[int, ...]], dict[str, dict[str, Any]]]:
    """Select high-signal public feature blocks using predictive-fit samples only.

    Selection is task-agnostic.  For each evidence channel, blocks are ranked by the
    largest between-candidate to pooled-within-candidate variance ratio among their
    coordinates.  Held-out observations never participate in selection.
    """

    if block_size <= 0 or block_count <= 0:
        raise ValueError("online feature block dimensions must be positive")
    if not np.isfinite(variance_floor) or variance_floor <= 0.0:
        raise ValueError("online feature selection variance floor must be positive")
    feature_indices: dict[str, tuple[int, ...]] = {}
    reports: dict[str, dict[str, Any]] = {}
    for evidence_action_id in evidence_action_ids:
        candidate_arrays: list[np.ndarray] = []
        for candidate_id in candidate_ids:
            values = np.asarray(
                samples[(str(candidate_id), str(evidence_action_id))],
                dtype=float,
            )
            if values.ndim != 2 or values.shape[0] < 4:
                raise ValueError(
                    "online feature selection requires at least four samples "
                    "per candidate/evidence channel"
                )
            candidate_arrays.append(values)
        dimensions = {array.shape[1] for array in candidate_arrays}
        if len(dimensions) != 1:
            raise ValueError("online feature selection dimensions are inconsistent")
        dimension = dimensions.pop()
        if dimension % block_size != 0:
            raise ValueError("public feature dimension is not divisible by the declared block size")
        available_blocks = dimension // block_size
        if block_count > available_blocks:
            raise ValueError("requested online feature blocks exceed the public feature dimension")

        candidate_means = np.stack(
            [np.mean(array, axis=0) for array in candidate_arrays],
            axis=0,
        )
        within_variance = np.mean(
            np.stack(
                [np.var(array, axis=0, ddof=1) for array in candidate_arrays],
                axis=0,
            ),
            axis=0,
        )
        between_variance = np.var(candidate_means, axis=0, ddof=0)
        coordinate_scores = between_variance / np.maximum(
            within_variance,
            variance_floor,
        )
        block_scores = np.max(
            coordinate_scores.reshape(available_blocks, block_size),
            axis=1,
        )
        ranked_blocks = sorted(
            range(available_blocks),
            key=lambda index: (-float(block_scores[index]), index),
        )
        selected_blocks = tuple(sorted(ranked_blocks[:block_count]))
        selected_indices = tuple(
            block * block_size + offset for block in selected_blocks for offset in range(block_size)
        )
        feature_indices[str(evidence_action_id)] = selected_indices
        reports[str(evidence_action_id)] = {
            "available_block_count": available_blocks,
            "selected_block_count": block_count,
            "selected_block_indices": list(selected_blocks),
            "selected_block_scores": [float(block_scores[index]) for index in selected_blocks],
            "selected_feature_indices": list(selected_indices),
        }
    return feature_indices, reports


def _select_information_maximum(
    values: Mapping[str, float],
    *,
    minimum_spread_nats: float,
    context: str,
) -> tuple[str, dict[str, Any]]:
    """Select a deterministic EIG maximum and fail closed on a saturated ranking."""

    if not values:
        raise ValueError(f"{context} information ranking is empty")
    if not np.isfinite(minimum_spread_nats) or minimum_spread_nats < 0.0:
        raise ValueError(f"{context} minimum information spread must be finite and nonnegative")
    normalized = {str(key): float(value) for key, value in values.items()}
    if not all(np.isfinite(value) for value in normalized.values()):
        raise ValueError(f"{context} information ranking contains non-finite values")
    minimum = min(normalized.values())
    maximum = max(normalized.values())
    spread = maximum - minimum
    unique_at_12dp = len({round(value, 12) for value in normalized.values()})
    non_saturated = spread >= minimum_spread_nats and (
        minimum_spread_nats == 0.0 or unique_at_12dp > 1
    )
    report = {
        "candidate_count": len(normalized),
        "minimum_nats": minimum,
        "maximum_nats": maximum,
        "spread_nats": spread,
        "minimum_required_spread_nats": minimum_spread_nats,
        "unique_values_at_12_decimal_places": unique_at_12dp,
        "non_saturated": non_saturated,
        "tie_break": "maximum_information_then_lexicographic_action_id",
    }
    if not non_saturated:
        raise ValueError(
            f"{context} information ranking is saturated: "
            f"spread={spread}, required={minimum_spread_nats}, "
            f"unique_at_12dp={unique_at_12dp}"
        )
    selected = min(
        normalized,
        key=lambda item: (-normalized[item], item),
    )
    return selected, report


def _dimension_normalized_likelihood_scale(
    *,
    effective_dimension_count: float,
    selected_feature_dimension: int,
) -> float:
    """Average correlated diagonal evidence over its selected feature dimension."""

    if (
        not np.isfinite(effective_dimension_count)
        or effective_dimension_count <= 0.0
        or selected_feature_dimension <= 0
        or effective_dimension_count > selected_feature_dimension
    ):
        raise ValueError(
            "likelihood normalization requires a positive effective dimension "
            "no greater than the selected feature dimension"
        )
    return float(effective_dimension_count / selected_feature_dimension)


def _uses_dimension_normalized_likelihood(
    fit_plan: Mapping[str, Any],
) -> bool:
    """Resolve likelihood semantics from the declared contract, not a plan version."""

    normalization = fit_plan.get("likelihood_normalization")
    if normalization is None:
        return False
    if normalization != "inverse_selected_feature_dimension":
        raise ValueError(
            f"unsupported likelihood normalization contract: {normalization!r}"
        )
    return True


def _update_online_change_point_posterior(
    posterior: Mapping[tuple[str, int | None], float],
    *,
    action_id: str,
    observation: Sequence[float],
    reference_action_id: str | None = None,
    reference_experiment_index: int | None,
    current_experiment_index: int,
    predictives: Mapping[
        tuple[str, str],
        tuple[np.ndarray, np.ndarray],
    ],
    feature_indices: Mapping[str, Sequence[int]] | None = None,
    likelihood_scale: float = 1.0,
) -> dict[tuple[str, int | None], float]:
    """Apply reference-age-aware public likelihoods without reviving zero states."""

    values = np.asarray(observation, dtype=float).reshape(-1)
    if not np.isfinite(likelihood_scale) or likelihood_scale <= 0.0:
        raise ValueError("online likelihood scale must be finite and positive")
    log_weights: dict[tuple[str, int | None], float] = {}
    for hypothesis, probability in posterior.items():
        if probability <= 0.0:
            continue
        predictive_key = _online_hypothesis_evidence_channel(
            hypothesis,
            action_id,
            reference_action_id=reference_action_id,
            reference_experiment_index=reference_experiment_index,
            current_experiment_index=current_experiment_index,
        )
        try:
            mean, variance = predictives[predictive_key]
        except KeyError as error:
            raise ValueError(
                "online predictives do not cover the latent evidence channel"
            ) from error
        selected_values = values
        if feature_indices is not None:
            try:
                indices = np.asarray(
                    feature_indices[predictive_key[1]],
                    dtype=int,
                )
            except KeyError as error:
                raise ValueError(
                    "online feature selection does not cover the evidence channel"
                ) from error
            if (
                indices.ndim != 1
                or indices.size == 0
                or np.any(indices < 0)
                or np.any(indices >= values.size)
            ):
                raise ValueError("online selected feature indices are invalid")
            selected_values = values[indices]
        if selected_values.shape != mean.shape:
            raise ValueError("observation dimension does not match online predictive")
        log_likelihood = float(
            -0.5
            * likelihood_scale
            * np.sum(np.log(2.0 * math.pi * variance) + ((selected_values - mean) ** 2) / variance)
        )
        log_weights[hypothesis] = math.log(float(probability)) + log_likelihood
    if not log_weights:
        raise ValueError("online posterior update has no active hypotheses")
    maximum = max(log_weights.values())
    weights = {
        hypothesis: math.exp(log_weight - maximum) for hypothesis, log_weight in log_weights.items()
    }
    total = sum(weights.values())
    return {hypothesis: weight / total for hypothesis, weight in weights.items() if weight > 0.0}


def _balanced_hidden_change_times(
    change_times: Sequence[int],
    *,
    trial_count: int,
    seed: int,
    task_id: str,
) -> list[int]:
    """Return a deterministic shuffle whose category counts differ by at most one."""

    if trial_count <= 0 or not change_times:
        raise ValueError("balanced change-time assignment requires trials and categories")
    assignments = [int(change_times[index % len(change_times)]) for index in range(trial_count)]
    rng = np.random.default_rng(
        _stable_seed(seed, f"{task_id}:online-policy-balanced-change-times")
    )
    rng.shuffle(assignments)
    return assignments


def _reference_predictive_adequacy(
    *,
    observations: Sequence[tuple[str, np.ndarray]],
    required_action_ids: Sequence[str],
    predictives: Mapping[tuple[str, str], tuple[np.ndarray, np.ndarray]],
    feature_indices: Mapping[str, Sequence[int]],
    maximum_mean_standardized_squared_error: float,
    maximum_per_action_mean_standardized_squared_error: float,
) -> dict[str, Any]:
    """Cross-fit a campaign reference against frozen development predictives.

    Development data freeze the feature map, baseline predictive family, and
    thresholds.  Each held-out old-world observation is then scored using a
    nuisance offset and conservative scale estimated only from the *other*
    pre-change observations in the same campaign.  No post-change observation
    or realized mechanism-family label enters this calculation.
    """

    if (
        maximum_mean_standardized_squared_error <= 0.0
        or maximum_per_action_mean_standardized_squared_error <= 0.0
    ):
        raise ValueError("predictive-adequacy thresholds must be positive")
    required = {str(item) for item in required_action_ids}
    by_action: dict[str, list[float]] = {action_id: [] for action_id in required}
    prepared: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]] = []
    for raw_action_id, raw_observation in observations:
        action_id = str(raw_action_id)
        channel = _online_evidence_channel("absolute_trace", action_id)
        try:
            mean, variance = predictives[("no_change", channel)]
            indices = np.asarray(feature_indices[channel], dtype=int)
        except KeyError as error:
            raise ValueError(
                "reference predictive check is missing a baseline action channel"
            ) from error
        observation = np.asarray(raw_observation, dtype=float)
        selected = observation[indices]
        if selected.shape != mean.shape or mean.shape != variance.shape:
            raise ValueError("reference predictive check dimensions are inconsistent")
        prepared.append(
            (
                action_id,
                selected,
                np.asarray(mean, dtype=float),
                np.asarray(variance, dtype=float),
            )
        )
    if len(prepared) < 2:
        return {
            "schema_version": "chemworld-reference-predictive-adequacy-0.2",
            "status": "failed",
            "pass": False,
            "source": "within_campaign_pre_change_leave_one_experiment_out",
            "observation_count": len(prepared),
            "checks": {
                "cross_fitting_has_at_least_two_pre_change_observations": False,
                "all_observed_actions_have_cross_fitted_predictive_checks": False,
                "mean_standardized_squared_error_within_limit": False,
                "per_action_standardized_squared_error_within_limit": False,
            },
        }
    dimension_count = 0
    squared_standardized_error = 0.0
    log_score = 0.0
    covered_95 = 0
    cross_fit_parameters: list[dict[str, float | str]] = []
    standardized = [
        (selected - mean) / np.sqrt(variance)
        for _action_id, selected, mean, variance in prepared
    ]
    for held_out_index, (action_id, selected, mean, variance) in enumerate(prepared):
        training_residuals = np.concatenate(
            [
                residual
                for index, residual in enumerate(standardized)
                if index != held_out_index
            ]
        )
        nuisance_offset = float(np.median(training_residuals))
        nuisance_scale = max(
            1.0,
            float(
                np.sqrt(
                    np.mean((training_residuals - nuisance_offset) ** 2)
                )
            ),
        )
        calibrated_mean = mean + nuisance_offset * np.sqrt(variance)
        calibrated_variance = variance * nuisance_scale**2
        residual_squared = ((selected - calibrated_mean) ** 2) / calibrated_variance
        held_out_action_msse = float(np.mean(residual_squared))
        by_action.setdefault(action_id, []).append(held_out_action_msse)
        squared_standardized_error += float(np.sum(residual_squared))
        covered_95 += int(np.sum(np.sqrt(residual_squared) <= 1.959963984540054))
        log_score += float(
            -0.5
            * np.sum(
                np.log(2.0 * math.pi * calibrated_variance)
                + residual_squared
            )
        )
        dimension_count += int(selected.size)
        cross_fit_parameters.append(
            {
                "held_out_action_id": action_id,
                "nuisance_offset_standardized": nuisance_offset,
                "nuisance_scale": nuisance_scale,
            }
        )
    action_msse_by_action: dict[str, float] = {
        action_id: float(np.mean(values))
        for action_id, values in sorted(by_action.items())
        if values
    }
    all_actions_checked = set(action_msse_by_action) == required
    mean_msse = (
        squared_standardized_error / dimension_count
        if dimension_count
        else math.inf
    )
    max_action_msse = max(action_msse_by_action.values(), default=math.inf)
    checks = {
        "cross_fitting_has_at_least_two_pre_change_observations": (
            len(prepared) >= 2
        ),
        "all_observed_actions_have_cross_fitted_predictive_checks": (
            all_actions_checked
        ),
        "mean_standardized_squared_error_within_limit": (
            math.isfinite(mean_msse)
            and mean_msse <= maximum_mean_standardized_squared_error
        ),
        "per_action_standardized_squared_error_within_limit": (
            math.isfinite(max_action_msse)
            and max_action_msse
            <= maximum_per_action_mean_standardized_squared_error
        ),
    }
    passed = all(checks.values())
    return {
        "schema_version": "chemworld-reference-predictive-adequacy-0.2",
        "status": "passed" if passed else "failed",
        "pass": passed,
        "source": (
            "within_campaign_pre_change_leave_one_experiment_out_against_"
            "development_frozen_no_change_predictive_structure"
        ),
        "development_role": (
            "freeze feature map, predictive family, and pass thresholds only"
        ),
        "campaign_parameter_fit": (
            "leave_one_experiment_out_standardized_nuisance_location_and_"
            "non_shrinking_scale"
        ),
        "post_change_observations_used": False,
        "truth_family_used_for_reference_fit": False,
        "observation_count": len(prepared),
        "selected_feature_dimension_count": dimension_count,
        "mean_standardized_squared_error": mean_msse,
        "maximum_action_mean_standardized_squared_error": max_action_msse,
        "prediction_interval_95_coverage": (
            covered_95 / dimension_count if dimension_count else None
        ),
        "mean_predictive_log_score_per_dimension": (
            log_score / dimension_count if dimension_count else None
        ),
        "action_mean_standardized_squared_error": action_msse_by_action,
        "cross_fit_parameters": cross_fit_parameters,
        "thresholds": {
            "maximum_mean_standardized_squared_error": (
                maximum_mean_standardized_squared_error
            ),
            "maximum_per_action_mean_standardized_squared_error": (
                maximum_per_action_mean_standardized_squared_error
            ),
        },
        "checks": checks,
    }


def _cluster_bootstrap_metric_interval(
    *,
    cluster_ids: Sequence[str],
    statistic: Callable[[list[int]], float],
    seed: int,
    draws: int = 2_000,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Bootstrap a row-level statistic at the independent-world cluster level."""

    if not cluster_ids:
        raise ValueError("cluster bootstrap requires at least one row")
    if draws <= 0:
        raise ValueError("cluster bootstrap draws must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("cluster bootstrap confidence must be in (0, 1)")
    rows_by_cluster: dict[str, list[int]] = {}
    for index, cluster_id in enumerate(cluster_ids):
        rows_by_cluster.setdefault(str(cluster_id), []).append(index)
    clusters = sorted(rows_by_cluster)
    rng = np.random.default_rng(seed)
    estimates: list[float] = []
    for _ in range(draws):
        sampled_clusters = rng.choice(clusters, size=len(clusters), replace=True)
        sampled_indices = [
            row_index
            for cluster_id in sampled_clusters
            for row_index in rows_by_cluster[str(cluster_id)]
        ]
        estimate = float(statistic(sampled_indices))
        if not math.isfinite(estimate):
            raise ValueError("cluster bootstrap statistic must be finite")
        estimates.append(estimate)
    tail = (1.0 - confidence) / 2.0
    return (
        float(np.quantile(estimates, tail)),
        float(np.quantile(estimates, 1.0 - tail)),
    )


def _binary_rate(
    labels: Sequence[bool],
    predictions: Sequence[bool],
    *,
    positive_label: bool,
) -> float:
    """Return the positive prediction rate inside one truth class."""

    if len(labels) != len(predictions):
        raise ValueError("binary labels and predictions must have equal length")
    selected = [
        bool(prediction)
        for label, prediction in zip(labels, predictions, strict=True)
        if bool(label) is positive_label
    ]
    if not selected:
        raise ValueError("binary rate requires the requested truth class")
    return float(np.mean(selected))


def _binary_auroc_for_indices(
    labels: Sequence[bool],
    scores: Sequence[float],
    indices: Sequence[int],
) -> float:
    """Compute tie-aware AUROC over a resampled set of row indices."""

    positive_scores = [float(scores[index]) for index in indices if labels[index]]
    negative_scores = [float(scores[index]) for index in indices if not labels[index]]
    if not positive_scores or not negative_scores:
        raise ValueError("AUROC requires both changed and no-change rows")
    pairwise_credit = sum(
        1.0 if positive > negative else 0.5 if positive == negative else 0.0
        for positive in positive_scores
        for negative in negative_scores
    )
    return pairwise_credit / (len(positive_scores) * len(negative_scores))


def _online_capability_chain_summary(
    *,
    trials: Sequence[Mapping[str, Any]],
    post_change_budget: int,
    frozen_criteria: Mapping[str, Any],
    bootstrap_seed: int,
) -> dict[str, Any]:
    """Report the frozen A3 reference-policy capability chain.

    Changed and never campaigns have different estimands.  Attribution is
    undefined for ``never`` and is therefore never placed in that denominator.
    Threshold-crossing events control recall, horizon FPR, and delay; checkpoint
    probabilities control AUROC and Brier calibration.
    """

    if not trials:
        raise ValueError("online capability-chain summary requires trials")
    threshold = float(
        frozen_criteria["change_decision_probability_threshold"]
    )
    bootstrap_draws = int(frozen_criteria.get("cluster_bootstrap_draws", 2_000))
    bootstrap_confidence = float(
        frozen_criteria.get("confidence_level", 0.95)
    )
    raw_checkpoints = frozen_criteria.get(
        "detection_checkpoint_experiments_after_change",
        [post_change_budget],
    )
    if isinstance(raw_checkpoints, Sequence) and not isinstance(
        raw_checkpoints, str | bytes
    ):
        frozen_checkpoints = sorted({int(item) for item in raw_checkpoints})
    else:
        frozen_checkpoints = [int(raw_checkpoints)]
    integrated_checkpoints = [
        checkpoint
        for checkpoint in frozen_checkpoints
        if checkpoint <= post_change_budget
    ]
    if not integrated_checkpoints or post_change_budget not in frozen_checkpoints:
        raise ValueError(
            "post-change budget must be one of the frozen detection checkpoints"
        )

    def detection_event(
        row: Mapping[str, Any],
        *,
        budget: int,
    ) -> tuple[bool, int | None]:
        delay = next(
            (
                int(update["phase_local_experiment_index"])
                for update in row["post_change_updates"]
                if int(update["phase_local_experiment_index"]) <= budget
                and 1.0 - float(update["posterior"]["no_change"])
                >= threshold
            ),
            None,
        )
        return delay is not None, delay

    reference_pass = [
        row["reference_sufficiency_certificate"]["pass"] is True
        for row in trials
    ]
    cluster_reference: dict[str, bool] = {}
    for row, passed in zip(trials, reference_pass, strict=True):
        cluster_id = str(row["statistical_cluster_id"])
        previous = cluster_reference.setdefault(cluster_id, passed)
        if previous != passed:
            raise ValueError(
                "paired candidate arms disagree on reference sufficiency"
            )
    reference_hits = sum(cluster_reference.values())
    reference_cluster_count = len(cluster_reference)
    reference_interval = wilson_interval(
        reference_hits,
        reference_cluster_count,
    )
    changed: list[bool] = []
    probabilities: list[float] = []
    detections: list[bool] = []
    delays: list[int | None] = []
    qualified_cluster_ids: list[str] = []
    detected_changed_rows: list[Mapping[str, Any]] = []
    changed_end_to_end_values: list[bool] = []
    changed_end_to_end_cluster_ids: list[str] = []
    never_horizon_alarm_values: list[bool] = []
    never_horizon_alarm_cluster_ids: list[str] = []
    changed_reference_clusters: dict[str, bool] = {}
    never_reference_clusters: dict[str, bool] = {}
    for row, reference_sufficient in zip(
        trials,
        reference_pass,
        strict=True,
    ):
        truth_id = str(row["truth_id"])
        truth_changed = truth_id != "no_change"
        posterior = row["posterior_by_post_budget"][str(post_change_budget)]
        change_probability = 1.0 - float(posterior["no_change"])
        predicted_family = str(row["predictions_by_post_budget"][str(post_change_budget)])
        detected, detection_delay = detection_event(
            row,
            budget=post_change_budget,
        )
        cluster_id = str(row["statistical_cluster_id"])
        reference_by_truth = (
            changed_reference_clusters
            if truth_changed
            else never_reference_clusters
        )
        previous_reference = reference_by_truth.setdefault(
            cluster_id,
            reference_sufficient,
        )
        if previous_reference != reference_sufficient:
            raise ValueError(
                "paired truth arms disagree on reference sufficiency"
            )
        if reference_sufficient:
            changed.append(truth_changed)
            probabilities.append(change_probability)
            detections.append(detected)
            qualified_cluster_ids.append(cluster_id)
            delays.append(detection_delay if truth_changed else None)
            if truth_changed and detected:
                detected_changed_rows.append(row)
        if truth_changed:
            changed_end_to_end_cluster_ids.append(cluster_id)
            changed_end_to_end_values.append(
                reference_sufficient
                and detected
                and predicted_family == truth_id
            )
        else:
            never_horizon_alarm_cluster_ids.append(cluster_id)
            never_horizon_alarm_values.append(detected)

    qualified_count = len(changed)
    if qualified_count and any(changed) and not all(changed):
        detection = change_detection_summary(
            changed=changed,
            probabilities=probabilities,
            detection_delays=delays,
            detections=detections,
            threshold=threshold,
            right_censor_time=post_change_budget,
            bootstrap_seed=bootstrap_seed,
        )
        detection["sensitivity_cluster_bootstrap_interval"] = list(
            _cluster_bootstrap_metric_interval(
                cluster_ids=qualified_cluster_ids,
                seed=bootstrap_seed + 1,
                draws=bootstrap_draws,
                confidence=bootstrap_confidence,
                statistic=lambda indices: _binary_rate(
                    [changed[index] for index in indices],
                    [detections[index] for index in indices],
                    positive_label=True,
                ),
            )
        )
        detection["false_positive_rate_cluster_bootstrap_interval"] = list(
            _cluster_bootstrap_metric_interval(
                cluster_ids=qualified_cluster_ids,
                seed=bootstrap_seed + 2,
                draws=bootstrap_draws,
                confidence=bootstrap_confidence,
                statistic=lambda indices: _binary_rate(
                    [changed[index] for index in indices],
                    [detections[index] for index in indices],
                    positive_label=False,
                ),
            )
        )
        detection["auroc_cluster_bootstrap_interval"] = list(
            _cluster_bootstrap_metric_interval(
                cluster_ids=qualified_cluster_ids,
                seed=bootstrap_seed + 3,
                draws=bootstrap_draws,
                confidence=bootstrap_confidence,
                statistic=lambda indices: _binary_auroc_for_indices(
                    changed,
                    probabilities,
                    indices,
                ),
            )
        )
    else:
        detection = {
            "status": "not_evaluable_without_reference_sufficient_both_class_trials",
            "campaign_count": qualified_count,
        }
    checkpoint_brier: dict[str, float] = {}
    for checkpoint in integrated_checkpoints:
        checkpoint_rows = [
            row
            for row, passed in zip(trials, reference_pass, strict=True)
            if passed
        ]
        if checkpoint_rows:
            losses_by_class: dict[bool, list[float]] = {
                True: [],
                False: [],
            }
            for row in checkpoint_rows:
                truth_changed = str(row["truth_id"]) != "no_change"
                probability = 1.0 - float(
                    row["posterior_by_post_budget"][str(checkpoint)][
                        "no_change"
                    ]
                )
                losses_by_class[truth_changed].append(
                    (
                        probability
                        - (1.0 if truth_changed else 0.0)
                    )
                    ** 2
                )
            if all(losses_by_class.values()):
                checkpoint_brier[str(checkpoint)] = float(
                    np.mean(
                        [
                            np.mean(losses_by_class[True]),
                            np.mean(losses_by_class[False]),
                        ]
                    )
                )
    integrated_mean_brier = (
        float(np.mean(list(checkpoint_brier.values())))
        if len(checkpoint_brier) == len(integrated_checkpoints)
        else math.inf
    )
    attribution_hits = sum(
        str(row["predictions_by_post_budget"][str(post_change_budget)])
        == str(row["truth_id"])
        for row in detected_changed_rows
    )
    attribution_interval = (
        wilson_interval(attribution_hits, len(detected_changed_rows))
        if detected_changed_rows
        else (0.0, 1.0)
    )
    attribution_cluster_interval = (
        _cluster_bootstrap_metric_interval(
            cluster_ids=[
                str(row["statistical_cluster_id"])
                for row in detected_changed_rows
            ],
            seed=bootstrap_seed + 4,
            draws=bootstrap_draws,
            confidence=bootstrap_confidence,
            statistic=lambda indices: float(
                np.mean(
                    [
                        str(
                            detected_changed_rows[index][
                                "predictions_by_post_budget"
                            ][str(post_change_budget)]
                        )
                        == str(detected_changed_rows[index]["truth_id"])
                        for index in indices
                    ]
                )
            ),
        )
        if detected_changed_rows
        else (0.0, 1.0)
    )
    end_to_end_hits = sum(changed_end_to_end_values)
    end_to_end_interval = wilson_interval(
        end_to_end_hits,
        len(changed_end_to_end_values),
    )
    end_to_end_cluster_interval = _cluster_bootstrap_metric_interval(
        cluster_ids=changed_end_to_end_cluster_ids,
        seed=bootstrap_seed + 5,
        draws=bootstrap_draws,
        confidence=bootstrap_confidence,
        statistic=lambda indices: float(
            np.mean([changed_end_to_end_values[index] for index in indices])
        ),
    )
    never_horizon_fpr = float(np.mean(never_horizon_alarm_values))
    never_horizon_fpr_cluster_interval = _cluster_bootstrap_metric_interval(
        cluster_ids=never_horizon_alarm_cluster_ids,
        seed=bootstrap_seed + 6,
        draws=bootstrap_draws,
        confidence=bootstrap_confidence,
        statistic=lambda indices: float(
            np.mean([never_horizon_alarm_values[index] for index in indices])
        ),
    )
    changed_reference_interval = wilson_interval(
        sum(changed_reference_clusters.values()),
        len(changed_reference_clusters),
    )
    never_reference_interval = wilson_interval(
        sum(never_reference_clusters.values()),
        len(never_reference_clusters),
    )
    no_false_alarm_given_reference = [
        not detected
        for detected, truth_changed in zip(detections, changed, strict=True)
        if not truth_changed
    ]
    changed_detection_given_reference = [
        detected
        for detected, truth_changed in zip(detections, changed, strict=True)
        if truth_changed
    ]
    maximum_brier = float(
        frozen_criteria.get(
            "maximum_integrated_mean_change_probability_brier_score",
            frozen_criteria.get("maximum_change_probability_brier_score"),
        )
    )
    criteria = {
        "reference_acquisition_rate": (
            reference_interval[0]
            >= float(
                frozen_criteria[
                    "minimum_reference_acquisition_rate_wilson_lower_bound"
                ]
            )
        ),
        "changed_detection_recall": (
            isinstance(
                detection.get("sensitivity_cluster_bootstrap_interval"),
                Sequence,
            )
            and float(
                detection["sensitivity_cluster_bootstrap_interval"][0]
            )
            >= float(
                frozen_criteria[
                    "minimum_changed_detection_recall_cluster_bootstrap_lower_bound"
                ]
            )
        ),
        "no_change_false_positive_rate": (
            float(never_horizon_fpr_cluster_interval[1])
            <= float(
                frozen_criteria[
                    "maximum_no_change_false_positive_rate_cluster_bootstrap_upper_bound"
                ]
            )
        ),
        "change_detection_auroc": (
            isinstance(
                detection.get("auroc_cluster_bootstrap_interval"),
                Sequence,
            )
            and float(detection["auroc_cluster_bootstrap_interval"][0])
            >= float(
                frozen_criteria[
                    "minimum_change_detection_auroc_cluster_bootstrap_lower_bound"
                ]
            )
        ),
        "integrated_mean_change_probability_brier_score": (
            math.isfinite(integrated_mean_brier)
            and integrated_mean_brier <= maximum_brier
        ),
        "attribution_given_detection_and_reference": (
            bool(detected_changed_rows)
            and attribution_cluster_interval[0]
            >= float(
                frozen_criteria[
                    "minimum_attribution_given_detection_and_reference_cluster_bootstrap_lower_bound"
                ]
            )
        ),
        "end_to_end_online_success": (
            end_to_end_cluster_interval[0]
            >= float(
                frozen_criteria[
                    "minimum_end_to_end_success_rate_cluster_bootstrap_lower_bound"
                ]
            )
        ),
    }
    gate_pass = all(criteria.values())
    return {
        "schema_version": "chemworld-online-capability-chain-0.2",
        "status": "passed" if gate_pass else "failed",
        "gate_pass": gate_pass,
        "post_change_budget": post_change_budget,
        "trial_count": len(trials),
        "statistical_cluster_count": reference_cluster_count,
        "statistical_unit": (
            "task_id_and_world_seed; candidate truth arms sharing the same "
            "stable-prefix world are one resampling cluster"
        ),
        "p_reference_sufficient": (
            reference_hits / reference_cluster_count
        ),
        "reference_sufficient_interval": list(reference_interval),
        "changed_campaigns": {
            "p_reference_sufficient_given_changed": (
                sum(changed_reference_clusters.values())
                / len(changed_reference_clusters)
            ),
            "reference_sufficient_interval": list(
                changed_reference_interval
            ),
            "p_detection_given_reference_and_changed": (
                float(np.mean(changed_detection_given_reference))
                if changed_detection_given_reference
                else None
            ),
            "p_attribution_given_detection_and_reference_and_changed": (
                attribution_hits / len(detected_changed_rows)
                if detected_changed_rows
                else None
            ),
            "p_reference_detection_attribution_given_changed": (
                end_to_end_hits / len(changed_end_to_end_values)
            ),
            "end_to_end_cluster_bootstrap_interval": list(
                end_to_end_cluster_interval
            ),
        },
        "no_change_campaigns": {
            "p_reference_sufficient_given_never": (
                sum(never_reference_clusters.values())
                / len(never_reference_clusters)
            ),
            "reference_sufficient_interval": list(never_reference_interval),
            "p_no_false_alarm_given_reference_and_never": (
                float(np.mean(no_false_alarm_given_reference))
                if no_false_alarm_given_reference
                else None
            ),
            "fpr_horizon": never_horizon_fpr,
            "fpr_horizon_cluster_bootstrap_interval": list(
                never_horizon_fpr_cluster_interval
            ),
            "attribution_defined": False,
        },
        "change_detection_conditional_on_reference": detection,
        "change_probability_brier_by_checkpoint": checkpoint_brier,
        "integrated_mean_change_probability_brier_score": (
            integrated_mean_brier
        ),
        "brier_aggregation": (
            "equal_weight_changed_never_class_mean_then_equal_weight_mean_"
            "across_frozen_checkpoints_up_to_budget"
        ),
        "p_attribution_given_detection_and_reference": (
            attribution_hits / len(detected_changed_rows)
            if detected_changed_rows
            else None
        ),
        "attribution_given_detection_and_reference_interval": list(
            attribution_interval
        ),
        "attribution_given_detection_and_reference_cluster_bootstrap_interval": list(
            attribution_cluster_interval
        ),
        "p_end_to_end_reference_detection_attribution_success": (
            end_to_end_hits / len(changed_end_to_end_values)
        ),
        "end_to_end_success_interval": list(end_to_end_interval),
        "end_to_end_success_cluster_bootstrap_interval": list(
            end_to_end_cluster_interval
        ),
        "reference_acquisition_failures_retained_in_end_to_end_denominator": True,
        "end_to_end_denominator": "changed_campaigns_only",
        "never_attribution_excluded_as_undefined": True,
        "frozen_criteria": dict(frozen_criteria),
        "criteria_pass": criteria,
    }


def _sample_online_predictive_job(job: Mapping[str, Any]) -> dict[str, list[float]]:
    """Return absolute, transition, and stable public predictive encodings."""

    action_id = str(job["action_id"])
    action_library = {action_id: np.asarray(job["action_vector"], dtype=float)}
    with PublicCampaignObservationSession(
        task_id=str(job["task_id"]),
        seed=int(job["world_seed"]),
        interventions=(),
        action_library=action_library,
        experiment_horizon=1,
        observation_seed=int(job["pre_observation_seed"]),
    ) as pre_session:
        pre = np.asarray(pre_session.observe(action_id), dtype=float)
    with PublicCampaignObservationSession(
        task_id=str(job["task_id"]),
        seed=int(job["world_seed"]),
        interventions=job["interventions"],
        action_library=action_library,
        experiment_horizon=1,
        observation_seed=int(job["stable_reference_observation_seed"]),
    ) as stable_reference_session:
        stable_reference = np.asarray(
            stable_reference_session.observe(action_id),
            dtype=float,
        )
    with PublicCampaignObservationSession(
        task_id=str(job["task_id"]),
        seed=int(job["world_seed"]),
        interventions=job["interventions"],
        action_library=action_library,
        experiment_horizon=1,
        observation_seed=int(job["post_observation_seed"]),
    ) as post_session:
        post = np.asarray(post_session.observe(action_id), dtype=float)
    return {
        "transition_contrast": (post - pre).tolist(),
        "stable_contrast": (post - stable_reference).tolist(),
        "absolute_trace": post.tolist(),
    }


def _unreachable_sample(_candidate_id: str, _action_id: str, _seed: int) -> list[float]:
    raise RuntimeError("loaded Gate A predictives do not sample during evaluation")


def _execute_gate_a_trial_job(job: Mapping[str, Any]) -> dict[str, Any]:
    task_id = str(job["task_id"])
    truth_id = str(job["truth_id"])
    trial_seed = int(job["trial_seed"])
    budgets = [int(item) for item in job["budgets"]]
    maximum_budget = max(budgets)
    action_ids = [str(item) for item in job["action_ids"]]
    action_library = {
        str(key): np.asarray(value, dtype=float) for key, value in job["action_library"].items()
    }

    def loaded_oracle() -> GaussianMechanismOracle:
        oracle = GaussianMechanismOracle(
            candidate_ids=[str(item) for item in job["candidate_ids"]],
            action_ids=action_ids,
            sample_public_observation=_unreachable_sample,
            samples_per_candidate=4,
            variance_floor=float(job["variance_floor"]),
            seed=int(job["oracle_seed"]),
        )
        oracle.load_predictives(job["predictives"])
        return oracle

    active_oracle = loaded_oracle()
    active_rows: dict[int, dict[str, Any]] = {}
    selected_actions: list[str] = []
    with PublicCampaignObservationSession(
        task_id=task_id,
        seed=trial_seed,
        interventions=job["interventions"],
        action_library=action_library,
        experiment_horizon=maximum_budget,
    ) as active_session:
        for experiment_index in range(1, maximum_budget + 1):
            information = active_oracle.expected_information_by_action(
                draws=int(job["information_draws"])
            )
            eligible = list(action_ids)
            if job.get("repeat_policy") == "without_replacement_within_budget":
                unused = [item for item in action_ids if item not in selected_actions]
                if unused:
                    eligible = unused
            action_id = max(eligible, key=information.__getitem__)
            selected_actions.append(action_id)
            active_oracle.update(
                action_id=action_id,
                observation=active_session.observe(action_id),
            )
            if experiment_index in budgets:
                active_rows[experiment_index] = {
                    "truth_id": truth_id,
                    "world_seed": trial_seed,
                    "prediction": max(
                        active_oracle.posterior,
                        key=active_oracle.posterior.__getitem__,
                    ),
                    "actions": list(selected_actions),
                    "posterior": dict(active_oracle.posterior),
                }

    decoder_oracle = loaded_oracle()
    decoder_rows: dict[int, dict[str, Any]] = {}
    fixed_actions = action_ids[:maximum_budget]
    with PublicCampaignObservationSession(
        task_id=task_id,
        seed=trial_seed,
        interventions=job["interventions"],
        action_library=action_library,
        experiment_horizon=maximum_budget,
    ) as decoder_session:
        for experiment_index, action_id in enumerate(fixed_actions, start=1):
            decoder_oracle.update(
                action_id=action_id,
                observation=decoder_session.observe(action_id),
            )
            if experiment_index in budgets:
                decoder_rows[experiment_index] = {
                    "truth_id": truth_id,
                    "world_seed": trial_seed,
                    "prediction": max(
                        decoder_oracle.posterior,
                        key=decoder_oracle.posterior.__getitem__,
                    ),
                    "actions": fixed_actions[:experiment_index],
                    "posterior": dict(decoder_oracle.posterior),
                }
    return {"active": active_rows, "decoder": decoder_rows}


def _execute_jobs(
    function: Any,
    jobs: Sequence[Mapping[str, Any]],
    *,
    workers: int,
) -> list[Any]:
    if workers <= 0:
        raise ValueError("Gate A execution workers must be positive")
    if workers == 1:
        return [function(job) for job in jobs]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(function, jobs))


def _paired_action_id(action_ids: Sequence[str]) -> str:
    return "+".join(str(item) for item in action_ids)


def _fit_paired_batch_oracle(
    *,
    candidate_ids: Sequence[str],
    action_ids: Sequence[str],
    per_action_samples: Mapping[tuple[str, str], Sequence[Sequence[float]]],
    batch_size: int,
    variance_floor: float,
    seed: int,
    likelihood_scale: float = 1.0,
) -> tuple[GaussianMechanismOracle, dict[str, tuple[str, ...]]]:
    """Fit batch predictives from nuisance-aligned per-action contrast samples."""

    batches = {
        _paired_action_id(batch): tuple(batch)
        for batch in itertools.combinations(action_ids, batch_size)
    }
    if not batches:
        raise ValueError("paired Gate A batch size exceeds the public action library")
    batch_samples: dict[tuple[str, str], list[list[float]]] = {}
    for candidate_id in candidate_ids:
        counts = {
            len(per_action_samples[(str(candidate_id), action_id)]) for action_id in action_ids
        }
        if len(counts) != 1:
            raise ValueError("paired Gate A samples must align by nuisance seed")
        sample_count = counts.pop()
        for batch_id, batch in batches.items():
            batch_samples[(str(candidate_id), batch_id)] = [
                np.concatenate(
                    [
                        np.asarray(per_action_samples[(str(candidate_id), action_id)][index])
                        for action_id in batch
                    ]
                ).tolist()
                for index in range(sample_count)
            ]
    oracle = GaussianMechanismOracle(
        candidate_ids=candidate_ids,
        action_ids=list(batches),
        sample_public_observation=_unreachable_sample,
        samples_per_candidate=4,
        variance_floor=variance_floor,
        likelihood_scale=likelihood_scale,
        seed=seed,
    )
    oracle.fit_predictives_from_samples(batch_samples)
    return oracle, batches


def _execute_paired_gate_a_trial_job(job: Mapping[str, Any]) -> dict[str, Any]:
    candidate_ids = [str(item) for item in job["candidate_ids"]]
    batch_ids = [str(item) for item in job["batch_ids"]]
    oracle = GaussianMechanismOracle(
        candidate_ids=candidate_ids,
        action_ids=batch_ids,
        sample_public_observation=_unreachable_sample,
        samples_per_candidate=4,
        variance_floor=float(job["variance_floor"]),
        seed=int(job["oracle_seed"]),
    )
    oracle.load_predictives(job["predictives"])
    rows: dict[str, dict[str, Any]] = {}
    contrasts: dict[str, list[float]] = {}
    for role in ("active", "decoder"):
        batch_id = str(job[f"{role}_batch_id"])
        if batch_id not in contrasts:
            action_ids = [str(item) for item in job["batches"][batch_id]]
            contrasts[batch_id] = _sample_paired_contrast_job(
                {
                    "task_id": job["task_id"],
                    "world_seed": job["world_seed"],
                    "interventions": job["interventions"],
                    "action_ids": action_ids,
                    "action_library": {
                        action_id: job["action_library"][action_id] for action_id in action_ids
                    },
                    "pre_observation_seed": _stable_seed(
                        int(job["world_seed"]), f"gate-a-v0.2.2:{batch_id}:pre"
                    ),
                    "post_observation_seed": _stable_seed(
                        int(job["world_seed"]), f"gate-a-v0.2.2:{batch_id}:post"
                    ),
                    "feature_indices": job.get("feature_indices"),
                }
            )
        oracle.reset_posterior()
        oracle.update(action_id=batch_id, observation=contrasts[batch_id])
        rows[role] = {
            "truth_id": str(job["truth_id"]),
            "world_seed": int(job["world_seed"]),
            "pre_observation_seed": _stable_seed(
                int(job["world_seed"]), f"gate-a-v0.2.2:{batch_id}:pre"
            ),
            "post_observation_seed": _stable_seed(
                int(job["world_seed"]), f"gate-a-v0.2.2:{batch_id}:post"
            ),
            "prediction": max(oracle.posterior, key=oracle.posterior.__getitem__),
            "actions": list(job["batches"][batch_id]),
            "batch_id": batch_id,
            "posterior": dict(oracle.posterior),
        }
    return rows


def _execute_online_policy_trial_job(job: Mapping[str, Any]) -> dict[str, Any]:
    """Run one reference-age-aware latent change-point diagnosis trial."""

    candidate_ids = [str(item) for item in job["candidate_ids"]]
    action_ids = [str(item) for item in job["action_ids"]]
    reference_action_ids = [str(item) for item in job["reference_action_ids"]]
    if not set(reference_action_ids).issubset(action_ids):
        raise ValueError("online reference actions must belong to the action library")
    if "no_change" not in candidate_ids:
        raise ValueError("online hidden-change filtering requires a no_change state")
    action_library = {
        str(key): np.asarray(value, dtype=float) for key, value in job["action_library"].items()
    }
    raw_relational_evidence = job.get("relational_evidence_enabled", False)
    if not isinstance(raw_relational_evidence, bool):
        raise ValueError("online trial relational evidence flag must be boolean")
    relational_evidence_enabled = raw_relational_evidence
    evidence_action_ids = [
        _online_evidence_channel(encoding, action_id)
        for action_id in action_ids
        for encoding in (
            "transition_contrast",
            "stable_contrast",
            "absolute_trace",
        )
    ]
    if relational_evidence_enabled:
        evidence_action_ids.extend(
            _online_relational_evidence_channel(
                encoding,
                action_id,
                reference_action_id,
            )
            for action_id, reference_action_id in itertools.permutations(
                action_ids,
                2,
            )
            for encoding in (
                "relational_transition_contrast",
                "relational_stable_contrast",
            )
        )
    expected_predictive_keys = {
        (candidate_id, action_id)
        for candidate_id in candidate_ids
        for action_id in evidence_action_ids
    }
    predictives = _online_predictive_lookup(job["predictives"])
    if set(predictives) != expected_predictive_keys:
        raise ValueError("online predictives must cover every candidate/evidence channel")
    feature_indices = {
        str(key): tuple(int(index) for index in value)
        for key, value in job["feature_indices"].items()
    }
    if set(feature_indices) != set(evidence_action_ids):
        raise ValueError("online feature selection must cover every evidence channel")
    relational_reference_priority = {
        str(action_id): [str(item) for item in raw_priority]
        for action_id, raw_priority in job.get(
            "relational_reference_priority",
            {},
        ).items()
    }
    declared_relational_reference_actions = {
        str(action_id): [str(item) for item in raw_references]
        for action_id, raw_references in job.get(
            "declared_relational_reference_actions",
            {},
        ).items()
    }
    if relational_evidence_enabled:
        if set(relational_reference_priority) != set(reference_action_ids):
            raise ValueError(
                "relational reference priorities must cover every policy action"
            )
        if set(declared_relational_reference_actions) != set(reference_action_ids):
            raise ValueError(
                "declared relational references must cover every policy action"
            )
        for action_id, priority in relational_reference_priority.items():
            if (
                len(priority) != len(reference_action_ids) - 1
                or set(priority) != set(reference_action_ids) - {action_id}
            ):
                raise ValueError(
                    "relational reference priority must order every other policy action"
                )
            declared_references = declared_relational_reference_actions[action_id]
            if (
                len(declared_references) != len(set(declared_references))
                or not set(declared_references).issubset(
                    set(reference_action_ids) - {action_id}
                )
            ):
                raise ValueError(
                    "declared relational references must be unique other policy actions"
                )
    elif relational_reference_priority or declared_relational_reference_actions:
        raise ValueError("legacy online evidence cannot declare relational references")
    likelihood_scale = float(job["likelihood_scale"])
    change_time = int(job["change_time"])
    allowed_change_times = [int(item) for item in job["change_time_candidates"]]
    post_budgets = sorted({int(item) for item in job["post_change_budget_checkpoints"]})
    minimum_reference_experiments = int(job["minimum_reference_experiments"])
    enforce_reference_sufficiency = bool(
        job.get("enforce_reference_sufficiency", False)
    )
    if not post_budgets:
        raise ValueError("online policy requires post-change budget checkpoints")
    post_budget = max(post_budgets)
    raw_hazards = job["change_hazards"]
    if not isinstance(raw_hazards, Mapping):
        raise ValueError("online policy change hazards must be a mapping")
    hazards = {str(key): float(value) for key, value in raw_hazards.items()}
    if (
        change_time <= 0
        or minimum_reference_experiments <= 0
        or post_budget <= 0
        or (
            enforce_reference_sufficiency
            and change_time < minimum_reference_experiments
        )
    ):
        raise ValueError("online policy trial horizons must be positive")
    if set(hazards) != {str(item) for item in allowed_change_times} or any(
        not 0.0 < value <= 1.0 for value in hazards.values()
    ):
        raise ValueError(
            "online policy change hazards must cover every allowed point"
        )

    hypothesis_posterior: dict[tuple[str, int | None], float] = {("no_change", None): 1.0}
    family_posterior = _online_family_posterior(
        hypothesis_posterior,
        candidate_ids,
    )
    references: dict[str, tuple[np.ndarray, int]] = {}
    pre_actions: list[str] = []
    pre_updates: list[dict[str, Any]] = []
    post_actions: list[str] = []
    post_updates: list[dict[str, Any]] = []
    pre_observations: list[tuple[str, np.ndarray]] = []
    posterior_by_post_budget: dict[str, dict[str, float]] = {}
    experiment_index = 0

    def take_policy_step(
        session: PublicCampaignObservationSession,
        *,
        phase_index: int,
        update_sink: list[dict[str, Any]],
        refresh_old_world_reference: bool,
    ) -> None:
        nonlocal experiment_index
        nonlocal hypothesis_posterior
        nonlocal family_posterior
        experiment_index += 1
        hypothesis_posterior = _advance_online_change_point_hypotheses(
            hypothesis_posterior,
            change_after_experiment=experiment_index - 1,
            allowed_change_times=allowed_change_times,
            candidate_ids=candidate_ids,
            hazard=hazards,
        )
        action_id = reference_action_ids[(experiment_index - 1) % len(reference_action_ids)]

        observation = np.asarray(session.observe(action_id), dtype=float)
        first_observation = action_id not in references
        reference_action_id, reference_source = _select_online_reference_action(
            action_id=action_id,
            available_reference_action_ids=references,
            relational_evidence_enabled=relational_evidence_enabled,
            declared_relational_references=declared_relational_reference_actions.get(
                action_id,
                (),
            ),
            fit_ranked_relational_references=relational_reference_priority.get(
                action_id,
                (),
            ),
        )
        reference = (
            references.get(reference_action_id)
            if reference_action_id is not None
            else None
        )
        if reference is None:
            reference_action_id = None
            reference_experiment_index = None
            evidence = observation
        else:
            reference_experiment_index = reference[1]
            evidence = observation - reference[0]
        hypothesis_posterior = _update_online_change_point_posterior(
            hypothesis_posterior,
            action_id=action_id,
            observation=evidence.tolist(),
            reference_action_id=reference_action_id,
            reference_experiment_index=reference_experiment_index,
            current_experiment_index=experiment_index,
            predictives=predictives,
            feature_indices=feature_indices,
            likelihood_scale=likelihood_scale,
        )
        family_posterior = _online_family_posterior(
            hypothesis_posterior,
            candidate_ids,
        )
        if first_observation or refresh_old_world_reference:
            references[action_id] = (observation, experiment_index)
        if refresh_old_world_reference:
            pre_observations.append((action_id, observation))
        update_sink.append(
            {
                "experiment_index": experiment_index,
                "phase_local_experiment_index": phase_index,
                "action_id": action_id,
                "evidence_encoding": (
                    "latent_mixture_absolute"
                    if reference_action_id is None
                    else "latent_mixture_relational_transition_or_stable_contrast"
                    if reference_action_id != action_id
                    else "latent_mixture_transition_or_stable_contrast"
                ),
                "reference_established": first_observation,
                "reference_action_id": reference_action_id,
                "reference_experiment_index": reference_experiment_index,
                "reference_source": reference_source,
                "policy_cycle_position": ((experiment_index - 1) % len(reference_action_ids)),
                "active_change_point_hypothesis_count": len(hypothesis_posterior),
                "posterior": dict(family_posterior),
            }
        )

    with PublicCampaignObservationSession(
        task_id=str(job["task_id"]),
        seed=int(job["world_seed"]),
        interventions=(),
        action_library=action_library,
        experiment_horizon=change_time,
        observation_seed=int(job["pre_observation_seed"]),
    ) as pre_session:
        for phase_index in range(1, change_time + 1):
            take_policy_step(
                pre_session,
                phase_index=phase_index,
                update_sink=pre_updates,
                refresh_old_world_reference=True,
            )
            pre_actions.append(pre_updates[-1]["action_id"])
        pre_noise_audit = list(pre_session.observation_noise_audit)

    actual_pre_change_reference_ids = list(references)
    predictive_contract = job["reference_predictive_adequacy"]
    if not isinstance(predictive_contract, Mapping):
        raise ValueError("online trial requires predictive reference criteria")
    predictive_adequacy = _reference_predictive_adequacy(
        observations=pre_observations,
        required_action_ids=list(dict.fromkeys(pre_actions)),
        predictives=predictives,
        feature_indices=feature_indices,
        maximum_mean_standardized_squared_error=float(
            predictive_contract[
                "maximum_mean_standardized_squared_error"
            ]
        ),
        maximum_per_action_mean_standardized_squared_error=float(
            predictive_contract[
                "maximum_per_action_mean_standardized_squared_error"
            ]
        ),
    )
    reference_ages = {
        action_id: change_time - reference_index
        for action_id, (_observation, reference_index) in references.items()
    }
    reference_certificate = reference_sufficiency_certificate(
        pre_change_action_ids=pre_actions,
        required_reference_action_ids=reference_action_ids,
        declared_relation_groups={
            str(key): value
            for key, value in job["declared_relation_groups"].items()
        },
        minimum_reference_experiments=minimum_reference_experiments,
        predictive_information_non_saturated=bool(
            job["predictive_information_non_saturated"]
        ),
        predictive_adequacy=predictive_adequacy,
        universal_candidate_family_scope=(
            job.get("reference_scope")
            == "universal_all_declared_candidate_families_without_truth_conditioning"
        ),
        maximum_reference_age_experiments=int(
            predictive_contract["maximum_reference_age_experiments"]
        ),
        observed_reference_ages=reference_ages,
    )
    with PublicCampaignObservationSession(
        task_id=str(job["task_id"]),
        seed=int(job["world_seed"]),
        interventions=job["interventions"],
        action_library=action_library,
        experiment_horizon=post_budget,
        observation_seed=int(job["post_observation_seed"]),
    ) as post_session:
        for phase_index in range(1, post_budget + 1):
            take_policy_step(
                post_session,
                phase_index=phase_index,
                update_sink=post_updates,
                refresh_old_world_reference=False,
            )
            post_actions.append(post_updates[-1]["action_id"])
            if phase_index in post_budgets:
                posterior_by_post_budget[str(phase_index)] = dict(family_posterior)
        post_noise_audit = list(post_session.observation_noise_audit)

    observation_noise_audit = {
        "schema_version": "chemworld-mechanism-paired-noise-audit-0.1",
        "mode": "keyed",
        "pre_change": pre_noise_audit,
        "post_checkpoint": post_noise_audit,
    }
    observation_noise_audit["bundle_sha256"] = canonical_sha256(
        observation_noise_audit
    )

    return {
        "task_id": str(job["task_id"]),
        "truth_id": str(job["truth_id"]),
        "world_seed": int(job["world_seed"]),
        "statistical_cluster_id": (
            f"{job['task_id']}:{int(job['world_seed'])}"
        ),
        "change_time": change_time,
        "truth_change_time": (
            "never" if str(job["truth_id"]) == "no_change" else change_time
        ),
        "evaluator_pseudo_checkpoint": (
            change_time if str(job["truth_id"]) == "no_change" else None
        ),
        "paired_runtime_contract": {
            "same_world_seed_across_truth_twins": True,
            "same_pre_change_observation_seed_across_truth_twins": True,
            "same_post_checkpoint_observation_seed_across_truth_twins": True,
            "common_random_numbers_across_truth_twins": True,
            "keyed_observation_noise": True,
            "sequential_rng_position_used_for_pairing": False,
            "identical_pre_post_session_boundary": True,
            "pseudo_checkpoint_runtime_side_effect": False,
            "reset_or_instance_identifier_exposed_to_policy": False,
        },
        "policy_received_change_time": False,
        "policy_received_change_time_support": False,
        "policy_received_minimum_stable_prefix": False,
        "policy_received_reference_certificate": False,
        "actual_pre_change_experiment_count": change_time,
        "reference_sufficiency_certificate": reference_certificate,
        "pre_change_reference_action_ids": actual_pre_change_reference_ids,
        "all_observation_reference_action_ids": list(references),
        "pre_change_actions": pre_actions,
        "post_change_actions": post_actions,
        "pre_change_updates": pre_updates,
        "post_change_updates": post_updates,
        "observation_noise_audit": observation_noise_audit,
        "predictions_by_post_budget": {
            budget: max(posterior, key=posterior.__getitem__)
            for budget, posterior in posterior_by_post_budget.items()
        },
        "posterior_by_post_budget": posterior_by_post_budget,
    }


def validate_precomputed_design_audit(
    protocol: Mapping[str, Any],
    plan: Mapping[str, Any],
    report: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate a frozen design audit before reusing it in an expensive Gate run."""

    expected = {
        "pass": True,
        "protocol_id": protocol.get("protocol_id"),
        "gate_a_plan_id": plan.get("plan_id"),
        "protocol_sha256": canonical_sha256(protocol),
        "gate_a_plan_sha256": canonical_sha256(plan),
        "diagnostic_relation_graph_sha256": plan[
            "diagnostic_relation_graph"
        ]["expected_graph_sha256"],
    }
    mismatches = [key for key, value in expected.items() if report.get(key) != value]
    if mismatches:
        raise ValueError(
            "precomputed mechanism design audit is stale or failed: " + ", ".join(mismatches)
        )
    return dict(report)


def _validate_release_qualification(
    protocol: Mapping[str, Any],
    plan: Mapping[str, Any],
    repository_root: Path,
) -> dict[str, Any] | None:
    contract = plan.get("release_qualification")
    if contract is None:
        return None
    if (
        not isinstance(contract, Mapping)
        or contract.get("required_before_formal_a2_or_a3") is not True
        or not isinstance(contract.get("report"), str)
    ):
        raise ValueError("invalid release-qualification contract")
    report = load_json_object(repository_root / str(contract["report"]))
    expected = {
        "qualified": True,
        "protocol_sha256": canonical_sha256(protocol),
        "gate_a_plan_sha256": canonical_sha256(plan),
        "formal_result": False,
        "formal_cohorts_consumed": False,
    }
    mismatches = [
        key for key, value in expected.items() if report.get(key) != value
    ]
    if mismatches:
        raise ValueError(
            "mechanism release qualification is missing, failed, or stale: "
            + ", ".join(mismatches)
        )
    return report


def _validate_frozen_registration(
    protocol: Mapping[str, Any],
    plan: Mapping[str, Any],
    repository_root: Path,
) -> None:
    graph_contract = plan.get("diagnostic_relation_graph")
    sample_contract = plan.get("sample_size_audit")
    preregistration_contract = plan.get("preregistration")
    if not all(
        isinstance(item, Mapping)
        for item in (
            graph_contract,
            sample_contract,
            preregistration_contract,
        )
    ):
        raise ValueError(
            "formal execution requires graph, sample-size, and "
            "preregistration contracts"
        )
    graph = load_json_object(
        repository_root / str(graph_contract["report"])  # type: ignore[index]
    )
    graph_errors = validate_diagnostic_relation_graph(protocol, plan, graph)
    if graph_errors:
        raise ValueError(
            "invalid diagnostic relation graph: " + "; ".join(graph_errors)
        )
    sample_size = load_json_object(
        repository_root / str(sample_contract["report"])  # type: ignore[index]
    )
    if sample_size.get("pass") is not True:
        raise ValueError("formal execution sample-size audit is missing or failed")
    preregistration = load_json_object(
        repository_root / str(preregistration_contract["manifest"])  # type: ignore[index]
    )
    errors = validate_mechanism_preregistration(
        preregistration,
        repository_root=repository_root,
        protocol=protocol,
        plan=plan,
        relation_graph=graph,
        sample_size_audit=sample_size,
    )
    if errors:
        raise ValueError(
            "invalid mechanism-adaptation preregistration: "
            + "; ".join(errors)
        )


def _paired_noise_coordinate_map(
    noise_audit: Mapping[str, Any],
) -> dict[str, str]:
    """Index truth-invariant noise keys by shared semantic coordinates."""

    indexed: dict[str, str] = {}
    for phase in ("pre_change", "post_checkpoint"):
        experiments = noise_audit.get(phase)
        if not isinstance(experiments, list):
            raise ValueError("paired-noise audit phase must be a list")
        for experiment in experiments:
            if not isinstance(experiment, Mapping):
                raise ValueError("paired-noise experiment entry must be an object")
            operation_noise = experiment.get("operation_noise")
            if not isinstance(operation_noise, list):
                raise ValueError("paired-noise operation list is missing")
            for operation_slot, provenance in enumerate(operation_noise):
                if (
                    not isinstance(provenance, Mapping)
                    or provenance.get("mode") != "keyed"
                    or provenance.get("sequential_rng_position_used") is not False
                    or not isinstance(provenance.get("noise_key_sha256"), str)
                    or not isinstance(provenance.get("coordinate"), Mapping)
                ):
                    raise ValueError(
                        "paired execution requires keyed semantic noise provenance"
                    )
                coordinate = {
                    "phase": phase,
                    "experiment_slot": experiment.get("experiment_index"),
                    "operation_slot": operation_slot,
                    "coordinate": provenance["coordinate"],
                }
                coordinate_id = f"{phase}:{canonical_sha256(coordinate)}"
                noise_key = str(provenance["noise_key_sha256"])
                previous = indexed.setdefault(coordinate_id, noise_key)
                if previous != noise_key:
                    raise ValueError(
                        "one trial reused a semantic noise coordinate with two keys"
                    )
    return indexed


def run_online_attainability_certificate(
    protocol: Mapping[str, Any],
    plan: Mapping[str, Any],
    *,
    design_validity_audit: Mapping[str, Any] | None = None,
    progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
    trial_store_root: Path | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    """Execute the frozen reference policy's online-attainability certificate.

    This certificate proves that the benchmark admits one compliant online
    solution.  It does not evaluate, redefine, or excuse a participant Agent.
    """

    errors = validate_mechanism_adaptation_protocol(protocol)
    if errors:
        raise ValueError("invalid mechanism-adaptation protocol: " + "; ".join(errors))
    _validate_release_qualification(
        protocol,
        plan,
        configuration_root().parent,
    )
    requirement = plan.get("online_attainability_certificate")
    if not isinstance(requirement, Mapping):
        raise ValueError("Gate A plan has no online-attainability certificate contract")
    if (
        requirement.get("scientific_name")
        != "A3_online_attainability_certificate"
        or requirement.get("certification_subject")
        != "frozen_reference_diagnostic_policy"
        or requirement.get("participant_agent_evaluation") is not False
    ):
        raise ValueError(
            "A3 must certify benchmark attainability by a frozen reference "
            "diagnostic policy, not participant-Agent performance"
        )
    if requirement.get("hidden_change_time") is not True:
        raise ValueError("online-attainability certificate must hide change time")
    if requirement.get("phase_invariant_action_rule") is not True:
        raise ValueError("online-policy action rule must be phase invariant")
    if requirement.get("changed_mechanism_states_are_absorbing") is not True:
        raise ValueError("online-policy changed states must be absorbing")
    if requirement.get("policy_receives_phase_or_reset_indicator") is not False:
        raise ValueError("online policy must not receive a phase or reset indicator")
    for private_field in (
        "policy_receives_reference_certificate",
        "policy_receives_change_time_support",
        "policy_receives_minimum_stable_prefix",
    ):
        if requirement.get(private_field) is not False:
            raise ValueError(f"online policy visibility contract requires {private_field}=false")
    if requirement.get("no_change_condition_included") is not True:
        raise ValueError("Gate A3 must include a first-class no-change condition")
    track_id = str(
        requirement.get("evaluation_track_id", PRIMARY_CHANGE_TRACK_ID)
    )
    if protocol.get("schema_version") == "chemworld-mechanism-adaptation-protocol-0.3.0":
        track = evaluation_track(protocol, track_id)
        if track_id != PRIMARY_CHANGE_TRACK_ID:
            raise ValueError("Gate A3 must use the calibrated online change track")
        if track.get("controls_gate_a3") is not True:
            raise ValueError("calibrated online change track must control Gate A3")
        minimum_reference_experiments = int(
            track["minimum_stable_prefix_experiments"]
        )
        protocol_timing = track
    else:
        protocol_timing = protocol["design"]
        minimum_reference_experiments = int(
            protocol_timing["pre_change_experiments"]
        )
    change_times = [int(item) for item in requirement["change_time_candidates"]]
    truth_change_time_support = requirement.get("truth_change_time_support")
    if truth_change_time_support != ["never", *change_times]:
        raise ValueError(
            "Gate A3 truth change-time support must be never plus the numeric points"
        )
    protocol_change_times = {
        int(item)
        for item in protocol_timing["change_after_experiments"]
        if item != "never"
    }
    if not change_times or not set(change_times).issubset(protocol_change_times):
        raise ValueError("online-policy change times must come from the protocol")
    if min(change_times) < minimum_reference_experiments:
        raise ValueError(
            "Gate A3 change times cannot precede the stable reference prefix"
        )
    if (
        requirement.get("changepoint_semantics")
        != "tau_is_completed_old_world_experiment_count"
    ):
        raise ValueError("Gate A3 must freeze completed-experiment changepoint semantics")
    post_budgets = sorted(
        {int(item) for item in requirement["post_change_experiment_budget_checkpoints"]}
    )
    online_gate_budget = int(requirement["online_policy_gate_budget"])
    controlled_primary_budget = int(plan["held_out_certificate"]["primary_gate_budget"])
    gate_a = protocol["gates"]["gate_a"]
    if post_budgets != sorted(
        {
            int(item)
            for item in gate_a[
                "a3_online_checkpoint_experiments_after_change"
            ]
        }
    ):
        raise ValueError("online-policy checkpoints must match the frozen Gate A checkpoints")
    if (
        online_gate_budget not in post_budgets
        or online_gate_budget
        != int(gate_a["a3_primary_checkpoint_experiments_after_change"])
    ):
        raise ValueError(
            "online-policy gate budget must equal the frozen A3 primary checkpoint"
        )
    if int(requirement["controlled_matched_primary_budget"]) != controlled_primary_budget:
        raise ValueError("online plan must bind the controlled primary budget")
    ranking_batch_size = int(requirement["information_ranking_batch_size"])
    if (
        protocol.get("schema_version")
        == "chemworld-mechanism-adaptation-protocol-0.3.0"
        and not 1 <= ranking_batch_size <= minimum_reference_experiments
    ):
        raise ValueError(
            "online information-ranking batch must fit inside the stable prefix"
        )
    if (
        protocol.get("schema_version")
        != "chemworld-mechanism-adaptation-protocol-0.3.0"
        and ranking_batch_size != minimum_reference_experiments
    ):
        raise ValueError(
            "online-policy information-ranking batch size must equal the protocol pre-change budget"
        )
    policy_action_count = int(requirement["online_policy_action_count"])
    _validate_online_policy_action_count(
        action_count=policy_action_count,
        gate_budget=(
            minimum_reference_experiments
            if protocol.get("schema_version")
            == "chemworld-mechanism-adaptation-protocol-0.3.0"
            else online_gate_budget
        ),
    )
    if protocol.get("schema_version") == "chemworld-mechanism-adaptation-protocol-0.3.0":
        if requirement.get("all_declared_relations_required_before_change") is not True:
            raise ValueError("Gate A3 must require all declared diagnostic relations")
        if (
            requirement.get("relation_closure_controls_reference_certificate")
            is not True
            or requirement.get(
                "canonical_witness_action_set_controls_reference_certificate"
            )
            is not False
        ):
            raise ValueError(
                "Gate A3 reference sufficiency must use relation closure, not "
                "canonical recipe-ID closure"
            )
    raw_hazards = requirement.get("change_hazard_by_allowed_change_point")
    if not isinstance(raw_hazards, Mapping):
        raise ValueError("Gate A3 must freeze a hazard for every allowed change point")
    hazards = {str(key): float(value) for key, value in raw_hazards.items()}
    if set(hazards) != {str(item) for item in change_times} or any(
        not 0.0 < value <= 1.0 for value in hazards.values()
    ):
        raise ValueError("Gate A3 change hazards do not match its numeric support")
    remaining_no_change_mass = 1.0
    prior_mass: list[float] = []
    for change_time in change_times:
        point_hazard = hazards[str(change_time)]
        prior_mass.append(remaining_no_change_mass * point_hazard)
        remaining_no_change_mass *= 1.0 - point_hazard
    prior_mass.append(remaining_no_change_mass)
    if max(prior_mass) - min(prior_mass) > 1e-12:
        raise ValueError(
            "Gate A3 hazards must give never and each numeric change time equal prior mass"
        )

    predictive_reference_contract = requirement.get(
        "reference_predictive_adequacy"
    )
    if not isinstance(predictive_reference_contract, Mapping):
        raise ValueError("Gate A3 must freeze predictive reference criteria")
    frozen_pass_criteria = requirement.get("frozen_pass_criteria")
    if not isinstance(frozen_pass_criteria, Mapping):
        raise ValueError("Gate A3 must freeze pass criteria before execution")
    probability_criteria = (
        "change_decision_probability_threshold",
        "minimum_reference_acquisition_rate_wilson_lower_bound",
        "minimum_changed_detection_recall_cluster_bootstrap_lower_bound",
        "maximum_no_change_false_positive_rate_cluster_bootstrap_upper_bound",
        "minimum_change_detection_auroc_cluster_bootstrap_lower_bound",
        "maximum_integrated_mean_change_probability_brier_score",
        "minimum_attribution_given_detection_and_reference_cluster_bootstrap_lower_bound",
        "minimum_end_to_end_success_rate_cluster_bootstrap_lower_bound",
    )
    if any(
        not 0.0 <= float(frozen_pass_criteria[field]) <= 1.0
        for field in probability_criteria
    ):
        raise ValueError("Gate A3 frozen probability criteria must be in [0, 1]")
    if (
        int(frozen_pass_criteria.get("cluster_bootstrap_draws", 0)) < 2_000
        or not isinstance(
            frozen_pass_criteria.get("cluster_bootstrap_seed_rule"),
            str,
        )
    ):
        raise ValueError(
            "Gate A3 must freeze at least 2000 cluster-bootstrap draws and "
            "a deterministic seed rule"
        )
    if (
        float(predictive_reference_contract["maximum_mean_standardized_squared_error"])
        <= 0.0
        or float(
            predictive_reference_contract[
                "maximum_per_action_mean_standardized_squared_error"
            ]
        )
        <= 0.0
    ):
        raise ValueError("Gate A3 predictive reference thresholds must be positive")
    if (
        predictive_reference_contract.get("cross_fitting")
        != "leave_one_experiment_out_within_campaign_pre_change_only"
        or predictive_reference_contract.get(
            "post_change_observations_allowed"
        )
        is not False
        or predictive_reference_contract.get("truth_family_allowed")
        is not False
    ):
        raise ValueError(
            "Gate A3 predictive reference must use within-campaign pre-change "
            "cross-fitting without post-change data or truth labels"
        )
    frozen_checkpoints = frozen_pass_criteria.get(
        "detection_checkpoint_experiments_after_change"
    )
    if (
        frozen_checkpoints != post_budgets
        or frozen_pass_criteria.get(
            "primary_detection_checkpoint_experiments_after_change"
        )
        != online_gate_budget
        or frozen_pass_criteria.get("undetected_campaign_handling")
        != f"right_censored_at_k_{online_gate_budget}"
    ):
        raise ValueError(
            "Gate A3 must freeze time-resolved checkpoints, primary horizon, "
            "and right-censoring"
        )
    cohort_partition = plan.get("cohort_partition")
    if not isinstance(cohort_partition, Mapping):
        raise ValueError("Gate A plan must declare disjoint cohort partitions")
    cohort_namespaces = {
        str(cohort_id): int(cohort["seed_namespace_start"])
        for cohort_id, cohort in cohort_partition.items()
        if isinstance(cohort, Mapping) and "seed_namespace_start" in cohort
    }
    legacy_cohorts = {
        "development",
        "a2_certification",
        "a3_certification",
        "private_confirmation",
    }
    split_private_cohorts = {
        "development",
        "a2_certification",
        "a3_certification",
        "private_environment_confirmation",
        "private_agent_confirmation",
    }
    if (
        set(cohort_namespaces) not in (legacy_cohorts, split_private_cohorts)
        or len(set(cohort_namespaces.values())) != len(cohort_namespaces)
        or cohort_partition.get("cross_cohort_world_seed_reuse_allowed") is not False
    ):
        raise ValueError("Gate A cohort namespaces must be complete and disjoint")
    if int(requirement["seed_namespace_start"]) != cohort_namespaces["a3_certification"]:
        raise ValueError("online certificate must use only the A3 cohort namespace")
    pairing = plan.get("paired_phase_design")
    if not isinstance(pairing, Mapping) or any(
        pairing.get(field) is not expected
        for field, expected in {
            "same_hidden_world_seed_across_phases": True,
            "same_world_seed_reused_across_candidate_twins": True,
            "common_random_numbers_across_changed_no_change_twins": True,
            "identical_reset_rule_across_changed_no_change_twins": True,
            "reset_or_instance_identifier_agent_visible": False,
            "pseudo_checkpoint_has_runtime_side_effect": False,
            "campaign_metadata_invariant_across_checkpoint": True,
        }.items()
    ):
        raise ValueError(
            "Gate A3 changed/no-change twins must freeze matched world, reset, "
            "metadata, and common-random-number controls"
        )
    graph_contract = plan.get("diagnostic_relation_graph")
    if not isinstance(graph_contract, Mapping):
        raise ValueError("Gate A plan must bind a diagnostic relation graph")
    graph_path = configuration_root().parent / str(graph_contract["report"])
    relation_graph = load_json_object(graph_path)
    graph_errors = validate_diagnostic_relation_graph(
        protocol,
        plan,
        relation_graph,
    )
    if graph_errors:
        raise ValueError("invalid diagnostic relation graph: " + "; ".join(graph_errors))
    repository_root = configuration_root().parent
    sample_size_contract = plan.get("sample_size_audit")
    preregistration_contract = plan.get("preregistration")
    if not isinstance(sample_size_contract, Mapping) or not isinstance(
        preregistration_contract,
        Mapping,
    ):
        raise ValueError(
            "Gate A3 requires sample-size and preregistration contracts"
        )
    sample_size_audit = load_json_object(
        repository_root / str(sample_size_contract["report"])
    )
    if (
        sample_size_audit.get("pass") is not True
        or int(
            sample_size_audit[
                "selected_independent_world_clusters_per_family"
            ]
        )
        != int(requirement["world_seeds_per_family"])
    ):
        raise ValueError("Gate A3 sample-size audit is missing or failed")
    preregistration = load_json_object(
        repository_root / str(preregistration_contract["manifest"])
    )
    preregistration_errors = validate_mechanism_preregistration(
        preregistration,
        repository_root=repository_root,
        protocol=protocol,
        plan=plan,
        relation_graph=relation_graph,
        sample_size_audit=sample_size_audit,
    )
    if preregistration_errors:
        raise ValueError(
            "invalid mechanism-adaptation preregistration: "
            + "; ".join(preregistration_errors)
        )
    action_plan = plan["action_library"]
    fit_plan = plan["candidate_predictive_fit"]
    certificate_plan = plan["held_out_certificate"]
    gate = protocol["gates"]["gate_a"]
    feature_encoding = str(fit_plan["public_feature_encoding"])
    if feature_encoding != PUBLIC_EXPERIMENT_FEATURE_ENCODING_VERSION:
        raise ValueError("online public-feature encoding contract is unsupported")
    feature_block_size = int(fit_plan["summary_block_size"])
    if feature_block_size != PUBLIC_EXPERIMENT_FEATURE_BLOCK_SIZE:
        raise ValueError("online public-feature block-size contract is unsupported")
    discriminative_block_count = int(fit_plan["online_discriminative_block_count"])
    raw_relational_evidence = fit_plan.get(
        "online_relational_evidence_enabled",
        False,
    )
    if not isinstance(raw_relational_evidence, bool):
        raise ValueError("online relational evidence flag must be boolean")
    relational_evidence_enabled = raw_relational_evidence
    if relational_evidence_enabled:
        if requirement.get("declared_relational_action_coverage") is not True:
            raise ValueError(
                "relational online evidence requires declared action-group coverage"
            )
        if requirement.get("one_likelihood_channel_per_experiment") is not True:
            raise ValueError(
                "relational online evidence must use one likelihood channel per experiment"
            )
    selected_feature_dimension = discriminative_block_count * feature_block_size
    dimension_normalized_likelihood = _uses_dimension_normalized_likelihood(
        fit_plan
    )
    if dimension_normalized_likelihood:
        likelihood_effective_dimension_count = float(
            fit_plan["likelihood_effective_dimension_count"]
        )
        online_likelihood_scale = _dimension_normalized_likelihood_scale(
            effective_dimension_count=likelihood_effective_dimension_count,
            selected_feature_dimension=selected_feature_dimension,
        )
        online_batch_likelihood_scale = _dimension_normalized_likelihood_scale(
            effective_dimension_count=likelihood_effective_dimension_count,
            selected_feature_dimension=selected_feature_dimension * ranking_batch_size,
        )
    else:
        likelihood_effective_dimension_count = float(selected_feature_dimension)
        online_likelihood_scale = float(fit_plan["online_likelihood_scale"])
        online_batch_likelihood_scale = online_likelihood_scale
    online_batch_information_minimum_spread = float(
        fit_plan.get("online_batch_information_minimum_spread_nats", 0.0)
    )
    maximum_likelihood_scale = 1.0 / feature_block_size
    if (
        not np.isfinite(online_likelihood_scale)
        or online_likelihood_scale <= 0.0
        or online_likelihood_scale > maximum_likelihood_scale
        or not np.isfinite(online_batch_likelihood_scale)
        or online_batch_likelihood_scale <= 0.0
        or online_batch_likelihood_scale > maximum_likelihood_scale
    ):
        raise ValueError(
            "online likelihood scale must be positive and no greater than one "
            "effective contribution per public summary block"
        )
    if (
        not np.isfinite(online_batch_information_minimum_spread)
        or online_batch_information_minimum_spread < 0.0
        or (
            dimension_normalized_likelihood
            and online_batch_information_minimum_spread == 0.0
        )
    ):
        raise ValueError(
            "online batch information minimum spread must be nonnegative "
            "and positive for dimension-normalized likelihoods"
        )
    action_libraries = {
        str(task_id): build_action_library(
            str(task_id),
            action_count=int(action_plan["action_count_per_task"]),
            seed=int(action_plan["design_seed"]),
            design_id=str(
                action_plan.get("design", DIAGNOSTIC_RECIPE_DESIGN_V1)
            ),
        )
        for task_id in protocol["design"]["tasks"]
    }
    if design_validity_audit is None:
        design_audit = audit_mechanism_design(
            protocol,
            plan,
            action_libraries=action_libraries,
        )
    else:
        design_audit = validate_precomputed_design_audit(
            protocol,
            plan,
            design_validity_audit,
        )
    if not design_audit["pass"]:
        raise ValueError("online-policy certificate requires a passing design audit")

    truths_by_budget: dict[int, list[str]] = {budget: [] for budget in post_budgets}
    predictions_by_budget: dict[int, list[str]] = {budget: [] for budget in post_budgets}
    sufficient_truths_by_budget: dict[int, list[str]] = {
        budget: [] for budget in post_budgets
    }
    sufficient_predictions_by_budget: dict[int, list[str]] = {
        budget: [] for budget in post_budgets
    }
    all_online_trials: list[dict[str, Any]] = []
    online_trial_manifests: dict[str, dict[str, Any]] = {}
    task_reports: dict[str, Any] = {}
    sample_count = int(fit_plan["samples_per_candidate_action"])
    world_seeds_per_family = int(requirement["world_seeds_per_family"])
    a3_seed_namespace_start = int(requirement["seed_namespace_start"])
    information_draws = int(certificate_plan["information_draws_per_batch"])
    qualified_candidates = [
        _qualified_candidate(str(task_id), str(candidate_id))
        for task_id in protocol["design"]["tasks"]
        for candidate_id in protocol["task_mechanism_contracts"][task_id]["candidate_ids"]
    ]

    for task_index, raw_task_id in enumerate(protocol["design"]["tasks"]):
        task_id = str(raw_task_id)
        contract = protocol["task_mechanism_contracts"][task_id]
        candidate_ids = [str(item) for item in contract["candidate_ids"]]
        action_library = action_libraries[task_id]
        action_ids = list(action_library)
        fit_world_seeds = [
            int(fit_plan["nuisance_seed_namespace_start"])
            + 50_000_000
            + task_index * 1_000_000
            + index
            for index in range(sample_count)
        ]
        fit_keys: list[tuple[str, str]] = []
        fit_jobs: list[dict[str, Any]] = []
        for candidate_id in candidate_ids:
            for action_id in action_ids:
                for world_seed in fit_world_seeds:
                    fit_keys.append((candidate_id, action_id))
                    fit_jobs.append(
                        {
                            "task_id": task_id,
                            "world_seed": world_seed,
                            "interventions": contract["interventions"][candidate_id],
                            "action_id": action_id,
                            "action_vector": action_library[action_id].tolist(),
                            "pre_observation_seed": _stable_seed(
                                world_seed,
                                f"online-policy:{action_id}:fit-pre",
                            ),
                            "stable_reference_observation_seed": _stable_seed(
                                world_seed,
                                f"online-policy:{action_id}:fit-stable-reference",
                            ),
                            "post_observation_seed": _stable_seed(
                                world_seed,
                                f"online-policy:{action_id}:fit-post",
                            ),
                        }
                    )
        _emit_gate_a_progress(
            progress_callback,
            event="online_predictive_fit_started",
            task_id=task_id,
            job_count=len(fit_jobs),
        )
        if trial_store_root is None:
            fit_results = _execute_jobs(
                _sample_online_predictive_job,
                fit_jobs,
                workers=int(fit_plan.get("execution_workers", 1)),
            )
        else:
            online_fit_keys = [
                ConfirmatoryTrialKey(
                    task_id=task_id,
                    truth_family=candidate_id,
                    world_cluster=str(job["world_seed"]),
                    changepoint="predictive_fit",
                    arm=f"online_action:{job['action_id']}",
                )
                for (candidate_id, _action_id), job in zip(
                    fit_keys,
                    fit_jobs,
                    strict=True,
                )
            ]
            fit_results, fit_manifest = execute_jobs_resumable(
                _sample_online_predictive_job,
                fit_jobs,
                online_fit_keys,
                workers=int(fit_plan.get("execution_workers", 1)),
                store=ConfirmatoryTrialStore(
                    Path(trial_store_root) / task_id / "predictive-fit"
                ),
                resume=resume,
            )
            online_trial_manifests.setdefault(task_id, {})[
                "predictive_fit"
            ] = fit_manifest
        _emit_gate_a_progress(
            progress_callback,
            event="online_predictive_fit_completed",
            task_id=task_id,
            job_count=len(fit_jobs),
        )
        transition_samples: dict[
            tuple[str, str],
            list[Sequence[float]],
        ] = {
            (candidate_id, action_id): []
            for candidate_id in candidate_ids
            for action_id in action_ids
        }
        stable_samples: dict[
            tuple[str, str],
            list[Sequence[float]],
        ] = {
            (candidate_id, action_id): []
            for candidate_id in candidate_ids
            for action_id in action_ids
        }
        absolute_samples: dict[tuple[str, str], list[Sequence[float]]] = {
            (candidate_id, action_id): []
            for candidate_id in candidate_ids
            for action_id in action_ids
        }
        for key, values in zip(fit_keys, fit_results, strict=True):
            transition_samples[key].append(values["transition_contrast"])
            stable_samples[key].append(values["stable_contrast"])
            absolute_samples[key].append(values["absolute_trace"])

        evidence_action_ids = [
            _online_evidence_channel(encoding, action_id)
            for action_id in action_ids
            for encoding in (
                "transition_contrast",
                "stable_contrast",
                "absolute_trace",
            )
        ]
        relational_samples: dict[
            tuple[str, str],
            list[list[float]],
        ] = {}
        if relational_evidence_enabled:
            relational_samples = _relational_evidence_samples(
                candidate_ids=candidate_ids,
                action_ids=action_ids,
                transition_samples=transition_samples,
                stable_samples=stable_samples,
                absolute_samples=absolute_samples,
            )
            evidence_action_ids.extend(
                _online_relational_evidence_channel(
                    encoding,
                    action_id,
                    reference_action_id,
                )
                for action_id, reference_action_id in itertools.permutations(
                    action_ids,
                    2,
                )
                for encoding in (
                    "relational_transition_contrast",
                    "relational_stable_contrast",
                )
            )
        evidence_samples: dict[
            tuple[str, str], Sequence[Sequence[float]]
        ] = {
            (
                candidate_id,
                _online_evidence_channel(encoding, action_id),
            ): (
                transition_samples[(candidate_id, action_id)]
                if encoding == "transition_contrast"
                else (
                    stable_samples[(candidate_id, action_id)]
                    if encoding == "stable_contrast"
                    else absolute_samples[(candidate_id, action_id)]
                )
            )
            for candidate_id in candidate_ids
            for action_id in action_ids
            for encoding in (
                "transition_contrast",
                "stable_contrast",
                "absolute_trace",
            )
        }
        evidence_samples.update(relational_samples)
        feature_indices, feature_selection_report = _select_discriminative_feature_blocks(
            evidence_samples,
            candidate_ids=candidate_ids,
            evidence_action_ids=evidence_action_ids,
            block_size=feature_block_size,
            block_count=discriminative_block_count,
            variance_floor=float(fit_plan["variance_floor"]),
        )
        reduced_evidence_samples = {
            key: [
                np.asarray(sample, dtype=float)[
                    np.asarray(feature_indices[key[1]], dtype=int)
                ].tolist()
                for sample in samples
            ]
            for key, samples in evidence_samples.items()
        }
        reduced_transition_samples = {
            (candidate_id, action_id): reduced_evidence_samples[
                (
                    candidate_id,
                    _online_evidence_channel("transition_contrast", action_id),
                )
            ]
            for candidate_id in candidate_ids
            for action_id in action_ids
        }
        selection_oracle, batches = _fit_paired_batch_oracle(
            candidate_ids=candidate_ids,
            action_ids=action_ids,
            per_action_samples=reduced_transition_samples,
            batch_size=ranking_batch_size,
            variance_floor=float(fit_plan["variance_floor"]),
            likelihood_scale=online_batch_likelihood_scale,
            seed=_stable_seed(
                int(fit_plan["oracle_seed"]),
                f"{task_id}:online-batch-selection",
            ),
        )
        batch_information = selection_oracle.expected_information_by_action(draws=information_draws)
        selected_batch_id, batch_information_diagnostics = _select_information_maximum(
            batch_information,
            minimum_spread_nats=online_batch_information_minimum_spread,
            context=f"{task_id} online batch",
        )
        selected_actions = list(batches[selected_batch_id])
        online_oracle = GaussianMechanismOracle(
            candidate_ids=candidate_ids,
            action_ids=evidence_action_ids,
            sample_public_observation=_unreachable_sample,
            samples_per_candidate=4,
            variance_floor=float(fit_plan["variance_floor"]),
            likelihood_scale=online_likelihood_scale,
            seed=_stable_seed(
                int(fit_plan["oracle_seed"]),
                f"{task_id}:online-policy",
            ),
        )
        online_oracle.fit_predictives_from_samples(reduced_evidence_samples)
        evidence_information = online_oracle.expected_information_by_action(draws=information_draws)
        selected_actions.sort(
            key=lambda item: (
                -evidence_information[
                    _online_evidence_channel("transition_contrast", item)
                ],
                item,
            ),
        )
        remaining_actions = [
            action_id for action_id in action_ids if action_id not in selected_actions
        ]
        remaining_actions.sort(
            key=lambda item: (
                -evidence_information[
                    _online_evidence_channel("transition_contrast", item)
                ],
                item,
            ),
        )
        relational_coverage_actions: list[str] = []
        relational_coverage_report: dict[str, Any] = {
            "required": False,
            "declarations": {},
            "selected_groups": {},
            "selected_action_ids": [],
        }
        if relational_evidence_enabled:
            declaration_groups = _declared_relational_action_groups(
                task_id=task_id,
                contract=contract,
                action_library=action_library,
            )
            (
                relational_coverage_actions,
                relational_coverage_report,
            ) = _select_relational_coverage_actions(
                canonical_action_ids=action_ids,
                declaration_groups=declaration_groups,
                evidence_information=evidence_information,
                action_count=policy_action_count,
            )
        policy_actions = _canonical_policy_cycle(
            canonical_action_ids=action_ids,
            ranked_action_ids=(
                relational_coverage_actions + selected_actions + remaining_actions
            ),
            action_count=policy_action_count,
        )
        relational_reference_priority = (
            _relational_reference_priorities(
                action_ids=policy_actions,
                evidence_information=evidence_information,
            )
            if relational_evidence_enabled
            else {}
        )
        declared_relational_reference_actions = (
            _declared_relational_reference_actions(
                action_ids=policy_actions,
                selected_groups=relational_coverage_report["selected_groups"],
                evidence_information=evidence_information,
            )
            if relational_evidence_enabled
            else {}
        )
        serialized_actions = {
            action_id: action_library[action_id].tolist() for action_id in action_ids
        }
        assigned_change_times = _balanced_hidden_change_times(
            change_times,
            trial_count=world_seeds_per_family,
            seed=a3_seed_namespace_start,
            task_id=task_id,
        )
        trial_jobs: list[dict[str, Any]] = []
        for candidate_id in candidate_ids:
            for repeat_index in range(world_seeds_per_family):
                world_seed = (
                    a3_seed_namespace_start
                    + 100_000_000
                    + task_index * 1_000_000
                    + repeat_index
                )
                change_time = assigned_change_times[repeat_index]
                trial_jobs.append(
                    {
                        "task_id": task_id,
                        "truth_id": candidate_id,
                        "world_seed": world_seed,
                        "change_time": change_time,
                        "change_time_candidates": change_times,
                        "post_change_budget_checkpoints": post_budgets,
                        "minimum_reference_experiments": (
                            minimum_reference_experiments
                        ),
                        "enforce_reference_sufficiency": (
                            protocol.get("schema_version")
                            == "chemworld-mechanism-adaptation-protocol-0.3.0"
                        ),
                        "change_hazards": hazards,
                        "interventions": contract["interventions"][candidate_id],
                        "candidate_ids": candidate_ids,
                        "action_ids": action_ids,
                        "reference_action_ids": policy_actions,
                        "action_library": serialized_actions,
                        "predictives": online_oracle.export_predictives(),
                        "feature_indices": {
                            key: list(value) for key, value in feature_indices.items()
                        },
                        "relational_evidence_enabled": relational_evidence_enabled,
                        "relational_reference_priority": relational_reference_priority,
                        "declared_relational_reference_actions": (
                            declared_relational_reference_actions
                        ),
                        "declared_relation_groups": declaration_groups,
                        "predictive_information_non_saturated": (
                            batch_information_diagnostics["non_saturated"]
                        ),
                        "reference_scope": requirement["reference_scope"],
                        "reference_predictive_adequacy": (
                            predictive_reference_contract
                        ),
                        "likelihood_scale": online_likelihood_scale,
                        "pre_observation_seed": _stable_seed(
                            world_seed,
                            "online-policy:pre",
                        ),
                        "post_observation_seed": _stable_seed(
                            world_seed,
                            "online-policy:post",
                        ),
                    }
                )
        _emit_gate_a_progress(
            progress_callback,
            event="online_certificate_trials_started",
            task_id=task_id,
            job_count=len(trial_jobs),
        )
        trial_keys = [
            ConfirmatoryTrialKey(
                task_id=task_id,
                truth_family=str(job["truth_id"]),
                world_cluster=str(job["world_seed"]),
                changepoint=(
                    "never"
                    if str(job["truth_id"]) == "no_change"
                    else int(job["change_time"])
                ),
                arm="online_reference_policy",
            )
            for job in trial_jobs
        ]
        if trial_store_root is None:
            completed = _execute_jobs(
                _execute_online_policy_trial_job,
                trial_jobs,
                workers=int(requirement.get("execution_workers", 1)),
            )
        else:
            completed, task_trial_manifest = execute_jobs_resumable(
                _execute_online_policy_trial_job,
                trial_jobs,
                trial_keys,
                workers=int(requirement.get("execution_workers", 1)),
                store=ConfirmatoryTrialStore(
                    Path(trial_store_root) / task_id
                ),
                resume=resume,
            )
            online_trial_manifests.setdefault(task_id, {})[
                "certificate"
            ] = task_trial_manifest
        all_online_trials.extend(completed)
        _emit_gate_a_progress(
            progress_callback,
            event="online_certificate_trials_completed",
            task_id=task_id,
            job_count=len(trial_jobs),
        )
        for row in completed:
            row_reference_certificate = dict(
                row["reference_sufficiency_certificate"]
            )
            qualified_truth = _qualified_candidate(
                task_id,
                str(row["truth_id"]),
            )
            for budget in post_budgets:
                truths_by_budget[budget].append(qualified_truth)
                prediction = row["predictions_by_post_budget"][str(budget)]
                qualified_prediction = _qualified_candidate(
                    task_id,
                    str(prediction),
                )
                predictions_by_budget[budget].append(qualified_prediction)
                if row_reference_certificate["pass"] is True:
                    sufficient_truths_by_budget[budget].append(qualified_truth)
                    sufficient_predictions_by_budget[budget].append(
                        qualified_prediction
                    )
        task_reports[task_id] = {
            "candidate_ids": candidate_ids,
            "selected_reference_action_ids": selected_actions,
            "online_policy_action_ids": policy_actions,
            "minimum_stable_prefix_experiments": minimum_reference_experiments,
            "reference_sufficient_candidate_arm_count": sum(
                row["reference_sufficiency_certificate"]["pass"] is True
                for row in completed
            ),
            "reference_candidate_arm_count": len(completed),
            "reference_statistical_cluster_count": len(
                {
                    str(row["statistical_cluster_id"])
                    for row in completed
                }
            ),
            "reference_sufficient_cluster_count": len(
                {
                    str(row["statistical_cluster_id"])
                    for row in completed
                    if row["reference_sufficiency_certificate"]["pass"] is True
                }
            ),
            "online_policy_action_set_selection": (
                "predictive_fit_information_ranked_before_held_out_execution"
            ),
            "online_policy_cycle_order": "canonical_frozen_action_library_order",
            "relational_evidence_enabled": relational_evidence_enabled,
            "relational_coverage": relational_coverage_report,
            "relational_reference_priority": relational_reference_priority,
            "declared_relational_reference_actions": (
                declared_relational_reference_actions
            ),
            "one_likelihood_channel_per_experiment": True,
            "selected_batch_id": selected_batch_id,
            "selected_batch_information_nats": float(batch_information[selected_batch_id]),
            "batch_information_nats": {
                key: float(value) for key, value in batch_information.items()
            },
            "batch_information_diagnostics": batch_information_diagnostics,
            "single_action_information_nats": {
                action_id: float(
                    evidence_information[
                        _online_evidence_channel(
                            "transition_contrast",
                            action_id,
                        )
                    ]
                )
                for action_id in action_ids
            },
            "evidence_channel_information_nats": {
                key: float(value) for key, value in evidence_information.items()
            },
            "public_feature_encoding": feature_encoding,
            "public_feature_block_size": feature_block_size,
            "online_discriminative_block_count": discriminative_block_count,
            "online_likelihood_scale": online_likelihood_scale,
            "online_batch_likelihood_scale": online_batch_likelihood_scale,
            "likelihood_effective_dimension_count": (
                likelihood_effective_dimension_count
            ),
            "selected_feature_dimension_per_evidence_channel": (
                selected_feature_dimension
            ),
            "online_batch_information_minimum_spread_nats": (
                online_batch_information_minimum_spread
            ),
            "feature_selection": feature_selection_report,
            "trials": completed,
        }

    unstratified_certificates_by_budget = {
        budget: identifiability_certificate(
            truths=truths_by_budget[budget],
            predictions=predictions_by_budget[budget],
            candidate_ids=qualified_candidates,
            overall_lower_bound_threshold=float(gate["overall_top1_wilson_lower_bound"]),
            family_recall_lower_bound_threshold=float(gate["per_family_recall_wilson_lower_bound"]),
        )
        for budget in post_budgets
    }
    reference_pass_by_cluster: dict[str, bool] = {}
    noise_coordinates_by_cluster: dict[str, dict[str, str]] = {}
    pre_noise_signatures_by_cluster: dict[str, set[str]] = {}
    noise_pairing_mismatches: list[str] = []
    pre_signature_mismatches: list[str] = []
    for row in all_online_trials:
        cluster_id = str(row["statistical_cluster_id"])
        passed = row["reference_sufficiency_certificate"]["pass"] is True
        previous = reference_pass_by_cluster.setdefault(cluster_id, passed)
        if previous != passed:
            raise ValueError(
                "paired candidate arms disagree on reference sufficiency"
            )
        noise_audit = row.get("observation_noise_audit")
        if not isinstance(noise_audit, Mapping) or not isinstance(
            noise_audit.get("bundle_sha256"),
            str,
        ):
            raise ValueError("online trial lacks keyed-noise provenance")
        coordinate_map = _paired_noise_coordinate_map(noise_audit)
        cluster_coordinates = noise_coordinates_by_cluster.setdefault(
            cluster_id,
            {},
        )
        for coordinate, noise_key in coordinate_map.items():
            previous_noise_key = cluster_coordinates.setdefault(
                coordinate,
                noise_key,
            )
            if previous_noise_key != noise_key:
                noise_pairing_mismatches.append(
                    f"{cluster_id}:{coordinate}"
                )
        pre_signatures = {
            coordinate
            for coordinate in coordinate_map
            if coordinate.startswith("pre_change:")
        }
        previous_pre_signatures = pre_noise_signatures_by_cluster.setdefault(
            cluster_id,
            pre_signatures,
        )
        if previous_pre_signatures != pre_signatures:
            pre_signature_mismatches.append(cluster_id)
    keyed_noise_pairing_audit = {
        "schema_version": "chemworld-keyed-noise-pairing-audit-0.1",
        "statistical_cluster_count": len(noise_coordinates_by_cluster),
        "shared_coordinate_key_mismatch_count": len(
            set(noise_pairing_mismatches)
        ),
        "shared_coordinate_key_mismatches": sorted(
            set(noise_pairing_mismatches)
        ),
        "pre_change_signature_set_mismatch_count": len(
            set(pre_signature_mismatches)
        ),
        "pre_change_signature_set_mismatch_cluster_ids": sorted(
            set(pre_signature_mismatches)
        ),
        "paired_noise_keys_match_on_shared_semantic_coordinates": (
            not noise_pairing_mismatches
        ),
        "pre_change_noise_coordinate_sets_match_across_truth_twins": (
            not pre_signature_mismatches
        ),
        "post_change_policy_divergence_may_change_coordinate_sets": True,
        "sequential_rng_position_used_for_pairing": False,
        "gate_precondition_pass": (
            not noise_pairing_mismatches and not pre_signature_mismatches
        ),
    }
    trial_manifests_complete = (
        None
        if trial_store_root is None
        else (
            len(online_trial_manifests) == len(task_reports)
            and all(
                len(task_manifests) == 2
                and all(
                    manifest["complete"] is True
                    for manifest in task_manifests.values()
                )
                for task_manifests in online_trial_manifests.values()
            )
        )
    )
    sufficient_trial_count = sum(reference_pass_by_cluster.values())
    reference_trial_count = len(reference_pass_by_cluster)
    reference_interval = (
        wilson_interval(sufficient_trial_count, reference_trial_count)
        if reference_trial_count
        else (0.0, 1.0)
    )
    minimum_reference_lower_bound = float(
        frozen_pass_criteria[
            "minimum_reference_acquisition_rate_wilson_lower_bound"
        ]
    )
    reference_gate_pass = (
        reference_trial_count > 0
        and reference_interval[0] >= minimum_reference_lower_bound
    )
    reference_acquisition_certificate = {
        "schema_version": "chemworld-reference-acquisition-summary-0.2",
        "status": "passed" if reference_gate_pass else "failed",
        "gate_pass": reference_gate_pass,
        "required_wilson_lower_bound": minimum_reference_lower_bound,
        "statistical_unit": (
            "task_id_and_world_seed; candidate truth arms sharing the same "
            "stable-prefix world contribute one reference-acquisition outcome"
        ),
        "statistical_cluster_count": reference_trial_count,
        "sufficient_cluster_count": sufficient_trial_count,
        "pass_rate": (
            sufficient_trial_count / reference_trial_count
            if reference_trial_count
            else 0.0
        ),
        "pass_rate_interval": list(reference_interval),
        "minimum_stable_prefix_experiments": minimum_reference_experiments,
        "failure_interpretation": (
            "excluded from conditional mechanism-attribution denominator but retained "
            "as failure in the end-to-end online-success denominator"
        ),
    }
    capability_chain_by_budget = {
        budget: _online_capability_chain_summary(
            trials=all_online_trials,
            post_change_budget=budget,
            frozen_criteria=frozen_pass_criteria,
            bootstrap_seed=_stable_seed(
                a3_seed_namespace_start,
                f"online-capability-chain:{budget}",
            ),
        )
        for budget in post_budgets
    }
    task_capability_chain_by_budget = {
        budget: {
            task_id: _online_capability_chain_summary(
                trials=[
                    row
                    for row in all_online_trials
                    if str(row["task_id"]) == task_id
                ],
                post_change_budget=budget,
                frozen_criteria=frozen_pass_criteria,
                bootstrap_seed=_stable_seed(
                    a3_seed_namespace_start,
                    f"online-capability-chain:{budget}:task:{task_id}",
                ),
            )
            for task_id in sorted(task_reports)
        }
        for budget in post_budgets
    }
    family_capability_chain_by_budget: dict[
        int, dict[str, dict[str, Any]]
    ] = {}
    for budget in post_budgets:
        family_capability_chain_by_budget[budget] = {}
        for task_id in sorted(task_reports):
            family_ids = sorted(
                {
                    str(row["truth_id"])
                    for row in all_online_trials
                    if str(row["task_id"]) == task_id
                    and str(row["truth_id"]) != "no_change"
                }
            )
            for family_id in family_ids:
                stratum_id = f"{task_id}:{family_id}"
                family_capability_chain_by_budget[budget][
                    stratum_id
                ] = _online_capability_chain_summary(
                    trials=[
                        row
                        for row in all_online_trials
                        if str(row["task_id"]) == task_id
                        and str(row["truth_id"])
                        in {"no_change", family_id}
                    ],
                    post_change_budget=budget,
                    frozen_criteria=frozen_pass_criteria,
                    bootstrap_seed=_stable_seed(
                        a3_seed_namespace_start,
                        (
                            f"online-capability-chain:{budget}:"
                            f"family:{stratum_id}"
                        ),
                    ),
                )
    if protocol.get("schema_version") != "chemworld-mechanism-adaptation-protocol-0.3.0":
        certificates_by_budget = unstratified_certificates_by_budget
    elif sufficient_trial_count:
        certificates_by_budget = {
            budget: identifiability_certificate(
                truths=sufficient_truths_by_budget[budget],
                predictions=sufficient_predictions_by_budget[budget],
                candidate_ids=qualified_candidates,
                overall_lower_bound_threshold=float(
                    gate["overall_top1_wilson_lower_bound"]
                ),
                family_recall_lower_bound_threshold=float(
                    gate["per_family_recall_wilson_lower_bound"]
                ),
            )
            for budget in post_budgets
        }
    else:
        certificates_by_budget = {
            budget: {
                "gate_pass": False,
                "status": "not_evaluable_without_reference_sufficient_trials",
                "sample_count": 0,
            }
            for budget in post_budgets
        }
    certificate = certificates_by_budget[online_gate_budget]
    capability_certificate = capability_chain_by_budget[online_gate_budget]
    task_capability_certificate = task_capability_chain_by_budget[
        online_gate_budget
    ]
    family_capability_certificate = family_capability_chain_by_budget[
        online_gate_budget
    ]
    task_intersection_pass = all(
        item["gate_pass"] is True
        for item in task_capability_certificate.values()
    )
    family_intersection_pass = all(
        item["gate_pass"] is True
        for item in family_capability_certificate.values()
    )
    def family_macro(
        values: Sequence[float | None],
    ) -> float | None:
        numeric_values = [
            float(value)
            for value in values
            if isinstance(value, int | float)
        ]
        return (
            float(np.mean(numeric_values))
            if numeric_values and len(numeric_values) == len(values)
            else None
        )

    family_macro_average = {
        "p_reference_sufficient": family_macro(
            [
                item["p_reference_sufficient"]
                for item in family_capability_certificate.values()
            ]
        ),
        "p_detection_given_reference_and_changed": family_macro(
            [
                item["changed_campaigns"][
                    "p_detection_given_reference_and_changed"
                ]
                for item in family_capability_certificate.values()
            ]
        ),
        "p_reference_detection_attribution_given_changed": family_macro(
            [
                item["changed_campaigns"][
                    "p_reference_detection_attribution_given_changed"
                ]
                for item in family_capability_certificate.values()
            ]
        ),
        "gate_pass": family_intersection_pass,
    }
    execution_binding = gate_a_execution_contract_binding(protocol, plan)
    gate_pass = (
        certificate["gate_pass"] is True
        and capability_certificate["gate_pass"] is True
        and task_intersection_pass
        and family_intersection_pass
        and keyed_noise_pairing_audit["gate_precondition_pass"] is True
        and trial_manifests_complete is not False
        and (
            reference_acquisition_certificate["gate_pass"] is True
            or protocol.get("schema_version")
            != "chemworld-mechanism-adaptation-protocol-0.3.0"
        )
    )
    return {
        "schema_version": str(
            requirement.get(
                "certificate_schema_version",
                ONLINE_ATTAINABILITY_CERTIFICATE_VERSION,
            )
        ),
        "certificate_scope": (
            "online_attainability_of_frozen_reference_diagnostic_policy"
        ),
        "status": "passed" if gate_pass else "failed",
        "gate_pass": gate_pass,
        "formal_benchmark_result": False,
        "certification_subject": "frozen_reference_diagnostic_policy",
        "participant_agent_evaluation": False,
        "participant_agents_are_evaluated_in": [
            "gate_b_change_detection",
            "gate_c_feedback_use",
            "gate_d_adaptation_recovery",
            "gate_e_procedural_autonomy",
        ],
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": canonical_sha256(protocol),
        "gate_a_plan_id": plan["plan_id"],
        "gate_a_plan_sha256": canonical_sha256(plan),
        "execution_contract_binding": execution_binding,
        "execution_contract_binding_sha256": execution_binding["binding_sha256"],
        "controlled_matched_primary_budget": controlled_primary_budget,
        "online_policy_gate_budget": online_gate_budget,
        "post_change_experiment_budget_checkpoints": post_budgets,
        "evaluation_track_id": track_id,
        "minimum_stable_prefix_experiments": minimum_reference_experiments,
        "hidden_change_time": True,
        "policy_received_change_time": False,
        "policy_received_change_time_support": False,
        "policy_received_minimum_stable_prefix": False,
        "policy_received_reference_certificate": False,
        "policy_received_phase_or_reset_indicator": False,
        "change_time_candidates": change_times,
        "truth_change_time_support": ["never", *change_times],
        "changepoint_semantics": "tau_is_completed_old_world_experiment_count",
        "uses_actual_available_pre_change_history": True,
        "uses_actual_action_measurement_and_budget_contract": True,
        "agent_weight_updates_performed": False,
        "policy": {
            "type": ("bayesian_latent_change_point_decoder_with_information_ranked_fixed_schedule"),
            "change_hazard_by_allowed_change_point": hazards,
            "terminal_no_change_prior_mass": prior_mass[-1],
            "phase_invariant_action_rule": True,
            "changed_mechanism_states_are_absorbing": True,
            "allowed_change_points_only": True,
            "reference_age_aware_likelihoods": True,
            "cross_action_relational_likelihoods": relational_evidence_enabled,
            "one_likelihood_channel_per_experiment": True,
            "information_ranking_batch_size": ranking_batch_size,
            "online_policy_action_count": policy_action_count,
            "reference_sufficiency_basis": (
                "universal_structural_relation_coverage_and_held_out_predictive_adequacy"
            ),
            "post_change_experiment_budget_checkpoints": post_budgets,
            "online_policy_gate_budget": online_gate_budget,
            "public_feature_encoding": feature_encoding,
            "public_feature_block_size": feature_block_size,
            "online_discriminative_block_count": discriminative_block_count,
            "online_likelihood_scale": online_likelihood_scale,
            "online_batch_likelihood_scale": online_batch_likelihood_scale,
            "likelihood_effective_dimension_count": (
                likelihood_effective_dimension_count
            ),
            "selected_feature_dimension_per_evidence_channel": (
                selected_feature_dimension
            ),
            "online_batch_information_minimum_spread_nats": (
                online_batch_information_minimum_spread
            ),
            "action_selection": requirement["action_selection"],
            "reference_policy": requirement["reference_policy"],
            "evidence_policy": requirement["evidence_policy"],
        },
        "reference_acquisition_certificate": reference_acquisition_certificate,
        "keyed_noise_pairing_audit": keyed_noise_pairing_audit,
        "trial_manifests": online_trial_manifests,
        "trial_manifests_complete": trial_manifests_complete,
        "online_capability_chain_certificate": capability_certificate,
        "online_capability_chain_by_post_change_budget": {
            str(budget): value
            for budget, value in capability_chain_by_budget.items()
        },
        "online_capability_chain_by_task_and_post_change_budget": {
            str(budget): value
            for budget, value in task_capability_chain_by_budget.items()
        },
        "online_capability_chain_by_family_and_post_change_budget": {
            str(budget): value
            for budget, value in family_capability_chain_by_budget.items()
        },
        "primary_task_intersection_pass": task_intersection_pass,
        "primary_family_intersection_pass": family_intersection_pass,
        "family_macro_average": family_macro_average,
        "pooled_micro_average": {
            "controls_gate": False,
            "summary": capability_certificate,
        },
        "aggregation_rule": (
            "gate_requires_overall_macro_all_tasks_and_all_declared_families; "
            "pooled_micro_results_are_supplemental"
        ),
        "identifiability_certificate": certificate,
        "identifiability_by_post_change_budget": {
            str(budget): value for budget, value in certificates_by_budget.items()
        },
        "unstratified_identifiability_by_post_change_budget": {
            str(budget): value
            for budget, value in unstratified_certificates_by_budget.items()
        },
        "task_reports": task_reports,
        "interpretation": (
            "Gate A3 certifies benchmark attainability by the frozen phase-invariant "
            "reference diagnostic policy. It first verifies that the policy acquired a "
            "sufficient old-world reference before scoring calibrated change "
            "attribution. Trials without that prerequisite are reported as reference "
            "acquisition failures, never as participant-Agent evidence. An evaluator-side "
            "Bayesian latent change-point decoder received no phase-change signal and "
            "selected exactly one absolute, "
            "same-action temporal"
            + (
                ", or fit-ranked cross-action relational"
                if relational_evidence_enabled
                else ""
            )
            + " likelihood channel per experiment. Transition-versus-stable "
            "semantics were selected "
            "from each reference's establishment time under the frozen post-change "
            "experiment budget."
        ),
        "publication_ready": False,
    }


def _emit_gate_a_progress(
    callback: Callable[[Mapping[str, Any]], None] | None,
    *,
    event: str,
    **details: Any,
) -> None:
    if callback is not None:
        callback({"stage": "gate-a", "event": event, **details})


def _validate_paired_public_contrast_encoding(
    phase_plan: Mapping[str, Any],
) -> None:
    """Accept the frozen same-recipe encoding and its relational extension."""

    supported = {
        "post_minus_pre_same_recipe",
        "post_minus_pre_same_recipe_or_declared_same_background_relation",
    }
    if phase_plan.get("public_contrast_encoding") not in supported:
        raise ValueError("unsupported paired Gate A public contrast encoding")


def _run_paired_gate_a(
    protocol: Mapping[str, Any],
    plan: Mapping[str, Any],
    *,
    online_attainability_certificate: Mapping[str, Any] | None = None,
    design_validity_audit: Mapping[str, Any] | None = None,
    progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
    trial_store_root: Path | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    """Run the pre/post paired controlled certificate and compose full Gate A."""

    errors = validate_mechanism_adaptation_protocol(protocol)
    if errors:
        raise ValueError("invalid mechanism-adaptation protocol: " + "; ".join(errors))
    release_qualification = _validate_release_qualification(
        protocol,
        plan,
        configuration_root().parent,
    )
    if release_qualification is not None:
        _validate_frozen_registration(
            protocol,
            plan,
            configuration_root().parent,
        )
    action_plan = plan["action_library"]
    fit_plan = plan["candidate_predictive_fit"]
    certificate_plan = plan["held_out_certificate"]
    phase_plan = plan["paired_phase_design"]
    gate = protocol["gates"]["gate_a"]
    feature_encoding = str(fit_plan["public_feature_encoding"])
    if feature_encoding != PUBLIC_EXPERIMENT_FEATURE_ENCODING_VERSION:
        raise ValueError("controlled public-feature encoding contract is unsupported")
    feature_block_size = int(fit_plan["summary_block_size"])
    if feature_block_size != PUBLIC_EXPERIMENT_FEATURE_BLOCK_SIZE:
        raise ValueError("controlled public-feature block-size contract is unsupported")
    dimension_normalized_likelihood = _uses_dimension_normalized_likelihood(
        fit_plan
    )
    if dimension_normalized_likelihood:
        controlled_block_count = int(fit_plan["controlled_discriminative_block_count"])
        likelihood_effective_dimension_count = float(
            fit_plan["likelihood_effective_dimension_count"]
        )
        fixed_controlled_likelihood_scale: float | None = None
        controlled_information_minimum_spread = float(
            fit_plan["controlled_primary_information_minimum_spread_nats"]
        )
    else:
        controlled_block_count = int(fit_plan["online_discriminative_block_count"])
        likelihood_effective_dimension_count = float(
            controlled_block_count * feature_block_size
        )
        fixed_controlled_likelihood_scale = float(fit_plan["online_likelihood_scale"])
        controlled_information_minimum_spread = 0.0
    selected_feature_dimension_per_action = controlled_block_count * feature_block_size
    maximum_likelihood_scale = 1.0 / feature_block_size
    if (
        controlled_block_count <= 0
        or (
            fixed_controlled_likelihood_scale is not None
            and (
                not np.isfinite(fixed_controlled_likelihood_scale)
                or fixed_controlled_likelihood_scale <= 0.0
                or fixed_controlled_likelihood_scale > maximum_likelihood_scale
            )
        )
    ):
        raise ValueError(
            "controlled evidence requires positive whole-block selection and a "
            "likelihood scale no greater than one effective contribution per block"
        )
    if (
        not np.isfinite(controlled_information_minimum_spread)
        or controlled_information_minimum_spread < 0.0
        or (
            dimension_normalized_likelihood
            and controlled_information_minimum_spread == 0.0
        )
    ):
        raise ValueError(
            "controlled primary information minimum spread must be nonnegative "
            "and positive for dimension-normalized likelihoods"
        )
    if fixed_controlled_likelihood_scale is None:
        _dimension_normalized_likelihood_scale(
            effective_dimension_count=likelihood_effective_dimension_count,
            selected_feature_dimension=selected_feature_dimension_per_action,
        )
    online_history_aligned_budget = int(phase_plan["pre_change_reference_experiments"])
    if protocol.get("schema_version") == (
        "chemworld-mechanism-adaptation-protocol-0.3.0"
    ):
        protocol_pre_change = int(
            protocol["evaluation_tracks"]["calibrated_online_change"][
                "minimum_stable_prefix_experiments"
            ]
        )
    else:
        protocol_pre_change = int(protocol["design"]["pre_change_experiments"])
    if online_history_aligned_budget != protocol_pre_change:
        raise ValueError("paired Gate A must use the protocol pre-change experiment budget")
    budgets = [int(item) for item in certificate_plan["budgets"]]
    gate_budget_field = (
        "a2_controlled_budget_checkpoints"
        if protocol.get("schema_version")
        == "chemworld-mechanism-adaptation-protocol-0.3.0"
        else "budget_checkpoints"
    )
    if budgets != [int(item) for item in gate[gate_budget_field]]:
        raise ValueError("paired Gate A budgets must match the frozen protocol checkpoints")
    primary_budget = int(certificate_plan["primary_gate_budget"])
    if primary_budget not in budgets:
        raise ValueError("paired Gate A primary budget must be a frozen checkpoint")
    if (
        protocol.get("schema_version")
        != "chemworld-mechanism-adaptation-protocol-0.3.0"
        and online_history_aligned_budget not in budgets
    ):
        raise ValueError("paired Gate A must report the online-history-aligned checkpoint")
    if int(action_plan["action_count_per_task"]) < max(budgets):
        raise ValueError("paired Gate A action library cannot underfill its largest decoder budget")
    if (
        len(budgets) > 1
        and phase_plan.get("diagnostic_curve_matches_reference_count_to_budget") is not True
    ):
        raise ValueError(
            "multi-budget paired Gate A must declare matched reference counts for its curve"
        )
    if phase_plan.get("same_hidden_world_seed_across_phases") is not True:
        raise ValueError("paired Gate A requires the same hidden-world seed across phases")
    if phase_plan.get("independent_observation_seed_across_phases") is not True:
        raise ValueError("paired Gate A requires independent phase observation seeds")
    _validate_paired_public_contrast_encoding(phase_plan)
    raw_controlled_relational_coverage = phase_plan.get(
        "controlled_primary_declared_relational_action_coverage",
        False,
    )
    if not isinstance(raw_controlled_relational_coverage, bool):
        raise ValueError(
            "controlled primary relational coverage flag must be boolean"
        )
    controlled_relational_coverage_required = (
        raw_controlled_relational_coverage
    )

    action_libraries = {
        str(task_id): build_action_library(
            str(task_id),
            action_count=int(action_plan["action_count_per_task"]),
            seed=int(action_plan["design_seed"]),
            design_id=str(
                action_plan.get("design", DIAGNOSTIC_RECIPE_DESIGN_V1)
            ),
        )
        for task_id in protocol["design"]["tasks"]
    }
    if controlled_relational_coverage_required:
        for task_id, action_library in action_libraries.items():
            declarations = _declared_relational_action_groups(
                task_id=task_id,
                contract=protocol["task_mechanism_contracts"][task_id],
                action_library=action_library,
            )
            coverage = relational_coverage_witness(
                declaration_groups=declarations,
                action_ids=list(action_library),
                budget=primary_budget,
            )
            if coverage["feasible"] is not True:
                raise ValueError(
                    f"{task_id} controlled primary budget {primary_budget} "
                    "cannot cover every declared relational intervention; "
                    f"minimum distinct actions={coverage['minimum_distinct_actions']}"
                )
    if design_validity_audit is None:
        _emit_gate_a_progress(progress_callback, event="design_audit_started")
        design_audit = audit_mechanism_design(
            protocol,
            plan,
            action_libraries=action_libraries,
        )
        _emit_gate_a_progress(progress_callback, event="design_audit_completed")
    else:
        design_audit = validate_precomputed_design_audit(
            protocol,
            plan,
            design_validity_audit,
        )
        _emit_gate_a_progress(progress_callback, event="design_audit_reused")
    if not design_audit["pass"]:
        failed = "; ".join(
            f"{item.get('task_id')}:{item['check']}" for item in design_audit["failures"]
        )
        raise ValueError(f"mechanism design audit failed: {failed}")

    task_reports: dict[str, Any] = {}
    active_truths: dict[int, list[str]] = {budget: [] for budget in budgets}
    active_predictions: dict[int, list[str]] = {budget: [] for budget in budgets}
    decoder_truths: dict[int, list[str]] = {budget: [] for budget in budgets}
    decoder_predictions: dict[int, list[str]] = {budget: [] for budget in budgets}
    candidate_union: list[str] = []
    sample_count = int(fit_plan["samples_per_candidate_action"])
    controlled_trial_manifests: dict[str, dict[str, Any]] = {}

    for task_index, task_id in enumerate(protocol["design"]["tasks"]):
        task_id = str(task_id)
        contract = protocol["task_mechanism_contracts"][task_id]
        candidate_ids = [str(item) for item in contract["candidate_ids"]]
        for candidate_id in candidate_ids:
            if candidate_id not in candidate_union:
                candidate_union.append(candidate_id)
        action_library = action_libraries[task_id]
        action_ids = list(action_library)
        controlled_relational_declarations = (
            _declared_relational_action_groups(
                task_id=task_id,
                contract=contract,
                action_library=action_library,
            )
            if controlled_relational_coverage_required
            else {}
        )
        fit_world_seeds = [
            int(fit_plan["nuisance_seed_namespace_start"]) + task_index * 1_000_000 + index
            for index in range(sample_count)
        ]
        fit_keys: list[tuple[str, str]] = []
        fit_jobs: list[dict[str, Any]] = []
        for candidate_id in candidate_ids:
            for action_id in action_ids:
                for world_seed in fit_world_seeds:
                    fit_keys.append((candidate_id, action_id))
                    fit_jobs.append(
                        {
                            "task_id": task_id,
                            "world_seed": world_seed,
                            "interventions": contract["interventions"][candidate_id],
                            "action_ids": [action_id],
                            "action_library": {action_id: action_library[action_id].tolist()},
                            "pre_observation_seed": _stable_seed(
                                world_seed, f"gate-a-v0.2.2:{action_id}:fit-pre"
                            ),
                            "post_observation_seed": _stable_seed(
                                world_seed, f"gate-a-v0.2.2:{action_id}:fit-post"
                            ),
                        }
                    )
        _emit_gate_a_progress(
            progress_callback,
            event="predictive_fit_started",
            task_id=task_id,
            job_count=len(fit_jobs),
        )
        if trial_store_root is None:
            fit_results = _execute_jobs(
                _sample_paired_contrast_job,
                fit_jobs,
                workers=int(fit_plan.get("execution_workers", 1)),
            )
            fit_manifest = None
        else:
            fit_trial_keys = [
                ConfirmatoryTrialKey(
                    task_id=task_id,
                    truth_family=candidate_id,
                    world_cluster=str(job["world_seed"]),
                    changepoint="predictive_fit",
                    arm=f"action:{job['action_ids'][0]}",
                )
                for (candidate_id, _action_id), job in zip(
                    fit_keys,
                    fit_jobs,
                    strict=True,
                )
            ]
            fit_results, fit_manifest = execute_jobs_resumable(
                _sample_paired_contrast_job,
                fit_jobs,
                fit_trial_keys,
                workers=int(fit_plan.get("execution_workers", 1)),
                store=ConfirmatoryTrialStore(
                    Path(trial_store_root) / task_id / "predictive-fit"
                ),
                resume=resume,
            )
            controlled_trial_manifests.setdefault(task_id, {})[
                "predictive_fit"
            ] = fit_manifest
        _emit_gate_a_progress(
            progress_callback,
            event="predictive_fit_completed",
            task_id=task_id,
            job_count=len(fit_jobs),
        )
        per_action_samples: dict[tuple[str, str], list[Sequence[float]]] = {
            (candidate_id, action_id): []
            for candidate_id in candidate_ids
            for action_id in action_ids
        }
        for key, values in zip(fit_keys, fit_results, strict=True):
            per_action_samples[key].append(values)
        controlled_feature_indices, controlled_feature_selection = (
            _select_discriminative_feature_blocks(
                per_action_samples,
                candidate_ids=candidate_ids,
                evidence_action_ids=action_ids,
                block_size=feature_block_size,
                block_count=controlled_block_count,
                variance_floor=float(fit_plan["variance_floor"]),
            )
        )
        reduced_per_action_samples = {
            key: [
                np.asarray(sample, dtype=float)[
                    np.asarray(controlled_feature_indices[key[1]], dtype=int)
                ].tolist()
                for sample in samples
            ]
            for key, samples in per_action_samples.items()
        }

        serialized_actions = {key: vector.tolist() for key, vector in action_library.items()}
        task_trials: dict[str, list[dict[str, Any]]] = {
            f"{role}_budget_{budget}": [] for budget in budgets for role in ("active", "decoder")
        }
        budget_designs: dict[str, Any] = {}
        for budget in budgets:
            oracle_seed = _stable_seed(int(fit_plan["oracle_seed"]), f"{task_id}:budget-{budget}")
            controlled_likelihood_scale = (
                fixed_controlled_likelihood_scale
                if fixed_controlled_likelihood_scale is not None
                else _dimension_normalized_likelihood_scale(
                    effective_dimension_count=likelihood_effective_dimension_count,
                    selected_feature_dimension=(
                        selected_feature_dimension_per_action * budget
                    ),
                )
            )
            oracle, batches = _fit_paired_batch_oracle(
                candidate_ids=candidate_ids,
                action_ids=action_ids,
                per_action_samples=reduced_per_action_samples,
                batch_size=budget,
                variance_floor=float(fit_plan["variance_floor"]),
                likelihood_scale=controlled_likelihood_scale,
                seed=oracle_seed,
            )
            information = oracle.expected_information_by_action(
                draws=int(certificate_plan["information_draws_per_batch"])
            )
            relationally_covered_batch_ids = (
                _relationally_covered_batch_ids(
                    batches=batches,
                    declaration_groups=controlled_relational_declarations,
                )
                if controlled_relational_coverage_required
                else list(batches)
            )
            eligible_relational_batch_ids = (
                relationally_covered_batch_ids
                if controlled_relational_coverage_required
                and budget == primary_budget
                else list(batches)
            )
            if not eligible_relational_batch_ids:
                raise ValueError(
                    f"{task_id} controlled primary budget cannot cover every "
                    "declared relational intervention"
                )
            selection_information = {
                batch_id: information[batch_id]
                for batch_id in eligible_relational_batch_ids
            }
            active_batch_id, information_diagnostics = _select_information_maximum(
                selection_information,
                minimum_spread_nats=(
                    controlled_information_minimum_spread
                    if budget == primary_budget
                    else 0.0
                ),
                context=f"{task_id} controlled budget {budget}",
            )
            decoder_batch_id = _paired_action_id(action_ids[:budget])
            trial_jobs = []
            trial_keys: list[tuple[str, int]] = []
            for candidate_id in candidate_ids:
                for repeat_index in range(int(certificate_plan["world_seeds_per_family"])):
                    world_seed = (
                        int(certificate_plan["seed_namespace_start"])
                        + task_index * 1_000_000
                        + repeat_index
                    )
                    trial_keys.append((candidate_id, repeat_index))
                    trial_jobs.append(
                        {
                            "task_id": task_id,
                            "truth_id": candidate_id,
                            "world_seed": world_seed,
                            "interventions": contract["interventions"][candidate_id],
                            "candidate_ids": candidate_ids,
                            "batch_ids": list(batches),
                            "batches": {key: list(value) for key, value in batches.items()},
                            "active_batch_id": active_batch_id,
                            "decoder_batch_id": decoder_batch_id,
                            "action_library": serialized_actions,
                            "predictives": oracle.export_predictives(),
                            "feature_indices": {
                                key: list(value)
                                for key, value in controlled_feature_indices.items()
                            },
                            "variance_floor": float(fit_plan["variance_floor"]),
                            "oracle_seed": oracle_seed,
                        }
                    )
            _emit_gate_a_progress(
                progress_callback,
                event="certificate_trials_started",
                task_id=task_id,
                budget=budget,
                job_count=len(trial_jobs),
            )
            confirmatory_keys = [
                ConfirmatoryTrialKey(
                    task_id=task_id,
                    truth_family=str(job["truth_id"]),
                    world_cluster=str(job["world_seed"]),
                    changepoint="not_applicable",
                    arm=f"controlled_budget_{budget}_active_and_decoder",
                )
                for job in trial_jobs
            ]
            if trial_store_root is None:
                completed_trials = _execute_jobs(
                    _execute_paired_gate_a_trial_job,
                    trial_jobs,
                    workers=int(certificate_plan.get("execution_workers", 1)),
                )
            else:
                completed_trials, certificate_manifest = execute_jobs_resumable(
                    _execute_paired_gate_a_trial_job,
                    trial_jobs,
                    confirmatory_keys,
                    workers=int(certificate_plan.get("execution_workers", 1)),
                    store=ConfirmatoryTrialStore(
                        Path(trial_store_root)
                        / task_id
                        / f"certificate-budget-{budget}"
                    ),
                    resume=resume,
                )
                controlled_trial_manifests.setdefault(task_id, {})[
                    f"certificate_budget_{budget}"
                ] = certificate_manifest
            _emit_gate_a_progress(
                progress_callback,
                event="certificate_trials_completed",
                task_id=task_id,
                budget=budget,
                job_count=len(trial_jobs),
            )
            for (truth_id, _repeat_index), completed in zip(
                trial_keys, completed_trials, strict=True
            ):
                qualified_truth = _qualified_candidate(task_id, truth_id)
                active = completed["active"]
                decoder = completed["decoder"]
                active_truths[budget].append(qualified_truth)
                active_predictions[budget].append(
                    _qualified_candidate(task_id, str(active["prediction"]))
                )
                decoder_truths[budget].append(qualified_truth)
                decoder_predictions[budget].append(
                    _qualified_candidate(task_id, str(decoder["prediction"]))
                )
                task_trials[f"active_budget_{budget}"].append(active)
                task_trials[f"decoder_budget_{budget}"].append(decoder)
            budget_designs[str(budget)] = {
                "matched_pre_change_reference_experiments": budget,
                "post_change_diagnostic_experiments": budget,
                "controlled_matched_pre_post_budget": True,
                "matches_a3_minimum_reference_prefix": (
                    budget == online_history_aligned_budget
                ),
                "likelihood_scale": controlled_likelihood_scale,
                "selected_feature_dimension": (
                    selected_feature_dimension_per_action * budget
                ),
                "active_batch_id": active_batch_id,
                "active_batch_information_nats": float(information[active_batch_id]),
                "batch_information_diagnostics": information_diagnostics,
                "declared_relational_coverage": {
                    "required": (
                        controlled_relational_coverage_required
                        and budget == primary_budget
                    ),
                    "declarations": {
                        key: [list(group) for group in groups]
                        for key, groups in controlled_relational_declarations.items()
                    },
                    "relationally_covered_batch_ids": (
                        relationally_covered_batch_ids
                    ),
                    "selection_eligible_batch_ids": (
                        eligible_relational_batch_ids
                    ),
                    "selected_batch_covers_all_declarations": (
                        active_batch_id in relationally_covered_batch_ids
                    ),
                },
                "decoder_batch_id": decoder_batch_id,
                "batch_information_nats": {key: float(value) for key, value in information.items()},
            }
        task_reports[task_id] = {
            "candidate_ids": candidate_ids,
            "action_library": serialized_actions,
            "controlled_public_feature_encoding": feature_encoding,
            "controlled_public_feature_block_size": feature_block_size,
            "controlled_discriminative_block_count": controlled_block_count,
            "controlled_likelihood_normalization": (
                "inverse_selected_feature_dimension"
                if fixed_controlled_likelihood_scale is None
                else "legacy_fixed_scale"
            ),
            "likelihood_effective_dimension_count": (
                likelihood_effective_dimension_count
            ),
            "selected_feature_dimension_per_action": (
                selected_feature_dimension_per_action
            ),
            "trial_manifests": controlled_trial_manifests.get(task_id, {}),
            "controlled_primary_information_minimum_spread_nats": (
                controlled_information_minimum_spread
            ),
            "controlled_primary_declared_relational_action_coverage": (
                controlled_relational_coverage_required
            ),
            "controlled_feature_selection": controlled_feature_selection,
            "budget_designs": budget_designs,
            "trials": task_trials,
        }

    qualified_candidates = [
        _qualified_candidate(str(task_id), str(candidate_id))
        for task_id in protocol["design"]["tasks"]
        for candidate_id in protocol["task_mechanism_contracts"][task_id]["candidate_ids"]
    ]
    active_certificates = {
        budget: identifiability_certificate(
            truths=active_truths[budget],
            predictions=active_predictions[budget],
            candidate_ids=qualified_candidates,
            overall_lower_bound_threshold=float(gate["overall_top1_wilson_lower_bound"]),
            family_recall_lower_bound_threshold=float(gate["per_family_recall_wilson_lower_bound"]),
        )
        for budget in budgets
    }
    decoder_certificates = {
        budget: identifiability_certificate(
            truths=decoder_truths[budget],
            predictions=decoder_predictions[budget],
            candidate_ids=qualified_candidates,
            overall_lower_bound_threshold=float(gate["overall_top1_wilson_lower_bound"]),
            family_recall_lower_bound_threshold=float(gate["per_family_recall_wilson_lower_bound"]),
        )
        for budget in budgets
    }
    controlled_manifests_complete = (
        None
        if trial_store_root is None
        else (
            len(controlled_trial_manifests) == len(task_reports)
            and all(
                len(task_manifests) == len(budgets) + 1
                and all(
                    manifest["complete"] is True
                    for manifest in task_manifests.values()
                )
                for task_manifests in controlled_trial_manifests.values()
            )
        )
    )
    controlled_gate_pass = bool(
        active_certificates[primary_budget]["gate_pass"]
        and controlled_manifests_complete is not False
    )
    decision = gate_a_certificate_decision(
        protocol,
        plan,
        physical_intervention_validity_pass=design_audit["pass"] is True,
        controlled_gate_pass=controlled_gate_pass,
        online_attainability_certificate=online_attainability_certificate,
    )
    return {
        "schema_version": GATE_A_REPORT_VERSION,
        "status": decision["status"],
        "formal_benchmark_result": False,
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": canonical_sha256(protocol),
        "gate_a_plan_id": plan["plan_id"],
        "gate_a_plan_sha256": canonical_sha256(plan),
        "execution_contract_binding": gate_a_execution_contract_binding(protocol, plan),
        "environment_role": protocol["environment_role"],
        "agent_weight_updates_performed": False,
        "design_validity_audit": design_audit,
        "public_feature_contract": dict(plan["public_feature_contract"]),
        "paired_phase_design": dict(phase_plan),
        "nuisance_integration": {
            "performed": True,
            "mode": "paired_common_hidden_world_with_independent_phase_observation_noise",
            "predictive_samples_per_candidate_action": sample_count,
            "held_out_world_seeds_per_family": int(certificate_plan["world_seeds_per_family"]),
            "same_world_seeds_reused_across_candidate_twins": True,
            "fit_and_certificate_seed_namespaces_disjoint": True,
        },
        "primary_gate_budget": primary_budget,
        "trial_manifests": controlled_trial_manifests,
        "trial_manifests_complete": controlled_manifests_complete,
        "active_oracle": {
            "type": "batch_information_gain_over_matched_pre_post_recipe_contrasts",
            "gate_pass": controlled_gate_pass,
            "primary_budget": primary_budget,
            "by_budget": {str(budget): active_certificates[budget] for budget in budgets},
        },
        "fixed_trajectory_decoder": {
            "controls_gate": False,
            "by_budget": {str(budget): decoder_certificates[budget] for budget in budgets},
        },
        "certificate_decision": decision,
        "online_attainability_certificate": decision["online_attainability_certificate"],
        "gate_a_pass": decision["gate_a_pass"],
        "task_reports": task_reports,
        "interpretation": (
            "A2 establishes controlled identifiability from equally sized matched "
            "pre/post public recipe budgets. It does not establish online change "
            "discovery. Full Gate A separately requires A3 online attainability by "
            "the frozen reference diagnostic policy after a sufficient old-world prefix."
        ),
        "candidate_family_names": candidate_union,
        "publication_ready": False,
    }


def run_gate_a(
    protocol: Mapping[str, Any],
    plan: Mapping[str, Any],
    *,
    online_attainability_certificate: Mapping[str, Any] | None = None,
    design_validity_audit: Mapping[str, Any] | None = None,
    progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
    trial_store_root: Path | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    """Run the frozen active-oracle and fixed-decoder identifiability checks."""

    if plan.get("schema_version") in {
        "chemworld-mechanism-adaptation-gate-a-plan-0.2.2",
        "chemworld-mechanism-adaptation-gate-a-plan-0.2.3",
        "chemworld-mechanism-adaptation-gate-a-plan-0.2.4",
        "chemworld-mechanism-adaptation-gate-a-plan-0.2.5",
        "chemworld-mechanism-adaptation-gate-a-plan-0.2.6",
        "chemworld-mechanism-adaptation-gate-a-plan-0.2.7",
        "chemworld-mechanism-adaptation-gate-a-plan-0.3.0",
    }:
        return _run_paired_gate_a(
            protocol,
            plan,
            online_attainability_certificate=online_attainability_certificate,
            design_validity_audit=design_validity_audit,
            progress_callback=progress_callback,
            trial_store_root=trial_store_root,
            resume=resume,
        )
    errors = validate_mechanism_adaptation_protocol(protocol)
    if errors:
        raise ValueError("invalid mechanism-adaptation protocol: " + "; ".join(errors))
    if plan.get("schema_version") != "chemworld-mechanism-adaptation-gate-a-plan-0.2.1":
        raise ValueError("unsupported Gate A plan schema")

    action_plan = plan["action_library"]
    fit_plan = plan["candidate_predictive_fit"]
    certificate_plan = plan["held_out_certificate"]
    gate = protocol["gates"]["gate_a"]
    budgets = [int(item) for item in certificate_plan["budgets"]]
    primary_budget = int(certificate_plan["primary_gate_budget"])
    gate_budget_field = (
        "a2_controlled_budget_checkpoints"
        if protocol.get("schema_version")
        == "chemworld-mechanism-adaptation-protocol-0.3.0"
        else "budget_checkpoints"
    )
    if budgets != [int(item) for item in gate[gate_budget_field]]:
        raise ValueError("Gate A plan budgets do not match the frozen protocol")
    if primary_budget not in budgets:
        raise ValueError("primary Gate A budget must be one of the frozen budgets")
    if int(action_plan["action_count_per_task"]) < max(budgets):
        raise ValueError("Gate A action library must cover the largest decoder budget")

    action_libraries = {
        str(task_id): build_action_library(
            str(task_id),
            action_count=int(action_plan["action_count_per_task"]),
            seed=int(action_plan["design_seed"]),
            design_id=str(
                action_plan.get("design", DIAGNOSTIC_RECIPE_DESIGN_V1)
            ),
        )
        for task_id in protocol["design"]["tasks"]
    }
    design_audit = audit_mechanism_design(
        protocol,
        plan,
        action_libraries=action_libraries,
    )
    if not design_audit["pass"]:
        failed = "; ".join(
            f"{item.get('task_id')}:{item['check']}" for item in design_audit["failures"]
        )
        raise ValueError(f"mechanism design audit failed: {failed}")

    active_by_budget: dict[int, dict[str, Any]] = {}
    decoder_by_budget: dict[int, dict[str, Any]] = {}
    task_reports: dict[str, Any] = {}
    active_truths: dict[int, list[str]] = {budget: [] for budget in budgets}
    active_predictions: dict[int, list[str]] = {budget: [] for budget in budgets}
    decoder_truths: dict[int, list[str]] = {budget: [] for budget in budgets}
    decoder_predictions: dict[int, list[str]] = {budget: [] for budget in budgets}
    candidate_union: list[str] = []

    for task_index, task_id in enumerate(protocol["design"]["tasks"]):
        contract = protocol["task_mechanism_contracts"][task_id]
        candidate_ids = [str(item) for item in contract["candidate_ids"]]
        for candidate_id in candidate_ids:
            if candidate_id not in candidate_union:
                candidate_union.append(candidate_id)
        action_library = action_libraries[task_id]
        action_ids = list(action_library)

        oracle = GaussianMechanismOracle(
            candidate_ids=candidate_ids,
            action_ids=action_ids,
            sample_public_observation=_unreachable_sample,
            samples_per_candidate=int(fit_plan["samples_per_candidate_action"]),
            variance_floor=float(fit_plan["variance_floor"]),
            seed=_stable_seed(int(fit_plan["oracle_seed"]), task_id),
        )
        fit_rng = np.random.default_rng(oracle.seed)
        fit_jobs: list[dict[str, Any]] = []
        fit_keys: list[tuple[str, str]] = []
        for action_id in action_ids:
            for candidate_id in candidate_ids:
                seeds = fit_rng.integers(
                    0,
                    np.iinfo(np.int32).max,
                    size=int(fit_plan["samples_per_candidate_action"]),
                )
                for nuisance_seed in seeds:
                    fit_keys.append((candidate_id, action_id))
                    fit_jobs.append(
                        {
                            "task_id": task_id,
                            "seed": int(fit_plan["nuisance_seed_namespace_start"])
                            + int(nuisance_seed) % 500_000_000,
                            "interventions": contract["interventions"][candidate_id],
                            "action_id": action_id,
                            "action_vector": action_library[action_id].tolist(),
                        }
                    )
        fit_results = _execute_jobs(
            _sample_experiment_job,
            fit_jobs,
            workers=int(fit_plan.get("execution_workers", 1)),
        )
        sample_groups: dict[tuple[str, str], list[Sequence[float]]] = {
            (candidate_id, action_id): []
            for action_id in action_ids
            for candidate_id in candidate_ids
        }
        for key, values in zip(fit_keys, fit_results, strict=True):
            sample_groups[key].append(values)
        oracle.fit_predictives_from_samples(sample_groups)
        task_trials: dict[str, list[dict[str, Any]]] = {
            f"active_budget_{budget}": [] for budget in budgets
        }
        task_trials.update({f"decoder_budget_{budget}": [] for budget in budgets})

        trial_keys = [
            (candidate_index, truth_id, repeat_index)
            for candidate_index, truth_id in enumerate(candidate_ids)
            for repeat_index in range(int(certificate_plan["world_seeds_per_family"]))
        ]
        exported_predictives = oracle.export_predictives()
        serialized_actions = {key: vector.tolist() for key, vector in action_library.items()}
        trial_jobs = [
            {
                "task_id": task_id,
                "truth_id": truth_id,
                "trial_seed": int(certificate_plan["seed_namespace_start"])
                + task_index * 1_000_000
                + candidate_index * 10_000
                + repeat_index,
                "interventions": contract["interventions"][truth_id],
                "budgets": budgets,
                "action_ids": action_ids,
                "action_library": serialized_actions,
                "candidate_ids": candidate_ids,
                "predictives": exported_predictives,
                "variance_floor": float(fit_plan["variance_floor"]),
                "oracle_seed": oracle.seed,
                "information_draws": int(certificate_plan["information_draws_per_action"]),
                "repeat_policy": str(action_plan["repeat_policy"]),
            }
            for candidate_index, truth_id, repeat_index in trial_keys
        ]
        completed_trials = _execute_jobs(
            _execute_gate_a_trial_job,
            trial_jobs,
            workers=int(certificate_plan.get("execution_workers", 1)),
        )

        for job, completed in zip(trial_keys, completed_trials, strict=True):
            _candidate_index, truth_id, _repeat_index = job
            qualified_truth = _qualified_candidate(task_id, truth_id)
            for budget in budgets:
                active = completed["active"][budget]
                decoder = completed["decoder"][budget]
                active_truths[budget].append(qualified_truth)
                active_predictions[budget].append(
                    _qualified_candidate(task_id, active["prediction"])
                )
                decoder_truths[budget].append(qualified_truth)
                decoder_predictions[budget].append(
                    _qualified_candidate(task_id, decoder["prediction"])
                )
                task_trials[f"active_budget_{budget}"].append(active)
                task_trials[f"decoder_budget_{budget}"].append(decoder)
        task_reports[task_id] = {
            "candidate_ids": candidate_ids,
            "action_library": {
                key: [float(item) for item in vector] for key, vector in action_library.items()
            },
            "trials": task_trials,
        }

    qualified_candidates = [
        _qualified_candidate(task_id, candidate_id)
        for task_id in protocol["design"]["tasks"]
        for candidate_id in protocol["task_mechanism_contracts"][task_id]["candidate_ids"]
    ]
    for budget in budgets:
        active_by_budget[budget] = identifiability_certificate(
            truths=active_truths[budget],
            predictions=active_predictions[budget],
            candidate_ids=qualified_candidates,
            overall_lower_bound_threshold=float(gate["overall_top1_wilson_lower_bound"]),
            family_recall_lower_bound_threshold=float(gate["per_family_recall_wilson_lower_bound"]),
        )
        decoder_by_budget[budget] = identifiability_certificate(
            truths=decoder_truths[budget],
            predictions=decoder_predictions[budget],
            candidate_ids=qualified_candidates,
            overall_lower_bound_threshold=float(gate["overall_top1_wilson_lower_bound"]),
            family_recall_lower_bound_threshold=float(gate["per_family_recall_wilson_lower_bound"]),
        )

    primary_active = active_by_budget[primary_budget]
    decision = gate_a_certificate_decision(
        protocol,
        plan,
        physical_intervention_validity_pass=design_audit["pass"] is True,
        controlled_gate_pass=bool(primary_active["gate_pass"]),
        online_attainability_certificate=online_attainability_certificate,
    )
    return {
        "schema_version": GATE_A_REPORT_VERSION,
        "status": decision["status"],
        "formal_benchmark_result": False,
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": canonical_sha256(protocol),
        "gate_a_plan_id": plan["plan_id"],
        "gate_a_plan_sha256": canonical_sha256(plan),
        "execution_contract_binding": gate_a_execution_contract_binding(protocol, plan),
        "environment_role": protocol["environment_role"],
        "agent_weight_updates_performed": False,
        "design_validity_audit": design_audit,
        "public_feature_contract": dict(plan["public_feature_contract"]),
        "nuisance_integration": {
            "performed": True,
            "predictive_samples_per_candidate_action": int(
                fit_plan["samples_per_candidate_action"]
            ),
            "held_out_world_seeds_per_family": int(certificate_plan["world_seeds_per_family"]),
            "fit_and_certificate_seed_namespaces_disjoint": True,
        },
        "primary_gate_budget": primary_budget,
        "active_oracle": {
            "gate_pass": bool(primary_active["gate_pass"]),
            "primary_budget": primary_budget,
            "by_budget": {str(key): value for key, value in active_by_budget.items()},
        },
        "fixed_trajectory_decoder": {
            "controls_gate": False,
            "by_budget": {str(key): value for key, value in decoder_by_budget.items()},
        },
        "certificate_decision": decision,
        "online_attainability_certificate": decision["online_attainability_certificate"],
        "gate_a_pass": decision["gate_a_pass"],
        "task_reports": task_reports,
        "interpretation": (
            "The controlled oracle establishes budgeted identifiability only. Full Gate A "
            "also requires a separately bound online-attainability certificate; neither "
            "certificate alone establishes evaluated-Agent mechanism discovery."
        ),
        "candidate_family_names": candidate_union,
        "publication_ready": False,
    }


def run_campaign_row(
    protocol: Mapping[str, Any],
    row: Mapping[str, Any],
    *,
    output_root: str | Path,
    llm_methods: Mapping[str, Any],
    method_id: str = "live_llm_b",
    spectrum_disclosure: str = "assigned",
    feedback_condition: FeedbackCondition = "true_feedback",
    progress_callback: (Callable[[str, Any, list[dict[str, Any]]], None] | None) = None,
) -> dict[str, Any]:
    """Execute one frozen changed/no-change row with a real DeepSeek-backed Agent."""

    agent = build_mechanism_agent(
        protocol,
        row,
        llm_methods=llm_methods,
        method_id=method_id,
        spectrum_disclosure=spectrum_disclosure,
    )
    task_id = str(row["task_id"])
    adapter = ContinuingPublicViewAgent(
        agent,
        method_id=method_id,
        feedback_condition=feedback_condition,
        critical_instrument=_CRITICAL_INSTRUMENTS[task_id],
    )
    change_time = int(row["phase_reset_after_experiment"])
    horizon = int(row["total_experiment_horizon"])
    development_override = row.get("development_horizon_override")
    minimum_reference_experiments = (
        int(development_override["frozen_pre_change_experiments"])
        if isinstance(development_override, Mapping)
        else change_time
    )
    condition_suffix = "" if feedback_condition == "true_feedback" else f"--{feedback_condition}"
    campaign_id = f"{row['pair_id']}--{row['arm']}{condition_suffix}"
    result = run_two_phase_campaign(
        task_id=task_id,
        adapter=adapter,
        seed=int(row["world_seed"]),
        pre_change_experiments=change_time,
        post_change_experiments=horizon - change_time,
        shifted_interventions=row["world_interventions"],
        output_root=Path(output_root) / "trajectories",
        campaign_id=campaign_id,
        observation_pair_id=str(row["pair_id"]),
        minimum_reference_experiments=minimum_reference_experiments,
        progress_callback=progress_callback,
    )
    result.update(
        {
            "schema_version": EXECUTION_SCHEMA_VERSION,
            "protocol_id": protocol["protocol_id"],
            "protocol_sha256": canonical_sha256(protocol),
            "matrix_row": dict(row),
            "truth_id": row["truth_id"],
            "hidden_law_changes": bool(row["hidden_law_changes"]),
            "feedback_condition": feedback_condition,
            "formal_result": False,
            "agent_weight_updates_performed": False,
        }
    )
    return result


def build_mechanism_agent(
    protocol: Mapping[str, Any],
    row: Mapping[str, Any],
    *,
    llm_methods: Mapping[str, Any],
    method_id: str = "live_llm_b",
    spectrum_disclosure: str = "assigned",
    client: Any | None = None,
) -> MechanismAdaptationLiveLLMAgent:
    """Build one leakage-resistant development participant Agent."""

    method = llm_methods["methods"][method_id]
    request = method["request_configuration"]
    planner = client or DeepSeekClient(
        model=str(method["model_id"]),
        thinking=bool(request["thinking"]),
        reasoning_effort=cast(Any, str(request.get("reasoning_effort") or "max")),
        timeout_s=float(request["timeout_s"]),
        max_attempts=int(request["max_attempts"]),
        retry_backoff_s=float(request["retry_backoff_s"]),
    )
    definitions = protocol["diagnosis_contract"]["candidate_definitions"]
    specs = tuple(
        MechanismCandidateSpec(
            candidate_id=str(candidate_id),
            public_definition=str(definitions[candidate_id]),
        )
        for candidate_id in row["candidate_ids"]
    )
    candidate_label_mode = _parse_candidate_label_mode(row["candidate_label_mode"])
    scaffold_id = str(method.get("scaffold_id") or "direct_reactive")
    if scaffold_id == "direct_reactive":
        agent_type = MechanismAdaptationLiveLLMAgent
    elif scaffold_id == "stateful_scientific":
        agent_type = StatefulScientificMechanismAgent
    else:
        raise ValueError(f"unsupported mechanism participant scaffold: {scaffold_id}")
    prompt_budget = method.get("prompt_budget_contract")
    prompt_budget_kwargs: dict[str, int] = {}
    if isinstance(prompt_budget, Mapping):
        prompt_budget_kwargs = {
            "environment_view_token_estimate_cap": int(
                prompt_budget["environment_view_max_estimated_tokens"]
            ),
            "agent_memory_token_estimate_cap": int(
                prompt_budget["agent_memory_max_estimated_tokens"]
            ),
            "prompt_token_estimate_cap": int(
                prompt_budget["per_decision_max_estimated_tokens"]
            ),
        }
    return agent_type(
        planner,
        role_id=f"mechanism_adaptation_{method_id}",
        spectrum_disclosure=spectrum_disclosure,
        response_max_tokens=int(request["max_tokens"]),
        fail_fast_on_unbillable_provider_failure=True,
        candidate_specs=specs,
        candidate_label_mode=candidate_label_mode,
        candidate_order_seed=int(row["candidate_order_seed"]),
        randomize_candidate_order=True,
        **prompt_budget_kwargs,
    )


def _parse_candidate_label_mode(value: Any) -> CandidateLabelMode:
    mode = str(value)
    if mode not in {"semantic", "anonymous"}:
        raise ValueError("candidate_label_mode must be semantic or anonymous")
    return cast(CandidateLabelMode, mode)


def selected_campaign_rows(
    protocol: Mapping[str, Any],
    *,
    tasks: Sequence[str] | None = None,
    pair_ids: Sequence[str] | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Select complete pairs from the frozen matrix for resumable pilots or runs."""

    rows = build_paired_campaign_matrix(protocol)
    task_filter = set(tasks or ())
    pair_filter = set(pair_ids or ())
    if task_filter:
        rows = [row for row in rows if row["task_id"] in task_filter]
    if pair_filter:
        rows = [row for row in rows if row["pair_id"] in pair_filter]
    if limit is not None:
        if limit <= 0:
            raise ValueError("campaign pair limit must be positive")
        ordered_pairs = list(dict.fromkeys(row["pair_id"] for row in rows))[:limit]
        retained = set(ordered_pairs)
        rows = [row for row in rows if row["pair_id"] in retained]
    return rows


def _qualified_candidate(task_id: str, candidate_id: str) -> str:
    return f"{task_id}::{candidate_id}"


def _stable_seed(seed: int, label: str) -> int:
    digest = hashlib.sha256(f"{seed}|{label}".encode()).digest()
    return int.from_bytes(digest[:4], "big") % np.iinfo(np.int32).max
