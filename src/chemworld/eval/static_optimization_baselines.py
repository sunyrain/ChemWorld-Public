"""Classic complete-experiment optimizers for the S0 static task."""

from __future__ import annotations

import copy
import math
import statistics
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from time import process_time
from typing import Any

import numpy as np
from sklearn.exceptions import ConvergenceWarning

from chemworld.agents._probability import normal_cdf as _normal_cdf
from chemworld.agents.crystallization_single_stage import (
    crystallization_single_stage_parameters_from_unit_vector,
)
from chemworld.agents.electrochemical_single_stage import (
    ELECTROCHEMICAL_SINGLE_STAGE_CATEGORICAL_COORDINATES,
    ELECTROCHEMICAL_SINGLE_STAGE_DIMENSION,
    electrochemical_single_stage_parameters_from_unit_vector,
)
from chemworld.agents.scientific_adaptation import scientific_measurement_slots
from chemworld.agents.static_optimization import (
    StaticOptimizationPlan,
)
from chemworld.agents.task_recipes import (
    electrochemical_recipe_parameters_from_unit_vector,
    task_recipe_categorical_coordinates,
    task_recipe_dimension,
    task_recipe_kind,
)
from chemworld.eval.provenance import canonical_json_sha256 as canonical_sha256
from chemworld.eval.static_optimization_execution import (
    StaticOptimizationExperimentSession,
    static_optimization_workflow_mode,
)
from chemworld.eval.static_optimization_protocol import (
    exploration_experiment_count,
    static_optimization_crystallization_material_family_id,
    static_optimization_material_family_id,
    static_optimization_scoring_contract_id,
    validate_development_seed_policy,
    validate_static_optimization_protocol,
)
from chemworld.eval.static_optimization_seeds import (
    exploration_observation_seed,
    validation_observation_seed,
)
from chemworld.materials import (
    STATIC_MATERIAL_INFORMATION_NOMINAL,
    STATIC_MATERIAL_INFORMATION_SHUFFLED,
    static_material_information_dossier,
)
from chemworld.physchem.electrochemical_task_contract import (
    ELECTROCHEMICAL_WORKFLOW_STATIC_SINGLE_STAGE,
    normalize_electrochemical_workflow_mode,
)
from chemworld.tasks import get_task

STATIC_BASELINE_RESULT_VERSION = "chemworld-static-optimization-baseline-result-0.1-s0-dev"
_NOMINAL_ONE_HOT_ENCODING = "nominal_one_hot"
_NOMINAL_PROPERTY_ENCODING = "nominal_properties"
_SHUFFLED_PROPERTY_ENCODING = "shuffled_properties"
_SURROGATE_ENCODINGS = frozenset(
    {
        _NOMINAL_ONE_HOT_ENCODING,
        _NOMINAL_PROPERTY_ENCODING,
        _SHUFFLED_PROPERTY_ENCODING,
    }
)


def _normal_pdf(z: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * np.square(z)) / math.sqrt(2.0 * math.pi)


def _expected_improvement(
    mu: np.ndarray,
    sigma: np.ndarray,
    *,
    best: float,
    xi: float = 0.01,
) -> np.ndarray:
    sigma = np.maximum(np.asarray(sigma, dtype=float), 1.0e-9)
    improvement = np.asarray(mu, dtype=float) - float(best) - float(xi)
    z = improvement / sigma
    return improvement * _normal_cdf(z) + sigma * _normal_pdf(z)


def _score_summary(values: list[float]) -> dict[str, Any]:
    return {
        "replicate_count": len(values),
        "mean": statistics.fmean(values) if values else None,
        "median": statistics.median(values) if values else None,
        "sample_standard_deviation": (statistics.stdev(values) if len(values) > 1 else None),
        "minimum": min(values) if values else None,
        "maximum": max(values) if values else None,
    }


def _balanced_design(
    *,
    count: int,
    dimension: int,
    categorical: tuple[tuple[int, int], ...],
    rng: np.random.Generator,
) -> np.ndarray:
    design = np.zeros((count, dimension), dtype=float)
    categorical_map = dict(categorical)
    for coordinate in range(dimension):
        if coordinate in categorical_map:
            category_count = categorical_map[coordinate]
            categories = np.resize(np.arange(category_count, dtype=int), count)
            rng.shuffle(categories)
            design[:, coordinate] = (categories + 0.5) / category_count
            continue
        bins = (np.arange(count, dtype=float) + rng.random(count)) / count
        rng.shuffle(bins)
        design[:, coordinate] = bins
    return design


def _nominal_property_encoding_contract(
    task_info: Mapping[str, Any],
    *,
    electrochemical_workflow_mode: str,
    material_information_config: Mapping[str, Any],
    electrochemical_material_family_id: str,
) -> tuple[str, dict[int, tuple[str, ...]], dict[int, np.ndarray]]:
    task_id = str(task_info.get("task_id", ""))
    if (
        task_id != "electrochemical-conversion"
        or electrochemical_workflow_mode != ELECTROCHEMICAL_WORKFLOW_STATIC_SINGLE_STAGE
    ):
        raise ValueError(
            "nominal-property surrogate encoding is audited only for static "
            "single-stage electrochemistry"
        )
    dossier = static_material_information_dossier(
        material_information_config,
        task_id=task_id,
        material_family_id=electrochemical_material_family_id,
    )
    if dossier is None:
        raise RuntimeError("nominal material dossier was not built")
    choices = dossier["choices"]
    fields_by_coordinate: dict[int, tuple[str, ...]] = {}
    matrix_by_coordinate: dict[int, np.ndarray] = {}
    for coordinate, material_field in ((0, "electrolyte_profile"), (1, "solvent")):
        rows = choices[material_field]
        property_fields = tuple(sorted(rows[0]["nominal_properties"]))
        if any(tuple(sorted(item["nominal_properties"])) != property_fields for item in rows):
            raise ValueError(f"{material_field} descriptor fields are inconsistent")
        raw = np.asarray(
            [
                [float(item["nominal_properties"][field]) for field in property_fields]
                for item in rows
            ],
            dtype=float,
        )
        if raw.shape[0] != 4 or not np.all(np.isfinite(raw)):
            raise ValueError(f"{material_field} descriptors are not a finite four-row table")
        minimum = np.min(raw, axis=0)
        span = np.max(raw, axis=0) - minimum
        normalized = np.divide(
            raw - minimum,
            span,
            out=np.zeros_like(raw),
            where=span > 0.0,
        )
        fields_by_coordinate[coordinate] = property_fields
        matrix_by_coordinate[coordinate] = normalized
    return canonical_sha256(dossier), fields_by_coordinate, matrix_by_coordinate


def _transport_prior_material_selection(
    fields_by_coordinate: Mapping[int, tuple[str, ...]],
    matrix_by_coordinate: Mapping[int, np.ndarray],
) -> tuple[dict[int, int], dict[str, Any]]:
    field_weights = {
        0: {
            "bulk_conductivity_S_m": 1.0,
            "diffusivity_m2_s": 1.0,
            "diffusion_layer_thickness_mm": -1.0,
        },
        1: {
            "relative_conductivity": 1.0,
            "relative_diffusivity": 1.0,
        },
    }
    selected: dict[int, int] = {}
    diagnostics: dict[str, Any] = {}
    for coordinate, weights in field_weights.items():
        fields = fields_by_coordinate[coordinate]
        matrix = matrix_by_coordinate[coordinate]
        scores = np.zeros(matrix.shape[0], dtype=float)
        for field, weight in weights.items():
            try:
                column = fields.index(field)
            except ValueError as error:
                raise ValueError(f"transport prior requires descriptor field {field}") from error
            values = matrix[:, column]
            scores += values if weight > 0.0 else 1.0 - values
        selected[coordinate] = int(np.argmax(scores))
        diagnostics[str(coordinate)] = {
            "fields": list(weights),
            "category_scores": [float(value) for value in scores],
            "selected_category": selected[coordinate],
        }
    return selected, diagnostics


@dataclass(frozen=True)
class BaselineObservation:
    experiment_index: int
    vector: tuple[float, ...]
    score: float
    peak_safety_risk: float
    plan: StaticOptimizationPlan
    telemetry: tuple[float, ...] = ()


@dataclass(frozen=True)
class BaselineDecision:
    vector: tuple[float, ...]
    phase: str
    selected_policy: str
    trained_experiment_count: int
    acquisition_value: float | None = None
    diagnostics: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "selected_policy": self.selected_policy,
            "trained_experiment_count": self.trained_experiment_count,
            "acquisition_value": self.acquisition_value,
            "diagnostics": copy.deepcopy(self.diagnostics or {}),
            "search_vector": list(self.vector),
        }


class CompleteExperimentOptimizer:
    """Select one complete recipe and learn from its terminal score."""

    def __init__(
        self,
        *,
        algorithm_id: str,
        task_info: Mapping[str, Any],
        horizon: int,
        seed: int,
        configuration: Mapping[str, Any],
        electrochemical_material_family_id: str,
        electrochemical_workflow_mode: str = (ELECTROCHEMICAL_WORKFLOW_STATIC_SINGLE_STAGE),
    ) -> None:
        self.algorithm_id = str(algorithm_id)
        self.task_info = dict(task_info)
        self.horizon = int(horizon)
        self.seed = int(seed)
        self.configuration = copy.deepcopy(dict(configuration))
        self.electrochemical_material_family_id = str(electrochemical_material_family_id)
        self.electrochemical_workflow_mode = normalize_electrochemical_workflow_mode(
            electrochemical_workflow_mode
        )
        self.rng = np.random.default_rng(self.seed)
        self.categorical: tuple[tuple[int, int], ...]
        if (
            str(self.task_info.get("task_id", "")) == "electrochemical-conversion"
            and self.electrochemical_workflow_mode == ELECTROCHEMICAL_WORKFLOW_STATIC_SINGLE_STAGE
        ):
            self.dimension = ELECTROCHEMICAL_SINGLE_STAGE_DIMENSION
            self.categorical = ELECTROCHEMICAL_SINGLE_STAGE_CATEGORICAL_COORDINATES
        else:
            self.dimension = task_recipe_dimension(self.task_info)
            self.categorical = task_recipe_categorical_coordinates(self.task_info)
        self.categorical_surrogate_encoding = str(
            self.configuration.get("categorical_surrogate_encoding", _NOMINAL_ONE_HOT_ENCODING)
        )
        raw_telemetry_metric_ids = self.configuration.get("telemetry_metric_ids", ())
        if not isinstance(raw_telemetry_metric_ids, list | tuple) or not all(
            isinstance(metric, str) and metric for metric in raw_telemetry_metric_ids
        ):
            raise ValueError("telemetry_metric_ids must be a string list")
        self.telemetry_metric_ids = tuple(raw_telemetry_metric_ids)
        if self.categorical_surrogate_encoding not in _SURROGATE_ENCODINGS:
            raise ValueError(
                f"unsupported categorical_surrogate_encoding: {self.categorical_surrogate_encoding}"
            )
        self.material_information_sha256: str | None = None
        self.descriptor_fields_by_coordinate: dict[int, tuple[str, ...]] = {}
        self.descriptor_matrix_by_coordinate: dict[int, np.ndarray] = {}
        if self.categorical_surrogate_encoding in {
            _NOMINAL_PROPERTY_ENCODING,
            _SHUFFLED_PROPERTY_ENCODING,
        }:
            material_information_config = (
                {"mode": STATIC_MATERIAL_INFORMATION_NOMINAL}
                if self.categorical_surrogate_encoding == _NOMINAL_PROPERTY_ENCODING
                else {
                    "mode": STATIC_MATERIAL_INFORMATION_SHUFFLED,
                    "descriptor_permutation": copy.deepcopy(
                        self.configuration.get("descriptor_permutation")
                    ),
                }
            )
            (
                self.material_information_sha256,
                self.descriptor_fields_by_coordinate,
                self.descriptor_matrix_by_coordinate,
            ) = _nominal_property_encoding_contract(
                self.task_info,
                electrochemical_workflow_mode=self.electrochemical_workflow_mode,
                material_information_config=material_information_config,
                electrochemical_material_family_id=(self.electrochemical_material_family_id),
            )
            if set(self.descriptor_matrix_by_coordinate) != {
                coordinate for coordinate, _count in self.categorical
            }:
                raise ValueError(
                    "nominal material descriptors do not cover every categorical coordinate"
                )
        self.observations: list[BaselineObservation] = []
        self.compute_events: list[dict[str, Any]] = []

    def propose(self) -> BaselineDecision:
        raise NotImplementedError

    def observe(self, observation: BaselineObservation) -> None:
        if observation.experiment_index != len(self.observations):
            raise ValueError("baseline observations must be appended in experiment order")
        if len(observation.telemetry) != len(self.telemetry_metric_ids):
            raise ValueError("baseline telemetry does not match its frozen metric contract")
        self.observations.append(observation)

    def best_observation(self) -> BaselineObservation:
        if not self.observations:
            raise RuntimeError("baseline optimizer has no completed observations")
        return max(self.observations, key=lambda item: item.score)

    def _time_compute(self, event_kind: str, operation: Any) -> Any:
        started = process_time()
        result = operation()
        self.compute_events.append(
            {
                "event_index": len(self.compute_events) + 1,
                "event_kind": event_kind,
                "cpu_time_s": process_time() - started,
            }
        )
        return result

    def _model_vector(self, vector: np.ndarray) -> np.ndarray:
        values = np.asarray(np.clip(vector, 0.0, 1.0), dtype=float)
        categorical_indices = {coordinate for coordinate, _ in self.categorical}
        encoded = [
            np.asarray(
                [value for index, value in enumerate(values) if index not in categorical_indices],
                dtype=float,
            )
        ]
        for coordinate, category_count in self.categorical:
            category = min(int(values[coordinate] * category_count), category_count - 1)
            if self.categorical_surrogate_encoding == _NOMINAL_ONE_HOT_ENCODING:
                one_hot = np.zeros(category_count, dtype=float)
                one_hot[category] = 1.0
                encoded.append(one_hot)
            else:
                encoded.append(self.descriptor_matrix_by_coordinate[coordinate][category])
        return np.concatenate(encoded)

    def _training_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x = np.vstack(
            [self._model_vector(np.asarray(item.vector, dtype=float)) for item in self.observations]
        )
        score = np.asarray([item.score for item in self.observations], dtype=float)
        risk = np.asarray([item.peak_safety_risk for item in self.observations], dtype=float)
        return x, score, risk

    def _mixed_candidates(
        self,
        *,
        count: int,
        local_fraction: float,
        local_scale: float,
    ) -> np.ndarray:
        global_count = max(1, round(count * (1.0 - local_fraction)))
        local_count = max(0, count - global_count)
        candidates = [self.rng.random((global_count, self.dimension))]
        if local_count and self.observations:
            best = np.asarray(self.best_observation().vector, dtype=float)
            local = np.tile(best, (local_count, 1))
            categorical_indices = {coordinate for coordinate, _ in self.categorical}
            continuous = [
                index for index in range(self.dimension) if index not in categorical_indices
            ]
            if continuous:
                local[:, continuous] = np.clip(
                    local[:, continuous]
                    + self.rng.normal(0.0, local_scale, size=(local_count, len(continuous))),
                    0.0,
                    1.0,
                )
            for row in local:
                for coordinate, category_count in self.categorical:
                    if self.rng.random() < 0.2:
                        category = int(self.rng.integers(0, category_count))
                        row[coordinate] = (category + 0.5) / category_count
            candidates.append(local)
        return np.vstack(candidates)

    def manifest(self) -> dict[str, Any]:
        return {
            "algorithm_id": self.algorithm_id,
            "algorithm_seed": self.seed,
            "configuration": copy.deepcopy(self.configuration),
            "decision_scope": "complete_experiment",
            "optimization_feedback": (
                "terminal_score_and_processed_telemetry"
                if self.telemetry_metric_ids
                else "terminal_summary.leaderboard_score"
            ),
            "telemetry_metric_ids": list(self.telemetry_metric_ids),
            "safety_label": "peak_safety_risk",
            "intermediate_measurement_reward_used": False,
            "static_world": True,
            "hidden_world_fields_supplied": False,
            "horizon_visible": True,
            "experiment_horizon": self.horizon,
            "algorithm_designed_experiments": self.horizon,
            "electrochemical_workflow_mode": self.electrochemical_workflow_mode,
            "electrochemical_material_family_id": (self.electrochemical_material_family_id),
            "internal_representation": "unit_vector",
            "categorical_surrogate_encoding": self.categorical_surrogate_encoding,
            "material_information_condition": (
                STATIC_MATERIAL_INFORMATION_NOMINAL
                if self.categorical_surrogate_encoding == _NOMINAL_PROPERTY_ENCODING
                else STATIC_MATERIAL_INFORMATION_SHUFFLED
                if self.categorical_surrogate_encoding == _SHUFFLED_PROPERTY_ENCODING
                else "opaque_codes"
            ),
            "material_information_sha256": self.material_information_sha256,
            "descriptor_fields_by_coordinate": {
                str(coordinate): list(fields)
                for coordinate, fields in sorted(self.descriptor_fields_by_coordinate.items())
            },
        }

    def resource_usage(self) -> dict[str, Any]:
        return {
            "schema_version": "chemworld-method-resource-usage-0.1",
            "accounting_complete": True,
            "usage_source": "instrumented_in_process_classic_compute",
            "model_call_count": 0,
            "input_token_count": 0,
            "output_token_count": 0,
            "monetary_cost_usd": 0.0,
            "training_environment_step_count": 0,
            "cpu_time_s": sum(float(item["cpu_time_s"]) for item in self.compute_events),
            "gpu_time_s": 0.0,
            "compute_events": copy.deepcopy(self.compute_events),
            "model_provenance": {},
        }


class RandomOptimizer(CompleteExperimentOptimizer):
    def propose(self) -> BaselineDecision:
        return BaselineDecision(
            vector=tuple(float(item) for item in self.rng.random(self.dimension)),
            phase="exploration",
            selected_policy="uniform_random",
            trained_experiment_count=len(self.observations),
        )


class LatinHypercubeOptimizer(CompleteExperimentOptimizer):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.design = _balanced_design(
            count=self.horizon,
            dimension=self.dimension,
            categorical=self.categorical,
            rng=self.rng,
        )

    def propose(self) -> BaselineDecision:
        index = len(self.observations)
        return BaselineDecision(
            vector=tuple(float(item) for item in self.design[index]),
            phase="space_filling",
            selected_policy="balanced_latin_hypercube",
            trained_experiment_count=index,
        )


class GreedyOptimizer(CompleteExperimentOptimizer):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.n_initial = int(self.configuration.get("n_initial", 6))
        self.initial_design = _balanced_design(
            count=self.n_initial,
            dimension=self.dimension,
            categorical=self.categorical,
            rng=self.rng,
        )

    def propose(self) -> BaselineDecision:
        count = len(self.observations)
        if count < self.n_initial:
            return BaselineDecision(
                vector=tuple(float(item) for item in self.initial_design[count]),
                phase="initial",
                selected_policy="shared_balanced_initial_design",
                trained_experiment_count=count,
            )
        best = np.asarray(self.best_observation().vector, dtype=float)
        candidate = np.array(best, copy=True)
        categorical_map = dict(self.categorical)
        continuous = [index for index in range(self.dimension) if index not in categorical_map]
        scale = float(self.configuration.get("perturbation_scale", 0.12))
        candidate[continuous] = np.clip(
            candidate[continuous] + self.rng.normal(0.0, scale, len(continuous)),
            0.0,
            1.0,
        )
        if self.rng.random() < float(self.configuration.get("exploration_probability", 0.2)):
            coordinate = int(self.rng.integers(0, self.dimension))
            if coordinate in categorical_map:
                category_count = categorical_map[coordinate]
                current = min(int(best[coordinate] * category_count), category_count - 1)
                alternatives = [item for item in range(category_count) if item != current]
                selected = int(self.rng.choice(alternatives))
                candidate[coordinate] = (selected + 0.5) / category_count
            else:
                candidate[coordinate] = float(self.rng.random())
        return BaselineDecision(
            vector=tuple(float(item) for item in candidate),
            phase="local_search",
            selected_policy="best_observed_local_perturbation",
            trained_experiment_count=count,
        )


class SurrogateOptimizer(CompleteExperimentOptimizer):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.n_initial = int(self.configuration.get("n_initial", 6))
        self.initial_design = _balanced_design(
            count=self.n_initial,
            dimension=self.dimension,
            categorical=self.categorical,
            rng=self.rng,
        )
        self.initial_material_policy = str(
            self.configuration.get("initial_material_policy", "shared_balanced")
        )
        self.initial_material_policy_diagnostics: dict[str, Any] = {}
        if self.initial_material_policy == "transport_prior_v0.1":
            if self.categorical_surrogate_encoding != _NOMINAL_PROPERTY_ENCODING:
                raise ValueError("transport_prior_v0.1 requires nominal-property encoding")
            categories, diagnostics = _transport_prior_material_selection(
                self.descriptor_fields_by_coordinate,
                self.descriptor_matrix_by_coordinate,
            )
            for coordinate, category_count in self.categorical:
                category = categories[coordinate]
                self.initial_design[0, coordinate] = (category + 0.5) / category_count
            self.initial_material_policy_diagnostics = diagnostics
        elif self.initial_material_policy != "shared_balanced":
            raise ValueError(f"unsupported initial_material_policy: {self.initial_material_policy}")

    def manifest(self) -> dict[str, Any]:
        payload = super().manifest()
        payload["initial_material_policy"] = self.initial_material_policy
        payload["initial_material_policy_diagnostics"] = copy.deepcopy(
            self.initial_material_policy_diagnostics
        )
        return payload

    def _initial_decision(self) -> BaselineDecision | None:
        count = len(self.observations)
        if count >= self.n_initial:
            return None
        return BaselineDecision(
            vector=tuple(float(item) for item in self.initial_design[count]),
            phase="initial",
            selected_policy="shared_balanced_initial_design",
            trained_experiment_count=count,
        )

    def _candidate_matrix(self) -> np.ndarray:
        return self._mixed_candidates(
            count=int(self.configuration.get("n_candidates", 2048)),
            local_fraction=float(self.configuration.get("local_candidate_fraction", 0.5)),
            local_scale=float(self.configuration.get("local_candidate_scale", 0.12)),
        )


class StructuredGPEIOptimizer(SurrogateOptimizer):
    def propose(self) -> BaselineDecision:
        initial = self._initial_decision()
        if initial is not None:
            return initial
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import Matern, WhiteKernel

        x_train, score_train, _risk_train = self._training_arrays()
        kernel = Matern(length_scale=np.ones(x_train.shape[1]), nu=2.5) + WhiteKernel(
            noise_level=1.0e-4
        )
        model = GaussianProcessRegressor(
            kernel=kernel,
            normalize_y=True,
            alpha=1.0e-8,
            random_state=self.seed,
        )

        def fit() -> None:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                model.fit(x_train, score_train)

        self._time_compute("fit_score_gp", fit)
        candidates = self._candidate_matrix()
        x_candidates = np.vstack([self._model_vector(item) for item in candidates])
        mu, sigma = self._time_compute(
            "score_gp_acquisition_prediction",
            lambda: model.predict(x_candidates, return_std=True),
        )
        acquisition = _expected_improvement(mu, sigma, best=float(np.max(score_train)))
        selected = int(np.argmax(acquisition))
        return BaselineDecision(
            vector=tuple(float(item) for item in candidates[selected]),
            phase="acquisition",
            selected_policy="structured_gp_expected_improvement",
            trained_experiment_count=len(self.observations),
            acquisition_value=float(acquisition[selected]),
            diagnostics={
                "predicted_score_mean": float(mu[selected]),
                "predicted_score_std": float(sigma[selected]),
                "candidate_count": len(candidates),
            },
        )


class StructuredRFEIOptimizer(SurrogateOptimizer):
    def propose(self) -> BaselineDecision:
        initial = self._initial_decision()
        if initial is not None:
            return initial
        from sklearn.ensemble import RandomForestRegressor

        x_train, score_train, _risk_train = self._training_arrays()
        model = RandomForestRegressor(
            n_estimators=int(self.configuration.get("n_estimators", 192)),
            min_samples_leaf=2,
            random_state=self.seed,
            n_jobs=1,
        )
        self._time_compute("fit_score_rf", lambda: model.fit(x_train, score_train))
        candidates = self._candidate_matrix()
        x_candidates = np.vstack([self._model_vector(item) for item in candidates])

        def predict() -> tuple[np.ndarray, np.ndarray]:
            tree_predictions = np.vstack([tree.predict(x_candidates) for tree in model.estimators_])
            return tree_predictions.mean(axis=0), tree_predictions.std(axis=0)

        mu, sigma = self._time_compute("score_rf_acquisition_prediction", predict)
        acquisition = _expected_improvement(mu, sigma, best=float(np.max(score_train)))
        selected = int(np.argmax(acquisition))
        return BaselineDecision(
            vector=tuple(float(item) for item in candidates[selected]),
            phase="acquisition",
            selected_policy="structured_rf_expected_improvement",
            trained_experiment_count=len(self.observations),
            acquisition_value=float(acquisition[selected]),
            diagnostics={
                "predicted_score_mean": float(mu[selected]),
                "predicted_score_std": float(sigma[selected]),
                "candidate_count": len(candidates),
            },
        )


class TelemetryAwareStructuredRFEIOptimizer(SurrogateOptimizer):
    """RF-EI with shared multi-output splits over score and public telemetry."""

    def propose(self) -> BaselineDecision:
        initial = self._initial_decision()
        if initial is not None:
            return initial
        if not self.telemetry_metric_ids:
            raise ValueError("telemetry-aware RF requires telemetry_metric_ids")
        from sklearn.ensemble import RandomForestRegressor

        x_train, score_train, _risk_train = self._training_arrays()
        telemetry_train = np.asarray(
            [observation.telemetry for observation in self.observations],
            dtype=float,
        )
        targets = np.column_stack((score_train, telemetry_train))
        model = RandomForestRegressor(
            n_estimators=int(self.configuration.get("n_estimators", 192)),
            min_samples_leaf=2,
            random_state=self.seed,
            n_jobs=1,
        )
        self._time_compute("fit_score_and_telemetry_rf", lambda: model.fit(x_train, targets))
        candidates = self._candidate_matrix()
        x_candidates = np.vstack([self._model_vector(item) for item in candidates])

        def predict() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            tree_predictions = np.stack(
                [tree.predict(x_candidates) for tree in model.estimators_],
                axis=0,
            )
            score_predictions = tree_predictions[:, :, 0]
            telemetry_predictions = tree_predictions[:, :, 1:]
            return (
                score_predictions.mean(axis=0),
                score_predictions.std(axis=0),
                telemetry_predictions.mean(axis=0),
            )

        score_mu, score_sigma, telemetry_mu = self._time_compute(
            "score_and_telemetry_rf_acquisition_prediction",
            predict,
        )
        acquisition = _expected_improvement(
            score_mu,
            score_sigma,
            best=float(np.max(score_train)),
        )
        selected = int(np.argmax(acquisition))
        return BaselineDecision(
            vector=tuple(float(item) for item in candidates[selected]),
            phase="acquisition",
            selected_policy="telemetry_aware_rf_expected_improvement",
            trained_experiment_count=len(self.observations),
            acquisition_value=float(acquisition[selected]),
            diagnostics={
                "predicted_score_mean": float(score_mu[selected]),
                "predicted_score_std": float(score_sigma[selected]),
                "predicted_telemetry": {
                    metric: float(telemetry_mu[selected, index])
                    for index, metric in enumerate(self.telemetry_metric_ids)
                },
                "candidate_count": len(candidates),
            },
        )


class StructuredSafeGPEIOptimizer(SurrogateOptimizer):
    def propose(self) -> BaselineDecision:
        initial = self._initial_decision()
        if initial is not None:
            return initial
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import Matern, WhiteKernel

        x_train, score_train, risk_train = self._training_arrays()
        kernel = Matern(length_scale=np.ones(x_train.shape[1]), nu=2.5) + WhiteKernel(
            noise_level=1.0e-4
        )
        score_model = GaussianProcessRegressor(
            kernel=kernel,
            normalize_y=True,
            alpha=1.0e-8,
            random_state=self.seed,
        )
        risk_model = GaussianProcessRegressor(
            kernel=copy.deepcopy(kernel),
            normalize_y=True,
            alpha=1.0e-8,
            random_state=self.seed + 1,
        )

        def fit() -> None:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                score_model.fit(x_train, score_train)
                risk_model.fit(x_train, risk_train)

        self._time_compute("fit_score_and_risk_gp", fit)
        candidates = self._candidate_matrix()
        x_candidates = np.vstack([self._model_vector(item) for item in candidates])

        def predict() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            score_mu, score_sigma = score_model.predict(x_candidates, return_std=True)
            risk_mu, risk_sigma = risk_model.predict(x_candidates, return_std=True)
            return score_mu, score_sigma, risk_mu, risk_sigma

        score_mu, score_sigma, risk_mu, risk_sigma = self._time_compute(
            "score_and_risk_gp_acquisition_prediction", predict
        )
        threshold = float(self.task_info.get("safety_limit", 0.65))
        beta = float(self.configuration.get("risk_confidence_beta", 2.0))
        risk_upper = risk_mu + beta * risk_sigma
        safe = risk_upper <= threshold
        observed_safe = risk_train <= threshold
        best_safe = (
            float(np.max(score_train[observed_safe]))
            if np.any(observed_safe)
            else float(np.max(score_train))
        )
        acquisition = _expected_improvement(score_mu, score_sigma, best=best_safe)
        if np.any(safe):
            acquisition = np.where(safe, acquisition, -np.inf)
            selected = int(np.argmax(acquisition))
            policy = "structured_safe_gp_expected_improvement"
        else:
            selected = int(np.argmin(risk_upper))
            policy = "structured_safe_gp_minimum_risk_fallback"
        return BaselineDecision(
            vector=tuple(float(item) for item in candidates[selected]),
            phase="acquisition",
            selected_policy=policy,
            trained_experiment_count=len(self.observations),
            acquisition_value=float(acquisition[selected]),
            diagnostics={
                "predicted_score_mean": float(score_mu[selected]),
                "predicted_score_std": float(score_sigma[selected]),
                "predicted_risk_mean": float(risk_mu[selected]),
                "predicted_risk_std": float(risk_sigma[selected]),
                "predicted_risk_upper": float(risk_upper[selected]),
                "risk_threshold": threshold,
                "risk_confidence_beta": beta,
                "safe_candidate_count": int(np.count_nonzero(safe)),
                "candidate_count": len(candidates),
            },
        )


OPTIMIZER_TYPES = {
    "random": RandomOptimizer,
    "lhs": LatinHypercubeOptimizer,
    "greedy": GreedyOptimizer,
    "structured_gp_ei": StructuredGPEIOptimizer,
    "structured_rf_ei": StructuredRFEIOptimizer,
    "structured_safe_gp_ei": StructuredSafeGPEIOptimizer,
    "descriptor_gp_ei": StructuredGPEIOptimizer,
    "descriptor_rf_ei": StructuredRFEIOptimizer,
    "descriptor_telemetry_rf_ei": TelemetryAwareStructuredRFEIOptimizer,
    "shuffled_descriptor_gp_ei": StructuredGPEIOptimizer,
    "shuffled_descriptor_rf_ei": StructuredRFEIOptimizer,
    "transport_prior_gp_ei": StructuredGPEIOptimizer,
    "transport_prior_rf_ei": StructuredRFEIOptimizer,
    "telemetry_rf_ei": TelemetryAwareStructuredRFEIOptimizer,
}


def make_optimizer(
    *,
    algorithm_id: str,
    task_info: Mapping[str, Any],
    horizon: int,
    seed: int,
    configuration: Mapping[str, Any],
    electrochemical_material_family_id: str,
    electrochemical_workflow_mode: str = (ELECTROCHEMICAL_WORKFLOW_STATIC_SINGLE_STAGE),
) -> CompleteExperimentOptimizer:
    try:
        optimizer_type = OPTIMIZER_TYPES[algorithm_id]
    except KeyError as error:
        raise ValueError(f"unknown S0 baseline algorithm: {algorithm_id}") from error
    return optimizer_type(
        algorithm_id=algorithm_id,
        task_info=task_info,
        horizon=horizon,
        seed=seed,
        configuration=configuration,
        electrochemical_material_family_id=electrochemical_material_family_id,
        electrochemical_workflow_mode=electrochemical_workflow_mode,
    )


def plan_from_baseline_decision(
    decision: BaselineDecision,
    *,
    algorithm_id: str,
    task_info: Mapping[str, Any],
    electrochemical_workflow_mode: str = (ELECTROCHEMICAL_WORKFLOW_STATIC_SINGLE_STAGE),
) -> StaticOptimizationPlan:
    vector = np.asarray(decision.vector, dtype=float)
    recipe_kind = task_recipe_kind(dict(task_info))
    measurement_slots: tuple[str, ...]
    if recipe_kind == "reaction_crystallization":
        parameters = crystallization_single_stage_parameters_from_unit_vector(vector)
        measurement_slots = ("diagnostic-01-hplc", "diagnostic-02-hplc")
        measurement_objective = (
            "execute the frozen complete reaction-crystallization experiment and score "
            "its final assay"
        )
    elif recipe_kind == "electrochemical":
        parameters = (
            electrochemical_single_stage_parameters_from_unit_vector(vector)
            if electrochemical_workflow_mode == ELECTROCHEMICAL_WORKFLOW_STATIC_SINGLE_STAGE
            else electrochemical_recipe_parameters_from_unit_vector(vector)
        )
        measurement_slots = (
            ("diagnostic-01-ph_meter", "diagnostic-02-uvvis")
            if electrochemical_workflow_mode == ELECTROCHEMICAL_WORKFLOW_STATIC_SINGLE_STAGE
            else (
                "diagnostic-01-ph_meter",
                "diagnostic-02-uvvis",
                "diagnostic-03-uvvis",
            )
        )
        measurement_objective = (
            "execute the frozen complete electrochemical experiment and score its final assay"
        )
    else:
        parameters = None
        measurement_slots = tuple(
            str(item["slot_id"]) for item in scientific_measurement_slots(task_info)
        )
        measurement_objective = (
            "execute the frozen complete task experiment and score its final assay"
        )
    return StaticOptimizationPlan(
        experiment_intent=f"{algorithm_id} {decision.selected_policy}",
        search_vector=tuple(float(item) for item in vector),
        requested_measurement_slots=measurement_slots,
        measurement_objective=measurement_objective,
        expected_effect=(
            "provide one terminal leaderboard score for complete-experiment optimization"
        ),
        uncertainty=0.5,
        recipe_parameters=parameters,
    )


def _execute_validation_target(
    *,
    protocol: Mapping[str, Any],
    task_id: str,
    world_seed: int,
    target: str,
    plan: StaticOptimizationPlan,
    replicate_count: int,
    experiment_index_offset: int,
) -> dict[str, Any]:
    replicates: list[dict[str, Any]] = []
    for replicate_index in range(replicate_count):
        observation_seed = validation_observation_seed(
            task_id, world_seed, "paired-replicate", replicate_index
        )
        namespace = (
            f"{protocol['observation_noise_namespace']}-{task_id}-validation-"
            f"paired-replicate-{replicate_index:03d}"
        )
        with StaticOptimizationExperimentSession(
            task_id=task_id,
            seed=world_seed,
            experiment_horizon=1,
            experiment_index_offset=experiment_index_offset + replicate_index,
            observation_seed=observation_seed,
            observation_noise_namespace=namespace,
            electrochemical_workflow_mode=static_optimization_workflow_mode(protocol),
            electrochemical_material_family_id=(static_optimization_material_family_id(protocol)),
            crystallization_material_family_id=(
                static_optimization_crystallization_material_family_id(protocol)
            ),
            scoring_contract_id=static_optimization_scoring_contract_id(protocol),
        ) as session:
            result = session.execute(plan)
        replicates.append(
            {
                "replicate_index": replicate_index,
                "observation_seed": observation_seed,
                "observation_noise_namespace": namespace,
                "result": result.to_dict(),
            }
        )
    scores = [float(item["result"]["terminal_summary"]["leaderboard_score"]) for item in replicates]
    return {
        "target": target,
        "plan": plan.to_dict(),
        "plan_sha256": canonical_sha256(plan.to_dict()),
        "score_summary": _score_summary(scores),
        "scores": scores,
        "replicates": replicates,
    }


def run_baseline_cell(
    *,
    protocol: Mapping[str, Any],
    algorithm_id: str,
    algorithm_seed: int,
) -> dict[str, Any]:
    validate_static_optimization_protocol(protocol)
    validate_development_seed_policy(protocol, algorithm_seed=algorithm_seed)
    task_id = str(protocol["tasks"][0])
    task_info = get_task(task_id).to_dict()
    horizon = exploration_experiment_count(protocol)
    world_seed = int(protocol["world_policy"]["world_seed"])
    configuration = protocol["algorithms"][algorithm_id]
    workflow_mode = static_optimization_workflow_mode(protocol)
    material_family_id = static_optimization_material_family_id(protocol)
    optimizer = make_optimizer(
        algorithm_id=algorithm_id,
        task_info=task_info,
        horizon=horizon,
        seed=algorithm_seed,
        configuration=configuration,
        electrochemical_material_family_id=material_family_id,
        electrochemical_workflow_mode=workflow_mode,
    )
    experiments: list[dict[str, Any]] = []
    history: list[dict[str, Any]] = []
    observation_seed = exploration_observation_seed(task_id, world_seed)
    for experiment_index in range(horizon):
        decision = optimizer.propose()
        plan = plan_from_baseline_decision(
            decision,
            algorithm_id=algorithm_id,
            task_info=task_info,
            electrochemical_workflow_mode=workflow_mode,
        )
        with StaticOptimizationExperimentSession(
            task_id=task_id,
            seed=world_seed,
            experiment_horizon=1,
            experiment_index_offset=experiment_index,
            observation_seed=observation_seed,
            observation_noise_namespace=(
                f"{protocol['observation_noise_namespace']}-{task_id}-"
                f"experiment-{experiment_index:03d}"
            ),
            electrochemical_workflow_mode=workflow_mode,
            electrochemical_material_family_id=material_family_id,
            crystallization_material_family_id=(
                static_optimization_crystallization_material_family_id(protocol)
            ),
            scoring_contract_id=static_optimization_scoring_contract_id(protocol),
        ) as session:
            result = session.execute(plan)
        score = float(result.terminal_summary["leaderboard_score"])
        telemetry: tuple[float, ...] = ()
        if optimizer.telemetry_metric_ids:
            processed_estimate = None
            for evidence in reversed(result.measurement_evidence):
                candidate = evidence.get("processed_estimate")
                if isinstance(candidate, Mapping):
                    processed_estimate = candidate
                    break
            if processed_estimate is None:
                raise RuntimeError("telemetry-aware baseline lacks processed measurement evidence")
            missing = [
                metric
                for metric in optimizer.telemetry_metric_ids
                if metric not in processed_estimate
            ]
            if missing:
                raise RuntimeError(f"telemetry-aware baseline lacks metrics: {missing}")
            telemetry = tuple(
                float(processed_estimate[metric]) for metric in optimizer.telemetry_metric_ids
            )
        optimizer.observe(
            BaselineObservation(
                experiment_index=experiment_index,
                vector=decision.vector,
                score=score,
                peak_safety_risk=float(result.peak_safety_risk),
                plan=plan,
                telemetry=telemetry,
            )
        )
        public_record = result.public_record()
        history.append(public_record)
        experiments.append(
            {
                "result": result.to_dict(),
                "decision_audit": {
                    "schema_version": STATIC_BASELINE_RESULT_VERSION,
                    "algorithm_id": algorithm_id,
                    "algorithm_seed": algorithm_seed,
                    "decision_source": "algorithm",
                    "algorithm_decision_consumed": True,
                    "decision": decision.to_dict(),
                    "feedback_received_after_execution": {
                        "leaderboard_score": score,
                        "peak_safety_risk": float(result.peak_safety_risk),
                        "processed_telemetry": {
                            metric: telemetry[index]
                            for index, metric in enumerate(optimizer.telemetry_metric_ids)
                        },
                    },
                    "reward_contract": copy.deepcopy(protocol["reward_contract"]),
                    "hidden_world_fields_supplied": False,
                },
            }
        )
    incumbent = optimizer.best_observation()
    incumbent_plan = incumbent.plan
    validation_config = protocol["validation_budget"]
    incumbent_count = int(validation_config["incumbent_replicates"])
    recommendation_count = int(validation_config["recommendation_replicates"])
    incumbent_validation = _execute_validation_target(
        protocol=protocol,
        task_id=task_id,
        world_seed=world_seed,
        target="incumbent",
        plan=incumbent_plan,
        replicate_count=incumbent_count,
        experiment_index_offset=horizon,
    )
    recommendation_validation = _execute_validation_target(
        protocol=protocol,
        task_id=task_id,
        world_seed=world_seed,
        target="recommendation",
        plan=incumbent_plan,
        replicate_count=recommendation_count,
        experiment_index_offset=horizon + incumbent_count,
    )
    validated_mean = float(recommendation_validation["score_summary"]["mean"])
    incumbent_mean = float(incumbent_validation["score_summary"]["mean"])
    method_id = f"{algorithm_id}_seed{algorithm_seed}"
    recommendation = {
        "schema_version": "chemworld-static-final-synthesis-0.3-s0-dev",
        "recommended_search_vector": list(incumbent.plan.search_vector),
        "recommended_recipe_parameters": copy.deepcopy(incumbent.plan.recipe_parameters),
        "recommended_measurement_slots": list(incumbent.plan.requested_measurement_slots),
        "recommendation_type": "tested",
        "source_experiment_indices": [incumbent.experiment_index],
        "predicted_score": incumbent.score,
        "confidence": None,
        "method_summary": "deterministic best-observed baseline submission",
        "evidence_refs": [],
        "remaining_risks": [],
        "recommended_followup": "paired blind validation",
    }
    scores = [item.score for item in optimizer.observations]
    validation = {
        "blind": True,
        "feedback_returned_to_agent": False,
        "incumbent_source_experiment_index": incumbent.experiment_index,
        "incumbent_observed_score": incumbent.score,
        "incumbent": incumbent_validation,
        "recommendation": recommendation_validation,
        "primary_validated_recommendation_score_mean": validated_mean,
        "validated_incumbent_score_mean": incumbent_mean,
        "recommendation_gain_over_incumbent_mean": validated_mean - incumbent_mean,
    }
    protocol_hash = canonical_sha256(protocol)
    method_config = {
        "algorithm_id": algorithm_id,
        "algorithm_seed": algorithm_seed,
        "configuration": copy.deepcopy(configuration),
    }
    return {
        "schema_version": STATIC_BASELINE_RESULT_VERSION,
        "formal_result": bool(protocol.get("formal_result", False)),
        "benchmark_claim_allowed": bool(protocol.get("benchmark_claim_allowed", False)),
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": protocol_hash,
        "method_config_freeze_id": protocol["freeze_id"],
        "method_config_sha256": canonical_sha256(method_config),
        "method_id": method_id,
        "provider_mode": "local_classic_optimizer",
        "method": method_config,
        "cell": {
            "cell_id": f"{method_id}:{task_id}",
            "task_id": task_id,
            "world_seed": world_seed,
            "algorithm_seed": algorithm_seed,
            "world_policy": "static_for_entire_campaign",
        },
        "world_policy": copy.deepcopy(protocol["world_policy"]),
        "reward_contract": copy.deepcopy(protocol.get("reward_contract", {})),
        "scientific_campaign_budget": copy.deepcopy(protocol["scientific_campaign_budget"]),
        "executor_contract": copy.deepcopy(protocol["executor_contract"]),
        "recommendation_stage_present": True,
        "cell_status": "completed",
        "failure": None,
        "agent_manifest": optimizer.manifest(),
        "resources": optimizer.resource_usage(),
        "planned_experiment_count": horizon,
        "experiment_count": len(experiments),
        "completed_experiment_count": len(experiments),
        "planned_algorithm_decision_count": horizon,
        "completed_algorithm_decision_count": sum(
            bool(item["decision_audit"]["algorithm_decision_consumed"]) for item in experiments
        ),
        "scores": scores,
        "experiments": experiments,
        "public_history": history,
        "planned_synthesis_call_count": 0,
        "completed_synthesis_call_count": 0,
        "final_synthesis": {
            "recommendation": recommendation,
            "synthesis_audit": {
                "mode": "deterministic_best_observed_selection",
                "provider_model": None,
                "provider_attempts": 0,
                "recommendation_sha256": canonical_sha256(recommendation),
                "validation_feedback_returned_to_agent": False,
            },
            "executes_experiment": False,
            "validation_feedback_returned_to_agent": False,
        },
        "planned_validation_experiment_count": incumbent_count + recommendation_count,
        "completed_validation_experiment_count": incumbent_count + recommendation_count,
        "planned_predictive_validation_experiment_count": 0,
        "completed_predictive_validation_experiment_count": 0,
        "total_physical_experiment_count": horizon + incumbent_count + recommendation_count,
        "predictive_validation": None,
        "validation": validation,
        "primary_score": validated_mean,
    }


def aggregate_baseline_cells(cells: list[Mapping[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for cell in cells:
        grouped.setdefault(str(cell["method"]["algorithm_id"]), []).append(cell)
    algorithms: list[dict[str, Any]] = []
    for algorithm_id, rows in sorted(grouped.items()):
        rows = sorted(rows, key=lambda item: int(item["method"]["algorithm_seed"]))
        first_values = [float(item["scores"][0]) for item in rows]
        best_values = [max(float(value) for value in item["scores"]) for item in rows]
        validated = [float(item["primary_score"]) for item in rows]
        best_rounds = [
            [float(value) for value in item["scores"]].index(max(item["scores"])) + 1
            for item in rows
        ]
        best_curves: list[list[float]] = []
        for item in rows:
            current = float("-inf")
            curve: list[float] = []
            for value in item["scores"]:
                current = max(current, float(value))
                curve.append(current)
            best_curves.append(curve)
        auc_values = [statistics.fmean(curve) for curve in best_curves]
        algorithms.append(
            {
                "algorithm_id": algorithm_id,
                "run_count": len(rows),
                "algorithm_seeds": [int(item["method"]["algorithm_seed"]) for item in rows],
                "first_score": _score_summary(first_values),
                "best_exploration_score": _score_summary(best_values),
                "best_so_far_area_under_curve": _score_summary(auc_values),
                "validated_final_score": _score_summary(validated),
                "best_round": _score_summary([float(value) for value in best_rounds]),
                "best_so_far_curve": [
                    {
                        "round": round_index + 1,
                        **_score_summary([curve[round_index] for curve in best_curves]),
                    }
                    for round_index in range(len(best_curves[0]))
                ],
                "cpu_time_s": _score_summary(
                    [float(item["resources"]["cpu_time_s"]) for item in rows]
                ),
                "runs": [
                    {
                        "method_id": item["method_id"],
                        "algorithm_seed": int(item["method"]["algorithm_seed"]),
                        "first_score": float(item["scores"][0]),
                        "best_exploration_score": max(item["scores"]),
                        "best_so_far_area_under_curve": auc_values[index],
                        "best_round": list(item["scores"]).index(max(item["scores"])) + 1,
                        "validated_final_score": float(item["primary_score"]),
                    }
                    for index, item in enumerate(rows)
                ],
            }
        )
    return {
        "schema_version": "chemworld-static-optimization-baseline-aggregate-0.1-s0-dev",
        "formal_result": False,
        "benchmark_claim_allowed": False,
        "reward_contract": ("terminal leaderboard score only; safety risk modeled separately"),
        "algorithm_count": len(algorithms),
        "run_count": len(cells),
        "algorithms": algorithms,
    }


__all__ = [
    "OPTIMIZER_TYPES",
    "STATIC_BASELINE_RESULT_VERSION",
    "BaselineDecision",
    "BaselineObservation",
    "CompleteExperimentOptimizer",
    "aggregate_baseline_cells",
    "make_optimizer",
    "plan_from_baseline_decision",
    "run_baseline_cell",
]
