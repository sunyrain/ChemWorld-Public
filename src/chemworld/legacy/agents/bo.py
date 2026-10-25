"""Surrogate-model optimization baselines."""

from __future__ import annotations

import warnings
from collections.abc import Callable
from math import erf, pi, sqrt
from time import perf_counter, process_time
from typing import Any, TypeVar

import numpy as np
from sklearn.exceptions import ConvergenceWarning

from chemworld.agents.base import BaseAgent, HistoryRecord
from chemworld.agents.interaction import InteractionCapabilities
from chemworld.agents.recipe_sequence import RecipeSequenceMixin
from chemworld.agents.task_recipes import (
    TASK_RECIPE_SPACE_VERSION,
    sample_conservative_task_recipe,
    sample_task_recipe,
    task_recipe_event_count,
    task_recipe_to_model_vector,
    task_recipe_to_vector,
)

T = TypeVar("T")


def _normal_cdf(z: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.vectorize(erf)(z / sqrt(2.0)))


def _normal_pdf(z: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * z**2) / sqrt(2.0 * pi)


def _expected_improvement(
    mu: np.ndarray,
    sigma: np.ndarray,
    best: float,
    xi: float = 0.01,
) -> np.ndarray:
    sigma = np.maximum(sigma, 1.0e-9)
    improvement = mu - best - xi
    z = improvement / sigma
    return improvement * _normal_cdf(z) + sigma * _normal_pdf(z)


def _probability_improvement(
    mu: np.ndarray,
    sigma: np.ndarray,
    best: float,
    xi: float = 0.01,
) -> np.ndarray:
    sigma = np.maximum(sigma, 1.0e-9)
    return _normal_cdf((mu - best - xi) / sigma)


class CandidateSurrogateMixin:
    rng: np.random.Generator
    task_info: dict[str, Any]

    def _start_recipe(self, action: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    def _reset_surrogate_diagnostics(self) -> None:
        self._decision_trace: list[dict[str, Any]] = []
        self._formal_compute_events: list[dict[str, Any]] = []

    def _timed_compute(self, event_kind: str, operation: Callable[[], T]) -> T:
        """Measure one non-overlapping surrogate fit or acquisition component."""

        cpu_started = process_time()
        wall_started = perf_counter()
        result = operation()
        event = {
            "event_index": len(self._formal_compute_events) + 1,
            "event_kind": event_kind,
            "cpu_time_s": process_time() - cpu_started,
            "wall_time_s": perf_counter() - wall_started,
        }
        self._formal_compute_events.append(event)
        return result

    def formal_compute_events(self) -> list[dict[str, Any]]:
        """Return cumulative component timings for formal resource accounting."""

        return [dict(item) for item in getattr(self, "_formal_compute_events", [])]

    def method_resource_usage(self) -> dict[str, Any]:
        return {
            "schema_version": "chemworld-method-resource-usage-0.1",
            "accounting_complete": True,
            "usage_source": "instrumented_in_process_classic_compute",
            "model_call_count": 0,
            "input_token_count": 0,
            "output_token_count": 0,
            "monetary_cost_usd": 0.0,
            "training_environment_step_count": 0,
            "cpu_time_s": sum(
                float(item["cpu_time_s"])
                for item in getattr(self, "_formal_compute_events", [])
            ),
            "gpu_time_s": 0.0,
            "model_provenance": {},
        }

    def _start_surrogate_recipe(
        self,
        action: dict[str, Any],
        *,
        phase: str,
        trained_recipe_count: int,
        best_observed_score: float | None = None,
        acquisition_value: float | None = None,
        selected_policy: str,
        decision_diagnostics: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        trace = {
            "trace_type": "surrogate_recipe_decision",
            "phase": phase,
            "trained_recipe_count": trained_recipe_count,
            "used_surrogate": phase == "acquisition",
            "selected_policy": selected_policy,
            "best_observed_score": best_observed_score,
            "acquisition_value": acquisition_value,
            "selected_recipe": dict(action),
        }
        if decision_diagnostics:
            trace["decision_diagnostics"] = dict(decision_diagnostics)
        self._decision_trace.append(trace)
        return self._start_recipe(action)

    def agent_trace(self) -> list[dict[str, Any]]:
        return [dict(item) for item in getattr(self, "_decision_trace", [])]

    def _candidate_actions(self, count: int) -> list[dict[str, Any]]:
        return [sample_task_recipe(self.task_info, self.rng) for _ in range(count)]

    def _xy(self, history: list[HistoryRecord]) -> tuple[np.ndarray, np.ndarray]:
        x = np.vstack([self._model_vector(record.action) for record in history])
        y = np.asarray([record.reward for record in history], dtype=float)
        return x, y

    def _model_vector(self, action: dict[str, Any]) -> np.ndarray:
        return task_recipe_to_vector(action)


class GaussianProcessBOAgent(RecipeSequenceMixin, CandidateSurrogateMixin, BaseAgent):
    name = "gp_bo"

    def __init__(self, n_initial: int = 4, n_candidates: int = 512) -> None:
        self.n_initial = n_initial
        self.n_candidates = n_candidates
        self.effective_n_initial = n_initial

    def interaction_capabilities(self) -> InteractionCapabilities:
        capabilities = super().interaction_capabilities()
        return InteractionCapabilities(
            decision_scope=capabilities.decision_scope,
            consumes_intermediate_observations=capabilities.consumes_intermediate_observations,
            consumes_spectra=capabilities.consumes_spectra,
            adapts_within_experiment=capabilities.adapts_within_experiment,
            adapts_across_experiments=True,
            emits_structured_decision_audit=capabilities.emits_structured_decision_audit,
        )

    def reset(self, task_info: dict[str, Any], seed: int) -> None:
        super().reset(task_info, seed)
        self.rng = np.random.default_rng(seed)
        experiment_capacity = max(
            1,
            int(task_info.get("budget", 1)) // task_recipe_event_count(task_info),
        )
        self.effective_n_initial = min(
            self.n_initial,
            max(1, experiment_capacity - 1),
        )
        self._reset_surrogate_diagnostics()

    def act(self, history: list[HistoryRecord]) -> dict[str, Any]:
        del history
        pending = self._pop_pending_event()
        if pending is not None:
            return pending

        recipe_history = self._recipe_history
        if len(recipe_history) < self.effective_n_initial:
            return self._start_surrogate_recipe(
                sample_task_recipe(self.task_info, self.rng),
                phase="initial",
                trained_recipe_count=len(recipe_history),
                selected_policy="random_initial_design",
            )

        candidates, y_train, mu, sigma = self._candidate_predictions(recipe_history)
        acquisition = _expected_improvement(mu, sigma, best=float(np.max(y_train)))
        selected_index = int(np.argmax(acquisition))
        return self._start_surrogate_recipe(
            candidates[selected_index],
            phase="acquisition",
            trained_recipe_count=len(recipe_history),
            best_observed_score=float(np.max(y_train)),
            acquisition_value=float(acquisition[selected_index]),
            selected_policy="gp_expected_improvement",
        )

    def _candidate_predictions(
        self,
        recipe_history: list[HistoryRecord],
    ) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray, np.ndarray]:
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import Matern, WhiteKernel

        x_train, y_train = self._xy(recipe_history)
        kernel = Matern(length_scale=np.ones(x_train.shape[1]), nu=2.5) + WhiteKernel(
            noise_level=1.0e-4
        )
        model = GaussianProcessRegressor(kernel=kernel, normalize_y=True, random_state=self.seed)
        def fit_model() -> None:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                model.fit(x_train, y_train)

        self._timed_compute("fit", fit_model)

        def optimize_acquisition() -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray]:
            candidates = self._candidate_actions(self.n_candidates)
            x_candidates = np.vstack([self._model_vector(action) for action in candidates])
            mu, sigma = model.predict(x_candidates, return_std=True)
            return candidates, mu, sigma

        candidates, mu, sigma = self._timed_compute(
            "acquisition_optimization", optimize_acquisition
        )
        return candidates, y_train, mu, sigma

    def manifest(self) -> dict[str, Any]:
        manifest = super().manifest()
        manifest.update(
            {
                "surrogate_family": "gaussian_process",
                "n_initial": self.n_initial,
                "effective_n_initial": self.effective_n_initial,
                "n_candidates": self.n_candidates,
                "search_space_version": TASK_RECIPE_SPACE_VERSION,
            }
        )
        return manifest


class GaussianProcessPIAgent(GaussianProcessBOAgent):
    """Gaussian-process optimization using probability of improvement."""

    name = "gp_pi"

    def __init__(self, n_initial: int = 4, n_candidates: int = 512, xi: float = 0.01) -> None:
        super().__init__(n_initial=n_initial, n_candidates=n_candidates)
        self.xi = xi

    def act(self, history: list[HistoryRecord]) -> dict[str, Any]:
        del history
        pending = self._pop_pending_event()
        if pending is not None:
            return pending
        recipe_history = self._recipe_history
        if len(recipe_history) < self.effective_n_initial:
            return self._start_surrogate_recipe(
                sample_task_recipe(self.task_info, self.rng),
                phase="initial",
                trained_recipe_count=len(recipe_history),
                selected_policy="random_initial_design",
            )
        candidates, y_train, mu, sigma = self._candidate_predictions(recipe_history)
        acquisition = _probability_improvement(
            mu,
            sigma,
            best=float(np.max(y_train)),
            xi=self.xi,
        )
        selected_index = int(np.argmax(acquisition))
        return self._start_surrogate_recipe(
            candidates[selected_index],
            phase="acquisition",
            trained_recipe_count=len(recipe_history),
            best_observed_score=float(np.max(y_train)),
            acquisition_value=float(acquisition[selected_index]),
            selected_policy="gp_probability_improvement",
        )

    def manifest(self) -> dict[str, Any]:
        manifest = super().manifest()
        manifest.update({"acquisition_function": "probability_improvement", "xi": self.xi})
        return manifest


class StructuredGaussianProcessPIAgent(GaussianProcessPIAgent):
    """GP probability-of-improvement with typed categorical coordinates."""

    name = "structured_gp_pi"

    def _model_vector(self, action: dict[str, Any]) -> np.ndarray:
        return task_recipe_to_model_vector(self.task_info, action)

    def manifest(self) -> dict[str, Any]:
        manifest = super().manifest()
        manifest.update({"recipe_encoding": "continuous_plus_material_one_hot"})
        return manifest


class StructuredGaussianProcessBOAgent(GaussianProcessBOAgent):
    """GP-EI with one-hot material choices and continuous process coordinates."""

    name = "structured_gp_bo"

    def _model_vector(self, action: dict[str, Any]) -> np.ndarray:
        return task_recipe_to_model_vector(self.task_info, action)

    def manifest(self) -> dict[str, Any]:
        manifest = super().manifest()
        manifest.update({"recipe_encoding": "continuous_plus_material_one_hot"})
        return manifest


class GaussianProcessUCBAgent(GaussianProcessBOAgent):
    """Gaussian-process optimization using an upper-confidence bound."""

    name = "gp_ucb"

    def __init__(
        self,
        n_initial: int = 4,
        n_candidates: int = 512,
        exploration_weight: float = 2.0,
    ) -> None:
        super().__init__(n_initial=n_initial, n_candidates=n_candidates)
        self.exploration_weight = exploration_weight

    def act(self, history: list[HistoryRecord]) -> dict[str, Any]:
        del history
        pending = self._pop_pending_event()
        if pending is not None:
            return pending
        recipe_history = self._recipe_history
        if len(recipe_history) < self.effective_n_initial:
            return self._start_surrogate_recipe(
                sample_task_recipe(self.task_info, self.rng),
                phase="initial",
                trained_recipe_count=len(recipe_history),
                selected_policy="random_initial_design",
            )
        candidates, y_train, mu, sigma = self._candidate_predictions(recipe_history)
        acquisition = mu + self.exploration_weight * sigma
        selected_index = int(np.argmax(acquisition))
        return self._start_surrogate_recipe(
            candidates[selected_index],
            phase="acquisition",
            trained_recipe_count=len(recipe_history),
            best_observed_score=float(np.max(y_train)),
            acquisition_value=float(acquisition[selected_index]),
            selected_policy="gp_upper_confidence_bound",
        )

    def manifest(self) -> dict[str, Any]:
        manifest = super().manifest()
        manifest.update(
            {
                "acquisition_function": "upper_confidence_bound",
                "exploration_weight": self.exploration_weight,
            }
        )
        return manifest


class StructuredGaussianProcessUCBAgent(GaussianProcessUCBAgent):
    """GP upper-confidence-bound search with typed categorical coordinates."""

    name = "structured_gp_ucb"

    def _model_vector(self, action: dict[str, Any]) -> np.ndarray:
        return task_recipe_to_model_vector(self.task_info, action)

    def manifest(self) -> dict[str, Any]:
        manifest = super().manifest()
        manifest.update({"recipe_encoding": "continuous_plus_material_one_hot"})
        return manifest


class RandomForestEIAgent(RecipeSequenceMixin, CandidateSurrogateMixin, BaseAgent):
    name = "rf_ei"

    def interaction_capabilities(self) -> InteractionCapabilities:
        capabilities = super().interaction_capabilities()
        return InteractionCapabilities(
            decision_scope=capabilities.decision_scope,
            consumes_intermediate_observations=capabilities.consumes_intermediate_observations,
            consumes_spectra=capabilities.consumes_spectra,
            adapts_within_experiment=capabilities.adapts_within_experiment,
            adapts_across_experiments=True,
            emits_structured_decision_audit=capabilities.emits_structured_decision_audit,
        )

    def __init__(
        self,
        n_initial: int = 4,
        n_candidates: int = 512,
        n_estimators: int = 128,
    ) -> None:
        self.n_initial = n_initial
        self.n_candidates = n_candidates
        self.n_estimators = n_estimators
        self.effective_n_initial = n_initial

    def reset(self, task_info: dict[str, Any], seed: int) -> None:
        super().reset(task_info, seed)
        self.rng = np.random.default_rng(seed)
        experiment_capacity = max(
            1,
            int(task_info.get("budget", 1)) // task_recipe_event_count(task_info),
        )
        self.effective_n_initial = min(
            self.n_initial,
            max(1, experiment_capacity - 1),
        )
        self._reset_surrogate_diagnostics()

    def act(self, history: list[HistoryRecord]) -> dict[str, Any]:
        del history
        pending = self._pop_pending_event()
        if pending is not None:
            return pending

        recipe_history = self._recipe_history
        if len(recipe_history) < self.effective_n_initial:
            return self._start_surrogate_recipe(
                sample_task_recipe(self.task_info, self.rng),
                phase="initial",
                trained_recipe_count=len(recipe_history),
                selected_policy="random_initial_design",
            )

        from sklearn.ensemble import RandomForestRegressor

        x_train, y_train = self._xy(recipe_history)
        model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            min_samples_leaf=2,
            random_state=self.seed,
        )
        self._timed_compute("fit", lambda: model.fit(x_train, y_train))

        def optimize_acquisition() -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray]:
            candidates = self._candidate_actions(self.n_candidates)
            x_candidates = np.vstack([self._model_vector(action) for action in candidates])
            tree_predictions = np.vstack(
                [tree.predict(x_candidates) for tree in model.estimators_]
            )
            return candidates, tree_predictions.mean(axis=0), tree_predictions.std(axis=0)

        candidates, mu, sigma = self._timed_compute(
            "acquisition_optimization", optimize_acquisition
        )
        acquisition = _expected_improvement(mu, sigma, best=float(np.max(y_train)))
        selected_index = int(np.argmax(acquisition))
        return self._start_surrogate_recipe(
            candidates[selected_index],
            phase="acquisition",
            trained_recipe_count=len(recipe_history),
            best_observed_score=float(np.max(y_train)),
            acquisition_value=float(acquisition[selected_index]),
            selected_policy="rf_expected_improvement",
        )

    def manifest(self) -> dict[str, Any]:
        manifest = super().manifest()
        manifest.update(
            {
                "surrogate_family": "random_forest",
                "n_initial": self.n_initial,
                "effective_n_initial": self.effective_n_initial,
                "n_candidates": self.n_candidates,
                "n_estimators": self.n_estimators,
                "search_space_version": TASK_RECIPE_SPACE_VERSION,
            }
        )
        return manifest


class StructuredRandomForestEIAgent(RandomForestEIAgent):
    """Random-forest expected improvement with typed categorical coordinates."""

    name = "structured_rf_ei"

    def _model_vector(self, action: dict[str, Any]) -> np.ndarray:
        return task_recipe_to_model_vector(self.task_info, action)

    def manifest(self) -> dict[str, Any]:
        manifest = super().manifest()
        manifest.update({"recipe_encoding": "continuous_plus_material_one_hot"})
        return manifest


class SafetyConstrainedBOAgent(GaussianProcessBOAgent):
    name = "safe_gp_bo"

    def __init__(
        self,
        n_initial: int = 4,
        n_candidates: int = 768,
        risk_threshold: float = 0.65,
        risk_confidence_beta: float = 2.0,
        initial_perturbation_scale: float = 0.06,
    ) -> None:
        super().__init__(n_initial=n_initial, n_candidates=n_candidates)
        if risk_confidence_beta < 0.0:
            raise ValueError("risk_confidence_beta must be non-negative")
        if not 0.0 <= initial_perturbation_scale <= 0.25:
            raise ValueError("initial_perturbation_scale must be between zero and 0.25")
        self.risk_threshold = risk_threshold
        self.risk_confidence_beta = risk_confidence_beta
        self.initial_perturbation_scale = initial_perturbation_scale
        self.effective_risk_threshold = risk_threshold

    def reset(self, task_info: dict[str, Any], seed: int) -> None:
        super().reset(task_info, seed)
        task_limit = float(task_info.get("safety_limit", self.risk_threshold))
        self.effective_risk_threshold = min(self.risk_threshold, task_limit)

    def act(self, history: list[HistoryRecord]) -> dict[str, Any]:
        del history
        pending = self._pop_pending_event()
        if pending is not None:
            return pending

        recipe_history = self._recipe_history
        if len(recipe_history) < self.effective_n_initial:
            return self._start_surrogate_recipe(
                sample_conservative_task_recipe(
                    self.task_info,
                    self.rng,
                    perturbation_scale=self.initial_perturbation_scale,
                ),
                phase="initial",
                trained_recipe_count=len(recipe_history),
                selected_policy="conservative_initial_design",
                decision_diagnostics={
                    "risk_label": "experiment_peak_safety_risk",
                    "risk_threshold": self.effective_risk_threshold,
                },
            )

        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import Matern, WhiteKernel

        x_train, y_train = self._xy(recipe_history)
        risk_train = np.asarray(
            [
                record.observation.get(
                    "experiment_peak_safety_risk",
                    record.observation.get("safety_risk", 1.0),
                )
                for record in recipe_history
            ],
            dtype=float,
        )
        kernel = Matern(length_scale=np.ones(x_train.shape[1]), nu=2.5) + WhiteKernel(
            noise_level=1.0e-4
        )

        score_model = GaussianProcessRegressor(
            kernel=kernel,
            normalize_y=True,
            random_state=self.seed,
        )
        risk_model = GaussianProcessRegressor(
            kernel=kernel,
            normalize_y=True,
            random_state=self.seed + 1,
        )
        def fit_models() -> None:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                score_model.fit(x_train, y_train)
                risk_model.fit(x_train, risk_train)

        self._timed_compute("fit", fit_models)

        def optimize_acquisition() -> tuple[
            list[dict[str, Any]], np.ndarray, np.ndarray, np.ndarray, np.ndarray
        ]:
            candidates = self._candidate_actions(self.n_candidates)
            x_candidates = np.vstack([self._model_vector(action) for action in candidates])
            mu, sigma = score_model.predict(x_candidates, return_std=True)
            risk_mu, risk_sigma = risk_model.predict(x_candidates, return_std=True)
            return candidates, mu, sigma, risk_mu, risk_sigma

        candidates, mu, sigma, risk_mu, risk_sigma = self._timed_compute(
            "acquisition_optimization", optimize_acquisition
        )
        observed_safe = risk_train <= self.effective_risk_threshold
        best_safe_score = (
            float(np.max(y_train[observed_safe]))
            if np.any(observed_safe)
            else float(np.min(y_train))
        )
        acquisition = _expected_improvement(mu, sigma, best=best_safe_score)

        safety_margin = risk_mu + self.risk_confidence_beta * risk_sigma
        safe_mask = safety_margin <= self.effective_risk_threshold
        if np.any(safe_mask):
            acquisition = np.where(safe_mask, acquisition, -np.inf)
            selected_index = int(np.argmax(acquisition))
            return self._start_surrogate_recipe(
                candidates[selected_index],
                phase="acquisition",
                trained_recipe_count=len(recipe_history),
                best_observed_score=best_safe_score,
                acquisition_value=float(acquisition[selected_index]),
                selected_policy="safe_gp_expected_improvement",
                decision_diagnostics={
                    "risk_label": "experiment_peak_safety_risk",
                    "risk_threshold": self.effective_risk_threshold,
                    "risk_confidence_beta": self.risk_confidence_beta,
                    "predicted_risk_mean": float(risk_mu[selected_index]),
                    "predicted_risk_std": float(risk_sigma[selected_index]),
                    "predicted_risk_upper": float(safety_margin[selected_index]),
                    "safe_candidate_count": int(np.count_nonzero(safe_mask)),
                    "candidate_count": len(candidates),
                },
            )

        selected_index = int(np.argmin(safety_margin))
        return self._start_surrogate_recipe(
            candidates[selected_index],
            phase="acquisition",
            trained_recipe_count=len(recipe_history),
            best_observed_score=best_safe_score,
            acquisition_value=float(acquisition[selected_index]),
            selected_policy="safe_gp_risk_fallback",
            decision_diagnostics={
                "risk_label": "experiment_peak_safety_risk",
                "risk_threshold": self.effective_risk_threshold,
                "risk_confidence_beta": self.risk_confidence_beta,
                "predicted_risk_mean": float(risk_mu[selected_index]),
                "predicted_risk_std": float(risk_sigma[selected_index]),
                "predicted_risk_upper": float(safety_margin[selected_index]),
                "safe_candidate_count": 0,
                "candidate_count": len(candidates),
            },
        )

    def manifest(self) -> dict[str, Any]:
        manifest = super().manifest()
        manifest.update(
            {
                "surrogate_family": "safe_gaussian_process",
                "configured_risk_threshold": self.risk_threshold,
                "risk_threshold": self.effective_risk_threshold,
                "risk_observation": "experiment_peak_safety_risk",
                "risk_confidence_beta": self.risk_confidence_beta,
                "initial_design": "public_conservative_low_intensity",
                "initial_perturbation_scale": self.initial_perturbation_scale,
            }
        )
        return manifest


class StructuredSafetyConstrainedBOAgent(SafetyConstrainedBOAgent):
    """Safety-constrained GP-EI with typed categorical material coordinates."""

    name = "structured_safe_gp_bo"

    def _model_vector(self, action: dict[str, Any]) -> np.ndarray:
        return task_recipe_to_model_vector(self.task_info, action)

    def manifest(self) -> dict[str, Any]:
        manifest = super().manifest()
        manifest.update({"recipe_encoding": "continuous_plus_material_one_hot"})
        return manifest
