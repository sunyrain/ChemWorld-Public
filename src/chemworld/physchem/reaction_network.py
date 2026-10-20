"""General reaction-network engine for ChemWorld."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from functools import cached_property
from math import isfinite
from pathlib import Path
from typing import Any, Literal

import numpy as np
import yaml
from scipy.integrate import solve_ivp

from chemworld.physchem import reaction_network_specs as network_specs
from chemworld.physchem import reaction_rate_laws as rate_laws
from chemworld.physchem import reaction_reference_cases as reference_cases
from chemworld.physchem import reaction_sensitivity as sensitivity
from chemworld.physchem.elements import element_matrix

Arrow = network_specs.Arrow
RateLawSpec = network_specs.RateLawSpec
ReactionSpec = network_specs.ReactionSpec
SpeciesSpec = network_specs.SpeciesSpec
parse_reaction_equation = network_specs.parse_reaction_equation
reaction_from_dict = network_specs.reaction_from_dict
species_from_dict = network_specs.species_from_dict
R_J_PER_MOL_K = rate_laws.R_J_PER_MOL_K
reverse_rate_constant_from_equilibrium = rate_laws.reverse_rate_constant_from_equilibrium
effective_third_body_concentration = rate_laws.effective_third_body_concentration
falloff_reduced_pressure = rate_laws.falloff_reduced_pressure
lindemann_falloff_rate_constant = rate_laws.lindemann_falloff_rate_constant
prefixed_arrhenius_params = rate_laws.prefixed_arrhenius_params
third_body_efficiencies = rate_laws.third_body_efficiencies
troe_broadening_factor = rate_laws.troe_broadening_factor
troe_falloff_rate_constant = rate_laws.troe_falloff_rate_constant
_arrhenius_k = rate_laws.arrhenius_k
_evaluate_rate_law = rate_laws.evaluate_rate_law
_float_param = rate_laws.float_param
_reaction_order_delta = rate_laws.reaction_order_delta
AnalyticalODECase = reference_cases.AnalyticalODECase
ReactionODEReferenceCase = reference_cases.ReactionODEReferenceCase
ReactionODEReferenceResult = reference_cases.ReactionODEReferenceResult
IndependentSciPyReferenceResult = reference_cases.IndependentSciPyReferenceResult
cantera_comparable_reaction_cases = reference_cases.cantera_comparable_reaction_cases
integrate_reaction_ode_reference_case = reference_cases.integrate_reaction_ode_reference_case
evaluate_reaction_ode_reference_case = reference_cases.evaluate_reaction_ode_reference_case
evaluate_against_independent_scipy = reference_cases.evaluate_against_independent_scipy
ReactionSensitivityEntry = sensitivity.ReactionSensitivityEntry
ReactionSensitivityReport = sensitivity.ReactionSensitivityReport
finite_difference_reaction_sensitivities = sensitivity.finite_difference_reaction_sensitivities
kinetic_sensitivity_parameter_candidates = sensitivity.kinetic_sensitivity_parameter_candidates

SolverMethod = Literal["LSODA", "BDF", "Radau", "RK45", "DOP853"]


@dataclass(frozen=True)
class BatchSolverOptions:
    """Explicit numerical contract for a batch reaction-network integration."""

    method: SolverMethod = "LSODA"
    rtol: float = 1.0e-8
    atol_mol: float = 1.0e-12
    max_step_s: float | None = None
    use_jacobian: bool = False
    nonnegative_tolerance_mol: float = 1.0e-9

    def __post_init__(self) -> None:
        if self.rtol <= 0.0 or not isfinite(self.rtol):
            raise ValueError("rtol must be finite and positive")
        if self.atol_mol <= 0.0 or not isfinite(self.atol_mol):
            raise ValueError("atol_mol must be finite and positive")
        if self.max_step_s is not None and (
            self.max_step_s <= 0.0 or not isfinite(self.max_step_s)
        ):
            raise ValueError("max_step_s must be finite and positive when provided")
        if self.nonnegative_tolerance_mol < 0.0 or not isfinite(self.nonnegative_tolerance_mol):
            raise ValueError("nonnegative_tolerance_mol must be finite and nonnegative")
        if self.use_jacobian and self.method not in {"LSODA", "BDF", "Radau"}:
            raise ValueError("use_jacobian requires LSODA, BDF, or Radau")

    @property
    def stiffness_class(self) -> str:
        if self.method in {"LSODA", "BDF", "Radau"}:
            return "stiff_capable"
        return "nonstiff_explicit"

    def to_dict(self) -> dict[str, object]:
        return {
            "solver_id": f"chemworld_reaction_network_{self.method.lower()}_v2",
            "backend": "scipy.integrate.solve_ivp",
            "method": self.method,
            "rtol": self.rtol,
            "atol": self.atol_mol,
            "max_step_s": self.max_step_s,
            "use_jacobian": self.use_jacobian,
            "stiffness_class": self.stiffness_class,
            "nonnegative_tolerance_mol": self.nonnegative_tolerance_mol,
        }


@dataclass(frozen=True)
class BatchTerminationEvent:
    """Terminal crossing of a named species-amount threshold."""

    event_id: str
    species_id: str
    threshold_mol: float
    direction: Literal[-1, 0, 1] = 0

    def __post_init__(self) -> None:
        if not self.event_id.strip():
            raise ValueError("event_id cannot be empty")
        if not self.species_id.strip():
            raise ValueError("event species_id cannot be empty")
        if self.threshold_mol < 0.0 or not isfinite(self.threshold_mol):
            raise ValueError("event threshold_mol must be finite and nonnegative")


@dataclass(frozen=True)
class MechanismDiagnostic:
    """Structural and initial-state feasibility evidence for a reaction network."""

    network_id: str
    species_count: int
    reaction_count: int
    stoichiometric_rank: int
    conservation_law_dimension: int
    element_balance_residuals: dict[str, dict[str, float]]
    charge_balance_residuals: dict[str, float]
    duplicate_stoichiometric_columns: tuple[tuple[str, ...], ...]
    blocked_reactions: tuple[str, ...]
    unreachable_species: tuple[str, ...]
    violations: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not self.violations

    def to_dict(self) -> dict[str, object]:
        return {
            "network_id": self.network_id,
            "species_count": self.species_count,
            "reaction_count": self.reaction_count,
            "stoichiometric_rank": self.stoichiometric_rank,
            "conservation_law_dimension": self.conservation_law_dimension,
            "element_balance_residuals": self.element_balance_residuals,
            "charge_balance_residuals": self.charge_balance_residuals,
            "duplicate_stoichiometric_columns": [
                list(group) for group in self.duplicate_stoichiometric_columns
            ],
            "blocked_reactions": list(self.blocked_reactions),
            "unreachable_species": list(self.unreachable_species),
            "violations": list(self.violations),
            "passed": self.passed,
        }


@dataclass(frozen=True)
class ReactionNetworkSpec:
    network_id: str
    species: tuple[SpeciesSpec, ...]
    reactions: tuple[ReactionSpec, ...]
    units: dict[str, str] = field(default_factory=lambda: {"amount": "mol", "volume": "L"})
    metadata: dict[str, object] = field(default_factory=dict)
    stochastic: dict[str, object] | None = None

    def __post_init__(self) -> None:
        if not self.network_id:
            raise ValueError("network_id cannot be empty")
        species_ids = [species.species_id for species in self.species]
        reaction_ids = [reaction.reaction_id for reaction in self.reactions]
        if len(species_ids) != len(set(species_ids)):
            raise ValueError("Duplicate species_id values are not allowed")
        if len(reaction_ids) != len(set(reaction_ids)):
            raise ValueError("Duplicate reaction_id values are not allowed")
        known = set(species_ids)
        expected_units = {"amount": "mol", "volume": "L"}
        if any(self.units.get(key) != value for key, value in expected_units.items()):
            raise ValueError("reaction network units must declare amount='mol' and volume='L'")
        for reaction in self.reactions:
            referenced = (
                set(reaction.stoichiometry)
                | set(reaction.forward_orders)
                | set(reaction.reverse_orders)
            )
            missing = sorted(referenced - known)
            if missing:
                raise ValueError(
                    f"Reaction {reaction.reaction_id} references unknown species: {missing}"
                )
        self.check_conservation(raise_on_error=True)
        if self.stochastic is not None:
            from chemworld.physchem.stochastic_kinetics import StochasticSpec, compile_channels

            spec = StochasticSpec(**self.stochastic)
            object.__setattr__(self, "stochastic", spec.to_dict())
            compile_channels(self, spec)

    @cached_property
    def stochastic_channels(self):
        from chemworld.physchem.stochastic_kinetics import StochasticSpec, compile_channels

        if self.stochastic is None:
            raise ValueError("network has no declared stochastic strategy")
        return compile_channels(self, StochasticSpec(**self.stochastic))

    @property
    def species_ids(self) -> tuple[str, ...]:
        return tuple(species.species_id for species in self.species)

    @property
    def reaction_ids(self) -> tuple[str, ...]:
        return tuple(reaction.reaction_id for reaction in self.reactions)

    @property
    def species_index(self) -> dict[str, int]:
        return {species_id: idx for idx, species_id in enumerate(self.species_ids)}

    def stoichiometric_matrix(self) -> tuple[tuple[float, ...], ...]:
        index = self.species_index
        matrix = [[0.0 for _ in self.reactions] for _ in self.species]
        for reaction_idx, reaction in enumerate(self.reactions):
            for species_id, coefficient in reaction.stoichiometry.items():
                matrix[index[species_id]][reaction_idx] = coefficient
        return tuple(tuple(row) for row in matrix)

    def element_matrix(self) -> tuple[tuple[tuple[float, ...], ...], tuple[str, ...]]:
        return element_matrix([species.composition for species in self.species])

    def element_balance_residuals(self) -> dict[str, dict[str, float]]:
        matrix, element_order = self.element_matrix()
        stoich = self.stoichiometric_matrix()
        residuals: dict[str, dict[str, float]] = {}
        for reaction_idx, reaction in enumerate(self.reactions):
            reaction_residuals = {}
            for element_idx, element in enumerate(element_order):
                residual = sum(
                    matrix[species_idx][element_idx] * stoich[species_idx][reaction_idx]
                    for species_idx in range(len(self.species))
                )
                if abs(residual) > 1e-12:
                    reaction_residuals[element] = residual
            residuals[reaction.reaction_id] = reaction_residuals
        return residuals

    def check_element_balance(self, *, raise_on_error: bool = False) -> bool:
        residuals = self.element_balance_residuals()
        passed = all(not reaction_residuals for reaction_residuals in residuals.values())
        if raise_on_error and not passed:
            raise ValueError(f"Reaction network is not element balanced: {residuals}")
        return passed

    def charge_balance_residuals(self) -> dict[str, float]:
        stoich = self.stoichiometric_matrix()
        charges = [float(species.charge) for species in self.species]
        residuals: dict[str, float] = {}
        for reaction_idx, reaction in enumerate(self.reactions):
            residual = sum(
                charges[species_idx] * stoich[species_idx][reaction_idx]
                for species_idx in range(len(self.species))
            )
            if abs(residual) > 1.0e-12:
                residuals[reaction.reaction_id] = residual
        return residuals

    def check_conservation(self, *, raise_on_error: bool = False) -> bool:
        element_residuals = self.element_balance_residuals()
        charge_residuals = self.charge_balance_residuals()
        passed = (
            all(not residuals for residuals in element_residuals.values()) and not charge_residuals
        )
        if raise_on_error and not passed:
            raise ValueError(
                "Reaction network is not element balanced or charge balanced: "
                f"elements={element_residuals}, charge={charge_residuals}"
            )
        return passed

    def diagnose_mechanism(
        self,
        initial_amounts_mol: Mapping[str, float] | None = None,
    ) -> MechanismDiagnostic:
        matrix = np.asarray(self.stoichiometric_matrix(), dtype=float)
        rank = int(np.linalg.matrix_rank(matrix)) if matrix.size else 0
        columns: dict[tuple[float, ...], list[str]] = {}
        for reaction_index, reaction_id in enumerate(self.reaction_ids):
            column = tuple(float(value) for value in matrix[:, reaction_index])
            columns.setdefault(column, []).append(reaction_id)
        duplicate_columns = tuple(tuple(ids) for ids in columns.values() if len(ids) > 1)

        blocked: tuple[str, ...] = ()
        unreachable: tuple[str, ...] = ()
        if initial_amounts_mol is not None:
            unknown = sorted(set(initial_amounts_mol) - set(self.species_ids))
            if unknown:
                raise ValueError(f"initial state references unknown species: {unknown}")
            available = {
                species_id
                for species_id in self.species_ids
                if float(initial_amounts_mol.get(species_id, 0.0)) > 0.0
            }
            changed = True
            executable: set[str] = set()
            while changed:
                changed = False
                for reaction in self.reactions:
                    if reaction.reaction_id in executable:
                        continue
                    required = {
                        species_id
                        for species_id, order in reaction.kinetic_forward_orders.items()
                        if order > 0.0
                    }
                    if required.issubset(available):
                        executable.add(reaction.reaction_id)
                        before = len(available)
                        available.update(reaction.products)
                        changed = changed or len(available) != before
            blocked = tuple(
                reaction_id for reaction_id in self.reaction_ids if reaction_id not in executable
            )
            unreachable = tuple(
                species_id for species_id in self.species_ids if species_id not in available
            )

        element_residuals = self.element_balance_residuals()
        charge_residuals = self.charge_balance_residuals()
        violations: list[str] = []
        if any(element_residuals.values()):
            violations.append("element_balance")
        if charge_residuals:
            violations.append("charge_balance")
        if any(np.all(np.abs(matrix[:, index]) <= 1.0e-15) for index in range(matrix.shape[1])):
            violations.append("zero_stoichiometric_column")
        if initial_amounts_mol is not None and blocked:
            violations.append("initial_state_blocked_reactions")
        return MechanismDiagnostic(
            network_id=self.network_id,
            species_count=len(self.species),
            reaction_count=len(self.reactions),
            stoichiometric_rank=rank,
            conservation_law_dimension=len(self.species) - rank,
            element_balance_residuals=element_residuals,
            charge_balance_residuals=charge_residuals,
            duplicate_stoichiometric_columns=duplicate_columns,
            blocked_reactions=blocked,
            unreachable_species=unreachable,
            violations=tuple(violations),
        )

    def reaction_rates(
        self,
        amounts_mol: Mapping[str, float],
        *,
        volume_L: float,
        temperature_K: float,
        species_thermo: Mapping[str, Any] | None = None,
        activity_coefficients: Mapping[str, float] | None = None,
    ) -> dict[str, float]:
        concentrations = self._concentrations(amounts_mol, volume_L=volume_L)
        if species_thermo is None and "gas_thermochemistry" in self.metadata:
            species_thermo = self.imported_gas_thermochemistry
        return {
            reaction.reaction_id: evaluate_rate_law(
                reaction,
                concentrations_mol_L=concentrations,
                temperature_K=temperature_K,
                species_thermo=species_thermo,
                activity_coefficients=activity_coefficients,
            )
            for reaction in self.reactions
        }

    @cached_property
    def imported_gas_thermochemistry(self) -> dict[str, Any]:
        """Rebuild declared NASA7 data once, including after native JSON replay."""
        from chemworld.physchem.thermochemistry import NASA7SpeciesThermo

        return {
            key: NASA7SpeciesThermo.from_cantera_yaml_thermo(species_id=key, **node)
            for key, node in self.metadata.get("gas_thermochemistry", {}).items()
        }

    def amount_derivatives(
        self,
        amounts_mol: Mapping[str, float],
        *,
        volume_L: float,
        temperature_K: float,
        species_thermo: Mapping[str, Any] | None = None,
        activity_coefficients: Mapping[str, float] | None = None,
    ) -> dict[str, float]:
        rates = self.reaction_rates(
            amounts_mol,
            volume_L=volume_L,
            temperature_K=temperature_K,
            species_thermo=species_thermo,
            activity_coefficients=activity_coefficients,
        )
        return self.amount_derivatives_from_rates(rates, volume_L=volume_L)

    def amount_derivatives_from_rates(
        self,
        rates: Mapping[str, float],
        *,
        volume_L: float,
    ) -> dict[str, float]:
        """Apply the same stoichiometry to already evaluated progress rates."""
        derivatives = dict.fromkeys(self.species_ids, 0.0)
        for reaction in self.reactions:
            rate_mol_L_s = rates[reaction.reaction_id]
            for species_id, coefficient in reaction.stoichiometry.items():
                derivatives[species_id] += coefficient * rate_mol_L_s * volume_L
        return derivatives

    def integrate_batch(
        self,
        initial_amounts_mol: Mapping[str, float],
        *,
        volume_L: float,
        temperature_K: float,
        duration_s: float,
        evaluation_times_s: Sequence[float] | None = None,
        species_thermo: Mapping[str, Any] | None = None,
        activity_coefficients: Mapping[str, float] | None = None,
        solver_options: BatchSolverOptions | None = None,
        termination_events: Sequence[BatchTerminationEvent] = (),
    ) -> BatchIntegrationResult:
        if self.stochastic is not None:
            raise ValueError(
                "discrete network requires advance_stochastic, not implicit ODE switching"
            )
        if duration_s < 0 or not isfinite(duration_s):
            raise ValueError("duration_s cannot be negative")
        if volume_L <= 0 or not isfinite(volume_L):
            raise ValueError("volume_L must be finite and positive")
        if temperature_K <= 0 or not isfinite(temperature_K):
            raise ValueError("temperature_K must be finite and positive")
        unknown = sorted(set(initial_amounts_mol) - set(self.species_ids))
        if unknown:
            raise ValueError(f"initial state references unknown species: {unknown}")
        initial_values = [
            float(initial_amounts_mol.get(species_id, 0.0)) for species_id in self.species_ids
        ]
        if any(value < 0.0 or not isfinite(value) for value in initial_values):
            raise ValueError("initial species amounts must be finite and nonnegative")
        y0 = np.asarray(initial_values, dtype=float)
        options = BatchSolverOptions() if solver_options is None else solver_options
        event_ids = [event.event_id for event in termination_events]
        if len(event_ids) != len(set(event_ids)):
            raise ValueError("termination event ids cannot contain duplicates")
        for event in termination_events:
            if event.species_id not in self.species_ids:
                raise ValueError(f"event references unknown species: {event.species_id}")

        def rhs(_time_s: float, y: np.ndarray) -> np.ndarray:
            amounts = {
                species_id: max(float(value), 0.0)
                for species_id, value in zip(self.species_ids, y, strict=True)
            }
            derivatives = self.amount_derivatives(
                amounts,
                volume_L=volume_L,
                temperature_K=temperature_K,
                species_thermo=species_thermo,
                activity_coefficients=activity_coefficients,
            )
            values = np.array([derivatives[species_id] for species_id in self.species_ids])
            # Project outward derivatives at the nonnegative boundary.  This is
            # a safeguard for zero-order/custom-order laws, not post-hoc clipping.
            values[(y <= 0.0) & (values < 0.0)] = 0.0
            return values

        def jacobian(time_s: float, y: np.ndarray) -> np.ndarray:
            baseline = rhs(time_s, y)
            matrix = np.empty((len(y), len(y)), dtype=float)
            for column in range(len(y)):
                step = 1.0e-7 * max(1.0, abs(float(y[column])))
                shifted = y.copy()
                shifted[column] += step
                matrix[:, column] = (rhs(time_s, shifted) - baseline) / step
            return matrix

        if evaluation_times_s is None:
            t_eval = None
        else:
            t_eval = np.array(tuple(evaluation_times_s), dtype=float)
            if (
                t_eval.ndim != 1
                or not np.all(np.isfinite(t_eval))
                or np.any(t_eval < 0.0)
                or np.any(t_eval > duration_s)
                or np.any(np.diff(t_eval) < 0.0)
            ):
                raise ValueError("evaluation_times_s must be finite, sorted, and within duration_s")
        if duration_s == 0.0:
            times = np.array((0.0,), dtype=float)
            values = y0[:, None]
            diagnostic: dict[str, object] = {
                "policy": options.to_dict(),
                "success": True,
                "message": "zero-duration initial state",
                "status": 0,
                "nfev": 0,
                "njev": 0,
                "nlu": 0,
                "t_start_s": 0.0,
                "t_end_s": 0.0,
                "evaluation_count": 1,
                "event_count": 0,
                "triggered_events": [],
                "final_time_s": 0.0,
            }
        else:
            scipy_events = []
            species_index = self.species_index
            for event_spec in termination_events:
                index = species_index[event_spec.species_id]

                def threshold_event(
                    _time_s: float,
                    y: np.ndarray,
                    *,
                    _index: int = index,
                    _threshold: float = event_spec.threshold_mol,
                ) -> float:
                    return float(y[_index]) - _threshold

                threshold_event.terminal = True  # type: ignore[attr-defined]
                threshold_event.direction = event_spec.direction  # type: ignore[attr-defined]
                scipy_events.append(threshold_event)

            kwargs: dict[str, object] = {
                "method": options.method,
                "rtol": options.rtol,
                "atol": options.atol_mol,
                "dense_output": bool(scipy_events),
            }
            if options.max_step_s is not None:
                kwargs["max_step"] = options.max_step_s
            if t_eval is not None and len(t_eval):
                kwargs["t_eval"] = t_eval
            if scipy_events:
                kwargs["events"] = tuple(scipy_events)
            if options.use_jacobian:
                kwargs["jac"] = jacobian
            result = solve_ivp(rhs, (0.0, duration_s), y0, **kwargs)
            if not result.success:
                raise RuntimeError(f"Reaction-network integration failed: {result.message}")
            times = np.asarray(result.t, dtype=float)
            values = np.asarray(result.y, dtype=float)
            triggered_events: list[str] = []
            event_times: list[float] = []
            result_event_times = getattr(result, "t_events", None) or ()
            for event_spec, crossings in zip(
                termination_events,
                result_event_times,
                strict=True,
            ):
                if len(crossings):
                    triggered_events.append(event_spec.event_id)
                    event_times.extend(float(value) for value in crossings)
            if event_times:
                final_event_time = min(event_times)
                if not len(times) or abs(float(times[-1]) - final_event_time) > 1.0e-12:
                    if result.sol is None:
                        raise RuntimeError("terminal event state is unavailable")
                    times = np.append(times, final_event_time)
                    values = np.column_stack((values, result.sol(final_event_time)))
            diagnostic = {
                "policy": options.to_dict(),
                "success": True,
                "message": str(result.message),
                "status": int(result.status),
                "nfev": int(result.nfev),
                "njev": None if result.njev is None else int(result.njev),
                "nlu": None if result.nlu is None else int(result.nlu),
                "t_start_s": 0.0,
                "t_end_s": duration_s,
                "evaluation_count": len(times),
                "event_count": len(triggered_events),
                "triggered_events": triggered_events,
                "final_time_s": float(times[-1]),
            }

        minimum_raw = float(np.min(values))
        nonnegative_passed = minimum_raw >= -options.nonnegative_tolerance_mol
        diagnostic["minimum_raw_amount_mol"] = minimum_raw
        diagnostic["nonnegative_passed"] = nonnegative_passed
        diagnostic["jacobian_used"] = options.use_jacobian
        diagnostic["mechanism"] = self.diagnose_mechanism(initial_amounts_mol).to_dict()
        if not nonnegative_passed:
            raise RuntimeError(
                "Reaction-network integration violated nonnegativity: "
                f"minimum={minimum_raw:.6g} mol"
            )
        clipped_values = np.maximum(values, 0.0)
        diagnostic["maximum_conservation_drift_mol"] = self._maximum_invariant_drift(clipped_values)
        final = {
            species_id: float(clipped_values[idx, -1])
            for idx, species_id in enumerate(self.species_ids)
        }
        return BatchIntegrationResult(
            network_id=self.network_id,
            species_ids=self.species_ids,
            times_s=tuple(float(value) for value in times),
            amounts_mol=tuple(
                tuple(float(value) for value in clipped_values[idx])
                for idx in range(len(self.species_ids))
            ),
            final_amounts_mol=final,
            solver_diagnostic=diagnostic,
        )

    def _maximum_invariant_drift(self, trajectory: np.ndarray) -> float:
        stoich = np.asarray(self.stoichiometric_matrix(), dtype=float)
        if not stoich.size or trajectory.shape[1] <= 1:
            return 0.0
        left_vectors, singular_values, _right = np.linalg.svd(stoich, full_matrices=True)
        tolerance = (
            max(stoich.shape)
            * np.finfo(float).eps
            * (float(singular_values[0]) if singular_values.size else 0.0)
        )
        rank = int(np.sum(singular_values > tolerance))
        invariants = left_vectors[:, rank:].T
        if not invariants.size:
            return 0.0
        inventory = invariants @ trajectory
        return float(np.max(np.abs(inventory - inventory[:, [0]])))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ReactionNetworkSpec:
        species = tuple(species_from_dict(item) for item in payload["species"])
        reactions = tuple(reaction_from_dict(item) for item in payload["reactions"])
        return cls(
            network_id=str(payload["network_id"]),
            species=species,
            reactions=reactions,
            units=dict(payload.get("units", {"amount": "mol", "volume": "L"})),
            metadata=dict(payload.get("metadata", {})),
            stochastic=payload.get("stochastic"),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "network_id": self.network_id,
            **({"stochastic": dict(self.stochastic)} if self.stochastic is not None else {}),
            "species": [species.to_dict() for species in self.species],
            "reactions": [reaction.to_dict() for reaction in self.reactions],
            "stoichiometric_matrix": [list(row) for row in self.stoichiometric_matrix()],
            "element_balance_residuals": self.element_balance_residuals(),
            "charge_balance_residuals": self.charge_balance_residuals(),
            "mechanism_diagnostic": self.diagnose_mechanism().to_dict(),
            "units": dict(self.units),
            "metadata": dict(self.metadata),
        }

    def _concentrations(
        self,
        amounts_mol: Mapping[str, float],
        *,
        volume_L: float,
    ) -> dict[str, float]:
        if volume_L <= 0:
            raise ValueError("volume_L must be positive")
        concentrations = {}
        for species_id in self.species_ids:
            amount = float(amounts_mol.get(species_id, 0.0))
            if amount < -1e-15:
                raise ValueError(f"Species amount cannot be negative: {species_id}={amount}")
            concentrations[species_id] = max(amount, 0.0) / volume_L
        return concentrations


@dataclass(frozen=True)
class BatchIntegrationResult:
    network_id: str
    species_ids: tuple[str, ...]
    times_s: tuple[float, ...]
    amounts_mol: tuple[tuple[float, ...], ...]
    final_amounts_mol: dict[str, float]
    solver_diagnostic: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "network_id": self.network_id,
            "species_ids": list(self.species_ids),
            "times_s": list(self.times_s),
            "amounts_mol": [list(row) for row in self.amounts_mol],
            "final_amounts_mol": dict(self.final_amounts_mol),
            "solver_diagnostic": dict(self.solver_diagnostic),
        }


@dataclass(frozen=True)
class ThermochemicalDetailedBalanceResult:
    """Forward and reverse rate constants linked by reaction thermochemistry."""

    reaction_id: str
    temperature_K: float
    forward_rate_constant: float
    reverse_rate_constant: float
    concentration_equilibrium_constant: float
    dimensionless_equilibrium_constant: float
    delta_g_J_mol: float
    reaction_order_delta: float
    standard_concentration_mol_L: float
    source: str = "nasa7"

    def to_dict(self) -> dict[str, object]:
        return {
            "reaction_id": self.reaction_id,
            "temperature_K": self.temperature_K,
            "forward_rate_constant": self.forward_rate_constant,
            "reverse_rate_constant": self.reverse_rate_constant,
            "concentration_equilibrium_constant": (self.concentration_equilibrium_constant),
            "dimensionless_equilibrium_constant": (self.dimensionless_equilibrium_constant),
            "delta_g_J_mol": self.delta_g_J_mol,
            "reaction_order_delta": self.reaction_order_delta,
            "standard_concentration_mol_L": self.standard_concentration_mol_L,
            "source": self.source,
        }


def thermochemical_detailed_balance(
    reaction: ReactionSpec,
    *,
    species_thermo: Mapping[str, Any],
    temperature_K: float,
    standard_concentration_mol_L: float = 1.0,
) -> ThermochemicalDetailedBalanceResult:
    """Compute reverse rate constant from NASA7 reaction thermochemistry.

    The concentration equilibrium constant is consistent with ChemWorld's
    mass-action rate powers:

    ``K_c = K_dimensionless * C0 ** sum(nu_i)``.
    """

    if reaction.rate_law.equation_id != "reversible_arrhenius":
        raise ValueError("thermochemical detailed balance requires reversible_arrhenius")
    if (
        reaction.kinetic_forward_orders != reaction.reactants
        or reaction.kinetic_reverse_orders != reaction.products
    ):
        raise ValueError(
            "thermochemical detailed balance requires kinetic orders to match "
            "stoichiometric reactant/product coefficients"
        )
    forward_rate_constant = _arrhenius_k(reaction.rate_law.parameters, temperature_K)
    concentration_equilibrium_constant, dimensionless_equilibrium_constant, delta_g = (
        thermochemical_concentration_equilibrium_constant(
            reaction,
            species_thermo=species_thermo,
            temperature_K=temperature_K,
            standard_concentration_mol_L=standard_concentration_mol_L,
        )
    )
    return ThermochemicalDetailedBalanceResult(
        reaction_id=reaction.reaction_id,
        temperature_K=temperature_K,
        forward_rate_constant=forward_rate_constant,
        reverse_rate_constant=reverse_rate_constant_from_equilibrium(
            forward_rate_constant=forward_rate_constant,
            concentration_equilibrium_constant=(
                dimensionless_equilibrium_constant
                if reaction.rate_law.uses_activities
                else concentration_equilibrium_constant
            ),
        ),
        concentration_equilibrium_constant=concentration_equilibrium_constant,
        dimensionless_equilibrium_constant=dimensionless_equilibrium_constant,
        delta_g_J_mol=delta_g,
        reaction_order_delta=_reaction_order_delta(reaction),
        standard_concentration_mol_L=standard_concentration_mol_L,
    )


def thermochemical_concentration_equilibrium_constant(
    reaction: ReactionSpec,
    *,
    species_thermo: Mapping[str, Any],
    temperature_K: float,
    standard_concentration_mol_L: float = 1.0,
) -> tuple[float, float, float]:
    """Return ``(K_c, K_dimensionless, Delta G)`` from species thermo."""

    if temperature_K <= 0:
        raise ValueError("temperature_K must be positive")
    if standard_concentration_mol_L <= 0 or not isfinite(standard_concentration_mol_L):
        raise ValueError("standard_concentration_mol_L must be finite and positive")
    from chemworld.physchem.thermochemistry import reaction_thermochemistry

    thermo_result = reaction_thermochemistry(
        reaction_id=reaction.reaction_id,
        stoichiometry=reaction.stoichiometry,
        species_thermo=species_thermo,
        temperature_K=temperature_K,
    )
    concentration_equilibrium_constant = (
        thermo_result.equilibrium_constant
        * standard_concentration_mol_L ** _reaction_order_delta(reaction)
    )
    if concentration_equilibrium_constant <= 0 or not isfinite(concentration_equilibrium_constant):
        raise ValueError("thermochemical concentration equilibrium constant is invalid")
    return (
        concentration_equilibrium_constant,
        thermo_result.equilibrium_constant,
        thermo_result.delta_g_J_mol,
    )


def evaluate_rate_law(
    reaction: ReactionSpec,
    *,
    concentrations_mol_L: Mapping[str, float],
    temperature_K: float,
    species_thermo: Mapping[str, Any] | None = None,
    activity_coefficients: Mapping[str, float] | None = None,
) -> float:
    return _evaluate_rate_law(
        reaction,
        concentrations_mol_L=concentrations_mol_L,
        temperature_K=temperature_K,
        species_thermo=species_thermo,
        activity_coefficients=activity_coefficients,
        thermochemical_reverse_rate_constant=_thermochemical_reverse_rate_constant,
    )


def _thermochemical_reverse_rate_constant(
    reaction: ReactionSpec,
    params: Mapping[str, object],
    temperature_K: float,
    species_thermo: Mapping[str, Any],
) -> float:
    standard_concentration = _float_param(
        params,
        "standard_concentration_mol_L",
        default=1.0,
    )
    return thermochemical_detailed_balance(
        reaction,
        species_thermo=species_thermo,
        temperature_K=temperature_K,
        standard_concentration_mol_L=standard_concentration,
    ).reverse_rate_constant


def load_mechanism(path: str | Path) -> ReactionNetworkSpec:
    source = Path(path)
    text = source.read_text(encoding="utf-8")
    if source.suffix.lower() == ".json":
        payload = json.loads(text)
    elif source.suffix.lower() in {".yaml", ".yml"}:
        payload = yaml.safe_load(text)
    else:
        raise ValueError(f"Unsupported mechanism file extension: {source.suffix}")
    if not isinstance(payload, dict):
        raise ValueError("Mechanism file must contain a mapping")
    return ReactionNetworkSpec.from_dict(payload)


def perturb_network_parameters(
    network: ReactionNetworkSpec,
    *,
    seed: int,
    relative_std: float = 0.05,
) -> ReactionNetworkSpec:
    if relative_std < 0:
        raise ValueError("relative_std cannot be negative")
    rng = np.random.default_rng(seed)
    reactions = []
    for reaction in network.reactions:
        params = dict(reaction.rate_law.parameters)
        if "A" in params and isinstance(params["A"], int | float):
            factor = float(np.exp(rng.normal(0.0, relative_std)))
            params["A"] = float(params["A"]) * factor
        elif "k" in params and isinstance(params["k"], int | float):
            factor = float(np.exp(rng.normal(0.0, relative_std)))
            params["k"] = float(params["k"]) * factor
        reactions.append(
            replace(
                reaction,
                rate_law=replace(reaction.rate_law, parameters=params),
            )
        )
    return replace(
        network,
        reactions=tuple(reactions),
        metadata={**network.metadata, "parameter_perturbation_seed": seed},
    )
