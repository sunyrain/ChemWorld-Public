"""Provider-neutral physical-model result and interface contracts."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite
from typing import Any, Protocol

from chemworld.foundation import WorldState
from chemworld.foundation.outcomes import OPERATION_OUTCOMES
from chemworld.physchem.maturity import ModelProviderContract


@dataclass(frozen=True)
class ModelProviderResult:
    """Provider-neutral result used by independently implemented physical models."""

    outputs: Mapping[str, Any]
    diagnostics: Mapping[str, Any]
    warnings: tuple[str, ...] = ()
    success: bool = True
    failure_reason: str | None = None
    provenance: tuple[str, ...] = ()
    outcome: str | None = None
    state: WorldState | None = None
    actual_duration_s: float = 0.0
    measurement_state: WorldState | None = None
    # Optional local heat increments for a grouped physical advance. The group's
    # resource ledger is aggregate; it must not overwrite an anchor's local heat.
    vessel_thermal_deltas: Mapping[str, Mapping[str, float]] | None = None

    def __post_init__(self) -> None:
        outcome = self.outcome or ("success" if self.success else "numerical_failure")
        if outcome not in OPERATION_OUTCOMES:
            raise ValueError(f"unknown operation outcome: {outcome}")
        object.__setattr__(self, "outcome", outcome)
        if not isfinite(self.actual_duration_s) or self.actual_duration_s < 0.0:
            raise ValueError("actual_duration_s must be finite and nonnegative")
        if outcome in {"rejected", "numerical_failure"} and self.actual_duration_s != 0.0:
            raise ValueError("unexecuted provider result cannot report elapsed physical time")
        if self.success != (outcome in {"success", "stopped"}):
            raise ValueError("provider success flag and outcome disagree")
        if self.success and self.failure_reason is not None:
            raise ValueError("successful provider result cannot include failure_reason")
        if not self.success and not str(self.failure_reason or "").strip():
            raise ValueError("failed provider result requires failure_reason")
        if self.vessel_thermal_deltas is not None:
            fields = {"energy_jacket_J", "heat_reaction_J", "heat_loss_J"}
            if not self.success or any(
                not isinstance(vessel, str) or not vessel or set(values) - fields
                or any(isinstance(v, bool) or not isfinite(v) for v in values.values())
                for vessel, values in self.vessel_thermal_deltas.items()
            ):
                raise ValueError("local thermal deltas require successful finite declared changes")
            object.__setattr__(self, "vessel_thermal_deltas", {
                vessel: dict(values) for vessel, values in self.vessel_thermal_deltas.items()
            })

    def to_dict(self) -> dict[str, Any]:
        return {
            "outputs": dict(self.outputs),
            "diagnostics": dict(self.diagnostics),
            "warnings": list(self.warnings),
            "success": self.success,
            "failure_reason": self.failure_reason,
            "provenance": list(self.provenance),
            "outcome": self.outcome,
            "actual_duration_s": self.actual_duration_s,
            **({"vessel_thermal_deltas": {
                vessel: dict(values) for vessel, values in self.vessel_thermal_deltas.items()
            }} if self.vessel_thermal_deltas is not None else {}),
        }


class PhysicalModelProvider(Protocol):
    """Minimum contract that parallel model teams implement before runtime wiring."""

    @property
    def model_contract(self) -> ModelProviderContract:
        ...

    def validate_domain(self, inputs: Mapping[str, Any]) -> tuple[str, ...]:
        """Return domain violations; an empty tuple means the input is supported."""
        ...

    def evaluate(self, inputs: Mapping[str, Any]) -> ModelProviderResult:
        ...


@dataclass(frozen=True)
class ContractModelProviderStub:
    """Deterministic contract fixture for model teams before runtime integration.

    The stub deliberately performs no physical calculation. It verifies the
    provider result shape and makes that limitation visible in every result.
    """

    model_contract: ModelProviderContract
    outputs: Mapping[str, Any]
    diagnostics: Mapping[str, Any]
    warnings: tuple[str, ...] = (
        "contract stub only; no physical model evaluation was performed",
    )

    def __post_init__(self) -> None:
        self._require_exact_fields(
            label="outputs",
            expected=self.model_contract.output_fields,
            actual=self.outputs,
        )
        self._require_exact_fields(
            label="diagnostics",
            expected=self.model_contract.diagnostic_fields,
            actual=self.diagnostics,
        )
        object.__setattr__(self, "outputs", dict(self.outputs))
        object.__setattr__(self, "diagnostics", dict(self.diagnostics))

    @staticmethod
    def _require_exact_fields(
        *,
        label: str,
        expected: tuple[str, ...],
        actual: Mapping[str, Any],
    ) -> None:
        missing = sorted(set(expected) - set(actual))
        unexpected = sorted(set(actual) - set(expected))
        if missing or unexpected:
            raise ValueError(
                f"stub {label} do not match provider contract: "
                f"missing={missing}, unexpected={unexpected}"
            )

    def validate_domain(self, inputs: Mapping[str, Any]) -> tuple[str, ...]:
        return tuple(
            f"missing required input: {field_name}"
            for field_name in self.model_contract.input_fields
            if field_name not in inputs
        )

    def evaluate(self, inputs: Mapping[str, Any]) -> ModelProviderResult:
        violations = self.validate_domain(inputs)
        if violations:
            return ModelProviderResult(
                outputs={},
                diagnostics=dict.fromkeys(
                    self.model_contract.diagnostic_fields,
                    None,
                ),
                warnings=self.warnings,
                success=False,
                failure_reason="; ".join(violations),
                provenance=(*self.model_contract.provenance, "contract-stub"),
            )
        return ModelProviderResult(
            outputs=dict(self.outputs),
            diagnostics=dict(self.diagnostics),
            warnings=self.warnings,
            provenance=(*self.model_contract.provenance, "contract-stub"),
        )
