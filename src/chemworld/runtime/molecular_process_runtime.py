"""Install existing steady molecular paths in the original local session."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from math import isclose, isfinite
from typing import TYPE_CHECKING, Any, cast

from chemworld.foundation import WorldState
from chemworld.foundation.balances import MaterialExchange, TransitionBalance
from chemworld.physchem.molecular_recycle import MolecularRecyclePath
from chemworld.runtime.kernel_contracts import ModelProviderResult
from chemworld.runtime.molecular_process_path import (
    MolecularProcessPath,
    advance_molecular_process_receiver,
)

if TYPE_CHECKING:
    from chemworld.world.composition import WorldCompositionSpec


@dataclass(frozen=True)
class MolecularProcessReceiverSpec:
    """Immutable equipment boundary; the original phase owns all received material."""

    path: MolecularProcessPath | MolecularRecyclePath
    phase_id: str
    maximum_amount_mol: float

    def __post_init__(self) -> None:
        if (
            not isinstance(self.path, (MolecularProcessPath, MolecularRecyclePath))
            or not isinstance(self.phase_id, str)
            or not self.phase_id
            or isinstance(self.maximum_amount_mol, bool)
            or not isfinite(self.maximum_amount_mol)
            or self.maximum_amount_mol <= 0
        ):
            raise ValueError("process receiver requires a supported path, phase and finite limit")

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_type": "molecular_process_receiver_v1",
            "path": self.path.to_dict(),
            "phase_id": self.phase_id,
            "maximum_amount_mol": self.maximum_amount_mol,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> MolecularProcessReceiverSpec:
        if payload.get("model_type") != "molecular_process_receiver_v1":
            raise ValueError("unsupported molecular receiver model")
        value = payload["path"]
        path: MolecularProcessPath | MolecularRecyclePath
        if value.get("model") == "molecular-process-path-v1":
            path = MolecularProcessPath.from_dict(value)
        elif value.get("model") == "molecular-single-recycle-v1":
            path = MolecularRecyclePath.from_dict(value)
        else:
            raise ValueError("unsupported molecular receiver path")
        return cls(path, payload["phase_id"], payload["maximum_amount_mol"])


class MolecularProcessRuntime:
    supported_operations = ("wait",)

    def __init__(self, composition: WorldCompositionSpec | None = None) -> None:
        self.composition = composition

    @property
    def models(self) -> Mapping[str, MolecularProcessReceiverSpec]:
        if self.composition is None or self.composition.physics is None:
            return {}
        return cast(
            Mapping[str, MolecularProcessReceiverSpec],
            self.composition.physics.models_of("molecular_process"),
        )

    def supports(self, state: WorldState) -> bool:
        return state.vessel_id in self.models

    def temperature_bounds_k(self, state: WorldState) -> tuple[float, float]:
        return self.models[state.vessel_id].path.domain.temperature_range_K

    def validate_state(self, state: WorldState) -> None:
        validated = self.advance(state, {"duration_s": 0.0})
        if not validated.success:
            raise ValueError("invalid process receiver: " + str(validated.diagnostics))

    def material_boundary(
        self,
        before: WorldState,
        after: WorldState,
        action: Mapping[str, Any],
    ) -> TransitionBalance:
        """Authenticate metered external feed, never infer it from inventory deficits."""
        spec = self.models[before.vessel_id]
        receipt = after.metadata.get("last_molecular_process_execution", {})
        elapsed = after.ledger.time_s - before.ledger.time_s
        if (
            action.get("operation") != "wait"
            or not 0 <= elapsed <= float(action["duration_s"]) + 1e-8
            or receipt.get("start_time_s") != before.ledger.time_s
            or not isclose(receipt.get("duration_s", -1.0), elapsed, abs_tol=1e-8)
        ):
            raise ValueError("process receipt does not identify this actual interval")
        expected = {
            k: x * spec.path.inlet.molar_rate_mol_s * elapsed
            for k, x in zip(
                spec.path.domain.model.component_ids,
                spec.path.inlet.state.mole_fractions,
                strict=True,
            )
        }
        actual = dict.fromkeys(expected, 0.0)
        seen = set()
        for row in receipt.get("exchanges", ()):
            key = row.get("inventory_key")
            if (
                key not in actual
                or key in seen
                or row.get("source") != spec.path.source_id
                or row.get("destination") != before.vessel_id
                or row.get("direction") != "in"
                or row.get("unit") != "mol"
            ):
                raise ValueError("process receipt has undeclared material endpoints")
            actual[key] = row["amount"]
            seen.add(key)
        if any(not isclose(actual[k], n, rel_tol=1e-8, abs_tol=1e-12) for k, n in expected.items()):
            raise ValueError("process receipt disagrees with its declared metered feed")
        return TransitionBalance(
            (
                MaterialExchange(
                    "molecular-process-feed",
                    f"supply:{spec.path.source_id}",
                    f"vessel:{before.vessel_id}",
                    "in",
                    tuple(sorted(actual.items())),
                ),
            )
        )

    def validate_world(self) -> None:
        if not self.models or self.composition is None or self.composition.physics is None:
            return
        physics = self.composition.physics
        if (
            physics.lab_clock is not None
            or len(physics.initial_states) != 1
            or set(self.models) != set(physics.initial_states)
            or physics.network.reactions
            or self.composition.connections
        ):
            raise ValueError("molecular process session requires one local nonreacting receiver")
        for vessel, spec in self.models.items():
            state = physics.initial_states[vessel]
            if (
                set(physics.network.species_ids) != set(spec.path.domain.model.component_ids)
                or state.phase not in {"gas", "liquid"}
                or any(s.phase != state.phase for s in physics.network.species)
                or any(
                    c.vessel == vessel and self.composition.modules.components()[c.kind].processes
                    for c in self.composition.components
                )
            ):
                raise ValueError("receiver needs its declared homogeneous nonreacting species")
            self.validate_state(state)

    def advance(
        self,
        state: WorldState,
        action: Mapping[str, Any],
        *,
        heat: bool = False,
        jacket_program: Any = None,
    ) -> ModelProviderResult:
        if heat or jacket_program is not None or "target_temperature_K" in action:
            return ModelProviderResult(
                {},
                success=False,
                outcome="rejected",
                failure_reason="unsupported_process_control",
                diagnostics={"scope": "wait consumes the declared steady path; no local heater"},
            )
        spec = self.models.get(state.vessel_id)
        if spec is None:
            return ModelProviderResult(
                {},
                success=False,
                outcome="rejected",
                failure_reason="missing_process_receiver",
                diagnostics={},
            )
        result = advance_molecular_process_receiver(
            state,
            spec.path,
            float(action["duration_s"]),
            phase_id=spec.phase_id,
            maximum_amount_mol=spec.maximum_amount_mol,
        )
        if result.success and result.state is not None:
            result = replace(
                result,
                state=result.state.replace(
                    metadata={
                        **result.state.metadata,
                        "last_molecular_process_execution": {
                            "start_time_s": state.ledger.time_s,
                            "duration_s": result.actual_duration_s,
                            "exchanges": list(result.outputs["actual_exchanges"]),
                        },
                    }
                ),
            )
        return result
