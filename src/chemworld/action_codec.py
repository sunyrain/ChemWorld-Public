"""Action abstraction layer for ChemWorld event and Gym-style actions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from chemworld.foundation.samples import CONTAINER_IDS
from chemworld.world.actions import CATALYSTS, ELECTROLYTE_PROFILES, SOLVENTS
from chemworld.world.control_contract import (
    CONTROL_ACTION_FIELDS,
    CONTROL_CHOICES,
    CONTROL_NUMERIC,
    CONTROL_VECTOR_FIELDS,
)
from chemworld.world.operations import (
    CAMPAIGN_CONTROL_OPERATIONS,
    INSTRUMENTS,
    OPERATION_TYPES,
    instrument_name,
    operation_contracts,
)
from chemworld.world.spectral_contract import SPECTRAL_BOUNDS, SPECTRAL_FIELDS
from chemworld.world.stream_contract import STREAM_FIELDS

PHASES = ("reactor_liquid", "aqueous", "organic")
EXTRACTANTS = SOLVENTS
FEED_IDS = tuple(f"feed-{i}" for i in range(16))
GYM_ACTION_KEYS = (
    "operation",
    "amount_mol",
    "volume_L",
    "catalyst_amount_mol",
    "target_temperature_K",
    "duration_s",
    "stirring_speed_rpm",
    "sample_volume_L",
    "instrument",
    "catalyst",
    "solvent",
    "phase",
    "target_phase",
    "extractant",
    "wash_volume_L",
    "transfer_fraction",
    "seed_mass_g",
    "reflux_ratio",
    "flow_rate_mL_min",
    "residence_time_s",
    "potential_V",
    "current_mA",
    "electrolyte_profile",
    "container",
    "source_container",
    "destination_container",
    "capacity_L",
    "mixing",
    "component",
    "vessel",
    "connection",
    *CONTROL_VECTOR_FIELDS,
    *SPECTRAL_FIELDS,
)


@dataclass(frozen=True)
class ActionCodec:
    """Convert between semantic event actions and numeric Gym-style vectors."""

    operation_types: tuple[str, ...] = OPERATION_TYPES
    instruments: tuple[str, ...] = INSTRUMENTS
    catalysts: tuple[str, ...] = CATALYSTS
    solvents: tuple[str, ...] = SOLVENTS
    phases: tuple[str, ...] = PHASES
    extractants: tuple[str, ...] = EXTRACTANTS
    electrolyte_profiles: tuple[str, ...] = ELECTROLYTE_PROFILES
    vessels: tuple[str, ...] = tuple(f"vessel-{i}" for i in range(16))
    connections: tuple[str, ...] = tuple(f"connection-{i}" for i in range(64))

    def canonicalize(self, action: dict[str, Any]) -> dict[str, Any]:
        """Normalize event-action JSON into canonical names and flat payload."""

        if "operation" not in action:
            raise ValueError("Event actions must include an operation field")
        canonical = dict(action)
        if isinstance(canonical.get("payload"), dict):
            payload = dict(canonical.pop("payload"))
            payload["operation"] = canonical["operation"]
            payload.update({key: value for key, value in canonical.items() if key != "operation"})
            canonical = payload
        canonical["operation"] = self._operation_name(canonical["operation"])
        canonical = self._apply_aliases(canonical)
        for field, choices in CONTROL_CHOICES.items():
            if field in canonical:
                canonical[field] = self._index(canonical[field], choices)
        control_numeric_fields = set(CONTROL_NUMERIC) | (
            set(CONTROL_ACTION_FIELDS.get(canonical["operation"], ())) - set(CONTROL_CHOICES)
        )
        control_numeric_fields |= set(STREAM_FIELDS.get(canonical["operation"], ())) - {
            "connection"
        }
        control_numeric_fields |= set(SPECTRAL_FIELDS)
        for field in control_numeric_fields:
            if field in canonical:
                if np.asarray(canonical[field]).dtype.kind == "b":
                    raise ValueError(f"{field} must be numeric, not boolean")
                canonical[field] = self._float(canonical, field, 0)
                low, high = CONTROL_NUMERIC[field][:2] if field in CONTROL_NUMERIC else (0.0, 0.0)
                for boundary in (low, high):
                    if abs(canonical[field] - boundary) <= abs(boundary) * 1e-7:
                        canonical[field] = boundary
        if "component" in canonical:
            canonical["component"] = self._index(canonical["component"], FEED_IDS)
        for field, choices in (("vessel", self.vessels), ("connection", self.connections)):
            if field in canonical:
                canonical[field] = self._index(canonical[field], choices)
        for field in ("container", "source_container", "destination_container"):
            if field in canonical:
                canonical[field] = self._index(canonical[field], CONTAINER_IDS)
        if canonical["operation"] in {"create_container", "transfer_material", "route_material"}:
            for field in ("capacity_L", "transfer_fraction"):
                if field in canonical:
                    canonical[field] = self._float(canonical, field, 0)
            if "mixing" in canonical:
                canonical["mixing"] = self._index(canonical["mixing"], ("empty_only", "allow"))
        if "solvent" in canonical:
            canonical["solvent"] = self._choice_index(canonical["solvent"], self.solvents)
        if "catalyst" in canonical:
            canonical["catalyst"] = self._choice_index(
                canonical["catalyst"],
                self.catalysts,
            )
        if "instrument" in canonical:
            canonical["instrument"] = instrument_name(canonical["instrument"])
        if "phase" in canonical:
            canonical["phase"] = self.phase_name(canonical["phase"])
        if "target_phase" in canonical:
            canonical["target_phase"] = self.phase_name(canonical["target_phase"])
        if "extractant" in canonical:
            canonical["extractant"] = self.extractant_name(canonical["extractant"])
        if "electrolyte_profile" in canonical:
            canonical["electrolyte_profile"] = self._choice_index(
                canonical["electrolyte_profile"],
                self.electrolyte_profiles,
            )
        return canonical

    def _operation_name(self, value: Any) -> str:
        if isinstance(value, str):
            if value not in self.operation_types:
                raise ValueError(f"Unsupported operation: {value}")
            return value
        index = self._index(value, self.operation_types)
        return self.operation_types[index]

    def _apply_aliases(self, action: dict[str, Any]) -> dict[str, Any]:
        """Accept common user-facing names while preserving canonical output."""

        operation = str(action["operation"])
        canonical = dict(action)
        if (
            operation == "add_catalyst"
            and "catalyst_amount_mol" not in canonical
            and "amount_mol" in canonical
        ):
            canonical["catalyst_amount_mol"] = canonical.pop("amount_mol")
        elif operation == "add_catalyst":
            canonical.pop("amount_mol", None)
        if (
            operation in {"heat", "cool_crystallize", "evaporate", "distill", "run_flow"}
            and "target_temperature_K" not in canonical
            and "temperature_K" in canonical
        ):
            canonical["target_temperature_K"] = canonical.pop("temperature_K")
        elif operation in {
            "heat",
            "cool_crystallize",
            "evaporate",
            "distill",
            "run_flow",
        }:
            canonical.pop("temperature_K", None)
        if operation in {"heat", "wait", "mix"} and "stirring_speed_rpm" not in canonical:
            if "stirring_rpm" in canonical:
                canonical["stirring_speed_rpm"] = canonical["stirring_rpm"]
            elif "stirring_speed" in canonical:
                canonical["stirring_speed_rpm"] = canonical["stirring_speed"]
        if operation in {"heat", "wait", "mix"}:
            canonical.pop("stirring_rpm", None)
            canonical.pop("stirring_speed", None)
        if operation == "sample" and "sample_volume_L" not in canonical and "volume_L" in canonical:
            canonical["sample_volume_L"] = canonical.pop("volume_L")
        elif operation == "sample":
            canonical.pop("volume_L", None)
        if (
            operation == "add_extractant"
            and "extractant" not in canonical
            and "solvent" in canonical
        ):
            canonical["extractant"] = canonical.pop("solvent")
        elif operation == "add_extractant":
            canonical.pop("solvent", None)
        if (
            operation == "separate_phase"
            and "target_phase" not in canonical
            and "phase" in canonical
        ):
            canonical["target_phase"] = canonical.pop("phase")
        elif operation == "separate_phase":
            canonical.pop("phase", None)
        if operation == "wash" and "wash_volume_L" not in canonical and "volume_L" in canonical:
            canonical["wash_volume_L"] = canonical.pop("volume_L")
        elif operation == "wash":
            canonical.pop("volume_L", None)
        return canonical

    def encode_vector(self, action: dict[str, Any]) -> np.ndarray:
        """Encode canonical event JSON as a stable numeric vector."""

        action = self.canonicalize(action)
        values = [
            self.operation_types.index(str(action["operation"])),
            self._float(action, "amount_mol", 0.0),
            self._float(action, "volume_L", 0.0),
            self._float(action, "catalyst_amount_mol", 0.0),
            self._float(action, "target_temperature_K", 298.15),
            self._float(action, "duration_s", 0.0),
            self._float(action, "stirring_speed_rpm", 600.0),
            self._float(action, "sample_volume_L", 0.0),
            self._index(action.get("instrument", "hplc"), self.instruments),
            self._float(action, "catalyst", 0.0),
            self._float(action, "solvent", 0.0),
            self._index(action.get("phase", "reactor_liquid"), self.phases),
            self._index(action.get("target_phase", "organic"), self.phases),
            self._index(action.get("extractant", 0), self.extractants),
            self._float(action, "wash_volume_L", 0.0),
            self._float(action, "transfer_fraction", 1.0),
            self._float(action, "seed_mass_g", 0.0),
            self._float(action, "reflux_ratio", 1.5),
            self._float(action, "flow_rate_mL_min", 1.0),
            self._float(action, "residence_time_s", 600.0),
            self._float(action, "potential_V", 1.2),
            self._float(action, "current_mA", 50.0),
            self._float(action, "electrolyte_profile", 1.0),
            self._float(action, "container", 1.0),
            self._float(action, "source_container", 0.0),
            self._float(action, "destination_container", 1.0),
            self._float(action, "capacity_L", 0.1),
            self._float(action, "mixing", 0.0),
            self._float(action, "component", 0.0),
            self._float(action, "vessel", 0.0),
            self._float(action, "connection", 0.0),
        ]
        control_values = [
            self._float(action, key, CONTROL_NUMERIC[key][3] if key in CONTROL_NUMERIC else 0)
            for key in CONTROL_VECTOR_FIELDS
        ]
        spectral_values = [
            self._float(action, key, SPECTRAL_BOUNDS[key][3]) for key in SPECTRAL_FIELDS
        ]
        vector = np.asarray([*values, *control_values, *spectral_values], dtype=np.float32)
        if not np.all(np.isfinite(vector)):
            raise ValueError("Encoded action vector must contain only finite values")
        return vector

    def decode_vector(self, vector: Any) -> dict[str, Any]:
        """Decode a numeric carrier into one operation-conditional event.

        The fixed-width vector is a transport representation only.  Returning
        inactive coordinates would turn it into an over-declared semantic
        payload and make the result incompatible with the exact operation
        contracts enforced by :class:`OperationValidator`.
        """

        array = np.asarray(vector, dtype=float).reshape(-1)
        if array.shape != (len(GYM_ACTION_KEYS),) or not np.all(np.isfinite(array)):
            raise ValueError(f"Action vector must be finite with shape ({len(GYM_ACTION_KEYS)},)")
        operation_index = int(np.clip(round(array[0]), 0, len(self.operation_types) - 1))
        instrument_index = int(np.clip(round(array[8]), 0, len(self.instruments) - 1))
        phase_index = int(np.clip(round(array[11]), 0, len(self.phases) - 1))
        target_phase_index = int(np.clip(round(array[12]), 0, len(self.phases) - 1))
        operation = self.operation_types[operation_index]
        decoded: dict[str, Any] = {
            "operation": operation,
            "amount_mol": float(array[1]),
            "volume_L": float(array[2]),
            "catalyst_amount_mol": float(array[3]),
            "target_temperature_K": float(array[4]),
            "duration_s": float(array[5]),
            "stirring_speed_rpm": float(array[6]),
            "sample_volume_L": float(array[7]),
            "instrument": self.instruments[instrument_index],
            "catalyst": int(np.clip(round(array[9]), 0, len(self.catalysts) - 1)),
            "solvent": int(np.clip(round(array[10]), 0, len(self.solvents) - 1)),
            "phase": self.phases[phase_index],
            "target_phase": self.phases[target_phase_index],
            "extractant": int(np.clip(round(array[13]), 0, len(self.extractants) - 1)),
            "wash_volume_L": float(array[14]),
            "transfer_fraction": float(array[15]),
            "seed_mass_g": float(array[16]),
            "reflux_ratio": float(array[17]),
            "flow_rate_mL_min": float(array[18]),
            "residence_time_s": float(array[19]),
            "potential_V": float(array[20]),
            "current_mA": float(array[21]),
            "electrolyte_profile": int(
                np.clip(round(array[22]), 0, len(self.electrolyte_profiles) - 1)
            ),
        }
        required = operation_contracts(
            include_campaign_controls=operation in CAMPAIGN_CONTROL_OPERATIONS
        )[operation].required_fields
        for index, key in enumerate(("container", "source_container", "destination_container"), 23):
            decoded[key] = int(np.clip(round(array[index]), 0, len(CONTAINER_IDS) - 1))
        decoded["capacity_L"] = float(array[26])
        decoded["mixing"] = int(np.clip(round(array[27]), 0, 1))
        decoded["component"] = int(np.clip(round(array[28]), 0, len(FEED_IDS) - 1))
        decoded["vessel"] = int(np.clip(round(array[29]), 0, max(len(self.vessels) - 1, 0)))
        decoded["connection"] = int(np.clip(round(array[30]), 0, max(len(self.connections) - 1, 0)))
        for index, key in enumerate(CONTROL_VECTOR_FIELDS, 31):
            decoded[key] = (
                int(np.clip(round(array[index]), 0, len(CONTROL_CHOICES[key]) - 1))
                if key in CONTROL_CHOICES
                else float(array[index])
            )
        for index, key in enumerate(SPECTRAL_FIELDS, 31 + len(CONTROL_VECTOR_FIELDS)):
            decoded[key] = float(array[index])
        return {"operation": operation, **{key: decoded[key] for key in required}}

    def phase_name(self, value: Any) -> str:
        if isinstance(value, str):
            # Keep unknown strings intact so validation rejects them atomically
            # instead of silently converting them to the organic phase.
            return value
        return self.phases[self._numeric_choice_index(value, self.phases, "phase")]

    def extractant_name(self, value: Any) -> int:
        if isinstance(value, str):
            normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
            # Backward-compatible role alias used by early event recipes.
            if normalized == "organic":
                return self.extractants.index("toluene")
        return self._choice_index(value, self.extractants)

    @staticmethod
    def _float(action: dict[str, Any], key: str, default: float) -> float:
        value = action.get(key, default)
        values = np.asarray(value).reshape(-1)
        if values.size != 1:
            raise ValueError(f"{key} must be a scalar numeric value")
        result = float(values[0])
        if not np.isfinite(result):
            raise ValueError(f"{key} must be finite")
        return result

    @staticmethod
    def _index(value: Any, choices: tuple[str, ...]) -> int:
        if isinstance(value, str) and value in choices:
            return choices.index(value)
        return ActionCodec._numeric_choice_index(value, choices, "choice")

    @staticmethod
    def _choice_index(value: Any, choices: tuple[str, ...]) -> int:
        if isinstance(value, str):
            display_name = value.split("·", 1)[0].split("路", 1)[0]
            normalized = display_name.strip().lower().replace("-", "_").replace(" ", "_")
            candidates = [normalized]
            if normalized.startswith("catalyst_") and len(normalized) == len("catalyst_") + 1:
                candidates.append(f"cat_{normalized[-1]}")
            for candidate in candidates:
                if candidate in choices:
                    return choices.index(candidate)
            if normalized.isdigit():
                value = int(normalized)
            else:
                allowed = ", ".join(choices)
                raise ValueError(f"Unknown material choice {value!r}; allowed: {allowed}")
        return ActionCodec._numeric_choice_index(value, choices, "material choice")

    @staticmethod
    def _numeric_choice_index(value: Any, choices: tuple[str, ...], label: str) -> int:
        if isinstance(value, (bool, np.bool_)):
            raise ValueError(f"{label} must be an integer categorical index, not boolean")
        try:
            values = np.asarray(value).reshape(-1)
            if values.size != 1:
                raise ValueError(f"{label} must be a scalar categorical index")
            coordinate = float(values[0])
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{label} must be a scalar categorical index") from exc
        if not np.isfinite(coordinate) or not coordinate.is_integer():
            raise ValueError(f"{label} must be a finite integer categorical index")
        index = int(coordinate)
        if not 0 <= index < len(choices):
            raise ValueError(f"{label} index {index} is outside [0, {len(choices) - 1}]")
        return index
