"""Typed phase-ledger helpers for the transactional runtime."""

from __future__ import annotations

from typing import Any

import numpy as np

from chemworld.foundation import WorldState, process_with_metrics
from chemworld.foundation.state import (
    EquipmentLedger,
    PhaseLedger,
    PhaseRecord,
    selected_phase_id,
)
from chemworld.runtime.species import MechanismSpeciesView
from chemworld.world.separation_kernel import downstream_truth_values
from chemworld.world.species_roles import PHASE_PRODUCT_AMOUNT_KEY

SELECTED_PHASE_UNSET = object()
PHASE_PROCESS_METRIC_KEYS = frozenset(
    {
        "purity",
        "recovery",
        "phase_ratio",
        "product_in_organic",
        "product_in_aqueous",
        "impurity_signal",
        "solvent_loss",
        "process_mass_balance_error",
    }
)
PHASE_PRIMARY_METADATA_KEYS = frozenset(
    {
        PHASE_PRODUCT_AMOUNT_KEY,
        "impurity_mol",
        "solvent_loss",
    }
)


def action_float(action: dict[str, Any], key: str, default: float) -> float:
    value = action.get(key, default)
    return float(np.asarray(value).reshape(-1)[0])


def empty_phase(volume_L: float = 0.0) -> dict[str, float]:
    return {
        "volume_L": volume_L,
        PHASE_PRODUCT_AMOUNT_KEY: 0.0,
        "impurity_mol": 0.0,
        "solvent_loss": 0.0,
    }


def phase_type(phase_name: str) -> str:
    if phase_name in {"organic", "aqueous", "solid"}:
        return phase_name
    return "liquid"


class ChemWorldPhaseLedgerServices:
    """Normalize typed phase ledgers and derived process metrics."""

    def __init__(self, species_view: MechanismSpeciesView) -> None:
        self.species_view = species_view

    def phase_ledger(self, state: WorldState) -> dict[str, dict[str, float]]:
        ledger: dict[str, dict[str, float]] = {}
        if state.phases is not None:
            for phase_name, phase in state.phases.phases.items():
                ledger[str(phase_name)] = self._phase_record_to_entry(phase)
            if ledger:
                return ledger
        if "reactor_liquid" not in ledger:
            ledger["reactor_liquid"] = {
                "volume_L": state.volume_L,
                PHASE_PRODUCT_AMOUNT_KEY: self.phase_product_amount(state),
                "impurity_mol": self.phase_impurity_amount(state),
                "solvent_loss": 0.0,
            }
        return ledger

    def phase_ledger_records(
        self,
        state: WorldState,
        phase_ledger: dict[str, dict[str, float]],
        *,
        settled: bool | None = None,
        selected_phase: str | None = None,
        preserve_selection: bool = True,
    ) -> PhaseLedger:
        species_template = dict.fromkeys(state.species_amounts, 0.0)
        phases: dict[str, PhaseRecord] = {}
        has_split_phases = bool({"organic", "aqueous"} & set(phase_ledger))

        for phase_name, values in phase_ledger.items():
            previous_phase = (
                None if state.phases is None else state.phases.phases.get(phase_name)
            )
            phase_settled = (
                bool(previous_phase.settled)
                if settled is None and previous_phase is not None
                else bool(settled)
            )
            if selected_phase is None:
                phase_selected = (
                    bool(previous_phase.selected)
                    if preserve_selection and previous_phase
                    else False
                )
            else:
                phase_selected = selected_phase == phase_name
            species_amounts = species_template.copy()
            if phase_name == "reactor_liquid":
                species_amounts.update(
                    {key: float(value) for key, value in state.species_amounts.items()}
                )
            else:
                species_amounts[self.product_species_for_phase(phase_name, state)] = float(
                    values.get(PHASE_PRODUCT_AMOUNT_KEY, 0.0)
                )
                species_amounts[self.impurity_species_for_phase(phase_name, state)] = float(
                    values.get("impurity_mol", 0.0)
                )
            phases[phase_name] = PhaseRecord(
                phase_id=phase_name,
                vessel_id=state.vessel_id,
                phase_type=phase_type(phase_name),
                volume_L=float(values.get("volume_L", 0.0)),
                species_amounts_mol=species_amounts,
                settled=phase_settled,
                selected=phase_selected,
                metadata={},
            )

        if has_split_phases and "reactor_liquid" not in phases:
            carrier_phase = "aqueous" if "aqueous" in phases else next(iter(phases))
            split_species = (
                self.product_candidate_species()
                | self.partition_impurity_candidate_species()
            )
            for species_id, amount in state.species_amounts.items():
                if species_id in split_species:
                    continue
                destination = self.declared_phase_for_species(species_id)
                if destination not in phases:
                    destination = carrier_phase
                phase = phases[destination]
                species_amounts = phase.species_amounts_mol.copy()
                species_amounts[species_id] = float(amount)
                phases[destination] = PhaseRecord(
                    phase_id=phase.phase_id,
                    vessel_id=phase.vessel_id,
                    phase_type=phase.phase_type,
                    volume_L=phase.volume_L,
                    species_amounts_mol=species_amounts,
                    settled=phase.settled,
                    selected=phase.selected,
                    metadata=self._phase_metadata_without_primary_state(phase.metadata),
                )

        return PhaseLedger(phases)

    def phase_metadata(
        self,
        state: WorldState,
        phase_ledger: dict[str, dict[str, float]],
        *,
        updates: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        _ = phase_ledger
        metadata = state.metadata.copy()
        metadata.pop("phase_ledger", None)
        metadata.pop("phase_system", None)
        metadata.pop("phase_settled", None)
        metadata.pop("selected_phase", None)
        metadata.pop("stirring_speed_rpm", None)
        if updates:
            metadata.update(updates)
        return metadata

    def phase_process_metrics(
        self,
        state: WorldState,
        phase_ledger: dict[str, dict[str, float]],
    ) -> dict[str, float]:
        truth_values = downstream_truth_values(
            state,
            phase_ledger,
            target_species=self.species_view.target_species_for_state(state),
            impurity_species=self.species_view.impurity_species_for_state(state),
        )
        metrics = {
            key: float(value)
            for key, value in truth_values.items()
            if key in PHASE_PROCESS_METRIC_KEYS
        }
        metrics["solvent_loss"] = max(
            metrics.get("solvent_loss", 0.0),
            *(
                float(values.get("solvent_loss", 0.0))
                for values in phase_ledger.values()
            ),
        )
        return metrics

    def with_phase_ledger(
        self,
        state: WorldState,
        phase_ledger: dict[str, dict[str, float]],
        *,
        metadata_updates: dict[str, Any] | None = None,
        phase_settled: bool | None = None,
        selected_phase: str | None | object = SELECTED_PHASE_UNSET,
        ledger: Any | None = None,
        volume_L: float | None = None,
        equipment: EquipmentLedger | None = None,
    ) -> WorldState:
        metadata_updates = {} if metadata_updates is None else metadata_updates.copy()
        preserve_selection = selected_phase is SELECTED_PHASE_UNSET
        selected_phase_value: str | None = None
        if preserve_selection:
            selected_phase_value = selected_phase_id(state.phases)
        elif selected_phase is None:
            selected_phase_value = None
        elif selected_phase is not None:
            selected_phase_value = str(selected_phase)
        metadata = self.phase_metadata(state, phase_ledger, updates=metadata_updates)
        phases = self.phase_ledger_records(
            state,
            phase_ledger,
            settled=None if phase_settled is None else bool(phase_settled),
            selected_phase=selected_phase_value,
            preserve_selection=preserve_selection,
        )
        species_amounts = phases.total_amounts_mol()
        process = process_with_metrics(
            state.process,
            **self.phase_process_metrics(state, phase_ledger),
        )
        return state.replace(
            species_amounts=species_amounts,
            phases=phases,
            volume_L=state.volume_L if volume_L is None else volume_L,
            ledger=state.ledger if ledger is None else ledger,
            metadata=metadata,
            equipment=state.equipment if equipment is None else equipment,
            process=process,
        )

    def product_candidate_species(self) -> set[str]:
        candidates: set[str] = set()
        for species_id in self.species_view.target_species:
            candidates.update(self._phase_suffix_family(species_id))
        for alias in ("product_organic", "product_aqueous"):
            alias_species_id = self._species_with_alias(alias)
            if alias_species_id is not None:
                candidates.update(self._phase_suffix_family(alias_species_id))
        return candidates

    def impurity_candidate_species(self) -> set[str]:
        candidates: set[str] = set()
        for species_id in self.species_view.impurity_species:
            candidates.update(self._phase_suffix_family(species_id))
        for species_id in self.species_view.byproduct_species:
            candidates.update(self._phase_suffix_family(species_id))
        for species_id in self.species_view.degradation_species:
            candidates.update(self._phase_suffix_family(species_id))
        for alias in ("byproduct_organic", "byproduct_aqueous", "degradation_product"):
            alias_species_id = self._species_with_alias(alias)
            if alias_species_id is not None:
                candidates.update(self._phase_suffix_family(alias_species_id))
        return candidates

    def partition_impurity_candidate_species(self) -> set[str]:
        """Return only impurity identities that represent one partition family.

        The broader impurity role also contains chemically distinct degradation
        and coupling products.  Collapsing those species into the aqueous/organic
        byproduct pair changes elemental inventory during a closed mixing step.
        """

        candidates: set[str] = set()
        for alias in ("byproduct_organic", "byproduct_aqueous"):
            species_id = self._species_with_alias(alias)
            if species_id is not None:
                candidates.update(self._phase_suffix_family(species_id))
        if candidates:
            return candidates
        for species_id in self.species_view.byproduct_species:
            if species_id.endswith(("_org", "_aq")):
                candidates.update(self._phase_suffix_family(species_id))
        return candidates or self.impurity_candidate_species()

    def declared_phase_for_species(self, species_id: str) -> str | None:
        for species in self.species_view.mechanism.network.species:
            if species.species_id == species_id:
                return str(species.phase)
        return None

    def product_species_for_phase(self, phase_name: str, state: WorldState) -> str:
        if phase_name == "organic":
            return (
                self._species_with_alias("product_organic")
                or self.species_view.primary_target_species
            )
        if phase_name == "aqueous":
            aqueous = self._species_with_alias("product_aqueous")
            if aqueous is not None:
                return aqueous
        for species_id in sorted(self.product_candidate_species()):
            if species_id in state.species_amounts:
                return species_id
        return self.species_view.primary_target_species

    def impurity_species_for_phase(self, phase_name: str, state: WorldState) -> str:
        if phase_name == "organic":
            organic = self._species_with_alias("byproduct_organic")
            if organic is not None:
                return organic
        if phase_name == "aqueous":
            aqueous = self._species_with_alias("byproduct_aqueous")
            if aqueous is not None:
                return aqueous
        for species_id in sorted(self.impurity_candidate_species()):
            if species_id in state.species_amounts:
                return species_id
        return self.species_view.primary_impurity_species

    def phase_product_amount(self, state: WorldState) -> float:
        return sum(
            float(state.species_amounts.get(species_id, 0.0))
            for species_id in self.product_candidate_species()
        )

    def phase_impurity_amount(self, state: WorldState) -> float:
        return sum(
            float(state.species_amounts.get(species_id, 0.0))
            for species_id in self.partition_impurity_candidate_species()
        )

    def _phase_record_to_entry(self, phase: PhaseRecord) -> dict[str, float]:
        product_amount = sum(
            float(phase.species_amounts_mol.get(species_id, 0.0))
            for species_id in self.product_candidate_species()
        )
        impurity_amount = sum(
            float(phase.species_amounts_mol.get(species_id, 0.0))
            for species_id in self.partition_impurity_candidate_species()
        )
        return {
            "volume_L": float(phase.volume_L),
            PHASE_PRODUCT_AMOUNT_KEY: product_amount,
            "impurity_mol": impurity_amount,
            "solvent_loss": 0.0,
        }

    def _species_with_alias(self, alias: str) -> str | None:
        mechanism = self.species_view.mechanism
        for species_id, aliases in mechanism.species_roles.items():
            if alias in aliases:
                return species_id
        return None

    @staticmethod
    def _phase_suffix_family(species_id: str) -> set[str]:
        family = {species_id}
        for suffix in ("_org", "_aq"):
            if species_id.endswith(suffix):
                base = species_id[: -len(suffix)]
                family.update({base, f"{base}_org", f"{base}_aq"})
        return family

    @staticmethod
    def _phase_metadata_without_primary_state(metadata: dict[str, Any]) -> dict[str, Any]:
        cleaned = metadata.copy()
        for key in PHASE_PRIMARY_METADATA_KEYS:
            cleaned.pop(key, None)
        return cleaned


__all__ = [
    "PHASE_PRIMARY_METADATA_KEYS",
    "PHASE_PROCESS_METRIC_KEYS",
    "SELECTED_PHASE_UNSET",
    "ChemWorldPhaseLedgerServices",
    "action_float",
    "empty_phase",
    "phase_type",
]
