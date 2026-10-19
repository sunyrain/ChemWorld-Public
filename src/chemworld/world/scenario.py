"""Scenario specifications and generation for the shared ChemWorld law."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from random import Random
from typing import Any, Protocol

from chemworld.foundation import WorldState
from chemworld.foundation.state import SpeciesLedger
from chemworld.runtime.mechanisms import CompiledMechanism, compile_mechanism_for_scenario
from chemworld.world.parameters import (
    WORLD_FAMILY_VERSION,
    ChemWorldParameters,
    load_chemworld_parameters,
)

WORLD_LAW_ID = WORLD_FAMILY_VERSION


@dataclass(frozen=True)
class ScenarioSpec:
    scenario_id: str
    world_law_id: str
    family: str
    split: str
    difficulty: str
    hidden_parameter_seed: int
    initial_state_seed: int
    initial_state_id: str
    parameter_profile: str
    allowed_module_tags: tuple[str, ...]
    expected_qualitative_behavior: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "scenario_id": self.scenario_id,
            "world_law_id": self.world_law_id,
            "family": self.family,
            "split": self.split,
            "difficulty": self.difficulty,
            "hidden_parameter_seed": self.hidden_parameter_seed,
            "initial_state_seed": self.initial_state_seed,
            "initial_state_id": self.initial_state_id,
            "parameter_profile": self.parameter_profile,
            "allowed_module_tags": list(self.allowed_module_tags),
            "expected_qualitative_behavior": list(self.expected_qualitative_behavior),
        }


@dataclass(frozen=True)
class ScenarioFamilySpec:
    family_id: str
    world_law_id: str
    description: str
    module_tags: tuple[str, ...]
    split_policy: str

    def to_dict(self) -> dict[str, object]:
        return {
            "family_id": self.family_id,
            "world_law_id": self.world_law_id,
            "description": self.description,
            "module_tags": list(self.module_tags),
            "split_policy": self.split_policy,
        }


@dataclass(frozen=True)
class ScenarioInstance:
    spec: ScenarioSpec
    parameters: ChemWorldParameters
    initial_state: WorldState
    compiled_mechanism: CompiledMechanism

    def to_card(self) -> dict[str, object]:
        intervention_hash = self.initial_state.metadata.get("world_family_intervention_hash")
        return {
            **self.spec.to_dict(),
            "world_id": self.parameters.world_id,
            "world_provider": self.parameters.provider,
            "world_family_version": self.parameters.family_version,
            "world_family_intervention_version": self.initial_state.metadata.get(
                "world_family_intervention_version"
            ),
            "world_family_intervention_hash": intervention_hash,
            "mechanism_family_intervention_version": self.initial_state.metadata.get(
                "mechanism_family_intervention_version"
            ),
            "mechanism_family_intervention_hash": self.initial_state.metadata.get(
                "mechanism_family_intervention_hash"
            ),
            "equilibrium_entity_panel_version": self.initial_state.metadata.get(
                "equilibrium_entity_panel_version"
            ),
            "equilibrium_entity_panel_hash": self.initial_state.metadata.get(
                "equilibrium_entity_panel_hash"
            ),
            "electrochemical_material_family_contract_version": (
                self.initial_state.metadata.get(
                    "electrochemical_material_family_contract_version"
                )
            ),
            "electrochemical_material_family_id": self.initial_state.metadata.get(
                "electrochemical_material_family_id"
            ),
            "electrochemical_material_family_sha256": (
                self.initial_state.metadata.get(
                    "electrochemical_material_family_sha256"
                )
            ),
            "electrochemical_material_instance_sha256": (
                self.initial_state.metadata.get(
                    "electrochemical_material_instance_sha256"
                )
            ),
            "crystallization_material_family_contract_version": (
                self.initial_state.metadata.get(
                    "crystallization_material_family_contract_version"
                )
            ),
            "crystallization_material_family_id": self.initial_state.metadata.get(
                "crystallization_material_family_id"
            ),
            "crystallization_material_family_sha256": (
                self.initial_state.metadata.get(
                    "crystallization_material_family_sha256"
                )
            ),
            "crystallization_material_instance_sha256": (
                self.initial_state.metadata.get(
                    "crystallization_material_instance_sha256"
                )
            ),
            "mechanism": self.compiled_mechanism.to_dict(),
        }


class ScenarioGenerator(Protocol):
    def generate(
        self,
        spec: ScenarioSpec,
        seed: int,
        interventions: tuple[dict[str, Any], ...] = (),
    ) -> ScenarioInstance:
        """Generate a reproducible scenario instance."""


class DefaultScenarioGenerator:
    """Generate ChemWorld scenarios from split, seed, and initial-state policy."""

    def generate(
        self,
        spec: ScenarioSpec,
        seed: int,
        interventions: tuple[dict[str, Any], ...] = (),
    ) -> ScenarioInstance:
        from chemworld.world.state_factory import initial_chemworld_state

        profile_offset = _profile_offset(spec.parameter_profile)
        parameter_seed = seed + spec.hidden_parameter_seed + profile_offset
        initial_seed = seed + spec.initial_state_seed + profile_offset
        parameters = load_chemworld_parameters(spec.split, parameter_seed)
        compiled_mechanism = compile_mechanism_for_scenario(spec.scenario_id)
        mechanism_payloads = tuple(
            item for item in interventions if item.get("kind") == "mechanism_family"
        )
        material_law_payloads = tuple(
            item
            for item in interventions
            if item.get("kind") == "material_law_counterfactual"
        )
        entity_panel_payloads = tuple(
            item
            for item in interventions
            if item.get("kind") == "equilibrium_entity_panel"
        )
        if len(mechanism_payloads) > 1:
            raise ValueError("only one mechanism-family intervention is allowed")
        if len(material_law_payloads) > 1:
            raise ValueError("only one material-law counterfactual is allowed")
        if len(entity_panel_payloads) > 1:
            raise ValueError("only one equilibrium entity panel is allowed")
        parsed_mechanism_intervention = None
        if mechanism_payloads:
            from chemworld.world.mechanism_family import (
                MechanismFamilyIntervention,
                derive_mechanism_family,
            )

            parsed_mechanism_intervention = MechanismFamilyIntervention.from_dict(
                mechanism_payloads[0]
            )
            if parsed_mechanism_intervention.mode != "constitutive_law_family":
                compiled_mechanism = derive_mechanism_family(
                    compiled_mechanism,
                    parsed_mechanism_intervention,
                )
        initial_state = initial_chemworld_state(
            species_ids=tuple(compiled_mechanism.species_index),
            species_roles=compiled_mechanism.species_roles,
            initial_amounts_mol=compiled_mechanism.initial_amount_policy,
            initial_limiting_species=compiled_mechanism.score_spec.initial_limiting_species,
        )
        if spec.family == "partition":
            feed_rng = Random(parameter_seed ^ 0x8A17_2D41)
            feed_amounts = initial_state.species_amounts.copy()
            target_species = compiled_mechanism.score_spec.target_species
            impurity_species = compiled_mechanism.score_spec.impurity_species
            target_amount = feed_rng.uniform(0.006, 0.014)
            impurity_amount = feed_rng.uniform(0.001, 0.004)
            if target_species:
                feed_amounts[target_species[0]] = target_amount
            if impurity_species:
                feed_amounts[impurity_species[0]] = impurity_amount
            initial_state = initial_state.replace(
                species_amounts=feed_amounts,
                species=SpeciesLedger(
                    species_roles=compiled_mechanism.species_roles,
                    initial_amounts_mol={
                        **({target_species[0]: target_amount} if target_species else {}),
                        **({impurity_species[0]: impurity_amount} if impurity_species else {}),
                    },
                ),
            )
        metadata = initial_state.metadata.copy()
        metadata.update(
            {
                "scenario_id": spec.scenario_id,
                "parameter_profile": spec.parameter_profile,
                "initial_state_seed": spec.initial_state_seed,
            }
        )
        if spec.family == "equilibrium_characterization":
            equilibrium_rng = Random(parameter_seed ^ 0xC4E9_51B3)
            metadata.update(
                {
                    "hidden_equilibrium_pka": equilibrium_rng.uniform(3.8, 5.4),
                    "hidden_equilibrium_ksp": 10.0 ** equilibrium_rng.uniform(-10.4, -9.2),
                }
            )
        if spec.initial_state_seed:
            rng = Random(initial_seed)
            initial_state = initial_state.replace(
                temperature_K=298.15 + rng.uniform(-1.5, 1.5),
                pressure_Pa=101_325.0 + rng.uniform(-250.0, 250.0),
                metadata={
                    **metadata,
                    "initial_condition_jitter": {
                        "temperature_K": True,
                        "pressure_Pa": True,
                    },
                },
            )
        else:
            initial_state = initial_state.replace(metadata=metadata)
        mechanism_metadata = {
            **initial_state.metadata,
            "mechanism_id": compiled_mechanism.mechanism_id,
            "mechanism_hash": compiled_mechanism.mechanism_hash,
        }
        initial_state = initial_state.replace(metadata=mechanism_metadata)
        instance = ScenarioInstance(
            spec=spec,
            parameters=parameters,
            initial_state=initial_state,
            compiled_mechanism=compiled_mechanism,
        )
        axis_payloads = tuple(
            item
            for item in interventions
            if item.get("kind")
            not in {
                "mechanism_family",
                "material_law_counterfactual",
                "equilibrium_entity_panel",
            }
        )
        if (
            not mechanism_payloads
            and not material_law_payloads
            and not entity_panel_payloads
            and not axis_payloads
        ):
            return instance
        if mechanism_payloads:
            from chemworld.world.mechanism_family import apply_mechanism_family_intervention

            if parsed_mechanism_intervention is None:
                raise RuntimeError("mechanism-family intervention was not parsed")
            instance = apply_mechanism_family_intervention(
                instance,
                parsed_mechanism_intervention,
            )
        if material_law_payloads:
            from chemworld.world.material_counterfactual import (
                MaterialLawCounterfactual,
                apply_material_law_counterfactual,
            )

            instance = apply_material_law_counterfactual(
                instance,
                MaterialLawCounterfactual.from_dict(material_law_payloads[0]),
            )
        if entity_panel_payloads:
            from chemworld.world.equilibrium_entity import (
                EquilibriumEntityPanelIntervention,
                apply_equilibrium_entity_panel,
            )

            instance = apply_equilibrium_entity_panel(
                instance,
                EquilibriumEntityPanelIntervention.from_dict(
                    entity_panel_payloads[0]
                ),
            )
        if not axis_payloads:
            return instance
        from chemworld.world.world_family import (
            AxisIntervention,
            apply_axis_interventions,
        )

        parsed = tuple(AxisIntervention.from_dict(item) for item in axis_payloads)
        return apply_axis_interventions(instance, parsed)


def _profile_offset(parameter_profile: str) -> int:
    if parameter_profile == "teaching_assay":
        return 0
    digest = sha256(parameter_profile.encode()).digest()
    return int.from_bytes(digest[:2], "little")


SCENARIO_FAMILIES: dict[str, ScenarioFamilySpec] = {
    "reaction": ScenarioFamilySpec(
        family_id="reaction",
        world_law_id=WORLD_LAW_ID,
        description="Semi-mechanistic batch reaction with hidden kinetics and safety tradeoffs.",
        module_tags=("reaction", "thermal", "observation"),
        split_policy="same mechanism family across public/private splits; distinct hidden seeds",
    ),
    "reaction_separation": ScenarioFamilySpec(
        family_id="reaction_separation",
        world_law_id=WORLD_LAW_ID,
        description="Reaction followed by phase partition, separation, and purification.",
        module_tags=("reaction", "thermal", "phase_partition", "separation", "observation"),
        split_policy="same mechanism family across public/private splits; distinct hidden seeds",
    ),
    "partition": ScenarioFamilySpec(
        family_id="partition",
        world_law_id=WORLD_LAW_ID,
        description="Phase partition and downstream characterization under sparse observations.",
        module_tags=("phase_partition", "separation", "observation"),
        split_policy=(
            "same partition-law family across public/private splits; distinct hidden seeds"
        ),
    ),
    "reaction_crystallization": ScenarioFamilySpec(
        family_id="reaction_crystallization",
        world_law_id=WORLD_LAW_ID,
        description="Reaction followed by seeded cooling crystallization and filtration.",
        module_tags=("reaction", "thermal", "crystallization", "observation"),
        split_policy="same shared law; crystallization parameters vary by hidden seed",
    ),
    "reaction_distillation": ScenarioFamilySpec(
        family_id="reaction_distillation",
        world_law_id=WORLD_LAW_ID,
        description="Reaction followed by evaporation, distillation, and fraction collection.",
        module_tags=("reaction", "thermal", "distillation", "observation"),
        split_policy="same shared law; volatility and recovery profiles vary by hidden seed",
    ),
    "continuous_flow": ScenarioFamilySpec(
        family_id="continuous_flow",
        world_law_id=WORLD_LAW_ID,
        description="Continuous-flow projection of the same reaction kinetics.",
        module_tags=("reaction", "thermal", "continuous_flow", "observation"),
        split_policy="same shared law; flow residence-time response varies by hidden seed",
    ),
    "electrochemistry": ScenarioFamilySpec(
        family_id="electrochemistry",
        world_law_id=WORLD_LAW_ID,
        description="Electrochemical conversion slice with potential/current controls.",
        module_tags=("reaction", "electrochemistry", "observation"),
        split_policy="same shared law; electrochemical selectivity varies by hidden seed",
    ),
    "equilibrium_characterization": ScenarioFamilySpec(
        family_id="equilibrium_characterization",
        world_law_id=WORLD_LAW_ID,
        description="Bounded aqueous-equilibrium characterization with public pH observations.",
        module_tags=("equilibrium_chemistry", "observation", "instrumentation"),
        split_policy="same D4 equilibrium slice; hidden acidity profile varies by seed",
    ),
}


SCENARIO_REGISTRY: dict[str, ScenarioSpec] = {
    "reaction-optimization": ScenarioSpec(
        scenario_id="reaction-optimization",
        world_law_id=WORLD_LAW_ID,
        family="reaction",
        split="public-test",
        difficulty="standard",
        hidden_parameter_seed=0,
        initial_state_seed=0,
        initial_state_id="reaction-optimization:default",
        parameter_profile="balanced_hidden_kinetics",
        allowed_module_tags=("reaction", "thermal", "observation"),
        expected_qualitative_behavior=(
            "temperature accelerates desired and undesired reactions",
            "long residence time can degrade product",
            "catalyst and solvent interact nonlinearly",
        ),
    ),
    "reaction-safety": ScenarioSpec(
        scenario_id="reaction-safety",
        world_law_id=WORLD_LAW_ID,
        family="reaction",
        split="public-test",
        difficulty="hard",
        hidden_parameter_seed=11,
        initial_state_seed=3,
        initial_state_id="reaction-safety:default",
        parameter_profile="high_safety_sensitivity",
        allowed_module_tags=("reaction", "thermal", "observation"),
        expected_qualitative_behavior=(
            "high temperature and concentration quickly increase risk",
            "safe optimum is separated from maximum-yield region",
        ),
    ),
    "reaction-mechanism": ScenarioSpec(
        scenario_id="reaction-mechanism",
        world_law_id=WORLD_LAW_ID,
        family="reaction",
        split="public-test",
        difficulty="standard",
        hidden_parameter_seed=17,
        initial_state_seed=5,
        initial_state_id="reaction-mechanism:default",
        parameter_profile="mechanism_probe",
        allowed_module_tags=("reaction", "thermal", "observation"),
        expected_qualitative_behavior=(
            "solvent changes selectivity",
            "degradation signal grows after product peak",
        ),
    ),
    "reaction-to-assay": ScenarioSpec(
        scenario_id="reaction-to-assay",
        world_law_id=WORLD_LAW_ID,
        family="reaction",
        split="public-dev",
        difficulty="intro",
        hidden_parameter_seed=0,
        initial_state_seed=0,
        initial_state_id="reaction-to-assay:default",
        parameter_profile="teaching_assay",
        allowed_module_tags=("reaction", "thermal", "observation"),
        expected_qualitative_behavior=("valid final assay requires termination",),
    ),
    "reaction-to-purification": ScenarioSpec(
        scenario_id="reaction-to-purification",
        world_law_id=WORLD_LAW_ID,
        family="reaction_separation",
        split="public-test",
        difficulty="hard",
        hidden_parameter_seed=23,
        initial_state_seed=7,
        initial_state_id="reaction-to-purification:default",
        parameter_profile="downstream_processing",
        allowed_module_tags=("reaction", "thermal", "phase_partition", "separation", "observation"),
        expected_qualitative_behavior=(
            "reaction quality constrains purification ceiling",
            "phase split trades purity against recovery",
            "wash and concentration can improve purity but increase loss and cost",
        ),
    ),
    "reaction-to-crystallization": ScenarioSpec(
        scenario_id="reaction-to-crystallization",
        world_law_id=WORLD_LAW_ID,
        family="reaction_crystallization",
        split="public-test",
        difficulty="standard",
        hidden_parameter_seed=53,
        initial_state_seed=29,
        initial_state_id="reaction-to-crystallization:default",
        parameter_profile="seeded_crystallization",
        allowed_module_tags=("reaction", "thermal", "crystallization", "observation"),
        expected_qualitative_behavior=(
            "cooling and seeding improve crystallization yield",
            "over-aggressive cooling can occlude impurities",
            "filtration trades recovery against purity",
        ),
    ),
    "reaction-to-distillation": ScenarioSpec(
        scenario_id="reaction-to-distillation",
        world_law_id=WORLD_LAW_ID,
        family="reaction_distillation",
        split="public-test",
        difficulty="standard",
        hidden_parameter_seed=59,
        initial_state_seed=31,
        initial_state_id="reaction-to-distillation:default",
        parameter_profile="distillation_recovery",
        allowed_module_tags=("reaction", "thermal", "distillation", "observation"),
        expected_qualitative_behavior=(
            "reflux improves purity but increases cost",
            "high temperature improves recovery while increasing safety cost",
            "fraction collection controls isolated recovery",
        ),
    ),
    "flow-reaction-optimization": ScenarioSpec(
        scenario_id="flow-reaction-optimization",
        world_law_id=WORLD_LAW_ID,
        family="continuous_flow",
        split="public-test",
        difficulty="standard",
        hidden_parameter_seed=61,
        initial_state_seed=37,
        initial_state_id="flow-reaction-optimization:default",
        parameter_profile="continuous_flow_projection",
        allowed_module_tags=("reaction", "thermal", "continuous_flow", "observation"),
        expected_qualitative_behavior=(
            "residence time controls conversion",
            "flow operation can reduce peak batch risk",
            "temperature still affects desired and undesired reaction rates",
        ),
    ),
    "electrochemical-conversion": ScenarioSpec(
        scenario_id="electrochemical-conversion",
        world_law_id=WORLD_LAW_ID,
        family="electrochemistry",
        split="public-test",
        difficulty="standard",
        hidden_parameter_seed=67,
        initial_state_seed=41,
        initial_state_id="electrochemical-conversion:default",
        parameter_profile="electrochemical_selectivity",
        allowed_module_tags=("reaction", "electrochemistry", "observation"),
        expected_qualitative_behavior=(
            "potential controls selectivity",
            "current and time control conversion",
            "excess electrical input lowers energy efficiency and raises risk",
        ),
    ),
    "equilibrium-characterization": ScenarioSpec(
        scenario_id="equilibrium-characterization",
        world_law_id=WORLD_LAW_ID,
        family="equilibrium_characterization",
        split="public-test",
        difficulty="standard",
        hidden_parameter_seed=71,
        initial_state_seed=43,
        initial_state_id="equilibrium-characterization:default",
        parameter_profile="d4_equilibrium_characterization",
        allowed_module_tags=("equilibrium_chemistry", "observation", "instrumentation"),
        expected_qualitative_behavior=(
            "pH-meter observations reveal acid/base strength without exposing hidden constants",
            "precipitation signal appears only when public ion-product proxy crosses Ksp",
            "final assay reports equilibrium residual and confidence for scoring",
        ),
    ),
    "partition-discovery": ScenarioSpec(
        scenario_id="partition-discovery",
        world_law_id=WORLD_LAW_ID,
        family="partition",
        split="public-test",
        difficulty="standard",
        hidden_parameter_seed=31,
        initial_state_seed=11,
        initial_state_id="partition-discovery:default",
        parameter_profile="unknown_partition_coefficients",
        allowed_module_tags=("phase_partition", "separation", "observation"),
        expected_qualitative_behavior=(
            "extractant and solvent choices control product partition",
            "settling and entrainment affect mass balance",
        ),
    ),
    "purity-yield-tradeoff": ScenarioSpec(
        scenario_id="purity-yield-tradeoff",
        world_law_id=WORLD_LAW_ID,
        family="reaction_separation",
        split="public-test",
        difficulty="hard",
        hidden_parameter_seed=37,
        initial_state_seed=13,
        initial_state_id="purity-yield-tradeoff:default",
        parameter_profile="purity_recovery_tradeoff",
        allowed_module_tags=("reaction", "thermal", "phase_partition", "separation", "observation"),
        expected_qualitative_behavior=(
            "highest conversion does not imply highest isolated-purity score",
            "aggressive separation can sacrifice recovery",
        ),
    ),
    "generalization": ScenarioSpec(
        scenario_id="generalization",
        world_law_id=WORLD_LAW_ID,
        family="reaction",
        split="private-eval",
        difficulty="hard",
        hidden_parameter_seed=101,
        initial_state_seed=17,
        initial_state_id="generalization:default",
        parameter_profile="private_shifted_hidden_kinetics",
        allowed_module_tags=("reaction", "thermal", "observation"),
        expected_qualitative_behavior=("public optimum may not transfer exactly to hidden world",),
    ),
    "low-budget-characterization": ScenarioSpec(
        scenario_id="low-budget-characterization",
        world_law_id=WORLD_LAW_ID,
        family="reaction",
        split="public-test",
        difficulty="hard",
        hidden_parameter_seed=43,
        initial_state_seed=19,
        initial_state_id="low-budget-characterization:default",
        parameter_profile="sparse_measurement",
        allowed_module_tags=("reaction", "thermal", "observation"),
        expected_qualitative_behavior=("instrument choice matters under tight budget",),
    ),
    "tool-agent-planning": ScenarioSpec(
        scenario_id="tool-agent-planning",
        world_law_id=WORLD_LAW_ID,
        family="reaction_separation",
        split="public-dev",
        difficulty="standard",
        hidden_parameter_seed=47,
        initial_state_seed=23,
        initial_state_id="tool-agent-planning:default",
        parameter_profile="operation_language_planning",
        allowed_module_tags=("reaction", "thermal", "phase_partition", "separation", "observation"),
        expected_qualitative_behavior=("valid operation sequencing is part of the task",),
    ),
}


def list_scenarios() -> list[ScenarioSpec]:
    return [SCENARIO_REGISTRY[key] for key in sorted(SCENARIO_REGISTRY)]


def get_scenario(scenario_id: str, *, split: str | None = None) -> ScenarioSpec:
    try:
        spec = SCENARIO_REGISTRY[scenario_id]
    except KeyError as exc:
        available = ", ".join(sorted(SCENARIO_REGISTRY))
        raise KeyError(f"Unknown scenario_id={scenario_id!r}. Available: {available}") from exc
    if split is None or split == spec.split:
        return spec
    return ScenarioSpec(
        scenario_id=spec.scenario_id,
        world_law_id=spec.world_law_id,
        family=spec.family,
        split=split,
        difficulty=spec.difficulty,
        hidden_parameter_seed=spec.hidden_parameter_seed,
        initial_state_seed=spec.initial_state_seed,
        initial_state_id=spec.initial_state_id,
        parameter_profile=spec.parameter_profile,
        allowed_module_tags=spec.allowed_module_tags,
        expected_qualitative_behavior=spec.expected_qualitative_behavior,
    )


def get_scenario_card(scenario_id: str, *, split: str | None = None) -> dict[str, object]:
    spec = get_scenario(scenario_id, split=split)
    family = SCENARIO_FAMILIES[spec.family]
    return {
        **spec.to_dict(),
        "family_card": family.to_dict(),
        "generator": "DefaultScenarioGenerator",
    }


__all__ = [
    "SCENARIO_FAMILIES",
    "SCENARIO_REGISTRY",
    "DefaultScenarioGenerator",
    "ScenarioFamilySpec",
    "ScenarioGenerator",
    "ScenarioInstance",
    "ScenarioSpec",
    "get_scenario",
    "get_scenario_card",
    "list_scenarios",
]
