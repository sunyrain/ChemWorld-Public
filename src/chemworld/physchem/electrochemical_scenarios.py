"""Standalone electrochemical reference scenarios, not ChemWorld task runtime."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from math import isfinite, log10
from random import Random

from chemworld._serialization import canonical_json_sha256
from chemworld.physchem.electrochem_double_layer import DoubleLayerRCSpec
from chemworld.physchem.electrochem_transport import DiffusionLayerSpec
from chemworld.physchem.electrochemistry import (
    ElectrodeReactionSpec,
    ElectrolyteResistanceSpec,
)

SUPPORTED_ELECTROCHEMICAL_SPLITS = ("public-dev", "public-test", "private-eval")
ELECTROCHEMICAL_SCENARIO_EXECUTION_ROLE = (
    "standalone_reference_generator_not_electrochemical_conversion_runtime"
)
HIDDEN_PARAMETER_IDS = (
    "exchange_current_density_A_m2",
    "electrolyte_conductivity_S_m",
    "contact_resistance_ohm",
    "diffusivity_m2_s",
    "diffusion_layer_thickness_m",
    "double_layer_capacitance_F_m2",
    "charge_transfer_resistance_ohm",
    "faradaic_efficiency_ref",
    "product_selectivity_ref",
)


@dataclass(frozen=True)
class HiddenParameterRange:
    minimum: float
    maximum: float
    distribution: str = "uniform"

    def __post_init__(self) -> None:
        if self.minimum <= 0.0 or self.maximum <= self.minimum:
            raise ValueError("hidden parameter range must be positive and increasing")
        if not isfinite(self.minimum) or not isfinite(self.maximum):
            raise ValueError("hidden parameter range must be finite")
        if self.distribution not in {"uniform", "log_uniform"}:
            raise ValueError("hidden parameter distribution must be uniform or log_uniform")

    def sample(self, rng: Random) -> float:
        if self.distribution == "log_uniform":
            return 10.0 ** rng.uniform(log10(self.minimum), log10(self.maximum))
        return rng.uniform(self.minimum, self.maximum)

    def to_dict(self) -> dict[str, object]:
        return {
            "minimum": self.minimum,
            "maximum": self.maximum,
            "distribution": self.distribution,
        }


@dataclass(frozen=True)
class RedoxMetadata:
    reaction_id: str
    reactant_id: str
    product_id: str
    electrons_transferred: float
    standard_potential_V: float
    reaction_quotient_exponents: dict[str, float]

    def __post_init__(self) -> None:
        if not self.reaction_id or not self.reactant_id or not self.product_id:
            raise ValueError("redox ids cannot be empty")
        if self.reactant_id == self.product_id:
            raise ValueError("redox reactant and product must be distinct")
        _positive(self.electrons_transferred, "electrons_transferred")
        _finite(self.standard_potential_V, "standard_potential_V")
        if set(self.reaction_quotient_exponents) != {
            self.reactant_id,
            self.product_id,
        }:
            raise ValueError("reaction quotient exponents must match reactant and product")
        if any(
            value == 0.0 or not isfinite(value)
            for value in self.reaction_quotient_exponents.values()
        ):
            raise ValueError("reaction quotient exponents must be finite and nonzero")

    def to_dict(self) -> dict[str, object]:
        return {
            "reaction_id": self.reaction_id,
            "reactant_id": self.reactant_id,
            "product_id": self.product_id,
            "electrons_transferred": self.electrons_transferred,
            "standard_potential_V": self.standard_potential_V,
            "reaction_quotient_exponents": dict(self.reaction_quotient_exponents),
        }


@dataclass(frozen=True)
class ElectrochemicalScenarioCard:
    scenario_id: str
    title: str
    redox: RedoxMetadata
    electrode_area_m2: float
    electrode_gap_m: float
    electrolyte_volume_m3: float
    electrolyte_window_V: tuple[float, float]
    cathodic_side_reaction_onset_V: float
    anodic_side_reaction_onset_V: float
    hidden_parameter_ranges: dict[str, HiddenParameterRange]
    qualitative_behavior: tuple[str, ...]
    provenance_id: str
    schema_version: str = "chemworld-electrochemical-scenario-card-0.1"

    def __post_init__(self) -> None:
        if not self.scenario_id or not self.title or not self.provenance_id:
            raise ValueError("scenario id, title, and provenance cannot be empty")
        if self.schema_version != "chemworld-electrochemical-scenario-card-0.1":
            raise ValueError("unsupported electrochemical scenario-card schema")
        for name, value in (
            ("electrode_area_m2", self.electrode_area_m2),
            ("electrode_gap_m", self.electrode_gap_m),
            ("electrolyte_volume_m3", self.electrolyte_volume_m3),
        ):
            _positive(value, name)
        window_low, window_high = self.electrolyte_window_V
        if not window_low < window_high:
            raise ValueError("electrolyte window must be increasing")
        if not (
            window_low
            < self.cathodic_side_reaction_onset_V
            < self.redox.standard_potential_V
            < self.anodic_side_reaction_onset_V
            < window_high
        ):
            raise ValueError("side-reaction onsets must bracket E0 inside the electrolyte window")
        if set(self.hidden_parameter_ranges) != set(HIDDEN_PARAMETER_IDS):
            raise ValueError("hidden parameter ranges must match the required parameter ids")
        for fraction_id in ("faradaic_efficiency_ref", "product_selectivity_ref"):
            if self.hidden_parameter_ranges[fraction_id].maximum > 1.0:
                raise ValueError(f"{fraction_id} hidden range cannot exceed one")
        if not self.qualitative_behavior:
            raise ValueError("qualitative_behavior cannot be empty")

    def to_public_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "scenario_id": self.scenario_id,
            "execution_role": ELECTROCHEMICAL_SCENARIO_EXECUTION_ROLE,
            "title": self.title,
            "redox": self.redox.to_dict(),
            "electrode_area_m2": self.electrode_area_m2,
            "electrode_gap_m": self.electrode_gap_m,
            "electrolyte_volume_m3": self.electrolyte_volume_m3,
            "electrolyte_window_V": list(self.electrolyte_window_V),
            "cathodic_side_reaction_onset_V": (self.cathodic_side_reaction_onset_V),
            "anodic_side_reaction_onset_V": self.anodic_side_reaction_onset_V,
            "hidden_parameter_policy": {
                parameter_id: {"distribution": parameter_range.distribution}
                for parameter_id, parameter_range in self.hidden_parameter_ranges.items()
            },
            "qualitative_behavior": list(self.qualitative_behavior),
            "provenance_id": self.provenance_id,
        }

    def to_private_dict(self) -> dict[str, object]:
        return {
            **self.to_public_dict(),
            "hidden_parameter_ranges": {
                key: value.to_dict() for key, value in self.hidden_parameter_ranges.items()
            },
        }


@dataclass(frozen=True)
class ElectrochemicalHiddenParameters:
    exchange_current_density_A_m2: float
    electrolyte_conductivity_S_m: float
    contact_resistance_ohm: float
    diffusivity_m2_s: float
    diffusion_layer_thickness_m: float
    double_layer_capacitance_F_m2: float
    charge_transfer_resistance_ohm: float
    faradaic_efficiency_ref: float
    product_selectivity_ref: float

    def to_dict(self) -> dict[str, float]:
        return {field_name: float(getattr(self, field_name)) for field_name in HIDDEN_PARAMETER_IDS}


@dataclass(frozen=True)
class ElectrochemicalModelBundle:
    reaction: ElectrodeReactionSpec
    electrolyte_resistance: ElectrolyteResistanceSpec
    diffusion_layer: DiffusionLayerSpec
    double_layer: DoubleLayerRCSpec

    def to_dict(self) -> dict[str, object]:
        return {
            "reaction": self.reaction.to_dict(),
            "electrolyte_resistance": self.electrolyte_resistance.to_dict(),
            "diffusion_layer": self.diffusion_layer.to_dict(),
            "double_layer": self.double_layer.to_dict(),
        }


@dataclass(frozen=True)
class ElectrochemicalScenarioInstance:
    scenario_id: str
    split: str
    seed: int
    world_id: str
    hidden_parameter_digest: str
    hidden_parameters: ElectrochemicalHiddenParameters
    card: ElectrochemicalScenarioCard

    def build_model_bundle(self) -> ElectrochemicalModelBundle:
        hidden = self.hidden_parameters
        window_low, window_high = self.card.electrolyte_window_V
        reaction = ElectrodeReactionSpec(
            reaction_id=self.card.redox.reaction_id,
            electrons_transferred=self.card.redox.electrons_transferred,
            standard_potential_V=self.card.redox.standard_potential_V,
            reaction_quotient_exponents=dict(self.card.redox.reaction_quotient_exponents),
            exchange_current_density_A_m2=hidden.exchange_current_density_A_m2,
            electrode_area_m2=self.card.electrode_area_m2,
            faradaic_efficiency_ref=hidden.faradaic_efficiency_ref,
            product_selectivity_ref=hidden.product_selectivity_ref,
            metadata={"scenario_id": self.scenario_id, "world_id": self.world_id},
        )
        resistance = ElectrolyteResistanceSpec(
            electrolyte_conductivity_S_m=hidden.electrolyte_conductivity_S_m,
            electrode_gap_m=self.card.electrode_gap_m,
            electrode_area_m2=self.card.electrode_area_m2,
            contact_resistance_ohm=hidden.contact_resistance_ohm,
            voltage_window_V=max(abs(window_low), abs(window_high)),
            metadata={"scenario_id": self.scenario_id},
        )
        diffusion = DiffusionLayerSpec(
            model_id=f"{self.scenario_id}:diffusion-layer",
            electrons_transferred=self.card.redox.electrons_transferred,
            electrode_area_m2=self.card.electrode_area_m2,
            diffusivity_m2_s=hidden.diffusivity_m2_s,
            diffusion_layer_thickness_m=hidden.diffusion_layer_thickness_m,
            electrolyte_volume_m3=self.card.electrolyte_volume_m3,
            provenance_id=self.hidden_parameter_digest,
        )
        double_layer = DoubleLayerRCSpec(
            model_id=f"{self.scenario_id}:double-layer",
            double_layer_capacitance_F_m2=(hidden.double_layer_capacitance_F_m2),
            electrode_area_m2=self.card.electrode_area_m2,
            series_resistance_ohm=resistance.total_resistance_ohm,
            charge_transfer_resistance_ohm=(hidden.charge_transfer_resistance_ohm),
            provenance_id=self.hidden_parameter_digest,
        )
        return ElectrochemicalModelBundle(
            reaction=reaction,
            electrolyte_resistance=resistance,
            diffusion_layer=diffusion,
            double_layer=double_layer,
        )

    def to_public_dict(self) -> dict[str, object]:
        return {
            "scenario_id": self.scenario_id,
            "split": self.split,
            "seed": self.seed,
            "world_id": self.world_id,
            "hidden_parameter_digest": self.hidden_parameter_digest,
            "card": self.card.to_public_dict(),
        }


@dataclass(frozen=True)
class SideReactionAssessment:
    potential_V: float
    status: str
    cathodic_side_reaction: bool
    anodic_side_reaction: bool
    severity: float
    flags: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "potential_V": self.potential_V,
            "status": self.status,
            "cathodic_side_reaction": self.cathodic_side_reaction,
            "anodic_side_reaction": self.anodic_side_reaction,
            "severity": self.severity,
            "flags": list(self.flags),
        }


def generate_electrochemical_scenario(
    card: ElectrochemicalScenarioCard,
    *,
    split: str,
    seed: int,
    private_salt: str = "",
) -> ElectrochemicalScenarioInstance:
    if split not in SUPPORTED_ELECTROCHEMICAL_SPLITS:
        raise ValueError("unsupported electrochemical scenario split")
    if split == "private-eval" and not private_salt:
        raise ValueError("private-eval electrochemical scenarios require private_salt")
    seed_payload = f"{card.schema_version}:{card.scenario_id}:{split}:{seed}:{private_salt}"
    seed_digest = hashlib.sha256(seed_payload.encode("utf-8")).digest()
    rng = Random(int.from_bytes(seed_digest[:8], "big"))
    sampled = {
        parameter_id: card.hidden_parameter_ranges[parameter_id].sample(rng)
        for parameter_id in HIDDEN_PARAMETER_IDS
    }
    hidden = ElectrochemicalHiddenParameters(**sampled)
    hidden_digest = _sha256(hidden.to_dict())
    world_suffix = hashlib.sha256(seed_payload.encode("utf-8")).hexdigest()[:16]
    return ElectrochemicalScenarioInstance(
        scenario_id=card.scenario_id,
        split=split,
        seed=seed,
        world_id=f"ElectrochemicalScenario:{split}:{world_suffix}",
        hidden_parameter_digest=hidden_digest,
        hidden_parameters=hidden,
        card=card,
    )


def assess_side_reaction_thresholds(
    card: ElectrochemicalScenarioCard,
    *,
    potential_V: float,
) -> SideReactionAssessment:
    _finite(potential_V, "potential_V")
    window_low, window_high = card.electrolyte_window_V
    cathodic = potential_V <= card.cathodic_side_reaction_onset_V
    anodic = potential_V >= card.anodic_side_reaction_onset_V
    flags: list[str] = []
    if cathodic:
        flags.append("cathodic_side_reaction_threshold_crossed")
        severity = min(
            max(
                (card.cathodic_side_reaction_onset_V - potential_V)
                / (card.cathodic_side_reaction_onset_V - window_low),
                0.0,
            ),
            1.0,
        )
    elif anodic:
        flags.append("anodic_side_reaction_threshold_crossed")
        severity = min(
            max(
                (potential_V - card.anodic_side_reaction_onset_V)
                / (window_high - card.anodic_side_reaction_onset_V),
                0.0,
            ),
            1.0,
        )
    else:
        severity = 0.0
    if potential_V < window_low or potential_V > window_high:
        flags.append("electrolyte_window_exceeded")
        severity = 1.0
    return SideReactionAssessment(
        potential_V=potential_V,
        status="window_exceeded"
        if "electrolyte_window_exceeded" in flags
        else "side_reaction_risk"
        if cathodic or anodic
        else "inside_selective_window",
        cathodic_side_reaction=cathodic,
        anodic_side_reaction=anodic,
        severity=severity,
        flags=tuple(flags),
    )


def electrochemical_scenario_cards() -> tuple[ElectrochemicalScenarioCard, ...]:
    ranges = _default_hidden_ranges()
    return (
        ElectrochemicalScenarioCard(
            scenario_id="aqueous_selective_reduction",
            title="Aqueous Selective Two-Electron Reduction",
            redox=RedoxMetadata(
                reaction_id="substrate_to_reduced_product",
                reactant_id="substrate",
                product_id="reduced_product",
                electrons_transferred=2.0,
                standard_potential_V=-0.20,
                reaction_quotient_exponents={
                    "reduced_product": 1.0,
                    "substrate": -1.0,
                },
            ),
            electrode_area_m2=0.01,
            electrode_gap_m=0.002,
            electrolyte_volume_m3=1.0e-4,
            electrolyte_window_V=(-1.20, 1.00),
            cathodic_side_reaction_onset_V=-0.90,
            anodic_side_reaction_onset_V=0.75,
            hidden_parameter_ranges=dict(ranges),
            qualitative_behavior=(
                "moderate cathodic potential favors the tracked reduction",
                "more negative potential approaches hydrogen-evolution risk",
                "high current encounters diffusion limitation and startup charging",
            ),
            provenance_id="chemworld-curated-electrochemical-scenarios-v1",
        ),
        ElectrochemicalScenarioCard(
            scenario_id="organic_anodic_coupling",
            title="Organic-Electrolyte Anodic Coupling",
            redox=RedoxMetadata(
                reaction_id="substrate_to_coupled_product",
                reactant_id="substrate",
                product_id="coupled_product",
                electrons_transferred=1.0,
                standard_potential_V=0.65,
                reaction_quotient_exponents={
                    "coupled_product": 1.0,
                    "substrate": -1.0,
                },
            ),
            electrode_area_m2=0.008,
            electrode_gap_m=0.0015,
            electrolyte_volume_m3=8.0e-5,
            electrolyte_window_V=(-0.50, 1.50),
            cathodic_side_reaction_onset_V=-0.25,
            anodic_side_reaction_onset_V=1.20,
            hidden_parameter_ranges=dict(ranges),
            qualitative_behavior=(
                "anodic potential above E0 drives coupling",
                "high potential approaches solvent oxidation onset",
                "ohmic and double-layer parameters vary across benchmark splits",
            ),
            provenance_id="chemworld-curated-electrochemical-scenarios-v1",
        ),
    )


def _default_hidden_ranges() -> dict[str, HiddenParameterRange]:
    return {
        "exchange_current_density_A_m2": HiddenParameterRange(2.0, 50.0, "log_uniform"),
        "electrolyte_conductivity_S_m": HiddenParameterRange(0.5, 12.0, "log_uniform"),
        "contact_resistance_ohm": HiddenParameterRange(0.01, 0.20, "log_uniform"),
        "diffusivity_m2_s": HiddenParameterRange(2.0e-10, 2.0e-9, "log_uniform"),
        "diffusion_layer_thickness_m": HiddenParameterRange(25.0e-6, 250.0e-6, "log_uniform"),
        "double_layer_capacitance_F_m2": HiddenParameterRange(0.05, 0.40, "log_uniform"),
        "charge_transfer_resistance_ohm": HiddenParameterRange(5.0, 250.0, "log_uniform"),
        "faradaic_efficiency_ref": HiddenParameterRange(0.72, 0.98),
        "product_selectivity_ref": HiddenParameterRange(0.68, 0.96),
    }


def _sha256(payload: object) -> str:
    return canonical_json_sha256(payload, ensure_ascii=True)


def _positive(value: float, field_name: str) -> None:
    if value <= 0.0 or not isfinite(value):
        raise ValueError(f"{field_name} must be positive and finite")


def _finite(value: float, field_name: str) -> None:
    if not isfinite(value):
        raise ValueError(f"{field_name} must be finite")


__all__ = [
    "ELECTROCHEMICAL_SCENARIO_EXECUTION_ROLE",
    "ElectrochemicalHiddenParameters",
    "ElectrochemicalModelBundle",
    "ElectrochemicalScenarioCard",
    "ElectrochemicalScenarioInstance",
    "HiddenParameterRange",
    "RedoxMetadata",
    "SideReactionAssessment",
    "assess_side_reaction_thresholds",
    "electrochemical_scenario_cards",
    "generate_electrochemical_scenario",
]
