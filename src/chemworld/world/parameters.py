"""World parameter generation and split handling."""

from __future__ import annotations

import os
from dataclasses import dataclass
from hashlib import sha256

import numpy as np

from chemworld.world.actions import CATALYSTS, SOLVENTS
from chemworld.world.crystallization_material_family import (
    REACTION_CRYSTALLIZATION_LATENT_MATERIAL_FAMILY,
    crystallization_material_family,
)
from chemworld.world.electrochemical_material_family import (
    NOMINAL_PRIOR_MATERIAL_FAMILY,
    ElectrochemicalResidualGeneratorContract,
    electrochemical_material_family,
)

WORLD_FAMILY_VERSION = "chemworld-physical-chemistry-v0.5"
SUPPORTED_SPLITS = ("public-dev", "public-test", "private-eval")

# Public family-level activity profiles used to generate the generic reaction
# material effects. World-specific residual multipliers are sampled below and
# remain hidden. Keeping the nominal profiles explicit lets participant-facing
# prior dossiers stay synchronized with the executable generator without
# reading a realized world instance.
REACTION_NOMINAL_CATALYST_ACTIVITY_PROFILES = (
    (1.00, 1.05, 0.92, 0.95, 0.88),
    (1.30, 0.92, 1.15, 1.08, 1.10),
    (0.82, 1.32, 0.90, 1.18, 0.94),
    (1.10, 0.86, 1.22, 0.90, 1.20),
)
REACTION_NOMINAL_SOLVENT_ACTIVITY_PROFILES = (
    (0.75, 0.72, 0.68, 0.70, 0.65),
    (0.96, 1.02, 1.00, 0.95, 1.05),
    (1.20, 0.98, 1.12, 1.15, 0.98),
    (1.05, 1.34, 1.28, 1.25, 1.18),
)

DEFAULT_DOMAIN_PARAMETERS: dict[str, float] = {
    "partition_coefficient_multiplier": 1.0,
    "partition_coefficient_exponent": 1.0,
    "partition_phase_volume_multiplier": 1.0,
    "crystallization_nucleation_multiplier": 1.0,
    "crystallization_growth_multiplier": 1.0,
    "crystallization_solubility_multiplier": 1.0,
    "distillation_relative_volatility_multiplier": 1.0,
    "flow_rate_multiplier": 1.0,
    "flow_residence_multiplier": 1.0,
    "flow_boundary_ua_multiplier": 1.0,
    "electro_exchange_current_multiplier": 1.0,
    "electro_resistance_multiplier": 1.0,
    "electro_selectivity_decay_multiplier": 1.0,
    "electro_standard_potential_multiplier": 1.0,
    "electro_transfer_asymmetry_multiplier": 1.0,
    "observation_noise_multiplier": 1.0,
}


@dataclass(frozen=True)
class ChemWorldParameters:
    world_id: str
    split: str
    provider: str
    family_version: str
    pre_exponential: np.ndarray
    activation_energy: np.ndarray
    catalyst_effects: np.ndarray
    solvent_effects: np.ndarray
    crystallization_catalyst_effects: np.ndarray
    crystallization_solvent_effects: np.ndarray
    crystallization_solvent_solubility_multipliers: np.ndarray
    crystallization_solvent_nucleation_multipliers: np.ndarray
    crystallization_solvent_growth_multipliers: np.ndarray
    crystallization_solvent_occlusion_multipliers: np.ndarray
    electrochemical_electrolyte_effects: np.ndarray
    electrochemical_solvent_effects: np.ndarray
    electrochemical_electrolyte_potential_residual_V: np.ndarray
    electrochemical_solvent_potential_residual_V: np.ndarray
    electrochemical_electrode_gap_m: float
    electrochemical_electrode_area_m2: float
    electrochemical_base_contact_resistance_ohm: float
    electrochemical_exchange_current_density_A_m2: float
    solvent_risks: np.ndarray
    solvent_costs: np.ndarray
    catalyst_costs: np.ndarray
    delta_h_J_per_mol: np.ndarray
    ua_W_per_K: float
    rho_cp_J_per_L_K: float
    environment_temperature_K: float
    crystallization_reference_solubility_mol_L: float
    domain_parameters: dict[str, float]

    def __post_init__(self) -> None:
        domain_parameters = {
            str(key): float(value) for key, value in self.domain_parameters.items()
        }
        missing = sorted(set(DEFAULT_DOMAIN_PARAMETERS) - set(domain_parameters))
        unknown = sorted(set(domain_parameters) - set(DEFAULT_DOMAIN_PARAMETERS))
        invalid = sorted(
            key
            for key, value in domain_parameters.items()
            if not np.isfinite(value) or value <= 0.0
        )
        if missing or unknown or invalid:
            raise ValueError(
                "invalid domain parameters: "
                f"missing={missing}, unknown={unknown}, nonpositive_or_nonfinite={invalid}"
            )
        object.__setattr__(self, "domain_parameters", domain_parameters)
        if (
            not np.isfinite(self.crystallization_reference_solubility_mol_L)
            or self.crystallization_reference_solubility_mol_L <= 0.0
        ):
            raise ValueError("crystallization reference solubility must be positive and finite")
        for field_name in (
            "crystallization_catalyst_effects",
            "crystallization_solvent_effects",
        ):
            values = np.asarray(getattr(self, field_name), dtype=float)
            if values.shape != (4, 5) or not np.all(np.isfinite(values)) or np.any(values <= 0.0):
                raise ValueError(f"{field_name} must be a finite positive 4x5 matrix")
        for field_name in (
            "crystallization_solvent_solubility_multipliers",
            "crystallization_solvent_nucleation_multipliers",
            "crystallization_solvent_growth_multipliers",
            "crystallization_solvent_occlusion_multipliers",
        ):
            values = np.asarray(getattr(self, field_name), dtype=float)
            if values.shape != (4,) or not np.all(np.isfinite(values)) or np.any(values <= 0.0):
                raise ValueError(f"{field_name} must be a finite positive four-element vector")
        for field_name in (
            "electrochemical_electrolyte_effects",
            "electrochemical_solvent_effects",
        ):
            values = np.asarray(getattr(self, field_name), dtype=float)
            if values.shape != (4, 7) or not np.all(np.isfinite(values)) or np.any(values <= 0.0):
                raise ValueError(f"{field_name} must be a finite positive 4x7 matrix")
        for field_name in (
            "electrochemical_electrolyte_potential_residual_V",
            "electrochemical_solvent_potential_residual_V",
        ):
            values = np.asarray(getattr(self, field_name), dtype=float)
            if values.shape != (4,) or not np.all(np.isfinite(values)):
                raise ValueError(f"{field_name} must be a finite four-element vector")
        for field_name in (
            "electrochemical_electrode_gap_m",
            "electrochemical_electrode_area_m2",
            "electrochemical_base_contact_resistance_ohm",
            "electrochemical_exchange_current_density_A_m2",
        ):
            value = float(getattr(self, field_name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{field_name} must be positive and finite")

    def domain_parameter(self, key: str) -> float:
        """Return a typed vNext provider parameter and fail on unknown keys."""

        try:
            return float(self.domain_parameters[key])
        except KeyError as exc:
            raise KeyError(f"Unknown domain parameter: {key}") from exc


def stable_parameter_seed(split: str, seed: int, private_salt: str = "") -> int:
    digest = sha256(f"{WORLD_FAMILY_VERSION}:{split}:{seed}:{private_salt}".encode()).digest()
    return int.from_bytes(digest[:8], "little") % (2**32)


def _latent_material_effects(
    rng: np.random.Generator,
    contract: ElectrochemicalResidualGeneratorContract,
    loadings: tuple[tuple[float, ...], ...],
) -> tuple[np.ndarray, np.ndarray]:
    latent = rng.normal(
        loc=0.0,
        scale=np.asarray(contract.latent_sigma, dtype=float),
        size=(4, len(contract.latent_factor_names)),
    )
    loading_matrix = np.asarray(loadings, dtype=float)
    if loading_matrix.shape != (
        len(contract.property_names),
        len(contract.latent_factor_names),
    ):
        raise RuntimeError("electrochemical residual factor loading matrix is malformed")
    independent = rng.normal(
        0.0,
        contract.independent_property_sigma,
        size=(4, len(contract.property_names)),
    )
    log_effects = latent @ loading_matrix.T + independent
    effects = np.clip(np.exp(log_effects), *contract.multiplier_bounds)
    potential = np.clip(
        latent @ np.asarray(contract.potential_loadings, dtype=float)
        + rng.normal(0.0, contract.potential_noise_sigma_V, size=4),
        *contract.potential_bounds_V,
    )
    return effects, potential


def load_chemworld_parameters(
    split: str = "public-dev",
    seed: int = 0,
) -> ChemWorldParameters:
    """Generate deterministic hidden world parameters for a split and seed."""

    if split not in SUPPORTED_SPLITS:
        allowed = ", ".join(SUPPORTED_SPLITS)
        raise ValueError(f"Unsupported world_split={split!r}. Allowed: {allowed}")

    private_salt = ""
    provider = "public-registry"
    if split == "private-eval":
        private_salt = os.environ.get("CHEMWORLD_PRIVATE_EVAL_SALT", "")
        provider = "external-private-registry" if private_salt else "public-placeholder-private"

    rng = np.random.default_rng(stable_parameter_seed(split, seed, private_salt))
    split_shift = {"public-dev": 0.0, "public-test": 0.06, "private-eval": -0.05}[split]
    pre_exponential = np.array([90.0, 190.0, 520.0, 65.0, 30.0])
    pre_exponential *= rng.lognormal(mean=split_shift, sigma=[0.10, 0.15, 0.18, 0.18, 0.14])
    activation_energy = np.array([31_000.0, 38_500.0, 45_000.0, 42_000.0, 36_000.0])
    activation_energy *= rng.lognormal(mean=0.0, sigma=[0.03, 0.05, 0.06, 0.06, 0.05])

    catalyst_effects = rng.lognormal(mean=0.0, sigma=0.22, size=(len(CATALYSTS), 5))
    catalyst_effects *= np.asarray(
        REACTION_NOMINAL_CATALYST_ACTIVITY_PROFILES,
        dtype=float,
    )

    solvent_effects = rng.lognormal(mean=0.0, sigma=0.20, size=(len(SOLVENTS), 5))
    solvent_effects *= np.asarray(
        REACTION_NOMINAL_SOLVENT_ACTIVITY_PROFILES,
        dtype=float,
    )

    crystallization_family = crystallization_material_family(
        REACTION_CRYSTALLIZATION_LATENT_MATERIAL_FAMILY
    )
    crystallization_residual = crystallization_family.residual_generator
    if crystallization_residual is None:
        raise RuntimeError("crystallization material family lacks a residual generator")
    crystallization_seed = stable_parameter_seed(
        split,
        seed,
        f"{private_salt}:{crystallization_residual.seed_namespace}",
    )
    crystallization_rng = np.random.default_rng(crystallization_seed)
    residual_bounds = crystallization_residual.residual_multiplier_bounds
    catalyst_nominal = np.asarray(
        [row["reaction_multipliers"] for row in crystallization_family.catalyst_profiles],
        dtype=float,
    )
    solvent_nominal = np.asarray(
        [row["reaction_multipliers"] for row in crystallization_family.solvent_profiles],
        dtype=float,
    )
    crystallization_catalyst_effects = catalyst_nominal * np.clip(
        crystallization_rng.lognormal(
            mean=0.0,
            sigma=crystallization_residual.reaction_log_sigma,
            size=catalyst_nominal.shape,
        ),
        *residual_bounds,
    )
    crystallization_solvent_effects = solvent_nominal * np.clip(
        crystallization_rng.lognormal(
            mean=0.0,
            sigma=crystallization_residual.reaction_log_sigma,
            size=solvent_nominal.shape,
        ),
        *residual_bounds,
    )

    def crystallization_profile_vector(field: str) -> np.ndarray:
        nominal = np.asarray(
            [row[field] for row in crystallization_family.solvent_profiles],
            dtype=float,
        )
        residual = np.clip(
            crystallization_rng.lognormal(
                mean=0.0,
                sigma=crystallization_residual.crystallization_log_sigma,
                size=nominal.shape,
            ),
            *residual_bounds,
        )
        return nominal * residual

    crystallization_solvent_solubility_multipliers = crystallization_profile_vector(
        "solubility_multiplier"
    )
    crystallization_solvent_nucleation_multipliers = crystallization_profile_vector(
        "nucleation_multiplier"
    )
    crystallization_solvent_growth_multipliers = crystallization_profile_vector("growth_multiplier")
    crystallization_solvent_occlusion_multipliers = crystallization_profile_vector(
        "impurity_occlusion_multiplier"
    )

    material_family = electrochemical_material_family(NOMINAL_PRIOR_MATERIAL_FAMILY)
    residual_contract = material_family.residual_generator
    if residual_contract is None:
        raise RuntimeError("nominal-prior material family lacks a residual generator contract")
    material_seed = stable_parameter_seed(
        split,
        seed,
        f"{private_salt}:{residual_contract.seed_namespace}",
    )
    material_rng = np.random.default_rng(material_seed)
    (
        electrochemical_electrolyte_effects,
        electrochemical_electrolyte_potential_residual_V,
    ) = _latent_material_effects(
        material_rng,
        residual_contract,
        residual_contract.electrolyte_loadings,
    )
    (
        electrochemical_solvent_effects,
        electrochemical_solvent_potential_residual_V,
    ) = _latent_material_effects(
        material_rng,
        residual_contract,
        residual_contract.solvent_loadings,
    )
    geometry_bounds = material_family.world_fixed_cell_geometry_bounds
    if geometry_bounds is None:
        raise RuntimeError("nominal-prior material family lacks geometry bounds")
    electrochemical_electrode_gap_m = float(material_rng.uniform(*geometry_bounds.electrode_gap_m))
    electrochemical_electrode_area_m2 = float(
        material_rng.uniform(*geometry_bounds.electrode_area_m2)
    )
    electrochemical_base_contact_resistance_ohm = float(
        material_rng.uniform(*geometry_bounds.base_contact_resistance_ohm)
    )
    electrochemical_exchange_current_density_A_m2 = float(
        np.clip(
            residual_contract.exchange_current_reference_A_m2
            * material_rng.lognormal(0.0, residual_contract.exchange_current_log_sigma),
            *residual_contract.exchange_current_bounds_A_m2,
        )
    )

    provider_label = "external" if provider == "external-private-registry" else "public"
    world_id = f"ChemWorld:{split}:{provider_label}:seed-{seed}"
    return ChemWorldParameters(
        world_id=world_id,
        split=split,
        provider=provider,
        family_version=WORLD_FAMILY_VERSION,
        pre_exponential=pre_exponential,
        activation_energy=activation_energy,
        catalyst_effects=catalyst_effects,
        solvent_effects=solvent_effects,
        crystallization_catalyst_effects=crystallization_catalyst_effects,
        crystallization_solvent_effects=crystallization_solvent_effects,
        crystallization_solvent_solubility_multipliers=(
            crystallization_solvent_solubility_multipliers
        ),
        crystallization_solvent_nucleation_multipliers=(
            crystallization_solvent_nucleation_multipliers
        ),
        crystallization_solvent_growth_multipliers=(crystallization_solvent_growth_multipliers),
        crystallization_solvent_occlusion_multipliers=(
            crystallization_solvent_occlusion_multipliers
        ),
        electrochemical_electrolyte_effects=electrochemical_electrolyte_effects,
        electrochemical_solvent_effects=electrochemical_solvent_effects,
        electrochemical_electrolyte_potential_residual_V=(
            electrochemical_electrolyte_potential_residual_V
        ),
        electrochemical_solvent_potential_residual_V=(electrochemical_solvent_potential_residual_V),
        electrochemical_electrode_gap_m=electrochemical_electrode_gap_m,
        electrochemical_electrode_area_m2=electrochemical_electrode_area_m2,
        electrochemical_base_contact_resistance_ohm=(electrochemical_base_contact_resistance_ohm),
        electrochemical_exchange_current_density_A_m2=(
            electrochemical_exchange_current_density_A_m2
        ),
        solvent_risks=np.array([0.05, 0.18, 0.28, 0.35]),
        solvent_costs=np.array([0.03, 0.08, 0.16, 0.11]),
        catalyst_costs=np.array([0.08, 0.18, 0.12, 0.22]),
        delta_h_J_per_mol=np.array([-42_000.0, -25_000.0, -18_000.0, -35_000.0, -5_000.0]),
        ua_W_per_K=float(rng.uniform(0.05, 0.12)),
        rho_cp_J_per_L_K=float(rng.uniform(3800.0, 4300.0)),
        environment_temperature_K=298.15,
        crystallization_reference_solubility_mol_L=float(rng.uniform(0.085, 0.105)),
        domain_parameters=dict(DEFAULT_DOMAIN_PARAMETERS),
    )


__all__ = [
    "DEFAULT_DOMAIN_PARAMETERS",
    "SUPPORTED_SPLITS",
    "WORLD_FAMILY_VERSION",
    "ChemWorldParameters",
    "load_chemworld_parameters",
    "stable_parameter_seed",
]
