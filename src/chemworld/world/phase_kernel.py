"""Phase and partition-law module for ChemWorld."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict

import numpy as np

from chemworld.foundation.solvents import SolventInventory
from chemworld.physchem.extraction_units import DistributionCoefficientModelSpec
from chemworld.physchem.phase_equilibrium_units import (
    LLEContactorSpec,
    StabilityAwareExtractionRequest,
    simulate_stability_aware_extraction,
)

INDEPENDENT_NOMINAL_SOLVENT_EXTRACTANT_PAIR_V1 = "independent-nominal-solvent-extractant-pair-v1"

# These are explicit nominal interaction tables.  Row/column indices are opaque
# category labels: adjacency, equality across the two axes, and numeric distance
# have no scientific meaning.
#
# The product table is calibrated on the v3 fixed-initial-target inventory basis.
# The 1.35 factor restores a useful, non-degenerate recovery range after replacing
# the historical moving denominator: several selective pairs can reach the frozen
# benchmark threshold, while weak pairs and under-extracted settings remain below it.
PARTITION_V3_PRODUCT_DISTRIBUTION_CALIBRATION = 1.35
_NOMINAL_PRODUCT_DISTRIBUTION_COEFFICIENTS = np.asarray(
    (
        (1.20, 3.40, 2.00, 3.80),
        (2.80, 1.50, 3.60, 2.20),
        (3.20, 2.60, 1.80, 3.50),
        (2.40, 3.30, 3.00, 1.60),
    ),
    dtype=float,
)
_NOMINAL_IMPURITY_DISTRIBUTION_COEFFICIENTS = np.asarray(
    (
        (0.42, 0.66, 0.51, 0.74),
        (0.58, 0.39, 0.69, 0.47),
        (0.63, 0.55, 0.36, 0.61),
        (0.49, 0.71, 0.57, 0.44),
    ),
    dtype=float,
)


def nominal_partition_pair_tables() -> dict[str, tuple[tuple[float, ...], ...]]:
    """Return the public nominal pair tables without any realized-world state."""

    return {
        "product_distribution_coefficients": tuple(
            tuple(float(value) for value in row)
            for row in _NOMINAL_PRODUCT_DISTRIBUTION_COEFFICIENTS
        ),
        "impurity_distribution_coefficients": tuple(
            tuple(float(value) for value in row)
            for row in _NOMINAL_IMPURITY_DISTRIBUTION_COEFFICIENTS
        ),
    }


class PartitionSplitResult(TypedDict):
    partition_coefficient: float
    impurity_partition_coefficient: float
    organic_product_mol: float
    aqueous_product_mol: float
    organic_impurity_mol: float
    aqueous_impurity_mol: float
    lle_phase_status: str
    lle_minimum_tpd_like: float
    lle_partition_log_spread: float
    extraction_model_id: str
    extraction_converged: bool
    extraction_material_balance_error_mol: float
    extraction_entrained_aqueous_volume_L: float
    extraction_provenance: tuple[str, ...]


def partition_split(
    *,
    product_mol: float,
    impurity_mol: float,
    solvent: int,
    extractant: int | None = None,
    nominal_pair_contract: str | None = None,
    temperature_K: float,
    duration_s: float,
    stirring_speed_rpm: float,
    organic_volume_L: float,
    aqueous_volume_L: float,
    coefficient_multiplier: float = 1.0,
    coefficient_exponent: float = 1.0,
    composition_coupling_multiplier: float = 1.0,
    phase_volume_multiplier: float = 1.0,
    solvent_inventory: SolventInventory | None = None,
    extractant_inventory: SolventInventory | None = None,
) -> PartitionSplitResult:
    if (
        coefficient_multiplier <= 0.0
        or coefficient_exponent <= 0.0
        or composition_coupling_multiplier < 1.0
        or phase_volume_multiplier <= 0.0
    ):
        raise ValueError("partition intervention multipliers must be positive")
    if solvent < 0 or solvent >= 4:
        raise ValueError("solvent must be one of four nominal choices")
    if nominal_pair_contract is not None and (
        nominal_pair_contract != INDEPENDENT_NOMINAL_SOLVENT_EXTRACTANT_PAIR_V1
    ):
        raise ValueError("unsupported nominal solvent/extractant pairing contract")
    if nominal_pair_contract == INDEPENDENT_NOMINAL_SOLVENT_EXTRACTANT_PAIR_V1:
        if extractant is None or extractant < 0 or extractant >= 4:
            raise ValueError("independent nominal pairing requires one of four extractants")
        feed_medium = solvent_inventory or SolventInventory.pure(solvent, 1.0)
        receiver_medium = extractant_inventory or SolventInventory.pure(extractant, 1.0)
        partition_base = PARTITION_V3_PRODUCT_DISTRIBUTION_CALIBRATION * feed_medium.log_property(
            tuple(
                receiver_medium.log_property(row)
                for row in _NOMINAL_PRODUCT_DISTRIBUTION_COEFFICIENTS
            )
        )
        impurity_partition_base = feed_medium.log_property(
            tuple(
                receiver_medium.log_property(row)
                for row in _NOMINAL_IMPURITY_DISTRIBUTION_COEFFICIENTS
            )
        )
        independent_nominal_pair = True
    else:
        historical_partition_base = np.array([0.65, 1.25, 2.20, 1.55])
        medium = extractant_inventory or solvent_inventory or SolventInventory.pure(solvent, 1.0)
        partition_base = medium.log_property(historical_partition_base)
        impurity_partition_base = 0.0
        independent_nominal_pair = False
    temperature_factor = 1.0 + 0.0025 * (temperature_K - 298.15)
    mix_factor = 0.75 + 0.25 * (1.0 - np.exp(-duration_s / 240.0)) * (
        0.70 + 0.30 * stirring_speed_rpm / 1200.0
    )
    partition = max(
        0.05,
        float(
            partition_base**coefficient_exponent
            * temperature_factor
            * mix_factor
            * coefficient_multiplier
        ),
    )
    v_org = max(organic_volume_L * phase_volume_multiplier, 1.0e-9)
    v_aq = max(aqueous_volume_L / phase_volume_multiplier, 1.0e-9)
    if independent_nominal_pair:
        impurity_partition = max(
            0.05,
            float(
                impurity_partition_base**coefficient_exponent
                * temperature_factor
                * mix_factor
                * coefficient_multiplier**0.50
            ),
        )
    else:
        impurity_partition = max(
            0.05,
            float(
                (
                    0.18 * partition / coefficient_multiplier
                    + medium.linear_property((0.08, 0.16, 0.24, 0.32))
                )
                / coefficient_multiplier**0.25
            ),
        )
    # The parent law (multiplier == 1) is composition independent.  The paired
    # private family introduces a feed-composition response that cannot be
    # absorbed by refitting one constant K across two feed anchors.  The
    # logarithmic form stays positive and is symmetric at equal product and
    # impurity fractions.
    feed_total = max(product_mol, 0.0) + max(impurity_mol, 0.0)
    if feed_total > 0.0 and composition_coupling_multiplier > 1.0:
        impurity_fraction = max(impurity_mol, 0.0) / feed_total
        log_strength = float(np.log(composition_coupling_multiplier))
        partition *= float(np.exp(-log_strength * impurity_fraction))
        impurity_partition *= float(np.exp(log_strength * impurity_fraction))
    feed = {"product": max(product_mol, 0.0), "impurity": max(impurity_mol, 0.0)}
    if sum(feed.values()) > 0.0:
        distribution_model = DistributionCoefficientModelSpec(
            model_id="runtime_activity_corrected_distribution_v1",
            component_ids=("product", "impurity"),
            intrinsic_partition_coefficients={
                "product": partition,
                "impurity": impurity_partition,
            },
            provenance_id="chemworld-world-law-vnext-partition-policy",
        )
        entrainment_fraction = float(
            np.clip(
                0.01 + 0.015 * stirring_speed_rpm / 1200.0 * (1.0 - np.exp(-duration_s / 120.0)),
                0.0,
                0.04,
            )
        )
        extraction = simulate_stability_aware_extraction(
            StabilityAwareExtractionRequest(
                feed_amounts_mol=feed,
                distribution_model=distribution_model,
                target_component="product",
                contactor=LLEContactorSpec(
                    aqueous_volume_L=v_aq,
                    organic_volume_L=v_org,
                    extraction_stages=1,
                    extraction_stage_efficiency=float(np.clip(mix_factor, 0.0, 1.0)),
                    extraction_entrainment_fraction=entrainment_fraction,
                    maximum_contact_volume_L=max(v_aq + v_org, 0.10),
                ),
                temperature_K=temperature_K,
            )
        )
        organic = extraction.outlet("extract")
        aqueous = extraction.outlet("raffinate")
        organic_product_mol = organic["product"]
        aqueous_product_mol = aqueous["product"]
        organic_impurity_mol = organic["impurity"]
        aqueous_impurity_mol = aqueous["impurity"]
        first_stage = extraction.stage_reports[0]
        diagnostic = first_stage.stability_diagnostic
        extraction_model_id = extraction.model_id
        extraction_converged = extraction.all_stages_converged
        extraction_balance_error = extraction.material_balance_error_mol
        extraction_entrained_volume = extraction.entrained_volume_L
        extraction_provenance = extraction.provenance
    else:
        organic_product_mol = aqueous_product_mol = 0.0
        organic_impurity_mol = aqueous_impurity_mol = 0.0
        diagnostic = {
            "phase_status": "single_liquid",
            "minimum_tpd_like": 0.0,
            "partition_log_spread": 0.0,
        }
        extraction_model_id = "chemworld_stability_aware_lle_vnext"
        extraction_converged = True
        extraction_balance_error = 0.0
        extraction_entrained_volume = 0.0
        extraction_provenance = ("ChemWorld stability-aware LLE empty-feed identity",)
    return {
        "partition_coefficient": partition,
        "impurity_partition_coefficient": impurity_partition,
        "organic_product_mol": organic_product_mol,
        "aqueous_product_mol": aqueous_product_mol,
        "organic_impurity_mol": organic_impurity_mol,
        "aqueous_impurity_mol": aqueous_impurity_mol,
        "lle_phase_status": str(diagnostic.get("phase_status", "single_liquid")),
        "lle_minimum_tpd_like": _diagnostic_float(diagnostic.get("minimum_tpd_like", 0.0)),
        "lle_partition_log_spread": _diagnostic_float(diagnostic.get("partition_log_spread", 0.0)),
        "extraction_model_id": extraction_model_id,
        "extraction_converged": extraction_converged,
        "extraction_material_balance_error_mol": extraction_balance_error,
        "extraction_entrained_aqueous_volume_L": extraction_entrained_volume,
        "extraction_provenance": extraction_provenance,
    }


@dataclass(frozen=True)
class PhaseModuleSpec:
    module_id: str = "phase_partition"
    version: str = "0.7"
    laws: tuple[str, ...] = (
        "aqueous_organic_phase_volume_balance",
        "benchmark_calibrated_intrinsic_distribution_coefficients",
        "contract_scoped_independent_nominal_solvent_extractant_interaction_table",
        "stability_gated_activity_corrected_extraction_train",
        "explicit_aqueous_entrainment_ledger",
        "tpd_style_phase_stability_diagnostic",
    )

    def to_dict(self) -> dict[str, object]:
        return {
            "module_id": self.module_id,
            "version": self.version,
            "laws": list(self.laws),
            "phases": ["reactor_liquid", "aqueous", "organic", "solid"],
            "observable_proxies": [
                "phase_ratio",
                "product_in_organic",
                "product_in_aqueous",
                "solvent_loss",
                "process_mass_balance_error",
            ],
        }


def _diagnostic_float(value: object, default: float = 0.0) -> float:
    return float(value) if isinstance(value, int | float) else default


__all__ = [
    "INDEPENDENT_NOMINAL_SOLVENT_EXTRACTANT_PAIR_V1",
    "PARTITION_V3_PRODUCT_DISTRIBUTION_CALIBRATION",
    "PartitionSplitResult",
    "PhaseModuleSpec",
    "nominal_partition_pair_tables",
    "partition_split",
]
