"""Participant-information-only endpoint map and candidate fitting for W2-87.

No simulator, realized world, scoring labels, or private random coordinates are
accepted here. The separate qualification driver checks this public model.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.optimize import least_squares, minimize_scalar

METRICS = ("product_in_organic", "product_in_aqueous", "phase_ratio", "score")
FAMILIES = ("FAMILY_A_LINEAR", "FAMILY_B_POWER", "FAMILY_C_SATURATING", "FAMILY_D_CONSTANT")


def public_contract(temperature_K: float) -> dict:
    return {
        "version": "w2-87-public-endpoint-1",
        "initial_temperature_K": temperature_K,
        "initial_liquid_volume_L": 0.0,
        "added_solvent_volume_L": 0.020,
        "coefficient_multiplier": 1.0,
        "phase_volume_multiplier": 1.0,
        "process": [
            "Fresh batch with a positive product and impurity inventory, temperature as stated.",
            "Add 0.020 L solvent, then aqueous phase, then extractant at the query volumes.",
            "One mix, one settle, HPLC, retain only organic phase, HPLC, terminate, final assay.",
            "At contact Vo=extractant_volume_L; Va=0.020+aqueous_phase_volume_L.",
            "T is constant. m=.75+.25*(1-exp(-mix_duration_s/240))"
            "*(.70+.30*stirring_speed_rpm/1200). process_factor=m*(1+.0025*(T-298.15)).",
            "D=max(.05,candidate_coefficient*process_factor). "
            "For linear candidate_coefficient=D_reference; for power D_reference**alpha; "
            "for saturating a+b*D_reference/(c+D_reference).",
            "In the stable ideal two-liquid domain, equilibrium organic fraction "
            "q=D*Vo/(D*Vo+Va); contacted fraction u=clip(m,0,1)*q.",
            "e=clip(.01+.015*stirring_speed_rpm/1200*(1-exp(-mix_duration_s/120)),0,.04). "
            "Organic retained fraction f=u+e*(1-u) if Vo>=Va, else u*(1-e).",
            "Settling has no further mass redistribution at these durations. "
            "Phase removal discards aqueous product; it preserves the initial-product denominator.",
            "Each HPLC consumes .00020 L and final assay consumes .00030 L. "
            "Readings describe the withdrawn sample's pre-withdrawal mother liquor. "
            "Committed sampling scales current and initial product inventory equally, so fractions "
            "cancel these sampling factors. Retained phase volume ratio is one.",
            "Noiseless final channels are product_in_organic=f, product_in_aqueous=0, "
            "phase_ratio=1. The constant-endpoint candidate instead fits constant channels.",
        ],
        "noise": {
            "distribution": "add independent zero-mean Gaussian noise per channel; clip to [0,1]",
            "std": dict(zip(METRICS[:3], (0.010, 0.010, 0.012), strict=True)),
            "reference_target_pairing": "Same query has identical Gaussian draws for reference "
            "and target, before clipping. Different queries have separate noise coordinates.",
            "reporting": "float32 public channel and score serialization",
        },
        "score": "clip(.85*observed_product_in_organic-.10*observed_product_in_aqueous"
        "-.10*observed_phase_ratio,0,1); no additional score noise or cost term",
        "qualification_scope": "One stable ideal two-liquid contact and this assay workflow.",
    }


def candidate_domains() -> dict:
    return {
        "FAMILY_A_LINEAR": "No free coefficient parameter",
        "FAMILY_B_POWER": {"alpha": [0.25, 3.0]},
        "FAMILY_C_SATURATING": {"a": [0.0, 20.0], "b": [0.0, 100.0], "c": [0.01, 100.0]},
        "FAMILY_D_CONSTANT": "One [0,1] constant per observable channel",
    }


def organic(
    queries: list[dict], contract: dict, family: str, parameters: list[float]
) -> np.ndarray:
    refs = np.array([q["reference_partition_coefficient"] for q in queries], dtype=float)
    controls = [q["feature_values"] for q in queries]
    duration = np.array([c["mix_duration_s"] for c in controls])
    rpm = np.array([c["stirring_speed_rpm"] for c in controls])
    vo = np.array([c["extractant_volume_L"] for c in controls])
    va = (
        np.array([c["aqueous_phase_volume_L"] for c in controls])
        + contract["added_solvent_volume_L"]
    )
    if family == FAMILIES[3]:
        return np.full(len(queries), parameters[0])
    if family == FAMILIES[0]:
        base = refs
    elif family == FAMILIES[1]:
        base = refs ** parameters[0]
    elif family == FAMILIES[2]:
        a, b, c = parameters
        base = a + b * refs / (c + refs)
    else:
        raise ValueError("unknown candidate family")
    mix = 0.75 + 0.25 * (1 - np.exp(-duration / 240)) * (0.70 + 0.30 * rpm / 1200)
    coefficient = np.maximum(
        0.05, base * mix * (1 + 0.0025 * (contract["initial_temperature_K"] - 298.15))
    )
    contacted = np.clip(mix, 0, 1) * coefficient * vo / (coefficient * vo + va)
    entrainment = np.clip(0.01 + 0.015 * rpm / 1200 * (1 - np.exp(-duration / 120)), 0, 0.04)
    return np.where(
        vo >= va, contacted + entrainment * (1 - contacted), contacted * (1 - entrainment)
    )


def clipped_normal_mean(mu: float, sigma: float) -> float:
    """Exact marginal mean after clipping a Gaussian to [0,1]."""
    cdf = lambda z: 0.5 * (1 + math.erf(z / math.sqrt(2)))  # noqa: E731
    pdf = lambda z: math.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)  # noqa: E731
    low, high = -mu / sigma, (1 - mu) / sigma
    return mu * (cdf(high) - cdf(low)) + sigma * (pdf(low) - pdf(high)) + 1 - cdf(high)


def endpoints(
    queries: list[dict], contract: dict, family: str, parameters: list[float], *, noisy_mean: bool
) -> dict:
    fractions = organic(queries, contract, family, parameters)
    result = {}
    for query, fraction in zip(queries, fractions, strict=True):
        values = [float(fraction), 0.0, 1.0]
        if noisy_mean:
            values = [
                clipped_normal_mean(mu, sigma)
                for mu, sigma in zip(values, (0.01, 0.01, 0.012), strict=True)
            ]
        # Score at marginal channel means; outer clipping can affect expected score.
        values.append(float(np.clip(0.85 * values[0] - 0.10 * values[1] - 0.10 * values[2], 0, 1)))
        result[query["query_id"]] = dict(zip(METRICS, values, strict=True))
    return result


def fit_public(packet: dict) -> dict[str, Any]:
    """Fit every family before any held-out truth is supplied to an evaluator.

    Paired differences cancel common noise on interior organic readings. A
    censored pair is an explicit unsupported qualification, never silently dropped.
    """
    evidence = packet["evidence"]
    contract = packet["complete_observation_contract"]
    observed = np.array([q["target_observations"]["product_in_organic"] for q in evidence])
    reference = np.array(
        [q["reference_linear_observations"]["product_in_organic"] for q in evidence]
    )
    if np.any((observed <= 0) | (observed >= 1) | (reference <= 0) | (reference >= 1)):
        raise ValueError(
            "censored organic pair: declared paired-difference fitter is not applicable"
        )
    baseline = organic(evidence, contract, FAMILIES[0], [])
    paired_target = baseline + observed - reference

    def residual(family: str, parameters: list[float]) -> np.ndarray:
        return organic(evidence, contract, family, parameters) - paired_target

    grid = np.linspace(0.25, 3, 276)
    costs = [float(np.mean(residual(FAMILIES[1], [x]) ** 2)) for x in grid]
    index = int(np.argmin(costs))
    power = minimize_scalar(
        lambda x: float(np.mean(residual(FAMILIES[1], [x]) ** 2)),
        bounds=(grid[max(0, index - 1)], grid[min(len(grid) - 1, index + 1)]),
        method="bounded",
        options={"xatol": 1e-12},
    )
    saturating = []
    for a in (0.0, 1.0):
        for c in (0.1, 1.0, 10.0, 90.0):
            fit = least_squares(
                lambda p: residual(FAMILIES[2], p),
                [a, 10.0, c],
                bounds=([0, 0, 0.01], [20, 100, 100]),
                ftol=1e-12,
                xtol=1e-12,
                gtol=1e-12,
                max_nfev=3000,
            )
            saturating.append(fit)
    best = min(saturating, key=lambda f: float(np.sum(f.fun**2)))
    parameters = [
        [],
        [float(power.x)],
        best.x.tolist(),
        [float(np.clip(np.mean(paired_target), 0, 1))],
    ]
    rows = []
    for family, params in zip(FAMILIES, parameters, strict=True):
        rows.append(
            {
                "family": family,
                "parameters": params,
                "organic_paired_rmse": float(np.sqrt(np.mean(residual(family, params) ** 2))),
                "predictions": endpoints(
                    packet["scoring_action_queries"], contract, family, params, noisy_mean=True
                ),
            }
        )
    ranked = sorted(rows, key=lambda r: r["organic_paired_rmse"])
    return {
        "families": rows,
        "selected_family": ranked[0]["family"],
        "organic_rmse_margin": ranked[1]["organic_paired_rmse"] - ranked[0]["organic_paired_rmse"],
        "power_optimizer_success": bool(power.success),
        "saturation_optimizer_successes": sum(bool(f.success) for f in saturating),
        "evidence_queries": len(evidence),
        "held_out_labels_accessed": False,
        "prediction_semantics": "Marginal clipped-channel means; score evaluated at those means.",
    }
