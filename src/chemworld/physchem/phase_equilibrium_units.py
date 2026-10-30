"""Stability-gated multistage liquid-liquid extraction and wash trains."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite, log
from typing import Any, Literal, cast

from chemworld.physchem.equilibrium import (
    ActivityModelSpec,
    LiquidEquilibriumResult,
    LiquidStabilityReport,
    activity_coefficients,
    entrain_liquid_amounts,
    equilibrate_liquids,
    lle_phase_stability_diagnostic,
)
from chemworld.physchem.extraction_units import DistributionCoefficientModelSpec
from chemworld.physchem.maturity import (
    MaturityLevel,
    ModelCard,
    ValidationEvidence,
)

PHASE_EQUILIBRIUM_MODEL_ID = "chemworld_stability_aware_lle_vnext"
PHASEPY_COMMIT = "9376df19c9ddf6723d10639d11cad64ee0b54047"
PHASEPY_LLE_PATH = "phasepy/equilibrium/lle.py"
PHASEPY_STABILITY_PATH = "phasepy/equilibrium/stability.py"

ContinuousPhasePolicy = Literal["auto", "aqueous", "organic"]


@dataclass(frozen=True)
class StrictLiquidSystemSpec:
    """Isothermal liquid material law, distinct from empirical contact policy.

    Pure-liquid chemical reference terms cancel between the equal-T phases.
    Only relative mixing potentials are supplied. Mixing heat/PV work is not
    an implemented energy closure in this finite isothermal model.
    """

    activity_model: ActivityModelSpec
    molar_holdups_L_mol: dict[str, float]
    temperature_K: float = 298.15
    pressure_Pa: float = 101325.0
    wash_feed_id: str = ""
    retained_phase: int = 1
    entrainment_fraction: float = .01
    max_iterations: int = 1000

    def __post_init__(self) -> None:
        if len(self.activity_model.component_ids) not in (2, 3):
            raise ValueError("strict liquid model supports binary/ternary material")
        volumes = dict(self.molar_holdups_L_mol)
        if (set(volumes) != set(self.activity_model.component_ids)
                or any(not isfinite(v) or v <= 0 for v in volumes.values())
                or any(not isfinite(v) or v <= 0 for v in (self.temperature_K, self.pressure_Pa))
                or self.retained_phase not in (0, 1) or isinstance(self.retained_phase, bool)
                or not 0 <= self.entrainment_fraction < 1 or isinstance(self.max_iterations, bool)
                or not isinstance(self.max_iterations, int) or self.max_iterations <= 0):
            raise ValueError("invalid strict liquid holdup/temperature/pressure/contact data")
        object.__setattr__(self, "molar_holdups_L_mol", volumes)

    def solve(self, amounts):
        feed = _amounts(amounts, label="strict liquid feed", require_positive_total=False)
        if set(feed) != set(self.activity_model.component_ids):
            raise ValueError("strict liquid feed identities differ from its model")
        if sum(feed.values()) == 0:
            stability = LiquidStabilityReport(self.temperature_K, feed, 0, feed, (), ())
            return LiquidEquilibriumResult(self.temperature_K, feed, (), stability, (), 0, 0, 0)
        return equilibrate_liquids(self.activity_model, feed, temperature_K=self.temperature_K,
                                   max_iterations=self.max_iterations)

    def volume_l(self, amounts):
        return sum(self.molar_holdups_L_mol[key] * amount for key, amount in amounts.items())

    def to_dict(self):
        from dataclasses import fields

        return {"mode": "strict_liquid_equilibrium_v1", **{
            f.name: (self.activity_model.to_dict() if f.name == "activity_model"
                     else getattr(self, f.name))
            for f in fields(self)}}

    @classmethod
    def from_dict(cls, payload):
        values = dict(payload)
        if values.pop("mode") != "strict_liquid_equilibrium_v1":
            raise ValueError("unsupported liquid equilibrium model mode")
        activity = dict(values["activity_model"])
        activity["component_ids"] = tuple(activity["component_ids"])
        values["activity_model"] = ActivityModelSpec(**activity)
        return cls(**values)


def _finite_positive(value: float, label: str) -> float:
    resolved = float(value)
    if not isfinite(resolved) or resolved <= 0.0:
        raise ValueError(f"{label} must be finite and positive")
    return resolved


def _finite_nonnegative(value: float, label: str) -> float:
    resolved = float(value)
    if not isfinite(resolved) or resolved < 0.0:
        raise ValueError(f"{label} must be finite and nonnegative")
    return resolved


def _fraction(value: float, label: str) -> float:
    resolved = float(value)
    if not isfinite(resolved) or not 0.0 <= resolved <= 1.0:
        raise ValueError(f"{label} must lie in [0, 1]")
    return resolved


def _open_fraction(value: float, label: str) -> float:
    resolved = float(value)
    if not isfinite(resolved) or not 0.0 <= resolved < 1.0:
        raise ValueError(f"{label} must lie in [0, 1)")
    return resolved


def _amounts(
    values: Mapping[str, float],
    *,
    label: str,
    require_positive_total: bool = True,
) -> dict[str, float]:
    resolved: dict[str, float] = {}
    for raw_component_id, raw_amount in values.items():
        component_id = str(raw_component_id).strip()
        if not component_id:
            raise ValueError(f"{label} component ids cannot be empty")
        if component_id in resolved:
            raise ValueError(f"{label} component ids collide after normalization")
        resolved[component_id] = _finite_nonnegative(
            raw_amount,
            f"{label}[{component_id!r}]",
        )
    if not resolved:
        raise ValueError(f"{label} cannot be empty")
    if require_positive_total and sum(resolved.values()) <= 0.0:
        raise ValueError(f"{label} must contain positive total material")
    return dict(sorted(resolved.items()))


def _zero_amounts(component_ids: tuple[str, ...]) -> dict[str, float]:
    return dict.fromkeys(component_ids, 0.0)


def _composition(amounts: Mapping[str, float]) -> dict[str, float]:
    total = sum(amounts.values())
    if total <= 0.0:
        return {component_id: 1.0 / len(amounts) for component_id in amounts}
    return {component_id: amount / total for component_id, amount in amounts.items()}


def _add(
    left: Mapping[str, float],
    right: Mapping[str, float],
) -> dict[str, float]:
    if set(left) != set(right):
        raise ValueError("amount ledgers must have exactly matching component ids")
    return {component_id: left[component_id] + right[component_id] for component_id in left}


def _balance_error(
    feed: Mapping[str, float],
    outlets: Mapping[str, Mapping[str, float]],
) -> float:
    return max(
        (
            abs(
                feed[component_id]
                - sum(outlet.get(component_id, 0.0) for outlet in outlets.values())
            )
            for component_id in feed
        ),
        default=0.0,
    )


def _phase_policy(value: str, label: str) -> ContinuousPhasePolicy:
    resolved = str(value).strip().lower()
    if resolved not in {"auto", "aqueous", "organic"}:
        raise ValueError(f"{label} must be auto, aqueous, or organic")
    return resolved  # type: ignore[return-value]


@dataclass(frozen=True)
class LLEContactorSpec:
    """Equipment and numerical policy for one extraction/wash train."""

    aqueous_volume_L: float
    organic_volume_L: float
    extraction_stages: int = 1
    wash_aqueous_volumes_L: tuple[float, ...] = ()
    extraction_stage_efficiency: float = 1.0
    wash_stage_efficiency: float = 1.0
    extraction_entrainment_fraction: float = 0.0
    wash_entrainment_fraction: float = 0.0
    extraction_continuous_phase: ContinuousPhasePolicy = "auto"
    wash_continuous_phase: ContinuousPhasePolicy = "auto"
    maximum_contact_volume_L: float = 100.0
    distribution_tolerance: float = 1.0e-10
    tpd_tolerance: float = 1.0e-9
    material_balance_tolerance_mol: float = 1.0e-10
    max_iterations: int = 100

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "aqueous_volume_L",
            _finite_positive(self.aqueous_volume_L, "aqueous_volume_L"),
        )
        object.__setattr__(
            self,
            "organic_volume_L",
            _finite_positive(self.organic_volume_L, "organic_volume_L"),
        )
        if isinstance(self.extraction_stages, bool) or self.extraction_stages <= 0:
            raise ValueError("extraction_stages must be a positive integer")
        if self.extraction_stages > 20:
            raise ValueError("extraction_stages exceeds the declared maximum of 20")
        wash_volumes = tuple(
            _finite_positive(value, "wash_aqueous_volumes_L")
            for value in self.wash_aqueous_volumes_L
        )
        if len(wash_volumes) > 20:
            raise ValueError("wash stage count exceeds the declared maximum of 20")
        object.__setattr__(self, "wash_aqueous_volumes_L", wash_volumes)
        for field_name in (
            "extraction_stage_efficiency",
            "wash_stage_efficiency",
        ):
            object.__setattr__(
                self,
                field_name,
                _fraction(getattr(self, field_name), field_name),
            )
        for field_name in (
            "extraction_entrainment_fraction",
            "wash_entrainment_fraction",
        ):
            object.__setattr__(
                self,
                field_name,
                _open_fraction(getattr(self, field_name), field_name),
            )
        object.__setattr__(
            self,
            "extraction_continuous_phase",
            _phase_policy(
                self.extraction_continuous_phase,
                "extraction_continuous_phase",
            ),
        )
        object.__setattr__(
            self,
            "wash_continuous_phase",
            _phase_policy(self.wash_continuous_phase, "wash_continuous_phase"),
        )
        object.__setattr__(
            self,
            "maximum_contact_volume_L",
            _finite_positive(
                self.maximum_contact_volume_L,
                "maximum_contact_volume_L",
            ),
        )
        object.__setattr__(
            self,
            "distribution_tolerance",
            _finite_positive(self.distribution_tolerance, "distribution_tolerance"),
        )
        object.__setattr__(
            self,
            "tpd_tolerance",
            _finite_nonnegative(self.tpd_tolerance, "tpd_tolerance"),
        )
        object.__setattr__(
            self,
            "material_balance_tolerance_mol",
            _finite_positive(
                self.material_balance_tolerance_mol,
                "material_balance_tolerance_mol",
            ),
        )
        if isinstance(self.max_iterations, bool) or self.max_iterations <= 0:
            raise ValueError("max_iterations must be a positive integer")
        if self.aqueous_volume_L + self.organic_volume_L > self.maximum_contact_volume_L:
            raise ValueError("extraction contact exceeds maximum_contact_volume_L")
        combined_organic_volume = self.extraction_stages * self.organic_volume_L
        if any(
            combined_organic_volume + wash_volume > self.maximum_contact_volume_L
            for wash_volume in wash_volumes
        ):
            raise ValueError("wash contact exceeds maximum_contact_volume_L")

    def to_dict(self) -> dict[str, Any]:
        return {
            "aqueous_volume_L": self.aqueous_volume_L,
            "organic_volume_L": self.organic_volume_L,
            "extraction_stages": self.extraction_stages,
            "wash_aqueous_volumes_L": list(self.wash_aqueous_volumes_L),
            "extraction_stage_efficiency": self.extraction_stage_efficiency,
            "wash_stage_efficiency": self.wash_stage_efficiency,
            "extraction_entrainment_fraction": self.extraction_entrainment_fraction,
            "wash_entrainment_fraction": self.wash_entrainment_fraction,
            "extraction_continuous_phase": self.extraction_continuous_phase,
            "wash_continuous_phase": self.wash_continuous_phase,
            "maximum_contact_volume_L": self.maximum_contact_volume_L,
            "distribution_tolerance": self.distribution_tolerance,
            "tpd_tolerance": self.tpd_tolerance,
            "material_balance_tolerance_mol": self.material_balance_tolerance_mol,
            "max_iterations": self.max_iterations,
        }


@dataclass(frozen=True)
class StabilityAwareExtractionRequest:
    feed_amounts_mol: dict[str, float]
    distribution_model: DistributionCoefficientModelSpec
    target_component: str
    contactor: LLEContactorSpec
    temperature_K: float = 298.15
    stability_activity_model: ActivityModelSpec | None = None

    def __post_init__(self) -> None:
        feed = _amounts(self.feed_amounts_mol, label="feed_amounts_mol")
        if len(feed) < 2:
            raise ValueError("stability-aware LLE requires at least two tracked components")
        component_ids = set(self.distribution_model.component_ids)
        if set(feed) != component_ids:
            raise ValueError("feed and distribution model component ids must exactly match")
        target_component = self.target_component.strip()
        if target_component not in feed or feed[target_component] <= 0.0:
            raise ValueError("target_component must have a positive feed amount")
        if not isinstance(self.contactor, LLEContactorSpec):
            raise ValueError("contactor must be an LLEContactorSpec")
        temperature = _finite_positive(self.temperature_K, "temperature_K")
        if self.stability_activity_model is not None and set(
            self.stability_activity_model.component_ids
        ) != set(feed):
            raise ValueError("stability activity model component ids must match the feed")
        for (
            component_id,
            coefficient,
        ) in self.distribution_model.intrinsic_partition_coefficients.items():
            if not 1.0e-12 <= coefficient <= 1.0e12:
                raise ValueError(
                    f"intrinsic partition coefficient for {component_id!r} "
                    "lies outside [1e-12, 1e12]"
                )
        object.__setattr__(self, "feed_amounts_mol", feed)
        object.__setattr__(self, "target_component", target_component)
        object.__setattr__(self, "temperature_K", temperature)

    def to_dict(self) -> dict[str, Any]:
        return {
            "feed_amounts_mol": dict(self.feed_amounts_mol),
            "distribution_model": self.distribution_model.to_dict(),
            "target_component": self.target_component,
            "contactor": self.contactor.to_dict(),
            "temperature_K": self.temperature_K,
            "stability_activity_model": (
                None
                if self.stability_activity_model is None
                else self.stability_activity_model.to_dict()
            ),
        }


@dataclass(frozen=True)
class StabilityAwareContactReport:
    stage_id: str
    mode: str
    phase_status: str
    aqueous_volume_L: float
    organic_volume_L: float
    final_phase_volumes_L: dict[str, float]
    continuous_phase: str
    dispersed_phase: str
    input_amounts_mol: dict[str, float]
    input_organic_amounts_mol: dict[str, float]
    organic_amounts_mol: dict[str, float]
    aqueous_amounts_mol: dict[str, float]
    distribution_coefficients: dict[str, float]
    aqueous_activity_coefficients: dict[str, float]
    organic_activity_coefficients: dict[str, float]
    recovery_to_organic: dict[str, float]
    entrained_amounts_mol: dict[str, float]
    entrained_volume_L: float
    entrained_from_phase: str | None
    entrained_to_phase: str | None
    stage_efficiency: float
    iterations: int
    distribution_residual: float
    material_balance_error_mol: float
    stability_diagnostic: dict[str, object]
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage_id": self.stage_id,
            "mode": self.mode,
            "phase_status": self.phase_status,
            "aqueous_volume_L": self.aqueous_volume_L,
            "organic_volume_L": self.organic_volume_L,
            "final_phase_volumes_L": dict(self.final_phase_volumes_L),
            "continuous_phase": self.continuous_phase,
            "dispersed_phase": self.dispersed_phase,
            "input_amounts_mol": dict(self.input_amounts_mol),
            "input_organic_amounts_mol": dict(self.input_organic_amounts_mol),
            "organic_amounts_mol": dict(self.organic_amounts_mol),
            "aqueous_amounts_mol": dict(self.aqueous_amounts_mol),
            "distribution_coefficients": dict(self.distribution_coefficients),
            "aqueous_activity_coefficients": dict(self.aqueous_activity_coefficients),
            "organic_activity_coefficients": dict(self.organic_activity_coefficients),
            "recovery_to_organic": dict(self.recovery_to_organic),
            "entrained_amounts_mol": dict(self.entrained_amounts_mol),
            "entrained_volume_L": self.entrained_volume_L,
            "entrained_from_phase": self.entrained_from_phase,
            "entrained_to_phase": self.entrained_to_phase,
            "stage_efficiency": self.stage_efficiency,
            "iterations": self.iterations,
            "distribution_residual": self.distribution_residual,
            "material_balance_error_mol": self.material_balance_error_mol,
            "stability_diagnostic": dict(self.stability_diagnostic),
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class StabilityAwareExtractionResult:
    model_id: str
    distribution_model_id: str
    distribution_provenance_id: str
    temperature_K: float
    target_component: str
    feed_amounts_mol: dict[str, float]
    outlets: dict[str, dict[str, float]]
    stage_reports: tuple[StabilityAwareContactReport, ...]
    target_recovery: float
    target_purity: float
    impurity_rejection: float
    material_balance_error_mol: float
    maximum_stage_material_balance_error_mol: float
    minimum_tpd_like: float
    maximum_distribution_residual: float
    entrained_volume_L: float
    all_stages_two_liquid: bool
    all_stages_converged: bool
    warnings: tuple[str, ...]
    provenance: tuple[str, ...]

    def outlet(self, outlet_id: str) -> dict[str, float]:
        return dict(self.outlets[outlet_id])

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "distribution_model_id": self.distribution_model_id,
            "distribution_provenance_id": self.distribution_provenance_id,
            "temperature_K": self.temperature_K,
            "target_component": self.target_component,
            "feed_amounts_mol": dict(self.feed_amounts_mol),
            "outlets": {key: dict(value) for key, value in self.outlets.items()},
            "stage_reports": [report.to_dict() for report in self.stage_reports],
            "target_recovery": self.target_recovery,
            "target_purity": self.target_purity,
            "impurity_rejection": self.impurity_rejection,
            "material_balance_error_mol": self.material_balance_error_mol,
            "maximum_stage_material_balance_error_mol": (
                self.maximum_stage_material_balance_error_mol
            ),
            "minimum_tpd_like": self.minimum_tpd_like,
            "maximum_distribution_residual": self.maximum_distribution_residual,
            "entrained_volume_L": self.entrained_volume_L,
            "all_stages_two_liquid": self.all_stages_two_liquid,
            "all_stages_converged": self.all_stages_converged,
            "warnings": list(self.warnings),
            "provenance": list(self.provenance),
        }


def ideal_organic_fraction(
    partition_coefficient: float,
    *,
    aqueous_volume_L: float,
    organic_volume_L: float,
) -> float:
    """Closed-form equilibrium organic fraction for one ideal contact."""

    coefficient = _finite_positive(partition_coefficient, "partition_coefficient")
    aqueous_volume = _finite_positive(aqueous_volume_L, "aqueous_volume_L")
    organic_volume = _finite_positive(organic_volume_L, "organic_volume_L")
    ratio = aqueous_volume / (coefficient * organic_volume)
    return 1.0 / (1.0 + ratio)


def simulate_stability_aware_extraction(
    request: StabilityAwareExtractionRequest,
) -> StabilityAwareExtractionResult:
    """Run extraction and wash contacts only where the declared LLE slice is stable."""

    if not isinstance(request, StabilityAwareExtractionRequest):
        raise ValueError("request must be a StabilityAwareExtractionRequest")
    component_ids = tuple(request.feed_amounts_mol)
    zero = _zero_amounts(component_ids)
    contactor = request.contactor
    reports: list[StabilityAwareContactReport] = []
    raffinate = dict(request.feed_amounts_mol)
    combined_extract = dict(zero)
    combined_organic_volume_L = 0.0
    current_aqueous_volume_L = contactor.aqueous_volume_L

    for index in range(1, contactor.extraction_stages + 1):
        report = _contact_stage(
            request=request,
            stage_id=f"extraction_{index}",
            mode="extraction",
            total_amounts_mol=raffinate,
            input_organic_amounts_mol=zero,
            aqueous_volume_L=current_aqueous_volume_L,
            organic_volume_L=contactor.organic_volume_L,
            stage_efficiency=contactor.extraction_stage_efficiency,
            entrainment_fraction=contactor.extraction_entrainment_fraction,
            continuous_phase_policy=contactor.extraction_continuous_phase,
        )
        reports.append(report)
        combined_extract = _add(combined_extract, report.organic_amounts_mol)
        raffinate = dict(report.aqueous_amounts_mol)
        combined_organic_volume_L += report.final_phase_volumes_L["organic"]
        current_aqueous_volume_L = report.final_phase_volumes_L["aqueous"]

    outlets: dict[str, dict[str, float]] = {"raffinate": dict(raffinate)}
    washed_extract = combined_extract
    for index, wash_volume_L in enumerate(
        contactor.wash_aqueous_volumes_L,
        start=1,
    ):
        report = _contact_stage(
            request=request,
            stage_id=f"wash_{index}",
            mode="wash",
            total_amounts_mol=washed_extract,
            input_organic_amounts_mol=washed_extract,
            aqueous_volume_L=wash_volume_L,
            organic_volume_L=combined_organic_volume_L,
            stage_efficiency=contactor.wash_stage_efficiency,
            entrainment_fraction=contactor.wash_entrainment_fraction,
            continuous_phase_policy=contactor.wash_continuous_phase,
        )
        reports.append(report)
        washed_extract = dict(report.organic_amounts_mol)
        outlets[f"wash_{index}"] = dict(report.aqueous_amounts_mol)
        combined_organic_volume_L = report.final_phase_volumes_L["organic"]
    outlets["extract"] = dict(washed_extract)

    material_error = _balance_error(request.feed_amounts_mol, outlets)
    if material_error > contactor.material_balance_tolerance_mol:
        raise RuntimeError(
            f"train material balance error exceeds tolerance: {material_error:.6g} mol"
        )
    target_feed = request.feed_amounts_mol[request.target_component]
    target_recovery = washed_extract[request.target_component] / target_feed
    extract_total = sum(washed_extract.values())
    target_purity = (
        0.0 if extract_total <= 0.0 else washed_extract[request.target_component] / extract_total
    )
    impurity_feed = sum(
        amount
        for component_id, amount in request.feed_amounts_mol.items()
        if component_id != request.target_component
    )
    impurity_extract = sum(
        amount
        for component_id, amount in washed_extract.items()
        if component_id != request.target_component
    )
    impurity_rejection = 1.0 if impurity_feed <= 0.0 else 1.0 - impurity_extract / impurity_feed
    warnings = tuple(dict.fromkeys(warning for report in reports for warning in report.warnings))
    provenance = (
        "coupled pre-contact TPD-style stability and gamma-corrected distribution solve",
        "per-component extraction, wash, and entrainment material ledgers",
        (f"PhasePy {PHASEPY_COMMIT}:{PHASEPY_STABILITY_PATH} stability workflow convention"),
        f"PhasePy {PHASEPY_COMMIT}:{PHASEPY_LLE_PATH} LLE solve convention",
        f"distribution profile {request.distribution_model.provenance_id}",
    )
    return StabilityAwareExtractionResult(
        model_id=PHASE_EQUILIBRIUM_MODEL_ID,
        distribution_model_id=request.distribution_model.model_id,
        distribution_provenance_id=request.distribution_model.provenance_id,
        temperature_K=request.temperature_K,
        target_component=request.target_component,
        feed_amounts_mol=dict(request.feed_amounts_mol),
        outlets=outlets,
        stage_reports=tuple(reports),
        target_recovery=target_recovery,
        target_purity=target_purity,
        impurity_rejection=impurity_rejection,
        material_balance_error_mol=material_error,
        maximum_stage_material_balance_error_mol=max(
            report.material_balance_error_mol for report in reports
        ),
        minimum_tpd_like=min(
            cast(float, report.stability_diagnostic["minimum_tpd_like"]) for report in reports
        ),
        maximum_distribution_residual=max(report.distribution_residual for report in reports),
        entrained_volume_L=sum(report.entrained_volume_L for report in reports),
        all_stages_two_liquid=all(report.phase_status == "two_liquid" for report in reports),
        all_stages_converged=True,
        warnings=warnings,
        provenance=provenance,
    )


def _contact_stage(
    *,
    request: StabilityAwareExtractionRequest,
    stage_id: str,
    mode: str,
    total_amounts_mol: Mapping[str, float],
    input_organic_amounts_mol: Mapping[str, float],
    aqueous_volume_L: float,
    organic_volume_L: float,
    stage_efficiency: float,
    entrainment_fraction: float,
    continuous_phase_policy: ContinuousPhasePolicy,
) -> StabilityAwareContactReport:
    component_ids = tuple(request.feed_amounts_mol)
    total = _amounts(total_amounts_mol, label=f"{stage_id}.total_amounts_mol")
    input_organic = _amounts(
        input_organic_amounts_mol,
        label=f"{stage_id}.input_organic_amounts_mol",
        require_positive_total=False,
    )
    if tuple(total) != component_ids or tuple(input_organic) != component_ids:
        raise ValueError("stage ledgers must exactly match request component ids")
    if any(input_organic[key] > total[key] for key in component_ids):
        raise ValueError("input organic amounts cannot exceed stage totals")
    if aqueous_volume_L + organic_volume_L > request.contactor.maximum_contact_volume_L:
        raise ValueError(f"{stage_id} contact exceeds maximum_contact_volume_L")

    stability = lle_phase_stability_diagnostic(
        total,
        partition_coefficients=(request.distribution_model.intrinsic_partition_coefficients),
        aqueous_volume_L=aqueous_volume_L,
        organic_volume_L=organic_volume_L,
        activity_model=request.stability_activity_model,
        temperature_K=request.temperature_K,
        initialization_policy=f"{stage_id}:partition_weighted",
        stage_efficiency=1.0,
        tpd_tolerance=request.contactor.tpd_tolerance,
    )
    no_phase_split_drive = "no_partition_or_nonideality_drive" in stability.warnings
    if stability.phase_status != "two_liquid" or no_phase_split_drive:
        raise ValueError(
            f"{stage_id} is outside the two-liquid domain: "
            f"minimum_tpd_like={stability.minimum_tpd_like:.6g}"
        )

    distribution = dict(request.distribution_model.intrinsic_partition_coefficients)
    aqueous_gamma = dict.fromkeys(component_ids, 1.0)
    organic_gamma = dict.fromkeys(component_ids, 1.0)
    residual = float("inf")
    iterations = 0
    for iteration in range(1, request.contactor.max_iterations + 1):
        iterations = iteration
        equilibrium_organic = {
            component_id: total[component_id]
            * ideal_organic_fraction(
                distribution[component_id],
                aqueous_volume_L=aqueous_volume_L,
                organic_volume_L=organic_volume_L,
            )
            for component_id in component_ids
        }
        equilibrium_aqueous = {
            component_id: total[component_id] - equilibrium_organic[component_id]
            for component_id in component_ids
        }
        aqueous_gamma = _activity_coefficients(
            request.distribution_model.aqueous_activity_model,
            equilibrium_aqueous,
            temperature_K=request.temperature_K,
        )
        organic_gamma = _activity_coefficients(
            request.distribution_model.organic_activity_model,
            equilibrium_organic,
            temperature_K=request.temperature_K,
        )
        updated = {
            component_id: request.distribution_model.intrinsic_partition_coefficients[component_id]
            * aqueous_gamma[component_id]
            / organic_gamma[component_id]
            for component_id in component_ids
        }
        if any(not isfinite(value) or not 1.0e-12 <= value <= 1.0e12 for value in updated.values()):
            raise RuntimeError(f"{stage_id} activity correction leaves the declared D range")
        residual = max(abs(log(updated[key] / distribution[key])) for key in component_ids)
        if residual <= request.contactor.distribution_tolerance:
            distribution = updated
            break
        distribution = {key: (distribution[key] * updated[key]) ** 0.5 for key in component_ids}
    else:
        raise RuntimeError(
            f"{stage_id} distribution solve did not converge after "
            f"{iterations} iterations; log residual={residual:.6g}"
        )

    equilibrium_organic = {
        component_id: total[component_id]
        * ideal_organic_fraction(
            distribution[component_id],
            aqueous_volume_L=aqueous_volume_L,
            organic_volume_L=organic_volume_L,
        )
        for component_id in component_ids
    }
    organic = {
        component_id: input_organic[component_id]
        + stage_efficiency * (equilibrium_organic[component_id] - input_organic[component_id])
        for component_id in component_ids
    }
    aqueous = {
        component_id: total[component_id] - organic[component_id] for component_id in component_ids
    }
    continuous_phase = _continuous_phase(
        continuous_phase_policy,
        aqueous_volume_L=aqueous_volume_L,
        organic_volume_L=organic_volume_L,
    )
    dispersed_phase = "aqueous" if continuous_phase == "organic" else "organic"
    entrained_amounts = dict.fromkeys(component_ids, 0.0)
    entrained_volume = 0.0
    entrained_from: str | None = None
    entrained_to: str | None = None
    if entrainment_fraction > 0.0:
        entrained_from = dispersed_phase
        entrained_to = continuous_phase
        source = aqueous if dispersed_phase == "aqueous" else organic
        target = organic if continuous_phase == "organic" else aqueous
        source, target, entrained_amounts = entrain_liquid_amounts(
            source, target, entrainment_fraction)
        if dispersed_phase == "aqueous":
            aqueous, organic = source, target
        else:
            organic, aqueous = source, target
        dispersed_volume = aqueous_volume_L if dispersed_phase == "aqueous" else organic_volume_L
        entrained_volume = entrainment_fraction * dispersed_volume

    final_volumes = {
        "aqueous": aqueous_volume_L,
        "organic": organic_volume_L,
    }
    final_volumes[dispersed_phase] -= entrained_volume
    final_volumes[continuous_phase] += entrained_volume
    recovery = {
        component_id: organic[component_id] / total[component_id]
        if total[component_id] > 0.0
        else 0.0
        for component_id in component_ids
    }
    material_error = _balance_error(total, {"organic": organic, "aqueous": aqueous})
    if material_error > request.contactor.material_balance_tolerance_mol:
        raise RuntimeError(
            f"{stage_id} material balance error exceeds tolerance: {material_error:.6g} mol"
        )
    warnings = list(stability.warnings)
    volume_ratio = max(aqueous_volume_L, organic_volume_L) / min(
        aqueous_volume_L,
        organic_volume_L,
    )
    if volume_ratio <= 1.2:
        warnings.append("near_phase_continuity_inversion")
    if continuous_phase_policy == "auto":
        warnings.append(f"auto_continuous_phase_{continuous_phase}")
    if entrainment_fraction > 0.0:
        warnings.append(f"{dispersed_phase}_entrainment_into_{continuous_phase}")
    return StabilityAwareContactReport(
        stage_id=stage_id,
        mode=mode,
        phase_status=stability.phase_status,
        aqueous_volume_L=aqueous_volume_L,
        organic_volume_L=organic_volume_L,
        final_phase_volumes_L=final_volumes,
        continuous_phase=continuous_phase,
        dispersed_phase=dispersed_phase,
        input_amounts_mol=dict(total),
        input_organic_amounts_mol=dict(input_organic),
        organic_amounts_mol=organic,
        aqueous_amounts_mol=aqueous,
        distribution_coefficients=distribution,
        aqueous_activity_coefficients=aqueous_gamma,
        organic_activity_coefficients=organic_gamma,
        recovery_to_organic=recovery,
        entrained_amounts_mol=entrained_amounts,
        entrained_volume_L=entrained_volume,
        entrained_from_phase=entrained_from,
        entrained_to_phase=entrained_to,
        stage_efficiency=stage_efficiency,
        iterations=iterations,
        distribution_residual=residual,
        material_balance_error_mol=material_error,
        stability_diagnostic=stability.to_dict(),
        warnings=tuple(dict.fromkeys(warnings)),
    )


def _activity_coefficients(
    model: ActivityModelSpec | None,
    amounts: Mapping[str, float],
    *,
    temperature_K: float,
) -> dict[str, float]:
    if model is None:
        return dict.fromkeys(amounts, 1.0)
    result = activity_coefficients(
        model,
        _composition(amounts),
        temperature_K=temperature_K,
    )
    if any(value <= 0.0 or not isfinite(value) for value in result.values()):
        raise RuntimeError("activity model returned non-positive or non-finite gamma")
    return result


def _continuous_phase(
    policy: ContinuousPhasePolicy,
    *,
    aqueous_volume_L: float,
    organic_volume_L: float,
) -> str:
    if policy != "auto":
        return policy
    return "organic" if organic_volume_L >= aqueous_volume_L else "aqueous"


def stability_aware_lle_model_card() -> ModelCard:
    return ModelCard(
        model_id=PHASE_EQUILIBRIUM_MODEL_ID,
        module_id="separations",
        title="Stability-Gated Activity-Corrected LLE Train",
        maturity=MaturityLevel.PROFESSIONAL_CANDIDATE,
        summary=(
            "A bounded multistage extraction and wash model that requires an "
            "explicit two-liquid stability result before each contact, iterates "
            "aqueous/organic activity corrections, and closes directional "
            "entrainment and component ledgers."
        ),
        equations=(
            "D_i = K_i^0 gamma_i^aqueous / gamma_i^organic",
            "n_i^org,eq/n_i = D_i V_org / (D_i V_org + V_aq)",
            "n_i^org = n_i^org,in + eta(n_i^org,eq - n_i^org,in)",
            "TPD_like,min < -tolerance is required before a liquid split",
            "n_i^entrained = f_entrainment n_i^dispersed",
            "n_i,feed = sum_outlets n_i,outlet",
        ),
        assumptions=(
            "The supplied distribution profile and activity models are valid "
            "at the declared temperature.",
            "Tracked components are dilute solutes; bulk solvent identities "
            "define aqueous and organic phase volumes.",
            "Each extraction stage uses fresh organic solvent and each wash "
            "stage uses fresh aqueous liquid.",
            "Stage efficiency is a linear approach from the declared incoming "
            "phase allocation to equilibrium.",
            "The larger phase is continuous under auto policy; an explicit "
            "policy can override that hydrodynamic heuristic.",
        ),
        validity_limits=(
            "At least two tracked components and intrinsic partition "
            "coefficients in [1e-12, 1e12] are required.",
            "At most 20 extraction and 20 wash stages are supported, within "
            "the declared contactor volume.",
            "Every contact must pass the bounded TPD-style two-liquid diagnostic.",
            "Activity-corrected distribution iteration must converge within "
            "the declared tolerance and iteration budget.",
            "This model does not predict solvent mutual solubility, density, "
            "interfacial tension, emulsion breakup, or mass-transfer coefficients.",
        ),
        failure_modes=(
            "Single-liquid classification fails explicitly instead of "
            "fabricating an extraction split.",
            "Invalid components, volumes, efficiencies, entrainment, "
            "temperature, or activity-model domains fail before evaluation.",
            "Non-converged activity iteration and material-balance residuals "
            "above tolerance fail the provider result.",
            "Near-equal phase volumes emit a phase-continuity-inversion warning; "
            "auto continuity is not a mechanistic hydrodynamics model.",
        ),
        units={
            "component amount": "mol",
            "phase/contact volume": "L",
            "temperature": "K",
            "partition/activity/stage efficiency/entrainment": "dimensionless",
            "material balance residual": "mol",
        },
        reference_reading=(
            f"reference_repos/phasepy/{PHASEPY_LLE_PATH} at {PHASEPY_COMMIT}: "
            "LLE successive-substitution conventions",
            f"reference_repos/phasepy/{PHASEPY_STABILITY_PATH} at "
            f"{PHASEPY_COMMIT}: stability initialization conventions",
            "Michelsen tangent-plane distance phase-stability criterion",
            "IDAES component material-balance and explicit inlet/outlet stream conventions",
        ),
        validation_evidence=(
            ValidationEvidence(
                evidence_id="lle-ideal-multistage-identity",
                evidence_type="analytical_test",
                description=(
                    "Checks fresh-solvent stage recovery against the closed-form "
                    "ideal distribution identity."
                ),
                status="implemented",
                command_or_path="tests/test_phase_equilibrium_units.py",
                tolerance="1e-12 relative and component closure <= 1e-10 mol",
            ),
            ValidationEvidence(
                evidence_id="lle-stability-convergence-and-inversion-domain",
                evidence_type="domain_sweep",
                description=(
                    "Covers single/two-liquid classification, activity iteration "
                    "failure, extreme K, wash tradeoffs, and both entrainment "
                    "directions."
                ),
                status="implemented",
                command_or_path="tests/test_phase_equilibrium_units.py",
                tolerance="declared solver and material-balance tolerances",
            ),
        ),
        model_limit_notes=(
            "Professional-candidate maturity applies only to this bounded "
            "solute-partition contact model.",
            "The TPD-style gate is an auditable benchmark diagnostic, not a "
            "rigorous global Gibbs phase-stability minimizer.",
            "Phase continuity and entrainment are declared policies, not CFD "
            "or scale-up predictions.",
            "WF-110 must review runtime state mapping and refreeze affected "
            "benchmarks before replacing the v0.3 providers.",
        ),
        intended_use=(
            "Budgeted solvent selection and extraction/wash sequencing benchmarks.",
            "Runtime provider candidate for mix and wash operations in a new World Law.",
            "Auditable recovery, purity, stability, entrainment, and conservation diagnostics.",
        ),
    )


__all__ = [
    "PHASEPY_COMMIT",
    "PHASEPY_LLE_PATH",
    "PHASEPY_STABILITY_PATH",
    "PHASE_EQUILIBRIUM_MODEL_ID",
    "LLEContactorSpec",
    "StabilityAwareContactReport",
    "StabilityAwareExtractionRequest",
    "StabilityAwareExtractionResult",
    "StrictLiquidSystemSpec",
    "ideal_organic_fraction",
    "simulate_stability_aware_extraction",
    "stability_aware_lle_model_card",
]
