"""Finite-capacity sorbent drying with explicit liquid and solid ledgers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from math import expm1, isfinite
from typing import Any

from chemworld.physchem.maturity import (
    MaturityLevel,
    ModelCard,
    ValidationEvidence,
)

DRYING_MODEL_ID = "chemworld_sorbent_drying_vnext"
IDAES_COMMIT = "4275c45bfa76cd5b05926beaa8eee58f7b0b05e8"
IDAES_ADSORPTION_PATH = (
    "idaes/models_extra/temperature_swing_adsorption/fixed_bed_tsa0d.py"
)
_EQUILIBRIUM_RELATIVE_TOLERANCE = 1.0e-13
_MAX_EQUILIBRIUM_ITERATIONS = 256


def _finite_nonnegative(value: float, label: str) -> float:
    resolved = float(value)
    if not isfinite(resolved) or resolved < 0.0:
        raise ValueError(f"{label} must be finite and nonnegative")
    return resolved


def _component_map(
    values: Mapping[str, float],
    *,
    label: str,
    require_positive_total: bool = False,
    require_positive_values: bool = False,
) -> dict[str, float]:
    resolved: dict[str, float] = {}
    for component_id, raw_value in values.items():
        key = str(component_id).strip()
        if not key:
            raise ValueError(f"{label} component ids cannot be empty")
        if key in resolved:
            raise ValueError(f"{label} component ids cannot collide after normalization")
        value = _finite_nonnegative(float(raw_value), f"{label}[{key!r}]")
        if require_positive_values and value <= 0.0:
            raise ValueError(f"{label}[{key!r}] must be positive")
        resolved[key] = value
    if require_positive_total and sum(resolved.values()) <= 0.0:
        raise ValueError(f"{label} must contain a positive component amount")
    return resolved


@dataclass(frozen=True)
class SorbentBedSpec:
    """Declared sorbent inventory and shared-site equilibrium parameters."""

    sorbent_id: str
    sorbent_mass_kg: float
    site_capacity_mol_per_kg: float
    affinity_L_per_mol: Mapping[str, float]
    mass_transfer_rate_per_s: float
    initial_loading_mol_per_kg: Mapping[str, float] = field(default_factory=dict)
    max_liquid_volume_L: float | None = None

    def __post_init__(self) -> None:
        if not self.sorbent_id.strip():
            raise ValueError("sorbent_id cannot be empty")
        mass = _finite_nonnegative(self.sorbent_mass_kg, "sorbent_mass_kg")
        capacity = _finite_nonnegative(
            self.site_capacity_mol_per_kg,
            "site_capacity_mol_per_kg",
        )
        affinity = _component_map(
            self.affinity_L_per_mol,
            label="affinity_L_per_mol",
            require_positive_values=True,
        )
        if not affinity:
            raise ValueError("affinity_L_per_mol must declare at least one component")
        rate = _finite_nonnegative(
            self.mass_transfer_rate_per_s,
            "mass_transfer_rate_per_s",
        )
        loading = _component_map(
            self.initial_loading_mol_per_kg,
            label="initial_loading_mol_per_kg",
        )
        unknown_loading = sorted(set(loading) - set(affinity))
        if unknown_loading:
            raise ValueError(
                "initial loading components require an affinity parameter: "
                f"{unknown_loading}"
            )
        if sum(loading.values()) > capacity + 1.0e-12:
            raise ValueError("initial sorbent loading exceeds shared site capacity")
        max_volume = self.max_liquid_volume_L
        if max_volume is not None:
            max_volume = _finite_nonnegative(max_volume, "max_liquid_volume_L")
        object.__setattr__(self, "sorbent_mass_kg", mass)
        object.__setattr__(self, "site_capacity_mol_per_kg", capacity)
        object.__setattr__(self, "affinity_L_per_mol", affinity)
        object.__setattr__(self, "mass_transfer_rate_per_s", rate)
        object.__setattr__(self, "initial_loading_mol_per_kg", loading)
        object.__setattr__(self, "max_liquid_volume_L", max_volume)


@dataclass(frozen=True)
class SorbentDryingRequest:
    """One well-mixed liquid contact followed by removal of the spent sorbent."""

    wet_liquid_amounts_mol: Mapping[str, float]
    liquid_volume_L: float
    drying_component_ids: tuple[str, ...]
    contact_time_s: float
    sorbent: SorbentBedSpec
    retained_liquid_volume_L: float = 0.0
    product_component_id: str | None = None
    target_residual_drying_fraction: float = 0.05
    balance_tolerance: float = 1.0e-10

    def __post_init__(self) -> None:
        feed = _component_map(
            self.wet_liquid_amounts_mol,
            label="wet_liquid_amounts_mol",
            require_positive_total=True,
        )
        volume = float(self.liquid_volume_L)
        if not isfinite(volume) or volume <= 0.0:
            raise ValueError("liquid_volume_L must be finite and positive")
        drying_ids = tuple(str(value).strip() for value in self.drying_component_ids)
        if not drying_ids or any(not value for value in drying_ids):
            raise ValueError("drying_component_ids must contain non-empty ids")
        if len(set(drying_ids)) != len(drying_ids):
            raise ValueError("drying_component_ids cannot contain duplicates")
        missing = sorted(set(drying_ids) - set(feed))
        if missing:
            raise ValueError(f"drying components are absent from wet liquid: {missing}")
        zero_inventory = [
            component_id
            for component_id in drying_ids
            if feed[component_id] <= 0.0
        ]
        if zero_inventory:
            raise ValueError(
                "drying components must have positive wet-liquid inventory: "
                f"{zero_inventory}"
            )
        contact_time = _finite_nonnegative(self.contact_time_s, "contact_time_s")
        retained_volume = _finite_nonnegative(
            self.retained_liquid_volume_L,
            "retained_liquid_volume_L",
        )
        tolerance = float(self.balance_tolerance)
        if not isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("balance_tolerance must be finite and positive")
        endpoint = float(self.target_residual_drying_fraction)
        if not isfinite(endpoint) or not 0.0 <= endpoint <= 1.0:
            raise ValueError("target_residual_drying_fraction must lie in [0, 1]")
        if retained_volume > volume + tolerance:
            raise ValueError("retained_liquid_volume_L cannot exceed liquid_volume_L")
        if (
            self.sorbent.max_liquid_volume_L is not None
            and volume > self.sorbent.max_liquid_volume_L + tolerance
        ):
            raise ValueError("liquid_volume_L exceeds the sorbent contactor maximum")
        product_id = self.product_component_id
        if product_id is not None:
            product_id = str(product_id).strip()
            if not product_id:
                raise ValueError("product_component_id cannot be empty")
            if product_id not in feed:
                raise ValueError("product_component_id must be present in wet liquid")
            if feed[product_id] <= 0.0:
                raise ValueError("product_component_id must have positive inventory")
            if product_id in drying_ids:
                raise ValueError("product_component_id cannot also be a drying component")
        object.__setattr__(self, "wet_liquid_amounts_mol", feed)
        object.__setattr__(self, "liquid_volume_L", volume)
        object.__setattr__(self, "drying_component_ids", drying_ids)
        object.__setattr__(self, "contact_time_s", contact_time)
        object.__setattr__(self, "retained_liquid_volume_L", retained_volume)
        object.__setattr__(self, "product_component_id", product_id)
        object.__setattr__(self, "target_residual_drying_fraction", endpoint)
        object.__setattr__(self, "balance_tolerance", tolerance)


@dataclass(frozen=True)
class _EquilibriumSolution:
    loading_mol_per_kg: dict[str, float]
    liquid_amounts_mol: dict[str, float]
    denominator: float
    residual: float
    iterations: int


def _solve_competitive_equilibrium(
    total_inventory_mol: Mapping[str, float],
    *,
    liquid_volume_L: float,
    sorbent_mass_kg: float,
    site_capacity_mol_per_kg: float,
    affinity_L_per_mol: Mapping[str, float],
) -> _EquilibriumSolution:
    """Solve the extended-Langmuir material balance through one scalar root."""

    affinity = dict(affinity_L_per_mol)

    def concentrations(denominator: float) -> dict[str, float]:
        values: dict[str, float] = {}
        for component_id, coefficient in affinity.items():
            total = float(total_inventory_mol.get(component_id, 0.0))
            effective_volume = liquid_volume_L + (
                sorbent_mass_kg
                * site_capacity_mol_per_kg
                * coefficient
                / denominator
            )
            values[component_id] = total / effective_volume
        return values

    def equation(denominator: float) -> float:
        values = concentrations(denominator)
        return (
            1.0
            + sum(affinity[key] * values[key] for key in affinity)
            - denominator
        )

    upper = 1.0 + sum(
        coefficient
        * float(total_inventory_mol.get(component_id, 0.0))
        / liquid_volume_L
        for component_id, coefficient in affinity.items()
    )
    if upper <= 1.0 + _EQUILIBRIUM_RELATIVE_TOLERANCE:
        denominator = 1.0
        iterations = 0
    else:
        lower = 1.0
        iterations = 0
        denominator = 0.5 * (lower + upper)
        for iteration in range(1, _MAX_EQUILIBRIUM_ITERATIONS + 1):
            iterations = iteration
            denominator = 0.5 * (lower + upper)
            value = equation(denominator)
            if abs(value) <= _EQUILIBRIUM_RELATIVE_TOLERANCE * max(
                1.0,
                denominator,
            ):
                break
            if value > 0.0:
                lower = denominator
            else:
                upper = denominator
            if upper - lower <= _EQUILIBRIUM_RELATIVE_TOLERANCE * max(
                1.0,
                denominator,
            ):
                denominator = 0.5 * (lower + upper)
                break
        else:
            raise RuntimeError("competitive sorption equilibrium did not converge")
    liquid_concentrations = concentrations(denominator)
    loading = {
        component_id: (
            site_capacity_mol_per_kg
            * coefficient
            * liquid_concentrations[component_id]
            / denominator
        )
        for component_id, coefficient in affinity.items()
    }
    liquid_amounts = {
        component_id: liquid_volume_L * liquid_concentrations[component_id]
        for component_id in affinity
    }
    return _EquilibriumSolution(
        loading_mol_per_kg=loading,
        liquid_amounts_mol=liquid_amounts,
        denominator=denominator,
        residual=abs(equation(denominator)),
        iterations=iterations,
    )


@dataclass(frozen=True)
class SorbentDryingResult:
    model_id: str
    sorbent_id: str
    wet_liquid_amounts_mol: dict[str, float]
    wet_liquid_volume_L: float
    initial_sorbent_loading_mol_per_kg: dict[str, float]
    initial_sorbent_amounts_mol: dict[str, float]
    equilibrium_loading_mol_per_kg: dict[str, float]
    final_sorbent_loading_mol_per_kg: dict[str, float]
    final_sorbent_amounts_mol: dict[str, float]
    net_sorption_amounts_mol: dict[str, float]
    dried_liquid_amounts_mol: dict[str, float]
    dried_liquid_volume_L: float
    retained_liquid_amounts_mol: dict[str, float]
    retained_liquid_volume_L: float
    spent_sorbent_inventory_mol: dict[str, float]
    drying_component_removal_fraction: dict[str, float]
    residual_drying_component_fraction: float
    target_residual_drying_fraction: float
    endpoint_met: bool
    product_recovery: float | None
    contact_fraction_of_equilibrium: float
    equilibrium_denominator: float
    equilibrium_residual: float
    equilibrium_iterations: int
    component_balance_error_mol: dict[str, float]
    material_balance_error_mol: float
    volume_balance_error_L: float
    warnings: tuple[str, ...]
    provenance: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "sorbent_id": self.sorbent_id,
            "wet_liquid_amounts_mol": dict(self.wet_liquid_amounts_mol),
            "wet_liquid_volume_L": self.wet_liquid_volume_L,
            "initial_sorbent_loading_mol_per_kg": dict(
                self.initial_sorbent_loading_mol_per_kg
            ),
            "initial_sorbent_amounts_mol": dict(self.initial_sorbent_amounts_mol),
            "equilibrium_loading_mol_per_kg": dict(
                self.equilibrium_loading_mol_per_kg
            ),
            "final_sorbent_loading_mol_per_kg": dict(
                self.final_sorbent_loading_mol_per_kg
            ),
            "final_sorbent_amounts_mol": dict(self.final_sorbent_amounts_mol),
            "net_sorption_amounts_mol": dict(self.net_sorption_amounts_mol),
            "dried_liquid_amounts_mol": dict(self.dried_liquid_amounts_mol),
            "dried_liquid_volume_L": self.dried_liquid_volume_L,
            "retained_liquid_amounts_mol": dict(self.retained_liquid_amounts_mol),
            "retained_liquid_volume_L": self.retained_liquid_volume_L,
            "spent_sorbent_inventory_mol": dict(self.spent_sorbent_inventory_mol),
            "drying_component_removal_fraction": dict(
                self.drying_component_removal_fraction
            ),
            "residual_drying_component_fraction": (
                self.residual_drying_component_fraction
            ),
            "target_residual_drying_fraction": (
                self.target_residual_drying_fraction
            ),
            "endpoint_met": self.endpoint_met,
            "product_recovery": self.product_recovery,
            "contact_fraction_of_equilibrium": self.contact_fraction_of_equilibrium,
            "equilibrium_denominator": self.equilibrium_denominator,
            "equilibrium_residual": self.equilibrium_residual,
            "equilibrium_iterations": self.equilibrium_iterations,
            "component_balance_error_mol": dict(self.component_balance_error_mol),
            "material_balance_error_mol": self.material_balance_error_mol,
            "volume_balance_error_L": self.volume_balance_error_L,
            "warnings": list(self.warnings),
            "provenance": list(self.provenance),
        }


def simulate_sorbent_drying(request: SorbentDryingRequest) -> SorbentDryingResult:
    """Contact a wet liquid with sorbent, then remove sorbent and retained liquid."""

    sorbent = request.sorbent
    component_ids = sorted(
        set(request.wet_liquid_amounts_mol)
        | set(sorbent.affinity_L_per_mol)
        | set(sorbent.initial_loading_mol_per_kg)
    )
    initial_loading = {
        component_id: sorbent.initial_loading_mol_per_kg.get(component_id, 0.0)
        for component_id in sorbent.affinity_L_per_mol
    }
    initial_sorbent_amounts = {
        component_id: loading * sorbent.sorbent_mass_kg
        for component_id, loading in initial_loading.items()
    }
    total_inventory = {
        component_id: (
            request.wet_liquid_amounts_mol.get(component_id, 0.0)
            + initial_sorbent_amounts.get(component_id, 0.0)
        )
        for component_id in component_ids
    }
    equilibrium = _solve_competitive_equilibrium(
        total_inventory,
        liquid_volume_L=request.liquid_volume_L,
        sorbent_mass_kg=sorbent.sorbent_mass_kg,
        site_capacity_mol_per_kg=sorbent.site_capacity_mol_per_kg,
        affinity_L_per_mol=sorbent.affinity_L_per_mol,
    )
    contact_fraction = -expm1(
        -sorbent.mass_transfer_rate_per_s * request.contact_time_s
    )
    from chemworld.physchem.sorbent_kinetics import isothermal_sorbent_loading

    final_loading = isothermal_sorbent_loading(sorbent, request.wet_liquid_amounts_mol,
        volume_L=request.liquid_volume_L, duration_s=request.contact_time_s)
    if sum(final_loading.values()) > (
        sorbent.site_capacity_mol_per_kg + request.balance_tolerance
    ):
        raise RuntimeError("transient sorbent loading exceeded shared site capacity")
    final_sorbent_amounts = {
        component_id: loading * sorbent.sorbent_mass_kg
        for component_id, loading in final_loading.items()
    }
    net_sorption = {
        component_id: (
            final_sorbent_amounts[component_id]
            - initial_sorbent_amounts[component_id]
        )
        for component_id in sorbent.affinity_L_per_mol
    }
    post_contact_liquid: dict[str, float] = {}
    for component_id in component_ids:
        amount = total_inventory[component_id] - final_sorbent_amounts.get(
            component_id,
            0.0,
        )
        if amount < -request.balance_tolerance:
            raise RuntimeError(
                f"sorption produced negative liquid inventory for {component_id!r}"
            )
        post_contact_liquid[component_id] = max(0.0, amount)
    retention_fraction = request.retained_liquid_volume_L / request.liquid_volume_L
    retained_liquid = {
        component_id: amount * retention_fraction
        for component_id, amount in post_contact_liquid.items()
    }
    dried_liquid = {
        component_id: amount - retained_liquid[component_id]
        for component_id, amount in post_contact_liquid.items()
    }
    dried_volume = request.liquid_volume_L - request.retained_liquid_volume_L
    spent_inventory = {
        component_id: (
            final_sorbent_amounts.get(component_id, 0.0)
            + retained_liquid.get(component_id, 0.0)
        )
        for component_id in component_ids
    }

    component_errors: dict[str, float] = {}
    for component_id in component_ids:
        incoming = (
            request.wet_liquid_amounts_mol.get(component_id, 0.0)
            + initial_sorbent_amounts.get(component_id, 0.0)
        )
        outgoing = (
            dried_liquid.get(component_id, 0.0)
            + retained_liquid.get(component_id, 0.0)
            + final_sorbent_amounts.get(component_id, 0.0)
        )
        component_errors[component_id] = abs(incoming - outgoing)
    material_error = sum(component_errors.values())
    volume_error = abs(
        request.liquid_volume_L
        - dried_volume
        - request.retained_liquid_volume_L
    )
    if material_error > request.balance_tolerance or volume_error > request.balance_tolerance:
        raise RuntimeError(
            "sorbent drying control volume failed closure: "
            f"material={material_error}, volume={volume_error}"
        )

    removal_fractions = {
        component_id: (
            1.0
            - dried_liquid.get(component_id, 0.0)
            / request.wet_liquid_amounts_mol[component_id]
        )
        for component_id in request.drying_component_ids
    }
    initial_drying_total = sum(
        request.wet_liquid_amounts_mol[component_id]
        for component_id in request.drying_component_ids
    )
    dried_drying_total = sum(
        dried_liquid.get(component_id, 0.0)
        for component_id in request.drying_component_ids
    )
    residual_drying_fraction = dried_drying_total / initial_drying_total
    endpoint_met = (
        residual_drying_fraction
        <= request.target_residual_drying_fraction + request.balance_tolerance
    )
    product_recovery: float | None = None
    if request.product_component_id is not None:
        product_input = (
            request.wet_liquid_amounts_mol[request.product_component_id]
            + initial_sorbent_amounts.get(request.product_component_id, 0.0)
        )
        product_recovery = (
            dried_liquid.get(request.product_component_id, 0.0)
            / product_input
        )

    warnings: list[str] = [
        "contact_fraction_of_equilibrium is the legacy nominal single-rate time factor; "
        "actual loadings use instantaneous finite-fluid Langmuir/LDF integration"
    ]
    if sorbent.sorbent_mass_kg <= request.balance_tolerance:
        warnings.append(
            "zero sorbent mass: only mechanical liquid retention changes the "
            "recovered product stream"
        )
    if sorbent.site_capacity_mol_per_kg <= request.balance_tolerance:
        warnings.append("zero sorbent capacity: no equilibrium uptake is possible")
    if contact_fraction <= request.balance_tolerance:
        warnings.append("zero effective contact: sorbent loading remains at its initial state")
    for component_id in request.drying_component_ids:
        if net_sorption.get(component_id, 0.0) < -request.balance_tolerance:
            warnings.append(
                f"{component_id} desorbs from the initial sorbent inventory"
            )
    if (
        sorbent.site_capacity_mol_per_kg > request.balance_tolerance
        and sum(final_loading.values())
        / sorbent.site_capacity_mol_per_kg
        >= 0.95
    ):
        warnings.append("spent sorbent is at least 95% loaded on the declared site basis")
    if dried_volume <= request.balance_tolerance:
        warnings.append("all liquid is mechanically retained with the spent sorbent")
    if product_recovery is not None and product_recovery < 1.0 - request.balance_tolerance:
        warnings.append("product loss is explicit in sorption and retained-liquid ledgers")
    if not endpoint_met:
        warnings.append("declared residual drying-component endpoint was not met")

    return SorbentDryingResult(
        model_id=DRYING_MODEL_ID,
        sorbent_id=sorbent.sorbent_id,
        wet_liquid_amounts_mol=dict(request.wet_liquid_amounts_mol),
        wet_liquid_volume_L=request.liquid_volume_L,
        initial_sorbent_loading_mol_per_kg=initial_loading,
        initial_sorbent_amounts_mol=initial_sorbent_amounts,
        equilibrium_loading_mol_per_kg=equilibrium.loading_mol_per_kg,
        final_sorbent_loading_mol_per_kg=final_loading,
        final_sorbent_amounts_mol=final_sorbent_amounts,
        net_sorption_amounts_mol=net_sorption,
        dried_liquid_amounts_mol=dried_liquid,
        dried_liquid_volume_L=dried_volume,
        retained_liquid_amounts_mol=retained_liquid,
        retained_liquid_volume_L=request.retained_liquid_volume_L,
        spent_sorbent_inventory_mol=spent_inventory,
        drying_component_removal_fraction=removal_fractions,
        residual_drying_component_fraction=residual_drying_fraction,
        target_residual_drying_fraction=request.target_residual_drying_fraction,
        endpoint_met=endpoint_met,
        product_recovery=product_recovery,
        contact_fraction_of_equilibrium=contact_fraction,
        equilibrium_denominator=equilibrium.denominator,
        equilibrium_residual=equilibrium.residual,
        equilibrium_iterations=equilibrium.iterations,
        component_balance_error_mol=component_errors,
        material_balance_error_mol=material_error,
        volume_balance_error_L=volume_error,
        warnings=tuple(warnings),
        provenance=(
            "extended-Langmuir shared-site equilibrium with exact component inventory",
            "shared finite-fluid Langmuir/LDF contribution through CoupledSystem",
            (
                f"IDAES {IDAES_COMMIT}:{IDAES_ADSORPTION_PATH} extended-isotherm "
                "convention; ChemWorld uses an independent bounded liquid-contact ledger"
            ),
        ),
    )


def sorbent_drying_model_card() -> ModelCard:
    return ModelCard(
        model_id=DRYING_MODEL_ID,
        module_id="separations",
        title="Finite-Capacity Selective Sorbent Drying",
        maturity=MaturityLevel.REFERENCE_VALIDATED,
        summary=(
            "A well-mixed liquid/sorbent contact with competitive shared-site "
            "equilibrium, finite contact kinetics, initial loading, mechanical "
            "retention, and explicit spent-sorbent material closure."
        ),
        equations=(
            "q_i* = Q b_i C_i / (1 + sum_j b_j C_j)",
            "N_i,total = V C_i* + m_s q_i*",
            "q_i(t) = q_i,0 + (1 - exp(-k t)) (q_i* - q_i,0)",
            "N_i,in = N_i,dried + N_i,retained + m_s q_i(t)",
            "endpoint_met iff sum_i,dry N_i,dried / sum_i,dry N_i,wet <= target",
        ),
        assumptions=(
            "one isothermal, well-mixed liquid contact with a shared class of sorption sites",
            (
                "the isothermal contact supplies one rate to the shared local-concentration "
                "Langmuir/LDF contribution; native component rates use explicit "
                "finite-site constraints"
            ),
            (
                "liquid volume is additive and unchanged by sorption before "
                "declared mechanical retention"
            ),
            (
                "spent sorbent is removed after contact and both sorbed and "
                "retained-liquid inventories persist"
            ),
        ),
        validity_limits=(
            (
                "declared affinity and capacity parameters only; no sorbent is "
                "inferred from a trade name"
            ),
            "the legacy request is one isothermal liquid contact without a caloric/heat claim",
            (
                "no intraparticle distribution, axial bed, pressure drop or scale-up; "
                "native persistent-carrier regeneration uses the shared contribution "
                "with caloric data"
            ),
            (
                "equilibrium parameters must be fitted or justified for the "
                "solvent, temperature, and sorbent lot"
            ),
        ),
        failure_modes=(
            "negative, nonfinite, duplicate, missing, or over-capacity inputs are rejected",
            "liquid volume above the declared contactor maximum is rejected",
            (
                "equilibrium non-convergence or material/volume closure outside "
                "tolerance is a hard failure"
            ),
            "initially loaded sorbent can desorb and produces an explicit warning",
        ),
        units={
            "component amount": "mol",
            "liquid volume": "L",
            "sorbent mass": "kg",
            "site capacity and loading": "mol/kg",
            "affinity": "L/mol",
            "mass-transfer rate": "1/s",
            "time": "s",
        },
        reference_reading=(
            (
                f"IDAES {IDAES_COMMIT}:{IDAES_ADSORPTION_PATH} extended Sips "
                "loading, explicit saturation capacity, and adsorbent mass conventions"
            ),
            "extended-Langmuir competitive adsorption and linear-driving-force identities",
        ),
        validation_evidence=(
            ValidationEvidence(
                evidence_id="drying-single-component-closed-form",
                evidence_type="analytic_test",
                description=(
                    "The scalar material-balance solve is checked against the positive "
                    "root of the single-component Langmuir quadratic."
                ),
                status="implemented",
                reference_backend="closed-form quadratic equilibrium",
                command_or_path="tests/test_drying_units.py",
                tolerance="1e-11 mol/L and mol/kg",
            ),
            ValidationEvidence(
                evidence_id="drying-multicomponent-independent-solve",
                evidence_type="reference_algorithm_test",
                description=(
                    "A competitive case is compared with an independently posed "
                    "SciPy nonlinear least-squares material-balance solve."
                ),
                status="implemented",
                reference_backend="scipy.optimize.least_squares",
                command_or_path="tests/test_drying_units.py",
                tolerance="1e-10 mol and mol/kg",
            ),
            ValidationEvidence(
                evidence_id="drying-control-volume-closure",
                evidence_type="invariant_test",
                description=(
                    "Wet liquid plus initial sorbent inventory closes component-wise "
                    "to dried liquid, retained liquid, and final sorbent inventory."
                ),
                status="implemented",
                reference_backend="analytic control-volume identity",
                command_or_path="tests/test_drying_units.py",
                tolerance="1e-10 mol and L",
            ),
        ),
        model_limit_notes=(
            (
                "Reference validation applies to the bounded equilibrium/contact "
                "ledger, not sorbent selection or plant design."
            ),
            (
                "The native persistent-carrier path requires authored caloric data; "
                "the legacy disposable path draws from an explicit finite media kit."
            ),
        ),
        intended_use=(
            "Declared finite sorbent contact with shared native kinetic contribution",
            (
                "agent environments that must trade drying capacity against "
                "product loss and waste inventory"
            ),
            "component-closure, contact-time, capacity, and retention strategy evaluation",
        ),
    )


__all__ = [
    "DRYING_MODEL_ID",
    "IDAES_ADSORPTION_PATH",
    "IDAES_COMMIT",
    "SorbentBedSpec",
    "SorbentDryingRequest",
    "SorbentDryingResult",
    "simulate_sorbent_drying",
    "sorbent_drying_model_card",
]
