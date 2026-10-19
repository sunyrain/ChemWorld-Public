"""State-to-provider mappings for the vNext downstream runtime.

The professional providers deliberately accept explicit, unit-bearing request
objects instead of ``WorldState``.  This module is the only place that maps the
benchmark's compact phase ledger to those contracts.  Keeping the mapping here
makes it possible to audit that an operation really executed the declared
provider and prevents the old proxy equations from surviving as a hidden
fallback.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace

from chemworld.foundation.solvents import (
    HEAT_CAPACITY_RATIOS,
    RELATIVE_VOLATILITIES,
    SolventInventory,
)
from chemworld.physchem.concentration_units import (
    ConcentrationComponentSpec,
    VacuumConcentrationRequest,
    VacuumConcentrationResult,
    VacuumConcentratorSpec,
    simulate_vacuum_concentration,
)
from chemworld.physchem.distillation_units import (
    DistillationComponentSpec,
    DutyLimitedDistillationRequest,
    DutyLimitedDistillationResult,
    ShortcutColumnSpec,
    simulate_duty_limited_distillation,
)
from chemworld.physchem.drying_units import (
    SorbentBedSpec,
    SorbentDryingRequest,
    SorbentDryingResult,
    simulate_sorbent_drying,
)
from chemworld.physchem.transfer_units import (
    TransferEquipmentSpec,
    TransferRequest,
    TransferUnitResult,
    simulate_transfer,
)

_CARRIER_ID = "runtime_carrier"
_DRYING_COMPONENT_ID = "residual_water"
_PRODUCT_ID = "product"
_IMPURITY_ID = "impurity"


@dataclass(frozen=True)
class PhaseSlice:
    """Minimal selected-phase state consumed by downstream providers."""

    product_mol: float
    impurity_mol: float
    volume_L: float
    solvent_loss: float = 0.0

    @property
    def has_material(self) -> bool:
        return self.volume_L > 1.0e-12 and self.product_mol + self.impurity_mol > 1.0e-12


def run_sorbent_drying(phase: PhaseSlice, *, water_mol: float | None = None) -> SorbentDryingResult:
    """Map a selected phase to the finite-capacity sorbent provider."""

    wetting_inventory = max(
        phase.volume_L * (0.02 + 0.08 * min(max(phase.solvent_loss, 0.0), 1.0)),
        1.0e-8,
    )
    if water_mol is not None:
        wetting_inventory = water_mol
    feed = {
        _PRODUCT_ID: max(phase.product_mol, 0.0),
        _IMPURITY_ID: max(phase.impurity_mol, 0.0),
        _DRYING_COMPONENT_ID: wetting_inventory,
    }
    sorbent = SorbentBedSpec(
        sorbent_id="molecular_sieve_runtime_vnext",
        sorbent_mass_kg=max(0.002, 0.12 * phase.volume_L),
        site_capacity_mol_per_kg=12.0,
        affinity_L_per_mol={
            _PRODUCT_ID: 0.02,
            _IMPURITY_ID: 0.10,
            _DRYING_COMPONENT_ID: 240.0,
        },
        mass_transfer_rate_per_s=0.018,
        max_liquid_volume_L=max(phase.volume_L, 1.0e-6),
    )
    request = SorbentDryingRequest(
        wet_liquid_amounts_mol=feed,
        liquid_volume_L=phase.volume_L,
        drying_component_ids=(_DRYING_COMPONENT_ID,),
        contact_time_s=300.0,
        sorbent=sorbent,
        product_component_id=_PRODUCT_ID if phase.product_mol > 0.0 else None,
        target_residual_drying_fraction=0.35,
    )
    return simulate_sorbent_drying(request)


def run_vacuum_concentration(
    phase: PhaseSlice,
    *,
    initial_temperature_K: float,
    duration_s: float,
    solvents: SolventInventory | None = None,
    carrier_heat_capacity_J_L_K: float = 4167.0,
) -> VacuumConcentrationResult:
    """Map a selected phase to an energy-limited vacuum concentration run."""

    operating_temperature = max(float(initial_temperature_K), 323.15)
    carrier_mol = max(
        (phase.volume_L - 0.018 * (phase.product_mol + phase.impurity_mol)) / 0.018,
        1.0e-7,
    )
    feed = {
        _PRODUCT_ID: max(phase.product_mol, 0.0),
        _IMPURITY_ID: max(phase.impurity_mol, 0.0),
        _CARRIER_ID: carrier_mol,
    }
    specifications = {
        _PRODUCT_ID: ConcentrationComponentSpec(
            _PRODUCT_ID,
            120.0,
            1.0,
            52_000.0,
            155.0,
            0.018,
            operating_temperature,
            465.0,
            "runtime-vnext-bounded-product",
        ),
        _IMPURITY_ID: ConcentrationComponentSpec(
            _IMPURITY_ID,
            4_500.0,
            1.0,
            48_000.0,
            130.0,
            0.018,
            operating_temperature,
            455.0,
            "runtime-vnext-bounded-impurity",
        ),
        _CARRIER_ID: ConcentrationComponentSpec(
            _CARRIER_ID,
            82_000.0,
            1.0,
            40_700.0,
            75.0,
            0.018,
            operating_temperature,
            500.0,
            "runtime-vnext-bounded-carrier",
        ),
    }
    solvent_ids: tuple[str, ...] = (_CARRIER_ID,)
    if solvents is not None:
        del feed[_CARRIER_ID]
        del specifications[_CARRIER_ID]
        for key in (_PRODUCT_ID, _IMPURITY_ID):
            specifications[key] = replace(specifications[key], liquid_molar_volume_L_mol=0.0)
        solvent_ids = tuple(f"carrier-{i}" for i in range(4))
        for i, key in enumerate(solvent_ids):
            feed[key] = solvents.volumes_L[i] / 0.018
            specifications[key] = ConcentrationComponentSpec(
                key,
                82_000.0 * RELATIVE_VOLATILITIES[i],
                1.0,
                40_700.0,
                carrier_heat_capacity_J_L_K * 0.018 * HEAT_CAPACITY_RATIOS[i],
                0.018,
                operating_temperature,
                500.0,
                "finite-surrogate-carrier-v1",
            )
    equipment = VacuumConcentratorSpec(
        equipment_id="bench_vacuum_concentrator_vnext",
        max_working_volume_L=max(0.20, phase.volume_L * 1.2),
        minimum_residual_volume_L=min(1.0e-5, phase.volume_L * 0.01),
        max_heater_power_W=120.0,
        max_evaporation_rate_mol_s=0.01,
        condenser_recovery_fraction=0.96,
        minimum_pressure_Pa=5_000.0,
        maximum_temperature_K=440.0,
        maximum_duration_s=3_600.0,
        provenance_id="chemworld-runtime-vnext-concentrator-card",
    )
    request = VacuumConcentrationRequest(
        feed_amounts_mol=feed,
        component_specs=specifications,
        target_component_id=_PRODUCT_ID,
        solvent_component_ids=solvent_ids,
        initial_temperature_K=initial_temperature_K,
        operating_temperature_K=operating_temperature,
        pressure_Pa=30_000.0,
        duration_s=min(max(duration_s, 0.0), 3_600.0),
        heater_power_W=90.0,
        equipment=equipment,
        target_solvent_remaining_fraction=0.45,
        minimum_target_recovery=0.97,
    )
    return simulate_vacuum_concentration(request)


def run_bounded_transfer(phase: PhaseSlice, *, fraction: float) -> TransferUnitResult:
    """Map a selected phase through a finite heel and line-holdup transfer."""

    carrier_mol = max(
        (phase.volume_L - 0.018 * (phase.product_mol + phase.impurity_mol)) / 0.018,
        1.0e-7,
    )
    request = TransferRequest(
        source_amounts_mol={
            _PRODUCT_ID: max(phase.product_mol, 0.0),
            _IMPURITY_ID: max(phase.impurity_mol, 0.0),
            _CARRIER_ID: carrier_mol,
        },
        source_volume_L=phase.volume_L,
        transfer_fraction=min(max(fraction, 0.0), 1.0),
        equipment=TransferEquipmentSpec(
            equipment_id="bench_transfer_line_vnext",
            source_heel_L=0.01 * phase.volume_L,
            line_holdup_L=0.005 * phase.volume_L,
            max_transfer_volume_L=phase.volume_L,
        ),
    )
    return simulate_transfer(request)


def run_duty_limited_distillation(
    feed_amounts_mol: Mapping[str, float],
    *,
    pressure_Pa: float,
    initial_temperature_K: float,
    operating_temperature_K: float,
    duration_s: float,
    reflux_ratio: float,
    requested_cut_fraction: float,
    relative_volatility_multiplier: float = 1.0,
    solvents: SolventInventory | None = None,
    carrier_heat_capacity_J_L_K: float = 4167.0,
) -> DutyLimitedDistillationResult:
    """Map the compact target/impurity cut to the bounded column provider."""

    if relative_volatility_multiplier <= 0.0:
        raise ValueError("relative_volatility_multiplier must be positive")
    feed = {
        _PRODUCT_ID: max(float(feed_amounts_mol.get(_PRODUCT_ID, 0.0)), 0.0),
        _IMPURITY_ID: max(float(feed_amounts_mol.get(_IMPURITY_ID, 0.0)), 0.0),
    }
    operating_temperature = max(initial_temperature_K, operating_temperature_K)
    specs = {
        _PRODUCT_ID: DistillationComponentSpec(
            _PRODUCT_ID,
            650_000.0 * relative_volatility_multiplier,
            38_000.0,
            145.0,
            operating_temperature,
            470.0,
            "runtime-vnext-light-key",
        ),
        _IMPURITY_ID: DistillationComponentSpec(
            _IMPURITY_ID,
            150_000.0 / relative_volatility_multiplier**0.25,
            55_000.0,
            165.0,
            operating_temperature,
            465.0,
            "runtime-vnext-heavy-key",
        ),
    }
    if solvents is not None:
        for i, volume in enumerate(solvents.volumes_L):
            key = f"carrier-{i}"
            feed[key] = volume / 0.018
            specs[key] = DistillationComponentSpec(
                key,
                250_000.0 * RELATIVE_VOLATILITIES[i],
                40_700.0,
                carrier_heat_capacity_J_L_K * 0.018 * HEAT_CAPACITY_RATIOS[i],
                operating_temperature,
                500.0,
                "finite-surrogate-carrier-v1",
            )
    total = sum(feed.values())
    column = ShortcutColumnSpec(
        theoretical_stages=min(max(2.0 + duration_s / 900.0, 2.0), 20.0),
        stage_efficiency=0.62,
        maximum_reboiler_power_W=420.0,
        maximum_condenser_power_W=420.0,
        maximum_internal_vapor_rate_mol_s=0.04,
        maximum_batch_amount_mol=max(total * 1.1, 1.0),
        minimum_bottoms_amount_mol=min(0.01 * total, 1.0e-5),
        maximum_distillate_cut_fraction=0.90,
        minimum_pressure_Pa=20_000.0,
        maximum_pressure_Pa=200_000.0,
        maximum_temperature_K=450.0,
        maximum_duration_s=14_400.0,
        maximum_reflux_ratio=10.0,
        provenance_id="chemworld-runtime-vnext-duty-limited-column",
    )
    return simulate_duty_limited_distillation(
        DutyLimitedDistillationRequest(
            feed_amounts_mol=feed,
            component_specs=specs,
            light_key=_PRODUCT_ID,
            heavy_key=_IMPURITY_ID,
            pressure_Pa=min(max(pressure_Pa, 20_000.0), 200_000.0),
            initial_temperature_K=initial_temperature_K,
            operating_temperature_K=operating_temperature,
            duration_s=min(max(duration_s, 0.0), 14_400.0),
            reflux_ratio=min(max(reflux_ratio, 0.0), 10.0),
            requested_distillate_cut_fraction=min(max(requested_cut_fraction, 0.0), 0.90),
            column=column,
        )
    )


__all__ = [
    "PhaseSlice",
    "run_bounded_transfer",
    "run_duty_limited_distillation",
    "run_sorbent_drying",
    "run_vacuum_concentration",
]
