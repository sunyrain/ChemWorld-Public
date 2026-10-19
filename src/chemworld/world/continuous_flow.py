"""Continuous-flow process module for the shared ChemWorld law."""

from __future__ import annotations

from dataclasses import dataclass

from chemworld.world.operations import FLOW_OPERATIONS

FIXED_FLOW_REACTOR_VOLUME_L = 0.018
FIXED_FLOW_REACTOR_INNER_DIAMETER_M = 0.004


@dataclass(frozen=True)
class ContinuousFlowModuleSpec:
    module_id: str = "continuous_flow"
    version: str = "0.4"
    operations: tuple[str, ...] = FLOW_OPERATIONS
    laws: tuple[str, ...] = (
        "fixed_hardware_volume_with_flow_derived_residence_time",
        "geometry_resolved_axial_profile",
        "darcy_weisbach_pressure_drop",
        "distributed_thermal_boundary",
        "material_and_energy_ledgers",
        "configure_then_single_run_transaction",
        "fail_closed_domain_and_replayable_provenance",
    )

    def to_dict(self) -> dict[str, object]:
        return {
            "module_id": self.module_id,
            "version": self.version,
            "operations": list(self.operations),
            "laws": list(self.laws),
            "tracked_metrics": ["flow_conversion", "yield", "safety_risk"],
            "runtime_provider_id": "chemworld_geometry_resolved_pfr_v2",
            "operation_semantics": {
                "set_flow_rate": (
                    "configure_only_no_physical_advance_fixed_volume_residence_derived_from_flow"
                ),
                "run_flow": "advance_one_new_configured_experiment",
                "repeat_run_flow": "requires a fresh configuration",
            },
        }


__all__ = [
    "FIXED_FLOW_REACTOR_INNER_DIAMETER_M",
    "FIXED_FLOW_REACTOR_VOLUME_L",
    "ContinuousFlowModuleSpec",
]
