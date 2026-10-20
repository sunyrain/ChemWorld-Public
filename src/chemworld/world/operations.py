"""Operation registry for the unified ChemWorld language."""

from __future__ import annotations

from typing import Any

import numpy as np

from chemworld.foundation import Operation
from chemworld.world.action_layout import (
    LEGACY_INSTRUMENT_ORDER,
    LEGACY_OBSERVATION_ORDER,
    LEGACY_OPERATION_ORDER,
)
from chemworld.world.control_contract import CONTROL_OPERATIONS
from chemworld.world.domain_modules import DomainModuleRegistry, OperationContract
from chemworld.world.ontology import chemworld_state_variables
from chemworld.world.stream_contract import STREAM_OPERATIONS

REACTION_OPERATIONS = (
    "add_reagent",
    "add_component",
    "add_solvent",
    "add_catalyst",
    "heat",
    "wait",
    "sample",
    "quench",
    "create_container",
    "transfer_material",
    "configure_instrument",
    *CONTROL_OPERATIONS,
    "terminate",
    "measure",
)
SEPARATION_OPERATIONS = (
    "add_phase",
    "add_extractant",
    "mix",
    "settle",
    "separate_phase",
    "wash",
    "dry",
    "concentrate",
    "transfer",
)
CRYSTALLIZATION_OPERATIONS = (
    "seed_crystals",
    "cool_crystallize",
    "filter_crystals",
    "resuspend_crystals",
)
DISTILLATION_OPERATIONS = ("evaporate", "distill", "collect_fraction")
FLOW_OPERATIONS = ("set_flow_rate", "run_flow", *STREAM_OPERATIONS)
ELECTROCHEMISTRY_OPERATIONS = ("set_potential", "electrolyze")
MACRO_OPERATIONS = ("wash", "dry", "concentrate")
TERMINAL_OPERATIONS = ("terminate",)
CAMPAIGN_CONTROL_OPERATIONS = ("discard_batch",)
DOMAIN_OPERATIONS = (
    *CRYSTALLIZATION_OPERATIONS,
    *DISTILLATION_OPERATIONS,
    *FLOW_OPERATIONS,
    *ELECTROCHEMISTRY_OPERATIONS,
)
PRIMITIVE_OPERATIONS = tuple(
    operation
    for operation in (
        *REACTION_OPERATIONS,
        *SEPARATION_OPERATIONS,
        *DOMAIN_OPERATIONS,
        *TERMINAL_OPERATIONS,
    )
    if operation not in (*MACRO_OPERATIONS, *DOMAIN_OPERATIONS, *TERMINAL_OPERATIONS)
)
PROCESS_OPERATIONS = (
    *CRYSTALLIZATION_OPERATIONS,
    *DISTILLATION_OPERATIONS,
    *FLOW_OPERATIONS,
    *ELECTROCHEMISTRY_OPERATIONS,
)
OPERATION_TYPES = LEGACY_OPERATION_ORDER
CAMPAIGN_OPERATION_TYPES = (*OPERATION_TYPES, *CAMPAIGN_CONTROL_OPERATIONS)
INSTRUMENTS = LEGACY_INSTRUMENT_ORDER
EXTENDED_INSTRUMENTS = (*INSTRUMENTS, "particle_size")

# Operation-specific input contracts are the single source of truth for values
# whose effective runtime domain is narrower than the shared Gym field space.
# Services may retain defensive clipping, but validated actions must never be
# silently reinterpreted by those guards.
_BUILTIN_OPERATIONS = DomainModuleRegistry.default().operation_specs()
OPERATION_FIELD_BOUNDS = {
    (key, f.name): f.runtime_bounds
    for key, spec in _BUILTIN_OPERATIONS.items()
    for f in spec.fields
    if f.runtime_bounds is not None
}
OPERATION_FIELD_CHOICES = {
    (key, f.name): f.runtime_choices
    for key, spec in _BUILTIN_OPERATIONS.items()
    for f in spec.fields
    if f.runtime_choices
}
OPERATION_CUMULATIVE_FIELD_LIMITS = {
    (key, f.name): f.cumulative_limit
    for key, spec in _BUILTIN_OPERATIONS.items()
    for f in spec.fields
    if f.cumulative_limit is not None
}
MATERIAL_STOCK_FIELDS = {
    key: spec.stock_field for key, spec in _BUILTIN_OPERATIONS.items() if spec.stock_field
}
DOWNSTREAM_OBSERVATION_KEYS = (
    "purity",
    "recovery",
    "phase_ratio",
    "product_in_organic",
    "product_in_aqueous",
    "impurity_signal",
    "solvent_loss",
    "process_mass_balance_error",
    "crystal_yield",
    "crystal_purity",
    "crystal_size",
    "distillate_purity",
    "distillate_recovery",
    "flow_conversion",
    "electrochemical_selectivity",
    "selective_product_yield",
    "electrochemical_conversion",
    "energy_efficiency",
    "crystal_csd_quality",
    "crystal_fines_fraction",
    "faradaic_efficiency",
    "transport_efficiency",
    "ohmic_efficiency",
)
EQUILIBRIUM_OBSERVATION_KEYS = (
    "pH_normalized",
    "acid_dissociation_fraction",
    "precipitation_signal",
    "equilibrium_residual",
    "equilibrium_confidence",
)
CORE_OBSERVATION_KEYS = (
    "yield",
    "selectivity",
    "conversion",
    "cost",
    "safety_risk",
    "score",
    "byproduct_signal",
    "degradation_warning",
    "virtual_spectrum_summary",
)
PUBLIC_OBSERVATION_KEYS = LEGACY_OBSERVATION_ORDER


def chemworld_operations(*, modules: DomainModuleRegistry | None = None) -> tuple[Operation, ...]:
    registry = modules or DomainModuleRegistry.default()
    specs = registry.operation_specs()
    return tuple(specs[name].operation for name in registry.operation_order() if name in specs)


def chemworld_state_variable_contracts() -> tuple[Any, ...]:
    return chemworld_state_variables()


def operation_contracts(
    *,
    include_campaign_controls: bool = False,
    modules: DomainModuleRegistry | None = None,
) -> dict[str, OperationContract]:
    contracts = (modules or DomainModuleRegistry.default()).contracts()
    if include_campaign_controls:
        contracts["discard_batch"] = OperationContract(
            operation_id="discard_batch",
            module="campaign_control",
            kind="campaign_control",
            required_fields=("reason",),
            preconditions=("campaign_batch_open", "campaign_vessel_started"),
        )
    return contracts


def operation_name(value: Any, *, modules: DomainModuleRegistry | None = None) -> str:
    operation_types = OPERATION_TYPES if modules is None else modules.operation_order()
    available = set(operation_types) if modules is None else set(modules.operation_specs())
    if isinstance(value, str):
        if value not in available:
            raise ValueError(f"Unsupported operation: {value}")
        return value
    name = operation_types[_discrete_index(value, len(operation_types), "operation")]
    if name not in available:
        raise ValueError(f"Operation is not installed: {name}")
    return name


def instrument_name(value: Any, *, modules: DomainModuleRegistry | None = None) -> str:
    available = (
        EXTENDED_INSTRUMENTS if modules is None else modules.instrument_specs(include_optional=True)
    )
    order = INSTRUMENTS if modules is None else modules.instrument_order()
    if isinstance(value, str):
        if value not in available:
            raise ValueError(f"Unsupported instrument: {value}")
        return value
    name = order[_discrete_index(value, len(order), "instrument")]
    if name not in available:
        raise ValueError(f"Instrument is not installed: {name}")
    return name


def _discrete_index(value: Any, cardinality: int, label: str) -> int:
    """Decode one finite categorical coordinate without silent remapping."""

    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{label} must be an integer categorical index, not boolean")
    try:
        values = np.asarray(value).reshape(-1)
        if values.size != 1:
            raise ValueError(f"{label} must be a scalar categorical index")
        coordinate = float(values[0])
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be a scalar categorical index") from exc
    if not np.isfinite(coordinate) or not coordinate.is_integer():
        raise ValueError(f"{label} must be a finite integer categorical index")
    index = int(coordinate)
    if not 0 <= index < cardinality:
        raise ValueError(f"{label} index {index} is outside [0, {cardinality - 1}]")
    return index


__all__ = [
    "CAMPAIGN_CONTROL_OPERATIONS",
    "CAMPAIGN_OPERATION_TYPES",
    "CORE_OBSERVATION_KEYS",
    "CRYSTALLIZATION_OPERATIONS",
    "DISTILLATION_OPERATIONS",
    "DOMAIN_OPERATIONS",
    "DOWNSTREAM_OBSERVATION_KEYS",
    "ELECTROCHEMISTRY_OPERATIONS",
    "EQUILIBRIUM_OBSERVATION_KEYS",
    "FLOW_OPERATIONS",
    "INSTRUMENTS",
    "MACRO_OPERATIONS",
    "OPERATION_CUMULATIVE_FIELD_LIMITS",
    "OPERATION_FIELD_BOUNDS",
    "OPERATION_FIELD_CHOICES",
    "OPERATION_TYPES",
    "PRIMITIVE_OPERATIONS",
    "PROCESS_OPERATIONS",
    "PUBLIC_OBSERVATION_KEYS",
    "REACTION_OPERATIONS",
    "SEPARATION_OPERATIONS",
    "TERMINAL_OPERATIONS",
    "OperationContract",
    "chemworld_operations",
    "chemworld_state_variable_contracts",
    "instrument_name",
    "operation_contracts",
    "operation_name",
]
