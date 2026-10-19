"""Operation registry for the unified ChemWorld language."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from chemworld.foundation import Operation
from chemworld.world.actions import SOLVENTS
from chemworld.world.control_contract import CONTROL_ACTION_FIELDS, CONTROL_OPERATIONS
from chemworld.world.ontology import chemworld_state_variables
from chemworld.world.spectral_contract import SPECTRAL_FIELDS, SPECTRAL_INSTRUMENTS
from chemworld.world.stream_contract import STREAM_FIELDS, STREAM_OPERATIONS

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
OPERATION_TYPES = (
    "select_vessel",
    "route_material",
    *REACTION_OPERATIONS[:-2],
    *SEPARATION_OPERATIONS,
    *PROCESS_OPERATIONS,
    "terminate",
    "measure",
)
CAMPAIGN_OPERATION_TYPES = (*OPERATION_TYPES, *CAMPAIGN_CONTROL_OPERATIONS)
INSTRUMENTS = ("hplc", "gc", "uvvis", "ph_meter", "final_assay", *SPECTRAL_INSTRUMENTS)
EXTENDED_INSTRUMENTS = (*INSTRUMENTS, "particle_size")

# Operation-specific input contracts are the single source of truth for values
# whose effective runtime domain is narrower than the shared Gym field space.
# Services may retain defensive clipping, but validated actions must never be
# silently reinterpreted by those guards.
OPERATION_FIELD_BOUNDS: dict[tuple[str, str], tuple[float, float]] = {
    ("set_flow_stream", "flow_rate_mL_min"): (0.0, 20.0),
    ("set_flow_rate", "flow_rate_mL_min"): (0.01, 20.0),
    ("advance_flow", "duration_s"): (1.0, 14400.0),
    ("advance_flow", "target_temperature_K"): (250.0, 430.0),
    ("advance_flow", "current_mA"): (0.0, 500.0),
    ("configure_control", "target_temperature_K"): (250.0, 430.0),
    ("queue_control_stage", "target_temperature_K"): (250.0, 430.0),
    ("advance_control", "duration_s"): (1.0, 14400.0),
    ("route_material", "transfer_fraction"): (0.0, 1.0),
    ("transfer_material", "transfer_fraction"): (0.0, 1.0),
    ("heat", "duration_s"): (1.0, 14_400.0),
    ("wait", "duration_s"): (1.0, 14_400.0),
    ("add_phase", "volume_L"): (0.0, 0.060),
    ("add_extractant", "volume_L"): (0.0, 0.060),
    ("mix", "duration_s"): (1.0, 14_400.0),
    ("settle", "duration_s"): (1.0, 14_400.0),
    ("wash", "wash_volume_L"): (1.0e-6, 0.040),
    ("concentrate", "duration_s"): (1.0, 14_400.0),
    ("transfer", "transfer_fraction"): (1.0e-4, 1.0),
    ("seed_crystals", "seed_mass_g"): (1.0e-6, 0.050),
    ("cool_crystallize", "target_temperature_K"): (250.0, 330.0),
    ("cool_crystallize", "duration_s"): (1.0, 14_400.0),
    ("resuspend_crystals", "volume_L"): (0.0001, 0.080),
    ("evaporate", "target_temperature_K"): (298.15, 390.0),
    ("evaporate", "duration_s"): (1.0, 14_400.0),
    ("distill", "target_temperature_K"): (298.15, 430.0),
    ("distill", "duration_s"): (1.0, 14_400.0),
    ("collect_fraction", "transfer_fraction"): (1.0e-4, 1.0),
    ("run_flow", "target_temperature_K"): (298.15, 430.0),
    ("run_flow", "duration_s"): (1.0, 14_400.0),
    ("set_potential", "current_mA"): (1.0e-3, 500.0),
    ("electrolyze", "duration_s"): (1.0, 14_400.0),
}
OPERATION_FIELD_CHOICES: dict[tuple[str, str], tuple[Any, ...]] = {
    ("add_phase", "phase"): ("aqueous", "organic"),
    ("add_extractant", "extractant"): tuple(range(len(SOLVENTS))),
    ("separate_phase", "target_phase"): ("aqueous", "organic"),
}
# Cumulative resource caps are operation contracts, not task-specific guards.
# They prevent repeated calls from bypassing a per-call physical domain limit.
OPERATION_CUMULATIVE_FIELD_LIMITS: dict[tuple[str, str], float] = {
    ("seed_crystals", "seed_mass_g"): 0.050,
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
PUBLIC_OBSERVATION_KEYS = (
    *CORE_OBSERVATION_KEYS,
    *DOWNSTREAM_OBSERVATION_KEYS,
    *EQUILIBRIUM_OBSERVATION_KEYS,
)


@dataclass(frozen=True)
class OperationContract:
    operation_id: str
    module: str
    kind: str
    required_fields: tuple[str, ...]
    preconditions: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "operation_id": self.operation_id,
            "module": self.module,
            "kind": self.kind,
            "required_fields": list(self.required_fields),
            "preconditions": list(self.preconditions),
        }


def chemworld_operations() -> tuple[Operation, ...]:
    """Return operation contracts for the shared event language."""

    return (
        Operation(
            "configure_instrument",
            "Configure finite spectral acquisition",
            ("instrument", *SPECTRAL_FIELDS),
            ("not_terminated",),
        ),
        *(
            Operation(name, "Finite-reservoir continuous streams", fields, ("not_terminated",))
            for name, fields in STREAM_FIELDS.items()
        ),
        *(
            Operation(name, "Persistent bounded process control", fields, ("not_terminated",))
            for name, fields in CONTROL_ACTION_FIELDS.items()
        ),
        Operation(
            "select_vessel", "Select a declared component vessel", ("vessel",), ("not_terminated",)
        ),
        Operation(
            "route_material",
            "Route a declared material output",
            ("connection", "transfer_fraction", "mixing"),
            ("not_terminated",),
        ),
        Operation(
            "create_container",
            "Open a single-use storage container",
            ("container", "capacity_L"),
            ("not_terminated",),
        ),
        Operation(
            "transfer_material",
            "Transfer or merge a representative material aliquot",
            ("source_container", "destination_container", "transfer_fraction", "mixing"),
            ("not_terminated", "material_routing_valid"),
        ),
        Operation(
            "resuspend_crystals",
            "Isolate filtrate and resuspend retained crystals in fresh solvent",
            ("volume_L", "solvent"),
            ("not_terminated", "has_material", "resuspend_requires_filtered_crystals"),
        ),
        Operation("add_reagent", "Add reagent", ("amount_mol",), ("not_terminated",)),
        Operation(
            "add_component",
            "Dose one catalog feed independently",
            ("component", "amount_mol"),
            ("not_terminated",),
        ),
        Operation("add_solvent", "Add solvent", ("volume_L", "solvent"), ("not_terminated",)),
        Operation(
            "add_catalyst",
            "Add catalyst",
            ("catalyst_amount_mol", "catalyst"),
            ("not_terminated",),
        ),
        Operation(
            "heat",
            "Heat",
            ("target_temperature_K", "duration_s", "stirring_speed_rpm"),
            ("has_volume", "has_material"),
        ),
        Operation(
            "wait",
            "Wait",
            ("duration_s", "stirring_speed_rpm"),
            ("has_volume", "has_material"),
        ),
        Operation("sample", "Sample", ("sample_volume_L",), ("has_volume",)),
        Operation("quench", "Quench", (), ("has_volume",)),
        Operation("add_phase", "Add phase", ("phase", "volume_L"), ("not_terminated",)),
        Operation(
            "add_extractant",
            "Add extractant",
            ("extractant", "volume_L"),
            ("has_volume", "has_material", "not_terminated"),
        ),
        Operation("mix", "Mix phases", ("duration_s", "stirring_speed_rpm"), ("has_phase_system",)),
        Operation("settle", "Settle phases", ("duration_s",), ("has_phase_system",)),
        Operation(
            "separate_phase",
            "Separate phase",
            ("target_phase",),
            ("has_phase_system", "phase_settled"),
        ),
        Operation("wash", "Wash product phase", ("wash_volume_L",), ("has_phase_system",)),
        Operation("dry", "Dry product phase", (), ("has_phase_system",)),
        Operation(
            "concentrate", "Concentrate product phase", ("duration_s",), ("has_phase_system",)
        ),
        Operation(
            "transfer", "Transfer product phase", ("transfer_fraction",), ("has_phase_system",)
        ),
        Operation("seed_crystals", "Seed crystallization", ("seed_mass_g",), ("has_volume",)),
        Operation(
            "cool_crystallize",
            "Cool crystallization",
            ("target_temperature_K", "duration_s"),
            ("has_volume", "has_material"),
        ),
        Operation(
            "filter_crystals",
            "Filter crystals",
            (),
            ("has_material", "filter_requires_crystallization"),
        ),
        Operation(
            "evaporate",
            "Evaporate solvent",
            ("target_temperature_K", "duration_s"),
            ("has_volume",),
        ),
        Operation(
            "distill",
            "Distill volatile fraction",
            ("target_temperature_K", "duration_s", "reflux_ratio"),
            ("has_volume", "has_material"),
        ),
        Operation(
            "collect_fraction",
            "Collect distillation fraction",
            ("transfer_fraction",),
            ("collect_fraction_requires_distillation",),
        ),
        Operation(
            "set_flow_rate",
            (
                "Configure fixed-volume continuous-flow rate; residence_time_s is a "
                "compatibility declaration and physical residence is derived from flow rate"
            ),
            ("flow_rate_mL_min", "residence_time_s"),
            ("not_terminated",),
        ),
        Operation(
            "run_flow",
            "Run continuous-flow reaction",
            ("target_temperature_K", "duration_s"),
            ("has_volume", "has_material", "run_flow_requires_flow_setup"),
        ),
        Operation(
            "set_potential",
            "Configure electrochemical cell",
            ("potential_V", "current_mA", "electrolyte_profile"),
            ("has_volume", "has_material"),
        ),
        Operation(
            "electrolyze",
            "Run electrolysis",
            ("duration_s",),
            ("has_volume", "has_material", "electrolyze_requires_potential"),
        ),
        Operation("terminate", "Terminate", (), ("has_material", "not_terminated")),
        Operation("measure", "Measure", ("instrument",), ("has_volume", "instrument_specific")),
    )


def chemworld_state_variable_contracts() -> tuple[Any, ...]:
    return chemworld_state_variables()


def operation_contracts(
    *,
    include_campaign_controls: bool = False,
) -> dict[str, OperationContract]:
    reaction = set(REACTION_OPERATIONS)
    separation = set(SEPARATION_OPERATIONS)
    crystallization = set(CRYSTALLIZATION_OPERATIONS)
    distillation = set(DISTILLATION_OPERATIONS)
    flow = set(FLOW_OPERATIONS)
    electrochemistry = set(ELECTROCHEMISTRY_OPERATIONS)
    macros = set(MACRO_OPERATIONS)
    terminals = set(TERMINAL_OPERATIONS)
    domains = set(DOMAIN_OPERATIONS)
    contracts: dict[str, OperationContract] = {}
    for operation in chemworld_operations():
        if operation.id in separation:
            module = "separation"
        elif operation.id in crystallization:
            module = "crystallization"
        elif operation.id in distillation:
            module = "distillation"
        elif operation.id in flow:
            module = "continuous_flow"
        elif operation.id in electrochemistry:
            module = "electrochemistry"
        elif operation.id in {"measure", "configure_instrument"}:
            module = "observation"
        elif operation.id in reaction or operation.id in {"select_vessel", "route_material"}:
            module = "reaction"
        else:
            module = "general"
        if operation.id in macros:
            kind = "macro"
        elif operation.id in terminals:
            kind = "terminal"
        elif operation.id in domains:
            kind = "domain"
        else:
            kind = "primitive"
        contracts[operation.id] = OperationContract(
            operation_id=operation.id,
            module=module,
            kind=kind,
            required_fields=operation.required_fields,
            preconditions=operation.preconditions,
        )
    if include_campaign_controls:
        contracts["discard_batch"] = OperationContract(
            operation_id="discard_batch",
            module="campaign_control",
            kind="campaign_control",
            required_fields=("reason",),
            preconditions=("campaign_batch_open", "campaign_vessel_started"),
        )
    return contracts


def operation_name(value: Any) -> str:
    if isinstance(value, str):
        if value not in OPERATION_TYPES:
            raise ValueError(f"Unsupported operation: {value}")
        return value
    return OPERATION_TYPES[_discrete_index(value, len(OPERATION_TYPES), "operation")]


def instrument_name(value: Any) -> str:
    if isinstance(value, str):
        if value not in (*INSTRUMENTS, "particle_size"):
            raise ValueError(f"Unsupported instrument: {value}")
        return value
    return INSTRUMENTS[_discrete_index(value, len(INSTRUMENTS), "instrument")]


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
