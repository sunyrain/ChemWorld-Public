"""Installed declaration for the finite gas/solid thermal apparatus."""

from chemworld.foundation.ontology import Operation
from chemworld.runtime import service_factories as factories
from chemworld.world.domain_modules import (
    ActionFieldSpec,
    AdvanceStrategySpec,
    ComponentCapability,
    DomainModuleSpec,
    OperationSpec,
)

THERMAL_FIXED_BED_MODULE = DomainModuleSpec(
    module_id="thermal_fixed_bed",
    scope_notes=(
        "Finite rigid ideal-gas compartments and original fixed sorbent lots; separate gas/solid "
        "constant-Cv energies, original Ergun/Langmuir/LDF, axial conduction and a declared finite "
        "thermal utility. Instantaneous control selects existing finite compartments and bounded "
        "utility settings, preserving material, capacity, thermal fields and the original clock.",
    ),
    operations=(OperationSpec(
        Operation("configure_thermal_bed", "Select actual finite bed ports and thermal controls",
                  ("bed_inlet_phase", "bed_outlet_phase"), ("not_terminated",)),
        "thermal", "reaction_thermal", "configure_thermal_bed",
        fields=(
            ActionFieldSpec("bed_inlet_phase", "categorical",
                            choice_provider=factories.thermal_bed_boundary_choices),
            ActionFieldSpec("bed_outlet_phase", "categorical",
                            choice_provider=factories.thermal_bed_boundary_choices),
            ActionFieldSpec("bed_heater_enabled", "categorical", choices=(0, 1), required=False),
            ActionFieldSpec("bed_target_temperature_K", "K", (200., 600.), required=False),
        ),
        function=factories.configure_thermal_bed,
        affected_ledgers=("equipment", "process"),
    ),),
    components=(("thermal_fixed_bed", ComponentCapability(
        operations=("wait", "configure_thermal_bed"), minimum_operations=1,
        interfaces=("material", "temperature", "sorbent"),
        state_owners=("sorbent_loading", "finite_fixed_bed_thermal_fields_v1"),
    )),),
    advance_strategies=(AdvanceStrategySpec(
        "finite-thermal-fixed-bed", factories.thermal_fixed_bed, shared_clock=True,
    ),),
    dependencies=("thermal", "sorbent_contact"),
)
