"""Lazy constructors used by built-in module declarations.

Imports happen at runtime construction, so physical declarations do not depend
on runtime initialization order. Factories receive only the current world context.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from chemworld.foundation import WorldState
    from chemworld.operation_validator import OperationValidator
    from chemworld.physchem.coupled_system import CoupledProcess
    from chemworld.runtime.aqueous_gas_transfer import AqueousGasTransferRuntime
    from chemworld.runtime.aqueous_mineral import AqueousMineralRuntime
    from chemworld.runtime.aqueous_reactive_transport import AqueousReactiveTransportRuntime
    from chemworld.runtime.battery import BatteryRuntime
    from chemworld.runtime.component_network import ComponentNetworkServices
    from chemworld.runtime.concentration import ConcentrationRuntime
    from chemworld.runtime.continuous_streams import ContinuousStreamServices
    from chemworld.runtime.control_program import ControlProgramServices
    from chemworld.runtime.coupled_thermal import CoupledThermalRuntime, ProcessFactoryContext
    from chemworld.runtime.crystal_thermal import CrystalThermalRuntime
    from chemworld.runtime.crystallization_services import ChemWorldCrystallizationServices
    from chemworld.runtime.distillation_services import ChemWorldDistillationServices
    from chemworld.runtime.dynamic_column import DynamicColumnRuntime
    from chemworld.runtime.electrochemical_services import ChemWorldElectrochemicalServices
    from chemworld.runtime.finite_supplies import FiniteCarrierSupply
    from chemworld.runtime.flash_thermal import FlashThermalRuntime
    from chemworld.runtime.flow_services import ChemWorldFlowServices
    from chemworld.runtime.gas_reactor_network import GasReactorNetworkRuntime
    from chemworld.runtime.instrument_cost_services import ChemWorldInstrumentCostServices
    from chemworld.runtime.kernel_contracts import ModelProviderResult
    from chemworld.runtime.legacy_liquid_advance import LegacyLiquidAdvance
    from chemworld.runtime.liquid_equilibrium import LiquidEquilibriumRuntime
    from chemworld.runtime.material_routing import ChemWorldMaterialRoutingServices
    from chemworld.runtime.membranes import MembraneRuntime
    from chemworld.runtime.mineral_kinetics import MineralKineticsRuntime
    from chemworld.runtime.molecular_process_runtime import MolecularProcessRuntime
    from chemworld.runtime.phase_separation_services import ChemWorldPhaseSeparationServices
    from chemworld.runtime.primitive_services import ChemWorldPrimitiveOperationServices
    from chemworld.runtime.reaction_thermal_services import ChemWorldReactionThermalServices
    from chemworld.runtime.sbml_models import SBMLAdvanceStrategy
    from chemworld.runtime.scheduling import LabScheduleServices
    from chemworld.runtime.solid_solutions import SolidSolutionRuntime
    from chemworld.runtime.stochastic import StochasticRuntime
    from chemworld.runtime.stochastic_sde import SDEStochasticRuntime
    from chemworld.runtime.thermal_fixed_bed_world import ThermalFixedBedRuntime
    from chemworld.runtime.thevenin import TheveninRuntime
    from chemworld.runtime.transient_electrode import TransientElectrodeRuntime
    from chemworld.runtime.transport_devices import TransportRuntime
    from chemworld.world.composition import WorldCompositionSpec
    from chemworld.world.domain_modules import ServiceContext


def configure_chromatography(state: WorldState, action: dict[str, Any]) -> WorldState:
    from chemworld.runtime.chromatography import configure

    return configure(state, action)


def calibrate_chromatography(state: WorldState, action: dict[str, Any]) -> WorldState:
    from chemworld.runtime.chromatography import calibrate

    return calibrate(state, action)


def primitive(context: ServiceContext) -> ChemWorldPrimitiveOperationServices:
    from chemworld.runtime.primitive_services import ChemWorldPrimitiveOperationServices

    return ChemWorldPrimitiveOperationServices(
        context.world, context.species_view, context.services["material_routing"]
    )


def material_routing(context: ServiceContext) -> ChemWorldMaterialRoutingServices:
    from chemworld.runtime.material_routing import ChemWorldMaterialRoutingServices

    return ChemWorldMaterialRoutingServices(
        None if context.composition is None else context.composition.physics
    )


def component_network(context: ServiceContext) -> ComponentNetworkServices:
    from chemworld.runtime.component_network import ComponentNetworkServices

    return ComponentNetworkServices(context.services["material_routing"], context.composition)


def reaction_thermal(context: ServiceContext) -> ChemWorldReactionThermalServices:
    from chemworld.runtime.reaction_thermal_services import ChemWorldReactionThermalServices

    return ChemWorldReactionThermalServices(
        context.world,
        context.species_view,
        composition=context.composition,
        modules=context.modules,
    )


def bulk_process(context: ProcessFactoryContext) -> tuple[CoupledProcess, ...]:
    from dataclasses import replace

    from chemworld.physchem.molecular_mixture import MolecularIsochoricThermodynamics
    from chemworld.physchem.reactor_solvers import reaction_network_process

    local_ids = {
        species.species_id for species in context.network.species
        if species.phase != "surface" and species.species_id in context.state.species_amounts
    }
    reactions = []
    for reaction in context.network.reactions:
        referenced = set(reaction.stoichiometry) | set(reaction.forward_orders) \
            | set(reaction.reverse_orders)
        if referenced <= local_ids:
            reactions.append(reaction)
        elif referenced & local_ids:
            raise ValueError("local bulk reaction omits a declared participant inventory")
    network = replace(
        context.network,
        species=tuple(
            species for species in context.network.species
            if species.phase != "surface" and species.species_id in context.state.species_amounts
        ),
        reactions=tuple(reactions),
        stochastic=None,
    )
    if isinstance(context.thermodynamics, MolecularIsochoricThermodynamics):
        from chemworld.physchem.molecular_reaction import molecular_reaction_process

        return (molecular_reaction_process(
            network, context.thermodynamics, process_id=context.component.id,
        ),)
    return (reaction_network_process(network, process_id=context.component.id),)


def surface_processes(context: ProcessFactoryContext) -> tuple[CoupledProcess, ...]:
    from chemworld.physchem.catalysis_network import (
        CATALYTIC_STATE_TYPE,
        CatalyticMechanismSpec,
        catalytic_process,
        validate_catalytic_thermodynamics,
    )
    from chemworld.physchem.surface_rates import surface_process
    from chemworld.physchem.surface_specs import (
        DEFAULT_SURFACE_MECHANISM,
        MULTISITE_STATE_TYPE,
        SURFACE_STATE_TYPE,
        MultisiteSurfaceSpec,
        surface_species_energies,
    )

    model_id = context.component.parameters.get("surface_id")
    if model_id is not None and model_id not in context.physics.models_of("surface"):
        raise ValueError("surface component must name a declared native mechanism")
    spec = (
        DEFAULT_SURFACE_MECHANISM if model_id is None
        else context.physics.models_of("surface")[model_id]
    )
    if isinstance(spec, CatalyticMechanismSpec):
        validate_catalytic_thermodynamics(
            spec, context.physics.models_of("thermal")[context.state.vessel_id].thermodynamics
        )
        slots = tuple(
            slot for slot in context.state.state_slots.slots
            if slot.spec.state_type == CATALYTIC_STATE_TYPE
            and slot.spec.mechanism_id == spec.mechanism_id
        )
        if not slots:
            raise ValueError("declared catalytic process requires its owned site inventory")
        return tuple(catalytic_process(slot, spec) for slot in slots)
    if isinstance(spec, MultisiteSurfaceSpec):
        from chemworld.physchem.coupled_system import ConstantCvThermodynamics

        caloric = context.physics.models_of("thermal")[context.state.vessel_id].thermodynamics
        if (
            not isinstance(caloric, ConstantCvThermodynamics)
            or caloric.reference_temperature_K != spec.reference_temperature_K
            or any(
                caloric.species.get(key) != value
                for key, value in surface_species_energies(spec).items()
            )
        ):
            raise ValueError(
                "reversible surface standard G and shared U/Cv must use the same declared table"
            )

    slots = tuple(
        slot
        for slot in context.state.state_slots.slots
        if slot.spec.state_type in {SURFACE_STATE_TYPE, MULTISITE_STATE_TYPE}
    )
    if not slots:
        # A material carrier may leave its vessel or be cleared at a declared
        # batch boundary. Its absence means zero active surface, not a new
        # inventory. Stationary/vessel-owned sites remain mandatory.
        initial = context.physics.initial_states.get(context.state.vessel_id)
        declared = () if initial is None else tuple(
            slot for slot in initial.state_slots.slots
            if slot.spec.state_type in {SURFACE_STATE_TYPE, MULTISITE_STATE_TYPE}
            and slot.spec.mechanism_id == spec.mechanism_id)
        if declared and all(slot.spec.owner_kind == 'material'
                            and slot.spec.batch_reset == 'clear' for slot in declared):
            return ()
        raise ValueError("surface process requires an owned surface inventory")
    return tuple(surface_process(slot, spec) for slot in slots)


def sorbent_processes(context: ProcessFactoryContext) -> tuple[CoupledProcess, ...]:
    from chemworld.physchem.sorbent_kinetics import (
        SORBENT_STATE_TYPE,
        sorbent_process,
        validate_sorbent_slot,
    )

    model_id = context.component.parameters["sorbent_id"]
    spec = context.physics.models_of("sorbent")[model_id]
    slots = tuple(
        s
        for s in context.state.state_slots.slots
        if s.spec.state_type == SORBENT_STATE_TYPE and s.spec.mechanism_id == model_id
    )
    if not slots:
        raise ValueError("sorbent contact requires an explicitly owned carrier lot")
    for slot in slots:
        validate_sorbent_slot(slot, spec)
    # An empty liquid compartment can heat its retained carrier. Contact
    # resumes when actual finite liquid is added; geometry is not ghost fluid.
    if context.state.volume_L == 0:
        return ()
    return tuple(sorbent_process(s, spec, thermodynamics=context.thermodynamics) for s in slots)


def control_program(context: ServiceContext) -> ControlProgramServices:
    from chemworld.runtime.control_program import ControlProgramServices

    return ControlProgramServices(context.services["reaction_thermal"])


def electrical_control_choices(
    state: WorldState, _validator: OperationValidator | None,
) -> tuple[str, ...]:
    from chemworld.runtime.control_program import electrical_device

    try:
        settings = electrical_device(state).settings["electrical_control"]
    except ValueError:
        return ()
    if settings.get("passive_load_modes"):
        return tuple(settings["passive_load_modes"])
    if settings.get("named_control_modes"):
        return tuple(settings["named_control_modes"])
    # Scalar A/V sources do not require the optional named-mode field. Keep a
    # nonempty lexical domain so action admission does not reject its absence.
    return tuple(mode for mode, unit in (("current", "A"), ("voltage", "V"))
                 if unit in settings.get("supported_units", ("A", "V")))


def transient_electrode(composition: WorldCompositionSpec | None) -> TransientElectrodeRuntime:
    from chemworld.runtime.transient_electrode import TransientElectrodeRuntime

    return TransientElectrodeRuntime(composition)


def aqueous_mineral(composition: WorldCompositionSpec | None) -> AqueousMineralRuntime:
    from chemworld.runtime.aqueous_mineral import AqueousMineralRuntime

    return AqueousMineralRuntime(composition)


def mineral_kinetics(composition: WorldCompositionSpec | None) -> MineralKineticsRuntime:
    from chemworld.runtime.mineral_kinetics import MineralKineticsRuntime

    return MineralKineticsRuntime(composition)


def battery_spm(composition: WorldCompositionSpec | None) -> BatteryRuntime:
    from chemworld.runtime.battery import BatteryRuntime

    return BatteryRuntime(composition)


def battery_thevenin(composition: WorldCompositionSpec | None) -> TheveninRuntime:
    from chemworld.runtime.thevenin import TheveninRuntime

    return TheveninRuntime(composition)


def finite_transport(composition: WorldCompositionSpec | None) -> TransportRuntime:
    from chemworld.runtime.transport_devices import TransportRuntime

    return TransportRuntime(composition)


def gas_reactor_network(composition: WorldCompositionSpec) -> GasReactorNetworkRuntime:
    from chemworld.runtime.gas_reactor_network import GasReactorNetworkRuntime

    return GasReactorNetworkRuntime(composition)


def thermal_fixed_bed(composition: WorldCompositionSpec | None) -> ThermalFixedBedRuntime:
    from chemworld.runtime.thermal_fixed_bed_world import ThermalFixedBedRuntime

    return ThermalFixedBedRuntime(composition)


def configure_thermal_bed(state: WorldState, action: dict[str, Any]) -> ModelProviderResult:
    from chemworld.runtime.thermal_fixed_bed_world import configure_thermal_bed as configure

    return configure(state, action)


def thermal_bed_boundary_choices(
    state: WorldState, validator: OperationValidator | None = None,
) -> tuple[str, ...]:
    from chemworld.runtime.thermal_fixed_bed_world import bed_boundary_choices

    return bed_boundary_choices(state, validator)


def aqueous_reactive_transport(
    composition: WorldCompositionSpec | None,
) -> AqueousReactiveTransportRuntime:
    from chemworld.runtime.aqueous_reactive_transport import AqueousReactiveTransportRuntime

    return AqueousReactiveTransportRuntime(composition)


def aqueous_gas_transfer(composition: WorldCompositionSpec | None) -> AqueousGasTransferRuntime:
    from chemworld.runtime.aqueous_gas_transfer import AqueousGasTransferRuntime

    return AqueousGasTransferRuntime(composition)


def molecular_process(composition: WorldCompositionSpec | None) -> MolecularProcessRuntime:
    from chemworld.runtime.molecular_process_runtime import MolecularProcessRuntime

    return MolecularProcessRuntime(composition)


def stochastic(composition: WorldCompositionSpec | None) -> StochasticRuntime:
    from chemworld.runtime.stochastic import StochasticRuntime

    return StochasticRuntime(composition)


def stochastic_controls(context: ServiceContext) -> SDEStochasticRuntime:
    return context.services['reaction_thermal'].advancer.strategies['gillespie-direct'].sde


def sbml(composition: WorldCompositionSpec | None) -> SBMLAdvanceStrategy:
    from chemworld.runtime.sbml_models import SBMLAdvanceStrategy

    return SBMLAdvanceStrategy(composition)


def formal_models(context: ServiceContext) -> SBMLAdvanceStrategy:
    return context.services["reaction_thermal"].formal_models


def ro_membrane(composition: WorldCompositionSpec | None) -> MembraneRuntime:
    from chemworld.runtime.membranes import MembraneRuntime

    return MembraneRuntime(composition)


def solid_solution(composition: WorldCompositionSpec | None) -> SolidSolutionRuntime:
    from chemworld.runtime.solid_solutions import SolidSolutionRuntime

    return SolidSolutionRuntime(composition)


def coupled_thermal(composition: WorldCompositionSpec | None) -> CoupledThermalRuntime:
    from chemworld.runtime.coupled_thermal import CoupledThermalRuntime

    return CoupledThermalRuntime(composition, None if composition is None else composition.modules)


def finite_carrier_supply(composition: WorldCompositionSpec | None) -> FiniteCarrierSupply:
    from chemworld.runtime.finite_supplies import FiniteCarrierSupply

    return FiniteCarrierSupply(composition)


def flash_thermal(composition: WorldCompositionSpec | None) -> FlashThermalRuntime:
    from chemworld.runtime.flash_thermal import FlashThermalRuntime

    return FlashThermalRuntime(composition)


def liquid_equilibrium(composition: WorldCompositionSpec | None) -> LiquidEquilibriumRuntime:
    from chemworld.runtime.liquid_equilibrium import LiquidEquilibriumRuntime

    return LiquidEquilibriumRuntime(composition)


def dynamic_column(composition: WorldCompositionSpec | None) -> DynamicColumnRuntime:
    from chemworld.runtime.dynamic_column import DynamicColumnRuntime

    return DynamicColumnRuntime(composition)


def concentration(composition: WorldCompositionSpec | None) -> ConcentrationRuntime:
    from chemworld.runtime.concentration import ConcentrationRuntime

    return ConcentrationRuntime(composition)


def crystal_thermal(composition: WorldCompositionSpec | None) -> CrystalThermalRuntime:
    from chemworld.runtime.crystal_thermal import CrystalThermalRuntime

    return CrystalThermalRuntime(composition)


def legacy_liquid(composition: WorldCompositionSpec | None) -> LegacyLiquidAdvance:
    from chemworld.runtime.legacy_liquid_advance import LegacyLiquidAdvance

    return LegacyLiquidAdvance(composition)


def configure_electrical_control(state: WorldState, action: dict[str, Any]) -> WorldState:
    from chemworld.runtime.control_program import configure_electrical_control as configure

    return configure(state, action)


def queue_electrical_stage(state: WorldState, action: dict[str, Any]) -> WorldState:
    from chemworld.runtime.control_program import queue_electrical_stage as queue

    return queue(state, action)


def phase_separation(context: ServiceContext) -> ChemWorldPhaseSeparationServices:
    from chemworld.runtime.phase_separation_services import ChemWorldPhaseSeparationServices

    return ChemWorldPhaseSeparationServices(
        context.world,
        context.species_view,
        nominal_pair_contract=context.partition_nominal_pair_contract,
        liquid_runtime=getattr(context.services["reaction_thermal"], "liquid", None),
        thermal=context.services["reaction_thermal"],
    )


def crystallization(context: ServiceContext) -> ChemWorldCrystallizationServices:
    from chemworld.runtime.crystallization_services import ChemWorldCrystallizationServices

    return ChemWorldCrystallizationServices(
        context.world, context.species_view, thermal=context.services["reaction_thermal"],
        physics=None if context.composition is None else context.composition.physics,
    )


def distillation(context: ServiceContext) -> ChemWorldDistillationServices:
    from chemworld.runtime.distillation_services import ChemWorldDistillationServices

    return ChemWorldDistillationServices(
        context.world, context.species_view, thermal=context.services["reaction_thermal"]
    )


def continuous_flow(context: ServiceContext) -> ChemWorldFlowServices:
    from chemworld.runtime.flow_services import ChemWorldFlowServices

    return ChemWorldFlowServices(context.species_view, context.services["reaction_thermal"])


def continuous_streams(context: ServiceContext) -> ContinuousStreamServices:
    from chemworld.runtime.continuous_streams import ContinuousStreamServices

    return ContinuousStreamServices(
        context.world, context.species_view, context.services["reaction_thermal"].advancer
    )


def electrochemistry(context: ServiceContext) -> ChemWorldElectrochemicalServices:
    from chemworld.runtime.electrochemical_services import ChemWorldElectrochemicalServices

    return ChemWorldElectrochemicalServices(context.world, context.species_view)


def instrument_cost(context: ServiceContext) -> ChemWorldInstrumentCostServices:
    from chemworld.runtime.instrument_cost_services import ChemWorldInstrumentCostServices

    return ChemWorldInstrumentCostServices(
        context.constitution,
        modules=context.modules,
        thermal=context.services["reaction_thermal"],
    )


def lab_schedule(context: ServiceContext) -> LabScheduleServices:
    from chemworld.runtime.scheduling import LabScheduleServices

    return LabScheduleServices(context)


def lab_heat_choices(
    state: WorldState, validator: OperationValidator | None = None,
) -> tuple[str, ...]:
    from chemworld.runtime.scheduling import lab_device_choices

    return lab_device_choices(state, "heat")


def lab_analyzer_choices(
    state: WorldState, validator: OperationValidator | None = None,
) -> tuple[str, ...]:
    from chemworld.runtime.scheduling import lab_device_choices

    return lab_device_choices(state, "measure", allowed_instruments=(
        None if validator is None else validator.allowed_instruments
    ))


def lab_heat_selected(state: WorldState, action: dict[str, Any]) -> bool:
    from chemworld.runtime.scheduling import lab_device_selected

    return lab_device_selected(state, action, "heat")


def lab_analyzer_selected(state: WorldState, action: dict[str, Any]) -> bool:
    from chemworld.runtime.scheduling import lab_device_selected

    return lab_device_selected(state, action, "measure")


def configure_instrument(state: WorldState, action: dict[str, Any]) -> WorldState:
    from chemworld.runtime.spectral_settings import configure_instrument as execute

    return execute(state, action)


def terminate(state: WorldState) -> WorldState:
    return state.replace(terminated=True)
