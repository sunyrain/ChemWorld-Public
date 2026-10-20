"""Lazy constructors used by built-in module declarations.

Imports happen at runtime construction, so physical declarations do not depend
on runtime initialization order. Factories receive only the current world context.
"""


def configure_chromatography(state, action):
    from chemworld.runtime.chromatography import configure

    return configure(state, action)


def calibrate_chromatography(state, action):
    from chemworld.runtime.chromatography import calibrate

    return calibrate(state, action)


def primitive(context):
    from chemworld.runtime.primitive_services import ChemWorldPrimitiveOperationServices

    return ChemWorldPrimitiveOperationServices(
        context.world, context.species_view, context.services["material_routing"]
    )


def material_routing(context):
    from chemworld.runtime.material_routing import ChemWorldMaterialRoutingServices

    return ChemWorldMaterialRoutingServices(
        None if context.composition is None else context.composition.physics
    )


def component_network(context):
    from chemworld.runtime.component_network import ComponentNetworkServices

    return ComponentNetworkServices(context.services["material_routing"], context.composition)


def reaction_thermal(context):
    from chemworld.runtime.reaction_thermal_services import ChemWorldReactionThermalServices

    return ChemWorldReactionThermalServices(
        context.world,
        context.species_view,
        composition=context.composition,
        modules=context.modules,
    )


def bulk_process(context):
    from dataclasses import replace

    from chemworld.physchem.reactor_solvers import reaction_network_process

    network = replace(
        context.network,
        species=tuple(species for species in context.network.species if species.phase != "surface"),
    )
    return (reaction_network_process(network, process_id=context.component.id),)


def surface_processes(context):
    from chemworld.physchem.surface_rates import surface_process
    from chemworld.physchem.surface_specs import (
        DEFAULT_SURFACE_MECHANISM,
        MULTISITE_STATE_TYPE,
        SURFACE_STATE_TYPE,
        MultisiteSurfaceSpec,
        surface_species_energies,
    )

    model_id = context.component.parameters.get("surface_id")
    if model_id is not None and model_id not in context.physics.surface_models:
        raise ValueError("surface component must name a declared native mechanism")
    spec = (
        DEFAULT_SURFACE_MECHANISM if model_id is None else context.physics.surface_models[model_id]
    )
    if isinstance(spec, MultisiteSurfaceSpec):
        from chemworld.physchem.coupled_system import ConstantCvThermodynamics

        caloric = context.physics.thermal_models[context.state.vessel_id].thermodynamics
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
        raise ValueError("surface process requires an owned surface inventory")
    return tuple(surface_process(slot, spec) for slot in slots)


def sorbent_processes(context):
    from chemworld.physchem.sorbent_kinetics import (
        SORBENT_STATE_TYPE,
        sorbent_process,
        validate_sorbent_slot,
    )

    model_id = context.component.parameters["sorbent_id"]
    spec = context.physics.sorbent_models[model_id]
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


def control_program(context):
    from chemworld.runtime.control_program import ControlProgramServices

    return ControlProgramServices(context.services["reaction_thermal"])


def transient_electrode(composition):
    from chemworld.runtime.transient_electrode import TransientElectrodeRuntime

    return TransientElectrodeRuntime(composition)


def aqueous_mineral(composition):
    from chemworld.runtime.aqueous_mineral import AqueousMineralRuntime

    return AqueousMineralRuntime(composition)


def mineral_kinetics(composition):
    from chemworld.runtime.mineral_kinetics import MineralKineticsRuntime

    return MineralKineticsRuntime(composition)


def battery_spm(composition):
    from chemworld.runtime.battery import BatteryRuntime

    return BatteryRuntime(composition)


def finite_transport(composition):
    from chemworld.runtime.transport_devices import TransportRuntime

    return TransportRuntime(composition)


def stochastic(composition):
    from chemworld.runtime.stochastic import StochasticRuntime

    return StochasticRuntime(composition)


def ro_membrane(composition):
    from chemworld.runtime.membranes import MembraneRuntime

    return MembraneRuntime(composition)


def solid_solution(composition):
    from chemworld.runtime.solid_solutions import SolidSolutionRuntime

    return SolidSolutionRuntime(composition)


def configure_electrical_control(state, action):
    from chemworld.runtime.control_program import configure_electrical_control as configure

    return configure(state, action)


def queue_electrical_stage(state, action):
    from chemworld.runtime.control_program import queue_electrical_stage as queue

    return queue(state, action)


def phase_separation(context):
    from chemworld.runtime.phase_separation_services import ChemWorldPhaseSeparationServices

    return ChemWorldPhaseSeparationServices(
        context.world,
        context.species_view,
        nominal_pair_contract=context.partition_nominal_pair_contract,
        liquid_runtime=context.services["reaction_thermal"].liquid,
        thermal=context.services["reaction_thermal"],
    )


def crystallization(context):
    from chemworld.runtime.crystallization_services import ChemWorldCrystallizationServices

    return ChemWorldCrystallizationServices(
        context.world, context.species_view, thermal=context.services["reaction_thermal"]
    )


def distillation(context):
    from chemworld.runtime.distillation_services import ChemWorldDistillationServices

    return ChemWorldDistillationServices(
        context.world, context.species_view, thermal=context.services["reaction_thermal"]
    )


def continuous_flow(context):
    from chemworld.runtime.flow_services import ChemWorldFlowServices

    return ChemWorldFlowServices(context.species_view, context.services["reaction_thermal"])


def continuous_streams(context):
    from chemworld.runtime.continuous_streams import ContinuousStreamServices

    return ContinuousStreamServices(
        context.world, context.species_view, context.services["reaction_thermal"].advancer
    )


def electrochemistry(context):
    from chemworld.runtime.electrochemical_services import ChemWorldElectrochemicalServices

    return ChemWorldElectrochemicalServices(context.world, context.species_view)


def instrument_cost(context):
    from chemworld.runtime.instrument_cost_services import ChemWorldInstrumentCostServices

    return ChemWorldInstrumentCostServices(
        context.constitution,
        modules=context.modules,
        thermal=context.services["reaction_thermal"],
    )


def lab_schedule(context):
    from chemworld.runtime.scheduling import LabScheduleServices

    return LabScheduleServices(context)


def configure_instrument(state, action):
    from chemworld.runtime.spectral_settings import configure_instrument as execute

    return execute(state, action)


def terminate(state):
    return state.replace(terminated=True)
