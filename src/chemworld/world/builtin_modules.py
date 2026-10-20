"""Built-in declarations: one owner for every operation and service factory."""

from chemworld.foundation.ontology import Operation
from chemworld.runtime import service_factories as factories
from chemworld.world.domain_modules import (
    ActionFieldSpec,
    ComponentCapability,
    DomainModuleSpec,
    InstrumentSpec,
    OperationSpec,
    ParameterRule,
    ServiceFactorySpec,
)
from chemworld.world.instruments import (
    _builtin_chemworld_instruments,
    _builtin_instrument_contracts,
)


def _instruments():
    from chemworld.world.aqueous_instruments import aqueous_instrument_specs
    from chemworld.world.biochemical_instruments import biochemical_instrument_specs
    from chemworld.world.chromatography_instruments import chromatography_instrument_specs
    from chemworld.world.electrode_instruments import electrode_instrument_specs
    from chemworld.world.flash_instruments import flash_instrument_specs
    from chemworld.world.gas_instruments import gas_instrument_specs
    from chemworld.world.membrane_instruments import membrane_instrument_spec
    from chemworld.world.solid_instruments import solid_instrument_spec
    from chemworld.world.transport_instruments import transport_instrument_spec

    contracts = _builtin_instrument_contracts(include_particle_size=True)
    return (
        tuple(
            InstrumentSpec(value, contracts[key], optional=key == "particle_size")
            for key, value in _builtin_chemworld_instruments(include_particle_size=True).items()
        )
        + gas_instrument_specs()
        + flash_instrument_specs()
        + electrode_instrument_specs()
        + chromatography_instrument_specs()
        + biochemical_instrument_specs()
        + (transport_instrument_spec(),)
        + aqueous_instrument_specs()
        + (membrane_instrument_spec(),)
        + (solid_instrument_spec(),)
    )


BUILTIN_MODULES = (
    DomainModuleSpec(
        module_id="lab_schedule",
        operations=(
            OperationSpec(
                Operation("queue_lab_heat", "Queue the shared heater",
                          ("duration_s", "target_temperature_K"), ("not_terminated",)),
                "lab_schedule", "lab_schedule", "queue",
                fields=(ActionFieldSpec("duration_s", "s", (1.0, 14400.0)),
                        ActionFieldSpec("target_temperature_K", "K", (1.0, 3000.0))),
                affected_ledgers=("equipment", "process"),
            ),
            OperationSpec(
                Operation("queue_lab_measure", "Queue the shared offline analyzer",
                          (), ("not_terminated",)),
                "lab_schedule", "lab_schedule", "queue", affected_ledgers=("equipment", "process"),
            ),
        ),
        services=(ServiceFactorySpec(
            "lab_schedule", "lab_schedule", factories.lab_schedule,
            "LabScheduleServices", ("reaction_thermal", "instrument_cost"),
        ),),
        components=(("lab_schedule", ComponentCapability(
            operations=("queue_lab_heat", "queue_lab_measure", "wait"),
            interfaces=("time",), state_owners=("lab_clock",),
        )),),
        dependencies=("thermal", "observation"),
    ),
    DomainModuleSpec(
        module_id="solid_solution",
        components=(("solid_solution", ComponentCapability(
            operations=("heat", "wait", "pause_control", "resume_control"),
            interfaces=("material", "temperature"), state_owners=("solid_phase_equilibrium",),
        )),),
        advance_strategies=(("binary-solid-equilibrium", factories.solid_solution),),
        dependencies=("thermal",),
    ),
    DomainModuleSpec(
        module_id="mineral_kinetics",
        components=(("mineral_kinetics", ComponentCapability(
            operations=("add_component", "wait", "sample", "route_material"),
            interfaces=("material", "aqueous_equilibrium"),
            state_owners=("phases", "calcite_surface_reference"),
            outputs=("aqueous", "calcite"), input_ports=("aqueous", "calcite"),
        )),),
        advance_strategies=(("finite-calcite-kinetics", factories.mineral_kinetics),),
        dependencies=("aqueous_mineral",),
    ),
    DomainModuleSpec(
        module_id="stochastic",
        components=(("stochastic", ComponentCapability(
            operations=("wait",), interfaces=("material",), minimum_operations=1,
            state_owners=("phases", "stochastic_continuation"),
        )),),
        advance_strategies=(("gillespie-direct", factories.stochastic),),
        dependencies=("thermal",),
    ),
    DomainModuleSpec(
        module_id="ro_membrane",
        components=(("ro_membrane", ComponentCapability(
            operations=("set_flow_rate", "run_flow", "wait"),
            interfaces=("material", "flow_rate", "temperature"),
            state_owners=("liquid_transport_enthalpy_v1", "membrane_bound_foulant_v1"),
        )),),
        advance_strategies=(("finite-dilute-ro", factories.ro_membrane),),
        dependencies=("continuous_flow",),
    ),
    DomainModuleSpec(
        module_id="finite_transport",
        components=(("finite_transport", ComponentCapability(
            operations=("set_flow_rate", "run_flow", "wait"),
            interfaces=("material", "flow_rate", "temperature"),
            state_owners=("liquid_transport_enthalpy_v1",),
        )),),
        advance_strategies=(("finite-pump-transport", factories.finite_transport),),
        dependencies=("continuous_flow",),
    ),
    DomainModuleSpec(
        module_id="aqueous_mineral",
        components=(("aqueous_mineral", ComponentCapability(
            operations=("add_component", "wait", "sample"),
            interfaces=("material", "aqueous_equilibrium"), state_owners=("phases",),
        )),),
        advance_strategies=(("finite-carbonate-calcite-equilibrium", factories.aqueous_mineral),),
        dependencies=("thermal", "core"),
    ),
    DomainModuleSpec(
        module_id="programmed_chromatography",
        operations=(
            OperationSpec(
                Operation(
                    "configure_chromatography",
                    "Select an installed chromatography program",
                    ("instrument", "program_index"),
                    ("not_terminated",),
                ),
                "observation",
                "instrument_cost",
                "configure",
                fields=(
                    ActionFieldSpec(
                        "instrument", "categorical", choices=("programmed_hplc", "programmed_gc")
                    ),
                    ActionFieldSpec("program_index", "dimensionless", (0, 15)),
                ),
                function=factories.configure_chromatography,
                affected_ledgers=("equipment", "process"),
            ),
            OperationSpec(
                Operation(
                    "calibrate_chromatography",
                    "Record acquired standard peak areas",
                    ("instrument", "standard_1_M", "standard_2_M", "standard_3_M"),
                    ("not_terminated",),
                ),
                "observation",
                "instrument_cost",
                "calibrate",
                fields=(
                    ActionFieldSpec(
                        "instrument", "categorical", choices=("programmed_hplc", "programmed_gc")
                    ),
                    *(ActionFieldSpec(f"standard_{i}_M", "mol/L", (0, 10)) for i in range(1, 4)),
                ),
                function=factories.calibrate_chromatography,
                affected_ledgers=("equipment", "process"),
            ),
        ),
        components=(
            (
                "programmed_chromatography",
                ComponentCapability(
                    operations=("configure_chromatography", "calibrate_chromatography"),
                    state_owners=("chromatography_retained_inventory_v1",),
                    interfaces=("material", "instrument"),
                ),
            ),
        ),
        dependencies=("observation",),
    ),
    DomainModuleSpec(
        module_id="sorbent_contact",
        processes=(("sorbent_contact", factories.sorbent_processes),),
        components=(
            (
                "sorbent_contact",
                ComponentCapability(
                    operations=("dry", "add_solvent"),
                    processes=("sorbent_contact",),
                    dependencies=("thermal",),
                    parameters=(("sorbent_id", ParameterRule("string")),),
                    state_owners=("sorbent_loading",),
                    interfaces=("material", "temperature", "sorbent"),
                ),
            ),
        ),
        dependencies=("thermal", "separation"),
    ),
    DomainModuleSpec(
        module_id="electrical_control",
        operations=(
            OperationSpec(
                Operation(
                    "configure_electrical_control",
                    "Configure an electrical step or ramp",
                    (),
                    ("not_terminated",),
                ),
                "electrical_control",
                "control_program",
                "apply",
                fields=(
                    ActionFieldSpec("potential_V", "V", (-10.0, 10.0), required=False),
                    ActionFieldSpec("current_A", "A", (-10.0, 10.0), required=False),
                    ActionFieldSpec("ramp_duration_s", "s", (0.0, 14400.0), required=False),
                ),
                function=factories.configure_electrical_control,
                affected_ledgers=("equipment", "process"),
            ),
            OperationSpec(
                Operation(
                    "queue_electrical_stage",
                    "Append a finite electrical control stage",
                    ("stage_duration_s",),
                    ("not_terminated",),
                ),
                "electrical_control",
                "control_program",
                "apply",
                fields=(
                    ActionFieldSpec("potential_V", "V", (-10.0, 10.0), required=False),
                    ActionFieldSpec("current_A", "A", (-10.0, 10.0), required=False),
                    ActionFieldSpec("stage_duration_s", "s", (0.000001, 14400.0)),
                    ActionFieldSpec("stop_voltage_below_V", "V", (-10.0, 10.0), required=False),
                    ActionFieldSpec("stop_voltage_above_V", "V", (-10.0, 10.0), required=False),
                    ActionFieldSpec("stop_abs_current_A", "A", (0.0, 10.0), required=False),
                ),
                function=factories.queue_electrical_stage,
                affected_ledgers=("equipment", "process"),
            ),
        ),
        components=(
            (
                "transient_electrode",
                ComponentCapability(
                    operations=("configure_electrical_control", "wait"),
                    interfaces=("material", "electrical_control", "spatial_diffusion"),
                    state_owners=("phases", "electrode_charge"),
                ),
            ),
        ),
        advance_strategies=(("finite-transient-electrode", factories.transient_electrode),),
        dependencies=("thermal",),
    ),
    DomainModuleSpec(
        module_id="battery_spm",
        components=(
            (
                "battery_spm",
                ComponentCapability(
                    operations=(
                        "configure_electrical_control",
                        "queue_electrical_stage",
                        "wait",
                        "heat",
                        "pause_control",
                        "resume_control",
                    ),
                    interfaces=("electrical_control", "spatial_diffusion"),
                    state_owners=("battery_inventory",),
                ),
            ),
        ),
        advance_strategies=(("single-particle-battery", factories.battery_spm),),
        dependencies=("electrical_control",),
    ),
    DomainModuleSpec(
        module_id="vacuum_concentration",
        components=(
            (
                "vacuum_concentration",
                ComponentCapability(
                    operations=("concentrate",),
                    dependencies=("thermal",),
                    state_owners=("phase_transition",),
                    interfaces=("material", "temperature", "phase_equilibrium"),
                ),
            ),
        ),
        dependencies=("separation", "thermal"),
    ),
    DomainModuleSpec(
        module_id="dynamic_distillation",
        components=(
            (
                "dynamic_distillation",
                ComponentCapability(
                    operations=("distill", "collect_fraction"),
                    dependencies=("thermal",),
                    state_owners=("phase_transition", "fraction_state"),
                    interfaces=("material", "temperature", "phase_equilibrium"),
                ),
            ),
        ),
        dependencies=("distillation", "thermal"),
    ),
    DomainModuleSpec(
        module_id="liquid_equilibrium",
        components=(
            (
                "liquid_equilibrium",
                ComponentCapability(
                    operations=("mix", "wash", "settle", "wait"),
                    dependencies=("phase",),
                    state_owners=("phases",),
                    interfaces=("material", "phase", "temperature"),
                ),
            ),
        ),
        dependencies=("separation", "thermal"),
    ),
    DomainModuleSpec(
        module_id="vapor_liquid_equilibrium",
        components=(
            (
                "vapor_liquid_equilibrium",
                ComponentCapability(
                    operations=(),
                    dependencies=("thermal",),
                    state_owners=("phases",),
                    interfaces=("material", "temperature", "phase_equilibrium"),
                ),
            ),
        ),
        dependencies=("thermal",),
    ),
    DomainModuleSpec(
        module_id="surface_kinetics",
        processes=(("surface_kinetics", factories.surface_processes),),
        components=(
            (
                "surface_reaction",
                ComponentCapability(
                    operations=(),
                    processes=("surface_kinetics",),
                    dependencies=("thermal",),
                    state_owners=("surface_kinetics",),
                    interfaces=("material", "temperature", "surface"),
                    parameters=(("surface_id", ParameterRule("string")),),
                ),
            ),
        ),
        dependencies=("thermal",),
    ),
    DomainModuleSpec(
        module_id="core",
        processes=(("bulk_reaction", factories.bulk_process),),
        operations=(
            OperationSpec(
                Operation(
                    "select_vessel",
                    "Select a declared component vessel",
                    ("vessel",),
                    ("not_terminated",),
                ),
                "reaction",
                "component_network",
                "select_vessel",
                kind="primitive",
                fields=(ActionFieldSpec("vessel", "categorical", None, tuple(range(16))),),
                affected_ledgers=(
                    "process",
                    "inactive_vessels",
                    "samples",
                    "species",
                    "phases",
                    "vessels",
                    "equipment",
                    "thermal",
                ),
            ),
            OperationSpec(
                Operation(
                    "route_material",
                    "Route a declared material output",
                    ("connection", "transfer_fraction", "mixing"),
                    ("not_terminated",),
                ),
                "reaction",
                "component_network",
                "route_material",
                kind="primitive",
                fields=(
                    ActionFieldSpec("connection", "categorical", None, tuple(range(64))),
                    ActionFieldSpec(
                        "transfer_fraction",
                        "dimensionless",
                        (0.0, 1.0),
                        (),
                        runtime_bounds=(0.0, 1.0),
                    ),
                    ActionFieldSpec("mixing", "categorical", None, (0, 1)),
                ),
                affected_ledgers=(
                    "process",
                    "inactive_vessels",
                    "samples",
                    "species",
                    "phases",
                    "vessels",
                    "equipment",
                    "thermal",
                ),
            ),
            OperationSpec(
                Operation(
                    "create_container",
                    "Open a single-use storage container",
                    ("container", "capacity_L"),
                    ("not_terminated",),
                ),
                "reaction",
                "material_routing",
                "create_container",
                kind="primitive",
                fields=(
                    ActionFieldSpec("container", "categorical", None, tuple(range(1, 17))),
                    ActionFieldSpec("capacity_L", "L", (0.0001, 0.1), ()),
                ),
                affected_ledgers=(
                    "process",
                    "samples",
                    "species",
                    "phases",
                    "vessels",
                    "equipment",
                ),
            ),
            OperationSpec(
                Operation(
                    "transfer_material",
                    "Transfer or merge a representative material aliquot",
                    ("source_container", "destination_container", "transfer_fraction", "mixing"),
                    ("not_terminated", "material_routing_valid"),
                ),
                "reaction",
                "material_routing",
                "transfer_material",
                kind="primitive",
                fields=(
                    ActionFieldSpec("source_container", "categorical", None, tuple(range(33))),
                    ActionFieldSpec("destination_container", "categorical", None, tuple(range(33))),
                    ActionFieldSpec(
                        "transfer_fraction",
                        "dimensionless",
                        (0.0, 1.0),
                        (),
                        runtime_bounds=(0.0, 1.0),
                    ),
                    ActionFieldSpec("mixing", "categorical", None, (0, 1)),
                ),
                affected_ledgers=(
                    "process",
                    "samples",
                    "species",
                    "phases",
                    "vessels",
                    "equipment",
                ),
            ),
            OperationSpec(
                Operation("add_reagent", "Add reagent", ("amount_mol",), ("not_terminated",)),
                "reaction",
                "primitive_operations",
                "add_reagent",
                kind="primitive",
                fields=(ActionFieldSpec("amount_mol", "mol", (0.0, 0.04), ()),),
                stock_field=("reagent_mol", "amount_mol"),
                affected_ledgers=("process", "species", "phases"),
            ),
            OperationSpec(
                Operation(
                    "add_component",
                    "Dose one catalog feed independently",
                    ("component", "amount_mol"),
                    ("not_terminated",),
                ),
                "reaction",
                "primitive_operations",
                "add_component",
                kind="primitive",
                fields=(
                    ActionFieldSpec("component", "unitless", None, tuple(range(16))),
                    ActionFieldSpec("amount_mol", "mol", (0.0, 0.04), ()),
                ),
                stock_field=("reagent_mol", "amount_mol"),
                affected_ledgers=("process", "species", "phases"),
            ),
            OperationSpec(
                Operation(
                    "add_solvent", "Add solvent", ("volume_L", "solvent"), ("not_terminated",)
                ),
                "reaction",
                "primitive_operations",
                "add_solvent",
                kind="primitive",
                fields=(
                    ActionFieldSpec("volume_L", "L", (0.0, 0.08), ()),
                    ActionFieldSpec("solvent", "categorical", None, tuple(range(4))),
                ),
                stock_field=("solvent_L", "volume_L"),
                affected_ledgers=(
                    "process",
                    "species",
                    "phases",
                    "thermal",
                    "vessels",
                    "equipment",
                ),
            ),
            OperationSpec(
                Operation(
                    "add_catalyst",
                    "Add catalyst",
                    ("catalyst_amount_mol", "catalyst"),
                    ("not_terminated",),
                ),
                "reaction",
                "primitive_operations",
                "add_catalyst",
                kind="primitive",
                fields=(
                    ActionFieldSpec("catalyst_amount_mol", "mol", (0.0, 0.005), ()),
                    ActionFieldSpec("catalyst", "categorical", None, tuple(range(4))),
                ),
                stock_field=("catalyst_mol", "catalyst_amount_mol"),
                affected_ledgers=("process", "species", "phases", "equipment"),
            ),
            OperationSpec(
                Operation("sample", "Sample", ("sample_volume_L",), ("has_volume",)),
                "reaction",
                "primitive_operations",
                "sample",
                kind="primitive",
                fields=(ActionFieldSpec("sample_volume_L", "L", (0.0, 0.002), ()),),
                affected_ledgers=("process", "species", "phases"),
            ),
            OperationSpec(
                Operation("quench", "Quench", (), ("has_volume",)),
                "reaction",
                "primitive_operations",
                "quench",
                kind="primitive",
                fields=(),
                takes_action=False,
                affected_ledgers=("process", "thermal", "vessels"),
            ),
            OperationSpec(
                Operation(
                    "evaporate",
                    "Evaporate solvent",
                    ("target_temperature_K", "duration_s"),
                    ("has_volume",),
                ),
                "distillation",
                "primitive_operations",
                "evaporate",
                kind="domain",
                fields=(
                    ActionFieldSpec(
                        "target_temperature_K",
                        "K",
                        (298.15, 390.0),
                        (),
                        runtime_bounds=(298.15, 390.0),
                    ),
                    ActionFieldSpec(
                        "duration_s", "s", (1.0, 14400.0), (), runtime_bounds=(1.0, 14400.0)
                    ),
                ),
                affected_ledgers=("process", "species", "phases", "thermal", "vessels"),
            ),
            OperationSpec(
                Operation("terminate", "Terminate", (), ("has_material", "not_terminated")),
                "reaction",
                "primitive_operations",
                "terminate",
                kind="terminal",
                fields=(),
                takes_action=False,
                function=factories.terminate,
                affected_ledgers=("process",),
            ),
        ),
        services=(
            ServiceFactorySpec(
                "primitive_operations",
                "primitive",
                factories.primitive,
                "ChemWorldPrimitiveOperationServices",
                ("material_routing",),
            ),
            ServiceFactorySpec(
                "material_routing",
                "material_routing",
                factories.material_routing,
                "ChemWorldMaterialRoutingServices",
                (),
            ),
            ServiceFactorySpec(
                "component_network",
                "component_network",
                factories.component_network,
                "ComponentNetworkServices",
                ("material_routing",),
            ),
        ),
        components=(
            (
                "material_input",
                ComponentCapability(
                    operations=("add_component",),
                    interfaces=("material",),
                ),
            ),
            (
                "reaction",
                ComponentCapability(
                    operations=("add_reagent", "add_component", "add_solvent", "add_catalyst"),
                    processes=("bulk_reaction",),
                    template="reaction-to-assay",
                    priority=20,
                    minimum_operations=4,
                    interfaces=("material", "reaction_event"),
                    state_owners=("material_transformation", "reaction_event"),
                    required_all=("add_reagent", "add_solvent"),
                    parameters=(
                        ("family", ParameterRule(value_kind="string")),
                        (
                            "controls",
                            ParameterRule(
                                value_kind="choices",
                                choices=("reagent", "catalyst", "solvent", "temperature", "time"),
                            ),
                        ),
                    ),
                ),
            ),
        ),
    ),
    DomainModuleSpec(
        module_id="thermal",
        operations=(
            OperationSpec(
                Operation(
                    "configure_control",
                    "Persistent bounded process control",
                    (
                        "target_temperature_K",
                        "pressure_Pa",
                        "atmosphere",
                        "ramp_rate_K_s",
                        "control_interval_s",
                        "headspace_L",
                    ),
                    ("not_terminated",),
                ),
                "reaction",
                "control_program",
                "apply",
                kind="primitive",
                fields=(
                    ActionFieldSpec(
                        "target_temperature_K",
                        "K",
                        (250.0, 430.0),
                        (),
                        runtime_bounds=(250.0, 430.0),
                    ),
                    ActionFieldSpec("pressure_Pa", "Pa", (50000.0, 400000.0), ()),
                    ActionFieldSpec("atmosphere", "categorical", None, (0, 1, 2)),
                    ActionFieldSpec("ramp_rate_K_s", "K/s", (0.001, 1.0), ()),
                    ActionFieldSpec("control_interval_s", "s", (1.0, 60.0), ()),
                    ActionFieldSpec("headspace_L", "L", (0.001, 0.1), ()),
                ),
                affected_ledgers=("process", "equipment", "species", "phases", "thermal"),
            ),
            OperationSpec(
                Operation(
                    "queue_control_stage",
                    "Persistent bounded process control",
                    (
                        "target_temperature_K",
                        "pressure_Pa",
                        "atmosphere",
                        "ramp_rate_K_s",
                        "stage_duration_s",
                    ),
                    ("not_terminated",),
                ),
                "reaction",
                "control_program",
                "apply",
                kind="primitive",
                fields=(
                    ActionFieldSpec(
                        "target_temperature_K",
                        "K",
                        (250.0, 430.0),
                        (),
                        runtime_bounds=(250.0, 430.0),
                    ),
                    ActionFieldSpec("pressure_Pa", "Pa", (50000.0, 400000.0), ()),
                    ActionFieldSpec("atmosphere", "categorical", None, (0, 1, 2)),
                    ActionFieldSpec("ramp_rate_K_s", "K/s", (0.001, 1.0), ()),
                    ActionFieldSpec("stage_duration_s", "s", (1.0, 14400.0), ()),
                ),
                affected_ledgers=("process", "equipment", "species", "phases", "thermal"),
            ),
            OperationSpec(
                Operation(
                    "set_control_feedback",
                    "Persistent bounded process control",
                    (
                        "feedback_sensor",
                        "feedback_threshold",
                        "feedback_direction",
                        "feedback_response",
                    ),
                    ("not_terminated",),
                ),
                "reaction",
                "control_program",
                "apply",
                kind="primitive",
                fields=(
                    ActionFieldSpec("feedback_sensor", "categorical", None, (0, 1, 2)),
                    ActionFieldSpec(
                        "feedback_threshold", "sensor_native_unit", (0.0, 550000.0), ()
                    ),
                    ActionFieldSpec("feedback_direction", "categorical", None, (0, 1)),
                    ActionFieldSpec("feedback_response", "categorical", None, (0, 1)),
                ),
                affected_ledgers=("process", "equipment", "species", "phases", "thermal"),
            ),
            OperationSpec(
                Operation(
                    "advance_control",
                    "Persistent bounded process control",
                    ("duration_s",),
                    ("not_terminated",),
                ),
                "reaction",
                "control_program",
                "apply",
                kind="primitive",
                fields=(
                    ActionFieldSpec(
                        "duration_s", "s", (1.0, 14400.0), (), runtime_bounds=(1.0, 14400.0)
                    ),
                ),
                affected_ledgers=("process", "equipment", "species", "phases", "thermal"),
            ),
            OperationSpec(
                Operation(
                    "pause_control", "Persistent bounded process control", (), ("not_terminated",)
                ),
                "reaction",
                "control_program",
                "apply",
                kind="primitive",
                fields=(),
                affected_ledgers=("process", "equipment", "species", "phases", "thermal"),
            ),
            OperationSpec(
                Operation(
                    "resume_control", "Persistent bounded process control", (), ("not_terminated",)
                ),
                "reaction",
                "control_program",
                "apply",
                kind="primitive",
                fields=(),
                affected_ledgers=("process", "equipment", "species", "phases", "thermal"),
            ),
            OperationSpec(
                Operation(
                    "heat",
                    "Heat",
                    ("target_temperature_K", "duration_s"),
                    ("has_volume", "has_material"),
                ),
                "reaction",
                "reaction_thermal",
                "integrate",
                kind="primitive",
                fields=(
                    ActionFieldSpec("target_temperature_K", "K", (250.0, 520.0), ()),
                    ActionFieldSpec(
                        "duration_s", "s", (1.0, 14400.0), (), runtime_bounds=(1.0, 14400.0)
                    ),
                    ActionFieldSpec(
                        "stirring_speed_rpm", "rpm", (100.0, 1200.0), (), required=False
                    ),
                ),
                method_kwargs=(("heat", True),),
                affected_ledgers=(
                    "process",
                    "species",
                    "phases",
                    "thermal",
                    "vessels",
                    "equipment",
                ),
            ),
            OperationSpec(
                Operation(
                    "wait",
                    "Wait",
                    ("duration_s",),
                    ("has_volume", "has_material"),
                ),
                "reaction",
                "reaction_thermal",
                "integrate",
                kind="primitive",
                fields=(
                    ActionFieldSpec(
                        "duration_s", "s", (1.0, 14400.0), (), runtime_bounds=(1.0, 14400.0)
                    ),
                    ActionFieldSpec(
                        "stirring_speed_rpm", "rpm", (100.0, 1200.0), (), required=False
                    ),
                ),
                method_kwargs=(("heat", False),),
                affected_ledgers=(
                    "process",
                    "species",
                    "phases",
                    "thermal",
                    "vessels",
                    "equipment",
                ),
            ),
        ),
        services=(
            ServiceFactorySpec(
                "reaction_thermal",
                "reaction_thermal",
                factories.reaction_thermal,
                "ChemWorldReactionThermalServices",
                (),
            ),
            ServiceFactorySpec(
                "control_program",
                "control_program",
                factories.control_program,
                "ControlProgramServices",
                ("reaction_thermal",),
            ),
        ),
        components=(
            (
                "thermal",
                ComponentCapability(
                    operations=(
                        "heat",
                        "wait",
                        "configure_control",
                        "queue_control_stage",
                        "set_control_feedback",
                        "advance_control",
                        "pause_control",
                        "resume_control",
                    ),
                    interfaces=("temperature", "time", "energy", "safety"),
                    state_owners=("temperature_control",),
                    required_any=("cool_crystallize", "distill", "heat", "run_flow", "wait"),
                    parameters=(
                        (
                            "temperature_range_K",
                            ParameterRule(
                                value_kind="range", unit="K", minimum=250.0, maximum=430.0
                            ),
                        ),
                        (
                            "duration_range_s",
                            ParameterRule(
                                value_kind="range", unit="s", minimum=1.0, maximum=14400.0
                            ),
                        ),
                    ),
                ),
            ),
        ),
    ),
    DomainModuleSpec(
        module_id="separation",
        operations=(
            OperationSpec(
                Operation("add_phase", "Add phase", ("phase", "volume_L"), ("not_terminated",)),
                "separation",
                "phase_separation",
                "add_phase",
                kind="primitive",
                fields=(
                    ActionFieldSpec(
                        "phase",
                        "categorical",
                        None,
                        ("aqueous", "organic"),
                        runtime_choices=("aqueous", "organic"),
                    ),
                    ActionFieldSpec("volume_L", "L", (0.0, 0.06), (), runtime_bounds=(0.0, 0.06)),
                ),
                stock_field=("phase_liquid_L", "volume_L"),
                affected_ledgers=("process", "species", "phases"),
            ),
            OperationSpec(
                Operation(
                    "add_extractant",
                    "Add extractant",
                    ("extractant", "volume_L"),
                    ("has_volume", "has_material", "not_terminated"),
                ),
                "separation",
                "phase_separation",
                "add_extractant",
                kind="primitive",
                fields=(
                    ActionFieldSpec(
                        "extractant",
                        "categorical",
                        None,
                        tuple(range(4)),
                        runtime_choices=tuple(range(4)),
                    ),
                    ActionFieldSpec("volume_L", "L", (0.0, 0.06), (), runtime_bounds=(0.0, 0.06)),
                ),
                stock_field=("extractant_L", "volume_L"),
                affected_ledgers=("process", "species", "phases"),
            ),
            OperationSpec(
                Operation(
                    "mix", "Mix phases", ("duration_s", "stirring_speed_rpm"), ("has_phase_system",)
                ),
                "separation",
                "phase_separation",
                "mix_phases",
                kind="primitive",
                fields=(
                    ActionFieldSpec(
                        "duration_s", "s", (1.0, 14400.0), (), runtime_bounds=(1.0, 14400.0)
                    ),
                    ActionFieldSpec("stirring_speed_rpm", "rpm", (100.0, 1200.0), ()),
                ),
                affected_ledgers=("process", "species", "phases", "equipment"),
            ),
            OperationSpec(
                Operation("settle", "Settle phases", ("duration_s",), ("has_phase_system",)),
                "separation",
                "phase_separation",
                "settle_phases",
                kind="primitive",
                fields=(
                    ActionFieldSpec(
                        "duration_s", "s", (1.0, 14400.0), (), runtime_bounds=(1.0, 14400.0)
                    ),
                ),
                affected_ledgers=("process", "species", "phases"),
            ),
            OperationSpec(
                Operation(
                    "separate_phase",
                    "Separate phase",
                    ("target_phase",),
                    ("has_phase_system", "phase_settled"),
                ),
                "separation",
                "phase_separation",
                "separate_phase",
                kind="primitive",
                fields=(
                    ActionFieldSpec(
                        "target_phase",
                        "categorical",
                        None,
                        ("aqueous", "organic"),
                        runtime_choices=("aqueous", "organic"),
                    ),
                ),
                affected_ledgers=("process", "species", "phases"),
            ),
            OperationSpec(
                Operation("wash", "Wash product phase", ("wash_volume_L",), ("has_phase_system",)),
                "separation",
                "phase_separation",
                "wash_phase",
                kind="macro",
                fields=(
                    ActionFieldSpec(
                        "wash_volume_L", "L", (1e-06, 0.04), (), runtime_bounds=(1e-06, 0.04)
                    ),
                ),
                stock_field=("wash_solvent_L", "wash_volume_L"),
                affected_ledgers=("process", "species", "phases"),
            ),
            OperationSpec(
                Operation("dry", "Dry product phase", (), ("has_phase_system",)),
                "separation",
                "phase_separation",
                "dry_phase",
                kind="macro",
                fields=(),
                takes_action=False,
                affected_ledgers=("process", "species", "phases"),
            ),
            OperationSpec(
                Operation(
                    "concentrate",
                    "Concentrate product phase",
                    ("duration_s",),
                    ("has_phase_system",),
                ),
                "separation",
                "phase_separation",
                "concentrate_phase",
                kind="macro",
                fields=(
                    ActionFieldSpec(
                        "duration_s", "s", (1.0, 14400.0), (), runtime_bounds=(1.0, 14400.0)
                    ),
                ),
                affected_ledgers=("process", "species", "phases"),
            ),
            OperationSpec(
                Operation(
                    "transfer",
                    "Transfer product phase",
                    ("transfer_fraction",),
                    ("has_phase_system",),
                ),
                "separation",
                "phase_separation",
                "transfer_phase",
                kind="primitive",
                fields=(
                    ActionFieldSpec(
                        "transfer_fraction",
                        "dimensionless",
                        (0.0001, 1.0),
                        (),
                        runtime_bounds=(0.0001, 1.0),
                    ),
                ),
                affected_ledgers=("process", "species", "phases"),
            ),
        ),
        services=(
            ServiceFactorySpec(
                "phase_separation",
                "phase_separation",
                factories.phase_separation,
                "ChemWorldPhaseSeparationServices",
                ("reaction_thermal",),
            ),
        ),
        components=(
            (
                "separation",
                ComponentCapability(
                    operations=(
                        "add_phase",
                        "add_extractant",
                        "mix",
                        "settle",
                        "separate_phase",
                        "wash",
                        "dry",
                        "concentrate",
                        "transfer",
                    ),
                    template="partition-discovery",
                    priority=30,
                    minimum_operations=4,
                    reactive_template="reaction-to-purification",
                    interfaces=("material", "phase", "volume", "sample_identity"),
                    dependencies=("phase",),
                    state_owners=("sample_routing",),
                    required_any=(
                        "add_extractant",
                        "add_phase",
                        "concentrate",
                        "dry",
                        "mix",
                        "separate_phase",
                        "settle",
                        "transfer",
                        "wash",
                    ),
                    parameters=(
                        (
                            "operations",
                            ParameterRule(
                                value_kind="choices",
                                choices=(
                                    "add_phase",
                                    "add_extractant",
                                    "mix",
                                    "settle",
                                    "separate_phase",
                                    "wash",
                                    "dry",
                                    "concentrate",
                                    "transfer",
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        ),
    ),
    DomainModuleSpec(
        module_id="crystallization",
        operations=(
            OperationSpec(
                Operation(
                    "resuspend_crystals",
                    "Isolate filtrate and resuspend retained crystals in fresh solvent",
                    ("volume_L", "solvent"),
                    ("not_terminated", "has_material", "resuspend_requires_filtered_crystals"),
                ),
                "crystallization",
                "crystallization",
                "resuspend_crystals",
                kind="domain",
                fields=(
                    ActionFieldSpec(
                        "volume_L", "L", (0.0001, 0.08), (), runtime_bounds=(0.0001, 0.08)
                    ),
                    ActionFieldSpec("solvent", "categorical", None, tuple(range(4))),
                ),
                stock_field=("solvent_L", "volume_L"),
                affected_ledgers=(
                    "process",
                    "species",
                    "phases",
                    "samples",
                    "vessels",
                    "equipment",
                ),
            ),
            OperationSpec(
                Operation(
                    "seed_crystals", "Seed crystallization", ("seed_mass_g",), ("has_volume",)
                ),
                "crystallization",
                "crystallization",
                "seed_crystals",
                kind="domain",
                fields=(
                    ActionFieldSpec(
                        "seed_mass_g",
                        "g",
                        (1e-06, 0.05),
                        (),
                        runtime_bounds=(1e-06, 0.05),
                        cumulative_limit=0.05,
                    ),
                ),
                stock_field=("seed_g", "seed_mass_g"),
                affected_ledgers=("process", "species", "phases", "equipment"),
            ),
            OperationSpec(
                Operation(
                    "cool_crystallize",
                    "Cool crystallization",
                    ("target_temperature_K", "duration_s"),
                    ("has_volume", "has_material"),
                ),
                "crystallization",
                "crystallization",
                "cool_crystallize",
                kind="domain",
                fields=(
                    ActionFieldSpec(
                        "target_temperature_K",
                        "K",
                        (250.0, 330.0),
                        (),
                        runtime_bounds=(250.0, 330.0),
                    ),
                    ActionFieldSpec(
                        "duration_s", "s", (1.0, 14400.0), (), runtime_bounds=(1.0, 14400.0)
                    ),
                ),
                affected_ledgers=("process", "species", "phases"),
            ),
            OperationSpec(
                Operation(
                    "filter_crystals",
                    "Filter crystals",
                    (),
                    ("has_material", "filter_requires_crystallization"),
                ),
                "crystallization",
                "crystallization",
                "filter_crystals",
                kind="domain",
                fields=(),
                takes_action=False,
                affected_ledgers=("process", "species", "phases"),
            ),
        ),
        services=(
            ServiceFactorySpec(
                "crystallization",
                "crystallization",
                factories.crystallization,
                "ChemWorldCrystallizationServices",
                ("reaction_thermal",),
            ),
        ),
        components=(
            (
                "crystal_thermal",
                ComponentCapability(
                    operations=(
                        "seed_crystals",
                        "cool_crystallize",
                        "filter_crystals",
                        "resuspend_crystals",
                        "create_container",
                        "transfer_material",
                    ),
                    interfaces=("temperature", "phase", "crystal_state", "sample"),
                    dependencies=("thermal",),
                    state_owners=("phase_transition", "crystal_state"),
                ),
            ),
            (
                "crystallization",
                ComponentCapability(
                    operations=(
                        "seed_crystals",
                        "cool_crystallize",
                        "filter_crystals",
                        "resuspend_crystals",
                    ),
                    template="reaction-to-crystallization",
                    priority=40,
                    minimum_operations=10,
                    minimum_measurements=2,
                    minimum_time_s=2.0,
                    interfaces=("temperature", "phase", "crystal_state", "sample"),
                    dependencies=("reaction", "thermal"),
                    state_owners=("phase_transition", "crystal_state"),
                    required_all=(
                        "cool_crystallize",
                        "filter_crystals",
                        "resuspend_crystals",
                        "seed_crystals",
                    ),
                    parameters=(
                        (
                            "temperature_range_K",
                            ParameterRule(
                                value_kind="range", unit="K", minimum=250.0, maximum=330.0
                            ),
                        ),
                        (
                            "seed_mass_range_g",
                            ParameterRule(
                                value_kind="range", unit="g", minimum=1e-06, maximum=0.05
                            ),
                        ),
                    ),
                ),
            ),
        ),
    ),
    DomainModuleSpec(
        module_id="distillation",
        operations=(
            OperationSpec(
                Operation(
                    "distill",
                    "Distill volatile fraction",
                    ("target_temperature_K", "duration_s", "reflux_ratio"),
                    ("has_volume", "has_material"),
                ),
                "distillation",
                "distillation",
                "distill",
                kind="domain",
                fields=(
                    ActionFieldSpec(
                        "target_temperature_K",
                        "K",
                        (298.15, 430.0),
                        (),
                        runtime_bounds=(298.15, 430.0),
                    ),
                    ActionFieldSpec(
                        "duration_s", "s", (1.0, 14400.0), (), runtime_bounds=(1.0, 14400.0)
                    ),
                    ActionFieldSpec("reflux_ratio", "dimensionless", (0.0, 10.0), ()),
                    ActionFieldSpec("full_reflux", choices=(0, 1), required=False),
                ),
                affected_ledgers=(
                    "process",
                    "species",
                    "phases",
                    "thermal",
                    "vessels",
                    "equipment",
                ),
            ),
            OperationSpec(
                Operation(
                    "collect_fraction",
                    "Collect distillation fraction",
                    ("transfer_fraction",),
                    ("collect_fraction_requires_distillation",),
                ),
                "distillation",
                "distillation",
                "collect_fraction",
                kind="domain",
                fields=(
                    ActionFieldSpec(
                        "transfer_fraction",
                        "dimensionless",
                        (0.0001, 1.0),
                        (),
                        runtime_bounds=(0.0001, 1.0),
                    ),
                ),
                affected_ledgers=("process", "species", "phases"),
            ),
        ),
        services=(
            ServiceFactorySpec(
                "distillation",
                "distillation",
                factories.distillation,
                "ChemWorldDistillationServices",
                ("reaction_thermal",),
            ),
        ),
        components=(
            (
                "distillation",
                ComponentCapability(
                    operations=("evaporate", "distill", "collect_fraction"),
                    template="reaction-to-distillation",
                    priority=50,
                    minimum_operations=6,
                    minimum_time_s=2.0,
                    outputs=("distillate", "bottoms", "collected_fraction"),
                    interfaces=("temperature", "phase", "volatility", "fraction"),
                    dependencies=("reaction", "thermal"),
                    state_owners=("phase_transition", "fraction_state"),
                    required_all=("collect_fraction", "distill"),
                    parameters=(
                        (
                            "temperature_range_K",
                            ParameterRule(
                                value_kind="range", unit="K", minimum=298.15, maximum=430.0
                            ),
                        ),
                        (
                            "reflux_ratio_range",
                            ParameterRule(
                                value_kind="range", unit="dimensionless", minimum=0.0, maximum=10.0
                            ),
                        ),
                        (
                            "fraction_count",
                            ParameterRule(value_kind="integer", minimum=1.0, maximum=24.0),
                        ),
                    ),
                ),
            ),
        ),
    ),
    DomainModuleSpec(
        module_id="continuous_flow",
        operations=(
            OperationSpec(
                Operation(
                    "set_flow_stream",
                    "Finite-reservoir continuous streams",
                    ("connection", "flow_rate_mL_min"),
                    ("not_terminated",),
                ),
                "continuous_flow",
                "continuous_streams",
                "apply",
                kind="domain",
                fields=(
                    ActionFieldSpec("connection", "categorical", None, tuple(range(64))),
                    ActionFieldSpec(
                        "flow_rate_mL_min", "mL/min", (0.0, 20.0), (), runtime_bounds=(0.0, 20.0)
                    ),
                ),
                affected_ledgers=(
                    "process",
                    "inactive_vessels",
                    "samples",
                    "equipment",
                    "species",
                    "phases",
                    "vessels",
                    "thermal",
                ),
            ),
            OperationSpec(
                Operation(
                    "advance_flow",
                    "Finite-reservoir continuous streams",
                    ("duration_s", "target_temperature_K", "current_mA"),
                    ("not_terminated",),
                ),
                "continuous_flow",
                "continuous_streams",
                "apply",
                kind="domain",
                fields=(
                    ActionFieldSpec(
                        "duration_s", "s", (1.0, 14400.0), (), runtime_bounds=(1.0, 14400.0)
                    ),
                    ActionFieldSpec(
                        "target_temperature_K",
                        "K",
                        (250.0, 430.0),
                        (),
                        runtime_bounds=(250.0, 430.0),
                    ),
                    ActionFieldSpec(
                        "current_mA", "mA", (0.0, 500.0), (), runtime_bounds=(0.0, 500.0)
                    ),
                ),
                affected_ledgers=(
                    "process",
                    "inactive_vessels",
                    "samples",
                    "equipment",
                    "species",
                    "phases",
                    "vessels",
                    "thermal",
                ),
            ),
            OperationSpec(
                Operation(
                    "set_flow_rate",
                    "Set requested flow; installed apparatus determines actual flow and residence",
                    ("flow_rate_mL_min",),
                    ("not_terminated",),
                ),
                "continuous_flow",
                "continuous_flow",
                "set_flow_rate",
                kind="domain",
                fields=(
                    ActionFieldSpec(
                        "flow_rate_mL_min", "mL/min", (0.0, 20.0), (), runtime_bounds=(0.0, 20.0)
                    ),
                    ActionFieldSpec("residence_time_s", "s", (1.0, 7200.0), (), required=False),
                    ActionFieldSpec("flow_path", "categorical", choices=("process", "rinse"),
                                    required=False),
                ),
                affected_ledgers=("process", "equipment"),
            ),
            OperationSpec(
                Operation(
                    "run_flow",
                    "Advance the installed flow apparatus",
                    ("duration_s",),
                    ("has_volume", "has_material", "run_flow_requires_flow_setup"),
                ),
                "continuous_flow",
                "continuous_flow",
                "run_flow",
                kind="domain",
                fields=(
                    ActionFieldSpec(
                        "target_temperature_K",
                        "K",
                        (298.15, 430.0),
                        (),
                        runtime_bounds=(298.15, 430.0),
                        required=False,
                    ),
                    ActionFieldSpec(
                        "duration_s", "s", (1.0, 14400.0), (), runtime_bounds=(1.0, 14400.0)
                    ),
                ),
                affected_ledgers=("process", "species", "phases", "thermal", "vessels"),
            ),
        ),
        services=(
            ServiceFactorySpec(
                "continuous_flow",
                "flow",
                factories.continuous_flow,
                "ChemWorldFlowServices",
                ("reaction_thermal",),
            ),
            ServiceFactorySpec(
                "continuous_streams",
                "continuous_streams",
                factories.continuous_streams,
                "ContinuousStreamServices",
                ("reaction_thermal",),
            ),
        ),
        components=(
            (
                "continuous_flow",
                ComponentCapability(
                    operations=("set_flow_rate", "run_flow", "set_flow_stream", "advance_flow"),
                    template="flow-reaction-optimization",
                    priority=60,
                    minimum_operations=6,
                    minimum_time_s=1.0,
                    interfaces=("flow_rate", "residence_time", "temperature", "material"),
                    dependencies=("reaction", "thermal"),
                    state_owners=("flow_residence_state",),
                    required_all=("advance_flow", "run_flow", "set_flow_rate", "set_flow_stream"),
                    parameters=(
                        (
                            "flow_rate_range_mL_min",
                            ParameterRule(
                                value_kind="range", unit="mL/min", minimum=0.01, maximum=20.0
                            ),
                        ),
                        (
                            "residence_time_range_s",
                            ParameterRule(
                                value_kind="range", unit="s", minimum=1.0, maximum=7200.0
                            ),
                        ),
                        (
                            "temperature_range_K",
                            ParameterRule(
                                value_kind="range", unit="K", minimum=298.15, maximum=430.0
                            ),
                        ),
                    ),
                ),
            ),
        ),
        dependencies=("thermal",),
    ),
    DomainModuleSpec(
        module_id="electrochemistry",
        operations=(
            OperationSpec(
                Operation(
                    "set_potential",
                    "Configure electrochemical cell",
                    ("potential_V", "current_mA", "electrolyte_profile"),
                    ("has_volume", "has_material"),
                ),
                "electrochemistry",
                "electrochemistry",
                "set_potential",
                kind="domain",
                fields=(
                    ActionFieldSpec("potential_V", "V", (-3.0, 3.0), ()),
                    ActionFieldSpec(
                        "current_mA", "mA", (0.001, 500.0), (), runtime_bounds=(0.001, 500.0)
                    ),
                    ActionFieldSpec("electrolyte_profile", "unitless", None, tuple(range(4))),
                ),
                affected_ledgers=("process", "equipment"),
            ),
            OperationSpec(
                Operation(
                    "electrolyze",
                    "Run electrolysis",
                    ("duration_s",),
                    ("has_volume", "has_material", "electrolyze_requires_potential"),
                ),
                "electrochemistry",
                "electrochemistry",
                "electrolyze",
                kind="domain",
                fields=(
                    ActionFieldSpec(
                        "duration_s", "s", (1.0, 14400.0), (), runtime_bounds=(1.0, 14400.0)
                    ),
                ),
                affected_ledgers=("process", "species", "phases", "equipment"),
            ),
        ),
        services=(
            ServiceFactorySpec(
                "electrochemistry",
                "electrochemical",
                factories.electrochemistry,
                "ChemWorldElectrochemicalServices",
                (),
            ),
        ),
        components=(
            (
                "electrochemistry",
                ComponentCapability(
                    operations=("set_potential", "electrolyze"),
                    template="electrochemical-conversion",
                    priority=70,
                    minimum_operations=7,
                    minimum_measurements=1,
                    minimum_time_s=1.0,
                    interfaces=("potential", "current", "charge", "energy", "material"),
                    dependencies=("reaction",),
                    state_owners=("electrical_state",),
                    required_all=("electrolyze", "set_potential"),
                    parameters=(
                        (
                            "potential_range_V",
                            ParameterRule(value_kind="range", unit="V", minimum=-2.5, maximum=2.5),
                        ),
                        (
                            "current_range_mA",
                            ParameterRule(
                                value_kind="range", unit="mA", minimum=0.001, maximum=500.0
                            ),
                        ),
                        (
                            "duration_range_s",
                            ParameterRule(
                                value_kind="range", unit="s", minimum=1.0, maximum=14400.0
                            ),
                        ),
                    ),
                ),
            ),
        ),
    ),
    DomainModuleSpec(
        module_id="observation",
        operations=(
            OperationSpec(
                Operation(
                    "configure_instrument",
                    "Configure finite spectral acquisition",
                    ("instrument", "scan_count", "resolution_factor", "dilution_factor"),
                    ("not_terminated",),
                ),
                "observation",
                "instrument_cost",
                "configure_instrument",
                kind="primitive",
                fields=(
                    ActionFieldSpec(
                        "instrument",
                        "categorical",
                        None,
                        (
                            "hplc",
                            "gc",
                            "uvvis",
                            "ph_meter",
                            "final_assay",
                            "nmr",
                            "ir",
                            "ms",
                            "particle_size",
                        ),
                    ),
                    ActionFieldSpec("scan_count", "count", (1.0, 128.0), ()),
                    ActionFieldSpec("resolution_factor", "dimensionless", (0.5, 8.0), ()),
                    ActionFieldSpec("dilution_factor", "dimensionless", (1.0, 1000.0), ()),
                ),
                function=factories.configure_instrument,
                affected_ledgers=("process", "equipment"),
            ),
            OperationSpec(
                Operation(
                    "measure", "Measure", ("instrument",), ("has_volume", "instrument_specific")
                ),
                "observation",
                "instrument_cost",
                "apply_measurement_cost",
                kind="primitive",
                fields=(
                    ActionFieldSpec(
                        "instrument",
                        "categorical",
                        None,
                        (
                            "hplc",
                            "gc",
                            "uvvis",
                            "ph_meter",
                            "final_assay",
                            "nmr",
                            "ir",
                            "ms",
                            "particle_size",
                        ),
                    ),
                ),
                affected_ledgers=("process", "equipment", "observation"),
            ),
        ),
        services=(
            ServiceFactorySpec(
                "instrument_cost",
                "instrument_cost",
                factories.instrument_cost,
                "ChemWorldInstrumentCostServices",
                ("reaction_thermal",),
            ),
        ),
        components=(
            (
                "observation",
                ComponentCapability(
                    operations=("measure",),
                    outputs=(),
                    interfaces=("public_measurement", "missingness", "cost", "latency", "sample"),
                    state_owners=("public_measurement",),
                    required_any=("measure", "queue_lab_measure"),
                    parameters=(
                        (
                            "instruments",
                            ParameterRule(
                                value_kind="instrument_ids",
                            ),
                        ),
                    ),
                ),
            ),
        ),
        instruments=_instruments(),
    ),
    DomainModuleSpec(
        module_id="phase",
        operations=(),
        services=(),
        components=(
            (
                "phase",
                ComponentCapability(
                    operations=("add_phase",),
                    template="equilibrium-characterization",
                    priority=10,
                    interfaces=("phase", "volume", "composition", "state_identity"),
                    state_owners=("phase_transition",),
                    parameters=(
                        (
                            "phases",
                            ParameterRule(
                                value_kind="choices",
                                choices=("reactor_liquid", "aqueous", "organic", "solid"),
                            ),
                        ),
                    ),
                ),
            ),
        ),
    ),
)
