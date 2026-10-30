"""Built-in declarations: one owner for every operation and service factory."""

from chemworld.foundation.ontology import Operation
from chemworld.runtime import service_factories as factories
from chemworld.world import builtin_physics_models as physics_models
from chemworld.world.domain_modules import (
    ActionFieldSpec,
    AdvanceStrategySpec,
    ComponentCapability,
    DeclaredParameterRule,
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
    from chemworld.world.chemostatted_catalysis_instruments import (
        chemostatted_coverage_instrument_spec,
    )
    from chemworld.world.chromatography_instruments import chromatography_instrument_specs
    from chemworld.world.electrode_instruments import electrode_instrument_specs
    from chemworld.world.flash_instruments import flash_instrument_specs
    from chemworld.world.gas_instruments import gas_instrument_specs
    from chemworld.world.membrane_instruments import membrane_instrument_spec
    from chemworld.world.sbml_instruments import formal_instrument_spec
    from chemworld.world.solid_instruments import calphad_instrument_spec, solid_instrument_spec
    from chemworld.world.stochastic_sde_instruments import (
        sde_aliquot_instrument_spec,
        sde_instrument_spec,
    )
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
        + (chemostatted_coverage_instrument_spec(),)
        + chromatography_instrument_specs()
        + biochemical_instrument_specs()
        + (sde_instrument_spec(), sde_aliquot_instrument_spec())
        + (transport_instrument_spec(),)
        + aqueous_instrument_specs()
        + (membrane_instrument_spec(),)
        + (formal_instrument_spec(),)
        + (solid_instrument_spec(), calphad_instrument_spec())
    )


def _named_electrical_fields():
    return (
        ActionFieldSpec("control_mode", choices=("current", "voltage", "power", "resistance",
            "differential_power", "differential_resistance", "cccv", "open_circuit"),
            required=False, choice_provider=factories.electrical_control_choices),
        ActionFieldSpec("power_W", "W", (-100., 100.), required=False),
        ActionFieldSpec("resistance_Ohm", "Ohm", (0., 1e6), required=False),
        ActionFieldSpec("current_limit_A", "A", (-10., 10.), required=False),
        ActionFieldSpec("branch_lower_A", "A", (-10., 10.), required=False),
        ActionFieldSpec("branch_upper_A", "A", (-10., 10.), required=False),
    )


BUILTIN_MODULES = (
    DomainModuleSpec(
        module_id="molecular_process",
        scope_notes=("Declared external source through homogeneous serial units or one recycle "
                     "loop into one finite receiver; wait and snapshots; local clock only.",),
        components=(("molecular_process", ComponentCapability(
            operations=("wait",), interfaces=("material", "temperature"),
            state_owners=("phases",),
        )),),
        models=(physics_models.MOLECULAR_PROCESS,),
        advance_strategies=(AdvanceStrategySpec(
            "molecular-process-receiver", factories.molecular_process,
        ),),
        dependencies=("thermal",),
    ),
    DomainModuleSpec(
        module_id="lab_schedule",
        scope_notes=("Single process; finite independent closed systems with explicitly declared "
                     "shared-clock strategies and atomic finite pump/membrane paths; "
                     "explicit finite heater/analyzer identities with FIFO occupancy; "
                     "no same-vessel hybrid or general loops.",),
        operations=(
            OperationSpec(
                Operation("queue_lab_heat", "Queue the shared heater",
                          ("duration_s", "target_temperature_K"),
                          ("not_terminated", "lab_device_selected")),
                "lab_schedule", "lab_schedule", "queue",
                fields=(ActionFieldSpec("duration_s", "s", (1.0, 14400.0)),
                        ActionFieldSpec("target_temperature_K", "K", (1.0, 3000.0)),
                        ActionFieldSpec("device", required=False,
                                        choice_provider=factories.lab_heat_choices)),
                precondition_checks=(("lab_device_selected", factories.lab_heat_selected),),
                affected_ledgers=("equipment", "process"),
            ),
            OperationSpec(
                Operation("queue_lab_measure", "Queue the shared offline analyzer",
                          (), ("not_terminated", "lab_device_selected")),
                "lab_schedule", "lab_schedule", "queue", affected_ledgers=("equipment", "process"),
                fields=(ActionFieldSpec("device", required=False,
                                        choice_provider=factories.lab_analyzer_choices),),
                precondition_checks=(("lab_device_selected", factories.lab_analyzer_selected),),
            ),
        ),
        services=(ServiceFactorySpec(
            "lab_schedule", "lab_schedule", factories.lab_schedule,
            "LabScheduleServices", ("reaction_thermal", "instrument_cost"),
        ),),
        components=(("lab_schedule", ComponentCapability(
            operations=("queue_lab_heat", "queue_lab_measure", "wait"),
            parameters=(("duration_range_s", ParameterRule(
                value_kind="range", unit="s", minimum=1e-6, maximum=14400.0)),),
            interfaces=("time",), state_owners=("lab_clock",),
        )),),
        dependencies=("thermal", "observation"),
    ),
    DomainModuleSpec(
        module_id="solid_solution",
        scope_notes=("Finite binary equilibrium, selected native TDB/CEF phase, or declared "
                     "multiphase CEF PH equilibrium with latent heat and finite thermal device; "
                     "finite phase and tangent search, without global uniqueness claims. "
                     "Fixed-T/P neutral vacancy-free ternary CEF sections retain requested "
                     "composition nodes, tie lines/triangles and failed points; read-only "
                     "world queries solve the exact current elemental composition, "
                     "without interpolating continuous boundaries or replacing live inventory.",),
        components=(("solid_solution", ComponentCapability(
            operations=("heat", "wait", "pause_control", "resume_control"),
            interfaces=("material", "temperature"), state_owners=("solid_phase_equilibrium",),
        )), ("calphad_phase", ComponentCapability(
            operations=("heat", "wait", "pause_control", "resume_control"),
            interfaces=("material", "temperature"), state_owners=("solid_phase_equilibrium",),
        ))),
        models=(physics_models.CALPHAD_MATERIAL,),
        advance_strategies=(AdvanceStrategySpec(
            "binary-solid-equilibrium", factories.solid_solution, shared_clock=True,
        ),),
        dependencies=("thermal",),
    ),
    DomainModuleSpec(
        module_id="mineral_kinetics",
        scope_notes=("Finite calcite dissolution/precipitation with fast aqueous equilibrium; "
                     "fixed 298.15 K; or declared database minerals at fixed TP with "
                     "fast aqueous reactions, finite seeded surface and same-reference heat/work. "
                     "Database mode currently permits wait only; no mineral nucleation.",),
        components=(("mineral_kinetics", ComponentCapability(
            operations=("add_component", "wait", "sample", "route_material"),
            interfaces=("material", "aqueous_equilibrium"),
            state_owners=("phases", "calcite_surface_reference",
                          "declared_mineral_surface_reference"),
            outputs=("aqueous", "calcite"), input_ports=("aqueous", "calcite"),
        )),),
        advance_strategies=(AdvanceStrategySpec(
            "finite-calcite-kinetics", factories.mineral_kinetics,
        ),),
        dependencies=("aqueous_mineral",),
    ),
    DomainModuleSpec(
        module_id="sbml",
        scope_notes=("Formal 3D SBML quantity coordinates and supported transverse events; "
                     "no declared chemical formula, carrier mass or heat model. Explicit "
                     "integer direct/next-reaction/tau modes use elementary source laws, "
                     "fixed source geometry and the original formal state/clock; source "
                     "rules, events and constraints are not coupled in integer mode. Read-only "
                     "finite-time perturbation growth for smooth autonomous reaction ODEs "
                     "uses an explicit mol metric and the original stoichiometric subspace; "
                     "periodic QR adds finite-duration Lyapunov estimates, while trajectory "
                     "spectra distinguish decay, growth, oscillation and resolved time scales; "
                     "no event/rate-rule tangents, ILDM/CSP reduction or asymptotic chaos claim. "
                     "Declared source controls and readouts support finite-start constrained "
                     "steady or finite-horizon optimization from the current inventory; "
                     "candidate states never replace live state and no global optimum is claimed. "
                     "Read-only "
                     "single-parameter steady-branch continuation holds the current "
                     "conservation class, uses declared amount/parameter scales, and "
                     "retains stable/unstable segments, fold brackets and failed steps; "
                     "no all-branch, branch-switching or periodic-orbit qualification.",),
        components=(("sbml_model", ComponentCapability(
            operations=("wait", "set_model_parameters"), minimum_operations=1,
            interfaces=("formal_quantities",),
            state_owners=("formal_model_coordinates_and_queue",), outputs=(), input_ports=(),
        )),),
        models=(physics_models.SBML,),
        operations=(OperationSpec(
            Operation("set_model_parameters", "Set declared model inputs", ("values",),
                      ("not_terminated",)),
            "sbml", "formal_models", "apply_parameters",
            fields=(ActionFieldSpec("values", "per_parameter_source_units",
                                    value_kind="number_map"),),
            affected_ledgers=("process", "equipment"),
        ),),
        services=(ServiceFactorySpec("formal_models", "formal_models", factories.formal_models,
            "SBMLAdvanceStrategy", ("reaction_thermal",)),),
        advance_strategies=(AdvanceStrategySpec(
            "formal-sbml-shared", factories.sbml, service_attribute="formal_models",
            shared_clock=True,
        ),),
        dependencies=("thermal", "observation"),
    ),
    DomainModuleSpec(
        module_id="stochastic",
        scope_notes=("Integer Gillespie direct, Gibson-Bruck next-reaction, or explicitly "
                     "approximate adaptive Poisson tau-leaping with critical-channel/SSA "
                     "fallback and persistent step-held pending batches; fixed volume and "
                     "temperature, declared mass action, one inventory and process RNG. "
                     "Fixed-partition ODE/SSA uses integrated hazards and explicit DOP853 "
                     "or RK45 integration; an explicit "
                     "stoichiometric-jump mode permits real molecular transfer between "
                     "integer and continuous stocks with resolved molecular quanta and "
                     "internal atom-transfer receipts. An explicit adaptive hybrid variant "
                     "uses lossless real mol, hysteretic classification, RK4 or LSODA drift "
                     "and persistent next-reaction targets with step-frozen propensities. "
                     "Its fixed-volume world supports instantaneous solute dosing and "
                     "nonconsuming mol measurement; consuming hybrid sampling and spatial "
                     "kMC remain unsupported. An explicitly declared continuous reaction "
                     "SDE uses RI5 with legacy linear or explicit fractional output, or "
                     "requested-endpoint truncation before bounded feasible-side physical "
                     "acceptance. Truncated output has no lookahead and wait splits can "
                     "change its numerical mesh. "
                     "Declared bounded inputs and representative offline aliquots invalidate "
                     "lookahead while retaining the process stream; an explicit volume floor "
                     "permits sampling boundaries between fixed-volume numerical intervals.",),
        components=(("stochastic", ComponentCapability(
            operations=("wait", "set_sde_parameters"), interfaces=("material",),
            minimum_operations=1,
            state_owners=("phases", "stochastic_continuation"),
        )),),
        advance_strategies=(AdvanceStrategySpec(
            # Stable dispatcher key; the physical declaration selects the algorithm.
            "gillespie-direct", factories.stochastic, shared_clock=True,
        ),),
        models=(physics_models.STOCHASTIC_HYBRID, physics_models.STOCHASTIC_SDE),
        operations=(OperationSpec(
            Operation('set_sde_parameters', 'Set declared SDE inputs', ('values',),
                      ('not_terminated',)),
            'stochastic', 'stochastic_controls', 'apply_parameters',
            fields=(ActionFieldSpec('values', 'per_input_coherent_SI_or_dimensionless',
                                    value_kind='number_map'),),
            affected_ledgers=('process',),
        ),),
        services=(ServiceFactorySpec('stochastic_controls', 'stochastic_controls',
            factories.stochastic_controls, 'SDEStochasticRuntime', ('reaction_thermal',)),),
        dependencies=("thermal",),
    ),
    DomainModuleSpec(
        module_id="ro_membrane",
        scope_notes=("Finite ideal dilute NaCl RO at 298.15 K with deposited foulant/rinse, "
                     "or declared binary/multisolute axial RO with real ion inventories, "
                     "neutral transport channels, pressure loss and three finite tanks; "
                     "explicit common-G aqueous thermodynamics supports isothermal "
                     "pressure-work recovery, while MCAS ideal-volume and empirical "
                     "Sharqawy/Nayar seawater properties use independent H/V/osmotic laws "
                     "without reversible Gibbs-work recovery; seawater TDS remains a "
                     "conserved kg quantity alongside molecular water; "
                     "or ideal dilute multi-ion Donnan/steric/Born NF with a resolved pore "
                     "and a mixed-feed 0D module; "
                     "or a periodic internal AEM/CEM electrodialysis pair with two finite "
                     "liquids, signed ion/water transfer and bounded CC/CV control; "
                     "or explicit passive reverse electrodialysis with open/short circuit, "
                     "finite resistive load, signed electrical output and isothermal bath heat; "
                     "or two-tank direct-contact Knudsen membrane distillation with finite "
                     "heat/cooling and signed water/energy transfer; model-specific "
                     "T/P/salinity bounds and operations, no membrane holdup or full WaterTAP "
                     "flowsheet.",),
        components=(("ro_membrane", ComponentCapability(
            operations=("set_flow_rate", "run_flow", "heat", "wait", "pause_control",
                        "resume_control", "configure_electrical_control"),
            interfaces=("material", "flow_rate", "temperature"),
            state_owners=("liquid_transport_enthalpy_v1", "membrane_bound_foulant_v1",
                          "seawater_tds_mass_v1"),
            requires_physics=True,
        )),),
        advance_strategies=(AdvanceStrategySpec(
            "finite-dilute-ro", factories.ro_membrane, shared_clock=True,
        ),),
        dependencies=("continuous_flow", "electrical_control"),
    ),
    DomainModuleSpec(
        module_id="finite_transport",
        scope_notes=("Finite pump, pipe, tanks and authored heat exchange; bounded working point; "
                     "or a declared molecular process graph with multiple algebraic tears, "
                     "selective flash/cascade, same-reference heat exchange and shaft recovery, "
                     "finite homogeneous tank n/H, separate heat/work ports and earliest events; "
                     "declared preloaded tanks and source molar-ratio feed control; "
                     "or four finite tanks and two pumps exchanging countercurrent heat at "
                     "declared UA with independent circuit flows; "
                     "or declared isothermal zero-order liquid treatment chains with finite "
                     "product/concentrate stocks and receipt-based engineering economics; "
                     "or finite pump-fed neutral reactive holdup with ideal overflow to a "
                     "finite product tank, volume-compatible chemistry and one shared bath; "
                     "or the complete empirical ASM1 13-component/eight-process single "
                     "CSTR or five-reactor recycle train, finite original water/quality "
                     "stocks, oxygen supply/offgas and sludge; source-bound COD/N/equivalent "
                     "budgets, not full molecular CHON, charge or biochemical reaction heat; "
                     "or finite neutral aqueous mixed-contact adsorption and rinse with "
                     "persistent competitive loading, five liquid stocks and one shared bath; "
                     "or finite ideal-dilute Na/Ca resin exchange with counterion release, "
                     "salt regeneration and rinse across six liquid stocks, weighted fixed "
                     "resin charge and one shared bath; "
                     "or finite neutral suspensions with a declared jar-test endpoint, "
                     "explicit dispersed/floc particle phases, empirical clarification "
                     "into actual clear-water/wet-sludge tanks, and mixing/pump work; "
                     "no additive-driven coagulation prediction or settling dynamics; "
                     "the adsorption-world feed also supports read-only single-solute fresh "
                     "GAC design and breakthrough queries at fixed influent, using actual "
                     "pump flow and feed volume above pickup; not remaining-life prediction "
                     "for existing LDF media. These are declared finite models, "
                     "not arbitrary multiphase process-network simulation.",),
        components=(("finite_transport", ComponentCapability(
            operations=("set_flow_rate", "run_flow", "wait"),
            parameters=(("duration_range_s", DeclaredParameterRule(
                value_kind="range", unit="s", minimum=1e-6, maximum=14400.0)),),
            interfaces=("material", "flow_rate", "temperature"),
            state_owners=("liquid_transport_enthalpy_v1", "finite_suspension_enthalpy"),
        )),),
        advance_strategies=(AdvanceStrategySpec(
            "finite-pump-transport", factories.finite_transport, shared_clock=True,
        ),),
        dependencies=("continuous_flow",),
    ),
    DomainModuleSpec(
        module_id="gas_reactor_network",
        scope_notes=("Finite isothermal ideal-gas reactors, storage and collection nodes; "
                     "declared molar/mass flow and one-way pressure valves; shared time, "
                     "material transfers and bath heat; or explicit per-node energy with "
                     "finite vessel heat capacities, source-temperature enthalpy transport, "
                     "environment/neighbor heat exchange and one queued anchor heater. "
                     "Complete imported ideal surfaces can share the same gas network, owned "
                     "site stocks and calorics. Also sealed or finite-feed/collector isothermal "
                     "low-Mach planar cells with common grid pressure, independent tank pressures, "
                     "continuity-derived bulk flow and whole-mixture diffusion; prescribed "
                     "boundary molar flow has no momentum or compressor model. "
                     "Explicit nonisothermal cells jointly close continuity, local energy "
                     "and common pressure with upwind enthalpy, diffusion enthalpy and "
                     "Fourier heat; finite tanks, node heat exchange and the anchor heater "
                     "retain original inventories and one shared clock. No Soret or "
                     "momentum equation is implied. "
                     "Alternatively one constant-load finite-stroke ideal-gas piston with "
                     "actual EOS volume and separately booked pressure work, without surfaces "
                     "or spatial cells. Constant-volume energy networks can instead declare "
                     "one finite-heat-capacity wall with its own persistent temperature, "
                     "gas contact conduction and grey radiation/conduction to an environment. "
                     "No wall/piston, wall/surface or wall/spatial coupling; "
                     "alternatively two nonreactive constant-volume energy nodes connected "
                     "by one zero-holdup compressor or expander, using actual endpoint "
                     "pressures, same-source PS/efficiency/PH calorics, finite shaft power "
                     "and separate inlet/outlet enthalpy with signed work; "
                     "no shaft/piston, shaft/wall, shaft/surface or shaft/spatial coupling; "
                     "no piston inertia, spatial flame field or arbitrary "
                     "member heat queues. Sourced finite harmonic state counting can compile "
                     "one-well RRKM channels into the original energy-grain populations and "
                     "bath collisions, with explicit ZPE and quantization; no rotor, tunneling "
                     "or general pressure-rate reduction. Read-only single-well QSD rates, "
                     "product branches and spectral gaps use current original grains and "
                     "actual bath inventory or explicit fixed T/bath-pressure grids; "
                     "no dynamic reduction, total-pressure PLog or new heat/inventory model. "
                     "Read-only trajectory sensitivities include signed species enthalpy "
                     "parameters through the original H/G/U and reverse rates; fixed-geometry "
                     "gas/surface steady queries distinguish a complete finite network from "
                     "explicitly held nonreacting boundary nodes, retain original residuals "
                     "and solver work, and never commit the hypothetical result.",),
        components=(("gas_reactor_network", ComponentCapability(
            operations=("wait",), minimum_operations=1,
            interfaces=("material", "temperature"), state_owners=("gas_network_accounting_v1",),
        )),),
        advance_strategies=(AdvanceStrategySpec(
            "finite-isothermal-gas-network", factories.gas_reactor_network, shared_clock=True,
        ),),
        dependencies=("thermal",),
    ),
    DomainModuleSpec(
        module_id="aqueous_reactive_transport",
        scope_notes=("Sealed one-dimensional ideal-dilute aqueous transport at fixed T/P; "
                     "finite stationary water, zero solute partial volumes, zero-current "
                     "electrodiffusion, fast equilibrium and declared slow neutral conversion. "
                     "Also finite feed/collector throughflow with conservative upwind advection, "
                     "common mechanical dispersion and actual water/solute/enthalpy transfer; "
                     "prescribed kinematic flow at common T/P, without a pump model. "
                     "Strictly positive ions; no pressure flow, mineral or external circuit.",),
        components=(("aqueous_reactive_transport", ComponentCapability(
            operations=("wait",), minimum_operations=1,
            interfaces=("material", "aqueous_equilibrium"), state_owners=("phases",),
        )),),
        advance_strategies=(AdvanceStrategySpec(
            "finite-aqueous-reactive-transport", factories.aqueous_reactive_transport,
            shared_clock=True,
        ),),
        dependencies=("thermal",),
    ),
    DomainModuleSpec(
        module_id="aqueous_gas_transfer",
        scope_notes=("Finite ideal-gas headspace and ideal aqueous liquid in one rigid "
                     "isothermal vessel; reversible sourced gas/liquid transfer and fast "
                     "aqueous reactions, actual gas pressure and same-reference internal-energy "
                     "bath duty; optional seeded pure-mineral kinetics with actual solid "
                     "volume displacing the headspace. Nonvolatile solvent, fixed liquid volume "
                     "and zero solute partial volumes; no infinite gas source, pressure flow "
                     "or spontaneous mineral nucleation.",),
        components=(("aqueous_gas_transfer", ComponentCapability(
            operations=("wait",), minimum_operations=1,
            interfaces=("material", "aqueous_equilibrium"), state_owners=("phases",),
        )),),
        advance_strategies=(AdvanceStrategySpec(
            "finite-aqueous-gas-transfer", factories.aqueous_gas_transfer,
        ),),
        dependencies=("thermal",),
    ),
    DomainModuleSpec(
        module_id="aqueous_mineral",
        scope_notes=("Authored finite aqueous acid/base/complexation and mineral equilibrium; "
                     "explicit thermodynamic data and ionic-strength domain, "
                     "not a geochemical database. Also finite same-T/P solution mixing, "
                     "nonideal equilibrium precipitation and dilution with liquid-only packets, "
                     "original solid inventory, same-reference enthalpy and pressure-volume work; "
                     "instantaneous equilibrium, not nucleation kinetics.",),
        components=(("aqueous_mineral", ComponentCapability(
            operations=("add_component", "wait", "sample"),
            interfaces=("material", "aqueous_equilibrium"), state_owners=("phases",),
        )),),
        advance_strategies=(AdvanceStrategySpec(
            "finite-carbonate-calcite-equilibrium", factories.aqueous_mineral,
        ),),
        dependencies=("thermal", "core"),
    ),
    DomainModuleSpec(
        module_id="programmed_chromatography",
        scope_notes=("Authored peak/retention response, finite sampling and persistent carryover; "
                     "not a fitted universal chromatographic transport model.",),
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
        models=(physics_models.SORBENT,),
        scope_notes=("Competitive finite sorbent kinetics, owned carrier and finite wash; "
                     "explicit condensed calorics and temperature domain.",),
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
                    *_named_electrical_fields(),
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
                    *_named_electrical_fields(),
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
        advance_strategies=(AdvanceStrategySpec(
            "finite-transient-electrode", factories.transient_electrode,
        ),),
        dependencies=("thermal",),
    ),
    DomainModuleSpec(
        module_id="battery_spm",
        scope_notes=("Synthetic SPM with calorics/SEI, or explicit isothermal SPMe-v1 "
                     "with conservative electrolyte transport; shared native cycling. "
                     "Explicit isothermal DFN has coupled spatial reactions and a fixed synthetic "
                     "qualification at 24 through-plane cells per region / 64 radial cells. "
                     "Explicit isothermal Newman-Tobias uses two average lithium inventories "
                     "and spatial porous electrical fields, without radial diffusion or dynamic "
                     "salt; its fixed CV grid-inventory qualification remains open. "
                     "Development MPM adds shared-potential size populations with independent "
                     "inventories; independent MPM scientific qualification remains pending. "
                     "Isothermal single-material SPM/MPM may declare current-sigmoid OCP "
                     "or one persistent Axen hysteresis coordinate per electrode; "
                     "no hysteresis thermal/SEI coupling or equilibrium-SOH interpretation. "
                     "Through-thickness thermal SPM uses three finite layers and two collectors, "
                     "constant net OCV and local BV temperature feedback with ambient cooling; "
                     "no arbitrary external thermal device or general material thermal claim. "
                     "Isothermal MSMR-SPM uses one or two explicit reactions per electrode "
                     "with potential-driven radial diffusion and shared branch kinetics; "
                     "no MSMR thermal, EIS or automatic equilibrium-SOH claim. "
                     "Isothermal composite DFN resolves two negative and one positive material "
                     "phase at each thickness cell with shared potentials and dynamic salt; "
                     "no composite-DFN thermal, side-reaction, SOH, EIS "
                     "or chemical-energy closure. "
                     "Collector SPM resolves 1+1D finite-conductivity current collectors "
                     "with position-resolved single particles and explicit tab potentials; "
                     "no electrolyte, thermal, side-reaction, PSD, hysteresis, SOH, EIS "
                     "or chemical-energy closure. "
                     "Composite SPM supports two independent negative material phases and one "
                     "positive phase with single-valued monotone OCV, including internal exchange "
                     "at zero terminal current; no thermal/SEI, EIS, SOH or energy closure. "
                     "Positive-working half-cell SPM uses finite lithium metal with explicitly "
                     "fixed geometry; no half-cell thermal/SEI, EIS, SOH or energy closure. "
                     "Coarser meshes and other parameters need convergence checks; "
                     "Closed volume-consistent isothermal lead-acid LOQS uses six real "
                     "chemical inventories and the same runtime; no heat, side reactions, "
                     "EIS or automatic SOH. Synthetic parameters are not wet-lab validation. "
                     "Split OCV-R uses two finite mean Li inventories, separate electrode OCV "
                     "and capacity, and series resistance; no particle diffusion, BV or heat. "
                     "Its EIS and equilibrium SOH reuse the same declared inventory model. "
                     "Versioned Chen2020 inputs retain all94 source fields and expose a checked "
                     "298.15K/fixed-electrolyte single-material SPM projection; nominal capacity "
                     "and occupied cell-envelope volume remain distinct from active inventories. "
                     "Explicit DFN half-cell uses separator/working-electrode dynamic salt, "
                     "finite fixed-geometry lithium metal and local working particles; "
                     "no half-cell DFN thermal, side, EIS or automatic-SOH interpretation. "
                     "New source-parameter and DFN-half-cell independent qualification is pending. "
                     "No PyBaMM runtime backend.",),
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
                    requires_physics=True,
                ),
            ),
            (
                "battery_cell",
                ComponentCapability(
                    operations=(
                        "configure_electrical_control", "queue_electrical_stage", "wait",
                        "pause_control", "resume_control",
                    ),
                    interfaces=("electrical_control",),
                    state_owners=("battery_inventory",),
                    requires_physics=True,
                ),
            ),
        ),
        advance_strategies=(AdvanceStrategySpec(
            "single-particle-battery", factories.battery_spm, shared_clock=True,
        ),),
        dependencies=("electrical_control",),
    ),
    DomainModuleSpec(
        module_id="battery_thevenin",
        scope_notes=("Sourced empirical RC and two-node cell/jig heat model; shared CC/CV "
                     "control and persistent apparatus. No chemical inventory/total energy, "
                     "diffusion element or independent released-model qualification yet.",),
        components=(("battery_thevenin", ComponentCapability(
            operations=("configure_electrical_control", "queue_electrical_stage", "wait"),
            interfaces=("electrical_control",), state_owners=("empirical_circuit_state",),
            input_ports=(), outputs=(),
            requires_physics=True,
        )),),
        advance_strategies=(AdvanceStrategySpec(
            "thevenin-empirical-circuit", factories.battery_thevenin, shared_clock=True,
        ),),
        dependencies=("electrical_control", "lab_schedule"),
    ),
    DomainModuleSpec(
        module_id="vacuum_concentration",
        advance_strategies=(
            AdvanceStrategySpec(
                "finite-vacuum-concentration-enthalpy", factories.concentration,
                service_attribute="concentration",
            ),
        ),
        models=(physics_models.CONCENTRATION,),
        scope_notes=("Finite variable-composition concentration with authored VLE/calorics; "
                     "unsupported kinetic coupling is rejected.",),
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
        advance_strategies=(
            AdvanceStrategySpec(
                "finite-equilibrium-column", factories.dynamic_column,
                service_attribute="column",
            ),
        ),
        models=(physics_models.COLUMN,),
        scope_notes=("Finite tray/receiver inventories, heat and timed cuts; "
                     "authored small dynamic column, not arbitrary plant design.",),
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
        advance_strategies=(
            AdvanceStrategySpec(
                "isothermal-liquid-equilibrium", factories.liquid_equilibrium,
                service_attribute="liquid",
            ),
        ),
        models=(physics_models.LIQUID,),
        scope_notes=("Strict finite liquid phase equilibrium using declared component properties "
                     "and supported composition domain; no silent extrapolation.",),
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
        advance_strategies=(
            AdvanceStrategySpec(
                "constant-pressure-flash-enthalpy", factories.flash_thermal,
                service_attribute="flash",
            ),
        ),
        models=(physics_models.FLASH,),
        scope_notes=("Finite TP/PH/UV closure with explicit phase/caloric models, including "
                     "declared multiphase molecular equilibrium, finite thermal devices, "
                     "constant-pressure boundary work or a fixed-volume energy balance; "
                     "no simultaneous undeclared kinetic process.",),
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
        models=(physics_models.SURFACE,),
        scope_notes=("Finite owned sites; declared single/multisite mean-field mechanisms "
                     "and symmetric interactions; optionally declared constant-activity "
                     "aqueous chemostats and fixed potential versus SHE, with actual liquid "
                     "stocks/ports, electron stoichiometry, original charge/work and bath heat. "
                     "Source electronic-energy scaling and free-energy BEP table fits can "
                     "author the same mechanisms with explicit known-sigma parameter uncertainty. "
                     "No dynamic pH, double layer, full-cell calorimetry or spatial kMC; "
                     "real-catalyst predictive accuracy requires separate validation.",),
        processes=(("surface_kinetics", factories.surface_processes),),
        components=(
            (
                "surface_reaction",
                ComponentCapability(
                    operations=("configure_electrical_control",),
                    processes=("surface_kinetics",),
                    dependencies=("thermal",),
                    state_owners=("surface_kinetics", "chemostatted_catalysis_electrical",
                                  "chemostatted_catalysis_exchange"),
                    interfaces=("material", "temperature", "surface"),
                    parameters=(("surface_id", ParameterRule("string")),),
                ),
            ),
        ),
        dependencies=("thermal",),
    ),
    DomainModuleSpec(
        module_id="core",
        scope_notes=("Declared bulk reaction networks use the selected caloric model. "
                     "An explicit homogeneous Peng-Robinson phase supports elementary "
                     "Arrhenius activity-concentration kinetics, same-source standard "
                     "thermochemistry and finite isochoric energy/pressure evolution; "
                     "local single-phase stability only, no general nonideal mechanism "
                     "family or multiphase reactive flash claim.",),
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
                "sample_storage",
                ComponentCapability(
                    operations=("create_container", "transfer_material"),
                    interfaces=("material", "sample_identity"), state_owners=("samples",),
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
        advance_strategies=(
            AdvanceStrategySpec(
                "owned-inventory-temperature", factories.coupled_thermal,
                service_attribute="coupled", shared_clock=True,
            ),
            AdvanceStrategySpec(
                "static-finite-carrier-stock", factories.finite_carrier_supply,
                timed_measurements=False, shared_clock=True,
            ),
            AdvanceStrategySpec("legacy-liquid-enthalpy", factories.legacy_liquid,
                                timed_measurements=False, legacy_input_default=True),
        ),
        models=(physics_models.THERMAL,),
        scope_notes=("Finite-power or explicitly isothermal device; shared energy accounting; "
                     "manual reconnection requires one disabled heater in local-clock mode. "
                     "Explicit finite-jacket P/PI/filtered PID uses actual temperature, "
                     "heater-owned persistent memory and realized-power anti-windup in the "
                     "original thermal/shared-clock advance; no extra controller clock, "
                     "automatic tuning or implicit coupling to other model families.",),
        operations=(
            OperationSpec(
                Operation("reconnect_heater", "Reconnect one disabled heater", ("vessel",),
                          ("not_terminated",)),
                "thermal", "component_network", "reconnect_heater",
                fields=(ActionFieldSpec("vessel", "categorical", None, tuple(range(16))),),
                affected_ledgers=("equipment", "inactive_vessels", "process"),
            ),
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
                    state_owners=("temperature_control", "thermal-feedback-memory-v1"),
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
                                value_kind="range", unit="s", minimum=1e-6, maximum=14400.0
                            ),
                        ),
                    ),
                ),
            ),
            ("heater_connection", ComponentCapability(
                operations=("reconnect_heater",), interfaces=("equipment",),
                state_owners=("equipment",),
            )),
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
        scope_notes=("Authored crystal seeding, cooling and wet-cake operations; also "
                     "declared same-T/P instantaneous aqueous mineral filtration into "
                     "two finite empty receivers, with explicit solid capture and mother-liquid "
                     "retention, actual leaked solids, wet-cake washing and repeat separation. "
                     "Database filtration has no drying, pressure-drop or filtration-rate model.",),
        advance_strategies=(
            AdvanceStrategySpec(
                "finite-crystal-population-temperature", factories.crystal_thermal,
                service_attribute="crystal", shared_clock=True,
            ),
        ),
        models=(physics_models.CRYSTAL,),
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
                "aqueous_filter",
                ComponentCapability(
                    operations=("filter_crystals",),
                    minimum_operations=1,
                    interfaces=("material", "phase"),
                    dependencies=("aqueous_mineral",),
                    outputs=("cake", "filtrate"),
                    input_ports=(),
                ),
            ),
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
                    ActionFieldSpec("flow_path", "categorical",
                                    choices=("process", "rinse", "left", "right"),
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
                    ActionFieldSpec("scan_count", "count", (1.0, 128.0), (), integer=True),
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
