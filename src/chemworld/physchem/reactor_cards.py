"""Reactor model cards for ChemWorld."""

from __future__ import annotations

from chemworld.physchem.maturity import MaturityLevel, ModelCard, ValidationEvidence


def reactor_model_cards() -> tuple[ModelCard, ...]:
    return (
        ModelCard(
            model_id="dynamic_batch_heat_release_jacket_sampling",
            module_id="reactors",
            title="Dynamic Batch And Semibatch Reactor Reference Slice",
            maturity=MaturityLevel.REFERENCE_VALIDATED,
            summary=(
                "Dynamic batch and prescribed-flow semibatch reactor slice with material "
                "balances, thermochemistry-derived reaction heat, wall/jacket "
                "heat transfer, destructive sampling events, and auditable "
                "material/energy ledgers."
            ),
            equations=(
                "dn/dt = S r(n, T)",
                "rhoCp V dT/dt = Q_jacket - Q_loss - sum_i DeltaH_i(T) r_i V",
                "Q_jacket = UA_jacket (T_jacket(t) - T) + Q_fixed",
                "Q_loss = UA_env (T - T_env)",
                "sample event: n_j <- n_j(1 - V_sample/V), V <- V - V_sample",
                "C_j = n_j/V; fixed-liquid P(t) = P_boundary when pressure is declared",
                "semibatch: dn_i/dt = V sum_j nu_ij r_j + F_i,in - q_out n_i/V",
                "semibatch: dV/dt = q_in - q_out",
            ),
            assumptions=(
                "well-mixed liquid-phase batch reactor",
                "constant density heat-capacity basis rhoCp V",
                "sampling removes a representative well-mixed fraction",
                "semibatch withdrawal composition equals instantaneous reactor composition",
                "reaction enthalpy uses supplied NASA7 species thermochemistry when available",
            ),
            validity_limits=(
                (
                    "pressure support is a fixed-liquid boundary only; no gas-headspace "
                    "dynamics, vapor-liquid equilibrium, or wall thermal inertia"
                ),
                "sample events must leave positive reactor volume",
                "prescribed semibatch schedules must leave positive reactor volume",
                "NASA7 reaction enthalpy requires thermochemistry for every reacting species",
            ),
            failure_modes=(
                "negative duration, volume, sample volume, or temperature raises errors",
                "missing species thermochemistry raises a KeyError when NASA7 heat is requested",
                "ODE solver failure raises RuntimeError instead of silently clipping",
                "a semibatch schedule that exhausts volume fails before integration",
            ),
            units={
                "amount": "mol",
                "volume": "L",
                "temperature": "K",
                "heat_capacity_density": "J/(L*K)",
                "heat_duty": "W",
                "energy_ledger": "J",
            },
            reference_reading=(
                (
                    "Cantera: reference_repos/cantera/src/zeroD/Reactor.cpp "
                    "separates species production, wall heat Qdot, inlet/outlet "
                    "enthalpy, and energy-equation terms."
                ),
                (
                    "Cantera: reference_repos/cantera/src/zeroD/"
                    "IdealGasReactor.cpp uses mass*cv*dT/dt as the energy "
                    "equation left-hand side for temperature-state reactors."
                ),
                (
                    "IDAES: modular property packages expose enthalpy/internal "
                    "energy flow and density terms for control-volume energy "
                    "balances."
                ),
            ),
            validation_evidence=(
                ValidationEvidence(
                    evidence_id="dynamic-batch-adiabatic-rise-test",
                    evidence_type="unit_test",
                    description=(
                        "Exothermic reaction with NASA7 enthalpy produces the "
                        "expected adiabatic temperature rise from the integrated "
                        "reaction-heat ledger."
                    ),
                    status="implemented",
                    command_or_path="tests/test_reactor_models.py; tests/test_reactor_reference.py",
                    tolerance="pytest.approx local tolerances",
                ),
                ValidationEvidence(
                    evidence_id="dynamic-batch-sampling-ledger-test",
                    evidence_type="unit_test",
                    description=(
                        "Destructive sampling reduces volume, records material_out, "
                        "and preserves elemental material balance."
                    ),
                    status="implemented",
                    command_or_path="tests/test_reactor_models.py; tests/test_reactor_reference.py",
                    tolerance="material balance error < 1e-8 mol",
                ),
                ValidationEvidence(
                    evidence_id="batch-first-order-and-idempotent-advance-test",
                    evidence_type="analytical_test",
                    description=(
                        "First-order isothermal batch conversion matches exp(-kt), and "
                        "a retried operation id does not advance time or energy twice."
                    ),
                    status="implemented",
                    command_or_path="tests/test_reactor_reference.py",
                    tolerance="2e-6 relative; exact retry state equality",
                ),
                ValidationEvidence(
                    evidence_id="semibatch-feed-withdrawal-ledger-closure",
                    evidence_type="analytical_test",
                    description=(
                        "Matched volumetric feed/withdrawal holds volume while material-in, "
                        "material-out, inventory, and elemental balance close."
                    ),
                    status="implemented",
                    command_or_path="tests/test_reactor_reference.py",
                    tolerance="material balance error < 1e-8 mol",
                ),
                ValidationEvidence(
                    evidence_id="semibatch-volume-exhaustion-failure",
                    evidence_type="failure_injection",
                    description="An exhaustive withdrawal schedule fails before ODE integration.",
                    status="implemented",
                    command_or_path="tests/test_reactor_reference.py",
                    tolerance="deterministic ValueError",
                ),
                ValidationEvidence(
                    evidence_id="formal-runtime-reaction-reactor-integration",
                    evidence_type="end_to_end_runtime_test",
                    description=(
                        "Formal heat/wait operations invoke the validated compiled reaction "
                        "network through DynamicBatchReactorModel, publish solver and "
                        "conservation diagnostics, and roll back failed domains."
                    ),
                    status="implemented",
                    command_or_path="tests/test_reaction_reactor_runtime_integration.py",
                    tolerance=(
                        "material/element/charge <= 1e-8 mol; invariant drift <= 1e-7 mol; "
                        "energy residual <= 1e-5 relative"
                    ),
                ),
            ),
            model_limit_notes=(
                "Current unquenched ChemWorld heat/wait uses Q_jacket = "
                "clip(4*(T_set-T), -70, 90) W; changing call boundaries does not reset its gain.",
                "Low-level prescribed semibatch schedules do not imply a public feed "
                "operation in the ChemWorld event-action contract.",
                (
                    "This slice closes the dynamic batch heat-release and sampling "
                    "kernel, not a full Cantera/IDAES reactor clone."
                ),
                (
                    "Pressure dynamics, gas expansion work, variable Cp mixtures, "
                    "and vapor-liquid phase change remain explicit future slices."
                ),
                "Prescribed semibatch flow is validated; closed-loop level control is not.",
            ),
            intended_use=(
                "reaction calorimetry task design",
                "heat-release-aware campaign simulation",
                "safe-operation benchmark scenarios with explicit material and energy ledgers",
                "semibatch feed-rate, quench, and residence-history reasoning",
                "formal transactional heat/wait runtime advancement",
            ),
        ),
        ModelCard(
            model_id="dynamic_cstr_startup_shutdown",
            module_id="reactors",
            title="Dynamic CSTR Startup, Shutdown, And Residence-Time Slice",
            maturity=MaturityLevel.REFERENCE_VALIDATED,
            summary=(
                "Constant-volume dynamic CSTR with coupled species and energy balances, "
                "explicit residence time, common inlet/outlet flow programs for startup "
                "and shutdown, and auditable material/energy ledgers."
            ),
            equations=(
                "dn_i/dt = V sum_j nu_ij r_j + s(t) F_i,in - s(t) q_out n_i/V",
                "tau = V/q_in",
                "rhoCp V dT/dt = Q_jacket + s(t) rhoCp q_in(T_in-T) - Q_loss - Q_rxn",
                "s(t) = step or linear flow-program interpolation",
            ),
            assumptions=(
                "well-mixed constant-volume liquid CSTR",
                "inlet and outlet share one flow scale so hydraulic volume remains fixed",
                "constant density heat-capacity basis",
                "reaction rates use the declared mechanism and current reactor temperature",
            ),
            validity_limits=(
                "constant volume requires equal nominal inlet and outlet volumetric flow",
                "no pressure controller, vapor holdup, or wall thermal inertia",
                "flow-program scale is finite and nonnegative",
                "steady-state multiplicity is validated by a separate first-order reference slice",
            ),
            failure_modes=(
                "negative or nonfinite flow-program values raise ValueError",
                "unsorted program times raise ValueError",
                "ODE failure raises RuntimeError with solver-policy diagnostics",
            ),
            units={
                "amount": "mol",
                "volume": "L",
                "volumetric_flow": "L/s",
                "residence_time": "s",
                "temperature": "K",
                "heat_duty": "W",
                "energy_ledger": "J",
            },
            reference_reading=(
                "Cantera continuous_reactor.py: stirred-reactor inlet/outlet controller semantics.",
                "IDAES cstr.py: dynamic 0D material, energy, and reaction-extent balances.",
                "Levenspiel, Chemical Reaction Engineering: ideal CSTR transient response.",
            ),
            validation_evidence=(
                ValidationEvidence(
                    evidence_id="cstr-startup-analytical-limit",
                    evidence_type="analytical_test",
                    description=(
                        "No-reaction startup matches the closed-form dilution response using "
                        "the time integral of the linear flow scale."
                    ),
                    status="implemented",
                    command_or_path="tests/test_reactor_models.py; tests/test_reactor_reference.py",
                    tolerance="2e-6 relative and material balance < 1e-8 mol",
                ),
                ValidationEvidence(
                    evidence_id="cstr-shutdown-hold-test",
                    evidence_type="analytical_test",
                    description=(
                        "No-reaction shutdown matches the integrated washout response and "
                        "holds material constant after flow reaches zero."
                    ),
                    status="implemented",
                    command_or_path="tests/test_reactor_models.py",
                    tolerance="2e-6 relative and material balance < 1e-8 mol",
                ),
                ValidationEvidence(
                    evidence_id="cstr-first-order-design-and-residual-gate",
                    evidence_type="analytical_test",
                    description=(
                        "The first-order steady state matches C_Af/(1+k tau), and the "
                        "generic dynamic solve must pass an explicit species-residual gate."
                    ),
                    status="implemented",
                    command_or_path="tests/test_reactor_reference.py",
                    tolerance="2e-6 relative; residual <= 1e-8 mol/s",
                ),
            ),
            model_limit_notes=(
                (
                    "The flow program models controlled constant-volume "
                    "startup/shutdown, not tank filling."
                ),
                (
                    "Hydraulic level control and pressure-controller dynamics "
                    "remain separate equipment slices."
                ),
            ),
            intended_use=(
                "startup and shutdown policy benchmarks",
                "residence-time and thermal-transient reasoning",
                "CSTR controller and safety task design",
            ),
        ),
        ModelCard(
            model_id="pfr_axial_hydraulics_heat_boundary",
            module_id="reactors",
            title="PFR Axial Reaction, Pressure Drop, And Heat-Transfer Slice",
            maturity=MaturityLevel.REFERENCE_VALIDATED,
            summary=(
                "Steady incompressible tubular PFR with residence-time/axial mapping, "
                "mechanism-based reaction integration, Darcy-Weisbach pressure loss, "
                "and a distributed thermal boundary."
            ),
            equations=(
                "z = L tau/tau_res and tau_res = V/q",
                "dC_i/dtau = sum_j nu_ij r_j(C,T)",
                "dP/dz = -f_D rho u^2/(2D)",
                "rhoCp V dT/dtau = UA(T_boundary-T) - Q_loss - Q_rxn",
            ),
            assumptions=(
                "steady one-dimensional plug flow with no axial dispersion",
                "constant incompressible volumetric flow, density, and viscosity",
                "Darcy friction factor uses the declared roughness and Reynolds number",
                "thermal boundary is represented by distributed UA per unit length",
            ),
            validity_limits=(
                "single-phase incompressible flow only",
                "geometry volume must match the declared reactor volume",
                "no compressible acceleration, two-phase multiplier, or catalyst-bed pressure drop",
                "absolute pressure must remain positive over the reactor length",
            ),
            failure_modes=(
                "inconsistent geometry and reactor volume raise ValueError",
                "axial evaluation points outside the tube raise ValueError",
                "nonpositive predicted absolute pressure raises RuntimeError",
            ),
            units={
                "axial_position": "m",
                "pressure": "Pa",
                "pressure_gradient": "Pa/m",
                "volumetric_flow": "L/s",
                "temperature": "K",
                "boundary_ua": "W/(m*K)",
            },
            reference_reading=(
                "Levenspiel, Chemical Reaction Engineering: ideal tubular-reactor design equation.",
                "Darcy-Weisbach equation with laminar/Colebrook friction-factor governance.",
                "IDAES PFR control-volume material and energy balance organization.",
            ),
            validation_evidence=(
                ValidationEvidence(
                    evidence_id="pfr-darcy-pressure-profile",
                    evidence_type="analytical_test",
                    description=(
                        "Incompressible constant-property pressure loss equals the declared "
                        "Darcy gradient times tube length."
                    ),
                    status="implemented",
                    command_or_path="tests/test_reactor_models.py",
                    tolerance="pytest.approx local tolerance",
                ),
                ValidationEvidence(
                    evidence_id="pfr-heat-boundary-analytical-limit",
                    evidence_type="analytical_test",
                    description=(
                        "A no-reaction tube approaches the boundary temperature according to "
                        "the closed-form constant-UA plug-flow solution."
                    ),
                    status="implemented",
                    command_or_path="tests/test_reactor_models.py",
                    tolerance="2e-6 relative",
                ),
            ),
            model_limit_notes=(
                "This closes the single-phase tubular slice, not a compressible or packed-bed PFR.",
                "Two-phase and Ergun pressure-drop models remain separate explicit tasks.",
            ),
            intended_use=(
                "PFR hotspot and residence-time benchmark design",
                "axial thermal and hydraulic constraint reasoning",
                "single-phase tubular reactor validation cases",
            ),
        ),
        ModelCard(
            model_id="cstr_exothermic_multiplicity_reference",
            module_id="reactors",
            title="Exothermic CSTR Multiple-Steady-State Reference Slice",
            maturity=MaturityLevel.REFERENCE_VALIDATED,
            summary=(
                "ChemWorld-owned steady-state CSTR reference problem for an "
                "exothermic first-order reaction. It solves the coupled "
                "steady-state material and energy balances as scalar energy "
                "roots and classifies local stability using the dynamic CSTR "
                "Jacobian."
            ),
            equations=(
                "0 = q(C_Af - C_A) - V k(T) C_A",
                "C_A(T) = C_Af / (1 + k(T) V/q)",
                "0 = (-DeltaH) V k(T) C_A(T) - rhoCp q(T - Tf) - UA(T - Tc)",
                "k(T) = A exp(-Ea/(RT))",
            ),
            assumptions=(
                "well-mixed liquid-phase CSTR",
                "constant volume and density heat capacity",
                "single irreversible A=>P first-order exothermic reaction",
                "heat removal represented by inlet sensible heat and UA coolant term",
            ),
            validity_limits=(
                "validated only for scalar first-order exothermic CSTR multiplicity",
                (
                    "no pressure dynamics, vapor phase, nonideal heat capacity, "
                    "or full plant hydraulics"
                ),
                "temperature roots must be bracketed inside the declared temperature bounds",
            ),
            failure_modes=(
                (
                    "invalid nonpositive flow, volume, heat capacity, or "
                    "Arrhenius parameters raise errors"
                ),
                "endothermic delta_h is rejected for this multiplicity case",
                "missing roots outside scan bounds are reported as absent rather than extrapolated",
            ),
            units={
                "concentration": "mol/L",
                "volumetric_flow": "L/s",
                "volume": "L",
                "temperature": "K",
                "heat_capacity_density": "J/(L*K)",
                "heat_duty": "W",
                "rate_constant": "1/s",
            },
            reference_reading=(
                (
                    "Cantera: reference_repos/cantera/doc/sphinx/userguide/"
                    "reactor-tutorial.md describes CSTR residence time and "
                    "advance_to_steady_state."
                ),
                (
                    "Cantera: reference_repos/cantera/samples/python/reactors/"
                    "continuous_reactor.py uses reservoirs, MassFlowController, "
                    "PressureController, and ReactorNet for a stirred reactor."
                ),
                (
                    "IDAES: reference_repos/idaes-pse/idaes/models/unit_models/"
                    "cstr.py builds a 0D control volume with material, energy, "
                    "and reaction extent balances."
                ),
                ("IDAES: cstr_performance_eqn sets rate_reaction_extent = volume * reaction_rate."),
            ),
            validation_evidence=(
                ValidationEvidence(
                    evidence_id="cstr-multiplicity-root-test",
                    evidence_type="unit_test",
                    description=(
                        "The default case has three energy-balance roots with "
                        "stable/unstable/stable local dynamics."
                    ),
                    status="implemented",
                    command_or_path="tests/test_reactor_models.py",
                    tolerance="root residual <= 1e-6 W",
                ),
                ValidationEvidence(
                    evidence_id="cstr-performance-equation-test",
                    evidence_type="unit_test",
                    description=(
                        "At each root, the material-balance concentration and "
                        "reaction heat satisfy the CSTR performance equation."
                    ),
                    status="implemented",
                    command_or_path="tests/test_reactor_models.py",
                    tolerance="pytest.approx local tolerances",
                ),
            ),
            model_limit_notes=(
                (
                    "This is a professional reference slice for multiplicity, "
                    "not a full IDAES clone."
                ),
                (
                    "Future work should add Cantera dynamic cross-checks and "
                    "plant-scale heat-transfer variants."
                ),
            ),
            intended_use=(
                "CSTR ignition/extinction task design",
                "reactor-model maturity reporting",
                "reference case for agents reasoning about multiple steady states",
            ),
        ),
    )


__all__ = [
    "reactor_model_cards",
]
