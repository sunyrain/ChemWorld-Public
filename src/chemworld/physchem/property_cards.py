"""Property-correlation model cards for ChemWorld."""

from __future__ import annotations

from chemworld.physchem.correlation_families import CORRELATION_FAMILIES
from chemworld.physchem.maturity import MaturityLevel, ModelCard, ValidationEvidence

_HEAT_ENTHALPY_REFERENCE_READING = (
    "reference_repos/chemicals/chemicals/dippr.py: EQ100 value and "
    "temperature integral",
    "reference_repos/chemicals/chemicals/heat_capacity.py: gas/liquid/solid "
    "heat-capacity method families and integral APIs",
    "reference_repos/chemicals/chemicals/phase_change.py: Watson latent-heat "
    "temperature correction",
    "reference_repos/thermo/thermo/chemical.py: phase-reference enthalpy "
    "paths across melting and boiling transitions",
)

_VOLUME_REFERENCE_READING = (
    "reference_repos/chemicals/chemicals/volume.py: Rackett liquid molar "
    "volume, Amgat liquid-mixture rule, ideal_gas, and CRC virial data notes",
    "reference_repos/chemicals/chemicals/virial.py: second-virial gas-volume "
    "families and mixture hooks",
    "reference_repos/thermo/thermo/volume.py: VolumeLiquid, VolumeGas, and "
    "VolumeLiquidMixture method organization and validity policy",
)

_TRANSPORT_REFERENCE_READING = (
    "reference_repos/chemicals/chemicals/viscosity.py: Andrade-like pure "
    "viscosity families and Wilke gas-mixture viscosity rule",
    "reference_repos/chemicals/chemicals/thermal_conductivity.py: DIPPR9B gas "
    "conductivity and DIPPR9H/DIPPR9I liquid-mixture conductivity rules",
    "reference_repos/thermo/thermo/viscosity.py and thermal_conductivity.py: "
    "method-governance organization for pure and mixture transport properties",
    "reference_repos/idaes-pse/.../viscosity_wilke.py: process-model Wilke "
    "callback pattern for gas-mixture viscosity",
    "reference_repos/idaes-pse/.../gas_phase_thermo.py: Fuller-style gas "
    "diffusivity scaling and mixture-diffusivity closure",
)

def property_correlation_model_cards() -> tuple[ModelCard, ...]:
    """Return model cards for generic property-correlation families."""

    return (
        ModelCard(
            model_id="native_dippr_correlation_families",
            module_id="properties",
            title="Native DIPPR equations and caloric integration",
            maturity=MaturityLevel.REFERENCE_VALIDATED,
            summary=("Fourteen declared formula variants with analytic first temperature "
                     "derivatives and adaptive definite integrals; existing caloric and "
                     "phase-path consumers use the same equations."),
            equations=tuple(key for key in CORRELATION_FAMILIES if key.startswith("dippr")),
            assumptions=("Coefficients use absolute K and the declared output-unit scale.",
                         "Each property_id must match its declared physical dimension."),
            validity_limits=("Synthetic same-equation validation; no material-data qualification.",
                             "Critical endpoints are excluded for 105/106/114/116.",
                             "Inventory admission requires an explicit U/Cv closure and "
                             "nonnegative Cv throughout its temperature interval."),
            failure_modes=("Unknown parameters, poles, invalid signs and nonfinite results fail.",
                           "Integration rejects crossed poles, unestablished positive heat "
                           "capacity and quadrature that misses the requested accuracy."),
            units={"temperature": "K", "value": "declared physical output unit",
                   "slope": "output unit/K", "integral": "output unit*K",
                   "integral_over_temperature": "output unit"},
            reference_reading=tuple(family.reference for key, family in CORRELATION_FAMILIES.items()
                                    if key.startswith("dippr")),
            validation_evidence=(ValidationEvidence(
                evidence_id="native-dippr-1.5.2-fixed-block",
                evidence_type="same_equation_reference",
                description="84 value/slope cases, 56 integrals and 2 caloric reference cases.",
                status="implemented", reference_backend="chemicals 1.5.2",
                command_or_path="tests/reference/test_native_dippr_reference.py",
                tolerance="value 2e-12; slope 3e-11; integral 5e-9 relative, explicit abs floors",
            ),),
            model_limit_notes=("A formula reference is not the source of a material parameter.",
                               "Integration error estimates are not experimental uncertainty.",
                               "Database coverage, ranking, coefficient fitting and higher "
                               "derivatives remain in NATIVE-CONV-22B through 22G."),
            intended_use=("Explicit-parameter caloric and property-package calculations.",),
        ),
        ModelCard(
            model_id="native_heat_capacity_families",
            module_id="properties",
            title="Native heat capacity families and piecewise caloric consumers",
            maturity=MaturityLevel.REFERENCE_VALIDATED,
            summary=("Fifteen gas/liquid/solid formula families with analytic first derivatives, "
                     "H/S quadrature, bounded temperature inversion and same-phase piecewise "
                     "curves consumed by native caloric inventories."),
            equations=tuple(key for key, family in CORRELATION_FAMILIES.items()
                            if family.reference_method),
            assumptions=("Coefficients use each formula's published convention and produce "
                         "J/(mol*K); Shomate coefficients use K rather than kK.",
                         "Similarity methods explicitly require MW and atoms per gram.",
                         "Rowlinson derivatives hold the supplied Cpgm parameter fixed.",
                         "Piecewise segments share a phase, capacity basis and energy datum; "
                         "the right segment owns an internal boundary."),
            validity_limits=("Formula reproduction with synthetic parameter variants, not "
                             "experimental material accuracy or complete parameter data.",
                             "Gas Cp-to-Cv conversion requires an explicit ideal-gas phase; "
                             "liquid/solid Cp alone does not determine U or Cv.",
                             "Mathematical bounds do not replace empirical validity ranges; "
                             "piecewise gaps and implicit extrapolation are rejected.",
                             "Statistical vibration supports polyatomic rigid rotors."),
            failure_modes=("Unknown parameters, critical poles and inconsistent phases fail.",
                           "Unestablished interval heat-capacity bounds or negative Cv fail.",
                           "Unreachable caloric targets and undeclared entropy data fail."),
            units={"temperature": "K", "capacity": "J/(mol*K)",
                   "slope": "J/(mol*K^2)", "enthalpy": "J/mol", "entropy": "J/(mol*K)"},
            reference_reading=tuple(family.reference for family in CORRELATION_FAMILIES.values()
                                    if family.reference_method),
            validation_evidence=(
                ValidationEvidence(
                    evidence_id="native-caloric-fixed-348",
                    evidence_type="same_equation_reference",
                    description=("348 units: 90 values/slopes, 180 integrals, 60 inverses, "
                                 "6 piecewise and 12 branch/limit units. Original 4 quasi-entropy "
                                 "reference discrepancies retained; a separate 9-unit diagnosis "
                                 "supports the explicit series-reference corrected block."),
                    status="implemented",
                    reference_backend="chemicals 1.5.2 and independent series",
                    command_or_path="tests/reference/run_heat_capacity_foundation_block.py",
                    tolerance="Cp 2e-11 relative; derivative 3e-7; integral 2e-10; root 2e-7 K",
                ),
                ValidationEvidence(
                    evidence_id="native-piecewise-caloric-consumer",
                    evidence_type="unit_test",
                    description="Native thermal operations and exact JSON snapshot continuation.",
                    status="implemented", command_or_path="tests/test_heat_capacity_families.py",
                    tolerance="exact snapshot equality; explicit thermodynamic tolerances",
                ),
            ),
            model_limit_notes=("Quadrature error estimates are not material uncertainty.",
                               "CaloricDatabase, PressureDatabase and PhaseChangeDatabase "
                               "read explicitly supplied external sources; data coverage "
                               "does not establish material prediction accuracy."),
            intended_use=("Declared caloric curves for property packages, phase-path H/S, "
                          "native gas thermal inventories and replayable energy closure.",),
        ),
        ModelCard(
            model_id="vapor_pressure_correlation_families",
            module_id="properties",
            title="Vapor-Pressure Correlation Families",
            maturity=MaturityLevel.REFERENCE_VALIDATED,
            summary=(
                "Formula-level vapor and sublimation pressure evaluators with "
                "explicit validity ranges, analytic temperature derivatives, "
                "and JSON-friendly reports."
            ),
            equations=(
                "Antoine: log_base(P) = A - B/(T + C)",
                "Wagner original 3,6 form: ln(P/Pc) = "
                "(a*tau + b*tau**1.5 + c*tau**3 + d*tau**6)/Tr",
                "DIPPR101: P = exp(A + B/T + C*ln(T) + D*T**E)",
                "Wagner2.5,5, TRC extended Antoine, Yaws, TDE and DIPPR115 pressure forms.",
                "Boiling/critical, Lee-Kesler, Ambrose-Walton, Sanjari and Edalat estimators; "
                "Clapeyron sublimation and explicit Arrhenius extrapolation.",
                "IF97 region-4 water saturation with one shared forward/inverse relation.",
                "dP/dT, d2P/dT2, d3P/dT3 and named coefficient Jacobians "
                "use analytic chain rules on the same formula as pressure.",
            ),
            assumptions=(
                "Coefficient units must match each PropertyCorrelation input "
                "and output unit declaration.",
                "Validity ranges are treated as benchmark contracts; callers "
                "choose warn, raise, or ignore policy.",
                "Sublimation pressure uses the same formula families when a "
                "caller supplies sublimation-pressure coefficients.",
            ),
            validity_limits=(
                "No automatic method ranking beyond the caller-provided "
                "correlation order in ComponentPropertyPackage.",
                "No data-table vendoring; only explicitly curated coefficients "
                "are shipped.",
                "Critical-region behavior is limited to declared correlation "
                "validity bounds.",
            ),
            failure_modes=(
                "Unsupported equations fail before derivative evaluation.",
                "Antoine singularities fail when T + C is nonpositive.",
                "Legacy Wagner3,6 requires T<Tc. Wagner2.5,5 supports the critical value "
                "and first derivative; divergent higher derivatives fail explicitly.",
                "Underflow to zero pressure and undefined TRC branch derivatives fail.",
                "Out-of-range calls can hard-fail with validity_policy='raise'.",
            ),
            units={
                "temperature": "K or declared temperature unit",
                "pressure": "Pa or declared pressure unit",
                "dP_dT": "declared_pressure_unit/K",
                "d2P_dT2": "declared_pressure_unit/K^2",
                "d3P_dT3": "declared_pressure_unit/K^3",
                "dlnP_dT": "1/K",
            },
            reference_reading=(
                "reference_repos/chemicals/chemicals/vapor_pressure.py: "
                "Antoine, Wagner, dWagner_dT, vapor-pressure data families",
                "reference_repos/chemicals/chemicals/dippr.py: EQ101 and "
                "order=1 derivative",
                "reference_repos/thermo/thermo/vapor_pressure.py: "
                "VaporPressure method ranking, validity limits, and derivative API",
            ),
            validation_evidence=(
                ValidationEvidence(
                    evidence_id="complete-pressure-methods-and-sources",
                    evidence_type="reference_comparison",
                    description=("Fixed 72-unit public-method/derivative/Jacobian/recovery block "
                                 "and separate complete 13-source external-data block; source "
                                 "missing/invalid data and all original failures stay explicit."),
                    status="implemented",
                    command_or_path="workstreams/world_foundation/NATIVE_CONV22C3_EXPERIMENT.md",
                    tolerance="Fixed per-measurement tolerances in the experiment note",
                    reference_backend="chemicals 1.5.2",
                ),
                ValidationEvidence(
                    evidence_id="antoine-vapor-pressure-derivative-test",
                    evidence_type="unit_test",
                    description=(
                        "Checks Antoine pressure and analytic derivative "
                        "against central finite differences."
                    ),
                    status="implemented",
                    command_or_path="tests/test_physchem_properties.py",
                    tolerance="rtol=1e-5",
                ),
                ValidationEvidence(
                    evidence_id="dippr101-vapor-pressure-derivative-test",
                    evidence_type="unit_test",
                    description=(
                        "Checks DIPPR101 analytic dP/dT against central finite "
                        "differences for curated compounds."
                    ),
                    status="implemented",
                    command_or_path="tests/test_physchem_properties.py",
                    tolerance="rtol=1e-5",
                ),
            ),
            model_limit_notes=(
                "Complete selected pressure sources use an explicit external directory; "
                "coefficients are not redistributed or treated as validated material predictions.",
                "VDI interpolation is explicit ln(P) versus 1/T, returned as native source "
                "intervals; no cross-knot derivative or missing-data filling.",
                "IF97 region 4 is a saturation curve, not the full water EOS or phase envelope.",
            ),
            intended_use=(
                "Flash, distillation, volatility-risk, and safety-envelope "
                "tasks requiring auditable vapor-pressure values and slopes.",
                "Benchmark datasets that need replayable property reports with "
                "validity status.",
            ),
        ),
        ModelCard(
            model_id="pure_fluid_saturation_solver",
            module_id="properties",
            title="Pure-Fluid Saturation Solver",
            maturity=MaturityLevel.REFERENCE_VALIDATED,
            summary=(
                "Auditable pure-fluid saturation reports built on explicit "
                "vapor-pressure correlations, analytic pressure slopes, "
                "bracketed temperature inversion, and critical-region guards."
            ),
            equations=(
                "Forward saturation: P_sat = vapor_pressure_report(T).",
                "Inverse saturation: solve ln(P_sat(T)/P_target) = 0.",
                "Temperature inversion uses bracketed bisection on the "
                "log-pressure residual.",
                "Critical guards require T < Tc and P < Pc when critical "
                "properties are supplied.",
            ),
            assumptions=(
                "The vapor-pressure correlation remains the source of the "
                "saturation curve; the solver does not infer hidden data.",
                "Temperature brackets come from declared validity ranges unless "
                "the caller supplies an explicit bracket.",
                "Critical warnings are metadata for benchmark agents, not "
                "high-accuracy critical-region thermodynamics.",
            ),
            validity_limits=(
                "Pure-fluid only; mixture bubble/dew and azeotrope behavior are "
                "handled by later flash tasks.",
                "No EOS phase-envelope tracing or superancillary equations are "
                "implemented in this slice.",
                "A non-bracketing target pressure hard-fails instead of "
                "extrapolating silently.",
                "A bracketed root does not certify global uniqueness; Sanjari requires an "
                "explicit bracket. Explicit extrapolation uses a separately constructed curve.",
            ),
            failure_modes=(
                "Nonpositive pressure or temperature fails.",
                "T >= Tc or P >= Pc fails when critical properties are supplied.",
                "No bracketing sign change fails with residual diagnostics.",
                "Unsupported vapor-pressure equations fail through the lower "
                "property-correlation layer.",
            ),
            units={
                "temperature": "K",
                "pressure": "Pa",
                "log_pressure_residual": "dimensionless",
            },
            reference_reading=(
                "reference_repos/thermo/thermo/vapor_pressure.py: "
                "VaporPressure method governance and validity limits",
                "reference_repos/coolprop/Web/coolprop/HighLevelAPI.rst: "
                "saturation inputs with vapor quality and critical-region "
                "warnings",
                "reference_repos/phasepy/phasepy/mixtures.py: Psat/Tsat "
                "inverse API shape for Antoine-like correlations",
            ),
            validation_evidence=(
                ValidationEvidence(
                    evidence_id="pure-saturation-temperature-inversion-test",
                    evidence_type="unit_test",
                    description=(
                        "Checks water Antoine Psat/Tsat closure and normal "
                        "boiling-point inversion with residual diagnostics."
                    ),
                    status="implemented",
                    command_or_path="tests/test_physchem_properties.py",
                    tolerance="relative temperature tolerance 5e-4",
                ),
                ValidationEvidence(
                    evidence_id="pure-saturation-critical-guard-test",
                    evidence_type="unit_test",
                    description=(
                        "Checks near-critical warnings, critical hard-fails, "
                        "and non-bracketing pressure failures."
                    ),
                    status="implemented",
                    command_or_path="tests/test_physchem_properties.py",
                    tolerance="guardrail behavior",
                ),
            ),
            model_limit_notes=(
                "This is a reference-validated numerical contract for "
                "benchmark saturation calls, not a replacement for CoolProp "
                "or full EOS phase-envelope tracing.",
            ),
            intended_use=(
                "Flash, distillation, vapor-loss, pressure-risk, and scenario "
                "generation tasks needing replayable pure-fluid saturation "
                "points.",
                "Dataset exports that need explicit residuals, brackets, and "
                "critical warning metadata.",
            ),
        ),
        ModelCard(
            model_id="phase_heat_capacity_enthalpy_package",
            module_id="properties",
            title="Phase-Aware Heat Capacity And Enthalpy Package",
            maturity=MaturityLevel.REFERENCE_VALIDATED,
            summary=(
                "Phase-tagged molar heat-capacity evaluation, analytic polynomial "
                "and declared-correlation quadrature integrals, signed latent-heat transitions, "
                "and mixture enthalpy ledgers for reactor and separation duties."
            ),
            equations=(
                "DIPPR100-style Cp polynomial: Cp = a + bT + cT^2 + dT^3 + eT^4.",
                "Sensible enthalpy: Delta H = integral Cp(T) dT along one phase.",
                "Watson latent heat: Hvap(T) = Hvap_ref*((1-Tr)/(1-Tr_ref))^n.",
                "Clapeyron, Pitzer, SMK/MK, Velasco, Alibakhshi and PPDS12 latent-heat families.",
                "Riedel/Chen/Liu/Vetere give normal-boiling anchors; "
                "Watson_n recovers the exponent.",
                "Phase paths sum sensible segments plus signed latent heats.",
                "Mixture duty: Delta H_mix = sum_i n_i Delta H_i.",
            ),
            assumptions=(
                "Heat-capacity correlations are molar and declare their phase "
                "in metadata when a phase-specific API is used.",
                "Reference-state temperature is explicit in every report; no "
                "global hidden enthalpy zero is assumed.",
                "Transitions are caller-declared and traversed by phase labels "
                "rather than inferred from hidden data tables.",
            ),
            validity_limits=(
                "Polynomial, DIPPR and fifteen caloric formula families support "
                "same-phase integration and external data through CaloricDatabase.",
                "PhaseChangeDatabase preserves full sources, single-temperature observations "
                "and explicit table intervals; missing source conditions must be supplied.",
                "Fusion, vaporization and sublimation have distinct positive directions. "
                "Critical values do not imply finite critical derivatives; Clapeyron's "
                "T derivative holds its declared Psat/dZ fixed.",
                "No EOS departure enthalpy, pressure correction, or broad "
                "component database is included here.",
            ),
            failure_modes=(
                "Unsupported Cp equations fail before integration.",
                "Missing phase Cp, invalid phase labels, and disconnected phase "
                "paths fail explicitly.",
                "Negative or unestablished heat capacity over the integration "
                "interval fails instead of being clipped.",
                "Latent heat direction is signed; invalid transition direction "
                "raises a validation error.",
            ),
            units={
                "temperature": "K",
                "heat_capacity": "J/(mol*K)",
                "molar_enthalpy": "J/mol",
                "mixture_enthalpy": "J",
            },
            reference_reading=_HEAT_ENTHALPY_REFERENCE_READING,
            validation_evidence=(
                ValidationEvidence(
                    evidence_id="native-latent-fixed-methods-and-sources",
                    evidence_type="same_equation_reference",
                    description="63 method units and 112703 complete source records; "
                                "computed curves, points and missing data counted separately.",
                    status="implemented", reference_backend="chemicals 1.5.2",
                    command_or_path="tests/reference/run_phase_change_methods_block.py; "
                                    "tests/reference/run_phase_change_data_block.py",
                    tolerance="values 3e-10 relative/1e-8 J/mol; "
                              "slopes 5e-7 relative/3e-7 J/(mol K)",
                ),
                ValidationEvidence(
                    evidence_id="phase-cp-integral-regression-test",
                    evidence_type="unit_test",
                    description=(
                        "Checks gas, liquid, and solid Cp polynomial integrals "
                        "and reference-state zero behavior."
                    ),
                    status="implemented",
                    command_or_path="tests/test_physchem_properties.py",
                    tolerance="rtol=1e-12",
                ),
                ValidationEvidence(
                    evidence_id="phase-transition-ledger-test",
                    evidence_type="unit_test",
                    description=(
                        "Checks signed vaporization/condensation latent heat "
                        "and mole-weighted mixture enthalpy ledger closure."
                    ),
                    status="implemented",
                    command_or_path="tests/test_physchem_properties.py",
                    tolerance="rtol=1e-12",
                ),
            ),
            model_limit_notes=(
                "This closes a path/ledger slice needed by ChemWorld reactor "
                "and flash duties. Source formula agreement is not material validation "
                "or a general EOS departure-property package. VLE free-energy models "
                "keep their own consistent Psat-derived enthalpy difference.",
            ),
            intended_use=(
                "Dynamic reactor heat-duty ledgers and flash/distillation "
                "energy-balance checks.",
                "Benchmark trajectories that need auditable sensible and "
                "latent heat contributions with provenance.",
            ),
        ),
        ModelCard(
            model_id="density_molar_volume_package",
            module_id="properties",
            title="Density And Molar-Volume Package",
            maturity=MaturityLevel.REFERENCE_VALIDATED,
            summary=(
                "Fifteen pure volume/density methods with saturation, molten-liquid, "
                "solid and ideal-gas paths, analytic derivatives and parameter Jacobians; "
                "second-virial gas reports and Amgat-style mixture volume ledgers."
            ),
            equations=(
                "Rackett: Vm = R Tc/Pc * Zc**(1 + (1 - T/Tc)**(2/7)).",
                "Ideal gas: Vm = R T/P, Z = 1.",
                "CRC second virial polynomial: B = (a1 + t(a2 + t(a3 + "
                "t(a4 + a5 t))))*1e-6, t = 298.15/T - 1.",
                "Virial gas root: P Vm^2 - R T Vm - R T B = 0.",
                "Amgat mixture: Vm_mix = sum_i x_i Vm_i.",
            ),
            assumptions=(
                "Liquid Rackett reports are low-pressure saturated-liquid style "
                "estimates and require caller-supplied critical constants.",
                "Virial gas reports are intended for low to moderate density; "
                "compressibility status is reported from |B/Vm|.",
                "Mixture volume ledger assumes zero excess volume.",
            ),
            validity_limits=(
                "The legacy Rackett alias requires T < Tc; explicit pure volume families "
                "allow finite critical values but reject divergent derivatives and T > Tc. "
                "Saturation slopes are not fixed-pressure liquid responses.",
                "Campbell hydroxyl_polar_term=1 follows the chemicals 1.5.2 public API; "
                "=0 follows its rendered hydroxyl beta equation. Source disagreement is retained.",
                "Only a CRC-style second-virial coefficient polynomial hook is "
                "implemented; broader Tsonopoulos/Pitzer families are future "
                "deepening work.",
                "No bulk component density database is vendored.",
            ),
            failure_modes=(
                "Nonpositive Tc, Pc, Zc, temperature, pressure, density, "
                "molecular weight, or molar volume raises ValueError.",
                "Virial roots fail when the quadratic has no positive gas "
                "molar-volume root.",
                "Mixture ledgers fail on fraction mismatch, missing component "
                "volume, or missing molecular weight.",
            ),
            units={
                "temperature": "K",
                "pressure": "Pa",
                "molar_volume": "m^3/mol",
                "density": "kg/m^3",
                "molecular_weight": "g/mol",
                "second_virial": "m^3/mol",
            },
            reference_reading=_VOLUME_REFERENCE_READING,
            validation_evidence=(
                ValidationEvidence(
                    evidence_id="pure-volume-methods-d2a",
                    evidence_type="unit_test",
                    description=("221 fixed development units cover all 15 pure methods, "
                        "33 parameter sets, full Bhirud table, analytic derivatives, "
                        "JSON and property-package consumption; 18 additional hydroxyl "
                        "convention units retain the documented/API distinction."),
                    status="implemented",
                    command_or_path="tests/reference/run_volume_pure_block.py",
                    tolerance="Fixed value/derivative tolerances in NATIVE_CONV22D2A_EXPERIMENT.md",
                ),
                ValidationEvidence(
                    evidence_id="rackett-liquid-volume-test",
                    evidence_type="unit_test",
                    description=(
                        "Checks Rackett liquid molar volume and density "
                        "conversion against a hand calculation."
                    ),
                    status="implemented",
                    command_or_path="tests/test_physchem_properties.py",
                    tolerance="rtol=1e-12",
                ),
                ValidationEvidence(
                    evidence_id="virial-gas-volume-root-test",
                    evidence_type="unit_test",
                    description=(
                        "Checks the second-virial gas-volume quadratic root, "
                        "compressibility factor, and warning status."
                    ),
                    status="implemented",
                    command_or_path="tests/test_physchem_properties.py",
                    tolerance="rtol=1e-12",
                ),
            ),
            model_limit_notes=(
                "This is a density/molar-volume ledger slice for ChemWorld "
                "tasks. It is not CoolProp density coverage, compressed-liquid "
                "Tait modeling, COSTALD mixture fitting, or a full virial "
                "correlation library.",
            ),
            intended_use=(
                "Flash, distillation, extraction, and safety tasks that need "
                "auditable volume/density values.",
                "Benchmark records that need compressibility status instead of "
                "unlabeled gas-density corrections.",
            ),
        ),
        ModelCard(
            model_id="transport_property_package",
            module_id="properties",
            title="Transport Property Package",
            maturity=MaturityLevel.REFERENCE_VALIDATED,
            summary=(
                "Pure-component viscosity and thermal-conductivity reports, "
                "Wilke gas-mixture viscosity ledgers, DIPPR9H liquid-mixture "
                "conductivity ledgers, Fuller-style gas diffusivity estimates, "
                "and thermal-diffusivity reports with explicit uncertainty and "
                "validity metadata."
            ),
            equations=(
                "Andrade/Arrhenius viscosity: mu = A exp(B/T).",
                "Sutherland gas viscosity: mu = mu_ref (T/T_ref)^1.5 "
                "(T_ref + S)/(T + S).",
                "Linear conductivity: k = k_ref(1 + alpha(T - T_ref)); "
                "polynomial conductivity is also supported.",
                "DIPPR9B gas conductivity from gas viscosity, Cv, molecular "
                "weight, molecule type, and critical temperature.",
                "Wilke gas mixture: mu_mix = sum_i y_i mu_i / sum_j y_j phi_ij.",
                "DIPPR9H liquid mixture: k_mix = (sum_i w_i/k_i^2)^-0.5.",
                "Fuller gas diffusivity scales as T^1.75/P and diffusion-volume "
                "groups; mixture effective diffusivity uses resistance summation.",
                "Thermal diffusivity: alpha = k/(rho Cp).",
            ),
            assumptions=(
                "Pure-property correlations use caller-supplied coefficients "
                "with declared units and validity ranges.",
                "Wilke and Fuller paths are low-pressure gas estimates; reported "
                "uncertainty metadata must remain visible to tasks and datasets.",
                "Liquid mixture conductivity uses the nonaqueous DIPPR9H rule "
                "unless a future task selects another ledger.",
            ),
            validity_limits=(
                "No broad viscosity, conductivity, or diffusion-volume database "
                "is vendored.",
                "No high-pressure gas viscosity correction, electrolyte "
                "transport, multicomponent Maxwell-Stefan solver, or "
                "temperature-dependent calibrated diffusion model is included.",
                "DIPPR9H warnings are emitted for broad use and when component "
                "conductivities differ by more than 2x.",
            ),
            failure_modes=(
                "Nonpositive temperature, pressure, viscosity, conductivity, "
                "diffusivity, density, heat capacity, molecular weight, or "
                "diffusion volume raises ValueError.",
                "Composition/key mismatches fail explicitly before mixture "
                "ledger construction.",
                "Unsupported molecule types or missing critical temperature for "
                "linear DIPPR9B gas conductivity fail explicitly.",
            ),
            units={
                "temperature": "K",
                "pressure": "Pa",
                "dynamic_viscosity": "Pa*s",
                "thermal_conductivity": "W/(m*K)",
                "diffusivity": "m^2/s",
                "molecular_weight": "g/mol",
            },
            reference_reading=_TRANSPORT_REFERENCE_READING,
            validation_evidence=(
                ValidationEvidence(
                    evidence_id="transport-viscosity-conductivity-report-test",
                    evidence_type="unit_test",
                    description=(
                        "Checks Andrade/Sutherland reports, conductivity "
                        "validity, and unit conversion."
                    ),
                    status="implemented",
                    command_or_path="tests/test_physchem_properties.py",
                    tolerance="rtol=1e-12",
                ),
                ValidationEvidence(
                    evidence_id="transport-mixture-diffusivity-ledger-test",
                    evidence_type="unit_test",
                    description=(
                        "Checks Wilke gas viscosity, DIPPR9H liquid "
                        "conductivity, Fuller binary gas diffusivity, and "
                        "mixture effective diffusivity ledgers."
                    ),
                    status="implemented",
                    command_or_path="tests/test_physchem_properties.py",
                    tolerance="rtol=1e-12",
                ),
            ),
            model_limit_notes=(
                "This is a compact transport-property reporting and ledger "
                "slice for ChemWorld tasks. It is not a replacement for "
                "chemicals/thermo/CoolProp transport coverage or a full "
                "Maxwell-Stefan transport backend.",
            ),
            intended_use=(
                "Heat-transfer, reactor, distillation, separation, and "
                "instrument tasks that need explicit transport properties.",
                "Benchmark logs that must expose uncertainty and method family "
                "rather than silently using constants.",
            ),
        ),
    )




__all__ = [
    "_HEAT_ENTHALPY_REFERENCE_READING",
    "_TRANSPORT_REFERENCE_READING",
    "_VOLUME_REFERENCE_READING",
    "property_correlation_model_cards",
]
