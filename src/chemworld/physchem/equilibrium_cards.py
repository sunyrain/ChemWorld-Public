"""Phase-equilibrium model cards for ChemWorld."""

from __future__ import annotations

from chemworld.physchem.maturity import MaturityLevel, ModelCard, ValidationEvidence


def activity_model_cards() -> tuple[ModelCard, ...]:
    """Return model cards for the implemented activity-coefficient models."""

    return (
        ModelCard(
            model_id="activity_excess_thermodynamics",
            module_id="phase_equilibrium",
            title="Excess Gibbs Thermodynamics and Activity Calibration",
            maturity=MaturityLevel.REFERENCE_VALIDATED,
            summary=(
                "Ideal, Margules, Wilson, NRTL, UNIQUAC, RegularSolution, Flory-Huggins "
                "Hansen and all six UNIFAC variants share one excess Gibbs "
                "potential for activity, heat, analytic derivatives and bounded calibration."
            ),
            equations=(
                "ln(gamma_i) = partial(n*GE)/partial(n_i)/(R*T)",
                "HE = GE - T*dGE/dT; SE = -dGE/dT; CpE = -T*d2GE/dT2",
                "Margules GE/(RT) = sum_i<j xi*xj*((Aij+Aji)/2+(Aji-Aij)*(xi-xj)/2)",
                "Regular GE = sum_mn xm*xn*Vm*Vn*Amn / sum(x*V)",
                "Flory-Huggins adds RT*sum_i xi*ln(Vi/sum(x*V))",
                "Hansen Amn = alpha/2*(delta_d_difference^2 + "
                "0.25*delta_p_difference^2 + 0.25*delta_h_difference^2)",
            ),
            assumptions=(
                "Reported energies are excess quantities at the same temperature, "
                "pressure and composition, not absolute phase energy references.",
                "Composition derivatives distinguish independent x coordinates from "
                "normalized mole amounts; mole derivatives use a 1 mol basis.",
                "A single explicit Margules pair value declares symmetric interactions.",
                "Regular-solution volumes and cohesive parameters are constant reference "
                "inputs; the model does not invent their temperature dependence.",
            ),
            validity_limits=(
                "Requires explicit complete interaction pairs and finite positive temperature.",
                "Infinite dilution uses exact zero solute fraction with nonzero remaining solvent.",
                "Binary and ternary fixed same-equation references validate numerical behavior; "
                "they do not establish material prediction accuracy or database completeness.",
            ),
            failure_modes=(
                "Missing pairs never silently become ideal interactions.",
                "Nonpositive structural factors or singular denominators are rejected.",
                "A failed or rank-deficient fit is retained in the shared calibration report.",
            ),
            units={"GE/HE": "J/mol", "SE/CpE": "J/(mol*K)",
                   "temperature": "K", "gamma": "dimensionless"},
            reference_reading=(
                "https://thermo.readthedocs.io/thermo.activity.html: GibbsExcess public equations",
                "thermo 0.6.0 NRTL, Wilson and UNIQUAC public class documentation",
                "thermo 0.6.0 regular_solution public equations and Hansen Aijs source "
                "lines 1546-1547: the class formula omits the double-sum factor 1/2",
                "https://thermo.readthedocs.io/thermo.unifac.html: UNIFAC public equations, "
                "temperature variants and explicit group data; KT combinatorial is original",
            ),
            validation_evidence=(ValidationEvidence(
                evidence_id="native-conv23-activity-thermodynamics",
                evidence_type="reference_comparison",
                description=(
                    "90 states, 25 pure endpoints and 25 exact dilution limits: 4390 "
                    "80-digit equation comparisons and three heldout fits pass. The original "
                    "double-precision thermo comparison retains 12 UNIQUAC trace "
                    "Hessian differences."
                ),
                status="implemented", reference_backend="thermo 0.6.0 and mpmath 80 digits",
                command_or_path="workstreams/world_foundation/archive/20261023-completed/convergence/23/NATIVE_CONV23A_RESULTS.json",
                tolerance="rtol=2e-9, atol=2e-8; fits use their fixed criteria",
            ), ValidationEvidence(
                evidence_id="native-conv23-regular-solution-sources",
                evidence_type="reference_comparison",
                description=(
                    "Three regular-solution models, 54 states, 15 endpoints, 15 dilution "
                    "limits and all 2460 source Hansen vectors: 34614 comparisons pass "
                    "against independent 80-digit potentials and source responses. Five "
                    "fits verify predictions and the unidentifiable antisymmetric lambda. "
                    "234 upstream numerical-Hessian differences remain recorded."
                ),
                status="implemented", reference_backend="thermo 0.6.0 and mpmath 80 digits",
                command_or_path="workstreams/world_foundation/archive/20261023-completed/convergence/23/NATIVE_CONV23B_RESULTS.json",
                tolerance="rtol=2e-9, atol=2e-8; 23 fixed fitting criteria",
            ), ValidationEvidence(
                evidence_id="native-conv23-unifac-all-variants",
                evidence_type="reference_comparison",
                description=(
                    "All six variants: 108 states, 30 endpoints and 30 exact dilution "
                    "limits give 5178 passing comparisons; six shared heldout fits pass "
                    "36 criteria. All 936 subgroups, 17689 interaction rows, 31778 "
                    "InChIKey and 31754 CAS assignment rows are consumed with 492948 "
                    "passing source criteria. Source-row probes use authored unit "
                    "groups, not measured material validation. Initial KT and source "
                    "format/underflow failures remain recorded."
                ),
                status="implemented", reference_backend="thermo 0.6.0",
                command_or_path="workstreams/world_foundation/archive/20261023-completed/convergence/23/NATIVE_CONV23C_RESULTS.json",
                tolerance="rtol=2e-9, atol=2e-8; 36 fixed fitting criteria",
            )),
            model_limit_notes=(
                "Further group property estimates, EOS mixing, Twu alpha and unified "
                "phase objects remain separate unfinished parts of NATIVE-CONV-23.",
                "Asymmetric legacy Margules values are intentionally corrected to a "
                "thermodynamically integrable potential; frozen historical evidence is unchanged.",
            ),
            intended_use=("Native nonideal liquid properties and activity-parameter calibration.",),
        ),
        ModelCard(
            model_id="wilson_activity_coefficients",
            module_id="phase_equilibrium",
            title="Wilson Activity Coefficients",
            maturity=MaturityLevel.REFERENCE_VALIDATED,
            summary=(
                "JSON-friendly Wilson gamma model for VLE-oriented benchmark "
                "tasks with explicit asymmetric interaction parameters."
            ),
            equations=(
                "ln(gamma_i) = 1 - ln(sum_j x_j Lambda_ij) - "
                "sum_j x_j Lambda_ji / sum_k x_k Lambda_jk",
                "Lambda_ij = exp(a_ij + b_ij/T + c_ij ln(T) + "
                "d_ij T + e_ij/T**2 + f_ij T**2)",
            ),
            assumptions=(
                "Liquid mole fractions are normalized before evaluation.",
                "All off-diagonal Wilson interactions are directional and must "
                "be declared explicitly.",
                "Wilson is intended for liquid-phase VLE activity coefficients, "
                "not LLE prediction.",
            ),
            validity_limits=(
                "Requires positive finite Lambda values for all off-diagonal "
                "pairs.",
                "All six temperature coefficients and binary/ternary analytic responses "
                "are checked by the shared excess-thermodynamics evidence.",
            ),
            failure_modes=(
                "Missing directional interaction parameters fail during spec "
                "construction.",
                "Nonpositive Lambda values fail before gamma evaluation.",
                "Near-singular composition sums raise validation errors rather "
                "than being clipped.",
            ),
            units={
                "temperature": "K",
                "composition": "mole fraction",
                "gamma": "dimensionless",
                "lambda_a/lambda_c": "dimensionless",
                "lambda_b": "K",
                "lambda_d": "1/K",
                "lambda_e": "K^2",
                "lambda_f": "1/K^2",
            },
            reference_reading=(
                "reference_repos/thermo/thermo/wilson.py: Wilson_gammas and "
                "Wilson class",
                "reference_repos/phasepy/phasepy/actmodels/wilson.py: compact "
                "ln gamma API",
            ),
            validation_evidence=(
                ValidationEvidence(
                    evidence_id="thermo-wilson-binary-gammas",
                    evidence_type="optional_reference_test",
                    description=(
                        "Compares fixed-lambda Wilson activity coefficients "
                        "against thermo.wilson.Wilson_gammas."
                    ),
                    status="implemented",
                    reference_backend="thermo",
                    command_or_path=(
                        "tests/reference/test_optional_reference_backends.py"
                    ),
                    tolerance="rtol=1e-12",
                ),
            ),
            model_limit_notes=(
                "This card validates the gamma equations and parameter contract; "
                "it is not a complete Wilson parameter database.",
            ),
            intended_use=(
                "Reference-validated nonideal VLE benchmark slices.",
                "Solvent and volatility tasks where liquid nonideality matters.",
            ),
        ),
        ModelCard(
            model_id="nrtl_activity_coefficients",
            module_id="phase_equilibrium",
            title="NRTL Activity Coefficients",
            maturity=MaturityLevel.REFERENCE_VALIDATED,
            summary=(
                "General asymmetric NRTL gamma model for binary and "
                "multicomponent benchmark mixtures."
            ),
            equations=(
                "tau_ij = A_ij + B_ij/T + E_ij ln(T) + F_ij T + "
                "G_ij/T**2 + H_ij T**2",
                "alpha_ij = c_ij + d_ij T",
                "G_ij = exp(-alpha_ij tau_ij)",
                "ln(gamma_i) follows the standard local-composition NRTL sum "
                "over directional pair interactions.",
            ),
            assumptions=(
                "Liquid mole fractions are normalized before evaluation.",
                "Every off-diagonal tau and alpha interaction is directional "
                "and explicit.",
                "The shared GE block validates binary/ternary temperature-dependent "
                "coefficients; explicit declarations support additional components.",
            ),
            validity_limits=(
                "Requires positive alpha values for off-diagonal pairs.",
                "Temperature-dependent tau/alpha coefficients are accepted but "
                "need system-specific validation.",
            ),
            failure_modes=(
                "Missing tau or alpha parameters fail during spec construction.",
                "Nonpositive alpha values fail before gamma evaluation.",
                "Singular NRTL denominator states raise validation errors rather "
                "than being clipped.",
            ),
            units={
                "temperature": "K",
                "composition": "mole fraction",
                "gamma": "dimensionless",
                "tau_a/tau_e/alpha_c": "dimensionless",
                "tau_b": "K",
                "tau_f": "1/K",
                "tau_g": "K^2",
                "tau_h": "1/K^2",
                "alpha_d": "1/K",
            },
            reference_reading=(
                "reference_repos/thermo/thermo/nrtl.py: NRTL_gammas_binaries "
                "and NRTL class",
                "reference_repos/phasepy/phasepy/actmodels/nrtl.py: compact "
                "matrix tau/G API",
            ),
            validation_evidence=(
                ValidationEvidence(
                    evidence_id="thermo-nrtl-binary-gammas",
                    evidence_type="optional_reference_test",
                    description=(
                        "Compares fixed binary NRTL activity coefficients "
                        "against thermo.nrtl.NRTL_gammas_binaries."
                    ),
                    status="implemented",
                    reference_backend="thermo",
                    command_or_path=(
                        "tests/reference/test_optional_reference_backends.py"
                    ),
                    tolerance="rtol=1e-12",
                ),
            ),
            model_limit_notes=(
                "This card validates the NRTL equation path and parameter "
                "contract; it does not provide a public interaction-parameter "
                "database.",
            ),
            intended_use=(
                "Reference-validated nonideal VLE/LLE benchmark slices.",
                "Future solvent-selection and liquid-phase separation tasks.",
            ),
        ),
        ModelCard(
            model_id="uniquac_activity_coefficients",
            module_id="phase_equilibrium",
            title="UNIQUAC Activity Coefficients",
            maturity=MaturityLevel.REFERENCE_VALIDATED,
            summary=(
                "JSON-friendly UNIQUAC gamma model with explicit structural "
                "r/q parameters, directional tau interactions, and auditable "
                "combinatorial/residual log-gamma terms."
            ),
            equations=(
                "Phi_i = x_i r_i / sum_j x_j r_j",
                "theta_i = x_i q_i / sum_j x_j q_j",
                "l_i = z/2 (r_i - q_i) - (r_i - 1)",
                "ln(gamma_i) = ln(gamma_i^comb) + ln(gamma_i^res)",
                "tau_ij = exp(a_ij + b_ij/T + c_ij ln(T) + d_ij T + "
                "e_ij/T**2 + f_ij T**2), or direct positive tau_ij",
            ),
            assumptions=(
                "Liquid mole fractions are normalized before evaluation.",
                "Every component declares positive structural r and q "
                "parameters.",
                "Every off-diagonal tau interaction is directional and "
                "explicit.",
                "Default coordination number z is 10 unless the caller "
                "declares a positive z parameter.",
            ),
            validity_limits=(
                "Requires finite positive r, q, tau, temperature, and z.",
                "Validated on the thermo UNIQUAC binary example and local "
                "ternary directional-parameter regression tests.",
                "No public parameter database is bundled; callers must provide "
                "system-specific structural and interaction parameters.",
            ),
            failure_modes=(
                "Missing r/q structural parameters fail during spec "
                "construction.",
                "Missing off-diagonal tau parameters fail during spec "
                "construction.",
                "Nonpositive tau/r/q/z values fail rather than being clipped.",
                "Overflow in tau or gamma exponentiation is reported as a "
                "validation error.",
            ),
            units={
                "temperature": "K",
                "composition": "mole fraction",
                "gamma": "dimensionless",
                "r/q/z": "dimensionless",
                "tau/tau_a/tau_c": "dimensionless",
                "tau_b": "K",
                "tau_d": "1/K",
                "tau_e": "K^2",
                "tau_f": "1/K^2",
            },
            reference_reading=(
                "reference_repos/thermo/thermo/uniquac.py: UNIQUAC_gammas "
                "equation contract and binary examples",
                "reference_repos/phasepy/phasepy/mixtures.py: UNIQUAC r/q "
                "mixture parameter conventions",
            ),
            validation_evidence=(
                ValidationEvidence(
                    evidence_id="thermo-uniquac-binary-gammas",
                    evidence_type="optional_reference_test",
                    description=(
                        "Compares the documented thermo UNIQUAC binary "
                        "gamma example against ChemWorld's report path."
                    ),
                    status="implemented",
                    reference_backend="thermo",
                    command_or_path=(
                        "tests/reference/test_optional_reference_backends.py"
                    ),
                    tolerance="rtol=1e-12",
                ),
                ValidationEvidence(
                    evidence_id="chemworld-uniquac-report-structure",
                    evidence_type="unit_test",
                    description=(
                        "Checks phi/theta normalization, tau matrix exposure, "
                        "directional ternary parameters, and fail-fast "
                        "parameter validation."
                    ),
                    status="implemented",
                    command_or_path="tests/test_phase_equilibrium.py",
                    tolerance="pytest.approx formula closure",
                ),
            ),
            model_limit_notes=(
                "This card validates the UNIQUAC equation path and parameter "
                "contract; it does not provide a public UNIQUAC parameter "
                "database or a full LLE stability solver.",
            ),
            intended_use=(
                "Nonideal VLE/LLE and extraction benchmark slices that need "
                "activity coefficients with explicit structural parameters.",
                "Scenario cards where public/private worlds vary liquid "
                "nonideality through auditable parameters.",
            ),
        ),
        ModelCard(
            model_id="ideal_gamma_vle_temperature_reports",
            module_id="phase_equilibrium",
            title="Raoult-Law Bubble/Dew Temperature Reports",
            maturity=MaturityLevel.REFERENCE_VALIDATED,
            summary=(
                "Auditable mixture bubble-temperature, dew-temperature, "
                "K-value, and Rachford-Rice diagnostic reports built from "
                "component vapor-pressure reports and activity coefficients."
            ),
            equations=(
                "K_i = gamma_i P_sat,i(T) / (phi_i P)",
                "bubble point: sum_i x_i K_i = 1",
                "dew point: sum_i y_i / K_i = 1",
                "Rachford-Rice: sum_i z_i (K_i - 1) / "
                "(1 + beta (K_i - 1)) = 0",
            ),
            assumptions=(
                "Default vapor fugacity coefficients are one.",
                "Temperature solves use bracketed log-pressure residuals.",
                "Component vapor pressures come from explicit "
                "PropertyCorrelation records and pure saturation reports.",
                "Liquid nonideality is supplied by the selected "
                "ActivityModelSpec; no EOS liquid-volume correction is applied.",
            ),
            validity_limits=(
                "Requires all components to have vapor-pressure correlations "
                "with overlapping temperature validity ranges.",
                "Pressure must be inside the requested bubble/dew bracket.",
                "Validated on curated ideal ethanol/water style regression "
                "cases and Rachford-Rice closure tests.",
            ),
            failure_modes=(
                "Missing vapor-pressure correlations fail before solving.",
                "Nonpositive pressure, K-values, or vapor pressures fail.",
                "No sign change in the requested temperature bracket raises a "
                "diagnostic error rather than silently extrapolating.",
            ),
            units={
                "temperature": "K",
                "pressure": "Pa",
                "composition": "mole fraction",
                "K_value": "dimensionless",
                "Rachford-Rice beta": "dimensionless vapor fraction",
            },
            reference_reading=(
                "reference_repos/thermo/thermo/flash/flash_base.py: ideal "
                "bubble/dew/flash workflow notes",
                "reference_repos/chemicals/chemicals/rachford_rice.py: "
                "Rachford-Rice objective conventions",
            ),
            validation_evidence=(
                ValidationEvidence(
                    evidence_id="curated-ideal-binary-bubble-dew-temperature",
                    evidence_type="unit_test",
                    description=(
                        "Checks curated vapor-pressure based ideal binary "
                        "bubble/dew ordering, residual closure, and K-value "
                        "sums."
                    ),
                    status="implemented",
                    command_or_path="tests/test_phase_equilibrium.py",
                    tolerance="log-pressure residual <= 1e-8",
                ),
                ValidationEvidence(
                    evidence_id="rachford-rice-diagnostic-closure",
                    evidence_type="unit_test",
                    description=(
                        "Checks two-phase vapor fraction, single-phase "
                        "classification, and objective residual diagnostics."
                    ),
                    status="implemented",
                    command_or_path="tests/test_phase_equilibrium.py",
                    tolerance="absolute objective residual <= 1e-10",
                ),
            ),
            model_limit_notes=(
                "This is a transparent benchmark VLE diagnostic layer, not a "
                "replacement for a full EOS/GE flash package or a parameter "
                "database.",
            ),
            intended_use=(
                "Distillation, evaporation, and volatility benchmark tasks "
                "that need auditable phase-boundary diagnostics.",
                "Teaching notebooks where students inspect K-values and "
                "Rachford-Rice phase status.",
            ),
        ),
        ModelCard(
            model_id="gamma_phi_k_values_and_azeotrope_diagnostics",
            module_id="phase_equilibrium",
            title="Gamma-Phi K-Values and Binary Azeotrope Diagnostics",
            maturity=MaturityLevel.REFERENCE_VALIDATED,
            summary=(
                "Auditable gamma-phi K-value reports and binary isothermal "
                "relative-volatility crossing diagnostics for flash and "
                "distillation benchmark tasks."
            ),
            equations=(
                "K_i = gamma_i P_sat,i phi_i^l,ref Poynting_i / (phi_i^v P)",
                "relative volatility: alpha_light,heavy = K_light / K_heavy",
                "azeotrope-risk residual: r(x) = ln(alpha_light,heavy(x))",
            ),
            assumptions=(
                "Vapor fugacity coefficients, liquid reference fugacity "
                "coefficients, and Poynting factors are caller-supplied "
                "positive dimensionless mappings or default to one.",
                "The azeotrope diagnostic is an isothermal composition-grid "
                "scan of relative-volatility crossing, not a full phase "
                "stability or pressure-composition azeotrope solver.",
                "Activity coefficients come from ActivityModelSpec, so model "
                "parameter provenance remains outside the solver.",
            ),
            validity_limits=(
                "Requires finite positive pressure, vapor pressures, fugacity "
                "coefficients, and Poynting factors.",
                "The binary diagnostic requires exactly two components and a "
                "composition grid strictly inside (0, 1).",
                "Validated on formula closure, nonideal crossing, and "
                "no-crossing regression tests.",
            ),
            failure_modes=(
                "Missing or extra phi/Poynting keys fail before K-value "
                "evaluation.",
                "Nonpositive fugacity coefficients fail rather than being "
                "clipped.",
                "No relative-volatility sign change is reported as a "
                "diagnostic status with warnings, not as an exception.",
            ),
            units={
                "temperature": "K",
                "pressure": "Pa",
                "composition": "mole fraction",
                "gamma": "dimensionless",
                "phi": "dimensionless",
                "Poynting": "dimensionless",
                "K_value": "dimensionless",
                "relative_volatility": "dimensionless",
            },
            reference_reading=(
                "reference_repos/chemicals/chemicals/flash_basic.py: K_value "
                "gamma-phi equation hierarchy",
                "reference_repos/thermo/thermo/phases/phase.py: fugacity "
                "coefficient reporting conventions",
            ),
            validation_evidence=(
                ValidationEvidence(
                    evidence_id="gamma-phi-factor-closure",
                    evidence_type="unit_test",
                    description=(
                        "Checks K-values against explicit gamma, vapor phi, "
                        "liquid-reference phi, and Poynting factors."
                    ),
                    status="implemented",
                    command_or_path="tests/test_phase_equilibrium.py",
                    tolerance="pytest.approx formula closure",
                ),
                ValidationEvidence(
                    evidence_id="binary-azeotrope-crossing-diagnostic",
                    evidence_type="unit_test",
                    description=(
                        "Checks a Margules binary case with ln(alpha) crossing "
                        "and an ideal no-crossing control case."
                    ),
                    status="implemented",
                    command_or_path="tests/test_phase_equilibrium.py",
                    tolerance="bracketed residual sign change",
                ),
            ),
            model_limit_notes=(
                "The diagnostic highlights azeotrope risk for benchmark "
                "planning. It is not a rigorous gamma-phi flash, tangent-plane "
                "stability test, or azeotrope curve tracer.",
            ),
            intended_use=(
                "Distillation and flash tasks that need visible K-value "
                "provenance and relative-volatility warnings.",
                "Scenario cards that vary public/private nonideality without "
                "hiding the diagnostic contract.",
            ),
        ),
    )




__all__ = [
    "activity_model_cards",
]
