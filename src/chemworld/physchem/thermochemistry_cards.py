"""Thermochemistry model cards for ChemWorld."""

from __future__ import annotations

from chemworld.physchem.maturity import MaturityLevel, ModelCard, ValidationEvidence


def thermochemistry_model_cards() -> tuple[ModelCard, ...]:
    """Return metadata for native NASA7 and source-derived thermochemistry."""

    return (
        ModelCard(
            model_id="nasa7_species_reaction_thermochemistry_v1",
            module_id="thermochemistry",
            title="NASA7 Species And Reaction Thermochemistry",
            maturity=MaturityLevel.REFERENCE_VALIDATED,
            summary=(
                "ChemWorld-local NASA7 thermochemistry slice for Cp/H/S/G "
                "species properties, reaction Delta H/Delta G/K_eq, and "
                "thermochemistry-coupled reversible-rate detailed balance."
            ),
            equations=(
                "Cp/R = a0 + a1 T + a2 T^2 + a3 T^3 + a4 T^4",
                "H/RT = a0 + a1 T/2 + a2 T^2/3 + a3 T^3/4 + a4 T^4/5 + a5/T",
                "S/R = a0 ln(T) + a1 T + a2 T^2/2 + a3 T^3/3 + a4 T^4/4 + a6",
                "G = H - T S",
                "Delta G_rxn = sum_i nu_i G_i; K = exp(-Delta G_rxn/RT)",
                "K_c = K_dimensionless * C0^(sum_i nu_i)",
                "k_reverse = k_forward / K_c for reversible Arrhenius detailed balance",
            ),
            assumptions=(
                "NASA7 coefficients represent standard-state species properties.",
                "Reaction thermochemistry is formed by stoichiometric summation.",
                "Activities and pressure corrections are outside this first slice.",
                "The validated reversible-rate slice treats concentrations as "
                "standard-state activity proxies.",
            ),
            validity_limits=(
                "Only NASA7 coefficient rows are supported.",
                "Temperature must lie inside at least one declared segment.",
                "NASA9, Shomate, group additivity, and pressure corrections are not included.",
                "Detailed-balance validation currently covers compact homogeneous "
                "reversible reaction-network examples, not pressure-dependent kinetics.",
            ),
            failure_modes=(
                "Unsupported thermo models raise ValueError.",
                "Missing species thermochemistry raises KeyError.",
                "Overlapping or invalid temperature ranges raise ValueError.",
                "Large equilibrium-constant exponents are clipped for numerical safety.",
                "NASA7 reversible Arrhenius rates fail fast when species thermo is absent.",
            ),
            units={
                "temperature": "K",
                "cp": "J/mol/K",
                "enthalpy": "J/mol",
                "entropy": "J/mol/K",
                "gibbs": "J/mol",
                "equilibrium_constant": "dimensionless",
            },
            reference_reading=(
                "Cantera NasaPoly1/NasaPoly2 formulas and continuity validation.",
                "Cantera YAML NASA7 species thermo format.",
                "RMG NASA polynomial export to Cantera YAML.",
            ),
            validation_evidence=(
                ValidationEvidence(
                    evidence_id="nasa7-identity-tests",
                    evidence_type="unit-test",
                    description=(
                        "NASA7 Cp/H/S identities, Cantera-style YAML parsing, "
                        "reaction Delta G to K_eq, and continuity diagnostics."
                    ),
                    status="implemented",
                    command_or_path="tests/test_thermochemistry.py",
                    tolerance="analytical identities checked at 1e-12 to 1e-9 relative tolerances",
                ),
                ValidationEvidence(
                    evidence_id="nasa7-detailed-balance-reaction-network-tests",
                    evidence_type="unit-test",
                    description=(
                        "NASA7 species Gibbs energies drive K_eq(T), concentration "
                        "standard-state conversion, reverse Arrhenius rate constants, "
                        "and reversible batch ODE equilibrium ratios."
                    ),
                    status="implemented",
                    command_or_path="tests/test_reaction_network.py",
                    tolerance=(
                        "reverse-rate equality to numerical precision; equilibrium ratio "
                        "at 5e-3 relative tolerance"
                    ),
                ),
            ),
            model_limit_notes=(
                "This closes a thermochemistry polynomial slice, not a full "
                "Cantera/RMG thermochemistry database or group-additivity engine.",
                "It also closes the compact thermochemistry-coupled reversibility "
                "slice used by the validated reaction-network contract; broader falloff "
                "and pressure-dependent kinetics remain separate work.",
            ),
            intended_use=(
                "Reaction-network thermochemistry sanity checks.",
                "Reversible Arrhenius detailed-balance checks.",
                "Reactor energy-balance validation when species thermochemistry is supplied.",
            ),
        ),
        ModelCard(
            model_id="native_caloric_data_formation_stoichiometry",
            module_id="thermochemistry",
            title="Caloric source data, formation references and exact stoichiometry",
            maturity=MaturityLevel.REFERENCE_VALIDATED,
            summary=("Explicit external caloric data become existing Cp/H/S curves, scientific "
                     "input cards and species/reaction states; exact atom/isotope/charge "
                     "balance and formation-reference conversions share native contracts."),
            equations=("ChemSep16: Cp = A + exp(B/T + C + D*T + E*T^2)",
                       "Sf = (Hf-Gf)/T; Gf = Delta H - T Delta S",
                       "Hf(g) = Hf(l) + Hvap at the same temperature",
                       "M nu = 0 with exact rational atom/isotope/charge coefficients"),
            assumptions=("Source selection, pressure, phase, missing Tc and molecular shape "
                         "are explicit; no data or reference-library runtime is vendored.",
                         "ChemSep coefficients retain their declared amount/output units.",
                         "Tabular interpolation is an explicit linear model; source values "
                         "and uncertainty availability remain distinguishable."),
            validity_limits=("Source reproduction does not establish experimental accuracy.",
                             "Zero-K data are queryable but are not positive-T caloric datums.",
                             "Repeated table temperatures retain both limits; only a selected "
                             "one-sided interval is usable without transition information.",
                             "The reaction mechanism importer remains NASA7-only until task 34."),
            failure_modes=("Missing coefficients, undeclared ranges and ambiguous phases fail.",
                           "Table gaps, implicit extrapolation and crossed discontinuities fail.",
                           "Underdetermined balances require additional fixed coefficients.",
                           "Absolute entropy cannot substitute for formation entropy."),
            units={"temperature": "K", "capacity": "J/(mol*K)", "enthalpy": "J/mol",
                   "entropy": "J/(mol*K)", "molecular_weight": "g/mol"},
            reference_reading=("chemicals 1.5.2 public heat_capacity and reaction interfaces",
                               "ChemSep Help, Table 3, equation 16: "
                               "https://studylib.es/doc/9235141/chemsep-help"),
            validation_evidence=(
                ValidationEvidence(evidence_id="caloric-complete-source-84230",
                    evidence_type="same_equation_reference", status="implemented",
                    description="All 25 selected sources; records, scalar data, nested curves "
                                "and table intervals counted separately, all failures retained.",
                    command_or_path="tests/reference/run_caloric_data_block.py",
                    reference_backend="chemicals 1.5.2 and independent Gauss quadrature",
                    tolerance="Cp 2e-10 relative/2e-8 absolute; H/S 2e-8/2e-6 SI"),
                ValidationEvidence(evidence_id="chemsep-formation-stoichiometry-23",
                    evidence_type="same_equation_reference", status="implemented",
                    description="ChemSep16 values/slopes/integrals, six exact balances, "
                                "molar/mass conversion and formation Gibbs data.",
                    command_or_path="tests/reference/run_formation_reference_block.py",
                    tolerance="exact rational balances; slope 3e-7 relative/2e-9 absolute"),
                ValidationEvidence(evidence_id="source-native-caloric-runtime",
                    evidence_type="unit_test", status="implemented",
                    description="Source to input, species/reaction thermo, heat/wait and snapshot.",
                    command_or_path="tests/test_caloric_data.py",
                    tolerance="explicit synthetic data; exact resumed snapshot equality"),
            ),
            model_limit_notes=("Synthetic Tc/shape/ranges in reference checks are not "
                               "material parameter estimates.",
                               "Source licensing and runtime data access remain separate "
                               "from the clean-room formula implementation."),
            intended_use=("Explicit-source property inputs and native caloric consumers.",
                          "Formation/reaction thermochemistry and conserved reaction balancing."),
        ),
    )

__all__ = [
    "thermochemistry_model_cards",
]
