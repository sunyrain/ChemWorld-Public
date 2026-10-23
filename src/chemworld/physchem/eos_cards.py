"""Equation-of-state model cards for ChemWorld."""

from __future__ import annotations

from chemworld.physchem.maturity import MaturityLevel, ModelCard, ValidationEvidence


def eos_model_cards() -> tuple[ModelCard, ...]:
    return (
        ModelCard(
            model_id="cubic_eos_pr_srk_residuals",
            module_id="eos",
            title="Peng-Robinson/SRK Cubic EOS Fugacity And Residual Properties",
            maturity=MaturityLevel.REFERENCE_VALIDATED,
            summary=(
                "PR/SRK cubic EOS with quadratic, predictive or excess-Gibbs mixing, explicit "
                "root selection, fugacity coefficients, and molar residual "
                "enthalpy/entropy/Gibbs properties."
            ),
            equations=(
                "a_i(T)=Omega_a R^2 Tc_i^2/Pc_i alpha_i(T)",
                "b_i=Omega_b R Tc_i/Pc_i",
                "a_mix=sum_i sum_j x_i x_j sqrt(a_i a_j)(1-k_ij)",
                "b_mix=sum_i x_i b_i",
                "GE rules: HV, exact MHV, MHV1, MHV2 and Wong-Sandler define a(T,x), b(T,x).",
                "A^R=-RT ln(1-b/V)-a/[b(s1-s2)] ln[(V+s1 b)/(V+s2 b)].",
                "GE-rule ln(phi_i)=d[n A^R/(RT)]/dn_i|T,V,n_j-ln(Z); "
                "H^R=A^R-T A^R_T+RT(Z-1), S^R=-A^R_T+R ln(Z).",
                "PR: H^R=RT(Z-1)+(T da/dT-a)/(2 sqrt(2) b) ln((Z+(1+sqrt(2))B)/(Z+(1-sqrt(2))B))",
                "SRK: H^R=RT(Z-1)+(T da/dT-a)/b ln((Z+B)/Z)",
                "S^R=R ln(Z-B)+(da/dT/c) ln(argument), with c=2 sqrt(2) b for PR and b for SRK",
            ),
            assumptions=(
                "Mixing explicitly selects quadratic, PPR78 or a source-bearing GE model. "
                "MHV1/2 q coefficients and all Wong-Sandler binary pairs are declared inputs.",
                "Pure-component alpha functions explicitly select the legacy PR/SRK "
                "Soave acentric-factor form or the three-coefficient Twu91 form.",
                "Residual properties are molar mixture departures from ideal gas "
                "at the same T/P/composition.",
            ),
            validity_limits=(
                "Requires positive critical temperature, critical pressure, "
                "pressure, and temperature.",
                "Root selection is explicit but no phase-stability or "
                "phase-envelope solve is claimed.",
                "Near-critical and highly associating/polar fluids require "
                "stronger EOS or fitted parameters.",
            ),
            failure_modes=(
                "No admissible real Z root raises ValueError.",
                "Invalid composition ids, negative composition entries, or "
                "nonpositive log arguments raise ValueError.",
                "Missing binary parameters default to k_ij=0 and should be "
                "governed by scenario metadata.",
            ),
            units={
                "temperature": "K",
                "pressure": "Pa",
                "molar_volume": "m^3/mol",
                "residual_enthalpy": "J/mol",
                "residual_entropy": "J/(mol*K)",
                "fugacity_coefficient": "dimensionless",
            },
            reference_reading=(
                "reference_repos/thermo/thermo/eos.py "
                "main_derivatives_and_departures and PR/SRK classes",
                "reference_repos/phasepy/phasepy/cubic/cubicpure.py and "
                "cubicmix.py logfug/EntropyR/EnthalpyR APIs",
                "reference_repos/thermopack/addon/pycThermopack/thermopack/"
                "thermo.py residual enthalpy/entropy API",
                "reference_repos/teqp/teqp/__init__.py exposes fugacity and "
                "critical/VLE architecture hooks",
            ),
            validation_evidence=(
                ValidationEvidence(
                    evidence_id="ge-cubic-mixing-potential-consumers",
                    evidence_type="independent_high_precision_reference",
                    description=(
                        "Five rules, PR/SRK and binary/ternary states: 120 mixing-response "
                        "and 480 liquid/vapor-root states pass 6480/6480 fixed comparisons "
                        "against independent 80-digit mixing and Helmholtz equations. "
                        "Covers temperature derivatives through third order, fugacity, "
                        "H/S/G/Cp and mechanical responses, including WS db/dT. "
                        "Nine GE families and all six UNIFAC variants reach original EOS."
                    ),
                    status="passing",
                    command_or_path="workstreams/world_foundation/archive/20261023-completed/convergence/23/NATIVE_CONV23G_RESULTS.json",
                    tolerance="2e-8 absolute/relative; existing numerical Cp 2e-6+2e-7 relative",
                ),
                ValidationEvidence(
                    evidence_id="ppr78-full-source-eos-consumer",
                    evidence_type="independent_high_precision_reference",
                    description=(
                        "All 225 original and 1600 extended interaction rows enter k(T), "
                        "three temperature derivatives and original PR78 mixture consumers. "
                        "The prospective domain-aware block passes 35942/35942 criteria: "
                        "1745 valid EOS states and 80 explicit negative-attraction rejections. "
                        "The initial block's 84 exceptions remain recorded."
                    ),
                    status="passing",
                    command_or_path="workstreams/world_foundation/archive/20261023-completed/convergence/23/NATIVE_CONV23F_RESULTS.json",
                    tolerance="2e-8 absolute + 2e-8 relative",
                ),
                ValidationEvidence(
                    evidence_id="bell2018-twu-full-source-consumer",
                    evidence_type="independent_high_precision_reference",
                    description=(
                        "All 2571 Bell2018 rows enter original PR and SRK EOS consumers. "
                        "Alpha through third temperature derivative, actual vapor states "
                        "and source JSON roundtrip pass 97698/97698 fixed criteria. "
                        "This establishes equation/data consumption, not material accuracy "
                        "or portability of fitted Bell coefficients between EOS families."
                    ),
                    status="passing",
                    command_or_path="workstreams/world_foundation/archive/20261023-completed/convergence/23/NATIVE_CONV23E_RESULTS.json",
                    tolerance="2e-8 absolute + 2e-9 relative",
                ),
                ValidationEvidence(
                    evidence_id="default-eos-residual-tests",
                    evidence_type="unit_tests",
                    description=(
                        "Default tests cover low-pressure ideal-gas limits, explicit "
                        "root policy, Gibbs consistency with fugacity coefficients, "
                        "and positive PR/SRK mixture fugacity coefficients."
                    ),
                    status="passing",
                    command_or_path="python -m pytest tests/test_eos.py",
                    tolerance="relative tolerances documented in tests",
                ),
                ValidationEvidence(
                    evidence_id="optional-thermo-eos-reference",
                    evidence_type="optional_reference_backend",
                    description=(
                        "Optional checks compare selected pure-fluid PR/SRK vapor "
                        "root Z, phi, H_dep, and S_dep against thermo.eos when "
                        "CHEMWORLD_RUN_REFERENCE_TESTS=1."
                    ),
                    status="optional",
                    reference_backend="thermo",
                    command_or_path=(
                        "CHEMWORLD_RUN_REFERENCE_TESTS=1 python -m pytest "
                        "tests/reference/test_optional_reference_backends.py"
                    ),
                    tolerance="1e-6 relative for reference backend comparisons",
                ),
            ),
            intended_use=(
                "Benchmark dense-gas and vapor-loss calculations.",
                "Future flash, distillation, and reactor energy-balance modules "
                "that need compact residual properties.",
            ),
        ),
        ModelCard(
            model_id="cubic_eos_volume_translation_root_governance",
            module_id="eos",
            title="Volume-Translated Cubic EOS And Root Governance Diagnostics",
            maturity=MaturityLevel.REFERENCE_VALIDATED,
            summary=(
                "Peneloux-style volume-translation reporting and root-governance "
                "diagnostics layered on the compact PR/SRK cubic EOS. The slice "
                "keeps translation shifts, binary-interaction provenance, root "
                "selection evidence, and translated molar volumes explicit in "
                "JSON-friendly reports."
            ),
            equations=(
                "c_mix=sum_i x_i c_i",
                "V_translated=V_eos-c_mix",
                "Z_translated=P V_translated/(R T)=Z_eos-c_mix P/(R T)",
                "Translated cubic roots use C=c_mix P/(R T) and admissibility Z>B-C.",
                "Stable-root diagnostics rank roots by mixture residual Gibbs score.",
            ),
            assumptions=(
                "Translation parameters are supplied by the caller or scenario card.",
                "Binary interaction parameters are local model inputs and can be "
                "required to carry explicit provenance records.",
                "Root-governance reports expose evidence rather than hiding root "
                "selection inside a solver side effect.",
            ),
            validity_limits=(
                "Only PR and SRK cubic EOS families are supported.",
                "Volume translation is a compact Peneloux-style molar-volume "
                "correction, not a broad density database or critical-region model.",
                "Translated fugacity-composition derivatives, saturation curves, "
                "and full phase-envelope algorithms remain future slices.",
            ),
            failure_modes=(
                "Translated molar volume <= 0 raises ValueError.",
                "Unknown or duplicate volume-translation component records raise ValueError.",
                "Missing or mismatched binary-interaction provenance can be required "
                "to raise ValueError before evaluation.",
                "No admissible translated cubic root raises ValueError.",
            ),
            units={
                "temperature": "K",
                "pressure": "Pa",
                "volume_translation_shift": "m^3/mol",
                "molar_volume": "m^3/mol",
                "compressibility_factor": "dimensionless",
                "binary_interaction": "dimensionless",
            },
            reference_reading=(
                "reference_repos/phasepy/phasepy/cubic/vtcubicpure.py and "
                "vtcubicmix.py expose translated cubic roots with C=cP/RT.",
                "reference_repos/thermopack/src/volume_shift.f90 applies "
                "Peneloux volume shifts to Z and tracks component c_i values.",
                "reference_repos/thermopack/src/cubic.f90 documents root flags "
                "including smallest, largest, and minimum-Gibbs root selection.",
                "reference_repos/thermo/thermo/eos_mix_methods.py separates "
                "translated b0, b, and c terms in PR/SRK translated lnphi paths.",
            ),
            validation_evidence=(
                ValidationEvidence(
                    evidence_id="volume-translation-root-governance-tests",
                    evidence_type="unit_tests",
                    description=(
                        "Default EOS tests cover pure-liquid volume translation, "
                        "translated cubic roots, vapor-root warnings, stable-root "
                        "diagnostic ranking, binary-interaction provenance, and "
                        "negative translated-volume failures."
                    ),
                    status="passing",
                    command_or_path="python -m pytest tests/test_eos.py",
                    tolerance="analytical consistency and pytest.approx checks",
                ),
            ),
            model_limit_notes=(
                "This slice deliberately does not vendor reference-library data tables.",
                "The compact report keeps residual/fugacity conventions visible; "
                "full translated fugacity derivatives are left for a later flash slice.",
            ),
            intended_use=(
                "Dense-fluid volume diagnostics in flash, distillation, and "
                "generalization tasks.",
                "Auditable scenario manifests with binary-interaction and "
                "volume-shift provenance.",
            ),
        ),
    )




__all__ = [
    "eos_model_cards",
]
