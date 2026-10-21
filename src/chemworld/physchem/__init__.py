# ruff: noqa: SIM905 -- compact declarative lazy-export table
"""Lazy public facade for ChemWorld physical-chemistry primitives.

Concrete implementations live in focused submodules. Public symbols are
resolved on first access so importing :mod:`chemworld.physchem` does not
eagerly initialize every model family and optional reference dependency.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_MODULE_EXPORTS: dict[str, list[str]] = {
    "tabulated_functions": 'TabulatedScalar'.split(),
    "battery_thevenin": 'TheveninSpec thevenin_system thevenin_fields'.split(),
    "battery_mpm": 'BatteryMPMSpec mpm_fields'.split(),
    "battery_particles": 'ParticleSizeDistribution lognormal_size_distribution'.split(),
    "battery_soh": (
        'ElectrodeSOHSpec HalfCellSOHSpec SOHResult solve_electrode_soh soh_spec_from_dict '
        'soh_from_battery battery_spec_at_soc soh_parameter_response'
    ).split(),
    "root_isolation": 'IsolationPolicy IsolatedRoots isolate_roots'.split(),
    "combustion": (
        'CombustionData combustion_stoichiometry combustion_products_mixture combustion_data '
        'combustion_enthalpy heating_value_convert heating_value_mass combustion_reference_heat '
        'octane_index ignition_delay_quality'
    ).split(),
    "combustion_streams": (
        'air_fuel_ratio_solver combustion_spec_solver mixture_combustion_spec'
    ).split(),
    "safety_environment": (
        'ppmv_to_mass_concentration mass_concentration_to_ppmv suzuki_flammability_limit '
        'crowl_louvar_flammability_limit flammability_estimate fire_mixture_limit '
        'nfpa_30_classification emission_impact octanol_water_partition '
        'gas_exposure_comparison source_flammable_mixture iso_10156_lower_limit '
        'ISO_10156_DILUENT_COEFFICIENTS'
    ).split(),
    "electro_optics": (
        'relative_permittivity_crc water_permittivity_iapws water_refractive_index_iapws '
        'refractive_index_tde molar_refractivity refractive_index_from_molar_refractivity '
        'polarizability_volume brix_to_refractive_index refractive_index_to_brix '
        'brix_reading parallel_plate_capacitance'
    ).split(),
    "henry": (
        'HENRY_SCALES henry_convert henry_pressure henry_pressure_derivatives '
        'henry_log_mixture henry_pressure_mixture henry_matrix henry_raoult_flash'
    ).split(),
    "solubility": (
        'HANSEN_SOURCES HansenParameters hansen_distance solid_solubility '
        'freezing_point_depression hildebrand_parameter'
    ).split(),
    "interface_kernels": ('INTERFACE_METHODS interface_value interface_derivatives').split(),
    "volume_kernels": (
        'VOLUME_METHODS volume_value volume_derivatives volume_parameter_jacobian '
        'volume_pressure_derivatives'
    ).split(),
    "volume_correlations": ['volume_correlation'],
    "volume_mixtures": (
        'MIXTURE_VOLUME_METHODS mixture_volume_value mixture_volume_derivatives '
        'mixture_volume_composition_jacobian mixture_volume_evaluation mixture_volume_correlation'
    ).split(),
    "interfaces": (
        'winterfeld_scriven_davis diguilio_teja weinaug_katz water_oil_interfacial_tension '
        'mixture_surface_tension parachor_surface_tension laplace_pressure capillary_force'
    ).split(),
    "conductivity_mixtures": (
        'dippr9h_conductivity dippr9i_conductivity filippov_conductivity '
        'wassiljewa_herning_zipperer_conductivity lindsay_bromley_conductivity '
        'mixture_conductivity_ledger'
    ).split(),
    "dense_conductivity": (
        'compressed_liquid_conductivity compressed_liquid_conductivity_derivatives '
        'compressed_liquid_conductivity_report chung_dense_conductivity'
    ).split(),
    "conductivity_kernels": (
        'CONDUCTIVITY_METHODS conductivity_value conductivity_derivatives'
    ).split(),
    "petroleum_viscosity": (
        'kinematic_viscosity dynamic_viscosity twu_1985_kinematic_viscosity '
        'twu_1985_viscosity viscosity_index'
    ).split(),
    "viscosity_scales": ('VISCOSITY_SCALES viscosity_converter').split(),
    "dense_viscosity": (
        'lucas_liquid_viscosity lucas_liquid_viscosity_report przedziecki_sridhar_viscosity '
        'przedziecki_sridhar_temperature_viscosity'
    ).split(),
    "water_transport": (
        'water_viscosity_iapws water_reference_compressibility water_correlation_length_nm '
        'water_viscosity_enhancement water_conductivity_iapws'
    ).split(),
    "air_transport": ['air_conductivity_lemmon'],
    "viscosity_mixtures": (
        'wilke_prefactors wilke_viscosity herning_zipperer_viscosity '
        'brokaw_viscosity lorentz_bray_clarke_viscosity '
        'gas_mixture_viscosity_ledger dense_mixture_viscosity_ledger'
    ).split(),
    "viscosity_kernels": (
        'VISCOSITY_METHODS viscosity_value viscosity_derivatives '
        'viscosity_yaws_parameter_jacobian'
    ).split(),
    "calphad_material": ['CefMaterialSpec', 'CefMaterialState'],
    "mqmqa": ['MQMQAPhaseModel'],
    "lennard_jones": (
        'LennardJonesSpec COLLISION_PAIRS EPSILON_METHODS DIAMETER_METHODS '
        'lennard_jones_energy lennard_jones_diameter lennard_jones_reduced_temperature '
        'lennard_jones_collision_integral lennard_jones_viscosity_report '
        'lennard_jones_diffusivity_report'
    ).split(),
    "critical_kernels": (
        'BOILING_CRITICAL_FITS MERSMANN_KIND_RADII_ANGSTROM '
        'MERSMANN_KIND_REGRESSED_RADII_ANGSTROM acentric_factor_from_pressure '
        'lee_kesler_acentric_factor stiel_polar_factor critical_surface '
        'boiling_critical_temperature mersmann_kind_volume mixture_critical_property'
    ).split(),
    "critical_properties": (
        'CriticalEstimate estimate_critical_property acentric_estimate_from_correlation '
        'eos_component_from_critical_inputs mixture_critical_estimate'
    ).split(),
    "vapor_pressure": (
        'VaporPressureJacobian vapor_pressure_parameter_jacobian '
        'antoine_coefficients_from_point dippr101_coefficients_from_point '
        'arrhenius_pressure_extrapolation'
    ).split(),
    "water_saturation": (
        'water_saturation_pressure water_saturation_temperature'
    ).split(),
    "stoichiometry": (
        'stoichiometric_matrix balance_stoichiometry standard_formation_reaction '
        'stoichiometry_molar_to_mass stoichiometry_mass_to_molar stoichiometry_mass_error'
    ).split(),
    "heat_capacity": (
        'PiecewiseHeatCapacity temperature_from_caloric_change'.split()
    ),
    "heat_capacity_kernels": (
        'vibrational_temperature'.split()
    ),
    "finite_volume": (
        'ConservativeDiffusion DiffusionBoundary DiffusionCoupling DiffusionMesh DiffusionRates'
    ).split(),
    "component_registry": (
        'COMPONENT_REGISTRY_SCHEMA_VERSION ComponentIdentityRegistry '
        'curated_component_registry'
    ).split(),
    "data_conflicts": (
        'DATA_PROVENANCE_CARD_SCHEMA_VERSION ComponentDataConflictReport '
        'DataConflictFinding DataSourceProvenance DatasetProvenanceCard '
        'audit_component_data_conflicts build_dataset_provenance_card'
    ).split(),
    "electrochemistry": (
        'FARADAY_C_PER_MOL ElectrodeReactionSpec ElectrolysisResult '
        'ElectrolyteResistanceSpec OhmicDropResult butler_volmer_current '
        'electrolyte_resistance_ohm equilibrium_potential_from_delta_g faradaic_extent_mol '
        'nernst_potential ohmic_drop reaction_quotient_from_activities '
        'reaction_quotient_log_from_activities run_electrolysis'
    ).split(),
    "mechanism_library": (
        'MECHANISM_SCENARIO_SCHEMA_VERSION MechanismLibraryValidationReport '
        'MechanismScenarioCard get_mechanism_card list_mechanism_cards list_mechanism_paths '
        'load_library_mechanism mechanism_library_root mechanism_scenario_library_path '
        'repository_root validate_mechanism_library'
    ).split(),
    "reaction_network": (
        'R_J_PER_MOL_K BatchIntegrationResult RateLawSpec ReactionNetworkSpec '
        'ReactionODEReferenceCase ReactionODEReferenceResult ReactionSensitivityEntry '
        'ReactionSensitivityReport ReactionSpec SpeciesSpec '
        'ThermochemicalDetailedBalanceResult cantera_comparable_reaction_cases '
        'effective_third_body_concentration evaluate_rate_law '
        'evaluate_reaction_ode_reference_case falloff_reduced_pressure '
        'finite_difference_reaction_sensitivities integrate_reaction_ode_reference_case '
        'kinetic_sensitivity_parameter_candidates lindemann_falloff_rate_constant '
        'load_mechanism parse_reaction_equation perturb_network_parameters '
        'prefixed_arrhenius_params reverse_rate_constant_from_equilibrium '
        'thermochemical_concentration_equilibrium_constant thermochemical_detailed_balance '
        'third_body_efficiencies troe_broadening_factor troe_falloff_rate_constant'
    ).split(),
    "properties": (
        'CorrelationIntegral caloric_correlation_change correlation_temperature_derivative '
        'integrate_correlation '
        'CaloricReference PropertyPackageThermodynamics SpeciesCaloricModel StatePropertyReport '
        'caloric_polynomial_change caloric_temperature evaluate_caloric_correlation '
        'evaluate_property_state '
        'STANDARD_PRESSURE_PA ComponentPropertyPackage MixtureEnthalpyLedger '
        'MixtureTransportLedger MixtureVolumeLedger MolarVolumeReport PhaseEnthalpyReport '
        'PhaseTransitionSpec PropertyEvaluation PureSaturationReport '
        'TransportPropertyReport VaporPressureReport binary_gas_diffusivity_fuller_report '
        'density_to_molar_volume_m3_mol evaluate_correlation '
        'gas_mixture_effective_diffusivity_ledger gas_thermal_conductivity_dippr9b_report '
        'heat_capacity_report ideal_gas_molar_volume_report '
        'liquid_mixture_thermal_conductivity_dippr9h_ledger mixture_density '
        'mixture_enthalpy_ledger mixture_molar_volume_ledger mixture_viscosity_log_rule '
        'molar_volume_report molar_volume_to_density_kg_m3 normal_boiling_point_report '
        'phase_path_enthalpy_report phase_sensible_enthalpy_report '
        'phase_transition_enthalpy pure_saturation_pressure_report '
        'pure_saturation_temperature_report second_virial_coefficient_report '
        'sensible_enthalpy_change thermal_diffusivity_report thermal_hazard_proxy '
        'transport_property_report vapor_pressure_report '
        'vapor_pressure_temperature_derivative virial_gas_molar_volume_report '
        'volatility_risk_from_psat wilke_gas_mixture_viscosity_ledger'
    ).split(),
    "equilibrium_chemistry": (
        'AcidBaseResult ChargeBalanceResult EquilibriumReactionSpec EquilibriumResult '
        'EquilibriumSystemSpec GibbsMinimizationDiagnostic GibbsMinimizationResult '
        'GibbsMinimizationSpec GibbsSpeciesSpec PHObservationResult PrecipitationHookResult '
        'PrecipitationResult SolubilityProductSpec apply_precipitation_hooks '
        'aqueous_ph_observation balance_charge_by_adjusting_ion diagnose_gibbs_minimization '
        'equilibrium_constant_vant_hoff ionic_strength ionic_strength_from_amounts '
        'net_charge_equivalents precipitate_if_supersaturated reaction_extent_bounds '
        'reaction_quotient reaction_quotient_log solid_solubility_mole_fraction '
        'solve_gibbs_minimization solve_mass_action_equilibrium solve_monoprotic_acid_base '
        'solve_reaction_extent water_ion_product'
    ).split(),
    "equilibrium": (
        'ActivityModelSpec AzeotropeScanPoint AzeotropeScanStatus '
        'BinaryAzeotropeDiagnosticReport FlashResult GammaPhiKValueReport '
        'LLEPhaseStabilityDiagnostic LLEStageResult LiquidEquilibriumResult LiquidStabilityReport '
        'RachfordRiceDiagnosticReport '
        'UNIQUACActivityReport VLETemperatureReport activity_coefficients '
        'binary_azeotrope_diagnostic_report bubble_pressure_pa bubble_temperature_report '
        'dew_pressure_pa dew_temperature_report entrain_liquid_amounts equilibrate_liquids '
        'flash_isothermal gamma_phi_k_value_report liquid_tpd_minimum '
        'liquid_liquid_split lle_phase_stability_diagnostic rachford_rice_diagnostic_report '
        'rachford_rice_vapor_fraction raoult_k_values uniquac_activity_report'
    ).split(),
    "phase_equilibrium_units": ['StrictLiquidSystemSpec'],
    "reactors": (
        'BatchReactorModel CSTRFlowProgram CSTRModel CSTRMultiplicityResult '
        'CSTRMultiplicitySpec CSTRSteadyStatePoint DynamicBatchReactorModel FeedStreamSpec '
        'HeatTransferSpec JacketTemperatureProgram PFRGeometrySpec PFRModel ReactorResult '
        'ReactorState SamplingEventSpec SemiBatchFeedSpec SemiBatchReactorModel '
        'cstr_multiple_steady_state_reference_case solve_cstr_multiple_steady_states'
    ).split(),
    "coupled_system": (
        'ConstantCvThermodynamics CoupledAdvanceResult CoupledEvent CoupledProcess '
        'CoupledRates CoupledState CoupledStateView CoupledSystem CoupledThermodynamics '
        'EnergyPort HeatFlux InternalTransfer MaterialPort ProcessContribution '
        'SpeciesInternalEnergy '
        'assemble_coupled_rates'
    ).split(),
    "surface_specs": (
        'DEFAULT_SURFACE_MECHANISM SurfaceMechanismSpec bulk_co_oxidation_network '
        'make_surface_slot surface_species surface_species_energies'
    ).split(),
    "surface_rates": (
        'SurfaceRates adsorbate_key gas_amounts_from_pressure make_surface_system '
        'simulate_surface surface_conservation_laws surface_inventory surface_process '
        'evaluate_surface_rates surface_state surface_thermodynamics updated_surface_slot'
    ).split(),
    "spectroscopy": (
        'BeerLambertBandSpec BeerLambertCalibrationResult CalibrationCurve '
        'ChromatographyCalibrationResult ChromatographyMethodSpec IRBandAssignmentReport '
        'IRFunctionalGroupBandSpec InstrumentSignalSpec SpectralFeatureSpec '
        'SpectralMeasurement assign_ir_functional_group_bands beer_lambert_absorbance '
        'build_signal_spec build_signal_spec_from_card chromatographic_baseline_peak_width '
        'chromatographic_resolution chromatographic_retention_factor '
        'chromatographic_retention_time chromatographic_theoretical_plates '
        'default_feature_specs detect_peak_overlap fit_beer_lambert_calibration '
        'fit_chromatography_calibration generate_beer_lambert_calibration synthesize_signal '
        'synthesize_signal_from_card'
    ).split(),
    "eos": (
        'BinaryInteractionProvenance CubicEOSSpec CubicPureParameters '
        'CubicResidualProperties EOSComponentSpec EOSMixtureParameters EOSState '
        'RootCandidateDiagnostic RootGovernanceReport VolumeTranslationReport '
        'VolumeTranslationSpec binary_interaction_provenance_map cubic_ab '
        'cubic_compressibility_roots cubic_departure_log_argument '
        'cubic_fugacity_coefficients cubic_mixture_parameters cubic_pure_parameters '
        'cubic_residual_properties cubic_root_governance_report cubic_root_selection_policy '
        'evaluate_cubic_eos evaluate_volume_translated_cubic_eos ideal_gas_molar_volume '
        'ideal_gas_pressure ideal_gas_state mixture_volume_translation select_cubic_root '
        'cubic_state_derivatives '
        'translated_cubic_compressibility_roots validate_binary_interaction_provenance'
    ).split(),
    "chromatography_methods": (
        'ChromatographyMethodReport DetectorResponseCalibrationResult '
        'EmpiricalChromatographyAnalyteSpec evaluate_chromatography_method '
        'fit_detector_response_calibration gc_linear_retention_index peak_shape_status'
    ).split(),
    "specs": (
        'ComponentConflictPolicy ComponentConflictResolution ComponentFieldCandidate '
        'ComponentProvenance ComponentSpec ComponentUncertainty MixtureSpec '
        'PropertyCorrelation component_alias_index mass_fractions_from_mole_fractions '
        'mole_fractions_from_mass_fractions normalize_component_token '
        'property_equation_contracts resolve_component_field_conflict '
        'resolve_component_identifier supported_property_equations'
    ).split(),
    "electrochem_control": (
        'ControlOperationLog ControlTracePoint ElectrochemicalControlExecution '
        'ElectrochemicalControlLimits ElectrochemicalControlRecipe '
        'ElectrochemicalControlSegment execute_electrochemical_control_recipe '
        'verify_electrochemical_control_replay'
    ).split(),
    "crystallization_units": (
        'CoolingCrystallizationResult CrystalSizeDistribution CrystallizationKineticsSpec '
        'CrystallizationStepReport SolubilityCurveSpec cooling_crystallization'
    ).split(),
    "curated_properties": (
        'CuratedPropertyCase curated_components curated_property_case_map '
        'curated_property_cases curated_property_correlations curated_property_package '
        'list_curated_property_packages'
    ).split(),
    "electrochem_transport": (
        'DiffusionLayerSpec MassTransferLimitedCurrentResult '
        'diffusion_layer_current_response'
    ).split(),
    "extraction_units": (
        'DistributionCoefficientModelSpec ExtractionStageReport ExtractionTrainResult '
        'activity_corrected_extraction_train'
    ).split(),
    "electrochem_double_layer": (
        'DoubleLayerRCSpec DoubleLayerTracePoint DoubleLayerTransientResult '
        'simulate_double_layer_current_step simulate_double_layer_potential_step'
    ).split(),
    "electrochemical_scenarios": (
        'ElectrochemicalHiddenParameters ElectrochemicalModelBundle '
        'ElectrochemicalScenarioCard ElectrochemicalScenarioInstance HiddenParameterRange '
        'RedoxMetadata SideReactionAssessment assess_side_reaction_thresholds '
        'electrochemical_scenario_cards generate_electrochemical_scenario'
    ).split(),
    "elements": (
        'ElementSpec atom_fractions element_matrix hill_formula mass_fractions_from_formula '
        'molecular_weight parse_formula formula_charge serialize_formula get_element get_isotope '
        'mixture_atomic_composition similarity_variable index_hydrogen_deficiency'
    ).split(),
    "atomic_data": (
        'ATOMIC_DATA_VERSION AtomicEstimate ElementData IsotopeData atomic_data_provenance '
        'element_data isotope_data'
    ).split(),
    "identifiers": (
        'validate_cas_number check_cas_number cas_to_int int_to_cas sorted_cas_key'
    ).split(),
    "equipment_specs": (
        'EquipmentCardSpec EquipmentConstraintCheck EquipmentConstraintReport '
        'EquipmentConstraintSpec column_equipment_card condenser_equipment_card '
        'evaluate_equipment_constraints heat_exchanger_equipment_card mixer_equipment_card '
        'pump_equipment_card vessel_equipment_card'
    ).split(),
    "heat_transfer_units": (
        'EquipmentHeatTransferResult FoulingEvolutionSpec HeatTransferEquipmentSpec '
        'PhaseChangeBoundarySpec equipment_heat_transfer'
    ).split(),
    "separations": (
        'FUGDistillationReport FUGDistillationSpec SeparationLedger SeparationResult '
        'crystallize downstream_score dry_solid evaporation_flash fenske_minimum_stages '
        'fenske_underwood_gilliland_sizing filter_cake gilliland_eduljee_stage_estimate '
        'liquid_liquid_extraction underwood_minimum_reflux_binary vle_shortcut_distillation'
    ).split(),
    "transport": (
        'FlowResult FluidState FrictionFactorResult HeatExchangerResult HeatTransferResult '
        'NusseltCorrelationResult PackedBedResult PipeSpec TwoPhasePressureDropResult '
        'counterflow_effectiveness darcy_friction_factor darcy_friction_factor_details '
        'flow_regime heat_exchanger_counterflow homogeneous_two_phase_pressure_drop '
        'internal_heat_transfer_coefficient jacket_heat_transfer mixing_power '
        'nusselt_internal_flow nusselt_internal_flow_details '
        'overall_heat_transfer_coefficient packed_bed_pressure_drop_ergun '
        'peclet_heat_number pipe_pressure_drop prandtl_number pump_work reynolds_number'
    ).split(),
    "mass_spectrometry": (
        'FragmentIonResult FragmentIonSpec IsotopeEnvelopePeak MassSpectrumAnalyteSpec '
        'MassSpectrumReport isotope_envelope simulate_mass_spectrum'
    ).split(),
    "two_phase_flow": (
        'LockhartMartinelliPressureDropResult lockhart_martinelli_pressure_drop'
    ).split(),
    "maturity": (
        'MaturityLevel ModelCard ModelCardTemplate ModuleMaturity TaskMaturitySpec '
        'ValidationEvidence model_card_template_map model_card_templates '
        'validate_model_card validate_task_maturity_policy'
    ).split(),
    "thermochemistry": (
        'NASA7SpeciesThermo NASA7TemperatureSegment ReactionThermoResult SpeciesThermoState '
        'equilibrium_constant_from_delta_g reaction_thermochemistry '
        'CorrelationSpeciesThermo formation_entropy formation_gibbs_energy '
        'formation_enthalpy_basis formation_reference_phase_change'
    ).split(),
    "phase_change_kernels": (
        'boiling_vaporization_enthalpy watson_exponent'
    ).split(),
    "nmr": (
        'NMRStickLine ProtonNMRMethodSpec ProtonNMRReport ProtonNMRSignalResult '
        'ProtonNMRSignalSpec simulate_proton_nmr'
    ).split(),
    "reference_validation": (
        'ReferenceBackendSpec ReferenceBackendStatus ReferenceComparison '
        'ReferenceToleranceProfile ReferenceValidationReport compare_scalar '
        'import_reference_module reference_backend_context reference_backend_specs '
        'reference_backend_status reference_repo_paths reference_repo_roots '
        'reference_repos_root reference_tolerance_profiles reference_validation_report '
        'skipped_reference_backends summarize_reference_comparisons '
        'write_reference_validation_report'
    ).split(),
    "safety_envelope": (
        'RunawayStateInput SafetyEnvelopeAssessment SafetyEnvelopeSpec '
        'arrhenius_heat_generation_slope assess_safety_envelope'
    ).split(),
    "flash_units": (
        'FlashSystemSpec TPFlashUnitResult VLEPropertyPackage ph_flash tp_flash '
        'tp_flash_with_energy_balance'
    ).split(),
    "equilibrium_cards": (
        'activity_model_cards'
    ).split(),
    "curated_property_cards": (
        'curated_property_model_cards'
    ).split(),
    "data_foundation_cards": (
        'data_foundation_model_cards'
    ).split(),
    "electrochemistry_cards": (
        'electrochemistry_model_cards'
    ).split(),
    "eos_cards": (
        'eos_model_cards'
    ).split(),
    "equilibrium_chemistry_cards": (
        'equilibrium_chemistry_model_cards'
    ).split(),
    "equipment_spec_cards": (
        'equipment_spec_model_cards'
    ).split(),
    "flash_cards": (
        'flash_unit_model_cards'
    ).split(),
    "property_cards": (
        'property_correlation_model_cards'
    ).split(),
    "reaction_network_cards": (
        'reaction_kinetics_model_cards'
    ).split(),
    "reactor_cards": (
        'reactor_model_cards'
    ).split(),
    "safety_cards": (
        'safety_envelope_model_cards'
    ).split(),
    "separation_cards": (
        'separation_model_cards'
    ).split(),
    "spectroscopy_cards": (
        'spectroscopy_model_cards'
    ).split(),
    "thermochemistry_cards": (
        'thermochemistry_model_cards'
    ).split(),
    "transport_cards": (
        'transport_model_cards'
    ).split(),
    "temperature_scales": (
        'TEMPERATURE_SCALES temperature_scale_bounds temperature_scale_conversion '
        'temperature_convert its90_68_difference temperature_quantity '
        'temperature_input temperature_observation'
    ).split(),
}
_EXPORTS = {
    name: module_name
    for module_name, names in _MODULE_EXPORTS.items()
    for name in names
}
__all__ = tuple(_EXPORTS)


def __getattr__(name: str) -> Any:
    """Load one public symbol from its owning module and cache it locally."""

    try:
        module_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    module = import_module(f"{__name__}.{module_name}")
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Expose lazy public names to interactive discovery."""

    return sorted(set(globals()) | set(__all__))
