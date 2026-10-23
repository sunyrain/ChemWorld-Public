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
    "tank_heads": ['HEAD_KINDS', 'TankHead'],
    "tank_geometry": (
        'TankGeometry TankCalibrationTable TankChebyshev cylindrical_wetted_area'
    ).split(),
    "solar_ephemeris": (
        'SolarParameters SolarPosition SolarEphemeris solar_delta_t_seconds'
    ).split(),
    "solar_events": ['SolarEvents', 'solar_events'],
    "solar_irradiance": ['SolarIrradiance', 'clear_sky_irradiance', 'extraterrestrial_irradiance'],
    "equipment_geometry": (
        'EquipmentGeometry solve_tube_pitch corrugated_plate_enlargement helical_coil_geometry '
        'plate_exchanger_geometry fin_exchanger_geometry air_cooled_geometry'
    ).split(),
    "geometric_primitives": (
        'cylinder_geometry circular_segment_area circular_segment_height '
        'partial_sphere_geometry shape_factor CoolingTowerGeometry'
    ).split(),
    "msis_atmosphere": ['MSISParameters', 'MSISAtmosphereState', 'MSISAtmosphere'],
    "standard_atmosphere": (
        'StandardAtmosphereState standard_atmosphere standard_air_properties standard_gravity '
        'standard_pressure_column AirColumnMass air_column_mass'
    ).split(),
    "fitting_curves": ['FittingCurve', 'FittingCurves'],
    "local_fittings": (
        'entrance_loss_coefficient transition_loss_coefficient staged_diffuser_loss'
    ).split(),
    "bend_fittings": ['bend_loss_coefficient', 'coil_loss_coefficient'],
    "valve_fittings": (
        'CRANE_VALVE_TYPES VALVE_LIFT_TYPES FittingCoefficientCatalog JunctionLossCoefficients '
        'crane_valve_coefficient minimum_valve_lift_velocity junction_loss_coefficients '
        'resistance_from_valve_coefficient valve_coefficient_from_resistance '
        'change_loss_velocity_basis darby_resistance hooper_resistance nominal_size_from_bore'
    ).split(),
    "engineering_numbers": (
        'ENGINEERING_GROUP_PARAMETERS engineering_group current_fluid_similarity '
        'thermal_diffusivity_si convert_viscosity engineering_resistance_conversion '
        'local_gravity ideal_gas_sound_speed'
    ).split(),
    "differential_meters": (
        'DifferentialMeterSpec DifferentialMeterResult LiquidMeterPressureResult '
        'liquid_differential_meter_pressure solve_differential_meter meter_discharge '
        'meter_flow_coefficient'
    ).split(),
    "meter_correlations": (
        'METER_TYPES METER_ALIASES meter_beta meter_expansibility miller_orifice_coefficient '
        'meter_discharge_coefficient meter_loss_coefficient meter_coefficient_from_loss '
        'wet_venturi_correction'
    ).split(),
    "meter_tables": ['MeterTableSurface'],
    "jet_pumps": (
        'LiquidJetPumpModel solve_jet_pump_inlet jet_pump_report '
        'solve_liquid_jet_pump vacuum_air_leakage'
    ).split(),
    "pump_engineering": (
        'corripio_pump_efficiency corripio_motor_efficiency pump_specific_speed '
        'pump_specific_diameter motor_synchronous_speed ideal_motor_current '
        'DriveEfficiencyResult DriveEfficiencyMap MotorEfficiencyCatalog MotorSizeCatalog '
        'MotorPartLoadCatalog ElectricalSupplyRecord ElectricalSupplyCatalog PumpDriveSpec'
    ).split(),
    "particle_distributions": (
        'ContinuousDiameterDistribution AnalyticDiameterDistribution CustomDiameterDistribution '
        'InterpolatedDiameterDistribution BinnedDiameterDistribution diameter_spacing'
    ).split(),
    "sieve_catalog": ['SieveSpecification', 'SieveCatalog'],
    "helmholtz": ['HelmholtzFluid', 'HelmholtzState'],
    "helmholtz_terms": ['HelmholtzTerm'],
    "helmholtz_thermal": ['HelmholtzIsochoricThermodynamics'],
    "helmholtz_equilibrium_thermal": ['HelmholtzEquilibriumIsochoricThermodynamics'],
    "helmholtz_processes": (
        'HelmholtzStream HelmholtzPressureStage HelmholtzPressureStageReport'
    ).split(),
    "helmholtz_flash": (
        'HelmholtzTwoPhaseState HelmholtzFlashReport coexistence_derivatives '
        'saturation_at_pressure solve_two_phase_state flash_at_pressure'
    ).split(),
    "helmholtz_solvers": (
        'HelmholtzAncillary HelmholtzPhaseModel HelmholtzSolveReport '
        'HelmholtzSaturationReport homogeneous_state_derivatives'
    ).split(),
    "flow_meters": (
        'LiquidOrificeSpec LiquidOrificeResult liquid_orifice_pressure liquid_orifice_mass_flow '
        'liquid_orifice_bore liquid_orifice_pressure_endpoint orifice_discharge_coefficient'
    ).split(),
    "gas_flow": (
        'IsothermalGasState IsothermalGasPipeResult isothermal_critical_pressure_ratio '
        'isothermal_gas_pipe solve_isothermal_gas_pipe'
    ).split(),
    "gas_flow_thermo": (
        'isothermal_gas_work_molar isentropic_gas_work_molar solve_isentropic_gas_work '
        'isentropic_gas_outlet_temperature gas_efficiency_conversion gas_polytropic_exponent '
        'critical_gas_pressure critical_gas_temperature is_critical_gas_flow '
        'stagnation_energy_J_kg stagnation_temperature_from_velocity gas_stagnation_pressure '
        'gas_stagnation_temperature GasNozzleResult ideal_gas_nozzle'
    ).split(),
    "relief_valves": (
        'API526_ORIFICES_IN2 API526_ORIFICES_M2 ReliefSizingResult '
        'api520_critical_coefficient api520_subcritical_factor api520_steam_pressure_factor '
        'api520_steam_superheat_factor api520_gas_backpressure_factor '
        'api520_liquid_backpressure_factor api520_viscosity_factor api526_select_orifice '
        'size_gas_relief_valve size_steam_relief_valve size_liquid_relief_valve '
        'api521_noise_curve api521_flare_noise vdi3732_flare_noise'
    ).split(),
    "gas_pipeline": (
        'GAS_PIPELINE_METHODS EmpiricalGasPipelineSpec EmpiricalGasPipelineResult '
        'solve_empirical_gas_pipeline'
    ).split(),
    "open_channel": (
        'WEIR_METHODS WeirFlowReport manning_velocity chezy_velocity manning_to_chezy '
        'chezy_to_manning uniform_channel_flow weir_flow manning_roughness_table '
        'select_manning_roughness'
    ).split(),
    "valve_characteristics": (
        'ValveOpeningCurve convert_valve_coefficient valve_cavitation_index '
        'gas_valve_choked liquid_valve_choked gas_valve_choke_pressure liquid_valve_choke_pressure'
    ).split(),
    "valve_noise": (
        'ValveAcousticGeometry ValveNoiseResult liquid_control_valve_noise gas_control_valve_noise'
    ).split(),
    "two_phase_holdup": (
        'TwoPhaseProperties VOIDAGE_REQUIREMENTS TWO_PHASE_VISCOSITY_METHODS '
        'available_voidage_methods available_two_phase_viscosity_methods martinelli_parameter '
        'two_phase_void_fraction two_phase_mixture_density void_fraction_from_density '
        'two_phase_mixture_viscosity'
    ).split(),
    "packed_beds": (
        'PACKED_BED_METHODS PackedBedSpec PackedBedPressureReport '
        'available_packed_bed_methods packed_bed_pressure packed_bed_voidage'
    ).split(),
    "friction_correlations": ['colebrook_friction_factor', 'pipe_friction_factor'],
    "straight_friction": (
        'STRAIGHT_FRICTION_RANGES available_straight_friction_methods '
        'straight_friction_correlation friction_transmission_factor crane_fully_turbulent_friction'
    ).split(),
    "curved_friction": (
        'HELICAL_TRANSITION_RANGES HELICAL_FRICTION_METHODS HelicalFrictionResult '
        'helical_transition_reynolds available_helical_friction_methods helical_friction_factor'
    ).split(),
    "plate_friction": ['PLATE_FRICTION_METHODS', 'plate_friction_factor'],
    "pneumatic_conveying": ['SALTATION_METHODS', 'SaltationResult', 'saltation_velocity'],
    "piping_catalog": ['PipeDimensions', 'PipeScheduleCatalog', 'WireGaugeSelection',
                       'WireGaugeCatalog', 'piping_erosional_velocity'],
    "packed_towers": ['TowerPackingSpec', 'TowerPressureResult', 'DemisterEfficiencyResult',
                      'mesh_voidage_from_mass', 'mesh_specific_area',
                      'demister_dry_pressure_drop', 'demister_wet_pressure_drop',
                      'demister_removal_efficiency', 'stichlmair_tower_pressure',
                      'stichlmair_flooding_velocity', 'robbins_tower_pressure_drop'],
    "particle_drag": ['PARTICLE_DRAG_RANGES', 'PARTICLE_ODE_POLICY', 'ParticleDragSpec',
                      'ParticleTerminalResult', 'ParticleMotionState', 'ParticleMotionResult',
                      'available_particle_drag_methods', 'particle_drag_coefficient',
                      'particle_terminal_velocity', 'time_to_stokes_terminal',
                      'advance_particle_motion'],
    "separator_capacity": ['WATKINS_METHODS', 'SeparatorCapacityCurve',
                           'SeparatorCapacityResult', 'separator_theoretical_capacity',
                           'separator_watkins_capacity', 'separator_demister_capacity',
                           'separator_gas_capacity'],
    "mixing_engineering": ['AgitatorMixingResult', 'agitator_mixing_time',
                           'ribbon_power_reynolds', 'ribbon_mixing_time', 'static_mixer_cov',
                           'static_mixer_loss_coefficient', 'StaticMixerResult',
                           'static_mixer_performance', 'TeeMixingResult', 'tee_mixing_cov'],
    "pipe_roughness": ['farshad_pipe_roughness', 'MaterialRoughnessResult',
                       'PipeRoughnessCatalog', 'smooth_pipe_observation_pairs'],
    "single_phase_pressure": ['single_phase_friction_pressure_drop',
                              'single_phase_acceleration_pressure_drop',
                              'single_phase_acceleration_gradient',
                              'single_phase_gravity_pressure_drop'],
    "flow_filters": (
        'FILTER_METHODS MESH_PATTERNS FilterCurve FilterPressureReport '
        'filter_loss_coefficient filter_pressure_drop'
    ).split(),
    "two_phase_pressure": (
        'TwoPhasePressureResult TWO_PHASE_PRESSURE_REQUIREMENTS TWO_PHASE_PRESSURE_GUIDANCE '
        'available_two_phase_pressure_methods two_phase_pressure_drop'
    ).split(),
    "two_phase_balances": (
        'two_phase_acceleration_pressure_drop two_phase_acceleration_gradient '
        'two_phase_gravity_gradient two_phase_gravity_pressure_drop'
    ).split(),
    "two_phase_regimes": (
        'TwoPhaseRegimeResult LogRegimeBoundary TaitelDuklerBoundaries '
        'mandhane_gregory_aziz_regime taitel_dukler_regime'
    ).split(),
    "gas_control_valves": (
        'GasControlValveSpec GasValveResult GasValveSizingResult size_gas_control_valve '
        'gas_valve_flow gas_valve_outlet_pressure gas_valve_piping_factors'
    ).split(),
    "fluids_devices": (
        'LiquidControlValveSpec LiquidValveResult LiquidValveSizingResult '
        'valve_critical_pressure_factor valve_piping_factors valve_reynolds valve_reynolds_factor '
        'liquid_valve_pressure_drop liquid_valve_flow size_liquid_control_valve'
    ).split(),
    "composition_math": (
        'normalized_fractions regularized_fractions fraction_basis convert_component_flows '
        'weighted_property combine_component_flows solve_stream_composition '
        'constrained_composition_gradient constrained_composition_hessian '
        'composition_gradient_to_moles partial_molar_from_mole_gradient '
        'partial_molar_properties partial_molar_composition_jacobian '
        'extensive_hessian_from_moles'
    ).split(),
    "fluid_response": (
        'mass_molar_property density_molar_volume velocity_basis gravity_scale '
        'watson_characterization vapor_mass_fraction parachor molecular_radius_of_gyration '
        'compressibility_factor isobaric_expansion isothermal_compressibility '
        'heat_capacity_difference joule_thomson acoustic_velocity isentropic_exponent '
        'isentropic_exponent_pv isentropic_exponent_pt isentropic_exponent_tv '
        'phase_identification_parameter phase_from_response'
    ).split(),
    "dippr": (
        'DIPPRTemperatureResponse DIPPRParameterJacobian DIPPRMixedDerivative '
        'dippr_temperature_response dippr_parameter_jacobian dippr_mixed_derivative '
        'dippr106_coefficients_from_point'
    ).split(),
    "property_selection": (
        'PropertySelectionPolicy PropertySelectionReport PropertySelectionError '
        'inspect_property_selection'
    ).split(),
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
    "virial_kernels": (
        'VIRIAL_METHODS TSONOPOULOS_CLASSES virial_value virial_derivatives virial_fast '
        'virial_vector virial_matrix tsonopoulos_polar_parameters meng_polar_parameter'
    ).split(),
    "virial_correlations": ['virial_correlation', 'virial_temperature_report'],
    "virial_thermodynamics": (
        'evaluate_virial_gas_state VirialGasThermodynamics virial_gas_batch_system'
    ).split(),
    "virial_mixtures": (
        'lee_kesler_cross_volumes tarakad_danner_interactions meng_duan_interactions '
        'virial_cross_parameters virial_coefficient_matrices VirialCoefficientResponse '
        'virial_mixture_response'
    ).split(),
    "virial_gas": (
        'virial_pressure pressure_virial_volume density_to_pressure_virial '
        'pressure_to_density_virial VirialGasState VirialRootError density_virial_state '
        'pressure_virial_state second_virial_from_compressibility '
        'compressibility_from_second_virial virial_volume_composition_response'
    ).split(),
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
        'compressed_liquid_conductivity_report chung_dense_conductivity '
        'stiel_thodos_dense_conductivity ely_hanley_dense_conductivity'
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
        'boiling_critical_temperature mersmann_kind_volume mixture_critical_property '
        'hekayati_raeissi_critical critical_compressibility'
    ).split(),
    "critical_properties": (
        'CriticalEstimate estimate_critical_property acentric_estimate_from_correlation '
        'eos_component_from_critical_inputs mixture_critical_estimate critical_correlation_input'
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
        'heat_capacity_report ideal_gas_molar_volume_report ideal_gas_transport_state '
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
    "activity_thermodynamics": (
        'ActivityInfiniteDilutionReport ActivityThermodynamicReport '
        'activity_infinite_dilution activity_parameter_jacobian activity_thermodynamic_report'
    ).split(),
    "regular_solution": ['regular_solution_model', 'hansen_activity_model'],
    "unifac": (
        'UNIFAC_VARIANTS UNIFACSubgroup unifac_activity_model unifac_rq unifac_psi'
    ).split(),
    "cubic_alpha": ['CubicAlphaResponse', 'cubic_alpha_response'],
    "ppr78": ["PPR78Spec", "ppr78_interaction_response"],
    "cubic_mixing": ["GibbsExcessMixingSpec"],
    "phase_evaluation": ["PhasePropertyModel", "PhaseEvaluation", "PhaseTransportSpec"],
    "group_contribution": (
        'GroupContributionTable GroupEstimateReport joback_group_values '
        'fedors_critical_volume wilson_jasperson_critical bondi_group_properties '
        'joback_estimate structural_group_estimate joback_property_correlations'
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
