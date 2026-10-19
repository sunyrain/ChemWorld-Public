"""Professional world-law layer for the unified ChemWorld environment."""

from chemworld.world.composition import (
    SUPPORTED_COMPONENT_KINDS,
    WORLD_COMPOSITION_SCHEMA_VERSION,
    CompiledWorldComposition,
    CompositionTaskRequest,
    WorldCompatibilityReport,
    WorldComponentRequest,
    WorldCompositionDiagnostic,
    WorldCompositionError,
    WorldCompositionSpec,
    check_world_composition_compatibility,
    compile_world_composition,
)
from chemworld.world.composition_coverage import (
    WORLD_COMPOSITION_COVERAGE_SCHEMA_VERSION,
    CompositionCoverageSuite,
    CompositionCoverageTarget,
    ContinuousCoverageAxis,
    DiscreteCoverageAxis,
    GeneratedCompositionCase,
    OrderedWorkflowTemplate,
    WorldCompositionCoverageError,
    generate_world_composition_coverage,
    latin_hypercube_coordinates,
    pairwise_covering_rows,
)
from chemworld.world.continuous_flow import ContinuousFlowModuleSpec
from chemworld.world.crystallization import CrystallizationModuleSpec
from chemworld.world.distillation import DistillationModuleSpec
from chemworld.world.electrochemistry import ElectrochemistryModuleSpec
from chemworld.world.equilibrium_entity import (
    EQUILIBRIUM_ENTITY_PANEL_VERSION,
    EquilibriumEntityPanelIntervention,
    EquilibriumEntityProfileSpec,
    apply_equilibrium_entity_panel,
    equilibrium_entity_panel_hash,
)
from chemworld.world.instruments import InstrumentContract, instrument_contracts
from chemworld.world.material_counterfactual import (
    MATERIAL_LAW_COUNTERFACTUAL_VERSION,
    MaterialLawCounterfactual,
    apply_material_law_counterfactual,
    material_law_counterfactual_hash,
)
from chemworld.world.mechanism_family import (
    MECHANISM_FAMILY_INTERVENTION_VERSION,
    MECHANISM_REACHABLE_TASKS,
    MechanismFamilyIntervention,
)
from chemworld.world.parameters import (
    SUPPORTED_SPLITS,
    WORLD_FAMILY_VERSION,
    ChemWorldParameters,
    load_chemworld_parameters,
)
from chemworld.world.recipes import compile_recipe, expand_macro_action, validate_recipe
from chemworld.world.scenario import (
    ScenarioFamilySpec,
    ScenarioGenerator,
    ScenarioInstance,
    ScenarioSpec,
    get_scenario,
    get_scenario_card,
    list_scenarios,
)
from chemworld.world.spectra import (
    final_assay_spectra,
    gc_chromatogram,
    hplc_chromatogram,
    ir_spectrum,
    nmr_spectrum,
    uvvis_spectrum,
)
from chemworld.world.state_factory import initial_chemworld_state
from chemworld.world.world_family import (
    WORLD_AXIS_REGISTRY,
    AxisIntervention,
    WorldAxisSpec,
    axes_for_task,
)
from chemworld.world.world_law import (
    MODULE_VERSIONS,
    constitution_rules,
    world_law_spec,
)

__all__ = [
    "EQUILIBRIUM_ENTITY_PANEL_VERSION",
    "MATERIAL_LAW_COUNTERFACTUAL_VERSION",
    "MECHANISM_FAMILY_INTERVENTION_VERSION",
    "MECHANISM_REACHABLE_TASKS",
    "MODULE_VERSIONS",
    "SUPPORTED_COMPONENT_KINDS",
    "SUPPORTED_SPLITS",
    "WORLD_AXIS_REGISTRY",
    "WORLD_COMPOSITION_COVERAGE_SCHEMA_VERSION",
    "WORLD_COMPOSITION_SCHEMA_VERSION",
    "WORLD_FAMILY_VERSION",
    "AxisIntervention",
    "ChemWorldParameters",
    "CompiledWorldComposition",
    "CompositionCoverageSuite",
    "CompositionCoverageTarget",
    "CompositionTaskRequest",
    "ContinuousCoverageAxis",
    "ContinuousFlowModuleSpec",
    "CrystallizationModuleSpec",
    "DiscreteCoverageAxis",
    "DistillationModuleSpec",
    "ElectrochemistryModuleSpec",
    "EquilibriumEntityPanelIntervention",
    "EquilibriumEntityProfileSpec",
    "GeneratedCompositionCase",
    "InstrumentContract",
    "MaterialLawCounterfactual",
    "MechanismFamilyIntervention",
    "OrderedWorkflowTemplate",
    "ScenarioFamilySpec",
    "ScenarioGenerator",
    "ScenarioInstance",
    "ScenarioSpec",
    "WorldAxisSpec",
    "WorldCompatibilityReport",
    "WorldComponentRequest",
    "WorldCompositionCoverageError",
    "WorldCompositionDiagnostic",
    "WorldCompositionError",
    "WorldCompositionSpec",
    "apply_equilibrium_entity_panel",
    "apply_material_law_counterfactual",
    "axes_for_task",
    "check_world_composition_compatibility",
    "compile_recipe",
    "compile_world_composition",
    "constitution_rules",
    "equilibrium_entity_panel_hash",
    "expand_macro_action",
    "final_assay_spectra",
    "gc_chromatogram",
    "generate_world_composition_coverage",
    "get_scenario",
    "get_scenario_card",
    "hplc_chromatogram",
    "initial_chemworld_state",
    "instrument_contracts",
    "ir_spectrum",
    "latin_hypercube_coordinates",
    "list_scenarios",
    "load_chemworld_parameters",
    "material_law_counterfactual_hash",
    "nmr_spectrum",
    "pairwise_covering_rows",
    "uvvis_spectrum",
    "validate_recipe",
    "world_law_spec",
]
