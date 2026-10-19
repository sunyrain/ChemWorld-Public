"""Public entrypoint for the ChemWorld physical-chemistry world environment."""

from chemworld.registration import ENV_ID, ENV_IDS, register_envs
from chemworld.validation import validate_action, validate_recipe
from chemworld.world.composition import (
    WORLD_COMPOSITION_SCHEMA_VERSION,
    CompiledWorldComposition,
    WorldCompatibilityReport,
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
)

__version__ = "0.5.0.dev0"
__all__ = [
    "ENV_ID",
    "ENV_IDS",
    "WORLD_COMPOSITION_COVERAGE_SCHEMA_VERSION",
    "WORLD_COMPOSITION_SCHEMA_VERSION",
    "CompiledWorldComposition",
    "CompositionCoverageSuite",
    "CompositionCoverageTarget",
    "ContinuousCoverageAxis",
    "DiscreteCoverageAxis",
    "GeneratedCompositionCase",
    "OrderedWorkflowTemplate",
    "WorldCompatibilityReport",
    "WorldCompositionCoverageError",
    "WorldCompositionDiagnostic",
    "WorldCompositionError",
    "WorldCompositionSpec",
    "__version__",
    "check_world_composition_compatibility",
    "compile_world_composition",
    "generate_world_composition_coverage",
    "register_envs",
    "validate_action",
    "validate_recipe",
]

register_envs()
