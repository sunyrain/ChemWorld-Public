"""Data schemas, trajectory logging, and validation helpers."""

from chemworld.data.logging import TrajectoryLogger, load_jsonl
from chemworld.data.records import trajectory_value
from chemworld.data.schema import (
    OUTCOME_LAYER_FIELDS,
    SUPPORTED_TRAJECTORY_SCHEMA_VERSIONS,
    TRAJECTORY_ALIAS_WRITE_REMOVAL_VERSION,
    TRAJECTORY_COMPATIBILITY_ALIASES,
    TRAJECTORY_SCHEMA_VERSION,
    TrajectoryRecordPayload,
)
from chemworld.data.submission import SUBMISSION_SCHEMA_VERSION, SubmissionManifest
from chemworld.data.validation import validate_record, validate_records

__all__ = [
    "GAUSSIAN_KNOWN_SIGMA",
    "OUTCOME_LAYER_FIELDS",
    "SUBMISSION_SCHEMA_VERSION",
    "SUPPORTED_TRAJECTORY_SCHEMA_VERSIONS",
    "TRAJECTORY_ALIAS_WRITE_REMOVAL_VERSION",
    "TRAJECTORY_COMPATIBILITY_ALIASES",
    "TRAJECTORY_SCHEMA_VERSION",
    "ActivityCalibration",
    "ActivityFitParameter",
    "ChemSageDatabase",
    "ChemSageParseError",
    "DiameterDistributionCalibration",
    "MolecularGroupAssignment",
    "MultifluidSourceCatalog",
    "SBMLCalibration",
    "SBMLCalibrationParameter",
    "SBMLModel",
    "SubmissionManifest",
    "TdbDatabase",
    "TdbParseError",
    "TrajectoryLogger",
    "TrajectoryRecordPayload",
    "UNIFACInteractionRecord",
    "bind_incompressible_liquid",
    "incompressible_catalog",
    "incompressible_fluid",
    "iter_ddbst_sqlite",
    "iter_ddbst_text",
    "load_helmholtz_fluid",
    "load_helmholtz_phase_model",
    "load_incompressible_fluid",
    "load_jsonl",
    "load_multifluid_catalog",
    "load_multifluid_model",
    "read_helmholtz_fluid",
    "read_multifluid_catalog",
    "read_multifluid_model",
    "read_unifac_interactions",
    "read_unifac_subgroups",
    "sbml_profile_likelihood",
    "select_unifac_interactions",
    "trajectory_value",
    "twu_eos_component_from_record",
    "unifac_model_from_assignments",
    "validate_record",
    "validate_records",
]


def __getattr__(name):
    if name in {"sbml_profile_likelihood", "GAUSSIAN_KNOWN_SIGMA"}:
        from chemworld.data import sbml_profile

        value = getattr(sbml_profile, name)
        globals()[name] = value
        return value
    if name in {"SBMLCalibration", "SBMLCalibrationParameter"}:
        from chemworld.data import sbml_calibration

        value = getattr(sbml_calibration, name)
        globals()[name] = value
        return value
    if name in {"MultifluidSourceCatalog", "load_multifluid_catalog", "read_multifluid_catalog"}:
        from chemworld.data import multifluid_catalog

        value = getattr(multifluid_catalog, name)
        globals()[name] = value
        return value
    if name in {"load_multifluid_model", "read_multifluid_model"}:
        from chemworld.data import multifluid

        value = getattr(multifluid, name)
        globals()[name] = value
        return value
    if name == "bind_incompressible_liquid":
        from chemworld.data.incompressible_bindings import bind_incompressible_liquid

        globals()[name] = bind_incompressible_liquid
        return bind_incompressible_liquid
    if name in {"load_incompressible_fluid", "incompressible_catalog", "incompressible_fluid"}:
        from chemworld.data import incompressible

        value = getattr(incompressible, name)
        globals()[name] = value
        return value
    if name == "DiameterDistributionCalibration":
        from chemworld.data.particle_distribution_calibration import DiameterDistributionCalibration

        globals()[name] = DiameterDistributionCalibration
        return DiameterDistributionCalibration
    if name == "twu_eos_component_from_record":
        from chemworld.data.twu_eos import twu_eos_component_from_record

        globals()[name] = twu_eos_component_from_record
        return twu_eos_component_from_record
    if name in {
        "MolecularGroupAssignment", "UNIFACInteractionRecord",
        "iter_ddbst_sqlite", "iter_ddbst_text", "read_unifac_interactions",
        "read_unifac_subgroups", "select_unifac_interactions",
        "unifac_model_from_assignments",
    }:
        from chemworld.data import unifac_sources

        value = getattr(unifac_sources, name)
        globals()[name] = value
        return value
    if name in {"load_helmholtz_fluid", "read_helmholtz_fluid"}:
        from chemworld.data import helmholtz

        value = getattr(helmholtz, name)
        globals()[name] = value
        return value
    if name == "load_helmholtz_phase_model":
        from chemworld.data.helmholtz_phase import load_helmholtz_phase_model

        globals()[name] = load_helmholtz_phase_model
        return load_helmholtz_phase_model
    if name in {"ActivityCalibration", "ActivityFitParameter"}:
        from chemworld.data import activity_calibration

        value = getattr(activity_calibration, name)
        globals()[name] = value
        return value
    if name == "SBMLModel":
        from chemworld.data.sbml import SBMLModel

        globals()[name] = SBMLModel
        return SBMLModel
    if name in {"ChemSageDatabase", "ChemSageParseError"}:
        from chemworld.data import chem_sage

        value = getattr(chem_sage, name)
        globals()[name] = value
        return value
    if name in {"TdbDatabase", "TdbParseError"}:
        from chemworld.data import tdb

        value = getattr(tdb, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
