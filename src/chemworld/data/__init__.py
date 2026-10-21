"""Data schemas, trajectory logging, and validation helpers."""

from chemworld.data.logging import TrajectoryLogger, load_jsonl
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
    "OUTCOME_LAYER_FIELDS",
    "SUBMISSION_SCHEMA_VERSION",
    "SUPPORTED_TRAJECTORY_SCHEMA_VERSIONS",
    "TRAJECTORY_ALIAS_WRITE_REMOVAL_VERSION",
    "TRAJECTORY_COMPATIBILITY_ALIASES",
    "TRAJECTORY_SCHEMA_VERSION",
    "ChemSageDatabase",
    "ChemSageParseError",
    "SBMLModel",
    "SubmissionManifest",
    "TdbDatabase",
    "TdbParseError",
    "TrajectoryLogger",
    "TrajectoryRecordPayload",
    "load_jsonl",
    "validate_record",
    "validate_records",
]


def __getattr__(name):
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
