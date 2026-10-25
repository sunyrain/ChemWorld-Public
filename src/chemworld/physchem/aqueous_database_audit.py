"""Coverage and provenance audits for the native aqueous database contract.

The audit is intentionally read-only.  It reports what a database declares,
which source and domain contracts are actually referenced, and the explicit
boundaries that remain outside the native Reaktoro-aligned consumer.  It does
not infer missing parameters or widen a model's temperature, pressure, or
ionic-strength domain.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from chemworld.physchem.aqueous_database import AqueousDatabase
from chemworld.physchem.electrolyte_activity import electrolyte_activity_catalog

AQUEOUS_DATABASE_AUDIT_SCHEMA_VERSION = "chemworld-aqueous-database-audit-v1"


def _used_source_ids(database: AqueousDatabase) -> set[str]:
    used = {
        *database.activity_model.source_ids(),
        *(item.source_id for item in database.species),
        *(item.standard_state.source_id for item in database.species),
        *(item.source_id for item in database.reactions),
    }
    for species in database.species:
        thermo = species.standard_state.thermo
        if species.standard_state.pressure_model == "aqueous-gibbs":
            used.update(
                {
                    thermo.source_id,
                    thermo.reference_basis.source_id,
                }
            )
    return used


def _source_usage(database: AqueousDatabase) -> dict[str, dict[str, list[str]]]:
    """Return the read-only source-to-consumer provenance graph."""

    fields = (
        "activity_models",
        "species",
        "standard_states",
        "thermochemistry",
        "reference_bases",
        "reactions",
    )
    usage = {
        source.source_id: {field: [] for field in fields}
        for source in database.sources
    }

    def record(source_id: str, field: str, value: str) -> None:
        # Preserve undeclared references in a diagnostic report for malformed
        # or deliberately replaced development objects.
        usage.setdefault(source_id, {name: [] for name in fields})[field].append(value)

    for source_id in database.activity_model.source_ids():
        record(source_id, "activity_models", database.activity_model.model_id)
    for species in database.species:
        record(species.source_id, "species", species.species_id)
        record(species.standard_state.source_id, "standard_states", species.species_id)
        thermo = species.standard_state.thermo
        if species.standard_state.pressure_model == "aqueous-gibbs":
            record(thermo.source_id, "thermochemistry", species.species_id)
            record(thermo.reference_basis.source_id, "reference_bases", species.species_id)
    for reaction in database.reactions:
        record(reaction.source_id, "reactions", reaction.reaction_id)
    return {
        source_id: {field: sorted(values) for field, values in categories.items()}
        for source_id, categories in sorted(usage.items())
    }


def _finite_domain(value: Any, label: str) -> list[float] | None:
    domain = getattr(value, label, None)
    if domain is None:
        return None
    if len(domain) != 2:
        return None
    return [float(domain[0]), float(domain[1])]


def audit_aqueous_database(database: AqueousDatabase) -> dict[str, Any]:
    """Return a JSON-safe coverage, provenance, and domain inventory."""

    if not isinstance(database, AqueousDatabase):
        raise TypeError("audit_aqueous_database expects AqueousDatabase")
    source_ids = {item.source_id for item in database.sources}
    used_sources = _used_source_ids(database)
    phases = Counter(item.phase for item in database.species)
    standard_models = Counter(item.standard_state.pressure_model for item in database.species)
    temperature_domains = [
        domain
        for item in database.species
        if (domain := _finite_domain(item.standard_state.thermo, "temperature_bounds_K"))
        is not None
    ]
    pressure_domains = [
        list(item.standard_state.pressure_bounds_Pa)
        for item in database.species
    ]
    missing_activity_parameters = list(database.activity_model.missing_parameters())
    redistribution = Counter(item.redistribution for item in database.sources)
    provenance = _source_usage(database)
    all_have_license_and_citation = all(
        item.citation.strip()
        and item.source_url.strip()
        and item.license_id.strip()
        and item.license_url.strip()
        for item in database.sources
    )
    return {
        "schema_version": AQUEOUS_DATABASE_AUDIT_SCHEMA_VERSION,
        "database_id": database.database_id,
        "valid": (
            not missing_activity_parameters
            and not (used_sources - source_ids)
            and all_have_license_and_citation
        ),
        "counts": {
            "sources": len(database.sources),
            "species": len(database.species),
            "reactions": len(database.reactions),
            "phases": dict(sorted(phases.items())),
        },
        "activity_model": {
            "model_id": database.activity_model.model_id,
            "family": database.activity_model.family,
            "species": len(database.activity_model.charges),
            "maximum_ionic_strength_mol_kg": database.activity_model.maximum_ionic_strength_mol_kg,
            "temperature_bounds_K": list(database.activity_model.temperature_bounds_K),
            "pressure_bounds_Pa": list(database.activity_model.pressure_bounds_Pa),
            "missing_parameters": missing_activity_parameters,
        },
        "standard_states": {
            "pressure_models": dict(sorted(standard_models.items())),
            "temperature_domains_K": temperature_domains,
            "pressure_domains_Pa": pressure_domains,
            "reference_basis_ids": sorted(
                {item.standard_state.reference_basis_id for item in database.species}
            ),
        },
        "sources": {
            "declared_ids": sorted(source_ids),
            "used_ids": sorted(used_sources),
            "unreferenced_ids": sorted(source_ids - used_sources),
            "undeclared_used_ids": sorted(used_sources - source_ids),
            "redistribution": dict(sorted(redistribution.items())),
            "all_have_license_and_citation": all_have_license_and_citation,
            "records": [
                {
                    "source_id": source.source_id,
                    "citation": source.citation,
                    "source_url": source.source_url,
                    "license_id": source.license_id,
                    "license_url": source.license_url,
                    "redistribution": source.redistribution,
                    "used": source.source_id in used_sources,
                    "usage": provenance.get(source.source_id, {}),
                }
                for source in sorted(database.sources, key=lambda item: item.source_id)
            ],
            "usage": provenance,
        },
        "unsupported": [
            "automatic parameter fitting or imputation",
            "unbounded T/P/ionic-strength extrapolation",
            "direct Reaktoro/PHREEQC file import without an explicit converter",
            "global multiphase stability and phase-split solving",
        ],
    }


def aqueous_database_capabilities() -> dict[str, Any]:
    """Return the native database contract and explicit source boundaries."""

    catalog = electrolyte_activity_catalog()
    from chemworld.physchem.aqueous_database_external import EXTERNAL_AQUEOUS_DATABASE_SCHEMA

    return {
        "schema_version": AQUEOUS_DATABASE_AUDIT_SCHEMA_VERSION,
        "source_specification": "chemworld-aqueous-database-v1",
        "phase_models": ["aqueous", "solvent", "gas", "mineral"],
        "standard_state_models": [
            "reference-only",
            "constant-volume",
            "gas-fugacity-reference",
            "aqueous-gibbs",
        ],
        "activity_models": sorted(catalog),
        "activity_contract": {
            family: {
                "parameters": details["parameters"],
                "interaction_kinds": sorted(details["interactions"]),
                "domain_policy": "declared finite T/P and ionic-strength bounds",
            }
            for family, details in catalog.items()
        },
        "source_contract": [
            "citation and source URL",
            "license identifier and license URL",
            "redistribution policy",
            "every consumed source explicitly declared",
        ],
        "external_json_contract": {
            "schema": EXTERNAL_AQUEOUS_DATABASE_SCHEMA,
            "providers": ["phreeqc", "reaktoro"],
            "normalization": "canonical SI units and lossless native declarations",
            "source_files": "not parsed; provider export must produce this envelope",
        },
        "qualification_boundaries": {
            "standard_quantities": "evaluate each declared standard state at requested T/P",
            "activities": "evaluate the selected excess-Gibbs activity family independently",
            "chemical_potentials": (
                "combine standard states and activities with finite solvent inventory"
            ),
            "reaction_affinity": "derive from the same chemical-potential report",
            "out_of_domain": "reject outside declared T/P/ionic-strength bounds",
        },
        "unsupported": [
            "automatic parameter estimation",
            "implicit dilute-solution fallback for missing interactions",
            "direct Reaktoro/PHREEQC source-file import",
            "global multiphase stability and phase-split solving",
        ],
    }
