"""Versioned public environmental and safety source declarations, no source algorithms."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from chemworld.data.constant_catalog import CONSTANT_SOURCES, ConstantSource

ASSESSMENT_SOURCES = (
    ConstantSource("IPCC_2007", "Environment/Official Global Warming Potentials 2007.tsv"),
    ConstantSource("IPCC_2014", "Environment/Official Global Warming Potentials 2014.tsv"),
    ConstantSource("IPCC_2021", "Environment/Official Global Warming Potentials 2021.tsv"),
    ConstantSource("ODP", "Environment/Ozone Depletion Potentials.tsv"),
    ConstantSource("CRC_LOGP", "Environment/CRC logP table.tsv"),
    ConstantSource("SYRRES", "Environment/Syrres logP data.csv.gz", kind="tsv-gzip"),
    ConstantSource("IEC_2010", "Safety/IS IEC 60079-20-1 2010.tsv"),
    ConstantSource("NFPA_2008", "Safety/NFPA 497 2008.tsv"),
    ConstantSource("SERAT", "Safety/DIPPR T_flash Serat.csv"),
    ConstantSource("IARC", "Safety/IARC Carcinogen Database.tsv"),
    ConstantSource("NTP", "Safety/National Toxicology Program Carcinogens.tsv"),
    ConstantSource("ONTARIO", "Safety/Ontario Exposure Limits.json", kind="mapping-json"),
    ConstantSource("ONTARIO_RAW", "Safety/Ontario Exposure Limits.tsv"),
    next(s for s in CONSTANT_SOURCES if s.source_id == "WIKIDATA"),
)
ASSESSMENT_SOURCE_MAP = MappingProxyType({s.source_id: s for s in ASSESSMENT_SOURCES})
IARC_CLASSIFICATIONS = MappingProxyType(
    {
        1: "Carcinogenic to humans (1)",
        11: "Probably carcinogenic to humans (2A)",
        12: "Possibly carcinogenic to humans (2B)",
        3: "Not classifiable as to its carcinogenicity to humans (3)",
        4: "Probably not carcinogenic to humans (4)",
    }
)
NTP_CLASSIFICATIONS = MappingProxyType({1: "Known", 2: "Reasonably Anticipated"})


@dataclass(frozen=True)
class AssessmentMethod:
    property_id: str
    method: str
    source_id: str
    field_id: str
    unit: str | None = "dimensionless"
    conditions: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "conditions", MappingProxyType(dict(self.conditions)))


def _climate_methods() -> list[AssessmentMethod]:
    methods = []
    for prop, years in (
        ("GWP", {2007: (20, 100, 500), 2014: (20, 100), 2021: (20, 100, 500)}),
        ("GTP", {2014: (20, 50, 100), 2021: (50, 100)}),
    ):
        for year, horizons in years.items():
            for horizon in horizons:
                methods.append(
                    AssessmentMethod(
                        prop,
                        f"IPCC ({year}) {horizon}yr",
                        f"IPCC_{year}",
                        f"{horizon}yr {prop}",
                        conditions={
                            "assessment_year": year,
                            "horizon_years": horizon,
                            "reference_substance": "CO2",
                            "basis": "equal emitted mass",
                            "response": "integrated forcing" if prop == "GWP" else "temperature",
                        },
                    )
                )
    methods.append(
        AssessmentMethod(
            "GWP",
            "IPCC (1995) 100yr",
            "IPCC_2007",
            "SAR 100yr",
            conditions={
                "assessment_year": 1995,
                "horizon_years": 100,
                "reference_substance": "CO2",
                "basis": "equal emitted mass",
                "response": "integrated forcing",
            },
        )
    )
    return methods


def _methods() -> tuple[AssessmentMethod, ...]:
    methods = _climate_methods()
    for version in (1, 2):
        for suffix, column, unit in (
            ("Max", " Max", "dimensionless"),
            ("Min", " Min", "dimensionless"),
            ("logarithmic average", " Design", "dimensionless"),
            ("string", "", None),
        ):
            methods.append(
                AssessmentMethod(
                    "ODP",
                    f"ODP{version} {suffix}",
                    "ODP",
                    f"ODP{version}{column}",
                    unit,
                    {
                        "version": version,
                        "reference_substance": "CFC-11",
                        "basis": "equal emitted mass",
                        "selection": suffix,
                    },
                )
            )
    for source, name in (("SYRRES", "SYRRES"), ("CRC_LOGP", "CRC"), ("WIKIDATA", "WIKIDATA")):
        methods.append(
            AssessmentMethod(
                "logP",
                name,
                source,
                "logP",
                conditions={"log_base": 10, "phases": "octanol/water", "solute": "un-ionized"},
            )
        )
    for prop in ("T_flash", "T_autoignition", "LFL", "UFL"):
        for source, name in (
            ("IEC_2010", "IEC 60079-20-1 (2010)"),
            ("NFPA_2008", "NFPA 497 (2008)"),
            ("SERAT", "Serat DIPPR (2017)"),
            ("WIKIDATA", "WIKIDATA"),
        ):
            if source == "SERAT" and prop != "T_flash":
                continue
            methods.append(
                AssessmentMethod(
                    prop,
                    name,
                    source,
                    prop,
                    "K" if prop.startswith("T_") else "dimensionless",
                    {"basis": "reported source condition; no inferred test protocol"},
                )
            )
    for prop in ("TWA", "STEL", "Ceiling", "Skin"):
        methods.append(
            AssessmentMethod(
                prop,
                "Ontario Limits",
                "ONTARIO",
                prop,
                None,
                {
                    "jurisdiction": "Ontario",
                    "edition": "chemicals 1.5.2 source snapshot",
                    "window": prop,
                    "source_units": "both original units retained",
                },
            )
        )
    for source, name, field_id in (
        ("IARC", "International Agency for Research on Cancer", "group"),
        ("NTP", "National Toxicology Program 13th Report on Carcinogens", "Listing"),
    ):
        methods.append(
            AssessmentMethod(
                "Carcinogen",
                name,
                source,
                field_id,
                None,
                {"edition": "chemicals 1.5.2 source snapshot", "classification_system": source},
            )
        )
    return tuple(methods)


ASSESSMENT_METHODS = _methods()
