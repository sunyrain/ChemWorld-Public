"""Complete external Phase Change directory, with scientific roles kept distinct."""

from chemworld.data.constant_catalog import ConstantSource

PHASE_CHANGE_SOURCES = tuple(
    ConstantSource(source_id, "Phase Change/" + filename)
    for source_id, filename in (
        ("ALIBAKHSHI", "Alibakhshi one-coefficient enthalpy of vaporization.tsv"),
        ("BELL_TWU", "Bell 2018 je7b00967_si_001.tsv"),
        ("CRC_FUSION", "CRC Handbook Heat of Fusion.tsv"),
        ("CRC_VAPORIZATION", "CRC Handbook Heat of Vaporization.tsv"),
        ("GHAZARATI_SUBLIMATION", "Ghazerati Appendix Sublimation Enthalpy.tsv"),
        ("GHAZARATI_VAPORIZATION", "Ghazerati Appendix Vaporization Enthalpy.tsv"),
        ("OPENNOTEBOOK_MELTING", "OpenNotebook Melting Points.tsv"),
        (
            "PERRY_VAPORIZATION",
            "Table 2-150 Heats of Vaporization of Inorganic and Organic Liquids.tsv",
        ),
        ("VDI_PPDS_VAPORIZATION", "VDI PPDS Enthalpies of vaporization.tsv"),
        ("YAWS_BOILING", "Yaws Boiling Points.tsv"),
    )
)
PHASE_CHANGE_SOURCE_BY_ID = {s.source_id: s for s in PHASE_CHANGE_SOURCES}
