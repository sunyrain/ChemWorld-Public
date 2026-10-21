"""Complete external chemicals 1.5.2 Vapor Pressure source catalog."""

from chemworld.data.constant_catalog import ConstantSource

PRESSURE_SOURCES = tuple(
    ConstantSource(source_id, "Vapor Pressure/" + filename)
    for source_id, filename in (
        ("ALCOCK_PRESSURE", "Alcock_Itkin_Horrigan_metalic_elements.tsv"),
        ("ALCOCK_SUBLIMATION", "Alcock_Itkin_Horrigan_metalic_elements_sublimation.tsv"),
        ("ANTOINE_POLING", "Antoine Collection Poling.tsv"),
        ("TRC_ANTOINE_POLING", "Antoine Extended Collection Poling.tsv"),
        ("LANDOLT_SUBLIMATION", "Landolt_antoine_sublimation_V20.tsv"),
        ("LANDOLT_PRESSURE", "Landolt_antoine_V20.tsv"),
        ("PERRY_PRESSURE", "Table 2-8 Vapor Pressure of Inorganic and Organic Liquids.tsv"),
        ("VDI_PPDS_PRESSURE", "VDI PPDS Boiling temperatures at different pressures.tsv"),
        ("WAGNER_POLING", "Wagner Collection Poling.tsv"),
        ("WAGNER_MCGARRY", "Wagner Original McGarry.tsv"),
    )
)
PRESSURE_SOURCE_BY_ID = {source.source_id: source for source in PRESSURE_SOURCES}
