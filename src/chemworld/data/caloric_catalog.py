"""Complete external caloric data layout for chemicals 1.5.2; no source data."""

from dataclasses import dataclass

from chemworld.data.constant_catalog import ConstantField, ConstantSource
from chemworld.physchem.data_conflicts import DataSourceProvenance

CALORIC_DATA_VERSION = "chemicals-caloric-data-1.5.2"


@dataclass(frozen=True)
class CaloricSource(ConstantSource):
    @property
    def provenance(self):
        return DataSourceProvenance(
            "chemicals-caloric:" + self.source_id,
            100,
            self.citation or f"chemicals 1.5.2 distribution: {self.relative_path}",
            CALORIC_DATA_VERSION,
            self.license,
        )


CALORIC_SOURCES = (
    CaloricSource(
        "CRC_THERMODYNAMIC",
        "Heat Capacity/CRC Standard Thermodynamic Properties of Chemical Substances.tsv",
    ),
    CaloricSource("CRC_SOLID_CP", "Heat Capacity/CRCHeatCapacitySolids.tsv"),
    CaloricSource("JANAF_GAS_CP", "Heat Capacity/JANAF_1998_gas_Cp.json", kind="caloric-json"),
    CaloricSource("JANAF_LIQUID_CP", "Heat Capacity/JANAF_1998_liq_Cp.json", kind="caloric-json"),
    CaloricSource("JANAF_SOLID_CP", "Heat Capacity/JANAF_1998_solid_Cp.json", kind="caloric-json"),
    CaloricSource("PERRY_LIQUID_100", "Heat Capacity/Perry_Table_2-153_DIPPR_100.tsv"),
    CaloricSource("PERRY_LIQUID_114", "Heat Capacity/Perry_Table_2-153_DIPPR_114.tsv"),
    CaloricSource("PERRY_151_CLEAN", "Heat Capacity/Perrys Table 2-151-2.tsv"),
    CaloricSource("PERRY_151_JSON", "Heat Capacity/Perrys Table 2-151.json", kind="caloric-json"),
    CaloricSource("PERRY_151_RAW", "Heat Capacity/Perrys Table 2-151.tsv"),
    CaloricSource("POLING_CP", "Heat Capacity/PolingDatabank.tsv"),
    CaloricSource(
        "PSI4_ADJUSTED_THETA",
        "Heat Capacity/psi4_adjusted_characteristic_temperatures.json",
        kind="caloric-json",
        nature="scaled quantum vibrational calculation",
    ),
    CaloricSource(
        "PSI4_UNADJUSTED_THETA",
        "Heat Capacity/psi4_unadjusted_characteristic_temperatures.json",
        kind="caloric-json",
        nature="unscaled quantum vibrational calculation",
    ),
    CaloricSource(
        "TRC_GAS", "Heat Capacity/TRC Thermodynamics of Organic Compounds in the Gas State.tsv"
    ),
    CaloricSource(
        "WEBBOOK_SHOMATE", "Heat Capacity/webbook_shomate_coefficients.json", kind="caloric-json"
    ),
    CaloricSource("ZABRANSKY", "Heat Capacity/Zabransky.tsv"),
    CaloricSource("API_TDB_GAS", "Reactions/API TDB Albahri Hf (g).tsv"),
    CaloricSource("ATCT_GAS", "Reactions/ATcT 1.112 (g).tsv"),
    CaloricSource("ATCT_LIQUID", "Reactions/ATcT 1.112 (l).tsv"),
    CaloricSource("JANAF_THERMODYNAMIC", "Reactions/JANAF_1998.tsv"),
    CaloricSource("YAWS_THERMODYNAMIC", "Reactions/Yaws Hf S0 (g).tsv"),
)
CALORIC_SOURCE_BY_ID = {source.source_id: source for source in CALORIC_SOURCES}


def _fields(names, unit, description):
    return {name: ConstantField(name, unit, description, "22C2") for name in names.split()}


CALORIC_FIELDS = {
    **_fields(
        "Hfs Hfl Hfg Gfs Gfl Gfg",
        "J/mol",
        "Standard molar formation quantity at 298.15 K; phase follows field name",
    ),
    **_fields("Hfg_0K Hfl_0K", "J/mol", "Standard formation enthalpy at 0 K, not at 298.15 K"),
    **_fields(
        "S0s S0l S0g",
        "J/(mol*K)",
        "Standard absolute molar entropy at 298.15 K, not formation entropy",
    ),
    **_fields("Cps Cpl Cpg", "J/(mol*K)", "Molar isobaric heat capacity at 298.15 K only"),
    **_fields("Tc Tmin Tmax", "K", "Source temperature bound or critical temperature"),
}
