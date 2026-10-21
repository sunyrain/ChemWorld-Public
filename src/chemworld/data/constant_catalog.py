"""Source and unit declarations for the external chemicals 1.5.2 data layout.

This module contains descriptions only. It never loads a data table or imports
the reference implementation. Unknown coefficient semantics remain explicit.
"""

from dataclasses import dataclass

from chemworld.physchem.data_conflicts import DataSourceProvenance

CONSTANT_DATA_VERSION = "chemicals-constant-data-1.5.2"


@dataclass(frozen=True)
class ConstantSource:
    source_id: str
    relative_path: str
    kind: str = "tsv"
    nature: str = "reported literature"
    license: str = "Data redistribution permission not established; external read-only input"
    citation: str = ""

    @property
    def provenance(self):
        return DataSourceProvenance(
            "chemicals-constants:" + self.source_id,
            100,
            self.citation or f"chemicals 1.5.2 distribution: {self.relative_path}",
            CONSTANT_DATA_VERSION,
            self.license,
        )


CONSTANT_SOURCES = (
    ConstantSource("PSRK", "Critical Properties/Appendix to PSRK Revision 4.tsv"),
    ConstantSource("CRC_CRITICAL", "Critical Properties/CRCCriticalOrganics.tsv"),
    ConstantSource("DIPPR_PINA_MARTINES", "Critical Properties/DIPPRPinaMartines.tsv"),
    ConstantSource("FEDORS", "Critical Properties/fedors_Vc_predictions.tsv", nature="prediction"),
    ConstantSource("IUPAC", "Critical Properties/IUPACOrganicCriticalProps.tsv"),
    ConstantSource("MATHEWS", "Critical Properties/Mathews1972InorganicCriticalProps.tsv"),
    ConstantSource(
        "OMEGA_PSAT", "Critical Properties/omega_Psat_Tc_predictions.tsv", nature="prediction"
    ),
    ConstantSource("PASSUT_DANNER", "Critical Properties/PassutDanner1973.tsv"),
    ConstantSource(
        "WILSON_JASPERSON",
        "Critical Properties/wilson_jasperson_Tc_Pc_predictions.tsv",
        nature="prediction",
    ),
    ConstantSource("YAWS", "Critical Properties/Yaws Collection.tsv"),
    ConstantSource("STAVELEY", "Triple Properties/Staveley 1981.tsv"),
    ConstantSource("ALSHEHRI_HANSEN", "Misc/alshehri_hansen_solubility_parameters.tsv"),
    ConstantSource("CCCBDB", "Misc/cccbdb.nist.gov Dipoles.csv"),
    ConstantSource(
        "CHEMSEP_XML",
        "Misc/ChemSep8.32.xml",
        kind="xml",
        license="ChemSep data: Artistic License 2.0 attribution in distribution",
        citation="Kooijman and Taylor; ChemSep 8.32 pure component data",
    ),
    ConstantSource(
        "CHEMSEP_RG",
        "Misc/chemsep_radius_of_gyrations.tsv",
        license="ChemSep data: Artistic License 2.0 attribution in distribution",
    ),
    ConstantSource("COMMON_CHEMISTRY", "Misc/common_chemistry_data.tsv"),
    ConstantSource("CRC_RI", "Misc/CRC Handbook Organic RI.csv"),
    ConstantSource("DAHMEN_IQT", "Misc/Dahmen_Marquardt_ignition_delay_IQT.tsv"),
    ConstantSource("LEGACY_ELEMENTS", "Misc/Element data.csv", nature="historical atomic table"),
    ConstantSource(
        "OPENBABEL_ELEMENTS",
        "Misc/element.txt",
        kind="elements",
        nature="historical atomic table with explicit defaults",
        license="Source header: Open Babel GNU General Public License (GPL)",
        citation="Blue Obelisk element data, Open Babel element.txt; source header",
    ),
    ConstantSource("FLORIAN_CETANE", "Misc/Florian_Liming_CETANE_experimental.tsv"),
    ConstantSource("FLORIAN_MON", "Misc/Florian_Liming_MON_experimental.tsv"),
    ConstantSource("FLORIAN_RON", "Misc/Florian_Liming_RON_experimental.tsv"),
    ConstantSource("FLORIAN_ANN", "Misc/Florian_Liming_RON_MON_ANN.tsv", nature="prediction"),
    ConstantSource("HEOS", "Misc/heos_constants.tsv", nature="EOS-derived constants"),
    ConstantSource("HSPIPY_HANSEN", "Misc/hspipy_hansen_solubility_parameters.tsv"),
    ConstantSource("JOBACK", "Misc/joback_predictions.tsv", nature="prediction"),
    ConstantSource("MULLER", "Misc/Muller Supporting Info Dipoles.csv"),
    ConstantSource("CRC_INORGANIC", "Misc/Physical Constants of Inorganic Compounds.csv"),
    ConstantSource("CRC_ORGANIC", "Misc/Physical Constants of Organic Compounds.csv"),
    ConstantSource("POLING", "Misc/Poling Dipole.csv"),
    ConstantSource("PSI4_DIPOLE", "Misc/psi4_dipoles.tsv", nature="quantum calculation"),
    ConstantSource("PSI4_LINEAR", "Misc/psi4_linear.tsv", nature="quantum calculation"),
    ConstantSource("PSI4_RG", "Misc/psi4_radius_of_gyrations.tsv", nature="quantum calculation"),
    ConstantSource("RUBEN_HANSEN", "Misc/ruben_manuel_hansen_solubility_parameters.tsv"),
    ConstantSource("SCHRIER_HANSEN", "Misc/schrier_hansen_solubility_parameters.tsv"),
    ConstantSource("COMBUSTDB_CETANE", "Misc/Travis_Kessler_Combustdb_CETANE.tsv"),
    ConstantSource("COMBUSTDB_MON", "Misc/Travis_Kessler_Combustdb_MON.tsv"),
    ConstantSource(
        "COMBUSTDB_PREDICTIONS",
        "Misc/Travis_Kessler_Combustdb_predictions.tsv",
        nature="prediction",
    ),
    ConstantSource("COMBUSTDB_RON", "Misc/Travis_Kessler_Combustdb_RON.tsv"),
    ConstantSource(
        "VDI",
        "Misc/VDI Saturation Compounds Data.json",
        kind="vdi",
        citation="VDI Heat Atlas, 2nd ed., 2010; distribution includes stated estimates",
    ),
    ConstantSource("WEBBOOK", "Misc/webbook_constants.tsv"),
    ConstantSource("WIKIDATA", "Misc/wikidata_properties.tsv"),
)


@dataclass(frozen=True)
class ConstantField:
    field_id: str
    unit: str | None
    description: str
    continuation: str = "22B3"


def _fields(names, unit, description):
    return {name: ConstantField(name, unit, description) for name in names.split()}


TABULAR_FIELDS = {
    **_fields(
        "Tc Tt Tm Tb RIT T_flash T_autoignition POUR_POINT CLOUD_POINT",
        "K",
        "Reported temperature; phase and condition follow the source field",
    ),
    **_fields("Pc Pt", "Pa", "Reported pressure"),
    **_fields("Vc Vms Vml", "m^3/mol", "Reported molar volume, phase follows the source field"),
    **_fields("MW", "g/mol", "Reported molecular weight"),
    **_fields(
        "omega Zc RI logP LFL UFL RON MON CETANE linear",
        "dimensionless",
        "Source-specific dimensionless quantity; no equivalence between different fields",
    ),
    **_fields(
        "Hfs Hfl Hfg Hf Hfus Hvap Hsub",
        "J/mol",
        "Reported molar enthalpy; unspecified phase Hf is not relabelled Hfg",
    ),
    **_fields("S0s S0l S0g S0", "J/(mol*K)", "Reported molar entropy; source reference retained"),
    **_fields("rho", "kg/m^3", "Reported mass density"),
    **_fields("RG", "m", "Radius of gyration; source molecular_geometry public data description"),
    **_fields("dipole_moment", "Debye", "Dipole moment; source dipole public data description"),
    **_fields("IGNITION_DELAY", "s", "Reported ignition delay"),
    **_fields(
        "HANSEN_DELTA_D HANSEN_DELTA_P HANSEN_DELTA_H",
        "Pa^0.5",
        "Hansen solubility parameter component",
    ),
    **_fields("Tc_error", "K", "Source error magnitude; statistical coverage not specified"),
    **_fields(
        "Pc_error Pt_uncertainty", "Pa", "Source error magnitude; no inferred one-sigma claim"
    ),
    **_fields("Vc_error", "m^3/mol", "Source error magnitude; no inferred one-sigma claim"),
}
for _coefficient in (
    "Cpg0",
    "Cpg1",
    "Cpg2",
    "Cpg3",
    "mul0",
    "mul1",
    "AntoineA",
    "AntoineB",
    "AntoineC",
    "AntoineTmin",
    "AntoineTmax",
):
    TABULAR_FIELDS[_coefficient] = ConstantField(
        _coefficient,
        None,
        "Raw correlation parameter; needs its complete equation/unit contract",
        "22C/22E",
    )

VDI_UNITS = {
    "T": "K",
    "P": "Pa",
    "Density (l)": "kg/m^3",
    "Density (g)": "kg/m^3",
    "Hvap": "J/mol",
    "Cp (l)": "J/(mol*K)",
    "Cp (g)": "J/(mol*K)",
    "Mu (l)": "Pa*s",
    "Mu (g)": "Pa*s",
    "K (l)": "W/(m*K)",
    "K (g)": "W/(m*K)",
    "Pr (l)": "dimensionless",
    "Pr (g)": "dimensionless",
    "sigma": "N/m",
    "Beta": "1/K",
    "Volume (l)": "m^3/mol",
    "Volume (g)": "m^3/mol",
    "MW": "g/mol",
    "Tc": "K",
}
XML_UNIT_NAMES = {
    "K": "K",
    "Pa": "Pa",
    "m3/kmol": "m^3/kmol",
    "_": "dimensionless",
    "kg/kmol": "kg/kmol",
    "J0.5/m1.5": "Pa^0.5",
    "Coulomb.m": "C*m",
    "J/kmol": "J/kmol",
    "J/kmol/K": "J/(kmol*K)",
    "m": "m",
    "kg0.25.m3/s0.5/kmol": "kg^0.25*m^3/(s^0.5*kmol)",
    "m2/kmol": "m^2/kmol",
}
XML_FIELD_ALIASES = {
    "Tc": "CriticalTemperature",
    "Pc": "CriticalPressure",
    "Vc": "CriticalVolume",
    "Zc": "CriticalCompressibility",
    "Tb": "NormalBoilingPointTemperature",
    "Tm": "NormalMeltingPointTemperature",
    "Tt": "TriplePointTemperature",
    "Pt": "TriplePointPressure",
    "MW": "MolecularWeight",
    "omega": "AcentricityFactor",
    "dipole_moment": "DipoleMoment",
    "RG": "RadiusOfGyration",
    "Hfg": "HeatOfFormation",
    "S0g": "AbsEntropy",
    "Hfus": "HeatOfFusionAtMeltingPoint",
}
ELEMENT_TEXT_FIELDS = (
    "atomic_number",
    "symbol",
    "allred_rochow",
    "covalent_radius",
    "bond_order_radius",
    "vdw_radius",
    "max_bond_valence",
    "historical_mass",
    "pauling",
    "ionization_energy",
    "electron_affinity",
    "red",
    "green",
    "blue",
    "name",
)
ELEMENT_TEXT_UNITS = {
    "atomic_number": "dimensionless",
    "allred_rochow": "dimensionless",
    "covalent_radius": "angstrom",
    "bond_order_radius": "angstrom",
    "vdw_radius": "angstrom",
    "max_bond_valence": "dimensionless",
    "pauling": "dimensionless",
    "ionization_energy": "eV",
    "electron_affinity": "eV",
    "red": "dimensionless",
    "green": "dimensionless",
    "blue": "dimensionless",
}
# Defaults explicitly documented by element.txt; indistinguishable measured
# values at the same literal are conservatively labelled ambiguous, not zero.
ELEMENT_TEXT_DEFAULTS = {
    "allred_rochow": 0.0,
    "covalent_radius": 1.6,
    "vdw_radius": 2.0,
    "max_bond_valence": 6.0,
    "pauling": 0.0,
    "ionization_energy": 0.0,
    "electron_affinity": 0.0,
}
