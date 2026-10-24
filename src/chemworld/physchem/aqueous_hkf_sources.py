"""Small attributed HKF parameter excerpt; source precision is not silently repaired.

Data: Cantera v3.1.0 test/data/pdss_hkft.yaml (BSD-3-Clause).
Only public numerical data were read; algorithms are independent implementations.
The OH- record is retained in full and rejected by the default precision check.

Copyright (c) 2001-2009, California Institute of Technology
All rights reserved.
Copyright (c) 2009 Sandia Corporation. Under the terms of
Contract AC04-94AL85000 with Sandia Corporation, the U.S. Government
retains certain rights in this software.
Copyright (c) 2011-2024, Cantera Developers.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
- Redistributions of source code must retain the above copyright notice,
  this list of conditions and the following disclaimer.
- Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.
- Neither the name of the California Institute of Technology, Sandia
  Corporation nor the names of other contributors may be used to endorse or
  promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
"""

from copy import deepcopy
from typing import Any

from chemworld.physchem.aqueous_database import (
    AqueousDatabase,
    AqueousDataSource,
    AqueousSpecies,
    AqueousStandardState,
)
from chemworld.physchem.aqueous_hkf import (
    AqueousReferenceBasis,
    IF97AqueousWaterThermo,
    RevisedHKFSpeciesThermo,
)
from chemworld.physchem.electrolyte_activity import ElectrolyteActivityModel

_URL = "https://raw.githubusercontent.com/Cantera/cantera/v3.1.0/test/data/pdss_hkft.yaml"
_SOURCE = "cantera-3.1.0-pdss-hkft-data"
_BASIS = "nist-chase1998-standard-entropy-Hplus-zero"
_RECORDS = {
    "Na+": {
        "composition": {"Na": 1},
        "charge": 1,
        "Hf": -57433,
        "Gf": None,
        "S": 13.96,
        "a": [0.1839, -228.5, 3.256, -27260],
        "c": [18.18, -29810],
        "omega": 33060,
    },
    "Cl-": {
        "composition": {"Cl": 1},
        "charge": -1,
        "Hf": None,
        "Gf": -31379,
        "S": 13.56,
        "a": [0.4032, 480.1, 5.563, -28470],
        "c": [-4.4, -57140],
        "omega": 145600,
    },
    "H+": {
        "composition": {"H": 1},
        "charge": 1,
        "Hf": None,
        "Gf": 0,
        "S": 0,
        "a": [0, 0, 0, 0],
        "c": [0, 0],
        "omega": 0,
    },
    "OH-": {
        "composition": {"H": 1, "O": 1},
        "charge": -1,
        "Hf": -54977,
        "Gf": -37595,
        "S": -2.56,
        "a": [0.12527, 7.38, 1.8423, -27821],
        "c": [4.15, -103460],
        "omega": 172460,
    },
}


def hkf_reference_basis() -> AqueousReferenceBasis:
    """NIST-JANAF/Chase (1998), per atom; explicit H+ conventional entropy zero.

    H2 130.680, O2 205.152, Cl2 223.079, Na(s) 51.46 J/mol/K at 298.15 K, 1 bar.
    No electronic/charge entropy is guessed from an uncharged species formula.
    """
    return AqueousReferenceBasis(
        _BASIS,
        {"H": 65.340, "O": 102.576, "Cl": 111.5395, "Na": 51.46},
        -65.340,
        "nist-janaf-1998-reference-datum",
    )


def hkf_source_records() -> dict[str, dict[str, Any]]:
    """Return complete printed records in cal/mol, bar and K; caller owns the copy."""
    return deepcopy(_RECORDS)


def hkf_species_from_record(
    species_id: str,
    record: dict[str, Any],
    *,
    reference_basis: AqueousReferenceBasis,
    source_id: str,
    units: str,
    consistency_tolerance_J_mol: float,
) -> RevisedHKFSpeciesThermo:
    """Explicit unit conversion; all source reference quantities participate in validation."""
    if units != "cal-mol-bar-K":
        raise ValueError("HKF record importer requires explicit cal-mol-bar-K units")
    required = {"composition", "charge", "Hf", "Gf", "S", "a", "c", "omega"}
    if set(record) != required:
        raise ValueError("HKF record has missing or unknown coefficients/reference quantities")
    if len(record["a"]) != 4 or len(record["c"]) != 2:
        raise ValueError("HKF record requires four a and two c coefficients")
    return RevisedHKFSpeciesThermo(
        species_id,
        record["composition"],
        record["charge"],
        reference_basis,
        record["S"] * 4.184,
        None if record["Hf"] is None else record["Hf"] * 4.184,
        None if record["Gf"] is None else record["Gf"] * 4.184,
        tuple(v * 4.184 / (1e5 if i in (0, 2) else 1) for i, v in enumerate(record["a"])),
        tuple(v * 4.184 for v in record["c"]),
        record["omega"] * 4.184,
        source_id,
        consistency_tolerance_J_mol,
    )


def hkf_source_species(species_id: str) -> RevisedHKFSpeciesThermo:
    """Strict source ingestion; OH- fails its printed Hf/Gf/S consistency check.

    Bound = 0.5 cal for each integer Hf/Gf + 0.005 cal/K for S, plus 0.001
    J/mol/K elemental-printing allowance; this is a rounding bound, not uncertainty.
    """
    if species_id not in _RECORDS:
        raise ValueError("unknown HKF source species")
    return hkf_species_from_record(
        species_id,
        hkf_source_records()[species_id],
        reference_basis=hkf_reference_basis(),
        source_id=_SOURCE,
        units="cal-mol-bar-K",
        consistency_tolerance_J_mol=4.184 * (1 + 298.15 * 0.005) + 0.29815,
    )


def hkf_source_water() -> IF97AqueousWaterThermo:
    """NIST liquid Hf=-285.830 kJ/mol and S=69.95 J/mol/K; converted IF97 zero."""
    return IF97AqueousWaterThermo(
        "H2O", hkf_reference_basis(), -285830, 69.95, "nist-janaf-1998-reference-datum"
    )


def hkf_aqueous_database() -> AqueousDatabase:
    """Actual Na+/Cl-/H+ + finite water, ideal molality; no fitted solution activity.

    OH- is deliberately absent because its full source record fails ingestion.
    This database therefore does not advertise water dissociation equilibrium.
    """
    sources = (
        AqueousDataSource(
            _SOURCE,
            "Cantera 3.1.0 HKFT test parameter data",
            _URL,
            "BSD-3-Clause",
            "https://raw.githubusercontent.com/Cantera/cantera/v3.1.0/License.txt",
            "permitted",
        ),
        AqueousDataSource(
            "nist-janaf-1998-reference-datum",
            "Chase 1998 NIST-JANAF reference scalars; H+ conventional zero",
            "https://janaf.nist.gov/",
            "NIST-SRD",
            "https://www.nist.gov/srd",
            "restricted",
        ),
        AqueousDataSource(
            "chemworld-ideal-molality",
            "Declared ideal molality solution, no empirical parameters",
            "urn:chemworld:ideal-molality",
            "CC0-1.0",
            "https://creativecommons.org/publicdomain/zero/1.0/",
            "synthetic",
        ),
    )
    water = hkf_source_water()
    thermos: list[IF97AqueousWaterThermo | RevisedHKFSpeciesThermo] = [
        water,
        *(hkf_source_species(k) for k in ("Na+", "Cl-", "H+")),
    ]
    species = tuple(
        AqueousSpecies(
            th.species_id,
            th.composition,
            getattr(th, "charge", 0),
            "solvent" if th.species_id == "H2O" else "aqueous",
            AqueousStandardState(
                th, th.reference_basis.basis_id, "aqueous-gibbs", (1, 1e8), th.source_id
            ),
            th.source_id,
        )
        for th in thermos
    )
    activity = ElectrolyteActivityModel(
        "hkf-ideal-molality",
        "ideal",
        (("Na+", 1), ("Cl-", -1), ("H+", 1)),
        "chemworld-ideal-molality",
        1.0,
        (273.15, 623.15),
        (1, 1e8),
    )
    return AqueousDatabase("HKF-IF97-qualified-source-subset", sources, species, (), activity)
