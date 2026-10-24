"""Fixed development coverage, including explicit synthetic native-consumer probes."""

from dataclasses import replace

from chemworld.foundation.state_ledgers import PhaseLedger, PhaseRecord, VesselRecord
from chemworld.physchem.aqueous_database import AqueousReaction
from chemworld.physchem.aqueous_hkf_sources import hkf_aqueous_database
from chemworld.physchem.aqueous_phase_properties import aqueous_phase_properties
from chemworld.physchem.aqueous_reaction_equilibrium import AqueousEquilibriumSpec

TP = ((298.15, 1e5), (323.15, 1e6), (373.15, 5e6), (473.15, 1e7), (573.15, 2e7), (623.15, 3e7))
DIELECTRIC = (
    (300, 55.31735, 77.74735),
    (300, 55.56148, 78.11269),
    (650, 2.24692, 1.26715),
    (650, 40.31090, 17.71733),
    (650, 52.58636, 26.62132),
    (870, 1.45275, 1.12721),
    (870, 20.98927, 4.98281),
    (870, 45.01376, 15.09746),
)


def inventory(db=None):
    db = db or hkf_aqueous_database()
    return {
        s.species_id: {"H2O": 55.5, "Na+": 0.1, "Cl-": 0.1}.get(s.species_id, 0) for s in db.species
    }


def synthetic_reaction():
    """Same-charge/composition synthetic partner; no claim of a real Cl- isomer.

    Delta H=2000 J/mol, delta S=3 J/mol/K; coefficient identity makes delta V=0.
    This is a declared software-consumer probe, not a qualified material record.
    """
    db = hkf_aqueous_database()
    chloride = next(s for s in db.species if s.species_id == "Cl-")
    original = chloride.standard_state.thermo
    thermo = replace(
        original,
        species_id="Cl-test",
        formation_gibbs_ref_J_mol=None,
        formation_enthalpy_ref_J_mol=original.enthalpy_ref_J_mol + 2000,
        entropy_ref_J_mol_K=original.entropy_ref_J_mol_K + 3,
        source_id="chemworld-ideal-molality",
    )
    species = replace(
        chloride,
        species_id="Cl-test",
        source_id="chemworld-ideal-molality",
        standard_state=replace(
            chloride.standard_state, thermo=thermo, source_id="chemworld-ideal-molality"
        ),
    )
    db = replace(
        db,
        database_id="synthetic-HKF-native-consumer-probe",
        species=(*db.species, species),
        reactions=(
            AqueousReaction(
                "synthetic-pair", {"Cl-": -1, "Cl-test": 1}, "chemworld-ideal-molality"
            ),
        ),
        activity_model=replace(
            db.activity_model, charges=(*db.activity_model.charges, ("Cl-test", -1))
        ),
    )
    return AqueousEquilibriumSpec(db, ("synthetic-pair",))


def native_inventory(T=473.15, P=1e7):
    spec = synthetic_reaction()
    amounts = inventory(spec.database)
    props = aqueous_phase_properties(spec.database, amounts, temperature_K=T, pressure_Pa=P)
    phase = PhaseRecord(
        "aqueous",
        "tank",
        "aqueous",
        props.volume_m3 * 1000,
        amounts,
        metadata={"carrier_representation": "species_inventory"},
    )
    other = PhaseRecord("other", "other-tank", "solid", 0.1, {"inert": 1})
    vessel = VesselRecord("tank", "HKF-TP", 2, 623.15, 1e8, temperature_K=T, pressure_Pa=P)
    return spec, PhaseLedger({"aqueous": phase, "other": other}), vessel
