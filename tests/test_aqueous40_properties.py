"""Aqueous caloric/volume response must derive from the same extensive Gibbs law."""

from dataclasses import replace

import pytest
from tests.reference.aqueous39_fixtures import activity, database

from chemworld.physchem.aqueous_phase_properties import aqueous_phase_properties


def liquid_database(family="Pitzer"):
    db = database()
    ids = {"H2O", "Na+", "Cl-"}
    species = tuple(s for s in db.species if s.species_id in ids)
    species = tuple(replace(s, standard_state=replace(
        s.standard_state, pressure_model="constant-volume",
        molar_volume_m3_mol=18e-6 if s.phase == "solvent" else -1e-6,
    )) for s in species)
    model = activity(family)
    model = replace(model, parameters={
        k: replace(v, per_K=0.0001, per_Pa=1e-9) for k, v in model.parameters.items()
    }, interactions=tuple(replace(i, parameters={
        k: replace(v, per_K=0.00001, per_Pa=1e-10) for k, v in i.parameters.items()
    }) for i in model.interactions))
    return replace(db, species=species, reactions=(), activity_model=model)


@pytest.mark.parametrize("family", [
    "ideal", "DebyeHuckelLimiting", "DebyeHuckel", "Davies", "SIT", "Pitzer",
])
def test_extensive_euler_caloric_and_pressure_identities(family):
    db = liquid_database(family)
    amounts = {"H2O": 55.5, "Na+": 0.3, "Cl-": 0.3}
    t, p = 310.0, 500000.0

    def properties(t, p, n=amounts):
        return aqueous_phase_properties(db, n, temperature_K=t, pressure_Pa=p)

    state = properties(t, p)
    # Independent symmetric changes of evaluated chemical potentials check
    # both pressure-dependent and temperature-dependent excess terms.
    dt, dp = 0.005, 100.0
    dg_dt = (properties(t + dt, p).gibbs_energy_J
             - properties(t - dt, p).gibbs_energy_J) / (2 * dt)
    dg_dp = (properties(t, p + dp).gibbs_energy_J
             - properties(t, p - dp).gibbs_energy_J) / (2 * dp)
    assert state.entropy_J_K == pytest.approx(-dg_dt, rel=2e-9, abs=1e-6)
    assert state.volume_m3 == pytest.approx(dg_dp, rel=2e-8)
    assert state.enthalpy_J == pytest.approx(state.gibbs_energy_J - t * dg_dt, abs=0.0005)
    assert state.internal_energy_J == pytest.approx(state.enthalpy_J - p * state.volume_m3)
    doubled = properties(t, p, {k: 2 * v for k, v in amounts.items()})
    for key in ("volume_m3", "enthalpy_J", "gibbs_energy_J", "entropy_J_K"):
        assert getattr(doubled, key) == pytest.approx(2 * getattr(state, key), rel=1e-12)


def test_volume_requires_declared_standard_state_and_no_other_phases():
    db = liquid_database()
    undeclared = replace(db, species=tuple(replace(s, standard_state=replace(
        s.standard_state, pressure_model="reference-only", molar_volume_m3_mol=None,
    )) for s in db.species))
    with pytest.raises(ValueError, match="constant-volume"):
        aqueous_phase_properties(undeclared, {"H2O": 55.5}, temperature_K=300, pressure_Pa=1e5)
    with pytest.raises(ValueError, match="water and solutes"):
        aqueous_phase_properties(db, {"H2O": 55.5, "gas": 1}, temperature_K=300, pressure_Pa=1e5)
    with pytest.raises(ValueError, match="positive V"):
        aqueous_phase_properties(
            replace(db, species=tuple(replace(s, standard_state=replace(
                s.standard_state, molar_volume_m3_mol=-1e-6,
            )) for s in db.species)), {"H2O": 55.5}, temperature_K=300, pressure_Pa=1e5,
        )
