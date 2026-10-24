"""Independent material identities and existing battery contracts for MSMR-SPM.

The scalar references use the site-occupancy definition and the original
occupancy form of exchange current, independently of the production inverse
and potential-form kinetic implementation.  These are functional contracts,
not an additional time-domain scientific qualification block.
"""

import json
from dataclasses import replace
from math import exp, log

import numpy as np
import pytest
from scipy.optimize import brentq

from chemworld.data.scientific_inputs import (
    InputSource,
    InputValue,
    QuantityColumn,
    ScientificRow,
    ScientificTable,
)
from chemworld.physchem.battery_models import (
    BatterySPMSpec,
    ParticleSpec,
    battery_chemical_energy_j,
    battery_chemical_potentials,
    battery_dissipation_w,
    battery_readout,
    battery_spec_from_dict,
)
from chemworld.physchem.battery_msmr import BatteryMSMRSPMSpec, msmr_fields
from chemworld.physchem.battery_msmr_materials import MSMRMaterialSpec
from chemworld.physchem.battery_soh import battery_spec_at_soc, soh_from_battery
from chemworld.physchem.data_conflicts import DataSourceProvenance

FARADAY = 96485.33212
GAS_CONSTANT = 8.31446261815324
UNITS = {
    "X": "dimensionless", "U0_V": "V", "w": "dimensionless",
    "jref_A_m2": "A/m^2", "alpha": "dimensionless",
}
REACTIONS = (
    {"X": 0.35, "U0_V": 0.12, "w": 1.2, "jref_A_m2": 2.3, "alpha": 0.27},
    {"X": 0.65, "U0_V": 0.20, "w": 2.4, "jref_A_m2": 4.7, "alpha": 0.73},
)


def source():
    return InputSource(
        DataSourceProvenance(
            "msmr-compatibility-synthetic", 0,
            "Explicit two-reaction material for source-definition contracts", "1", "CC0-1.0",
        ),
        "synthetic",
    )


def material(reactions=REACTIONS, *, material_id="negative", reverse=False):
    table = ScientificTable(
        material_id + "-reaction-table", "parameters",
        tuple(
            QuantityColumn(key, unit, "parameter", (-1e6, 1e6), "Explicit synthetic input")
            for key, unit in UNITS.items()
        ),
        (source(),),
        tuple(
            ScientificRow(
                f"reaction-{i}", source().provenance.source_id, "material", f"reaction-{i}",
                "parameters", {
                    key: InputValue(value, UNITS[key], None, "Fixed synthetic input")
                    for key, value in row.items()
                },
            )
            for i, row in enumerate(reactions)
        ),
    )
    row_ids = tuple(row.row_id for row in table.rows)
    return MSMRMaterialSpec(
        parameters=table, reaction_row_ids=row_ids[::-1] if reverse else row_ids,
        material_id=material_id, reference_electrolyte_concentration_mol_m3=1000.0,
    )


def scalar_reference(rows, potential, temperature, eta=0.0, concentration=1000.0):
    f = FARADAY / (GAS_CONSTANT * temperature)
    occupancies, first, second, exchange, currents = [], [], [], [], []
    dj_deta = dj_du = 0.0
    for row in rows:
        X, w, a = row["X"], row["w"], row["alpha"]
        p = 1 / (1 + exp(f * (potential - row["U0_V"]) / w))
        x = X * p
        occupancies.append(x)
        first.append(-f / w * X * p * (1 - p))
        second.append((f / w)**2 * X * p * (1 - p) * (1 - 2 * p))
        # Original occupancy form, not the production reformulation in potential.
        j0 = (row["jref_A_m2"] * x**(w * a) * (X - x)**(w * (1 - a))
              * (concentration / 1000.0)**(1 - a))
        positive, negative = exp((1 - a) * f * eta), exp(-a * f * eta)
        exchange.append(j0)
        currents.append(j0 * (positive - negative))
        dj_deta += j0 * f * ((1 - a) * positive + a * negative)
        dj_du += j0 * f * (p - a) * (positive - negative)
    return {
        "occupancies": occupancies,
        "derivatives": (sum(occupancies), sum(first), sum(second)),
        "exchange": exchange, "currents": currents,
        "deta_dj": 1 / dj_deta, "deta_dU": -dj_du / dj_deta,
    }


@pytest.mark.parametrize("potential,temperature", [(0.08, 285.0), (0.16, 298.15), (0.26, 320.0)])
def test_two_reaction_thermodynamics_and_inverse_use_declared_sites(potential, temperature):
    model = material()
    expected = scalar_reference(REACTIONS, potential, temperature)
    np.testing.assert_allclose(model.occupancies(potential, temperature),
                               expected["occupancies"], rtol=3e-13, atol=1e-15)
    np.testing.assert_allclose(model.stoichiometry_derivatives(potential, temperature),
                               expected["derivatives"], rtol=3e-12, atol=1e-12)
    x = expected["derivatives"][0]
    assert model.stoichiometry(potential, temperature) == pytest.approx(x, abs=2e-14)
    assert model.potential(x, temperature) == pytest.approx(potential, abs=2e-11, rel=0)
    assert expected["derivatives"][1] < 0


@pytest.mark.parametrize("width", [1.0, 2.0])
def test_single_reaction_inverse_matches_closed_form_and_differential_capacity(width):
    rows = ({"X": 1.0, "U0_V": 0.18, "w": width, "jref_A_m2": 2.0, "alpha": 0.5},)
    model = material(rows)
    temperature = 310.0
    thermal_voltage = GAS_CONSTANT * temperature / FARADAY
    for x in (0.03, 0.25, 0.5, 0.81, 0.97):
        expected = 0.18 + width * thermal_voltage * log((1 - x) / x)
        actual = model.potential(x, temperature)
        assert actual == pytest.approx(expected, rel=0, abs=2e-11)
        _, dx_du, _ = model.stoichiometry_derivatives(actual, temperature)
        assert 1 / dx_du == pytest.approx(-width * thermal_voltage / (x * (1 - x)),
                                         rel=2e-10, abs=1e-11)


@pytest.mark.parametrize(
    "eta,temperature,concentration", [(-0.04, 288., 600.), (0.03, 310., 1400.)],
)
def test_distinct_reaction_kinetics_and_implicit_derivatives_use_each_alpha(
    eta, temperature, concentration
):
    model, potential = material(), 0.16
    expected = scalar_reference(REACTIONS, potential, temperature, eta, concentration)
    np.testing.assert_allclose(
        model.exchange_current_densities(potential, temperature, concentration),
        expected["exchange"], rtol=3e-12, atol=1e-13,
    )
    np.testing.assert_allclose(model.reaction_currents(potential, eta, temperature, concentration),
                               expected["currents"], rtol=3e-12, atol=1e-13)
    target = sum(expected["currents"])
    actual_eta, currents, derivative_j, derivative_u = model.kinetic_response(
        potential, target, temperature, concentration,
    )
    assert actual_eta == pytest.approx(eta, rel=0, abs=2e-10)
    np.testing.assert_allclose(currents, expected["currents"], rtol=2e-9, atol=1e-10)
    assert sum(currents) == pytest.approx(target, rel=2e-9, abs=1e-10)
    assert derivative_j == pytest.approx(expected["deta_dj"], rel=2e-8, abs=1e-10)
    assert derivative_u == pytest.approx(expected["deta_dU"], rel=2e-8, abs=1e-10)
    assert derivative_j > 0
    # Distinct alpha values must not collapse to one common symmetric BV branch.
    reverse = scalar_reference(REACTIONS, potential, temperature, -eta, concentration)
    assert not np.isclose(sum(reverse["currents"]), -target, rtol=1e-3)


def test_material_codec_preserves_sources_row_identity_and_permutation_physics():
    model, reordered = material(), material(reverse=True)
    payload = json.loads(json.dumps(model.to_dict(), sort_keys=True, allow_nan=False))
    restored = MSMRMaterialSpec.from_dict(payload)
    assert restored.to_dict() == payload
    assert restored.parameters == model.parameters
    assert restored.reaction_row_ids == ("reaction-0", "reaction-1")
    assert restored.parameters.sources == (source(),)
    for potential in (0.1, 0.16, 0.24):
        assert reordered.stoichiometry(potential, 300.) == pytest.approx(
            model.stoichiometry(potential, 300.), abs=2e-14)
        np.testing.assert_allclose(reordered.occupancies(potential, 300.),
                                   model.occupancies(potential, 300.)[::-1], rtol=0, atol=0)
        assert sum(reordered.reaction_currents(potential, 0.02, 300., 1000.)) == pytest.approx(
            sum(model.reaction_currents(potential, 0.02, 300., 1000.)), abs=1e-13)
    with pytest.raises(ValueError):
        MSMRMaterialSpec.from_dict({**payload, "normalize_capacity": True})


@pytest.mark.parametrize("column,value", [
    ("X", -0.1), ("X", 0.35001), ("w", 0.0), ("jref_A_m2", 0.0),
    ("alpha", 0.0), ("alpha", 1.0),
])
def test_invalid_reaction_data_rejects_without_normalization(column, value):
    rows = ({**REACTIONS[0], column: value}, REACTIONS[1])
    with pytest.raises(ValueError):
        material(rows)


@pytest.mark.parametrize("fraction", [0.0, 1.0, -0.01, 1.01])
def test_inverse_requires_an_interior_total_inventory(fraction):
    instance = material()
    with pytest.raises(ValueError):
        instance.potential(fraction, 298.15)


@pytest.mark.parametrize("identities", [(), ("reaction-0", "reaction-0"), ("absent",)])
def test_reaction_identity_contract_rejects_missing_empty_or_reused_rows(identities):
    instance = material()
    with pytest.raises(ValueError):
        replace(instance, reaction_row_ids=identities)


def model():
    positive = tuple({**row, "U0_V": row["U0_V"] + 3.8} for row in REACTIONS)
    return BatteryMSMRSPMSpec(
        negative=ParticleSpec(cells=4, mesh_exponent=1.0),
        positive=ParticleSpec(cells=5, mesh_exponent=1.0),
        negative_material=material(), positive_material=material(positive, material_id="positive"),
    )


def test_uniform_radial_scope_refuses_legacy_stretched_particle_mesh():
    spec = model()
    with pytest.raises(ValueError):
        replace(spec, negative=replace(spec.negative, mesh_exponent=2.0))


def test_surface_kinetic_occupancies_are_distinct_from_extrapolated_source_observables():
    spec = model()
    state = spec.initial_state()
    potentials = np.array([0.14, 0.13, 0.12, 0.16])
    particle = spec.negative
    weights = particle.mesh.cell_volumes_m3 / particle.active_volume_m3
    # Rearrange Li inside this particle, preserving its recorded charge exactly.
    shift = brentq(lambda offset: sum(
        weight * scalar_reference(REACTIONS, u + offset, state.temperature_K)["derivatives"][0]
        for weight, u in zip(weights, potentials, strict=True)
    ) - spec.initial_negative_fraction, -0.1, 0.1, xtol=1e-14)
    potentials += shift
    expected_profiles = np.array([
        scalar_reference(REACTIONS, u, state.temperature_K)["occupancies"] for u in potentials
    ])
    amounts = dict(state.amounts_mol)
    amounts.update(zip(
        particle.keys("negative"),
        particle.mesh.cell_volumes_m3 * particle.maximum_concentration_mol_m3
        * expected_profiles.sum(axis=1), strict=True,
    ))
    changed = replace(state, amounts_mol=amounts)
    before = dict(changed.amounts_mol), dict(changed.coordinates), dict(changed.algebraic_values)
    fields = msmr_fields(spec, changed, current_A=0.0)
    surface_u = 1.5 * potentials[-1] - 0.5 * potentials[-2]
    expected_kinetic = scalar_reference(REACTIONS, surface_u, state.temperature_K)["occupancies"]
    expected_source = 1.5 * expected_profiles[-1] - 0.5 * expected_profiles[-2]
    np.testing.assert_allclose(fields["negative_particle_potentials_V"], potentials,
                               rtol=0, atol=2e-11)
    assert fields["negative_potential_V"] == pytest.approx(surface_u, rel=0, abs=2e-11)
    np.testing.assert_allclose(fields["negative_surface_occupancies"], expected_kinetic,
                               rtol=2e-9, atol=2e-11)
    np.testing.assert_allclose(fields["negative_source_surface_occupancies"], expected_source,
                               rtol=2e-9, atol=2e-11)
    assert np.max(np.abs(expected_source - expected_kinetic)) > 1e-3
    assert before == (dict(changed.amounts_mol), dict(changed.coordinates),
                      dict(changed.algebraic_values))


def test_msmr_roundtrip_keeps_one_total_li_inventory_and_source_materials():
    original = model()
    payload = json.loads(json.dumps(original.to_dict(), sort_keys=True, allow_nan=False))
    restored = battery_spec_from_dict(payload)
    assert type(restored) is BatteryMSMRSPMSpec
    assert restored.to_dict() == payload
    assert restored.negative_material.parameters == original.negative_material.parameters
    assert restored.positive_material.parameters == original.positive_material.parameters
    state = restored.initial_state()
    assert tuple(state.amounts_mol) == (
        *(f"negative:Li:{i}" for i in range(4)), *(f"positive:Li:{i}" for i in range(5)),
    )
    assert tuple(state.coordinates) == ("discharged_charge_C", "delivered_work_J")
    assert not any("potential" in key or "reaction" in key for key in state.amounts_mol)
    expected_u = [
        brentq(lambda u, rows=rows, x=x:
               scalar_reference(rows, u, original.temperature_K)["derivatives"][0] - x,
               lo, hi, xtol=1e-14)
        for rows, x, lo, hi in (
            (REACTIONS, 0.7, -0.2, 0.5),
            (tuple({**r, "U0_V": r["U0_V"] + 3.8} for r in REACTIONS), 0.3, 3.6, 4.3),
        )
    ]
    assert battery_readout(restored, state, 0.0).voltage_V == pytest.approx(
        expected_u[1] - expected_u[0], abs=4e-11, rel=0,
    )
    with pytest.raises((TypeError, ValueError)):
        battery_spec_from_dict({**payload, "normalize_capacity": True})


def test_unsupported_energy_impedance_and_soh_do_not_approximate_or_mutate_state():
    spec = model()
    state = spec.initial_state()
    before = (dict(state.amounts_mol), dict(state.coordinates), dict(state.algebraic_values))
    requests = (
        lambda: battery_chemical_potentials(spec, state),
        lambda: battery_chemical_energy_j(spec, state),
        lambda: battery_dissipation_w(spec, state, 0.1),
        lambda: soh_from_battery(spec, state),
        lambda: battery_spec_at_soc(spec, None, 0.5),
    )
    for request in requests:
        with pytest.raises(ValueError, match=r"MSMR.*unsupported"):
            request()
        assert before == (dict(state.amounts_mol), dict(state.coordinates),
                          dict(state.algebraic_values))


def test_old_spm_codec_and_readout_remain_plain_single_ocp():
    original = BatterySPMSpec(negative=ParticleSpec(cells=3), positive=ParticleSpec(cells=5))
    payload = json.loads(json.dumps(original.to_dict(), sort_keys=True, allow_nan=False))
    restored = battery_spec_from_dict(payload)
    assert type(restored) is BatterySPMSpec
    assert restored.to_dict() == payload
    state = restored.initial_state()
    assert tuple(state.amounts_mol) == (
        *(f"negative:Li:{i}" for i in range(3)), *(f"positive:Li:{i}" for i in range(5)),
    )
    assert tuple(state.coordinates) == ("discharged_charge_C", "delivered_work_J")
    assert not any("msmr" in name or "reaction" in name for name in state.amounts_mol)
    before = dict(state.amounts_mol), dict(state.coordinates)
    readout = battery_readout(restored, state, 0.0)
    expected = (4.2 - 0.8 * 0.3) - (0.2 - 0.1 * 0.7)
    assert readout.voltage_V == pytest.approx(expected, abs=1e-12)
    assert before == (dict(state.amounts_mol), dict(state.coordinates))
