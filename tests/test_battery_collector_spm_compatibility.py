"""Compatibility contracts for the one-dimensional collector SPM family.

The collector implementation is intentionally a new battery family.  These
checks keep the eleven already-public battery families on their old codec,
equipment slot and persistent-state contracts while exercising the small,
stateless collector geometry primitive.  Detailed collector equations and
the full native/source comparison belong to the physical and reference lines.
"""

import json
from collections.abc import Mapping
from dataclasses import fields
from importlib import import_module

import numpy as np
import pytest

from chemworld.examples.battery_composite import default_spec as composite_spec
from chemworld.examples.battery_dfn import BatteryDFNSpec
from chemworld.examples.battery_dfn_composite import default_spec as composite_dfn_spec
from chemworld.examples.battery_half_cell import default_spec as half_cell_spec
from chemworld.examples.battery_lead_acid import default_spec as lead_acid_spec
from chemworld.examples.battery_mpm import default_spec as mpm_spec
from chemworld.examples.battery_msmr import default_spec as msmr_spec
from chemworld.examples.battery_newman_tobias import default_spec as newman_tobias_spec
from chemworld.examples.battery_split_ocvr import default_spec as split_ocvr_spec
from chemworld.physchem.battery_collector_spm import BatteryCollectorSPMSpec
from chemworld.physchem.battery_electrolyte import BatteryElectrolyteSpec
from chemworld.physchem.battery_models import (
    BATTERY_COORDINATES,
    BatterySPMSpec,
    ParticleSpec,
    battery_spec_from_dict,
)
from chemworld.runtime.battery import initial_battery_slot, unpack_battery
from chemworld.world.battery_world import battery_world_spec


def _snapshot(state):
    """Return an immutable comparison view without serialising numeric arrays."""
    return {
        field.name: dict(value) if isinstance(value, Mapping) else value
        for field in fields(state)
        for value in (getattr(state, field.name),)
    }


def _old_family_specs():
    """Use the same eleven public examples used by the previous family set."""
    return {
        "spm": BatterySPMSpec(negative=ParticleSpec(cells=4), positive=ParticleSpec(cells=4)),
        "spme": BatterySPMSpec(
            negative=ParticleSpec(cells=4), positive=ParticleSpec(cells=4),
            electrolyte=BatteryElectrolyteSpec(cells=(2, 2, 2)),
        ),
        "mpm": mpm_spec(),
        "dfn": BatteryDFNSpec(
            negative=ParticleSpec(cells=4), positive=ParticleSpec(cells=4),
            electrolyte=BatteryElectrolyteSpec(cells=(2, 2, 2)),
        ),
        "composite-spm": composite_spec(cells=4),
        "composite-dfn": composite_dfn_spec(radial_cells=4, cells=(2, 2, 2)),
        "newman-tobias": newman_tobias_spec(cells=(3, 2, 3)),
        "msmr": msmr_spec(cells=4),
        "split-ocvr": split_ocvr_spec(),
        "half-cell": half_cell_spec(cells=4),
        "lead-acid": lead_acid_spec(),
    }


def _collector_spec():
    """Small explicit collector fixture; no example/default hides ownership."""
    return BatteryCollectorSPMSpec(
        negative=ParticleSpec(cells=3, active_volume_m3=1.2e-6),
        positive=ParticleSpec(cells=3, active_volume_m3=1.5e-6),
        initial_negative_fractions=(.62, .67, .72),
        initial_positive_fractions=(.28, .31, .34),
        collector_cells=3,
        provenance="collector-compatibility-v1",
    )


def test_old_eleven_families_keep_exact_codec_and_equipment_slot_contract():
    """Importing/dispatching collector support must not rewrite old families."""
    for family, spec in _old_family_specs().items():
        payload = json.loads(json.dumps(spec.to_dict(), allow_nan=False))
        restored = battery_spec_from_dict(payload)
        assert restored == spec, family
        assert restored.to_dict() == payload, family

        state = restored.initial_state()
        before = _snapshot(state)
        raw = battery_world_spec(restored).physics.initial_states["reactor"]
        raw_before = raw.to_dict()
        record, slot, decoded, unpacked, _ = unpack_battery(raw)
        assert decoded == restored, family
        assert unpacked == state, family
        assert slot == initial_battery_slot(record, decoded), family
        assert slot.spec.owner_kind == "equipment", family
        assert slot.spec.batch_reset == "preserve", family
        # DFN/SPMe slots intentionally place electrolyte stocks before particle
        # stocks; the compatibility contract is identity and cardinality, not
        # an accidental cross-family ordering.
        assert set(slot.values) == {*restored.inventory_keys, *restored.coordinate_keys}, family
        assert len(slot.values) == (
            len(restored.inventory_keys) + len(restored.coordinate_keys)
        ), family
        assert _snapshot(state) == before, family
        assert raw.to_dict() == raw_before, family


def test_old_family_coordinates_never_gain_collector_algebraics():
    """Legacy slots retain only their declared coordinates after new dispatch loads."""
    for family, spec in _old_family_specs().items():
        assert tuple(spec.coordinate_keys) == tuple(
            getattr(spec, "coordinate_keys", BATTERY_COORDINATES)
        ), family
        assert not any(
            key.startswith("collector:") or key.endswith(":local_current_A")
            for key in spec.coordinate_keys
        ), family
        assert not any(
            key.startswith("collector:") or key.endswith(":local_current_A")
            for key in spec.inventory_keys
        ), family


def test_collector_geometry_is_stateless_and_preserves_source_boundary_observation():
    """The shared collector primitive cannot mutate q or conflate source and tab V."""
    module = import_module("chemworld.physchem.battery_current_collector")
    geometry = module.CollectorGeometrySpec(cells=4, negative_tab="left")
    qn = np.array([.01, .02, .03, .04])
    qp = -qn.copy()
    qn_before, qp_before = qn.copy(), qp.copy()
    negative, positive = module.collector_potentials(geometry, qn, qp, 3.9)
    assert negative.shape == positive.shape == (4,)
    assert np.all(np.isfinite(negative)) and np.all(np.isfinite(positive))
    assert np.array_equal(qn, qn_before) and np.array_equal(qp, qp_before)

    faces_n, faces_p = geometry.face_currents(qn, qp)
    assert faces_n.shape == faces_p.shape == (5,)
    assert faces_n[0] == pytest.approx(np.sum(qn))
    assert faces_n[-1] == pytest.approx(0.0)
    assert faces_p[0] == pytest.approx(0.0)
    assert faces_p[-1] == pytest.approx(-np.sum(qp))

    physical_tab = 3.9  # The supplied physical boundary, not the last cell centre.
    source_tab = geometry.boundary_voltage_v(positive)
    assert source_tab != physical_tab
    assert source_tab - physical_tab == pytest.approx(
        -geometry.dz_m * qp[-1] / (2 * geometry.positive_sheet_conductance_s),
        rel=0,
        abs=1e-15,
    )
    assert geometry.boundary_voltage_v(positive, source_linear=False) == float(positive[-1])


@pytest.mark.parametrize("negative_tab", ["left", "right"])
@pytest.mark.parametrize("q", [
    (.01, .02, .03, .04),
    (-.01, -.02, -.03, -.04),
    (.11, -.04, .02, .01),
], ids=["discharge", "charge", "local-reversal"])
def test_nonuniform_collector_currents_obey_tab_boundaries_and_local_divergence(
    negative_tab, q,
):
    """Left-to-right face currents obey KCL and Ohm's law on both metal foils.

    A right tab changes the boundary location, not the physical ordering of
    the nonuniform cell currents.  Local reversal is allowed: these checks do
    not assume that all cells share the sign of the terminal current.
    """
    module = import_module("chemworld.physchem.battery_current_collector")
    geometry = module.CollectorGeometrySpec(
        cells=4, negative_tab=negative_tab, height_m=.12, width_m=.02,
        negative_conductivity_S_m=2e5, positive_conductivity_S_m=3e5,
    )
    qn = np.array(q)
    qp = -qn
    before = qn.copy(), qp.copy()
    physical_tab = 3.9
    phi_n, phi_p = module.collector_potentials(geometry, qn, qp, physical_tab)
    face_n, face_p = geometry.face_currents(qn, qp)
    current = float(np.sum(qn))

    if negative_tab == "left":
        assert face_n[[0, -1]] == pytest.approx([current, 0.0], rel=0, abs=1e-16)
        assert -geometry.negative_sheet_conductance_s * phi_n[0] / (
            .5 * geometry.dz_m
        ) == pytest.approx(current, rel=0, abs=1e-15)
    else:
        assert face_n[[0, -1]] == pytest.approx([0.0, -current], rel=0, abs=1e-16)
        assert geometry.negative_sheet_conductance_s * phi_n[-1] / (
            .5 * geometry.dz_m
        ) == pytest.approx(-current, rel=0, abs=1e-15)
    assert face_p[[0, -1]] == pytest.approx([0.0, current], rel=0, abs=1e-16)
    assert -geometry.positive_sheet_conductance_s * (physical_tab - phi_p[-1]) / (
        .5 * geometry.dz_m
    ) == pytest.approx(current, rel=0, abs=1e-13)

    for cell in range(geometry.cells):
        assert face_n[cell + 1] - face_n[cell] == pytest.approx(-qn[cell], rel=0, abs=1e-16)
        assert face_p[cell + 1] - face_p[cell] == pytest.approx(-qp[cell], rel=0, abs=1e-16)
    assert -geometry.negative_sheet_conductance_s * np.diff(phi_n) / (
        geometry.dz_m
    ) == pytest.approx(face_n[1:-1], rel=0, abs=1e-15)
    assert -geometry.positive_sheet_conductance_s * np.diff(phi_p) / (
        geometry.dz_m
    ) == pytest.approx(face_p[1:-1], rel=0, abs=1e-13)
    assert np.array_equal(qn, before[0]) and np.array_equal(qp, before[1])


def test_collector_state_has_only_local_li_and_global_charge_work():
    spec = _collector_spec()
    state = spec.initial_state()
    expected = tuple(
        f"{electrode}:z:{z}:Li:{r}"
        for electrode in ("negative", "positive")
        for z in range(spec.collector_cells)
        for r in range(spec.negative.cells if electrode == "negative" else spec.positive.cells)
    )
    assert spec.inventory_keys == expected
    assert set(spec.inventory_species.values()) == {"Li"}
    assert tuple(spec.coordinate_keys) == BATTERY_COORDINATES
    assert state.algebraic_values == {}
    assert spec.hysteresis_keys == ()
    assert len(state.amounts_mol) == 2 * spec.collector_cells * spec.negative.cells
    assert state.volume_L == pytest.approx(spec.volume_l, rel=0, abs=1e-18)
    assert state.volume_L == pytest.approx(
        1000 * (
            spec.height_m * spec.width_m
            * (spec.negative_collector_thickness_m + spec.positive_collector_thickness_m)
            + spec.negative.active_volume_m3 + spec.positive.active_volume_m3
        ),
        rel=0,
        abs=1e-18,
    )
    assert state.volume_L != pytest.approx(
        1000 * (spec.negative.active_volume_m3 + spec.positive.active_volume_m3)
        * spec.collector_cells,
        rel=0,
        abs=1e-18,
    )


def test_collector_codec_preserves_nonuniform_local_inputs_without_derived_fields():
    spec = _collector_spec()
    payload = json.loads(json.dumps(spec.to_dict(), allow_nan=False))
    restored = BatteryCollectorSPMSpec.from_dict(payload)
    assert restored == spec
    assert restored.to_dict() == payload
    assert payload["model_family"] == "collector-spm-isothermal-v1"
    assert "particle_bins" not in payload
    assert "algebraic_names" not in payload
    assert restored.initial_state() == spec.initial_state()


def test_public_battery_codec_dispatches_collector_family_without_legacy_fallback():
    spec = _collector_spec()
    payload = spec.to_dict()
    restored = battery_spec_from_dict(payload)
    assert isinstance(restored, BatteryCollectorSPMSpec)
    assert restored == spec
    assert restored.model_family == "collector-spm-isothermal-v1"


@pytest.mark.parametrize("field", [
    "thermal", "electrolyte", "sei", "plating",
])
def test_collector_spec_rejects_unsupported_parallel_physics(field):
    payload = _collector_spec().to_dict()
    payload[field] = {"declared": True}
    before = json.loads(json.dumps(payload))
    with pytest.raises(ValueError, match=r"(?i)collector.*(without|support|isothermal)|side"):
        BatteryCollectorSPMSpec.from_dict(payload)
    assert payload == before


@pytest.mark.parametrize("name", [
    "battery_chemical_energy_j",
    "battery_chemical_potentials",
    "battery_dissipation_w",
    "battery_sei_affinity_j_mol",
])
def test_collector_family_does_not_fallback_to_legacy_analysis(name):
    """The new family must reject undeclared analyses instead of averaging cells."""
    # This import is deliberately delayed: the production family is delivered
    # by the physical line after this compatibility contract is authored.
    spec = _collector_spec()
    state = spec.initial_state()
    before = _snapshot(state), spec.to_dict()
    from chemworld.physchem import battery_models

    calls = {
        "battery_chemical_energy_j": lambda: battery_models.battery_chemical_energy_j(spec, state),
        "battery_chemical_potentials": lambda: battery_models.battery_chemical_potentials(
            spec, state
        ),
        "battery_dissipation_w": lambda: battery_models.battery_dissipation_w(spec, state, .1),
        "battery_sei_affinity_j_mol": lambda: battery_models.battery_sei_affinity_j_mol(
            spec, state, .1
        ),
    }
    with pytest.raises((ValueError, TypeError), match=r"(?i)collector|unsupported|undeclared"):
        calls[name]()
    assert (_snapshot(state), spec.to_dict()) == before
