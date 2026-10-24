"""Spatial gas actually consumes original transport, inventories, clock and replay."""

import importlib
import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from chemworld.physchem.finite_volume import DiffusionMesh
from chemworld.physchem.gas_reactor_network import (
    GasFlowConnectionSpec,
    GasReactorNetworkSpec,
    advance_gas_reactor_network,
    gas_reactor_network_system,
    pack_gas_reactor_network_state,
)
from chemworld.physchem.gas_spatial_reactor import spatial_rates
from chemworld.physchem.gas_spatial_transport import GasSpatialTransportSpec


@pytest.fixture(scope="module")
def reference():
    directory = str(Path(__file__).parent / "reference")
    sys.path.insert(0, directory)
    try:
        yield importlib.import_module("run_gas_spatial36")
    finally:
        sys.path.remove(directory)


def assert_checks(record):
    assert all(c["passed"] for c in record.checks), [c for c in record.checks if not c["passed"]]


@pytest.mark.parametrize("model", ("stefan-maxwell", "mixture-averaged"))
@pytest.mark.parametrize("kind", ("ternary", "equimolar", "non-equimolar"))
def test_spatial_species_pressure_face_enthalpy_and_bath_match_independent_x_p(
    reference, model, kind
):
    record = reference.Record(kind)
    reference.physical(record, reference.cases.case(kind, model))
    assert_checks(record)


@pytest.mark.parametrize("kind", ("association", "dissociation", "pure", "pressure-event"))
def test_analytic_closed_limits_and_earliest_common_pressure_stop(reference, kind):
    record = reference.Record(kind)
    reference.physical(record, reference.cases.limit(kind), analytic=True)
    assert_checks(record)


@pytest.mark.parametrize("kind", ("world", "json", "process", "rollback"))
def test_original_world_cell_selection_unique_state_replay_and_rollback(reference, kind, tmp_path):
    record = reference.Record(kind)
    reference.world(record, kind, tmp_path)
    assert_checks(record)


def test_binary_continuous_neumann_solution_converges_under_mesh_refinement(reference):
    records = [reference.Record(str(n)) for n in (4, 8)]
    errors = [reference.grid(record, cells) for record, cells in zip(records, (4, 8), strict=True)]
    for record in records:
        assert_checks(record)
    assert errors[0] / errors[1] >= 3


def test_zero_molar_bulk_does_not_impose_zero_mass_velocity_or_spurious_bath(reference):
    spec, network, amounts = reference.cases.native(reference.cases.case("binary"))
    state = pack_gas_reactor_network_state(spec, network, amounts)
    _, fields = spatial_rates(spec, network, state)
    assert all(f["bulk_mol_s"] == 0 for f in fields["faces"].values())
    assert any(abs(f["mass_velocity_m_s"]) > 1e-8 for f in fields["faces"].values())
    result, diagnostic = advance_gas_reactor_network(spec, network, state, 0.2)
    assert result.final_state.temperature_K == spec.temperature_K
    assert all(abs(n["bath_heat_J"]) < 1e-10 for n in diagnostic["nodes"].values())
    assert all(abs(n["chemical_energy_change_J"]) < 1e-10 for n in diagnostic["nodes"].values())


def test_nonuniform_non_equimolar_reaction_needs_bulk_flow_and_one_pressure(reference):
    spec, network, amounts = reference.cases.native(reference.cases.case())
    state = pack_gas_reactor_network_state(spec, network, amounts)
    chemistry, fields = spatial_rates(spec, network, state)
    assert any(abs(f["bulk_mol_s"]) > 1e-10 for f in fields["faces"].values())
    assert len({sum(c.amount_rates_mol_s.values()) for c in chemistry.values()}) > 1
    result, diagnostic = advance_gas_reactor_network(spec, network, state, 0.4)
    pressures = [row["pressure_Pa"] for row in diagnostic["nodes"].values()]
    assert max(pressures) - min(pressures) < 1e-7
    assert pressures[0] != pytest.approx(100000)
    assert result.final_state.volume_L == state.volume_L


@pytest.mark.parametrize(
    "defect", ("unequal-pressure", "volume", "imposed-flow", "spherical", "mach", "temperature")
)
def test_rejects_inconsistent_spatial_closures(reference, defect):
    spec, network, amounts = reference.cases.native(reference.cases.case())
    spatial = spec.spatial_transport
    with pytest.raises(ValueError):
        if defect == "unequal-pressure":
            amounts["cell-000"]["A"] *= 2
            gas_reactor_network_system(spec, network).validate_state(
                pack_gas_reactor_network_state(spec, network, amounts)
            )
        elif defect == "volume":
            replace(spec, nodes=(replace(spec.nodes[0], volume_L=1), *spec.nodes[1:]))
        elif defect == "imposed-flow":
            replace(
                spec,
                connections=(
                    GasFlowConnectionSpec("flow", "cell-000", "cell-001", coefficient=0.1),
                ),
            )
        elif defect == "spherical":
            replace(spatial, mesh=DiffusionMesh.spherical(0.02, 4))
        elif defect == "mach":
            replace(spatial, maximum_mach=0.1)
        else:
            state = pack_gas_reactor_network_state(spec, network, amounts)
            gas_reactor_network_system(spec, network).validate_state(
                replace(state, temperature_K=500)
            )


def test_spatial_and_legacy_serialization_remain_separate(reference):
    spec, _, _ = reference.cases.native(reference.cases.case())
    payload = json.loads(json.dumps(spec.to_dict()))
    assert GasReactorNetworkSpec.from_dict(payload) == spec
    assert (
        GasSpatialTransportSpec.from_dict(spec.spatial_transport.to_dict())
        == spec.spatial_transport
    )
    old = replace(spec, spatial_transport=None)
    assert "spatial_transport" not in old.to_dict()
    assert old.to_dict()["model"] == "finite-isothermal-gas-network-v1"
    payload["model"] = old.to_dict()["model"]
    with pytest.raises(ValueError, match="model version"):
        GasReactorNetworkSpec.from_dict(payload)


def test_mass_mach_limit_catches_diffusive_mass_velocity_with_zero_molar_bulk(reference):
    spec, network, amounts = reference.cases.native(reference.cases.case("binary"))
    spec = replace(spec, spatial_transport=replace(spec.spatial_transport, maximum_mach=1e-12))
    with pytest.raises(ValueError, match="mass-average"):
        gas_reactor_network_system(spec, network).validate_state(
            pack_gas_reactor_network_state(spec, network, amounts)
        )


def test_nonuniform_mesh_uses_actual_volumes_and_face_positions(reference):
    card = reference.cases.case("binary")
    spec, network, _ = reference.cases.native(card)
    mesh = DiffusionMesh("planar", (0.0, 0.002, 0.005, 0.011, 0.02), area_m2=0.01)
    spatial = replace(spec.spatial_transport, mesh=mesh)
    spec = replace(
        spec,
        nodes=tuple(
            replace(n, volume_L=float(v) * 1000)
            for n, v in zip(spec.nodes, mesh.cell_volumes_m3, strict=True)
        ),
        spatial_transport=spatial,
    )
    amounts = {
        n.vessel_id: {
            s: x * card["pressure"] * n.volume_L / (1000 * reference.cases.R * card["temperature"])
            for s, x in row.items()
        }
        for n, row in zip(spec.nodes, card["fractions"], strict=True)
    }
    _, diagnostic = advance_gas_reactor_network(
        spec, network, pack_gas_reactor_network_state(spec, network, amounts), 0.2
    )
    assert all(
        r["pressure_Pa"] == pytest.approx(card["pressure"]) for r in diagnostic["nodes"].values()
    )
    assert diagnostic["energy_residual_J"] == pytest.approx(0, abs=1e-8)


@pytest.mark.parametrize("invalid", (False, True))
def test_single_cell_validates_transport_without_needing_an_internal_face(reference, invalid):
    card = reference.cases.case("pure", cells=1)
    spec, network, amounts = reference.cases.native(card)
    if invalid:
        other, _, _ = reference.cases.native(reference.cases.case("binary"))
        spec = replace(
            spec,
            spatial_transport=replace(
                spec.spatial_transport, transport=other.spatial_transport.transport
            ),
        )
        with pytest.raises(ValueError, match="exactly the original"):
            gas_reactor_network_system(spec, network)
    else:
        _, diagnostic = advance_gas_reactor_network(
            spec, network, pack_gas_reactor_network_state(spec, network, amounts), 1.0
        )
        assert diagnostic["nodes"]["cell-000"]["amounts_mol"] == amounts["cell-000"]
        assert diagnostic["spatial"]["faces"] == {}
