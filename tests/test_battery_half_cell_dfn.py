"""Two real salt domains, finite metal and the original coupled DFN residual."""

import json
from dataclasses import asdict, replace

import numpy as np
import pytest

from chemworld.data.scientific_inputs import (
    InputSource,
    InputValue,
    QuantityColumn,
    ScientificRow,
    ScientificTable,
)
from chemworld.physchem.battery_half_cell import (
    METAL_KEY,
    METAL_UNITS,
    BatteryHalfCellSPMSpec,
    LithiumMetalCounterElectrodeSpec,
    half_cell_readout,
)
from chemworld.physchem.battery_half_cell_dfn import (
    HALF_CELL_DFN_FAMILY,
    PHI_E,
    BatteryHalfCellDFNSpec,
    BatteryHalfCellElectrolyteSpec,
    LithiumMetalSaltKineticsSpec,
    advance_half_cell_dfn,
    half_cell_dfn_control_current,
    half_cell_dfn_control_mode,
    half_cell_dfn_fields,
    half_cell_dfn_readout,
    half_cell_dfn_system,
    validate_half_cell_dfn_state,
)
from chemworld.physchem.battery_materials import ElectrodeOCV, ElectrodeOCVEndpointPolicy
from chemworld.physchem.battery_models import BATTERY_ALGEBRAIC, F, ParticleSpec
from chemworld.physchem.coupled_system import CoupledStateView, assemble_coupled_rates
from chemworld.physchem.data_conflicts import DataSourceProvenance
from chemworld.physchem.reactor_shared import ScalarControlProgram
from chemworld.physchem.residual_system import finite_jacobian


def source():
    return InputSource(
        DataSourceProvenance(
            "half-cell-dfn-functional",
            0,
            "Authored synthetic two-domain half-cell functional inputs",
            "1",
            "CC0-1.0",
        ),
        "synthetic",
    )


def parameter_table(values, units, identity):
    return ScientificTable(
        identity,
        "parameters",
        tuple(
            QuantityColumn(k, units[k], "parameter", (0, 1e10), "Fixed synthetic functional input")
            for k in values
        ),
        (source(),),
        (
            ScientificRow(
                "parameters",
                source().provenance.source_id,
                "synthetic",
                "Li",
                "parameters",
                {
                    k: InputValue(v, units[k], None, "Fixed synthetic functional input")
                    for k, v in values.items()
                },
            ),
        ),
    )


def metal(**changes):
    values = {
        "area_m2": 0.01,
        "thickness_m": 1e-5,
        "conductivity_S_m": 1e7,
        "partial_molar_volume_m3_mol": 1.3e-5,
        "exchange_current_density_A_m2": 10.0,
        "reference_temperature_K": 298.15,
        "electrolyte_concentration_mol_m3": 1000.0,
    }
    values.update(changes)
    return LithiumMetalCounterElectrodeSpec(
        parameter_table(values, METAL_UNITS, "half-cell-dfn-metal"), "parameters"
    )


def kinetics(exponent=0.3):
    return LithiumMetalSaltKineticsSpec(
        parameter_table(
            {"salt_exponent": exponent},
            {"salt_exponent": "dimensionless"},
            "half-cell-dfn-kinetics",
        ),
        "parameters",
    )


def model(**changes):
    values = {
        "metal": metal(),
        "positive": ParticleSpec(cells=5, mesh_exponent=1, active_volume_m3=4e-7),
        "positive_ocv": ElectrodeOCV("working", source(), (4.2, -0.8)),
        "metal_kinetics": kinetics(),
        "electrolyte": BatteryHalfCellElectrolyteSpec(area_m2=0.01, cells=(3, 4)),
        "maximum_current_A": 0.4,
    }
    values.update(changes)
    return BatteryHalfCellDFNSpec(**values)


def program(value, unit="A"):
    return ScalarControlProgram(((0.0, value),), unit)


def test_codec_real_two_domain_layout_and_fixed_foil_geometry():
    spec = model()
    restored = BatteryHalfCellDFNSpec.from_dict(json.loads(json.dumps(spec.to_dict())))
    assert restored == spec
    assert restored.initial_state() == spec.initial_state()
    assert spec.model_family == spec.model_id == HALF_CELL_DFN_FAMILY
    assert not hasattr(spec, "negative")
    assert spec.electrolyte.region_slices == {"separator": slice(0, 3), "positive": slice(3, 7)}
    assert len(spec.inventory_keys) == 1 + 7 + 4 * 5
    assert not any(k.startswith("negative:") for k in spec.inventory_keys)
    assert len(spec.algebraic_names) == 4 + 2
    assert sum(p.active_volume_m3 for _, _, p, _, _ in spec.particle_bins) == pytest.approx(
        spec.positive.active_volume_m3
    )
    assert spec.volume_l == pytest.approx(1000 * 0.01 * (1e-5 + 2.5e-5 + 1e-4))
    assert spec.metal.initial_amount_mol == pytest.approx(0.01 * 1e-5 / 1.3e-5)
    assert spec.metal.resistance_ohm == pytest.approx(1e-5 / (1e7 * 0.01))
    assert spec.initial_stoichiometric_discharge_bound_c == spec.usable_charge_c
    with pytest.raises(ValueError, match="fields"):
        BatteryHalfCellDFNSpec.from_dict({**spec.to_dict(), "negative": {}})
    encoded = spec.to_dict()
    del encoded["metal_kinetics"]
    with pytest.raises(ValueError, match="fields"):
        BatteryHalfCellDFNSpec.from_dict(encoded)


@pytest.mark.parametrize(
    "changes",
    [
        {"cells": (0, 4)},
        {"cells": (3, 0, 4)},
        {"cells": (True, 4)},
        {"lengths_m": (0, 1e-4)},
        {"porosities": (0.5, 1)},
        {"cation_transference": 1.0},
        {"minimum_concentration_mol_m3": 1000.0},
    ],
)
def test_two_domain_geometry_rejects_false_negative_domains_and_invalid_salt(changes):
    with pytest.raises(ValueError):
        BatteryHalfCellElectrolyteSpec(**changes)


def test_spec_rejects_area_temperature_volume_and_material_mismatch():
    for changes in (
        {"metal": metal(area_m2=0.02)},
        {"temperature_K": 300.0},
        {"positive": ParticleSpec(active_volume_m3=1e-5)},
        {"positive_ocv": ElectrodeOCV("wrong", source(), (3.0, 0.1))},
        {"working_reference_concentration_mol_m3": 0.0},
    ):
        with pytest.raises(ValueError):
            model(**changes)
    with pytest.raises(ValueError):
        kinetics(-0.3)
    with pytest.raises(ValueError, match="thermal"):
        half_cell_dfn_system(model(), program(0), thermal_device=object())
    with pytest.raises(ValueError, match="programme"):
        half_cell_dfn_system(model(), program(0.5))


@pytest.mark.parametrize("current", [-0.2, 0.0, 0.2])
def test_local_bv_charge_boundaries_metal_gauge_and_voltage_decomposition(current):
    spec = model()
    initial = spec.initial_state()
    f = half_cell_dfn_fields(spec, initial, current_A=current)
    e = spec.electrolyte
    assert sum(f["reaction_current_A"]) == pytest.approx(-current, abs=1e-14)
    face = f["electrolyte_face_current_A_m2"]
    assert face[0] == pytest.approx(current / e.area_m2)
    assert face[-1] == pytest.approx(0.0, abs=1e-12)
    np.testing.assert_allclose(
        np.diff(face) * e.area_m2, np.r_[np.zeros(e.cells[0]), f["reaction_current_A"]], atol=1e-14
    )
    assert f["positive_solid_face_current_A_m2"][0] == pytest.approx(0.0, abs=1e-12)
    assert f["positive_solid_face_current_A_m2"][-1] == pytest.approx(current / e.area_m2)
    np.testing.assert_allclose(f["bv_residual_V"], 0, atol=1e-11)
    assert f["metal_boundary_residual_V"] == pytest.approx(0, abs=1e-11)
    assert f["metal_electrolyte_potential_V"] == pytest.approx(
        -f["metal_overpotential_V"] - f["metal_ohmic_V"], abs=1e-11
    )
    read = half_cell_dfn_readout(spec, initial, current)
    assert "negative_surface_fraction" not in asdict(read)
    assert read.voltage_V == pytest.approx(
        np.mean(f["open_circuit_potential_V"])
        + read.working_overpotential_V
        - read.metal_overpotential_V
        - read.metal_ohmic_V
        + read.concentration_overpotential_V
        + read.electrolyte_ohmic_V
        + read.solid_ohmic_V,
        abs=1e-11,
    )
    if current:
        assert np.ptp(f["reaction_current_A"]) > 1e-7
        assert (read.voltage_V - (4.2 - 0.8 * 0.3)) * current < 0
    else:
        np.testing.assert_allclose(f["reaction_current_A"], 0, atol=1e-12)
        assert read.voltage_V == pytest.approx(3.96)


@pytest.mark.parametrize("current", [-0.2, 0.2])
def test_quadratic_boundary_reconstruction_includes_the_real_nonzero_gradient(current):
    spec = model()
    e = spec.electrolyte
    initial = spec.initial_state()
    h = e.lengths_m[0] / e.cells[0]
    g = (
        -(1 - e.cation_transference)
        * current
        / (e.area_m2 * F * e.diffusivity_m2_s * e.transport_efficiencies[0])
    )
    a, curvature = 800.0, 1e10
    amounts = dict(initial.amounts_mol)
    for i in range(2):
        lo, hi = i * h, (i + 1) * h
        average = a + g * (lo + hi) / 2 + curvature * (lo * lo + lo * hi + hi * hi) / 3
        amounts[e.keys[i]] = average * e.diffusion.storage_volumes_m3[i]
    state = replace(initial, amounts_mol=amounts)
    assert e.boundary_concentrations(state, current)[0] == pytest.approx(a, abs=1e-10)
    assert state.amounts_mol == amounts


def test_boundary_cutoff_can_detect_salt_failure_before_any_cell_average():
    spec = model()
    initial = spec.initial_state()
    e = replace(spec.electrolyte, minimum_concentration_mol_m3=999.0)
    spec = replace(spec, electrolyte=e)
    current = -0.2
    assert min(e.concentrations(initial)) > e.minimum_concentration_mol_m3
    assert e.boundary_concentrations(initial, current)[0] < e.minimum_concentration_mol_m3
    event = next(
        event
        for event in half_cell_dfn_system(spec, program(current)).events
        if event.event_id == "half-cell-dfn:electrolyte:depleted"
    )
    assert event.value(initial) < 0


def test_metal_and_working_reference_salt_scales_do_not_drift_with_initial_salt():
    spec = model()
    low = replace(spec, electrolyte=replace(spec.electrolyte, initial_concentration_mol_m3=500.0))
    for actual in (spec, low):
        f = half_cell_dfn_fields(actual, actual.initial_state(), current_A=0)
        concentration = actual.electrolyte.initial_concentration_mol_m3
        assert f["metal_exchange_current_density_A_m2"] == pytest.approx(
            10 * (concentration / 1000) ** 0.3
        )
        assert f["exchange_current_density_A_m2"][0] == pytest.approx(
            actual.positive.exchange_current_scale_A_m2 * np.sqrt(concentration / 1000 * 0.3 * 0.7)
        )
    assert low.metal.native_parameters["electrolyte_concentration_mol_m3"] == 1000.0
    assert low.working_reference_concentration_mol_m3 == 1000.0


@pytest.mark.parametrize("current", [-0.2, 0.2])
def test_one_shared_rhs_has_exact_salt_and_both_solid_faradaic_sources(current):
    spec = model()
    initial = spec.initial_state()
    system = half_cell_dfn_system(spec, program(current))
    f = half_cell_dfn_fields(spec, initial, current_A=current)
    values = {
        **dict(zip(spec.reaction_keys, f["reaction_current_A"][:-1], strict=True)),
        PHI_E: f["metal_electrolyte_potential_V"],
        BATTERY_ALGEBRAIC[0]: current,
        BATTERY_ALGEBRAIC[1]: f["voltage_V"],
    }
    state = CoupledStateView(
        initial.amounts_mol,
        initial.temperature_K,
        initial.volume_L,
        coordinates=initial.coordinates,
        algebraic_values=values,
    )
    rates = assemble_coupled_rates(
        state,
        processes=system.processes,
        thermodynamics=system.thermodynamics,
        heat_flux=system.heat_flux,
        isothermal=True,
    )
    salt = np.array([rates.amount_rates_mol_s[k] for k in spec.electrolyte.keys])
    expected = np.r_[np.zeros(spec.electrolyte.cells[0]), f["reaction_current_A"]] * (1 - 0.4) / F
    expected[0] += (1 - 0.4) * current / F
    np.testing.assert_allclose(salt, expected, atol=1e-16)
    assert sum(salt) == pytest.approx(0, abs=1e-15)
    assert rates.amount_rates_mol_s[METAL_KEY] == pytest.approx(-current / F)
    assert sum(
        rates.amount_rates_mol_s[k] for _, _, _, keys, _ in spec.particle_bins for k in keys
    ) == pytest.approx(current / F, abs=1e-15)
    assert rates.coordinate_rates["discharged_charge_C"] == current
    assert rates.coordinate_rates["delivered_work_J"] == current * f["voltage_V"]


@pytest.mark.parametrize("value,unit", [(0.17, "A"), (3.93, "V"), (2.6, "V"), (4.4, "V")])
def test_complete_rhs_jacobian_matches_independent_scaled_differences(value, unit):
    spec = model()
    drive = program(value, unit)
    system = half_cell_dfn_system(spec, drive)
    initial = spec.initial_state()
    material, coords, names = spec.inventory_keys, spec.coordinate_keys, spec.algebraic_names
    identities = (*material, "temperature", *coords, *names)
    guess = {
        **initial.amounts_mol,
        "temperature": initial.temperature_K,
        **initial.coordinates,
        **system.algebraic.initial_values(initial),
    }
    y = np.array([guess[key] for key in identities])
    scales = np.r_[
        [system.inventory_scales_mol[k] for k in material],
        300.0,
        np.ones(2),
        [v.scale for v in system.algebraic.variables],
    ]

    def view(vector):
        values = dict(zip(identities, vector, strict=True))
        return CoupledStateView(
            {k: values[k] for k in material},
            values["temperature"],
            initial.volume_L,
            coordinates={k: values[k] for k in coords},
            algebraic_values={k: values[k] for k in names},
        )

    def rhs(vector):
        state = view(vector)
        rates = assemble_coupled_rates(
            state,
            processes=system.processes,
            thermodynamics=system.thermodynamics,
            heat_flux=system.heat_flux,
            isothermal=True,
        )
        return np.r_[
            [rates.amount_rates_mol_s[k] for k in material],
            0.0,
            [rates.coordinate_rates[k] for k in coords],
            -np.asarray(system.algebraic.residual(state)),
        ]

    expected = finite_jacobian(rhs, y, scales, relative_step=1e-6) * scales
    actual = system.residual_rhs_jacobian(view(y), identities).toarray() * scales
    np.testing.assert_allclose(actual, expected, rtol=3e-6, atol=3e-8)
    permutation = np.arange(len(identities))[::-1]
    reversed_names = tuple(identities[i] for i in permutation)
    reversed_jac = system.residual_rhs_jacobian(view(y), reversed_names).toarray()
    np.testing.assert_allclose(
        reversed_jac,
        system.residual_rhs_jacobian(view(y), identities).toarray()[
            np.ix_(permutation, permutation)
        ],
    )


@pytest.mark.parametrize(
    "target,mode",
    [(3.93, "constant_voltage"), (2.6, "positive_current_limit"), (4.4, "negative_current_limit")],
)
def test_finite_cv_jointly_solves_metal_working_and_the_real_source_bound(target, mode):
    spec = model()
    drive = program(target, "V")
    fields = half_cell_dfn_fields(spec, spec.initial_state(), program=drive)
    assert half_cell_dfn_control_mode(spec, spec.initial_state(), drive)[0] == mode
    current = half_cell_dfn_control_current(spec, spec.initial_state(), drive)
    assert current == pytest.approx(fields["current_A"])
    if mode == "constant_voltage":
        assert fields["voltage_V"] == pytest.approx(target, abs=1e-11)
        assert abs(current) < spec.maximum_current_A
    else:
        assert abs(current) == pytest.approx(spec.maximum_current_A)
        assert (fields["voltage_V"] - target) * current > 0


@pytest.mark.parametrize("value,unit", [(0.0, "A"), (0.15, "A"), (-0.15, "A"), (3.93, "V")])
def test_short_real_coupled_advance_preserves_all_unique_inventories_and_work(value, unit):
    spec = model()
    initial = spec.initial_state()
    result = advance_half_cell_dfn(spec, initial, program(value, unit), 2.0)
    assert result.outcome == "success", result.solver_diagnostic
    final = result.final_state
    validate_half_cell_dfn_state(spec, final)
    for name, residual in result.conservation_residuals.items():
        assert abs(residual) < (1e-8 if name.endswith("_C") else 1e-12)
    charge = final.coordinates["discharged_charge_C"]
    assert F * (final.amounts_mol[METAL_KEY] - initial.amounts_mol[METAL_KEY]) == pytest.approx(
        -charge, abs=1e-8
    )
    assert sum(final.amounts_mol[k] for k in spec.electrolyte.keys) == pytest.approx(
        sum(initial.amounts_mol[k] for k in spec.electrolyte.keys), abs=1e-13
    )
    if unit == "A":
        assert charge == pytest.approx(value * 2, abs=1e-8)
    if value == 0:
        assert final.amounts_mol == pytest.approx(initial.amounts_mol, abs=1e-13)
    else:
        assert np.ptp(spec.electrolyte.concentrations(final)) > 0.1
        assert final.coordinates["delivered_work_J"] * charge > 0


def test_rest_from_nonuniform_salt_permits_internal_opposed_local_currents():
    spec = model()
    first = advance_half_cell_dfn(spec, spec.initial_state(), program(0.2), 2.0)
    assert first.outcome == "success"
    rest = half_cell_dfn_fields(spec, first.final_state, current_A=0.0)
    q = rest["reaction_current_A"]
    assert min(q) < -1e-6 < 0 < 1e-6 < max(q)
    assert sum(q) == pytest.approx(0.0, abs=1e-14)
    final = advance_half_cell_dfn(spec, first.final_state, program(0.0), 2.0)
    assert final.outcome == "success"
    assert final.final_state.amounts_mol[METAL_KEY] == pytest.approx(
        first.final_state.amounts_mol[METAL_KEY], abs=1e-14
    )


def test_high_transport_limit_matches_the_actual_existing_spm_half_cell():
    base = model()
    spec = replace(
        base,
        electrolyte=replace(
            base.electrolyte,
            diffusivity_m2_s=1e-3,
            conductivity_S_m=1e8,
            positive_solid_conductivity_S_m=1e8,
        ),
    )
    spm = BatteryHalfCellSPMSpec(
        spec.metal, spec.positive, spec.positive_ocv, maximum_current_A=spec.maximum_current_A
    )
    for current in (-0.2, 0.2):
        actual = half_cell_dfn_readout(spec, spec.initial_state(), current)
        limit = half_cell_readout(spm, spm.initial_state(), current)
        assert actual.voltage_V == pytest.approx(limit.voltage_V, abs=2e-7)
        assert actual.working_surface_fraction == pytest.approx(
            limit.working_surface_fraction, abs=1e-8
        )


def test_fixed_interface_empty_metal_stops_forward_and_can_rest_or_plate_in_reverse():
    spec = model(metal=metal(thickness_m=1e-9))
    drive = program(0.2)
    result = advance_half_cell_dfn(spec, spec.initial_state(), drive, 1.0)
    assert result.outcome == "stopped", result.solver_diagnostic
    assert result.stopping_event == "half-cell-dfn:metal:depleted"
    final = result.final_state
    assert final.amounts_mol[METAL_KEY] == pytest.approx(0.0, abs=1e-12)
    assert final.time_s == pytest.approx(F * spec.metal.initial_amount_mol / 0.2, abs=1e-7)
    again = advance_half_cell_dfn(spec, final, drive, 0.1)
    assert again.outcome == "stopped"
    for value in (0.0, -0.1):
        resumed = advance_half_cell_dfn(spec, final, program(value), 0.1)
        assert resumed.outcome == "success", resumed.solver_diagnostic
        assert resumed.final_state.amounts_mol[METAL_KEY] >= final.amounts_mol[METAL_KEY]


def test_segmented_cc_rest_reverse_preserves_the_original_shared_clock():
    spec = model()
    drive = ScalarControlProgram(((0.0, 0.12), (1.0, 0.0), (2.0, -0.08)), "A")
    whole = advance_half_cell_dfn(spec, spec.initial_state(), drive, 3.0)
    first = advance_half_cell_dfn(spec, spec.initial_state(), drive, 1.0)
    final = advance_half_cell_dfn(spec, first.final_state, drive, 2.0)
    assert whole.outcome == first.outcome == final.outcome == "success"
    assert final.final_state.time_s == 3.0
    assert final.final_state.amounts_mol == pytest.approx(whole.final_state.amounts_mol, abs=1e-10)
    assert final.final_state.coordinates == pytest.approx(whole.final_state.coordinates, abs=1e-6)


@pytest.mark.parametrize(
    "fraction,current,label", [(0.9895, 0.1, "saturated"), (0.0105, -0.1, "depleted")]
)
def test_all_working_surfaces_stop_at_a_real_finite_time(fraction, current, label):
    spec = model(
        initial_positive_fraction=fraction,
        positive=ParticleSpec(
            cells=5, mesh_exponent=1, active_volume_m3=4e-7, diffusivity_m2_s=1e-12
        ),
    )
    result = advance_half_cell_dfn(spec, spec.initial_state(), program(current), 3.0)
    assert result.outcome == "stopped", result.solver_diagnostic
    assert result.stopping_event == "half-cell-dfn:working:" + label
    assert 0 < result.actual_duration_s < 3
    fields = half_cell_dfn_fields(spec, result.final_state, current_A=current)
    stopping = max(fields["surface_fraction"]) if current > 0 else min(fields["surface_fraction"])
    assert stopping == pytest.approx(
        1 - spec.surface_margin if current > 0 else spec.surface_margin, abs=1e-8
    )


def test_salt_event_stops_at_the_real_metal_boundary_before_any_cell_depletes():
    base = model()
    spec = replace(base, electrolyte=replace(base.electrolyte, minimum_concentration_mol_m3=995.0))
    current = -0.2
    initial = spec.initial_state()
    assert spec.electrolyte.minimum_resolved_concentration(initial, current) > 995.0
    result = advance_half_cell_dfn(spec, initial, program(current), 0.2)
    assert result.outcome == "stopped", result.solver_diagnostic
    assert result.stopping_event == "half-cell-dfn:electrolyte:depleted"
    assert 0 < result.actual_duration_s < 0.2
    e = spec.electrolyte
    assert e.boundary_concentrations(result.final_state, current)[0] == pytest.approx(
        995.0, abs=1e-6
    )
    assert min(e.concentrations(result.final_state)) > 995.0


@pytest.mark.parametrize(
    "current,limits,event",
    [(0.2, (3.85, 4.5), "minimum-voltage"), (-0.2, (2.5, 4.1), "maximum-voltage")],
)
def test_loaded_terminal_voltage_cutoff_is_distinct_from_the_initial_ocv(current, limits, event):
    spec = model(minimum_voltage_V=limits[0], maximum_voltage_V=limits[1])
    assert limits[0] < 3.96 < limits[1]
    result = advance_half_cell_dfn(spec, spec.initial_state(), program(current), 0.1)
    assert result.outcome == "stopped", result.solver_diagnostic
    assert result.stopping_event == "half-cell-dfn:" + event
    assert result.actual_duration_s == 0.0


def test_snapshot_validation_rejects_inconsistent_salt_and_metal_charge():
    spec = model()
    initial = spec.initial_state()
    for key in (METAL_KEY, spec.electrolyte.keys[0]):
        amounts = {**initial.amounts_mol, key: initial.amounts_mol[key] * 1.01}
        with pytest.raises(ValueError, match="balance"):
            validate_half_cell_dfn_state(spec, replace(initial, amounts_mol=amounts))


def test_two_physical_roots_from_metal_salt_fold_are_kept_in_cc_and_rejected_in_cv():
    base = model(metal=metal(exchange_current_density_A_m2=1e-6), metal_kinetics=kinetics(3.0))
    spec = replace(
        base,
        positive=replace(base.positive, exchange_current_scale_A_m2=1e10, diffusivity_m2_s=1e-10),
        positive_ocv=replace(base.positive_ocv, polynomial_V=(4.0,)),
        electrolyte=replace(
            base.electrolyte,
            diffusivity_m2_s=1e-14,
            conductivity_S_m=1e10,
            positive_solid_conductivity_S_m=1e10,
            thermodynamic_factor=1e-10,
        ),
    )
    # Independent review's two-root witness. Both CC states satisfy the same
    # V=3.8 CV constraint, with all real surfaces and salt strictly in domain.
    for current in (4.903217707214928e-7, 0.129204611156221):
        fields = half_cell_dfn_fields(spec, spec.initial_state(), current_A=current)
        assert fields["voltage_V"] == pytest.approx(3.8, abs=2e-10)
        assert min(fields["surface_fraction"]) > spec.surface_margin
        assert max(fields["surface_fraction"]) < 1 - spec.surface_margin
        assert (
            spec.electrolyte.minimum_resolved_concentration(spec.initial_state(), current)
            > spec.electrolyte.minimum_concentration_mol_m3
        )
        np.testing.assert_allclose(fields["bv_residual_V"], 0.0, atol=1e-10)
    with pytest.raises(ValueError, match="monotonicity certificate"):
        half_cell_dfn_fields(spec, spec.initial_state(), program=program(3.8, "V"))
    with pytest.raises(ValueError, match="monotonicity certificate"):
        half_cell_dfn_system(spec, program(3.8, "V"))


def test_physical_interval_ocv_monotonicity_does_not_certify_unbounded_fv_trials():
    base = model()
    # U'=-.8+.2*theta is negative on the accepted physical interval, but
    # becomes positive in the actual unbounded surface extrapolation.
    spec = replace(base, positive_ocv=replace(base.positive_ocv, polynomial_V=(4.2, -0.8, 0.1)))
    assert half_cell_dfn_fields(spec, spec.initial_state(), current_A=0.1)["current_A"] == 0.1
    with pytest.raises(ValueError, match="monotonicity certificate"):
        half_cell_dfn_fields(spec, spec.initial_state(), program=program(3.94, "V"))


@pytest.mark.parametrize("endpoint", [False, True])
def test_cv_certificate_covers_global_term_signs_or_the_complete_clipped_interval(endpoint):
    base = model()
    curve = (
        replace(
            base.positive_ocv,
            polynomial_V=(4.2, -0.8, 0.1),
            endpoint_policy=ElectrodeOCVEndpointPolicy(source()),
        )
        if endpoint
        else replace(
            base.positive_ocv,
            exponential_terms=((0.02, -3.0, 0.5),),
            tanh_terms=((-0.03, 4.0, 0.5),),
        )
    )
    spec = replace(base, positive_ocv=curve)
    assert spec.cv_monotone_certified
    fields = half_cell_dfn_fields(spec, spec.initial_state(), program=program(3.94, "V"))
    assert fields["voltage_V"] == pytest.approx(3.94, abs=1e-10)


def test_cv_certificate_rejects_nonpositive_reconstructed_salt_base():
    spec = model()
    initial = spec.initial_state()
    e = spec.electrolyte
    c = e.concentrations(initial)
    c[0], c[2] = 1.0, c[2] + 999.0
    state = replace(
        initial,
        amounts_mol={
            **initial.amounts_mol,
            **dict(zip(e.keys, c * e.diffusion.storage_volumes_m3, strict=True)),
        },
    )
    validate_half_cell_dfn_state(spec, state)
    assert e.boundary_concentrations(state, 0.0)[0] < 0
    with pytest.raises(ValueError, match="positive reconstructed salt base"):
        half_cell_dfn_fields(spec, state, program=program(3.94, "V"))


@pytest.mark.parametrize("change", [{"temperature_K": 310.0}, {"volume_L": 0.001485}])
def test_public_fields_and_readout_reject_undeclared_persistent_temperature_or_volume(change):
    spec = model()
    state = replace(spec.initial_state(), **change)
    with pytest.raises(ValueError, match="isothermal fixed volume"):
        half_cell_dfn_fields(spec, state, current_A=0.0)
    with pytest.raises(ValueError, match="isothermal fixed volume"):
        half_cell_dfn_readout(spec, state, 0.0)


def test_newton_view_can_still_probe_temperature_and_volume_without_mutating_the_device():
    spec = model()
    initial = spec.initial_state()
    trial = CoupledStateView(
        initial.amounts_mol, 310.0, initial.volume_L * 1.1, coordinates=initial.coordinates
    )
    assert half_cell_dfn_fields(spec, trial, current_A=0.0)["voltage_V"] == pytest.approx(3.96)
    assert initial.temperature_K == spec.temperature_K
    assert initial.volume_L == spec.volume_l


def test_nonuniform_empty_metal_cv_at_the_exact_zero_current_port_can_rest():
    spec = model(metal=metal(thickness_m=1e-9))
    initial = spec.initial_state()
    amounts = dict(initial.amounts_mol)
    ce = np.linspace(700.0, 1300.0, len(spec.electrolyte.keys))
    volumes = spec.electrolyte.diffusion.storage_volumes_m3
    ce += 1000.0 - np.average(ce, weights=volumes)
    amounts.update(zip(spec.electrolyte.keys, ce * volumes, strict=True))
    delta = spec.metal.initial_amount_mol / spec.positive.capacity_mol
    for theta, (_, _, particle, keys, _) in zip(
        (0.35, 0.32, 0.28, 0.25), spec.particle_bins, strict=True
    ):
        amounts.update(
            zip(
                keys,
                (theta + delta)
                * particle.maximum_concentration_mol_m3
                * particle.mesh.cell_volumes_m3,
                strict=True,
            )
        )
    amounts[METAL_KEY] = 0.0
    state = replace(
        initial,
        amounts_mol=amounts,
        coordinates={
            "discharged_charge_C": F * spec.metal.initial_amount_mol,
            "delivered_work_J": 0.0,
        },
    )
    validate_half_cell_dfn_state(spec, state)
    target = half_cell_dfn_fields(spec, state, current_A=0.0)["voltage_V"]
    drive = program(target, "V")
    fields = half_cell_dfn_fields(spec, state, program=drive)
    assert abs(fields["current_A"]) < 1e-12
    event = next(
        e
        for e in half_cell_dfn_system(spec, drive).events
        if e.event_id == "half-cell-dfn:metal:depleted"
    )
    assert event.value(state) > 0.0
    values = {
        **dict(zip(spec.reaction_keys, fields["reaction_current_A"][:-1], strict=True)),
        PHI_E: fields["metal_electrolyte_potential_V"],
        BATTERY_ALGEBRAIC[0]: fields["current_A"],
        BATTERY_ALGEBRAIC[1]: fields["voltage_V"],
    }
    trial = CoupledStateView(
        state.amounts_mol,
        state.temperature_K,
        state.volume_L,
        coordinates=state.coordinates,
        algebraic_values=values,
    )
    system = half_cell_dfn_system(spec, drive)
    rates = assemble_coupled_rates(
        trial,
        processes=system.processes,
        thermodynamics=system.thermodynamics,
        heat_flux=system.heat_flux,
        isothermal=True,
    )
    # The numerical-rest event classification never clips the actual signed rate.
    assert rates.amount_rates_mol_s[METAL_KEY] == -fields["current_A"] / F


@pytest.mark.parametrize("exponent", [0.0, 1.0])
def test_cv_certificate_includes_both_passive_salt_exponent_endpoints(exponent):
    spec = model(metal_kinetics=kinetics(exponent))
    assert spec.cv_monotone_certified
    fields = half_cell_dfn_fields(spec, spec.initial_state(), program=program(3.93, "V"))
    assert fields["voltage_V"] == pytest.approx(3.93, abs=1e-10)


def test_cv_trial_requires_the_bulk_fraction_used_in_its_global_passivity_proof():
    spec = model()
    initial = spec.initial_state()
    _, _, particle, keys, _ = spec.particle_bins[0]
    amounts = {
        **initial.amounts_mol,
        keys[-1]: 1.01 * particle.mesh.cell_volumes_m3[-1] * particle.maximum_concentration_mol_m3,
    }
    trial = CoupledStateView(
        amounts, initial.temperature_K, initial.volume_L, coordinates=initial.coordinates
    )
    with pytest.raises(ValueError, match="last-cell bulk fractions"):
        half_cell_dfn_fields(spec, trial, program=program(3.93, "V"))
