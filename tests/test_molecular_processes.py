"""Same-reference local closure and open-boundary steady process balances."""

import json
from dataclasses import replace

import pytest

from chemworld.physchem.molecular_mixture import MolecularStream
from chemworld.physchem.molecular_processes import (
    MolecularHeatDuty,
    MolecularPressureStage,
    MolecularProcessDomain,
    molecular_heat_duty,
    molecular_mix_streams,
    molecular_process_state,
    molecular_split_stream,
)
from chemworld.physchem.residual_system import RootPolicy
from reference.run_molecular_processes import domain, ideal, stream


@pytest.mark.parametrize("real", [False, True])
@pytest.mark.parametrize("coordinate", ["T", "H", "S"])
def test_local_tp_ph_ps_same_caloric_state(real, coordinate):
    d = domain(real)
    target = stream(d, 450, 3e5).state
    result = molecular_process_state(
        d,
        pressure_Pa=3e5,
        coordinate=coordinate,
        value={"T": target.temperature_K, "H": target.enthalpy_J_mol, "S": target.entropy_J_mol_K}[
            coordinate
        ],
        mole_fractions=target.mole_fractions,
        initial_temperature_K=400.0,
    )
    assert result.success, result.to_dict()
    assert result.state.temperature_K == pytest.approx(target.temperature_K, rel=1e-9)
    assert result.state.molar_density_mol_m3 == pytest.approx(target.molar_density_mol_m3, rel=1e-9)


@pytest.mark.parametrize(
    "mode,pressure", [("compressor", 4e5), ("expander", 1e5), ("throttle", 1e5)]
)
def test_pressure_devices_energy_entropy_and_efficiency(mode, pressure):
    d, feed = domain(), stream(domain())
    eta = 1.0 if mode == "throttle" else 0.8
    result = MolecularPressureStage(d, mode, pressure, eta).evaluate(feed)
    assert result.success, result.to_dict()
    assert result.diagnostics["energy_residual_W"] == pytest.approx(0.0, abs=1e-6)
    assert result.diagnostics["entropy_generation_W_K"] >= -1e-7
    assert result.outlet.molar_rate_mol_s == feed.molar_rate_mol_s
    cp = ideal(400, 100, feed.state.mole_fractions)["Cp"]
    iso = 400 * (pressure / 2e5) ** (d.model.gas_constant_J_mol_K / cp)
    expected = (
        400
        if mode == "throttle"
        else 400 + (iso - 400) / (eta if mode == "compressor" else 1 / eta)
    )
    assert result.outlet.state.temperature_K == pytest.approx(expected, rel=1e-9)


def test_steady_mix_split_and_heat_transfer_reservoir():
    d = domain()
    feeds = (
        stream(d, 350, fractions=(1.0, 0.0), rate=0.02),
        stream(d, 500, fractions=(0.0, 1.0), rate=0.03),
    )
    mixed = molecular_mix_streams(d, feeds, outlet_pressure_Pa=2e5, initial_temperature_K=400.0)
    assert mixed.success, mixed.to_dict()
    assert mixed.outlet.state.mole_fractions == pytest.approx((0.4, 0.6))
    assert mixed.diagnostics["entropy_generation_W_K"] > 0
    split = molecular_split_stream(d, mixed.outlet, (0.25, 0.75, 0.0))
    assert [s.molar_rate_mol_s for s in split.outlets] == pytest.approx((0.0125, 0.0375, 0.0))
    assert split.diagnostics["energy_residual_W"] == pytest.approx(0, abs=1e-8)
    for q, bath in ((20, 800), (-20, 250)):
        heated = molecular_heat_duty(d, mixed.outlet, heat_duty_W=q, boundary_temperature_K=bath)
        assert heated.success and heated.diagnostics["entropy_generation_W_K"] >= 0
        assert heated.outlet.enthalpy_rate_W - mixed.outlet.enthalpy_rate_W == pytest.approx(q)
    bad = molecular_heat_duty(d, mixed.outlet, heat_duty_W=20, boundary_temperature_K=300)
    assert not bad.success and bad.failure_reason == "incompatible_heat_reservoir"


def test_domain_reference_direction_and_unreachable_targets_are_not_coerced():
    d = domain()
    feed = stream(d)
    bad = molecular_process_state(
        d,
        pressure_Pa=2e5,
        coordinate="H",
        value=1e9,
        mole_fractions=(0.25, 0.75),
        initial_temperature_K=400.0,
    )
    assert not bad.success and bad.state is None
    limit = molecular_process_state(
        d,
        pressure_Pa=5e5,
        coordinate="T",
        value=600,
        mole_fractions=(0.25, 0.75),
        initial_temperature_K=300,
        policy=RootPolicy(maximum_iterations=1),
    )
    assert not limit.success
    with pytest.raises(ValueError, match="reference"):
        molecular_heat_duty(
            d, MolecularStream(replace(feed.state, enthalpy_J_mol=1e6), 0.05), heat_duty_W=0
        )
    with pytest.raises(ValueError, match="direction"):
        MolecularPressureStage(d, "compressor", 1e5).evaluate(feed)
    with pytest.raises(ValueError, match="zero stream"):
        molecular_heat_duty(d, MolecularStream(feed.state, 0), heat_duty_W=20)
    with pytest.raises(ValueError, match="sum to one"):
        molecular_split_stream(d, feed, (0.5, 0.6))
    with pytest.raises(ValueError, match="raise pressure"):
        molecular_mix_streams(d, (feed,), outlet_pressure_Pa=3e5, initial_temperature_K=400)


def test_native_declarations_roundtrip_reuse_existing_codec():
    d = domain(True)
    assert MolecularProcessDomain.from_dict(json.loads(json.dumps(d.to_dict()))) == d
    for device in (
        MolecularPressureStage(d, "compressor", 4e5, 0.8),
        MolecularHeatDuty(d, 10, boundary_temperature_K=800),
    ):
        restored = type(device).from_dict(json.loads(json.dumps(device.to_dict())))
        assert restored == device
        assert restored.evaluate(stream(d)).to_dict() == device.evaluate(stream(d)).to_dict()
