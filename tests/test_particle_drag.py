import json
from dataclasses import replace

import pytest

from chemworld.physchem.particle_drag import (
    ParticleDragSpec,
    ParticleMotionState,
    advance_particle_motion,
    available_particle_drag_methods,
    particle_drag_coefficient,
    particle_terminal_velocity,
    time_to_stokes_terminal,
)
from chemworld.physchem.transport import FluidState


def spec(**kwargs):
    return replace(ParticleDragSpec(1e-3, 3000, FluidState(1000, 0.001)), **kwargs)


def test_coefficient_domains_shapes_and_explicit_cap():
    assert len(available_particle_drag_methods(1, check_ranges=False)) == 21
    assert "Stokes" not in available_particle_drag_methods(1)
    assert "Ceylan" not in available_particle_drag_methods(0.01)
    with pytest.raises(ValueError, match="range"):
        particle_drag_coefficient(1e6 + 1)
    assert particle_drag_coefficient(
        2e6, cap_barati_high_at_million=True
    ) == particle_drag_coefficient(1e6)
    with pytest.raises(ValueError, match="shape"):
        particle_drag_coefficient(100, sphericity=0.8)
    assert particle_drag_coefficient(
        100, method="Song_Xu", sphericity=0.8
    ) > particle_drag_coefficient(100, method="Song_Xu")


def test_signed_terminal_neutral_buoyancy_and_branch_bracket():
    for density in (500, 3000):
        model = spec(particle_density_kg_m3=density)
        result = particle_terminal_velocity(model)
        assert result.velocity_m_s * (density - 1000) > 0
        assert abs(result.force_residual_N) < 1e-9 * abs(
            model.particle_mass_kg * model.driving_acceleration_m_s2
        )
        bracketed = particle_terminal_velocity(
            model, reynolds_bracket=(result.reynolds / 2, result.reynolds * 2)
        )
        assert bracketed.velocity_m_s == pytest.approx(result.velocity_m_s, rel=1e-9)
    neutral = particle_terminal_velocity(spec(particle_density_kg_m3=1000))
    assert (
        neutral.velocity_m_s == neutral.force_residual_N == 0 and neutral.drag_coefficient is None
    )
    with pytest.raises(ValueError, match="bracket"):
        particle_terminal_velocity(spec(), reynolds_bracket=(0.01, 0.1))


def test_upward_launch_dissipates_energy_and_crosses_to_settling():
    model = spec()
    initial = ParticleMotionState(velocity_m_s=-0.1)
    brief = advance_particle_motion(model, initial, 0.001)
    assert brief.state.velocity_m_s > initial.velocity_m_s
    assert brief.drag_dissipation_J > 0
    full = advance_particle_motion(model, initial, 1)
    assert full.state.velocity_m_s > 0
    assert full.drag_dissipation_J > 0
    assert abs(full.energy_residual_J) < 1e-7 * max(
        abs(full.buoyancy_gravity_work_J), full.drag_dissipation_J
    )
    rising = advance_particle_motion(spec(particle_density_kg_m3=500), ParticleMotionState(), 1)
    assert rising.state.velocity_m_s < 0 and rising.displacement_m < 0
    assert rising.drag_dissipation_J > 0


def test_continuation_identity_and_state_spec_replay():
    model = ParticleDragSpec.from_dict(json.loads(json.dumps(spec().to_dict())))
    initial = ParticleMotionState.from_dict(
        json.loads(json.dumps(ParticleMotionState(2, 4, -0.1).to_dict()))
    )
    identity = advance_particle_motion(model, initial, 0)
    assert identity.state == initial and identity.energy_residual_J == 0
    one = advance_particle_motion(model, initial, 0.4)
    first = advance_particle_motion(model, initial, 0.1)
    second = advance_particle_motion(model, first.state, 0.3)
    assert second.state.velocity_m_s == pytest.approx(one.state.velocity_m_s, rel=1e-8)
    assert second.state.position_m == pytest.approx(one.state.position_m, abs=1e-10)
    assert first.drag_dissipation_J + second.drag_dissipation_J == pytest.approx(
        one.drag_dissipation_J, rel=1e-8
    )


def test_stokes_stable_long_time_and_exact_tolerance():
    model = spec(diameter_m=1e-6, method="Stokes")
    terminal = particle_terminal_velocity(model).velocity_m_s
    result = advance_particle_motion(model, ParticleMotionState(velocity_m_s=0.1), 1000)
    assert result.state.velocity_m_s == terminal
    expected = terminal * 1000 + (0.1 - terminal) * model.stokes_relaxation_time_s
    assert result.displacement_m == pytest.approx(expected, rel=1e-14)
    duration = time_to_stokes_terminal(model, 0.1, relative_tolerance=1e-6)
    reached = advance_particle_motion(model, ParticleMotionState(velocity_m_s=0.1), duration)
    assert abs(reached.state.velocity_m_s - terminal) == pytest.approx(
        abs(terminal) * 1e-6, rel=1e-9
    )
    assert time_to_stokes_terminal(model, terminal) == 0
    neutral = spec(particle_density_kg_m3=1000, method="Stokes")
    with pytest.raises(ValueError, match="absolute"):
        time_to_stokes_terminal(neutral, 0.1)
    assert time_to_stokes_terminal(neutral, 0.1, absolute_velocity_tolerance_m_s=1e-4) > 0
