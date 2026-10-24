"""Independent reference self-checks precede any native qualification run."""

import pytest

from reference.calphad42_diagram_reference import R, binodal, equilibrium, interaction, surface


@pytest.mark.parametrize("family", ["REGULAR", "LOOP", "CROSSING"])
@pytest.mark.parametrize("temperature,x", [(500.0, 0.2), (700.0, 0.5), (900.0, 0.8)])
def test_reference_derivatives_and_chemical_potentials(family, temperature, x):
    phase = "BETA" if family == "CROSSING" else "ALPHA"
    point = surface(family, temperature, x, phase)
    dt, dx = 0.001, 1e-5
    gt = (
        surface(family, temperature + dt, x, phase)["G"]
        - surface(family, temperature - dt, x, phase)["G"]
    ) / (2 * dt)
    gx = (
        surface(family, temperature, x + dx, phase)["G"]
        - surface(family, temperature, x - dx, phase)["G"]
    ) / (2 * dx)
    assert point["H"] == pytest.approx(point["G"] - temperature * gt, abs=2e-5)
    assert point["S"] == pytest.approx(-gt, abs=1e-7)
    assert point["mu"]["FE"] - point["mu"]["NI"] == pytest.approx(gx, abs=2e-5)
    assert point["G"] == pytest.approx(
        x * point["mu"]["FE"] + (1 - x) * point["mu"]["NI"], abs=1e-10
    )


@pytest.mark.parametrize("family", ["REGULAR", "LOOP", "CROSSING"])
def test_reference_conservation_and_pure_endpoints(family):
    for t in (500.0, 600.0, 700.0, 800.0, 900.0):
        for z in (0.0, 0.25, 0.5, 0.75, 1.0):
            result = equilibrium(family, t, z)
            assert sum(p["fraction"] for p in result["phases"]) == pytest.approx(1.0, abs=1e-14)
            assert sum(p["fraction"] * p["x"] for p in result["phases"]) == pytest.approx(
                z, abs=1e-14
            )
            for p in result["phases"]:
                assert p["mu"] == pytest.approx(result["mu"], abs=2e-8)
            if z in (0, 1):
                assert len(result["phases"]) == 1 and result["phases"][0]["x"] == z


def test_reference_critical_roots_and_crossing_are_distinct():
    for family, t in (("REGULAR", 12000 / (2 * R)), ("LOOP", 550.0), ("LOOP", 850.0)):
        omega, _ = interaction(family, t)
        assert 4 * R * t - 2 * omega == pytest.approx(0.0, abs=1e-10)
        assert binodal(family, t) is None
    assert binodal("LOOP", 549.0) is None and binodal("LOOP", 551.0) is not None
    assert binodal("LOOP", 849.0) is not None and binodal("LOOP", 851.0) is None
    a, b = [surface("CROSSING", 700.0, 0.5, name) for name in ("ALPHA", "BETA")]
    assert a["G"] == b["G"]
    assert b["H"] - a["H"] == pytest.approx(10000.0)
    assert b["S"] - a["S"] == pytest.approx(10000 / 700)
    assert 4 * R * 700 > 0  # Its composition curvature remains strictly positive.
