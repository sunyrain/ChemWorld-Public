from datetime import UTC, datetime, timedelta, timezone

import pytest

from chemworld.physchem.solar_ephemeris import SolarEphemeris, SolarPosition, _GeocentricSun
from chemworld.physchem.solar_events import solar_events
from chemworld.physchem.solar_irradiance import clear_sky_irradiance, extraterrestrial_irradiance


class EquatorialAnalyticSun(SolarEphemeris):
    def __init__(self):
        pass

    def _geocentric(self, timestamp, delta_t):
        return _GeocentricSun(180, 0, (timestamp % 86400) / 86400 * 360, 1, 0)


def test_civil_timezone_day_and_polar_events_on_analytic_sun():
    model = EquatorialAnalyticSun()
    moment = datetime(2024, 6, 21, tzinfo=timezone(timedelta(hours=14)))
    events = solar_events(model, moment, 0, 0, delta_t_seconds=0)
    assert events.status == "normal"
    assert all(
        event.date() == moment.date() for event in (events.sunrise, events.sunset, events.transit)
    )
    assert events.transit.astimezone(UTC) == datetime(2024, 6, 20, 12, tzinfo=UTC)
    assert (events.sunset - events.sunrise).total_seconds() == pytest.approx(
        -86400 + 12 * 3600 + 2 * 0.8333 * 240, abs=0.001
    )
    polar = solar_events(model, moment, 90, 0, delta_t_seconds=0)
    assert polar.status == "polar_day" and polar.sunrise is polar.sunset is None
    assert events.to_dict()["sunrise"].endswith("+14:00")


def test_explicit_irradiance_energy_and_seasonal_convention():
    position = SolarPosition(30, 30, 60, 60, 180, 0, 149597870700)
    date = datetime(2024, 1, 1, tzinfo=UTC)
    kwargs = {
        "pressure_Pa": 101325,
        "elevation_m": 0,
        "linke_turbidity": 3,
        "surface_tilt_degrees": 30,
        "surface_azimuth_degrees": 180,
    }
    result = clear_sky_irradiance(position, date, **kwargs)
    source = clear_sky_irradiance(position, date, convention="source_constant", **kwargs)
    assert result.plane_total_W_m2 / source.plane_total_W_m2 == pytest.approx(
        result.extraterrestrial_W_m2 / 1366.1
    )
    assert result.absorbed_power_w(2, 0.7) == pytest.approx(1.4 * result.plane_total_W_m2)
    assert result.plane_total_W_m2 == pytest.approx(
        result.plane_direct_W_m2 + result.plane_sky_W_m2 + result.plane_ground_W_m2
    )
    with pytest.raises(ValueError, match="domain"):
        clear_sky_irradiance(position, date, **{**kwargs, "linke_turbidity": 0.5})
    with pytest.raises(ValueError, match="ephemeris"):
        extraterrestrial_irradiance(date, method="nrel")
    assert (
        extraterrestrial_irradiance(date, method="nrel", ephemeris=EquatorialAnalyticSun())
        == 1366.1
    )


def test_night_and_vacuum_refraction_are_explicit():
    model = EquatorialAnalyticSun()
    moment = datetime(2024, 1, 1, tzinfo=UTC)
    position = model.position(moment, 0, 0, pressure_Pa=0)
    assert position.apparent_elevation_degrees == position.elevation_degrees
    result = clear_sky_irradiance(
        position,
        moment,
        pressure_Pa=101325,
        elevation_m=0,
        linke_turbidity=3,
        surface_tilt_degrees=30,
        surface_azimuth_degrees=180,
    )
    assert result.plane_total_W_m2 == result.absorbed_power_w(10, 1) == 0
