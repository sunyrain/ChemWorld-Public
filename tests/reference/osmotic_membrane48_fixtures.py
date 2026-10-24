"""Authored synthetic finite-water osmosis fixtures, not material calibration."""

from chemworld.physchem.heat_transfer_units import ThermalDeviceSpec
from chemworld.physchem.osmotic_membrane import OsmoticMembraneSpec
from reference.ro47_fixtures import WATER_VOLUME, law
from reference.ro47_fixtures import spec as ro_spec


def spec(family="ideal", mode="FO", **kwargs):
    return OsmoticMembraneSpec(
        ro_spec(family).properties,
        law(2e-12, "m/(Pa*s)"),
        ThermalDeviceSpec(
            mode="isothermal",
            setpoint_K=310,
            maximum_power_W=0.1,
            maximum_cooling_power_W=0.1,
            minimum_setpoint_K=280,
            maximum_setpoint_K=340,
        ),
        0.01,
        (1e5, {"FO": 1e5, "PRO": 2e5, "reverse": 6e5}[mode]),
        (0.003, 0.003),
        **kwargs,
    )


def inventories(left=10.0, right=60.0, water=0.05):
    return tuple(
        {"H2O": water, "Na+": c * water * WATER_VOLUME, "Cl-": c * water * WATER_VOLUME}
        for c in (left, right)
    )
