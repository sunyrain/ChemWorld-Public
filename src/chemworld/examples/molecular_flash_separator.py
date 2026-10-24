"""Authored PR example for the original finite molecular flash station."""

from typing import Any

from chemworld.examples.molecular_cascade import molecular_cascade_example
from chemworld.physchem.heat_transfer_units import ThermalDeviceSpec
from chemworld.physchem.molecular_flash_separator import MolecularFlashSeparatorSpec
from chemworld.physchem.molecular_flash_separator_inventory import (
    MolecularFlashSeparatorInventorySpec,
)
from chemworld.physchem.molecular_mixture import MolecularStream


def molecular_flash_separator_example(
    *, temperature_K: float = 300.0, z: float = 0.6, hot_feed: bool = False
) -> tuple[Any, ...]:
    authored, _, _, _ = molecular_cascade_example(1)
    original = authored.cascade
    separator = MolecularFlashSeparatorSpec(
        original.model,
        original.vapor_basin,
        original.liquid_basin,
        6e5,
        ThermalDeviceSpec(
            mode="isothermal",
            setpoint_K=temperature_K,
            maximum_power_W=100.0,
            maximum_cooling_power_W=100.0,
        ),
        "gas" if hot_feed else "liquid",
    )
    bounds = (325.0, 360.0) if hot_feed else (265.0, 295.0)
    spec = MolecularFlashSeparatorInventorySpec(
        separator, tank_temperature_bounds_K=(bounds, (325.0, 360.0), (265.0, 295.0))
    )
    amounts = tuple({"propane": 0.02 * x, "butane": 0.02 * (1 - x)} for x in (z, 0.95, 0.05))
    temperatures = (340.0 if hot_feed else 280.0, 340.0, 280.0)
    inlet = MolecularStream(spec.at_tp(0, amounts[0], temperatures[0]), spec.feed_rate_mol_s)
    return spec, amounts, temperatures, inlet


def molecular_flash_separator_demo_world() -> Any:
    from chemworld.world.molecular_flash_separator_world import (
        build_molecular_flash_separator_world,
    )

    spec, amounts, temperatures, _ = molecular_flash_separator_example()
    return build_molecular_flash_separator_world(
        spec, amounts, temperatures, formulas={"propane": "C3H8", "butane": "C4H10"}
    )
