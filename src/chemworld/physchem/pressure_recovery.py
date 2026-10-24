"""Isothermal liquid pressure-work output from the original aqueous G/H/S/V.

This device has no material storage, generator or same-shaft pump coupling.
Its efficiency is relative to reversible isothermal Gibbs work, not isentropic
turbine work. Excess available work is dissipated at the declared isotherm.
"""

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from math import isfinite
from typing import Any

from chemworld.physchem.aqueous_transport import BinaryAqueousProperties


@dataclass(frozen=True)
class IsothermalLiquidPressureRecoverySpec:
    efficiency: float
    maximum_shaft_power_W: float
    maximum_pressure_drop_Pa: float
    maximum_inlet_volume_flow_m3_s: float

    def __post_init__(self) -> None:
        if any(isinstance(v, bool) or not isfinite(v) for v in asdict(self).values()):
            raise ValueError("pressure recovery needs finite declared ratings")
        if (
            not 0 <= self.efficiency <= 1
            or self.maximum_shaft_power_W < 0
            or self.maximum_pressure_drop_Pa <= 0
            or self.maximum_inlet_volume_flow_m3_s <= 0
        ):
            raise ValueError("pressure recovery ratings or isothermal efficiency are invalid")

    def to_dict(self) -> dict[str, Any]:
        return {"schema": "isothermal-liquid-pressure-recovery-v1", **asdict(self)}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "IsothermalLiquidPressureRecoverySpec":
        data = dict(value)
        if data.pop("schema", None) != "isothermal-liquid-pressure-recovery-v1":
            raise ValueError("unsupported isothermal pressure recovery schema")
        return cls(**data)


def evaluate_isothermal_liquid_pressure_recovery(
    spec: IsothermalLiquidPressureRecoverySpec,
    properties: BinaryAqueousProperties,
    component_rates_mol_s: Mapping[str, float],
    *,
    temperature_K: float,
    inlet_pressure_Pa: float,
    outlet_pressure_Pa: float,
) -> dict[str, Any]:
    """A pure rate calculation: original extensive properties evaluated per second."""
    if any(
        isinstance(v, bool) or not isfinite(v) or v <= 0
        for v in (temperature_K, inlet_pressure_Pa, outlet_pressure_Pa)
    ):
        raise ValueError("pressure recovery requires positive finite T and pressures")
    if inlet_pressure_Pa < outlet_pressure_Pa:
        raise ValueError("pressure recovery requires forward pressure drop")
    drop = inlet_pressure_Pa - outlet_pressure_Pa
    if drop > spec.maximum_pressure_drop_Pa:
        raise ValueError("pressure recovery pressure-drop rating exceeded")
    properties.validate_amounts(component_rates_mol_s, empty=True)
    # Validate declarations even at zero flow, without inventing a liquid state.
    for pressure in (inlet_pressure_Pa, outlet_pressure_Pa):
        properties.database.standard_states(temperature_K, pressure)
        properties.viscosity.evaluate(temperature_K, pressure, 0.0)
    result: dict[str, Any] = {
        "scope": "independent_isothermal_liquid_pressure_work",
        "component_rates_mol_s": dict(component_rates_mol_s),
        "temperature_K": temperature_K,
        "inlet_pressure_Pa": inlet_pressure_Pa,
        "outlet_pressure_Pa": outlet_pressure_Pa,
        "pressure_drop_Pa": drop,
        "inlet_volume_flow_m3_s": 0.0,
        "pressure_availability_W": 0.0,
        "enthalpy_rate_change_W": 0.0,
        "recovered_shaft_power_W": 0.0,
        "irreversibility_W": 0.0,
        "device_heat_into_liquid_W": 0.0,
        "entropy_production_W_K": 0.0,
        "power_limited": False,
        "no_material": not any(component_rates_mol_s.values()),
        "inlet": None,
        "outlet": None,
    }
    if result["no_material"]:
        return result
    inlet = properties.liquid(component_rates_mol_s, temperature_K, inlet_pressure_Pa)
    outlet = properties.liquid(component_rates_mol_s, temperature_K, outlet_pressure_Pa)
    if inlet.volume_m3 > spec.maximum_inlet_volume_flow_m3_s:
        raise ValueError("pressure recovery inlet-volume-flow rating exceeded")
    available = inlet.gibbs_energy_J - outlet.gibbs_energy_J
    if not isfinite(available) or available < 0:
        raise ValueError("declared liquid has negative isothermal pressure availability")
    target = spec.efficiency * available
    recovered = min(target, spec.maximum_shaft_power_W)
    change = outlet.enthalpy_J - inlet.enthalpy_J
    result.update(
        inlet=inlet.to_dict(),
        outlet=outlet.to_dict(),
        inlet_volume_flow_m3_s=inlet.volume_m3,
        pressure_availability_W=available,
        enthalpy_rate_change_W=change,
        recovered_shaft_power_W=recovered,
        irreversibility_W=available - recovered,
        device_heat_into_liquid_W=change + recovered,
        entropy_production_W_K=(available - recovered) / temperature_K,
        power_limited=target > spec.maximum_shaft_power_W,
    )
    return result
