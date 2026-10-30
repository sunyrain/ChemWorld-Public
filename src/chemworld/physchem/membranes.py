"""Dilute ideal RO with finite chambers and a bounded native hydraulic drive.

The isothermal dead-end chamber loses exactly the water/salt reaching its
collector. The liquid-volume convention neglects salt partial volume. No
upstream flowsheet or hidden infinite feed is used.
"""

from dataclasses import asdict, dataclass, field
from math import exp, isfinite

import numpy as np
from scipy.optimize import brentq

from chemworld.physchem.phase_inventory import PhaseInventoryRates, integrate_phase_inventory
from chemworld.physchem.transport import (
    FluidState,
    PipeSpec,
    PumpSpec,
    pipe_pressure_drop,
    pump_operating_point,
)
from chemworld.physchem.transport_devices import TransportInventory

R = 8.31446261815324
WATER_MOLAR_VOLUME_M3 = 0.01801528 / 1000.0
FOULANT_MOLAR_MASS_KG = 0.0600843  # inert SiO2 accounting units
VISCOSITY_PA_S = 0.001


@dataclass(frozen=True)
class MembraneFoulingSpec:
    deposition_fraction: float = 0.1
    detachment_s: float = 1e-5
    rinse_detachment_s: float = 0.01
    specific_resistance_m_kg: float = 1e16
    wash_vessel_id: str = "wash"
    waste_vessel_id: str = "waste"
    waste_capacity_L: float = 0.2

    def __post_init__(self) -> None:
        for key in (
            "deposition_fraction",
            "detachment_s",
            "rinse_detachment_s",
            "specific_resistance_m_kg",
        ):
            if not isfinite(getattr(self, key)) or getattr(self, key) < 0:
                raise ValueError("fouling coefficients must be finite and nonnegative")
        if self.deposition_fraction > 1:
            raise ValueError("deposition fraction cannot exceed one")
        if not isfinite(self.waste_capacity_L) or self.waste_capacity_L <= 0:
            raise ValueError("waste capacity must be finite and positive")
        if (
            not self.wash_vessel_id
            or not self.waste_vessel_id
            or (self.wash_vessel_id == self.waste_vessel_id)
        ):
            raise ValueError("wash and waste need distinct finite vessels")


@dataclass(frozen=True)
class MembraneProperties:
    component_ids: tuple[str, ...] = ("water", "nacl")


@dataclass(frozen=True)
class MembraneSpec:
    water_permeability_m_Pa_s: float = 1e-11
    salt_permeability_m_s: float = 1e-7
    mass_transfer_m_s: float = 1e-5
    area_m2: float = 0.01
    temperature_K: float = 298.15
    concentration_limit_mol_m3: float = 100.0
    pump: PumpSpec = field(default_factory=lambda: PumpSpec(800000, 0.2, 0.7))
    pipe: PipeSpec = field(default_factory=lambda: PipeSpec(0.1, 0.002))
    collector_vessel_id: str = "permeate"
    collector_capacity_L: float = 1.1
    pickup_volume_L: float = 1e-6
    fouling: MembraneFoulingSpec | None = None

    def __post_init__(self) -> None:
        for name in (
            "water_permeability_m_Pa_s",
            "mass_transfer_m_s",
            "area_m2",
            "temperature_K",
            "concentration_limit_mol_m3",
            "collector_capacity_L",
            "pickup_volume_L",
        ):
            if not isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not isfinite(self.salt_permeability_m_s) or self.salt_permeability_m_s < 0:
            raise ValueError("salt permeability must be finite and nonnegative")
        if self.temperature_K != 298.15 or self.concentration_limit_mol_m3 > 100:
            raise ValueError("first dilute membrane model is 298.15 K and <=100 mol/m3")
        if not self.collector_vessel_id or self.pipe.elevation_change_m != 0:
            raise ValueError("membrane requires a named collector at equal elevation")
        if self.fouling and self.collector_vessel_id in (
            self.fouling.wash_vessel_id,
            self.fouling.waste_vessel_id,
        ):
            raise ValueError("permeate, wash and waste must be separate vessels")

    @property
    def properties(self):
        return MembraneProperties(
            ("water", "nacl", "foulant") if self.fouling else ("water", "nacl")
        )

    def at_ph(self, index, amounts, enthalpy):
        if abs(enthalpy) > 1e-8:
            raise ValueError("isothermal ideal membrane inventories have zero datum enthalpy")
        return self.temperature_K

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        return cls(
            **{
                **value,
                "pump": PumpSpec(**value["pump"]),
                "pipe": PipeSpec(**value["pipe"]),
                "fouling": MembraneFoulingSpec(**value["fouling"])
                if value.get("fouling")
                else None,
            }
        )


def effective_water_permeability(spec, deposited_mass_kg=0.0):
    if not isfinite(deposited_mass_kg) or deposited_mass_kg < 0:
        raise ValueError("deposited mass must be finite and nonnegative")
    if spec.fouling is None:
        if deposited_mass_kg:
            raise ValueError("a clean-only membrane cannot own undeclared fouling")
        return spec.water_permeability_m_Pa_s
    resistance = 1 / (VISCOSITY_PA_S * spec.water_permeability_m_Pa_s)
    return 1 / (
        VISCOSITY_PA_S
        * (resistance + spec.fouling.specific_resistance_m_kg * deposited_mass_kg / spec.area_m2)
    )


def membrane_fouling_rate(spec, feed, water_flow_m3_s, deposited_mass_kg):
    """Signed film mass rate; equal and opposite transfer goes to the feed."""
    model = spec.fouling
    if model is None:
        return 0.0
    volume = liquid_volume_m3(feed)
    concentration = feed["foulant"] * FOULANT_MOLAR_MASS_KG / volume if volume else 0.0
    return (
        model.deposition_fraction * concentration * water_flow_m3_s
        - model.detachment_s * deposited_mass_kg
    )


def liquid_volume_m3(amounts):
    return amounts["water"] * WATER_MOLAR_VOLUME_M3


def salt_concentration(amounts):
    volume = liquid_volume_m3(amounts)
    if volume <= 0:
        if amounts["nacl"] > 0:
            raise ValueError("dry salt is outside the dilute liquid model")
        return 0.0
    return amounts["nacl"] / volume


def membrane_at_flux(spec, bulk_concentration, water_flux_m_s, deposited_mass_kg=0.0):
    """Inverse pressure law; analytic B>0 zero-flux limit avoids js/jw."""
    if not all(isfinite(v) and v >= 0 for v in (bulk_concentration, water_flux_m_s)):
        raise ValueError("membrane requires finite nonnegative concentration and flux")
    j, b = water_flux_m_s, spec.salt_permeability_m_s
    if b == 0:
        cp = 0.0
        difference = bulk_concentration * exp(j / spec.mass_transfer_m_s)
    else:
        denominator = b + j * exp(-j / spec.mass_transfer_m_s)
        cp = bulk_concentration * b / denominator
        difference = bulk_concentration * j / denominator
    return {
        "water_flux_m_s": j,
        "salt_flux_mol_m2_s": j * cp,
        "bulk_concentration_mol_m3": bulk_concentration,
        "interface_concentration_mol_m3": cp + difference,
        "permeate_concentration_mol_m3": cp,
        "osmotic_difference_Pa": 2 * R * spec.temperature_K * difference,
        "membrane_pressure_Pa": j / effective_water_permeability(spec, deposited_mass_kg)
        + 2 * R * spec.temperature_K * difference,
    }


def membrane_at_pressure(spec, bulk_concentration, pressure_difference_Pa, deposited_mass_kg=0.0):
    """Forward reference-facing closure with explicit unsupported reverse drive."""
    if not isfinite(pressure_difference_Pa):
        raise ValueError("membrane pressure must be finite")
    base = membrane_at_flux(spec, bulk_concentration, 0, deposited_mass_kg)
    if pressure_difference_Pa < base["membrane_pressure_Pa"]:
        return {**base, "stopping_event": "unsupported_reverse_drive"}
    if pressure_difference_Pa == base["membrane_pressure_Pa"]:
        return {**base, "stopping_event": "zero_net_drive"}
    flux = brentq(
        lambda j: (
            membrane_at_flux(spec, bulk_concentration, j, deposited_mass_kg)["membrane_pressure_Pa"]
            - pressure_difference_Pa
        ),
        0,
        pressure_difference_Pa * effective_water_permeability(spec, deposited_mass_kg),
        xtol=1e-16,
    )
    result = membrane_at_flux(spec, bulk_concentration, flux, deposited_mass_kg)
    return {
        **result,
        "stopping_event": (
            "concentration_domain"
            if max(
                result[k]
                for k in (
                    "bulk_concentration_mol_m3",
                    "interface_concentration_mol_m3",
                    "permeate_concentration_mol_m3",
                )
            )
            > spec.concentration_limit_mol_m3
            else None
        ),
    }


def membrane_operating_point(spec, amounts, requested_flow_m3_s, deposited_mass_kg=0.0):
    concentration = salt_concentration(amounts)
    fluid = FluidState(1000, 0.001, 4180, 0.6)

    def pipe_load(flow):
        return pipe_pressure_drop(
            spec.pipe, fluid, volumetric_flow_m3_s=flow
        ).pressure_drop_total_Pa

    def load(flow):
        return membrane_at_flux(spec, concentration, flow / spec.area_m2, deposited_mass_kg)[
            "membrane_pressure_Pa"
        ] + pipe_load(flow)

    point = pump_operating_point(spec.pump, requested_flow_m3_s, load)
    flow = point["actual_flow_m3_s"]
    membrane = membrane_at_flux(spec, concentration, flow / spec.area_m2, deposited_mass_kg)
    pipe_pressure = pipe_load(flow)
    actual_pressure = max(0.0, point["pressure_rise_Pa"] - pipe_pressure)
    return {
        **point,
        **membrane,
        "pipe_pressure_Pa": pipe_pressure,
        "membrane_pressure_Pa": actual_pressure,
        "net_driving_pressure_Pa": actual_pressure - membrane["osmotic_difference_Pa"],
    }


def rinse_operating_point(spec, requested_flow_m3_s):
    """Zero-holdup wash bypass; same bounded pump and pipe, no permeation."""
    return pump_operating_point(
        spec.pump,
        requested_flow_m3_s,
        lambda q: (
            pipe_pressure_drop(
                spec.pipe, FluidState(1000, VISCOSITY_PA_S, 4180, 0.6), volumetric_flow_m3_s=q
            ).pressure_drop_total_Pa
        ),
    )


def film_mass(spec, inventory):
    return inventory.amounts_mol[4]["foulant"] * FOULANT_MOLAR_MASS_KG if spec.fouling else 0.0


def advance_membrane(
    spec, state, requested_flow_m3_s, duration_s, *, max_step_s=None, flow_path="process"
):
    if not isfinite(requested_flow_m3_s) or requested_flow_m3_s < 0:
        raise ValueError("requested flow must be finite and nonnegative")
    if flow_path not in (("process", "rinse") if spec.fouling else ("process",)):
        raise ValueError("unsupported membrane flow path")
    if len(state.amounts_mol) != (5 if spec.fouling else 2):
        raise ValueError("membrane needs its declared fluid and bound inventories")
    rinsing = flow_path == "rinse"
    initial = (
        rinse_operating_point(spec, requested_flow_m3_s)
        if rinsing
        else membrane_operating_point(
            spec, state.amounts_mol[0], requested_flow_m3_s, film_mass(spec, state)
        )
    )
    trial_extensions = {"exhausted_feed_trials": 0}

    def point(candidate):
        if rinsing:
            return initial
        if liquid_volume_m3(candidate.amounts_mol[0]) <= 0:
            trial_extensions["exhausted_feed_trials"] += 1
            return initial
        return membrane_operating_point(
            spec, candidate.amounts_mol[0], requested_flow_m3_s, film_mass(spec, candidate)
        )

    def rates(candidate, temperatures):
        operating = point(candidate)
        water = operating["actual_flow_m3_s"] / WATER_MOLAR_VOLUME_M3
        power = operating["shaft_power_W"]
        amounts = np.zeros((len(candidate.amounts_mol), len(spec.properties.component_ids)))
        if rinsing:
            # Only actual finite wash contacts the film. No zero-flow cleaning.
            released = (
                spec.fouling.rinse_detachment_s * candidate.amounts_mol[4]["foulant"]
                if water > 0
                else 0.0
            )
            amounts[2, 0], amounts[3, 0] = -water, water
            amounts[4, 2], amounts[3, 2] = -released, released
        else:
            salt = operating["salt_flux_mol_m2_s"] * spec.area_m2
            amounts[0, :2], amounts[1, :2] = (-water, -salt), (water, salt)
            if spec.fouling:
                deposited = (
                    membrane_fouling_rate(
                        spec,
                        candidate.amounts_mol[0],
                        operating["actual_flow_m3_s"],
                        film_mass(spec, candidate),
                    )
                    / FOULANT_MOLAR_MASS_KG
                )
                amounts[0, 2], amounts[4, 2] = -deposited, deposited
        # At fixed T ideal mixing has zero H. All shaft work is released to
        # the explicit bath, including motor loss; no duplicate time solve.
        return PhaseInventoryRates(
            amounts,
            np.zeros(len(candidate.amounts_mol)),
            power,
            power,
            (),
        )

    stops = (
        [
            (
                "feed_pickup_exhausted",
                lambda s, t: liquid_volume_m3(s.amounts_mol[0]) * 1000 - spec.pickup_volume_L,
            ),
            (
                "collector_full",
                lambda s, t: spec.collector_capacity_L - liquid_volume_m3(s.amounts_mol[1]) * 1000,
            ),
            (
                "concentration_domain",
                lambda s, t: (
                    spec.concentration_limit_mol_m3
                    - max(
                        point(s)["interface_concentration_mol_m3"],
                        salt_concentration(s.amounts_mol[1]),
                    )
                ),
            ),
        ]
        if requested_flow_m3_s and not rinsing
        else []
    )
    if rinsing and requested_flow_m3_s:
        stops.extend(
            (
                (
                    "wash_pickup_exhausted",
                    lambda s, t: liquid_volume_m3(s.amounts_mol[2]) * 1000 - spec.pickup_volume_L,
                ),
                (
                    "waste_full",
                    lambda s, t: (
                        spec.fouling.waste_capacity_L - liquid_volume_m3(s.amounts_mol[3]) * 1000
                    ),
                ),
            )
        )
    if requested_flow_m3_s and initial["actual_flow_m3_s"] == 0:
        stops.append(("no_forward_pump_drive", lambda s, t: 0.0))
    result = integrate_phase_inventory(
        spec,
        state,
        duration_s=duration_s,
        rates=rates,
        closure_indices=tuple(range(len(state.amounts_mol))),
        enthalpy_bounds=lambda i, n: (-1e-9, 1e-9),
        stop_functions=tuple(stops),
        initial_boundary="stop",
        max_step_s=max_step_s,
        solver_id="finite_isothermal_ro_bdf_v1",
        diagnostic_counters=trial_extensions,
    )
    final = result.final_state
    work = final.heater_energy_J - state.heater_energy_J
    return result, {
        "initial": initial,
        "final": point(final),
        "shaft_work_J": work,
        "hydraulic_work_J": work * spec.pump.efficiency,
        "motor_loss_J": work * (1 - spec.pump.efficiency),
        "isothermal_heat_out_J": final.condenser_energy_J - state.condenser_energy_J,
        "flow_path": flow_path,
        "initial_film_mass_kg": film_mass(spec, state),
        "final_film_mass_kg": film_mass(spec, final),
    }


def membrane_inventory(
    feed_volume_L=1.0,
    concentration_mol_m3=10.0,
    *,
    fouling=False,
    foulant_concentration_kg_m3=0.1,
    wash_volume_L=0.1,
):
    if not isfinite(feed_volume_L) or feed_volume_L <= 0:
        raise ValueError("finite feed volume must be positive")
    if not isfinite(concentration_mol_m3) or not 0 <= concentration_mol_m3 <= 100:
        raise ValueError("feed salt concentration must be within 0-100 mol/m3")
    volume = feed_volume_L / 1000
    amounts = (
        {"water": volume / WATER_MOLAR_VOLUME_M3, "nacl": concentration_mol_m3 * volume},
        {"water": 0.0, "nacl": 0.0},
    )
    if fouling:
        if (
            not isfinite(foulant_concentration_kg_m3)
            or foulant_concentration_kg_m3 < 0
            or not isfinite(wash_volume_L)
            or wash_volume_L <= 0
        ):
            raise ValueError(
                "fouling inventory needs finite nonnegative pollutant and positive wash"
            )
        amounts = (
            {**amounts[0], "foulant": foulant_concentration_kg_m3 * volume / FOULANT_MOLAR_MASS_KG},
            {**amounts[1], "foulant": 0.0},
            {"water": wash_volume_L / 1000 / WATER_MOLAR_VOLUME_M3, "nacl": 0.0, "foulant": 0.0},
            {"water": 0.0, "nacl": 0.0, "foulant": 0.0},
            {"water": 0.0, "nacl": 0.0, "foulant": 0.0},
        )
    return TransportInventory(amounts, (0.0,) * len(amounts))
