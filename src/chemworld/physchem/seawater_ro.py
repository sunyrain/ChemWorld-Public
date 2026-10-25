"""Empirical seawater mass properties and a finite-volume RO channel.

Independent Sharqawy (2010), Nayar (2016), and Bartholomew (2019)
correlations, in the WaterTAP 1.8 seawater/flat-sheet convention. TDS is kg
of empirical sea salt. The colligative conversion is not a molecular identity.
These independently fitted H/V/osmotic laws do not supply G or reversible work.
"""

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from itertools import pairwise
from math import exp, isfinite
from typing import Any

import numpy as np
from scipy.optimize import brentq, least_squares

from chemworld.physchem.aqueous_transport import AqueousTransportLaw
from chemworld.physchem.transport import PumpSpec, pump_operating_point


def empirical_seawater_enthalpy(s: Any, temperature_K: Any, pressure_Pa: Any) -> Any:
    """Nayar independent caloric fit; permits the original derivative type."""
    t = temperature_K - 273.15
    hw = 141.355 + 4202.07 * t - 0.535 * t * t + 0.004 * t**3
    h0 = hw - s * (
        -23482.5
        + 315183 * s
        + 2802690 * s * s
        - 14460600 * s**3
        + 7826.07 * t
        - 44.1733 * t * t
        + 0.21394 * t**3
        - 19910.8 * s * t
        + 27784.6 * s * s * t
        + 97.2801 * s * t * t
    )
    return h0 + (pressure_Pa - 101325) / 1e6 * (
        996.7767
        - 3.2406 * t
        + 0.0127 * t * t
        - 4.7723e-5 * t**3
        + 1000 * s * (-1.1748 + 0.01169 * t - 2.6185e-5 * t * t + 7.0661e-8 * t**3)
    )


@dataclass(frozen=True)
class SeawaterROProperties:
    correlation: str = "sharqawy-nayar-watertap-1.8"

    def __post_init__(self) -> None:
        if self.correlation != "sharqawy-nayar-watertap-1.8":
            raise ValueError("unknown seawater property correlation")

    def evaluate(
        self, salinity_kg_kg: float, temperature_K: float, pressure_Pa: float
    ) -> dict[str, float]:
        s, t = salinity_kg_kg, temperature_K - 273.15
        if (
            not all(isfinite(v) for v in (s, t, pressure_Pa))
            or not 0 <= s <= 0.12
            or not 10 <= t <= 120
            or not 1e5 <= pressure_Pa <= 12e6
        ):
            raise ValueError("seawater shared property domain: 10-120 C, 0-120 g/kg, 0.1-12 MPa")
        rho_w = 999.9 + 0.02034 * t - 0.006162 * t * t + 2.261e-5 * t**3 - 4.657e-8 * t**4
        rho = (
            rho_w
            + 802 * s
            - 2.001 * s * t
            + 0.01677 * s * t * t
            - 3.060e-5 * s * t**3
            - 1.613e-5 * s * s * t * t
        )
        mu_w = 4.2844e-5 + 1 / (0.157 * (t + 64.993) ** 2 - 91.296)
        mu = mu_w * (
            1
            + (1.541 + 0.01998 * t - 9.52e-5 * t * t) * s
            + (7.974 - 0.07561 * t + 0.0004724 * t * t) * s * s
        )
        phi = (
            0.89453
            + 0.00041561 * t
            - 4.6262e-6 * t * t
            + 2.2211e-11 * t**4
            - 0.11445 * s
            - 0.0014783 * s * t
            - 1.3526e-8 * s * t**3
            + 7.0132 * s * s
            + 0.05696 * s * s * t
            - 0.00028624 * s * s * t * t
        )
        pi = phi * s / (1 - s) / 0.0314038218 * rho_w * 8.31446261815324 * temperature_K
        diffusion = 1.51e-9 - 2e-9 * s + 3.01e-8 * s * s - 1.22e-7 * s**3 + 1.53e-7 * s**4
        hw = 141.355 + 4202.07 * t - 0.535 * t * t + 0.004 * t**3
        h0 = hw - s * (
            -23482.5
            + 315183 * s
            + 2802690 * s * s
            - 14460600 * s**3
            + 7826.07 * t
            - 44.1733 * t * t
            + 0.21394 * t**3
            - 19910.8 * s * t
            + 27784.6 * s * s * t
            + 97.2801 * s * t * t
        )
        h = h0 + (pressure_Pa - 101325) / 1e6 * (
            996.7767
            - 3.2406 * t
            + 0.0127 * t * t
            - 4.7723e-5 * t**3
            + 1000 * s * (-1.1748 + 0.01169 * t - 2.6185e-5 * t * t + 7.0661e-8 * t**3)
        )
        return {
            "density_kg_m3": rho,
            "water_density_kg_m3": rho_w,
            "viscosity_Pa_s": mu,
            "osmotic_coefficient": phi,
            "osmotic_pressure_Pa": pi,
            "diffusivity_m2_s": diffusion,
            "specific_enthalpy_J_kg": h,
            "tds_concentration_kg_m3": rho * s,
        }

    def validate_amounts(self, masses: Mapping[str, float], *, empty: bool = False) -> None:
        if set(masses) != {"H2O", "TDS"} or any(not isfinite(v) or v < 0 for v in masses.values()):
            raise ValueError("seawater inventory requires nonnegative H2O/TDS masses in kg")
        if empty and not any(masses.values()):
            return
        if masses["H2O"] <= 0 or masses["TDS"] / sum(masses.values()) > 0.12:
            raise ValueError("seawater inventory lies outside its salinity domain")

    def volume(
        self, masses: Mapping[str, float], temperature_K: float, pressure_Pa: float
    ) -> float:
        total = sum(masses.values())
        return (
            total
            / self.evaluate(masses["TDS"] / total, temperature_K, pressure_Pa)["density_kg_m3"]
            if total
            else 0.0
        )

    def enthalpy(
        self, masses: Mapping[str, float], temperature_K: float, pressure_Pa: float
    ) -> float:
        total = sum(masses.values())
        return (
            total
            * self.evaluate(masses["TDS"] / total, temperature_K, pressure_Pa)[
                "specific_enthalpy_J_kg"
            ]
            if total
            else 0.0
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SeawaterROSpec:
    water_permeability: AqueousTransportLaw
    solute_permeability: AqueousTransportLaw
    pump: PumpSpec
    membrane_area_m2: float
    membrane_length_m: float
    temperature_K: float
    inlet_pressure_Pa: float
    reservoir_pressure_Pa: float = 101325.0
    channel_height_m: float = 0.001
    spacer_porosity: float = 0.9
    permeate_capacity_L: float = 1.0
    retentate_capacity_L: float = 2.0
    pickup_volume_L: float = 1e-5
    permeate_vessel_id: str = "permeate"
    retentate_vessel_id: str = "retentate"
    finite_elements: int = 20
    properties: SeawaterROProperties = SeawaterROProperties()

    def __post_init__(self) -> None:
        positive = (
            self.membrane_area_m2,
            self.membrane_length_m,
            self.channel_height_m,
            self.permeate_capacity_L,
            self.retentate_capacity_L,
            self.pickup_volume_L,
        )
        if (
            any(not isfinite(v) or v <= 0 for v in positive)
            or not 0 < self.spacer_porosity < 1
            or not self.reservoir_pressure_Pa < self.inlet_pressure_Pa
            or type(self.finite_elements) is not int
            or self.finite_elements < 1
            or self.finite_elements > 1000
            or not self.permeate_vessel_id
            or not self.retentate_vessel_id
            or self.permeate_vessel_id == self.retentate_vessel_id
            or self.water_permeability.unit != "m/(Pa*s)"
            or self.solute_permeability.unit != "m/s"
        ):
            raise ValueError(
                "seawater RO requires valid finite geometry, pressure and membrane laws"
            )
        self.properties.evaluate(0.035, self.temperature_K, self.inlet_pressure_Pa)
        self.properties.evaluate(0.0, self.temperature_K, self.reservoir_pressure_Pa)

    def to_dict(self) -> dict[str, Any]:
        return json.loads(json.dumps({"schema": "declared-seawater-ro-mass-v1", **asdict(self)}))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SeawaterROSpec":
        payload = dict(data)
        if payload.pop("schema") != "declared-seawater-ro-mass-v1":
            raise ValueError("unsupported empirical seawater RO declaration")
        for name in ("water_permeability", "solute_permeability"):
            payload[name] = AqueousTransportLaw(**payload[name])
        payload["pump"] = PumpSpec(**payload["pump"])
        payload["properties"] = SeawaterROProperties(**payload["properties"])
        return cls(**payload)


def seawater_ro_profile(
    spec: SeawaterROSpec, feed_mass_kg: Mapping[str, float], requested_flow_m3_s: float
) -> dict[str, Any]:
    prop, t, pr = spec.properties, spec.temperature_K, spec.reservoir_pressure_Pa
    prop.validate_amounts(feed_mass_kg)
    point = pump_operating_point(
        spec.pump, requested_flow_m3_s, lambda _: spec.inlet_pressure_Pa - pr
    )
    feed = np.array([feed_mass_kg[k] for k in ("H2O", "TDS")])
    feed *= point["actual_flow_m3_s"] / prop.volume(feed_mass_kg, t, pr)
    incoming = dict(zip(("H2O", "TDS"), map(float, feed), strict=True))
    if not any(feed):
        return {
            **point,
            "inlet_kg_s": incoming,
            "permeate_kg_s": incoming,
            "retentate_kg_s": incoming,
            "stations": [],
            "pressure_drop_Pa": 0.0,
            "water_recovery": 0.0,
        }
    width = spec.membrane_area_m2 / spec.membrane_length_m
    cross = spec.channel_height_m * width * spec.spacer_porosity
    dh = (
        4
        * spec.spacer_porosity
        / (2 / spec.channel_height_m + (1 - spec.spacer_porosity) * 8 / spec.channel_height_m)
    )
    scales = np.array([feed[0], max(feed[1], feed[0] * 1e-5), 1e6, 0.04, 0.001, 1e-5])
    previous = np.r_[feed, spec.inlet_pressure_Pa]
    stations: list[dict[str, Any]] = []
    # Match the explicit backward finite-volume coordinate convention; no source
    # optimizer or source runtime is a production dependency.
    grid = np.round(np.linspace(0.0, 1.0, spec.finite_elements + 1), 6)
    for left, right in pairwise(grid):
        area, length = (
            spec.membrane_area_m2 * (right - left),
            spec.membrane_length_m * (right - left),
        )

        def evaluate(
            vector: Any,
            previous: Any = previous,
            area: float = area,
            length: float = length,
            right: float = float(right),
        ) -> tuple[np.ndarray, dict[str, Any]]:
            water, salt, pressure, sm, sp, j = map(float, vector * scales)
            sb = salt / (water + salt)
            bulk = prop.evaluate(sb, t, pressure)
            interface = prop.evaluate(sm, t, pressure)
            permeate = prop.evaluate(sp, t, pr)
            rho, mu, d = bulk["density_kg_m3"], bulk["viscosity_Pa_s"], bulk["diffusivity_m2_s"]
            q = (water + salt) / rho
            re = (water + salt) * dh / (cross * mu)
            sc = mu / (rho * d)
            sh = 0.46 * (re * sc) ** 0.36
            k = d * sh / dh
            gradient = (0.42 + 189.3 / re) * rho * (q / cross) ** 2 / (2 * dh)
            # AqueousTransportLaw concentration coordinate is explicitly the
            # empirical TDS colligative concentration, not molecular moles.
            concentration = bulk["tds_concentration_kg_m3"] / 0.0314038218
            a = spec.water_permeability.evaluate(t, pressure, concentration)
            b = spec.solute_permeability.evaluate(t, pressure, concentration)
            js = 1000 * j * sp / (1 - sp)
            cm, cp, cb = (r["tds_concentration_kg_m3"] for r in (interface, permeate, bulk))
            residual = np.array(
                [
                    (water - previous[0] + area * 1000 * j) / scales[0],
                    (salt - previous[1] + area * js) / scales[1],
                    (pressure - previous[2] + length * gradient) / 1e6,
                    (
                        j
                        - a
                        * (
                            pressure
                            - pr
                            - interface["osmotic_pressure_Pa"]
                            + permeate["osmotic_pressure_Pa"]
                        )
                    )
                    / 1e-5,
                    (js - b * (cm - cp)) / 1e-5,
                    (cm - cb * exp(j / k) + js / j * (exp(j / k) - 1)) / 100.0,
                ]
            )
            return residual, {
                "axial_fraction": float(right),
                "pressure_Pa": pressure,
                "water_mass_flow_kg_s": water,
                "tds_mass_flow_kg_s": salt,
                "bulk_salinity_kg_kg": sb,
                "interface_salinity_kg_kg": sm,
                "permeate_salinity_kg_kg": sp,
                "water_flux_kg_m2_s": 1000 * j,
                "tds_flux_kg_m2_s": js,
                "water_flux_m_s": j,
                "film_coefficient_m_s": k,
                "pressure_gradient_Pa_m": gradient,
                "volume_flow_m3_s": q,
                "bulk": bulk,
                "interface": interface,
                "permeate": permeate,
                "concentration_polarization_modulus": cm / cb if cb else 1.0,
            }

        sb0 = previous[1] / sum(previous[:2])
        start = (
            np.array(
                [
                    previous[0] * 0.995,
                    previous[1] * 0.99999,
                    previous[2] * 0.999,
                    min(sb0 * 1.2, 0.119),
                    max(sb0 * 0.005, 1e-8),
                    1e-5,
                ]
            )
            / scales
        )
        lower = np.array([feed[0] * 1e-6, 0.0, pr, 0.0, 0.0, 1e-12]) / scales
        upper = np.array([previous[0], previous[1], previous[2], 0.12, 0.12, 1e-3]) / scales
        start = np.maximum(lower + 1e-12, np.minimum(start, upper - 1e-12))
        active = np.array([0, 2, 5]) if feed[1] == 0 else np.arange(6)
        if feed[1] == 0:
            start[[1, 3, 4]] = 0.0

        def expand(x: Any, start: Any = start, active: Any = active) -> Any:
            vector = start.copy()
            vector[active] = x
            return vector

        result = least_squares(
            lambda x, active=active: evaluate(expand(x))[0][
                [0, 2, 3] if len(active) == 3 else active
            ],
            start[active],
            bounds=(lower[active], upper[active]),
            jac="3-point",
            xtol=1e-14,
            ftol=1e-14,
            gtol=1e-14,
            max_nfev=500,
        )
        final_vector = expand(result.x)
        residual, row = evaluate(final_vector)
        if not result.success or np.max(np.abs(residual)) > 1e-8:
            raise RuntimeError(
                f"seawater RO cell {len(stations) + 1} did not close: {max(abs(residual))}"
            )
        row["maximum_scaled_residual"] = float(max(abs(residual)))
        row["solver_evaluations"] = result.nfev
        stations.append(row)
        previous = final_vector[:3] * scales[:3]
    retained = dict(zip(("H2O", "TDS"), map(float, previous[:2]), strict=True))
    permeate_mass = {k: incoming[k] - retained[k] for k in incoming}
    return {
        **point,
        "inlet_kg_s": incoming,
        "permeate_kg_s": permeate_mass,
        "retentate_kg_s": retained,
        "stations": stations,
        "pressure_drop_Pa": spec.inlet_pressure_Pa - previous[2],
        "water_recovery": permeate_mass["H2O"] / incoming["H2O"],
        "backpressure_dissipation_W": (previous[2] - pr) * stations[-1]["volume_flow_m3_s"],
        "property_boundary": "empirical H/V/osmotic fits; no common G or reversible recovery",
        "diffusion_boundary": "25 C concentration fit held temperature-independent as in source",
    }


@dataclass(frozen=True)
class SeawaterROInventory:
    masses_kg: tuple[dict[str, float], ...]
    enthalpies_J: tuple[float, ...]
    time_s: float = 0.0

    def __post_init__(self) -> None:
        if (
            len(self.masses_kg) != 3
            or len(self.enthalpies_J) != 3
            or not isfinite(self.time_s)
            or self.time_s < 0
            or any(not isfinite(h) for h in self.enthalpies_J)
        ):
            raise ValueError("seawater inventory requires three finite mass/H tank coordinates")


def seawater_ro_inventory(
    spec: SeawaterROSpec, feed_mass_kg: Mapping[str, float]
) -> SeawaterROInventory:
    spec.properties.validate_amounts(feed_mass_kg)
    masses = (dict(feed_mass_kg), {"H2O": 0.0, "TDS": 0.0}, {"H2O": 0.0, "TDS": 0.0})
    return SeawaterROInventory(
        masses,
        tuple(
            spec.properties.enthalpy(n, spec.temperature_K, spec.reservoir_pressure_Pa)
            for n in masses
        ),
    )


def advance_seawater_ro(
    spec: SeawaterROSpec,
    inventory: SeawaterROInventory,
    requested_flow_m3_s: float,
    duration_s: float,
) -> tuple[SeawaterROInventory, dict[str, Any]]:
    """Exact finite constant-composition drain; original world slots own state.

    No membrane holdup is introduced. Receiver mixture H is evaluated from its
    actual final composition; bath heat closes the resulting first-law boundary.
    """
    if not isfinite(duration_s) or duration_s < 0 or len(inventory.masses_kg) != 3:
        raise ValueError("finite seawater RO requires three tanks and nonnegative duration")
    p, t, prop = spec.reservoir_pressure_Pa, spec.temperature_K, spec.properties
    for i, amounts in enumerate(inventory.masses_kg):
        prop.validate_amounts(amounts, empty=i > 0)
        expected = prop.enthalpy(amounts, t, p)
        if abs(inventory.enthalpies_J[i] - expected) > 1e-6 + 1e-9 * abs(expected):
            raise ValueError("finite seawater H differs from the declared empirical liquid")
    profile = seawater_ro_profile(spec, inventory.masses_kg[0], requested_flow_m3_s)
    rates = (
        {k: -v for k, v in profile["inlet_kg_s"].items()},
        profile["permeate_kg_s"],
        profile["retentate_kg_s"],
    )
    dt, event = duration_s, None

    def amounts_at(index: int, elapsed: float) -> dict[str, float]:
        return {
            k: inventory.masses_kg[index][k] + elapsed * rates[index][k] for k in ("H2O", "TDS")
        }

    q = profile["actual_flow_m3_s"]
    if q:
        drain_time = (prop.volume(inventory.masses_kg[0], t, p) - spec.pickup_volume_L / 1000) / q
        if drain_time <= dt:
            dt, event = max(0.0, drain_time), "feed_pickup_exhausted"
        for i, capacity in ((1, spec.permeate_capacity_L), (2, spec.retentate_capacity_L)):

            def margin(elapsed: float, i: int = i, capacity: float = capacity) -> float:
                return capacity / 1000 - prop.volume(amounts_at(i, elapsed), t, p)

            if margin(0.0) < -1e-12:
                raise ValueError("seawater receiver already exceeds its finite capacity")
            if margin(dt) <= 0:
                dt = brentq(margin, 0.0, dt, xtol=1e-12) if margin(0.0) > 0 else 0.0
                event = "permeate_receiver_full" if i == 1 else "retentate_receiver_full"
    final_masses = tuple(amounts_at(i, dt) for i in range(3))
    for i, amounts in enumerate(final_masses):
        prop.validate_amounts(amounts, empty=i > 0)
    final_h = tuple(prop.enthalpy(n, t, p) for n in final_masses)
    shaft = profile["shaft_power_W"] * dt
    hydraulic = profile["hydraulic_power_W"] * dt
    bath = sum(final_h) - sum(inventory.enthalpies_J) - hydraulic
    final = SeawaterROInventory(final_masses, final_h, inventory.time_s + dt)
    diagnostic = {
        "profile": profile,
        "actual_duration_s": dt,
        "stopping_event": event,
        "shaft_work_J": shaft,
        "hydraulic_work_J": hydraulic,
        "motor_loss_J": shaft - hydraulic,
        "bath_heat_into_liquid_J": bath,
        "mass_residuals_kg": {
            k: sum(n[k] for n in final_masses) - sum(n[k] for n in inventory.masses_kg)
            for k in ("H2O", "TDS")
        },
        "enthalpy_residual_J": sum(final_h) - sum(inventory.enthalpies_J) - hydraulic - bath,
    }
    return final, diagnostic
