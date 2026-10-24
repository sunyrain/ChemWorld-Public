"""Bounded zero-holdup axial RO with original finite aqueous n/H consumers.

The membrane uses a solvent-volume SD/film approximation and the declared
water chemical-potential difference. It does not add pressure twice, infer
seawater data, or model independent-ion electrical selectivity.
"""

import json
from collections.abc import Mapping
from dataclasses import dataclass, replace
from math import exp, isclose, isfinite
from typing import Any, cast

import numpy as np
from scipy.optimize import brentq

from chemworld.physchem.aqueous_transport import AqueousTransportLaw, BinaryAqueousProperties
from chemworld.physchem.elements import molecular_weight
from chemworld.physchem.phase_inventory import PhaseInventoryRates, integrate_phase_inventory
from chemworld.physchem.pressure_recovery import (
    IsothermalLiquidPressureRecoverySpec,
    evaluate_isothermal_liquid_pressure_recovery,
)
from chemworld.physchem.solver_backend import ODESolverPolicy, solve_ode
from chemworld.physchem.transport import (
    FluidState,
    PipeSpec,
    PumpSpec,
    pipe_pressure_drop,
    pump_operating_point,
)
from chemworld.physchem.transport_devices import TransportInventory


@dataclass(frozen=True)
class DistributedROSpec:
    properties: BinaryAqueousProperties
    water_permeability: AqueousTransportLaw
    salt_permeability: AqueousTransportLaw | None
    mass_transfer: AqueousTransportLaw
    channel: PipeSpec
    pump: PumpSpec
    membrane_area_m2: float
    temperature_K: float
    reservoir_pressure_Pa: float
    inlet_pressure_Pa: float
    permeate_capacity_L: float
    retentate_capacity_L: float
    pickup_volume_L: float = 1e-5
    permeate_vessel_id: str = "permeate"
    retentate_vessel_id: str = "retentate"
    maximum_interface_concentration_mol_m3: float = 100.0
    support_heat_capacity_J_K: float = 5.0
    energy_recovery: IsothermalLiquidPressureRecoverySpec | None = None

    def __post_init__(self) -> None:
        numbers = (
            self.membrane_area_m2,
            self.temperature_K,
            self.reservoir_pressure_Pa,
            self.inlet_pressure_Pa,
            self.permeate_capacity_L,
            self.retentate_capacity_L,
            self.pickup_volume_L,
            self.maximum_interface_concentration_mol_m3,
            self.support_heat_capacity_J_K,
        )
        if any(isinstance(v, bool) or not isfinite(v) or v <= 0 for v in numbers):
            raise ValueError("distributed RO requires finite positive geometry and conditions")
        if (
            self.inlet_pressure_Pa <= self.reservoir_pressure_Pa
            or not self.permeate_vessel_id
            or not self.retentate_vessel_id
            or self.permeate_vessel_id == self.retentate_vessel_id
            or self.channel.elevation_change_m != 0
            or self.channel.fittings_loss_coefficient != 0
        ):
            raise ValueError("RO requires distinct outlets and a horizontal distributed channel")
        sources = {s.source_id for s in self.properties.database.sources}
        for law, unit in (
            (self.water_permeability, "m/(Pa*s)"),
            (self.mass_transfer, "m/s"),
            (self.salt_permeability, "m/s"),
        ):
            if law is None:
                continue
            if law.unit != unit or law.source_id not in sources:
                raise ValueError("membrane transport laws require compatible units and sources")
            law.evaluate(self.temperature_K, self.inlet_pressure_Pa, 0)
            law.evaluate(self.temperature_K, self.reservoir_pressure_Pa, 0)
        self.properties.viscosity.evaluate(self.temperature_K, self.inlet_pressure_Pa, 0)
        self.properties.database.chemical_potentials(
            self.properties.amounts(1, 0),
            temperature_K=self.temperature_K,
            pressure_Pa=self.reservoir_pressure_Pa,
        )

    def enthalpy(self, index: int, amounts: Mapping[str, float], temperature: float) -> float:
        return self.properties.enthalpy(amounts, temperature, self.reservoir_pressure_Pa) + (
            self.support_heat_capacity_J_K * (temperature - self.temperature_K)
        )

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema": "declared-distributed-ro-v1",
            "properties": self.properties.to_dict(),
            "water_permeability": self.water_permeability.to_dict(),
            "salt_permeability": None
            if self.salt_permeability is None
            else self.salt_permeability.to_dict(),
            "mass_transfer": self.mass_transfer.to_dict(),
            "channel": self.channel.to_dict(),
            "pump": self.pump.to_dict(),
            **{
                k: getattr(self, k)
                for k in (
                    "membrane_area_m2",
                    "temperature_K",
                    "reservoir_pressure_Pa",
                    "inlet_pressure_Pa",
                    "permeate_capacity_L",
                    "retentate_capacity_L",
                    "pickup_volume_L",
                    "permeate_vessel_id",
                    "retentate_vessel_id",
                    "maximum_interface_concentration_mol_m3",
                    "support_heat_capacity_J_K",
                )
            },
        }
        if self.energy_recovery is not None:
            payload["schema"] = "declared-distributed-ro-v2"
            payload["energy_recovery"] = self.energy_recovery.to_dict()
        # Equipment settings are a JSON declaration. Canonical containers also
        # normalize nested original aqueous-parameter bounds before persistence.
        return json.loads(json.dumps(payload))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "DistributedROSpec":
        data = dict(value)
        schema = data.pop("schema")
        if schema not in {"declared-distributed-ro-v1", "declared-distributed-ro-v2"}:
            raise ValueError("unsupported distributed RO schema")
        if schema == "declared-distributed-ro-v2":
            data["energy_recovery"] = IsothermalLiquidPressureRecoverySpec.from_dict(
                data["energy_recovery"]
            )
        elif "energy_recovery" in data:
            raise ValueError("legacy RO schema cannot declare pressure recovery")
        data["properties"] = BinaryAqueousProperties.from_dict(data["properties"])
        for key in ("water_permeability", "salt_permeability", "mass_transfer"):
            data[key] = None if data[key] is None else AqueousTransportLaw(**data[key])
        data["channel"], data["pump"] = PipeSpec(**data["channel"]), PumpSpec(**data["pump"])
        return cls(**data)


def local_ro_flux(
    spec: DistributedROSpec, bulk_concentration_mol_m3: float, pressure_Pa: float
) -> dict[str, Any]:
    """Solve SD/CP with c=n_s/(n_w Vw), Jw=A*(mu_w,f-mu_w,p)/Vw.

    Jw is solvent-volume flux. Js is salt-basis molar flux. Actual ions are
    transported only by their declared integer stoichiometric vector.
    """
    prop, t, cb = spec.properties, spec.temperature_K, bulk_concentration_mol_m3
    if not 0 <= cb <= spec.maximum_interface_concentration_mol_m3:
        raise ValueError("bulk concentration outside declared membrane interface domain")
    if pressure_Pa < spec.reservoir_pressure_Pa:
        raise ValueError("reverse pressure is outside this forward RO profile")
    a = spec.water_permeability.evaluate(t, pressure_Pa, cb)
    b = 0 if spec.salt_permeability is None else spec.salt_permeability.evaluate(t, pressure_Pa, cb)
    k = spec.mass_transfer.evaluate(t, pressure_Pa, cb)
    vw = prop.water_volume_m3_mol

    def evaluate(j: float) -> dict[str, Any]:
        cp = cb * b / (b + j * exp(-j / k)) if b else 0.0
        cm = cp * (1 + j / b) if b else cb * exp(j / k)
        if cm > spec.maximum_interface_concentration_mol_m3:
            raise ValueError("polarized interface outside declared concentration domain")
        mf = prop.database.chemical_potentials(
            prop.amounts(1, cm * vw), temperature_K=t, pressure_Pa=pressure_Pa
        )
        mp = prop.database.chemical_potentials(
            prop.amounts(1, cp * vw), temperature_K=t, pressure_Pa=spec.reservoir_pressure_Pa
        )
        delta = cast(float, mf.chemical_potentials_J_mol[prop.water_id]) - cast(
            float, mp.chemical_potentials_J_mol[prop.water_id]
        )
        return {
            "water_flux_m_s": j,
            "salt_flux_mol_m2_s": j * cp,
            "interface_concentration_mol_m3": cm,
            "permeate_concentration_mol_m3": cp,
            "water_chemical_potential_difference_J_mol": delta,
            "driving_pressure_Pa": delta / vw,
            "water_flux_residual_m_s": j - a * delta / vw,
        }

    zero = evaluate(0.0)
    if zero["driving_pressure_Pa"] <= 0:
        return {**zero, "stopping_event": "zero_or_reverse_water_drive"}
    upper = min(max(2 * a * zero["driving_pressure_Pa"], 1e-12), 40 * k)
    # The bracket may cross the declared interface boundary. Find the largest
    # admissible trial, then require the physical root inside it; no extrapolation.
    lower, boundary = 0.0, upper
    for _ in range(80):
        try:
            high = evaluate(upper)
            if high["water_flux_residual_m_s"] >= 0:
                break
            lower = upper
        except ValueError:
            boundary = upper
        upper = (lower + boundary) / 2
    else:
        raise ValueError("forward flux root lies beyond declared membrane property domain")
    flux = brentq(
        lambda j: evaluate(j)["water_flux_residual_m_s"], 0, upper, xtol=1e-15, rtol=1e-11
    )
    return {**evaluate(flux), "stopping_event": None}


def distributed_ro_profile(
    spec: DistributedROSpec,
    feed_amounts: Mapping[str, float],
    requested_flow_m3_s: float,
    *,
    samples: int = 17,
    rtol: float = 1e-7,
) -> dict[str, Any]:
    prop, t, pr = spec.properties, spec.temperature_K, spec.reservoir_pressure_Pa
    prop.validate_amounts(feed_amounts)
    if type(samples) is not int or samples < 2:
        raise ValueError("axial reporting requires at least two stations")
    point = pump_operating_point(
        spec.pump, requested_flow_m3_s, lambda q: spec.inlet_pressure_Pa - pr
    )
    q = point["actual_flow_m3_s"]
    volume = prop.volume(feed_amounts, t, pr)
    incoming = {k: v * q / volume for k, v in feed_amounts.items()}
    if q == 0:
        result = {
            **point,
            "inlet_mol_s": incoming,
            "permeate_mol_s": incoming,
            "retentate_mol_s": incoming,
            "stations": [],
            "pressure_drop_Pa": 0.0,
            "backpressure_dissipation_W": 0.0,
            "solver": {},
            "water_recovery": 0.0,
        }
        if spec.energy_recovery is not None:
            result["pressure_recovery"] = evaluate_isothermal_liquid_pressure_recovery(
                spec.energy_recovery,
                prop,
                incoming,
                temperature_K=t,
                inlet_pressure_Pa=pr,
                outlet_pressure_Pa=pr,
            )
        return result
    key, nu = prop.salt_stoichiometry[0]
    w0, s0 = incoming[prop.water_id], incoming[key] / nu
    scales = np.array([w0, max(s0, w0 * 1e-5), 1e6])
    mass = {
        s.species_id: molecular_weight(dict(s.composition)) / 1000 for s in prop.database.species
    }
    unit_pipe = replace(spec.channel, length_m=1.0)

    def station(vector: Any) -> dict[str, Any]:
        w, s, p = map(float, vector * scales)
        if w <= w0 * 1e-5 or s < -1e-14:
            raise ValueError("axial recovery exhausts the forward finite feed stream")
        n = prop.amounts(w, max(s, 0.0))
        liquid = prop.liquid(n, t, p)
        concentration = prop.concentration(n)
        viscosity = prop.viscosity.evaluate(t, p, concentration)
        density = sum(n[k] * mass[k] for k in n) / liquid.volume_m3
        # Hydraulics consumes only rho and mu. Explicit inert placeholders for
        # unused thermal transport fields carry no water-property claim.
        fluid = FluidState(density, viscosity, 1.0, 1.0)
        drop = pipe_pressure_drop(
            unit_pipe, fluid, volumetric_flow_m3_s=liquid.volume_m3
        ).pressure_drop_total_Pa
        flux = local_ro_flux(spec, concentration, p)
        return {
            **flux,
            "water_mol_s": w,
            "salt_mol_s": s,
            "pressure_Pa": p,
            "bulk_concentration_mol_m3": concentration,
            "volume_flow_m3_s": liquid.volume_m3,
            "pressure_gradient_Pa_m": drop,
        }

    def rhs(x: float, y: Any) -> Any:
        row = station(y)
        return (
            np.array(
                [
                    -spec.membrane_area_m2 * row["water_flux_m_s"] / prop.water_volume_m3_mol,
                    -spec.membrane_area_m2 * row["salt_flux_mol_m2_s"],
                    -spec.channel.length_m * row["pressure_gradient_Pa_m"],
                ]
            )
            / scales
        )

    solved = solve_ode(
        rhs,
        np.array([w0, s0, spec.inlet_pressure_Pa]) / scales,
        time_span_s=(0.0, 1.0),
        evaluation_times_s=np.linspace(0, 1, samples),
        policy=ODESolverPolicy(
            "axial_ro_normalized_coordinate_v1", "DOP853", rtol, 1e-10, max_step_s=0.125
        ),
    )
    solved.raise_for_failure("distributed reverse osmosis")
    stations = [
        {"axial_fraction": float(x), **station(y)}
        for x, y in zip(solved.t, solved.y.T, strict=True)
    ]
    final = stations[-1]
    retained = prop.amounts(final["water_mol_s"], final["salt_mol_s"])
    permeate = {k: incoming[k] - retained[k] for k in incoming}
    if min(permeate.values()) < -1e-12:
        raise RuntimeError("axial RO generated negative collected flux")
    result = {
        **point,
        "inlet_mol_s": incoming,
        "permeate_mol_s": permeate,
        "retentate_mol_s": retained,
        "stations": stations,
        "pressure_drop_Pa": spec.inlet_pressure_Pa - final["pressure_Pa"],
        "backpressure_dissipation_W": (final["pressure_Pa"] - pr) * final["volume_flow_m3_s"],
        "water_recovery": permeate[prop.water_id] / w0,
        "solver": solved.diagnostic.to_dict(),
    }
    if spec.energy_recovery is not None:
        recovery = evaluate_isothermal_liquid_pressure_recovery(
            spec.energy_recovery,
            prop,
            retained,
            temperature_K=t,
            inlet_pressure_Pa=final["pressure_Pa"],
            outlet_pressure_Pa=pr,
        )
        result["pressure_recovery"] = recovery
        result["backpressure_dissipation_W"] = recovery["irreversibility_W"]
    return result


def distributed_ro_inventory(
    spec: DistributedROSpec,
    feed_amounts: Mapping[str, float],
    *,
    permeate: Mapping[str, float] | None = None,
    retentate: Mapping[str, float] | None = None,
) -> TransportInventory:
    prop = spec.properties
    rows = tuple(
        dict(n)
        for n in (
            feed_amounts,
            permeate or dict.fromkeys(prop.component_ids, 0.0),
            retentate or dict.fromkeys(prop.component_ids, 0.0),
        )
    )
    for n in rows:
        prop.validate_amounts(n, empty=True)
    return TransportInventory(
        rows, tuple(spec.enthalpy(i, n, spec.temperature_K) for i, n in enumerate(rows))
    )


def advance_distributed_ro(
    spec: DistributedROSpec,
    state: TransportInventory,
    requested_flow_m3_s: float,
    duration_s: float,
    *,
    progress: Any = None,
) -> tuple[Any, dict[str, Any]]:
    """The existing n/H strategy owns all three finite reservoirs and utilities."""
    prop, t, p = spec.properties, spec.temperature_K, spec.reservoir_pressure_Pa
    if len(state.amounts_mol) != 3 or len(state.enthalpies_J) != 3:
        raise ValueError("distributed RO needs finite feed, permeate and retentate tanks")
    for i, n in enumerate(state.amounts_mol):
        prop.validate_amounts(n, empty=i > 0)
        if not isclose(state.enthalpies_J[i], spec.enthalpy(i, n, t), rel_tol=1e-8, abs_tol=1e-6):
            raise ValueError("RO inventory H differs from declared isothermal aqueous table")
    profile = distributed_ro_profile(spec, state.amounts_mol[0], requested_flow_m3_s)
    flows = (
        {k: -v for k, v in profile["inlet_mol_s"].items()},
        profile["permeate_mol_s"],
        profile["retentate_mol_s"],
    )
    matrix = np.array([[n[k] for k in prop.component_ids] for n in flows])
    shaft, hydraulic = profile["shaft_power_W"], profile["hydraulic_power_W"]
    recovered = profile.get("pressure_recovery", {}).get("recovered_shaft_power_W", 0.0)

    def rates(candidate: Any, unused: Any) -> PhaseInventoryRates:
        # At fixed isotherm/pressure the draining feed composition and axial
        # split are constant. Receiver n(t)=n0+t*flow is exact, so evaluate
        # caloric derivatives on that physical path rather than BDF's off-face
        # finite-difference probes (which can put salt into an empty dry tank).
        # The original integrator still owns and commits every n/H coordinate.
        elapsed = candidate.time_s - state.time_s
        caloric_amounts = (
            state.amounts_mol[0],
            *(
                {k: state.amounts_mol[i][k] + elapsed * flows[i][k] for k in prop.component_ids}
                for i in (1, 2)
            ),
        )
        hdot = np.array(
            [
                prop.enthalpy_rate(n, flow, t, p)
                for n, flow in zip(caloric_amounts, flows, strict=True)
            ]
        )
        bath = float(sum(hdot)) - hydraulic
        if spec.energy_recovery is not None:
            bath += recovered
        # Common utility counters represent total boundary energy in/out.
        # The returned physical ledger separates shaft work, motor loss and bath.
        outgoing = shaft - hydraulic + max(-bath, 0.0)
        if spec.energy_recovery is not None:
            outgoing += recovered
        return PhaseInventoryRates(matrix, hdot, shaft + max(bath, 0.0), outgoing, ())

    def volume(n: Mapping[str, float]) -> float:
        return prop.volume(n, t, p) * 1000

    stops = [
        ("feed_pickup_exhausted", lambda s, _: volume(s.amounts_mol[0]) - spec.pickup_volume_L),
        ("permeate_full", lambda s, _: spec.permeate_capacity_L - volume(s.amounts_mol[1])),
        ("retentate_full", lambda s, _: spec.retentate_capacity_L - volume(s.amounts_mol[2])),
    ]
    if requested_flow_m3_s > 0 and profile["actual_flow_m3_s"] == 0:
        stops.insert(0, ("no_forward_pump_drive", lambda s, _: 0.0))
    result = integrate_phase_inventory(
        spec,
        state,
        duration_s=duration_s,
        rates=rates,
        closure_indices=(),
        stop_functions=stops,
        enthalpy_bounds=lambda i, n: (-np.inf, np.inf),
        initial_boundary="stop",
        progress=progress,
        solver_id="finite_distributed_ro_nH_v1",
    )
    dt, final = result.actual_duration_s, result.final_state
    for i, n in enumerate(final.amounts_mol):
        prop.validate_amounts(n, empty=i > 0)
        if not isclose(final.enthalpies_J[i], spec.enthalpy(i, n, t), rel_tol=1e-7, abs_tol=1e-5):
            raise RuntimeError("finite RO failed same-table isothermal H closure")
    change = sum(final.enthalpies_J) - sum(state.enthalpies_J)
    diagnostic = {
        "profile": profile,
        "shaft_work_J": shaft * dt,
        "hydraulic_work_J": hydraulic * dt,
        "motor_loss_J": (shaft - hydraulic) * dt,
        "bath_heat_into_liquid_J": change - hydraulic * dt,
        "backpressure_dissipation_J": profile["backpressure_dissipation_W"] * dt,
        "liquid_enthalpy_change_J": change,
        "energy_scope": "shaft - external motor loss + signed isothermal bath = liquid H",
    }
    if spec.energy_recovery is not None:
        diagnostic.update(
            recovered_shaft_work_J=recovered * dt,
            net_external_mechanical_work_J=(shaft - recovered) * dt,
            bath_heat_into_liquid_J=change - hydraulic * dt + recovered * dt,
            energy_scope="gross pump - motor loss - independent shaft output + bath = liquid H",
        )
    return result, diagnostic
