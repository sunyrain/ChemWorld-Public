"""Declared dilute-ion DSPM-DE on the original aqueous species and n/H solver.

The pore is resolved across its thickness; the single-pass module is 0D, with
mean inlet/retentate concentration and no film or pore inventory. No independent
salt identities, ion clipping or post-solve charge correction are introduced.
"""

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from math import exp, isclose, isfinite, log
from typing import Any, cast

import numpy as np
from scipy.optimize import brentq

from chemworld.physchem.multiion_membrane_properties import MultiIonMembraneProperties
from chemworld.physchem.phase_inventory import PhaseInventoryRates, integrate_phase_inventory
from chemworld.physchem.residual_system import (
    ResidualDerivatives,
    ResidualDomainError,
    ResidualEquation,
    ResidualSystem,
    ResidualVariable,
    RootPolicy,
    solve_steady_residual,
)
from chemworld.physchem.transport import PumpSpec, pump_operating_point
from chemworld.physchem.transport_devices import TransportInventory

ELEMENTARY_CHARGE_C = 1.602176634e-19
BOLTZMANN_J_K = 1.380649e-23
VACUUM_PERMITTIVITY_F_M = 8.8541878128e-12
DEFAULT_NF_ROOT_POLICY = RootPolicy()


@dataclass(frozen=True)
class PoreIon:
    species_id: str
    stokes_radius_m: float
    born_radius_m: float
    diffusivity_m2_s: float
    source_id: str

    def __post_init__(self) -> None:
        if (
            not self.species_id
            or not self.source_id
            or any(
                not isfinite(v) or v <= 0
                for v in (self.stokes_radius_m, self.born_radius_m, self.diffusivity_m2_s)
            )
        ):
            raise ValueError("pore ions require sourced positive radii and diffusivity")


@dataclass(frozen=True)
class NanofiltrationSpec:
    properties: MultiIonMembraneProperties
    ions: tuple[PoreIon, ...]
    pump: PumpSpec
    membrane_area_m2: float
    pore_radius_m: float
    effective_thickness_m: float
    fixed_charge_mol_m3: float
    bulk_relative_permittivity: float
    pore_relative_permittivity: float
    temperature_K: float
    reservoir_pressure_Pa: float
    inlet_pressure_Pa: float
    permeate_capacity_L: float
    retentate_capacity_L: float
    source_id: str
    pore_segments: int = 16
    maximum_concentration_mol_m3: float = 500.0
    pickup_volume_L: float = 1e-5
    permeate_vessel_id: str = "permeate"
    retentate_vessel_id: str = "retentate"
    support_heat_capacity_J_K: float = 5.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "ions", tuple(self.ions))
        sources = {s.source_id for s in self.properties.database.sources}
        if (
            self.properties.database.activity_model.family != "ideal"
            or self.source_id not in sources
            or tuple(i.species_id for i in self.ions) != self.properties.ion_ids
            or any(i.source_id not in sources for i in self.ions)
        ):
            raise ValueError("first NF slice requires sourced ideal dilute original ions")
        positive = (
            self.membrane_area_m2,
            self.pore_radius_m,
            self.effective_thickness_m,
            self.bulk_relative_permittivity,
            self.pore_relative_permittivity,
            self.temperature_K,
            self.reservoir_pressure_Pa,
            self.inlet_pressure_Pa,
            self.permeate_capacity_L,
            self.retentate_capacity_L,
            self.maximum_concentration_mol_m3,
            self.pickup_volume_L,
            self.support_heat_capacity_J_K,
        )
        if (
            any(not isfinite(v) or v <= 0 for v in positive)
            or not isfinite(self.fixed_charge_mol_m3)
            or self.inlet_pressure_Pa <= self.reservoir_pressure_Pa
            or type(self.pore_segments) is not int
            or not 1 <= self.pore_segments <= 128
            or not self.permeate_vessel_id
            or not self.retentate_vessel_id
            or self.permeate_vessel_id == self.retentate_vessel_id
        ):
            raise ValueError("NF requires positive declared geometry, pressure and finite outlets")
        if any(i.stokes_radius_m >= self.pore_radius_m for i in self.ions):
            raise ValueError("excluded ionic radius lies outside forward permeating-ion model")
        self.properties.viscosity.evaluate(self.temperature_K, self.inlet_pressure_Pa, 0)
        for pressure in (self.reservoir_pressure_Pa, self.inlet_pressure_Pa):
            self.properties.database.standard_states(self.temperature_K, pressure)

    def enthalpy(self, index: int, amounts: Mapping[str, float], temperature: float) -> float:
        return self.properties.enthalpy(amounts, temperature, self.reservoir_pressure_Pa) + (
            self.support_heat_capacity_J_K * (temperature - self.temperature_K)
        )

    def to_dict(self) -> dict[str, Any]:
        payload = {k: v for k, v in vars(self).items() if k not in {"properties", "ions", "pump"}}
        payload.update(
            schema="declared-multiion-nf-v1",
            properties=self.properties.to_dict(),
            ions=[asdict(i) for i in self.ions],
            pump=self.pump.to_dict(),
        )
        return json.loads(json.dumps(payload))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "NanofiltrationSpec":
        data = dict(payload)
        if data.pop("schema") != "declared-multiion-nf-v1":
            raise ValueError("unsupported nanofiltration declaration")
        data["properties"] = MultiIonMembraneProperties.from_dict(data["properties"])
        data["ions"] = tuple(PoreIon(**i) for i in data["ions"])
        data["pump"] = PumpSpec(**data["pump"])
        return cls(**data)


def pore_hindrance(radius_ratio: float) -> tuple[float, float]:
    """Diffusive and convective factors from public DSPM-DE equations."""
    x = radius_ratio
    if not isfinite(x) or not 0 <= x < 1:
        raise ValueError("pore radius ratio must lie in [0,1)")
    if x == 0:
        return 1.0, 1.0
    kd = (
        0.984 * ((1 - x) / x) ** 2.5
        if x > 0.95
        else (
            1
            + 9 / 8 * x * log(x)
            - 1.56034 * x
            + 0.528155 * x * x
            + 1.91521 * x**3
            - 2.81903 * x**4
            + 0.270788 * x**5
            - 1.10115 * x**6
            - 0.435933 * x**7
        )
        / (1 - x) ** 2
    )
    kc = (1 + 3.867 * x - 1.907 * x * x - 0.834 * x**3) / (1 + 1.867 * x - 0.741 * x * x)
    return kd, kc


def pore_partition_factors(spec: NanofiltrationSpec) -> dict[str, dict[str, float]]:
    rows = {}
    for ion in spec.ions:
        ratio = ion.stokes_radius_m / spec.pore_radius_m
        z = spec.properties.charges[ion.species_id]
        born = (
            z
            * z
            * ELEMENTARY_CHARGE_C**2
            / (8 * np.pi * VACUUM_PERMITTIVITY_F_M * ion.born_radius_m)
            * (1 / spec.pore_relative_permittivity - 1 / spec.bulk_relative_permittivity)
        )
        kd, kc = pore_hindrance(ratio)
        rows[ion.species_id] = {
            "steric": (1 - ratio) ** 2,
            "born": exp(-born / (BOLTZMANN_J_K * spec.temperature_K)),
            "diffusive_hindrance": kd,
            "convective_hindrance": kc,
        }
    return rows


def donnan_partition(
    spec: NanofiltrationSpec, concentrations: Mapping[str, float]
) -> dict[str, Any]:
    ids, z = spec.properties.ion_ids, spec.properties.charges
    if set(concentrations) != set(ids) or any(
        c <= 0 or not isfinite(c) for c in concentrations.values()
    ):
        raise ValueError("Donnan query requires every declared ion at positive concentration")
    partition = pore_partition_factors(spec)
    base = {k: concentrations[k] * partition[k]["steric"] * partition[k]["born"] for k in ids}

    def charge(phi: float) -> float:
        return sum(z[k] * base[k] * exp(-z[k] * phi) for k in ids) + spec.fixed_charge_mol_m3

    phi = brentq(
        charge,
        -100 / max(abs(q) for q in z.values()),
        100 / max(abs(q) for q in z.values()),
        xtol=1e-14,
    )
    return {
        "dimensionless_potential": phi,
        "pore_concentrations_mol_m3": {k: base[k] * exp(-z[k] * phi) for k in ids},
    }


def nanofiltration_point(
    spec: NanofiltrationSpec,
    concentrations: Mapping[str, float],
    *,
    solvent_flow_m3_s: float = float("inf"),
    policy: RootPolicy = DEFAULT_NF_ROOT_POLICY,
) -> dict[str, Any]:
    """Common residual solve for pore concentrations, electrical potential and flux.

    With infinite inlet flow this is a local membrane query. A finite solvent
    inlet couples the 0D mean concentration to the actual retentate stream.
    """
    prop, ids, t = spec.properties, spec.properties.ion_ids, spec.temperature_K
    n, m = len(ids), spec.pore_segments
    if (
        set(concentrations) != set(ids)
        or solvent_flow_m3_s <= 0
        or any(
            not isfinite(v) or not 0 <= v <= spec.maximum_concentration_mol_m3
            for v in concentrations.values()
        )
    ):
        raise ValueError("NF input outside declared ion/solvent-flow domain")
    prop.validate_amounts(prop.amounts(1, concentrations))
    ci = np.array([concentrations[k] for k in ids])
    z = np.array([prop.charges[k] for k in ids])
    vw = prop.water_volume_m3_mol
    mu = prop.viscosity.evaluate(t, spec.inlet_pressure_Pa, float(sum(ci)))
    hydraulic = spec.pore_radius_m**2 / (8 * mu * spec.effective_thickness_m)
    dp = spec.inlet_pressure_Pa - spec.reservoir_pressure_Pa
    jscale = hydraulic * dp
    if not any(ci):
        if spec.fixed_charge_mol_m3:
            raise ValueError("charged pore requires mobile counterions")
        if spec.membrane_area_m2 * jscale >= solvent_flow_m3_s:
            raise ValueError("membrane recovery exhausts actual inlet solvent")
        zeros = dict.fromkeys(ids, 0.0)
        return {
            "status": "converged",
            "water_flux_m_s": jscale,
            "permeate_concentrations_mol_m3": zeros,
            "ion_fluxes_mol_m2_s": zeros,
            "mean_feed_concentrations_mol_m3": zeros,
            "retentate_concentrations_mol_m3": zeros,
            "rejection": dict.fromkeys(ids, None),
            "pore_concentrations_mol_m3": [zeros] * (m + 1),
            "pore_dimensionless_potential": [0.0] * (m + 1),
            "permeate_dimensionless_potential": 0.0,
            "water_chemical_potential_difference_J_mol": vw * dp,
            "charge_residual_mol_m3": 0.0,
            "current_molar_charge_flux_mol_m2_s": 0.0,
            "solver": {"algorithm": "pure_water_analytic_limit", "scaled_residual_norm": 0.0},
        }
    if np.any(ci <= 0):
        raise ValueError(
            "ion transport requires every declared ion present; pure water is a separate limit"
        )
    factors = pore_partition_factors(spec)
    kpart = np.array([factors[k]["steric"] * factors[k]["born"] for k in ids])
    kd = np.array(
        [
            factors[k]["diffusive_hindrance"] * ion.diffusivity_m2_s
            for k, ion in zip(ids, spec.ions, strict=True)
        ]
    )
    kc = np.array([factors[k]["convective_hindrance"] for k in ids])
    cscale = max(float(max(ci)), abs(spec.fixed_charge_mol_m3), 1.0)
    fscale = kd * cscale / spec.effective_thickness_m + jscale * cscale
    size_c, size_phi = (m + 1) * n, m + 1

    def decode(y: Any) -> tuple[Any, Any, Any, float, float]:
        pore = np.exp(y[:size_c].reshape((m + 1, n)))
        phi = y[size_c : size_c + size_phi]
        cp = np.exp(y[size_c + size_phi : -2])
        return pore, phi, cp, float(y[-2]), jscale * exp(float(y[-1]))

    def external(cp: Any, j: float) -> tuple[Any, Any]:
        recovery = spec.membrane_area_m2 * j / solvent_flow_m3_s
        if recovery >= 1:
            raise ResidualDomainError("actual solvent stream exhausted")
        cr = (ci - recovery * cp) / (1 - recovery)
        cf = (ci + cr) / 2
        if np.any(cr <= 0) or np.any(cf > spec.maximum_concentration_mol_m3):
            raise ResidualDomainError("retentate outside positive declared ion domain")
        return cf, cr

    def water_drive(cf: Any, cp: Any) -> float:
        values = []
        for c, pressure in ((cf, spec.inlet_pressure_Pa), (cp, spec.reservoir_pressure_Pa)):
            potentials = prop.database.chemical_potentials(
                prop.amounts(1.0, dict(zip(ids, c, strict=True))),
                temperature_K=t,
                pressure_Pa=pressure,
            )
            values.append(cast(float, potentials.chemical_potentials_J_mol[prop.water_id]))
        return values[0] - values[1]

    def residual(time: float, y: Any, rate: Any, parameters: Any) -> Any:
        pore, phi, cp, pp, j = decode(y)
        cf, _ = external(cp, j)
        mids = (pore[1:] + pore[:-1]) / 2
        dx = spec.effective_thickness_m / m
        actual = (
            -kd * np.diff(pore, axis=0) / dx
            - z * kd * mids * np.diff(phi)[:, None] / dx
            + kc * mids * j
        )
        return np.concatenate(
            (
                np.log(pore[0] / (kpart * cf)) + z * phi[0],
                np.log(pore[-1] / (kpart * cp)) + z * (phi[-1] - pp),
                ((actual - j * cp) / fscale).ravel(),
                (pore @ z + spec.fixed_charge_mol_m3) / cscale,
                [float(cp @ z) / cscale, (j - hydraulic * water_drive(cf, cp) / vw) / jscale],
            )
        )

    start_cp = ci * 0.8
    feed_part = donnan_partition(spec, concentrations)
    product_part = donnan_partition(spec, dict(zip(ids, start_cp, strict=True)))
    left = np.array([feed_part["pore_concentrations_mol_m3"][k] for k in ids])
    right = np.array([product_part["pore_concentrations_mol_m3"][k] for k in ids])
    initial = np.concatenate(
        (
            np.log(np.linspace(left, right, m + 1)).ravel(),
            np.linspace(
                feed_part["dimensionless_potential"], product_part["dimensionless_potential"], m + 1
            ),
            np.log(start_cp),
            [0.0, log(0.7)],
        )
    )
    variables = tuple(
        ResidualVariable(
            f"log_c_{i}",
            "1",
            role="algebraic",
            lower=-35,
            upper=log(spec.maximum_concentration_mol_m3 * 100),
        )
        for i in range(size_c)
    )
    variables += tuple(
        ResidualVariable(f"phi_{i}", "1", role="algebraic", lower=-100, upper=100)
        for i in range(size_phi)
    )
    variables += tuple(
        ResidualVariable(
            f"log_cp_{k}",
            "1",
            role="algebraic",
            lower=-35,
            upper=log(spec.maximum_concentration_mol_m3),
        )
        for k in ids
    )
    variables += (
        ResidualVariable("phi_permeate", "1", role="algebraic", lower=-100, upper=100),
        ResidualVariable("log_relative_water_flux", "1", role="algebraic", lower=-25, upper=log(5)),
    )
    system = ResidualSystem(
        "dspm_de_midpoint_pore_v1",
        variables,
        tuple(ResidualEquation(f"pore_equation_{i}", "1") for i in range(len(variables))),
        residual,
        ResidualDerivatives("finite_difference"),
    )
    solved = solve_steady_residual(system, initial, policy=policy)
    diagnostic = {
        **solved.diagnostics,
        **solved.root.diagnostics,
        "pore_segments": m,
        "failure_reason": solved.root.failure_reason,
    }
    if not solved.root.success:
        return {"status": "not_converged", "solver": diagnostic}
    pore, phi, cp, pp, j = decode(solved.root.value)
    cf, cr = external(cp, j)
    return {
        "status": "converged",
        "water_flux_m_s": j,
        "permeate_concentrations_mol_m3": dict(zip(ids, map(float, cp), strict=True)),
        "ion_fluxes_mol_m2_s": dict(zip(ids, map(float, j * cp), strict=True)),
        "mean_feed_concentrations_mol_m3": dict(zip(ids, map(float, cf), strict=True)),
        "retentate_concentrations_mol_m3": dict(zip(ids, map(float, cr), strict=True)),
        "rejection": dict(zip(ids, map(float, 1 - cp / ci), strict=True)),
        "pore_concentrations_mol_m3": [
            dict(zip(ids, map(float, row), strict=True)) for row in pore
        ],
        "pore_dimensionless_potential": phi.tolist(),
        "permeate_dimensionless_potential": pp,
        "water_chemical_potential_difference_J_mol": water_drive(cf, cp),
        "charge_residual_mol_m3": float(
            max(np.max(np.abs(pore @ z + spec.fixed_charge_mol_m3)), abs(cp @ z))
        ),
        "current_molar_charge_flux_mol_m2_s": float(j * (cp @ z)),
        "solver": diagnostic,
    }


def nanofiltration_profile(
    spec: NanofiltrationSpec,
    feed_amounts: Mapping[str, float],
    requested_flow_m3_s: float,
    *,
    policy: RootPolicy = DEFAULT_NF_ROOT_POLICY,
) -> dict[str, Any]:
    prop, t, p = spec.properties, spec.temperature_K, spec.reservoir_pressure_Pa
    prop.validate_amounts(feed_amounts)
    point = pump_operating_point(
        spec.pump, requested_flow_m3_s, lambda q: spec.inlet_pressure_Pa - p
    )
    q = point["actual_flow_m3_s"]
    incoming = {k: v * q / prop.volume(feed_amounts, t, p) for k, v in feed_amounts.items()}
    if q == 0:
        return {
            **point,
            "inlet_mol_s": incoming,
            "permeate_mol_s": incoming,
            "retentate_mol_s": incoming,
            "water_recovery": 0.0,
            "backpressure_dissipation_W": 0.0,
            "pore": {"status": "idle"},
        }
    pore = nanofiltration_point(
        spec,
        prop.concentrations(feed_amounts),
        solvent_flow_m3_s=incoming[prop.water_id] * prop.water_volume_m3_mol,
        policy=policy,
    )
    if pore["status"] != "converged":
        raise RuntimeError(f"nanofiltration pore failed: {pore['solver']}")
    permeate = {
        prop.water_id: spec.membrane_area_m2 * pore["water_flux_m_s"] / prop.water_volume_m3_mol,
        **{k: spec.membrane_area_m2 * v for k, v in pore["ion_fluxes_mol_m2_s"].items()},
    }
    retained = {k: incoming[k] - permeate[k] for k in incoming}
    if min(retained.values()) < 0:
        raise ValueError("membrane exhausts actual ion inlet stream")
    return {
        **point,
        "inlet_mol_s": incoming,
        "permeate_mol_s": permeate,
        "retentate_mol_s": retained,
        "water_recovery": permeate[prop.water_id] / incoming[prop.water_id],
        "backpressure_dissipation_W": (spec.inlet_pressure_Pa - p) * prop.volume(retained, t, p),
        "pore": pore,
    }


def nanofiltration_inventory(
    spec: NanofiltrationSpec,
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


def advance_nanofiltration(
    spec: NanofiltrationSpec,
    state: TransportInventory,
    requested_flow_m3_s: float,
    duration_s: float,
    *,
    progress: Any = None,
) -> tuple[Any, dict[str, Any]]:
    prop, t, p = spec.properties, spec.temperature_K, spec.reservoir_pressure_Pa
    if len(state.amounts_mol) != 3 or len(state.enthalpies_J) != 3:
        raise ValueError("NF requires three original finite reservoirs")
    for i, n in enumerate(state.amounts_mol):
        prop.validate_amounts(n, empty=i > 0)
        if not isclose(state.enthalpies_J[i], spec.enthalpy(i, n, t), rel_tol=1e-8, abs_tol=1e-6):
            raise ValueError("NF inventory H differs from declared isotherm")
    profile = nanofiltration_profile(spec, state.amounts_mol[0], requested_flow_m3_s)
    flows = (
        {k: -v for k, v in profile["inlet_mol_s"].items()},
        profile["permeate_mol_s"],
        profile["retentate_mol_s"],
    )
    matrix = np.array([[n[k] for k in prop.component_ids] for n in flows])
    shaft, hydraulic = profile["shaft_power_W"], profile["hydraulic_power_W"]

    def rates(candidate: Any, unused: Any) -> PhaseInventoryRates:
        elapsed = candidate.time_s - state.time_s
        physical = (
            state.amounts_mol[0],
            *(
                {k: state.amounts_mol[i][k] + elapsed * flows[i][k] for k in prop.component_ids}
                for i in (1, 2)
            ),
        )
        hdot = np.array(
            [prop.enthalpy_rate(n, flow, t, p) for n, flow in zip(physical, flows, strict=True)]
        )
        bath = float(sum(hdot)) - hydraulic
        return PhaseInventoryRates(
            matrix, hdot, shaft + max(bath, 0.0), shaft - hydraulic + max(-bath, 0.0), ()
        )

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
        solver_id="finite_multiion_nf_nH_v1",
    )
    dt, final = result.actual_duration_s, result.final_state
    for i, n in enumerate(final.amounts_mol):
        prop.validate_amounts(n, empty=i > 0)
        if not isclose(final.enthalpies_J[i], spec.enthalpy(i, n, t), rel_tol=1e-7, abs_tol=1e-5):
            raise RuntimeError("finite NF failed same-table H closure")
    change = sum(final.enthalpies_J) - sum(state.enthalpies_J)
    return result, {
        "profile": profile,
        "shaft_work_J": shaft * dt,
        "hydraulic_work_J": hydraulic * dt,
        "motor_loss_J": (shaft - hydraulic) * dt,
        "bath_heat_into_liquid_J": change - hydraulic * dt,
        "backpressure_dissipation_J": profile["backpressure_dissipation_W"] * dt,
        "liquid_enthalpy_change_J": change,
        "energy_scope": "shaft - external motor loss + signed isothermal bath = liquid H",
    }
