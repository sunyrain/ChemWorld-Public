"""Periodic interior C|AEM|D|CEM|C with original finite aqueous n/H.

Terminal voltage covers this periodic cell pair, never end electrodes. The
accepted slice is powered ED at common fixed T/P, with no external matter.
"""

import json
from collections.abc import Mapping
from dataclasses import dataclass, replace
from math import isclose, isfinite
from typing import Any

import numpy as np

from chemworld.physchem.ion_exchange_pore import F, IonExchangePoreSpec, PoreResidualBlock
from chemworld.physchem.multiion_membrane_properties import MultiIonMembraneProperties
from chemworld.physchem.phase_inventory import (
    PhaseInventoryAdvance,
    PhaseInventoryRates,
    integrate_phase_inventory,
)
from chemworld.physchem.property_reports import R_J_PER_MOL_K as R
from chemworld.physchem.reactor_shared import ScalarControlProgram
from chemworld.physchem.residual_system import (
    ResidualDerivatives,
    ResidualEquation,
    ResidualSystem,
    ResidualVariable,
    RootPolicy,
    solve_steady_residual,
)
from chemworld.physchem.transport_devices import TransportInventory

DEFAULT_ED_POLICY = RootPolicy(tolerance=1e-11)


@dataclass(frozen=True)
class ElectrodialysisSpec:
    properties: MultiIonMembraneProperties
    aem: IonExchangePoreSpec
    cem: IonExchangePoreSpec
    membrane_area_m2: float
    temperature_K: float
    reservoir_pressure_Pa: float
    capacities_L: tuple[float, float]
    solution_resistance_Ohm_m2: float
    maximum_current_A: float
    maximum_voltage_V: float
    source_id: str
    concentrate_vessel_id: str = "concentrate"
    minimum_water_mol: float = 1e-4
    minimum_ion_concentration_mol_m3: float = 1e-3
    support_heat_capacity_J_K: float = 5.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "capacities_L", tuple(self.capacities_L))
        positive = (
            self.membrane_area_m2,
            self.temperature_K,
            self.reservoir_pressure_Pa,
            *self.capacities_L,
            self.maximum_current_A,
            self.maximum_voltage_V,
            self.minimum_water_mol,
            self.minimum_ion_concentration_mol_m3,
            self.support_heat_capacity_J_K,
        )
        if (
            any(not isfinite(v) or v <= 0 for v in positive)
            or len(self.capacities_L) != 2
            or not isfinite(self.solution_resistance_Ohm_m2)
            or self.solution_resistance_Ohm_m2 < 0
            or not self.concentrate_vessel_id
            or self.properties.database.activity_model.family != "ideal"
            or self.source_id not in {s.source_id for s in self.properties.database.sources}
        ):
            raise ValueError(
                "periodic ED needs sourced ideal finite liquids and bounded electrical source"
            )
        if not self.aem.fixed_charge_mol_m3 > 0 > self.cem.fixed_charge_mol_m3:
            raise ValueError("periodic orientation requires positive AEM then negative CEM")
        for pore in (self.aem, self.cem):
            if tuple(k for k, _ in pore.effective_diffusivities_m2_s) != self.properties.ion_ids:
                raise ValueError("both ED membranes must transport the same original ions")
        self.properties.database.standard_states(self.temperature_K, self.reservoir_pressure_Pa)

    def enthalpy(self, index: int, n: Mapping[str, float], temperature: float) -> float:
        return self.properties.enthalpy(
            n, temperature, self.reservoir_pressure_Pa
        ) + self.support_heat_capacity_J_K * (temperature - self.temperature_K)

    def to_dict(self) -> dict[str, Any]:
        data = {k: v for k, v in vars(self).items() if k not in {"properties", "aem", "cem"}}
        data.update(
            schema="periodic-interior-ed-v1",
            properties=self.properties.to_dict(),
            aem=self.aem.to_dict(),
            cem=self.cem.to_dict(),
        )
        return json.loads(json.dumps(data))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ElectrodialysisSpec":
        data = dict(payload)
        if data.pop("schema") != "periodic-interior-ed-v1":
            raise ValueError("unsupported electrodialysis declaration")
        data["properties"] = MultiIonMembraneProperties.from_dict(data["properties"])
        for name in ("aem", "cem"):
            data[name] = IonExchangePoreSpec(**data[name])
        return cls(**data)


def electrodialysis_point(
    spec: ElectrodialysisSpec,
    amounts: tuple[Mapping[str, float], ...],
    target: float,
    unit: str,
    *,
    policy: RootPolicy = DEFAULT_ED_POLICY,
    numerical_trial: bool = False,
) -> dict[str, Any]:
    if len(amounts) != 2 or unit not in {"A", "V", "Ohm"} or not isfinite(target):
        raise ValueError("ED needs two finite liquids and a finite current, voltage or load")
    if unit == "Ohm" and (isinstance(target, bool) or target < 0):
        raise ValueError("passive load resistance must be finite nonnegative Ohm")
    prop = spec.properties
    if not numerical_trial:
        for n in amounts:
            prop.validate_amounts(n)
    concentrations = tuple(prop.concentrations(n) for n in amounts)
    blocks = (
        PoreResidualBlock(
            spec.aem,
            prop,
            concentrations[1],
            concentrations[0],
            spec.temperature_K,
            spec.reservoir_pressure_Pa,
        ),
        PoreResidualBlock(
            spec.cem,
            prop,
            concentrations[0],
            concentrations[1],
            spec.temperature_K,
            spec.reservoir_pressure_Pa,
        ),
    )
    widths = [b.width for b in blocks]
    split = widths[0]
    size = sum(widths) + 1
    jscale = max(1.0, *[F * float(max(b.flux_scales)) for b in blocks])
    vscale = max(R * spec.temperature_K / F, 0.01)
    area = spec.membrane_area_m2
    approximate_resistance = spec.solution_resistance_Ohm_m2 + sum(
        R
        * spec.temperature_K
        * b.spec.thickness_m
        / (F * F * float(np.sum(b.z * b.z * b.diffusion * np.mean(b.end_c, axis=0))))
        for b in blocks
    )
    initial_j = (
        0.0 if unit == "Ohm" else target / area if unit == "A" else target / approximate_resistance
    )
    variables = tuple(
        replace(v, name=f"{name}:{v.name}")
        for name, b in zip(("aem", "cem"), blocks, strict=True)
        for v in b.variables
    )
    variables += (ResidualVariable("common_current_density", "1", role="algebraic"),)

    def voltage(y: Any, j: float) -> float:
        return (
            sum(
                R * spec.temperature_K / F * (b.phi_right_partition - part[b.nc + b.nphi - 1])
                for b, part in zip(blocks, (y[:split], y[split:-1]), strict=True)
            )
            + j * spec.solution_resistance_Ohm_m2
        )

    def residual(time: float, y: Any, rate: Any, parameters: Any) -> Any:
        j = y[-1] * jscale
        electrical = (
            (j - target / area) / jscale if unit == "A" else (voltage(y, j) - target) / vscale
        )
        if unit == "Ohm":
            electrical = (voltage(y, j) + area * j * target) / vscale
        return np.r_[
            blocks[0].residual(y[:split], j), blocks[1].residual(y[split:-1], j), electrical
        ]

    def jacobian(time: float, y: Any, rate: Any, parameters: Any) -> Any:
        jac = np.zeros((size, size))
        offset = 0
        j = y[-1] * jscale
        for b in blocks:
            width = b.width
            native, dj = b.jacobian(y[offset : offset + width], j)
            jac[offset : offset + width, offset : offset + width] = native
            jac[offset : offset + width, -1] = dj * jscale
            if unit in {"V", "Ohm"}:
                jac[-1, offset + b.nc + b.nphi - 1] = -R * spec.temperature_K / F / vscale
            offset += width
        jac[-1, -1] = 1 if unit == "A" else jscale * spec.solution_resistance_Ohm_m2 / vscale
        if unit == "Ohm":
            jac[-1, -1] += jscale * area * target / vscale
        return jac

    system = ResidualSystem(
        "periodic_dual_ion_exchange_pore_v1",
        variables,
        tuple(ResidualEquation(f"pair_equation_{i}", "1") for i in range(size)),
        residual,
        ResidualDerivatives(
            "analytic", state=jacobian, rate=lambda t, y, rate, p: np.zeros((size, size))
        ),
    )
    solved = solve_steady_residual(
        system, np.r_[*(b.initial(initial_j) for b in blocks), initial_j / jscale], policy=policy
    )
    diagnostic = {
        **solved.root.diagnostics,
        **solved.diagnostics,
        "failure_reason": solved.root.failure_reason,
    }
    if not solved.root.success:
        return {"status": "not_converged", "solver": diagnostic}
    y = solved.root.value
    j = float(y[-1] * jscale)
    aem, cem = [
        b.readout(part, j) for b, part in zip(blocks, (y[:split], y[split:-1]), strict=True)
    ]
    rates = {
        prop.water_id: area
        * (aem["water_flux_m_s"] - cem["water_flux_m_s"])
        / prop.water_volume_m3_mol,
        **{
            k: area * (aem["ion_fluxes_mol_m2_s"][k] - cem["ion_fluxes_mol_m2_s"][k])
            for k in prop.ion_ids
        },
    }
    current = j * area
    v = voltage(y, j)
    return {
        "status": "converged",
        "aem": aem,
        "cem": cem,
        "current_A": current,
        "voltage_V": v,
        "electrical_power_W": current * v,
        "diluate_rates_mol_s": rates,
        "concentrate_rates_mol_s": {k: -x for k, x in rates.items()},
        "dissipation_W": area
        * (
            aem["dissipation_W_m2"]
            + cem["dissipation_W_m2"]
            + j * j * spec.solution_resistance_Ohm_m2
        ),
        "solver": diagnostic,
        "voltage_scope": "periodic interior pair only; no end electrodes",
    }


def electrodialysis_inventory(
    spec: ElectrodialysisSpec, diluate: Mapping[str, float], concentrate: Mapping[str, float]
) -> TransportInventory:
    ns = (dict(diluate), dict(concentrate))
    for n in ns:
        spec.properties.validate_amounts(n)
    return TransportInventory(
        ns, tuple(spec.enthalpy(i, n, spec.temperature_K) for i, n in enumerate(ns))
    )


def advance_electrodialysis(
    spec: ElectrodialysisSpec,
    state: TransportInventory,
    program: ScalarControlProgram,
    duration_s: float,
    *,
    progress: Any = None,
) -> tuple[Any, dict[str, Any]]:
    prop, t, p = spec.properties, spec.temperature_K, spec.reservoir_pressure_Pa
    if (
        len(state.amounts_mol) != 2
        or len(state.enthalpies_J) != 2
        or not isfinite(duration_s)
        or duration_s < 0
    ):
        raise ValueError("ED requires two finite n/H tanks and nonnegative duration")
    for i, n in enumerate(state.amounts_mol):
        prop.validate_amounts(n)
        if not isclose(state.enthalpies_J[i], spec.enthalpy(i, n, t), rel_tol=1e-8, abs_tol=1e-6):
            raise ValueError("ED inventory H differs from fixed isotherm")
    counters = {"pore_solves": 0, "rhs_evaluations": 0, "off_neutral_numerical_probes": 0}
    cache: list[Any] = [None, None]

    def point(candidate: Any) -> dict[str, Any]:
        key = (
            candidate.time_s,
            tuple(tuple(n[k] for k in prop.component_ids) for n in candidate.amounts_mol),
        )
        if cache[0] == key:
            return cache[1]
        for n in candidate.amounts_mol:
            charge = sum(prop.charges[k] * n[k] for k in prop.ion_ids)
            if abs(charge) > 1e-12:
                counters["off_neutral_numerical_probes"] += 1
        row = electrodialysis_point(
            spec,
            candidate.amounts_mol,
            program.value_at(candidate.time_s),
            program.unit,
            numerical_trial=True,
        )
        counters["pore_solves"] += 2
        if row["status"] != "converged":
            raise RuntimeError(f"ED coupled pore failed: {row['solver']}")
        if row["electrical_power_W"] < -1e-10:
            raise ValueError("RED power generation is outside this powered-ED slice")
        cache[:] = [key, row]
        return row

    def rates(candidate: Any, flashes: Any) -> PhaseInventoryRates:
        row = point(candidate)
        counters["rhs_evaluations"] += 1
        flows = (row["diluate_rates_mol_s"], row["concentrate_rates_mol_s"])
        hdot = np.array(
            [
                prop.enthalpy_rate(n, f, t, p)
                for n, f in zip(candidate.amounts_mol, flows, strict=True)
            ]
        )
        power = row["electrical_power_W"]
        # Original generic utility integrals numerically own input/removal.
        # Runtime maps these once to electrical work and signed bath, not heater.
        return PhaseInventoryRates(
            np.array([[f[k] for k in prop.component_ids] for f in flows]),
            hdot,
            power,
            power - float(sum(hdot)),
            (),
        )

    stops: list[Any] = []
    for i in range(2):
        stops.extend(
            (
                (
                    f"water_minimum:{i}",
                    lambda s, _, i=i: s.amounts_mol[i][prop.water_id] - spec.minimum_water_mol,
                ),
                (
                    f"capacity:{i}",
                    lambda s, _, i=i: (
                        spec.capacities_L[i] - 1000 * prop.volume(s.amounts_mol[i], t, p)
                    ),
                ),
                (
                    f"ion_minimum:{i}",
                    lambda s, _, i=i: (
                        min(prop.concentrations(s.amounts_mol[i]).values())
                        - spec.minimum_ion_concentration_mol_m3
                    ),
                ),
            )
        )
    stops.extend(
        (
            (
                "voltage_compliance",
                lambda s, _: spec.maximum_voltage_V - abs(point(s)["voltage_V"]),
            ),
            (
                "current_compliance",
                lambda s, _: spec.maximum_current_A - abs(point(s)["current_A"]),
            ),
        )
    )
    initial_point = point(state)
    finish = state.time_s + duration_s
    boundaries = sorted(
        {float(x) for x, _ in program.setpoints if state.time_s < x < finish} | {finish}
    )
    current = state
    pieces = []
    trajectory = [state]
    for end in boundaries:
        result = integrate_phase_inventory(
            spec,
            current,
            duration_s=end - current.time_s,
            rates=rates,
            closure_indices=(),
            stop_functions=stops,
            enthalpy_bounds=lambda i, n: (-np.inf, np.inf),
            initial_boundary="stop",
            progress=progress,
            solver_id="finite_periodic_ed_nH_v1",
        )
        pieces.append(result)
        current = result.final_state
        trajectory.extend(result.trajectory[1:])
        if result.stopping_event:
            break
    for i, n in enumerate(current.amounts_mol):
        prop.validate_amounts(n)
        if not isclose(current.enthalpies_J[i], spec.enthalpy(i, n, t), rel_tol=1e-8, abs_tol=1e-5):
            raise RuntimeError("finite ED failed same-table isothermal H closure")
    work = current.heater_energy_J - state.heater_energy_J
    heat_out = current.condenser_energy_J - state.condenser_energy_J
    component = {
        k: sum(n[k] for n in current.amounts_mol) - sum(n[k] for n in state.amounts_mol)
        for k in prop.component_ids
    }
    energy = sum(current.enthalpies_J) - sum(state.enthalpies_J) - work + heat_out
    combined = PhaseInventoryAdvance(
        state,
        current,
        tuple(trajectory),
        pieces[-1].stopping_event,
        {**counters, "intervals": [r.solver_diagnostic for r in pieces]},
        component,
        energy,
    )
    return combined, {
        "initial": initial_point,
        "final": point(current),
        "electrical_work_J": work,
        "bath_heat_into_liquid_J": -heat_out,
        "liquid_enthalpy_change_J": sum(current.enthalpies_J) - sum(state.enthalpies_J),
        "energy_scope": "electrical work + signed isothermal bath = original closed ideal liquid H",
        "bath_allocation": "equal halves to the two declared isothermal reservoir baths",
    }
