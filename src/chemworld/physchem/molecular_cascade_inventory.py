"""Four finite moving-volume tanks for the original steady molecular cascade."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from math import isfinite
from typing import Any

import numpy as np

from chemworld.physchem.molecular_cascade import MolecularCascadeSpec, solve_molecular_cascade
from chemworld.physchem.molecular_mixture import (
    MolecularCaloricModel,
    MolecularCaloricState,
    MolecularStream,
)
from chemworld.physchem.molecular_processes import MolecularProcessDomain
from chemworld.physchem.molecular_tanks import (
    molecular_tank_ph,
    molecular_tank_tp,
    validate_molecular_amounts,
)
from chemworld.physchem.phase_inventory import (
    PhaseInventoryAdvance,
    PhaseInventoryRates,
    integrate_phase_inventory,
)
from chemworld.physchem.transport_devices import TransportInventory


@dataclass(frozen=True)
class MolecularCascadeInventorySpec:
    cascade: MolecularCascadeSpec
    vessel_ids: tuple[str, ...] = ("gas_source", "liquid_source", "reactor", "liquid_receiver")
    feed_rates_mol_s: tuple[float, float] = (0.001, 0.00075)
    tank_temperature_bounds_K: tuple[tuple[float, float], ...] = (
        (325.0, 360.0),
        (265.0, 295.0),
        (325.0, 360.0),
        (265.0, 295.0),
    )
    source_reserves_mol: tuple[float, float] = (1e-5, 1e-5)
    receiver_capacities_mol: tuple[float, float] = (0.1, 0.1)
    tank_capacities_L: tuple[float, ...] = (1.0, 1.0, 1.0, 1.0)

    def __post_init__(self) -> None:
        for key in (
            "vessel_ids",
            "feed_rates_mol_s",
            "source_reserves_mol",
            "receiver_capacities_mol",
            "tank_capacities_L",
        ):
            object.__setattr__(self, key, tuple(getattr(self, key)))
        object.__setattr__(
            self,
            "tank_temperature_bounds_K",
            tuple(tuple(b) for b in self.tank_temperature_bounds_K),
        )
        if (
            len(self.vessel_ids) != 4
            or len(set(self.vessel_ids)) != 4
            or any(not isinstance(v, str) or not v for v in self.vessel_ids)
            or any(
                len(getattr(self, key)) != 2
                for key in ("feed_rates_mol_s", "source_reserves_mol", "receiver_capacities_mol")
            )
            or len(self.tank_capacities_L) != 4
            or len(self.tank_temperature_bounds_K) != 4
            or any(
                isinstance(x, bool) or not isfinite(x) or x <= 0
                for x in (
                    *self.feed_rates_mol_s,
                    *self.source_reserves_mol,
                    *self.receiver_capacities_mol,
                    *self.tank_capacities_L,
                )
            )
        ):
            raise ValueError("cascade requires four finite owners and positive finite rates/limits")
        for i in range(4):
            self.domain(i)

    @property
    def properties(self) -> MolecularCaloricModel:
        """Original n/H integrator's component-identity protocol; no duplicate EOS."""
        return self.cascade.model

    def domain(self, index: int) -> MolecularProcessDomain:
        return MolecularProcessDomain(
            self.cascade.model,
            self.tank_temperature_bounds_K[index],
            self.cascade.basins[index % 2],
        )

    def validate_amounts(self, amounts: Mapping[str, float]) -> float:
        return validate_molecular_amounts(self.properties, amounts)

    def at_tp(
        self, index: int, amounts: Mapping[str, float], temperature: float
    ) -> MolecularCaloricState:
        return molecular_tank_tp(self.domain(index), self.cascade.pressure_Pa, amounts, temperature)

    def at_ph(
        self, index: int, amounts: Mapping[str, float], enthalpy: float
    ) -> MolecularCaloricState:
        return molecular_tank_ph(
            self.domain(index),
            self.cascade.pressure_Pa,
            amounts,
            enthalpy,
            self.cascade.root_policy,
        )

    def enthalpy_bounds(self, index: int, amounts: Mapping[str, float]) -> tuple[float, float]:
        total = self.validate_amounts(amounts)
        values = tuple(
            total * self.at_tp(index, amounts, t).enthalpy_J_mol
            for t in self.tank_temperature_bounds_K[index]
        )
        return values[0], values[1]

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_type": "molecular_cascade_four_tank_v1",
            "cascade": self.cascade.to_dict(),
            **{
                key: getattr(self, key)
                for key in (
                    "vessel_ids",
                    "feed_rates_mol_s",
                    "tank_temperature_bounds_K",
                    "source_reserves_mol",
                    "receiver_capacities_mol",
                    "tank_capacities_L",
                )
            },
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> MolecularCascadeInventorySpec:
        if data.get("model_type") != "molecular_cascade_four_tank_v1":
            raise ValueError("unknown molecular finite cascade declaration")
        return cls(
            MolecularCascadeSpec.from_dict(data["cascade"]),
            **{
                key: data[key]
                for key in (
                    "vessel_ids",
                    "feed_rates_mol_s",
                    "tank_temperature_bounds_K",
                    "source_reserves_mol",
                    "receiver_capacities_mol",
                    "tank_capacities_L",
                )
            },
        )


def molecular_cascade_inventory(
    spec: MolecularCascadeInventorySpec,
    amounts: Sequence[Mapping[str, float]],
    temperatures_K: Sequence[float],
) -> TransportInventory:
    if len(amounts) != 4 or len(temperatures_K) != 4:
        raise ValueError("cascade inventory requires all four original tank states")
    return TransportInventory(
        tuple(dict(n) for n in amounts),
        tuple(
            sum(n.values()) * spec.at_tp(i, n, temperatures_K[i]).enthalpy_J_mol
            for i, n in enumerate(amounts)
        ),
    )


class MolecularCascadeSolveError(ValueError):
    def __init__(self, reason: str, diagnostics: dict[str, Any]) -> None:
        super().__init__(reason)
        self.diagnostics = diagnostics


def advance_molecular_cascade(
    spec: MolecularCascadeInventorySpec,
    state: TransportInventory,
    duration_s: float,
    *,
    progress: Callable[[float, int], None] | None = None,
) -> tuple[PhaseInventoryAdvance, dict[str, Any]]:
    """Advance n/H once; constant-P homogeneous withdrawal keeps donor T/x fixed.

    Original donor PH is read at each action. Equipment contributes the actual
    inlet/outlet H rates exactly once; its Q separately meters utility energy.
    """
    if len(state.amounts_mol) != 4 or len(state.enthalpies_J) != 4:
        raise ValueError("cascade needs four finite tank inventories")
    initial_states = tuple(
        spec.at_ph(i, n, h)
        for i, (n, h) in enumerate(zip(state.amounts_mol, state.enthalpies_J, strict=True))
    )
    feeds = tuple(MolecularStream(initial_states[i], spec.feed_rates_mol_s[i]) for i in (0, 1))
    solved = solve_molecular_cascade(spec.cascade, *feeds)
    if not solved.success:
        raise MolecularCascadeSolveError(
            "molecular cascade failed: " + solved.failure_reason, solved.diagnostics
        )
    streams = (*feeds, *solved.outlets)
    rates = np.array(
        [
            (-1 if i < 2 else 1) * s.molar_rate_mol_s * np.array(s.state.mole_fractions)
            for i, s in enumerate(streams)
        ]
    )
    h_rates = np.array([(-1 if i < 2 else 1) * s.enthalpy_rate_W for i, s in enumerate(streams)])
    contribution = PhaseInventoryRates(
        rates, h_rates, solved.diagnostics["heater_W"], solved.diagnostics["condenser_W"], ()
    )
    stops = []
    for i in (0, 1):
        stops.append(
            (
                f"source_reserve:{spec.vessel_ids[i]}",
                lambda s, _, i=i: sum(s.amounts_mol[i].values()) - spec.source_reserves_mol[i],
            )
        )
    for i in (2, 3):
        stops.append(
            (
                f"receiver_capacity:{spec.vessel_ids[i]}",
                lambda s, _, i=i: (
                    spec.receiver_capacities_mol[i - 2] - sum(s.amounts_mol[i].values())
                ),
            )
        )
        stops.append(
            (
                f"tank_volume:{spec.vessel_ids[i]}",
                lambda s, phases, i=i: (
                    spec.tank_capacities_L[i]
                    - 1000 * sum(s.amounts_mol[i].values()) / phases[i].molar_density_mol_m3
                ),
            )
        )
    result = integrate_phase_inventory(
        spec,
        state,
        duration_s=duration_s,
        rates=lambda _state, _phases: contribution,
        closure_indices=(2, 3),
        stop_functions=stops,
        enthalpy_bounds=spec.enthalpy_bounds,
        progress=progress,
        initial_boundary="stop",
        solver_id="molecular_cascade_four_tank_bdf_v1",
    )
    final = result.final_state
    final_states = tuple(
        spec.at_ph(i, n, h)
        for i, (n, h) in enumerate(zip(final.amounts_mol, final.enthalpies_J, strict=True))
    )
    volumes = [
        tuple(
            sum(n.values()) / p.molar_density_mol_m3
            for n, p in zip(inv.amounts_mol, phases, strict=True)
        )
        for inv, phases in ((state, initial_states), (final, final_states))
    ]
    if any(
        1000 * v > cap + 1e-7 for v, cap in zip(volumes[1], spec.tank_capacities_L, strict=True)
    ):
        raise ValueError("accepted cascade tank exceeds its actual finite volume")
    work = spec.cascade.pressure_Pa * (sum(volumes[1]) - sum(volumes[0]))
    q = (
        final.heater_energy_J
        - state.heater_energy_J
        - final.condenser_energy_J
        + state.condenser_energy_J
    )
    du = sum(final.enthalpies_J) - sum(state.enthalpies_J) - work
    diagnostic = {
        "steady": solved.to_dict(),
        "actual_duration_s": result.actual_duration_s,
        "heat_J": q,
        "heater_energy_J": final.heater_energy_J - state.heater_energy_J,
        "condenser_energy_J": final.condenser_energy_J - state.condenser_energy_J,
        "pressure_work_J": work,
        "internal_energy_change_J": du,
        "internal_energy_residual_J": du - q + work,
        "shaft_work_J": 0.0,
        "initial_volumes_m3": volumes[0],
        "final_volumes_m3": volumes[1],
        "phase_scope": "declared_homogeneous_tanks_no_phase_switching",
    }
    return result, diagnostic
