"""One finite source and actual-phase finite products on the shared n/H solver."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from math import isfinite
from typing import Any

import numpy as np

from chemworld.physchem.molecular_flash_separator import (
    MolecularFlashSeparatorSpec,
    evaluate_molecular_flash_separator,
)
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
class MolecularFlashSeparatorInventorySpec:
    separator: MolecularFlashSeparatorSpec
    vessel_ids: tuple[str, ...] = ("source", "reactor", "liquid_receiver")
    feed_rate_mol_s: float = 0.001
    tank_temperature_bounds_K: tuple[tuple[float, float], ...] = (
        (265.0, 295.0),
        (325.0, 360.0),
        (265.0, 295.0),
    )
    source_reserve_mol: float = 1e-5
    receiver_capacities_mol: tuple[float, float] = (0.1, 0.1)
    tank_capacities_L: tuple[float, ...] = (1.0, 1.0, 1.0)

    def __post_init__(self) -> None:
        for key in ("vessel_ids", "receiver_capacities_mol", "tank_capacities_L"):
            object.__setattr__(self, key, tuple(getattr(self, key)))
        object.__setattr__(
            self,
            "tank_temperature_bounds_K",
            tuple(tuple(b) for b in self.tank_temperature_bounds_K),
        )
        if (
            len(self.vessel_ids) != 3
            or len(set(self.vessel_ids)) != 3
            or any(not isinstance(k, str) or not k for k in self.vessel_ids)
            or len(self.tank_temperature_bounds_K) != 3
            or len(self.tank_capacities_L) != 3
            or len(self.receiver_capacities_mol) != 2
            or any(
                isinstance(v, bool) or not isfinite(v) or v <= 0
                for v in (
                    self.feed_rate_mol_s,
                    self.source_reserve_mol,
                    *self.receiver_capacities_mol,
                    *self.tank_capacities_L,
                )
            )
        ):
            raise ValueError("separator requires three finite owners and positive rates/capacities")
        for i in range(3):
            self.domain(i)

    @property
    def properties(self) -> MolecularCaloricModel:
        return self.separator.model

    def phase_id(self, index: int) -> str:
        return (self.separator.feed_phase, "gas", "liquid")[index]

    def domain(self, index: int) -> MolecularProcessDomain:
        return MolecularProcessDomain(
            self.properties,
            self.tank_temperature_bounds_K[index],
            self.separator.basin(self.phase_id(index)),
        )

    def validate_amounts(self, amounts: Mapping[str, float]) -> float:
        return validate_molecular_amounts(self.properties, amounts)

    def at_tp(
        self, index: int, amounts: Mapping[str, float], temperature: float
    ) -> MolecularCaloricState:
        return molecular_tank_tp(
            self.domain(index), self.separator.pressure_Pa, amounts, temperature
        )

    def at_ph(
        self, index: int, amounts: Mapping[str, float], enthalpy: float
    ) -> MolecularCaloricState:
        return molecular_tank_ph(
            self.domain(index),
            self.separator.pressure_Pa,
            amounts,
            enthalpy,
            self.separator.root_policy,
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
            "model_type": "molecular_flash_three_tank_v1",
            "separator": self.separator.to_dict(),
            **{
                k: getattr(self, k)
                for k in (
                    "vessel_ids",
                    "feed_rate_mol_s",
                    "tank_temperature_bounds_K",
                    "source_reserve_mol",
                    "receiver_capacities_mol",
                    "tank_capacities_L",
                )
            },
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> MolecularFlashSeparatorInventorySpec:
        if data.get("model_type") != "molecular_flash_three_tank_v1":
            raise ValueError("unknown molecular flash finite inventory encoding")
        return cls(
            MolecularFlashSeparatorSpec.from_dict(data["separator"]),
            **{
                k: data[k]
                for k in (
                    "vessel_ids",
                    "feed_rate_mol_s",
                    "tank_temperature_bounds_K",
                    "source_reserve_mol",
                    "receiver_capacities_mol",
                    "tank_capacities_L",
                )
            },
        )


def molecular_flash_separator_inventory(
    spec: MolecularFlashSeparatorInventorySpec,
    amounts: Sequence[Mapping[str, float]],
    temperatures_K: Sequence[float],
) -> TransportInventory:
    if len(amounts) != 3 or len(temperatures_K) != 3:
        raise ValueError("separator needs all three original tank states")
    return TransportInventory(
        tuple(dict(n) for n in amounts),
        tuple(
            sum(n.values()) * spec.at_tp(i, n, temperatures_K[i]).enthalpy_J_mol
            for i, n in enumerate(amounts)
        ),
    )


@dataclass(frozen=True)
class _ParticipatingTanks:
    spec: MolecularFlashSeparatorInventorySpec
    indices: tuple[int, ...]

    @property
    def properties(self) -> MolecularCaloricModel:
        return self.spec.properties

    def at_ph(
        self, index: int, amounts: Mapping[str, float], enthalpy: float
    ) -> MolecularCaloricState:
        return self.spec.at_ph(self.indices[index], amounts, enthalpy)

    def enthalpy_bounds(self, index: int, amounts: Mapping[str, float]) -> tuple[float, float]:
        return self.spec.enthalpy_bounds(self.indices[index], amounts)


class MolecularFlashSeparatorError(ValueError):
    def __init__(self, reason: str, diagnostics: dict[str, Any]) -> None:
        super().__init__(reason)
        self.diagnostics = diagnostics


def advance_molecular_flash_separator(
    spec: MolecularFlashSeparatorInventorySpec,
    state: TransportInventory,
    duration_s: float,
    *,
    progress: Callable[[float, int], None] | None = None,
) -> tuple[PhaseInventoryAdvance, dict[str, Any]]:
    if len(state.amounts_mol) != 3 or len(state.enthalpies_J) != 3:
        raise ValueError("separator needs three actual n/H inventories")
    before = tuple(
        spec.at_ph(i, n, h)
        for i, (n, h) in enumerate(zip(state.amounts_mol, state.enthalpies_J, strict=True))
    )
    inlet = MolecularStream(before[0], spec.feed_rate_mol_s)
    solved = evaluate_molecular_flash_separator(spec.separator, inlet)
    if not solved.success:
        raise MolecularFlashSeparatorError(
            "molecular flash separator failed: " + solved.failure_reason, solved.diagnostics
        )
    labels = solved.diagnostics["outlet_phase_labels"]
    indices = (0, *(1 if label == "gas" else 2 for label in labels))
    view = _ParticipatingTanks(spec, indices)
    initial = replace(
        state,
        amounts_mol=tuple(state.amounts_mol[i] for i in indices),
        enthalpies_J=tuple(state.enthalpies_J[i] for i in indices),
    )
    streams = (inlet, *solved.outlets)
    contribution = PhaseInventoryRates(
        np.array(
            [
                (-1 if j == 0 else 1) * s.molar_rate_mol_s * np.array(s.state.mole_fractions)
                for j, s in enumerate(streams)
            ]
        ),
        np.array([(-1 if j == 0 else 1) * s.enthalpy_rate_W for j, s in enumerate(streams)]),
        max(solved.heat_duty_W, 0.0),
        max(-solved.heat_duty_W, 0.0),
        (),
    )
    stops: list[tuple[str, Callable[[Any, Any], float]]] = [
        (
            f"source_reserve:{spec.vessel_ids[0]}",
            lambda s, _: sum(s.amounts_mol[0].values()) - spec.source_reserve_mol,
        )
    ]
    for j, i in enumerate(indices[1:], 1):

        def capacity_margin(s: Any, _: Any, j: int = j, i: int = i) -> float:
            return spec.receiver_capacities_mol[i - 1] - sum(s.amounts_mol[j].values())

        def volume_margin(s: Any, phases: Any, j: int = j, i: int = i) -> float:
            return (
                spec.tank_capacities_L[i]
                - 1000 * sum(s.amounts_mol[j].values()) / phases[j].molar_density_mol_m3
            )

        stops.append(
            (
                f"receiver_capacity:{spec.vessel_ids[i]}",
                capacity_margin,
            )
        )
        stops.append(
            (
                f"tank_volume:{spec.vessel_ids[i]}",
                volume_margin,
            )
        )
    advanced = integrate_phase_inventory(
        view,
        initial,
        duration_s=duration_s,
        rates=lambda _s, _p: contribution,
        closure_indices=tuple(range(1, len(indices))),
        stop_functions=stops,
        enthalpy_bounds=view.enthalpy_bounds,
        initial_boundary="stop",
        progress=progress,
        solver_id="molecular_flash_three_tank_bdf_v1",
    )

    def full(partial: TransportInventory) -> TransportInventory:
        ns, hs = list(state.amounts_mol), list(state.enthalpies_J)
        for j, i in enumerate(indices):
            ns[i], hs[i] = partial.amounts_mol[j], partial.enthalpies_J[j]
        return replace(partial, amounts_mol=tuple(ns), enthalpies_J=tuple(hs))

    result = replace(
        advanced,
        initial_state=state,
        final_state=full(advanced.final_state),
        trajectory=tuple(full(s) for s in advanced.trajectory),
    )
    final = result.final_state
    after = tuple(
        spec.at_ph(i, n, h)
        for i, (n, h) in enumerate(zip(final.amounts_mol, final.enthalpies_J, strict=True))
    )
    volumes = [
        tuple(
            sum(n.values()) / p.molar_density_mol_m3
            for n, p in zip(inv.amounts_mol, phases, strict=True)
        )
        for inv, phases in ((state, before), (final, after))
    ]
    if any(
        1000 * v > cap + 1e-7 for v, cap in zip(volumes[1], spec.tank_capacities_L, strict=True)
    ):
        raise ValueError("accepted separator state exceeds finite tank volume")
    work = spec.separator.pressure_Pa * (sum(volumes[1]) - sum(volumes[0]))
    heat = (
        final.heater_energy_J
        - state.heater_energy_J
        - final.condenser_energy_J
        + state.condenser_energy_J
    )
    du = sum(
        sum(n.values()) * p.internal_energy_J_mol
        for n, p in zip(final.amounts_mol, after, strict=True)
    ) - sum(
        sum(n.values()) * p.internal_energy_J_mol
        for n, p in zip(state.amounts_mol, before, strict=True)
    )
    return result, {
        "steady": solved.to_dict(),
        "participating_tank_indices": indices,
        "actual_duration_s": result.actual_duration_s,
        "heat_J": heat,
        "pressure_work_J": work,
        "internal_energy_change_J": du,
        "internal_energy_residual_J": du - heat + work,
        "shaft_work_J": 0.0,
        "initial_volumes_m3": volumes[0],
        "final_volumes_m3": volumes[1],
        "heater_energy_J": final.heater_energy_J - state.heater_energy_J,
        "condenser_energy_J": final.condenser_energy_J - state.condenser_energy_J,
    }
