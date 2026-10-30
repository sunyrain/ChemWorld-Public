"""Finite, closed, selected CEF phase on the common inventory/enthalpy interface.

Amounts are mol of nonvacant atoms. Mechanical volume is an authored condensed
phase property, not a volume inferred from a pressure-independent database.
"""

from dataclasses import dataclass, replace
from math import isfinite
from types import MappingProxyType

import numpy as np

from chemworld.physchem.elements import ELEMENTS
from chemworld.physchem.gibbs_phase import GibbsPhaseModel, gibbs_phase_from_dict
from chemworld.physchem.residual_system import (
    ResidualDomainError,
    ResidualVariable,
    RootPolicy,
    solve_root,
)


@dataclass(frozen=True)
class CefMaterialState:
    temperature_K: float
    amounts_mol: object
    constitution: object
    gibbs_J: float
    enthalpy_J: float
    heat_capacity_J_K: float
    support_deficit_J_mol: None = None  # No global multiphase certificate.


@dataclass(frozen=True)
class CefMaterialSpec:
    phase_model: GibbsPhaseModel
    initial_sites: tuple
    minimum_temperature_K: float
    maximum_temperature_K: float
    reference_temperature_K: float
    molar_volume_L_mol: float
    phase_type: str = "solid"
    pressure_Pa: float = 101325.0
    support_heat_capacity_J_K: float = 0.2

    def __post_init__(self) -> None:
        if not isinstance(self.phase_model, GibbsPhaseModel):
            raise ValueError("CEF material requires an explicit native phase model")
        if self.phase_type not in {"solid", "liquid"}:
            raise ValueError("CEF material needs an authored condensed phase type")
        if "G" in self.phase_model._phase.options:
            raise ValueError("a TDB gas requires a mechanical EOS, not condensed CEF volume")
        positive = (self.minimum_temperature_K, self.maximum_temperature_K,
                    self.reference_temperature_K, self.molar_volume_L_mol, self.pressure_Pa)
        if any(isinstance(v, bool) or not isfinite(v) or v <= 0 for v in positive):
            raise ValueError("CEF material requires positive finite caloric/geometric parameters")
        if not self.minimum_temperature_K < self.maximum_temperature_K:
            raise ValueError("CEF material needs a finite temperature interval")
        if (isinstance(self.support_heat_capacity_J_K, bool)
                or not isfinite(self.support_heat_capacity_J_K)
                or self.support_heat_capacity_J_K < 0):
            raise ValueError("CEF holder heat capacity must be finite and nonnegative")
        sites = self.phase_model._fractions(self.initial_sites, check_constraints=False)
        if any(y <= 0 for subl in sites for y in subl.values()):
            raise ValueError("CEF material requires an interior initial constitution")
        object.__setattr__(self, "initial_sites", tuple(MappingProxyType(s) for s in sites))
        # Validate that each counter has a real native elemental formula.
        _ = self.element_symbols

    @property
    def properties(self):
        return self

    @property
    def component_ids(self):
        return tuple(sorted(self.phase_model._composition(self.initial_sites)[1]))

    @property
    def element_symbols(self):
        symbols = {symbol.upper(): symbol for symbol in ELEMENTS}
        if set(self.component_ids) - symbols.keys():
            raise ValueError("CEF material contains a non-elemental inventory counter")
        return {element: symbols[element] for element in self.component_ids}

    def with_sites(self, sites):
        """A disposable closure view seeded by the sole committed constitution."""
        return replace(self, initial_sites=sites)

    def to_dict(self):
        return {"model_type": "calphad_material_v1",
                "phase_model": self.phase_model.to_dict(),
                "initial_sites": [dict(s) for s in self.initial_sites],
                **{name: getattr(self, name) for name in (
                    "minimum_temperature_K", "maximum_temperature_K",
                    "reference_temperature_K", "molar_volume_L_mol", "phase_type",
                    "pressure_Pa", "support_heat_capacity_J_K")}}

    @classmethod
    def from_dict(cls, payload):
        from chemworld.data.scientific_inputs import _keys

        _keys(payload, ("model_type", "phase_model", "initial_sites", "minimum_temperature_K",
                        "maximum_temperature_K", "reference_temperature_K",
                        "molar_volume_L_mol", "phase_type", "pressure_Pa",
                        "support_heat_capacity_J_K"))
        if payload["model_type"] != "calphad_material_v1":
            raise ValueError("unknown CEF material model type")
        return cls(**{**{k: v for k, v in payload.items() if k != "model_type"},
                      "phase_model": gibbs_phase_from_dict(payload["phase_model"])})

    def at_tp(self, amounts, temperature):
        if (set(amounts) != set(self.component_ids)
                or any(isinstance(v, bool) or not isfinite(v) or v <= 0
                       for v in amounts.values())
                or isinstance(temperature, bool) or not isfinite(temperature)
                or not self.minimum_temperature_K <= temperature <= self.maximum_temperature_K):
            raise ValueError("CEF material needs positive declared atom stocks within its T domain")
        total = sum(amounts.values())
        constitution = self.phase_model.stationary_constitution(
            {e: n/total for e, n in amounts.items()}, self.initial_sites,
            temperature, self.pressure_Pa,
        )
        if not constitution.success:
            raise ValueError(f"CEF material constitution failed: {constitution.failure_reason}")
        cp = constitution.relaxed_heat_capacity_J_mol_K
        if cp is None:
            raise ValueError("CEF material lacks a finite relaxed caloric derivative")
        capacity = total*cp + self.support_heat_capacity_J_K
        if not isfinite(capacity) or capacity <= 0:
            raise ValueError("CEF material has nonpositive total effective heat capacity")
        properties = constitution.properties
        enthalpy = total*properties.enthalpy_J_mol + self.support_heat_capacity_J_K*(
            temperature-self.reference_temperature_K)
        return CefMaterialState(float(temperature), MappingProxyType(dict(amounts)), constitution,
                                total*properties.gibbs_J_mol, enthalpy, capacity)

    def at_ph(self, index, amounts, enthalpy):
        if isinstance(enthalpy, bool) or not isfinite(enthalpy):
            raise ValueError("CEF material enthalpy must be finite")
        low, high = self.minimum_temperature_K, self.maximum_temperature_K
        cache = {}

        def close(temperature):
            t = float(temperature)
            if t not in cache:
                try:
                    cache[t] = self.at_tp(amounts, t)
                except (ValueError, ArithmeticError) as exc:
                    raise ResidualDomainError(str(exc)) from exc
            return cache[t]

        hlow, hhigh = close(low).enthalpy_J, close(high).enthalpy_J
        tolerance = 1e-8 + 1e-10*max(abs(hlow), abs(hhigh), abs(enthalpy))
        if hhigh <= hlow or not hlow-tolerance <= enthalpy <= hhigh+tolerance:
            raise ValueError("CEF enthalpy leaves the selected monotone caloric branch")
        target = min(max(enthalpy, hlow), hhigh)
        guess = low+(high-low)*(target-hlow)/(hhigh-hlow)
        root = solve_root(
            lambda values: np.array([close(values[0]).enthalpy_J-target]), [guess],
            jacobian=lambda values: np.array([[close(values[0]).heat_capacity_J_K]]),
            variables=(ResidualVariable("temperature", "K", scale=high-low,
                                        lower=low, upper=high),),
            residual_scales=np.array([max(1., abs(target), abs(hhigh-hlow))]),
            policy=RootPolicy(tolerance=1e-11, maximum_iterations=40),
        )
        if not root.success:
            raise ValueError(f"CEF enthalpy closure failed: {root.failure_reason}")
        final = close(root.value[0])
        if abs(final.enthalpy_J-enthalpy) > tolerance:
            raise ValueError("CEF caloric root failed its physical enthalpy residual")
        return final
