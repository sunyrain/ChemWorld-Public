"""Common atom-molar Gibbs responses and constitution interface.

Phase-specific classes supply energy, composition and constraints. All material
consumers share the same caloric derivatives, constrained root and state.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import isclose, isfinite
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, TypeAlias

from chemworld.physchem.caloric_math import FirstDerivative, JetCoefficient
from chemworld.physchem.tdb_expressions import TdbScalar, scalar_value

if TYPE_CHECKING:
    from chemworld.physchem.calphad_response import CefDerivativeResponse, CefTemperatureResponse
    from chemworld.physchem.calphad_stationary import CefChemicalPotentials, CefConstitutionResult

SiteFractions: TypeAlias = Sequence[Mapping[str, float]]
JetSiteFractions: TypeAlias = Sequence[Mapping[str, TdbScalar]]


def gibbs_phase_from_dict(payload: Mapping[str, Any]) -> GibbsPhaseModel:
    from chemworld.physchem.calphad import CefPhaseModel
    from chemworld.physchem.mqmqa import MQMQAPhaseModel

    if not isinstance(payload, Mapping):
        raise ValueError("Gibbs phase model must be a declarative object")
    if payload.get("model_type") == "calphad_cef_v1":
        return CefPhaseModel.from_dict(payload)
    if payload.get("model_type") == "calphad_mqmqa_v1":
        return MQMQAPhaseModel.from_dict(payload)
    raise ValueError("unknown native Gibbs phase model type")


def _slope(value: JetCoefficient) -> JetCoefficient:
    return value.slope if isinstance(value, FirstDerivative) else 0.0



@dataclass(frozen=True)
class GibbsPhaseProperties:
    temperature_K: float
    pressure_Pa: float
    gibbs_J_mol: float
    enthalpy_J_mol: float
    entropy_J_mol_K: float
    pressure_derivative_m3_mol: float | None
    element_fractions: Mapping[str, float]
    contributions_J_mol: Mapping[str, float]
    diagnostics: tuple[str, ...]



class GibbsPhaseModel:
    """Shared native Gibbs interface; no phase-specific inventory or solver."""

    _diagnostics: tuple[str, ...]

    @property
    def constituents(self) -> tuple[tuple[str, ...], ...]:
        raise NotImplementedError("A phase model must declare its constituents")

    def _constraints(self, fractions: JetSiteFractions) -> tuple[TdbScalar, ...]:
        raise NotImplementedError("A phase model must define its constitution constraints")

    def _composition(
        self, fractions: JetSiteFractions,
    ) -> tuple[TdbScalar, Mapping[str, TdbScalar]]:
        raise NotImplementedError("A phase model must define its real-atom composition")

    def _energies(
        self, fractions: JetSiteFractions, temperature: TdbScalar, pressure: TdbScalar,
    ) -> Mapping[str, TdbScalar]:
        raise NotImplementedError("A phase model must define its energy contributions")

    def _response_diagnostics(
        self, fractions: SiteFractions, temperature: float, pressure: float,
    ) -> list[str]:
        return list(self._diagnostics)

    def _ordering_views(
        self, fractions: JetSiteFractions,
    ) -> tuple[GibbsPhaseModel, JetSiteFractions, JetSiteFractions] | None:
        return None

    def _fractions(
        self, fractions: SiteFractions, *, check_constraints: bool = True,
    ) -> tuple[dict[str, float], ...]:
        if not isinstance(fractions, (tuple, list)) or len(fractions) != len(self.constituents):
            raise ValueError("CEF needs one fraction mapping per declared sublattice")
        result = []
        for subl, expected in zip(fractions, self.constituents, strict=True):
            if not isinstance(subl, Mapping) or set(subl) != set(expected):
                raise ValueError("CEF site fractions must exactly match the active constituents")
            values = {s: scalar_value(subl[s]) for s in expected}
            if any(not isfinite(v) or v < 0 or v > 1 for v in values.values()):
                raise ValueError("CEF site fractions must be finite in [0,1]")
            if not isclose(sum(values.values()), 1, abs_tol=1e-10, rel_tol=0):
                raise ValueError("CEF site fractions must sum to one on every sublattice")
            result.append(values)
        normalized = tuple(result)
        constraints = self._constraints(normalized)
        if check_constraints and any(abs(scalar_value(c)) > 1e-9 for c in constraints):
            raise ValueError(
                "CEF site constitution violates charge neutrality or auxiliary balance")
        self._composition(normalized)
        return normalized


    def gibbs(
        self, fractions: SiteFractions, temperature_K: float, pressure_Pa: float = 101325.0,
    ) -> float:
        fractions = self._fractions(fractions)
        temperature, pressure = self._potentials(temperature_K, pressure_Pa)
        value = scalar_value(sum(self._energies(fractions, temperature, pressure).values()))
        if not isfinite(value):
            raise ValueError("nonfinite CEF Gibbs energy")
        return value


    @staticmethod
    def _potentials(temperature_K: float, pressure_Pa: float) -> tuple[float, float]:
        temperature, pressure = scalar_value(temperature_K), scalar_value(pressure_Pa)
        if not isfinite(temperature) or temperature <= 0 or not isfinite(pressure) or pressure <= 0:
            raise ValueError("CEF temperature and pressure must be finite and positive")
        return temperature, pressure


    def evaluate(
        self, fractions: SiteFractions, temperature_K: float, pressure_Pa: float = 101325.0,
        *, pressure_derivative: bool = True,
    ) -> GibbsPhaseProperties:
        fractions = self._fractions(fractions)
        temperature, pressure = self._potentials(temperature_K, pressure_Pa)
        energies = self._energies(fractions, FirstDerivative(temperature, 1), pressure)
        total = sum(energies.values())
        g, dg_dt = scalar_value(total), scalar_value(_slope(total))
        dg_dp = None
        if pressure_derivative:
            response = sum(self._energies(
                fractions, temperature, FirstDerivative(pressure, 1),
            ).values())
            dg_dp = scalar_value(_slope(response))
        if not all(isfinite(v) for v in (g, dg_dt, *(() if dg_dp is None else (dg_dp,)))):
            raise ValueError("nonfinite CEF energy or first derivative")
        diagnostics = self._response_diagnostics(fractions, temperature, pressure)
        if dg_dp == 0:
            diagnostics.append("zero pressure derivative is not a measured material volume")
        return GibbsPhaseProperties(
            temperature, pressure, g, g - temperature * dg_dt, -dg_dt, dg_dp,
            MappingProxyType({k: scalar_value(v)
                              for k, v in self._composition(fractions)[1].items()}),
            MappingProxyType({k: scalar_value(v) for k, v in energies.items()}), tuple(diagnostics),
        )


    def site_partials(
        self, fractions: SiteFractions, temperature_K: float, pressure_Pa: float = 101325.0,
    ) -> dict[tuple[int, str], float]:
        fractions = self._fractions(fractions)
        # Check thermodynamic domains before returning any partial result.
        self.gibbs(fractions, temperature_K, pressure_Pa)
        partials = {}
        for index, subl in enumerate(fractions):
            for species, value in subl.items():
                seeded: list[dict[str, TdbScalar]] = [dict(s) for s in fractions]
                seeded[index][species] = FirstDerivative(value, 1)
                energy = sum(self._energies(tuple(seeded), temperature_K, pressure_Pa).values())
                derivative = scalar_value(_slope(energy))
                if not isfinite(derivative):
                    raise ValueError("nonfinite CEF site partial")
                partials[index, species] = derivative
        return partials


    def temperature_response(
        self, fractions: SiteFractions, temperature_K: float, pressure_Pa: float = 101325.0,
        **kwargs: Any,
    ) -> CefTemperatureResponse:
        """G/H/S and Cp at fixed site fractions, rather than relaxed-site Cp."""
        from chemworld.physchem.calphad_response import temperature_response

        return temperature_response(self, fractions, temperature_K, pressure_Pa, **kwargs)


    def derivatives(
        self, fractions: SiteFractions, temperature_K: float, pressure_Pa: float = 101325.0,
        **kwargs: Any,
    ) -> CefDerivativeResponse:
        """Analytic T/P/site gradient and Hessian, with coordinate units."""
        from chemworld.physchem.calphad_response import derivatives

        return derivatives(self, fractions, temperature_K, pressure_Pa, **kwargs)


    def chemical_potentials(
        self, fractions: SiteFractions, temperature_K: float, pressure_Pa: float = 101325.0,
        **kwargs: Any,
    ) -> CefChemicalPotentials:
        """Determine elemental mu only when stationarity and rank permit it."""
        from chemworld.physchem.calphad_stationary import chemical_potentials

        return chemical_potentials(self, fractions, temperature_K, pressure_Pa, **kwargs)


    def stationary_constitution(
        self, element_fractions: Mapping[str, float], initial_sites: SiteFractions,
        temperature_K: float, pressure_Pa: float = 101325.0, **kwargs: Any,
    ) -> CefConstitutionResult:
        """A local constrained root; no global/multiphase minimum is promised."""
        from chemworld.physchem.calphad_stationary import stationary_constitution

        return stationary_constitution(self, element_fractions, initial_sites,
                                       temperature_K, pressure_Pa, **kwargs)
