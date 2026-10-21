"""Common atom-molar Gibbs responses and constitution interface.

Phase-specific classes supply energy, composition and constraints. All material
consumers share the same caloric derivatives, constrained root and state.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from math import isclose, isfinite
from types import MappingProxyType

from chemworld.physchem.caloric_math import FirstDerivative
from chemworld.physchem.tdb_expressions import scalar_value


def gibbs_phase_from_dict(payload):
    from chemworld.physchem.calphad import CefPhaseModel
    from chemworld.physchem.mqmqa import MQMQAPhaseModel

    if not isinstance(payload, Mapping):
        raise ValueError("Gibbs phase model must be a declarative object")
    if payload.get("model_type") == "calphad_cef_v1":
        return CefPhaseModel.from_dict(payload)
    if payload.get("model_type") == "calphad_mqmqa_v1":
        return MQMQAPhaseModel.from_dict(payload)
    raise ValueError("unknown native Gibbs phase model type")


def _slope(value):
    return value.slope if isinstance(value, FirstDerivative) else 0.0



@dataclass(frozen=True)
class GibbsPhaseProperties:
    temperature_K: float
    pressure_Pa: float
    gibbs_J_mol: float
    enthalpy_J_mol: float
    entropy_J_mol_K: float
    pressure_derivative_m3_mol: float | None
    element_fractions: object
    contributions_J_mol: object
    diagnostics: tuple[str, ...]



class GibbsPhaseModel:
    """Shared native Gibbs interface; no phase-specific inventory or solver."""

    def _ordering_views(self, fractions):
        return None

    def _fractions(self, fractions, *, check_constraints=True):
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
        result = tuple(result)
        constraints = self._constraints(result)
        if check_constraints and any(abs(scalar_value(c)) > 1e-9 for c in constraints):
            raise ValueError(
                "CEF site constitution violates charge neutrality or auxiliary balance")
        self._composition(result)
        return result


    def gibbs(self, fractions, temperature_K, pressure_Pa=101325.0):
        fractions = self._fractions(fractions)
        temperature, pressure = self._potentials(temperature_K, pressure_Pa)
        value = scalar_value(sum(self._energies(fractions, temperature, pressure).values()))
        if not isfinite(value):
            raise ValueError("nonfinite CEF Gibbs energy")
        return value


    @staticmethod
    def _potentials(temperature_K, pressure_Pa):
        temperature, pressure = scalar_value(temperature_K), scalar_value(pressure_Pa)
        if not isfinite(temperature) or temperature <= 0 or not isfinite(pressure) or pressure <= 0:
            raise ValueError("CEF temperature and pressure must be finite and positive")
        return temperature, pressure


    def evaluate(self, fractions, temperature_K, pressure_Pa=101325.0, *, pressure_derivative=True):
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
        diagnostics = list(self._diagnostics)
        hints = self._phase.model_hints
        if "magnetic_structure_factor" in hints:
            tc = self._sum({"TC"}, fractions, temperature, pressure)
            afm = hints["magnetic_afm_factor"]
            criticals = ([tc / afm if tc < 0 else tc] if afm != 0 else
                         [tc, self._sum({"NT"}, fractions, temperature, pressure)])
            if any(c > 0 and isclose(temperature, c, rel_tol=1e-12) for c in criticals):
                diagnostics.append("magnetic critical point: declared one-sided branch derivative")
        if dg_dp == 0:
            diagnostics.append("zero pressure derivative is not a measured material volume")
        return GibbsPhaseProperties(
            temperature, pressure, g, g - temperature * dg_dt, -dg_dt, dg_dp,
            MappingProxyType({k: scalar_value(v)
                              for k, v in self._composition(fractions)[1].items()}),
            MappingProxyType({k: scalar_value(v) for k, v in energies.items()}), tuple(diagnostics),
        )


    def site_partials(self, fractions, temperature_K, pressure_Pa=101325.0):
        fractions = self._fractions(fractions)
        # Check thermodynamic domains before returning any partial result.
        self.gibbs(fractions, temperature_K, pressure_Pa)
        partials = {}
        for index, subl in enumerate(fractions):
            for species, value in subl.items():
                seeded = [dict(s) for s in fractions]
                seeded[index][species] = FirstDerivative(value, 1)
                energy = sum(self._energies(tuple(seeded), temperature_K, pressure_Pa).values())
                derivative = scalar_value(_slope(energy))
                if not isfinite(derivative):
                    raise ValueError("nonfinite CEF site partial")
                partials[index, species] = derivative
        return partials


    def temperature_response(self, fractions, temperature_K, pressure_Pa=101325.0, **kwargs):
        """G/H/S and Cp at fixed site fractions, rather than relaxed-site Cp."""
        from chemworld.physchem.calphad_response import temperature_response

        return temperature_response(self, fractions, temperature_K, pressure_Pa, **kwargs)


    def derivatives(self, fractions, temperature_K, pressure_Pa=101325.0, **kwargs):
        """Analytic T/P/site gradient and Hessian, with coordinate units."""
        from chemworld.physchem.calphad_response import derivatives

        return derivatives(self, fractions, temperature_K, pressure_Pa, **kwargs)


    def chemical_potentials(self, fractions, temperature_K, pressure_Pa=101325.0, **kwargs):
        """Determine elemental mu only when stationarity and rank permit it."""
        from chemworld.physchem.calphad_stationary import chemical_potentials

        return chemical_potentials(self, fractions, temperature_K, pressure_Pa, **kwargs)


    def stationary_constitution(self, element_fractions, initial_sites, temperature_K,
                                pressure_Pa=101325.0, **kwargs):
        """A local constrained root; no global/multiphase minimum is promised."""
        from chemworld.physchem.calphad_stationary import stationary_constitution

        return stationary_constitution(self, element_fractions, initial_sites,
                                       temperature_K, pressure_Pa, **kwargs)
