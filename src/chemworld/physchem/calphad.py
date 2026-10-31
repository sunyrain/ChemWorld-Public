"""Native compound-energy phases over declarative TDB parameters.

This layer evaluates a specified phase constitution. It owns neither a world
inventory nor a multiphase equilibrium loop. Site partials are not chemical
potentials. All first derivatives use the existing caloric chain arithmetic.
G/H and elemental potentials use J per mole of nonvacant atoms; S/Cp use the
same atom-molar basis per kelvin. Vacancies and electrons add no real atoms.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable, Mapping
from dataclasses import dataclass, field
from itertools import permutations, product
from math import isclose, prod
from typing import Any, cast, overload

import numpy as np

from chemworld.data.tdb import AuxiliaryBalances, TdbDatabase, TdbFunction, TdbParameter, TdbPhase
from chemworld.physchem.caloric_math import FirstDerivative, JetCoefficient, exponential, logarithm
from chemworld.physchem.calphad_extrapolation import qkt_weight, validate_qkt
from chemworld.physchem.gibbs_phase import (
    GibbsPhaseModel,
    GibbsPhaseProperties,
    JetSiteFractions,
    SiteFractions,
    _slope,
)
from chemworld.physchem.second_order_pair import SecondOrderPair
from chemworld.physchem.tdb_expressions import TDB_GAS_CONSTANT_J_MOL_K as R
from chemworld.physchem.tdb_expressions import TdbScalar, scalar_value

CefPhaseProperties = GibbsPhaseProperties

_GIBBS_PARAMETERS = frozenset({
    "G", "L", "TC", "NT", "BMAGN", "GD", "THETA", "V0", "VA", "VK", "VC", "QKT",
})


def _zero_jet(value: JetCoefficient) -> bool:
    """Require every scalar coefficient to be zero, including nested directions."""
    if isinstance(value, FirstDerivative):
        return _zero_jet(value.value) and _zero_jet(value.slope)
    if isinstance(value, SecondOrderPair):
        return all(_zero_jet(getattr(value, name))
                   for name in ("value", "dx", "dy", "dxx", "dxy", "dyy"))
    if isinstance(value, np.ndarray):
        return all(_zero_jet(coefficient) for coefficient in value.flat)
    return bool(value == 0)


def _xlogx(value: TdbScalar) -> TdbScalar:
    if scalar_value(value) == 0:
        if not _zero_jet(value):
            raise ValueError("CEF site derivative diverges at zero site fraction")
        return 0.0
    return value * logarithm(value)


def _softplus(value: TdbScalar) -> TdbScalar:
    return (value + logarithm(1 + exponential(-value)) if scalar_value(value) > 0
            else logarithm(1 + exponential(value)))


def _einstein_energy(log_theta: TdbScalar, temperature: TdbScalar) -> TdbScalar:
    theta = exponential(log_theta)
    x = theta / temperature
    if scalar_value(x) <= 0:
        raise ValueError("CEF Einstein temperature must be positive")
    if scalar_value(x) < .01:
        # log(1-exp(-x)) with its small-x expansion avoids subtractive loss.
        log_population = logarithm(x) - x / 2 + x**2 / 24 - x**4 / 2880 + x**6 / 181440
    else:
        log_population = logarithm(1 - exponential(-x))
    return 1.5 * R * theta + 3 * R * temperature * log_population


def _magnetic_energy(
    temperature: TdbScalar, tc: TdbScalar, nt: TdbScalar, moment: TdbScalar,
    *, afm: float, structure: float,
) -> TdbScalar:
    if afm != 0:
        beta = moment / afm if scalar_value(moment) <= 0 else moment
        critical = tc / afm if scalar_value(tc) <= 0 else tc
        if scalar_value(critical) == 0:
            return 0.0
        tau = temperature / critical
        a = 518 / 1125 + 11692 / 15975 * (1 / structure - 1)
        if scalar_value(tau) < 1:
            shape = 1 - ((79 / (140 * structure)) / tau
                         + (474 / 497) * (1 / structure - 1)
                         * (tau**3 / 6 + tau**9 / 135 + tau**15 / 600)) / a
        else:
            shape = -(tau**-5 / 10 + tau**-15 / 315 + tau**-25 / 1500) / a
    else:
        beta = moment
        d = .33471979 + .49649686 * (1 / structure - 1)

        def xiong(critical: TdbScalar) -> TdbScalar:
            if scalar_value(critical) <= 0:
                return 0.0
            tau = temperature / critical
            if scalar_value(tau) <= 1:
                return 1 - ((.38438376 / structure) / tau
                            + .63570895 * (1 / structure - 1)
                            * (tau**3 / 6 + tau**9 / 135 + tau**15 / 600)
                            + tau**21 / 1617) / d
            return -(tau**-7 / 21 + tau**-21 / 630
                     + tau**-35 / 2975 + tau**-49 / 8232) / d

        shape = xiong(tc) + xiong(nt)
    if scalar_value(beta) <= -1:
        raise ValueError("CEF magnetic moment requires beta > -1")
    return R * temperature * logarithm(beta + 1) * shape


@dataclass(frozen=True)
class _Parameter:
    source: TdbParameter
    constituents: tuple[tuple[str, ...], ...]

    @property
    def kind(self) -> str:
        return self.source.kind

    @property
    def order(self) -> int:
        return self.source.order


def _symmetry_arrays(
    array: tuple[tuple[str, ...], ...], option: str,
) -> list[tuple[tuple[str, ...], ...]]:
    if len(array) < 4:
        raise ValueError("CEF four-sublattice symmetry needs at least four sublattices")
    indices: Iterable[tuple[int, ...]]
    if option == "F":
        indices = permutations(range(4))
    else:
        indices = (order for a in ((0, 1), (1, 0)) for b in ((2, 3), (3, 2))
                   for order in ((*a, *b), (*b, *a)))
    return sorted({tuple(array[i] for i in order) + array[4:] for order in indices})


@dataclass(frozen=True)
class CefPhaseModel(GibbsPhaseModel):
    database: TdbDatabase
    phase_name: str
    components: tuple[str, ...] | None = None
    extrapolation: str = "reject"
    unspecified_endmembers: str = "reject"
    activate_suppressed: bool = False
    _phase: TdbPhase = field(init=False, repr=False)
    _parameters: tuple[_Parameter, ...] = field(init=False, repr=False)
    _substitutional: tuple[int, ...] = field(init=False, repr=False, default=())
    _interstitial: tuple[int, ...] = field(init=False, repr=False, default=())
    _diagnostics: tuple[str, ...] = field(init=False, repr=False, default=())

    def __post_init__(self) -> None:
        if not isinstance(self.database, TdbDatabase):
            raise ValueError("CEF requires a validated native TdbDatabase")
        if type(self.activate_suppressed) is not bool:
            raise ValueError("source suppression activation must be an explicit boolean")
        if (not isinstance(self.extrapolation, str)
                or self.extrapolation not in {"reject", "extend"}):
            raise ValueError("CEF extrapolation must be reject or extend")
        if (not isinstance(self.unspecified_endmembers, str)
                or self.unspecified_endmembers not in {"reject", "zero"}):
            raise ValueError("CEF unspecified_endmembers must be reject or explicit zero")
        if not isinstance(self.phase_name, str):
            raise ValueError("CEF phase name must be text")
        name = self.phase_name.upper()
        if name not in self.database.phases:
            raise ValueError(f"unknown CEF phase {name}")
        object.__setattr__(self, "phase_name", name)
        if self.components is None:
            eligible = set(self.database.species)
        else:
            if (not isinstance(self.components, (list, tuple))
                    or any(not isinstance(c, str) for c in self.components)):
                raise ValueError("CEF components must be a list of declared names")
            components = tuple(c.upper() for c in self.components)
            if not components or set(components) - self.database.species.keys():
                raise ValueError("CEF components must name declared species or elements")
            elements = {e for c in components for e, _ in self.database.species[c].constituents}
            eligible = {s.name for s in self.database.species.values()
                        if {e for e, _ in s.constituents} <= elements}
            object.__setattr__(self, "components", components)
        original = self.database.phases[name]
        if original.model_hints.get("unsupported_physics"):
            raise ValueError("unconsumed ChemSage physical declarations: "+"; ".join(
                cast(tuple[str, ...], original.model_hints["unsupported_physics"])))
        if original.model_hints.get("phase_family") in {"SUBQ", "SUBG"}:
            raise ValueError("MQMQA requires its declared quadruplet model")
        if original.model_hints.get("chemical_group_overrides"):
            raise ValueError("ChemSage group overrides require their explicitly supported model")
        if not self.activate_suppressed and (
            original.model_hints.get("initial_status", "entered") != "entered"
            or original.model_hints.get("suppressed_endmembers")
        ):
            raise ValueError("source suppression requires explicit activation before physical use")
        phase = self._active_phase(original, eligible)
        definitions = {d.character: d for d in self.database.type_definitions}
        type_diagnostics = []
        for character in phase.type_characters:
            definition = definitions.get(character)
            if definition is None:
                type_diagnostics.append(
                    f"unassigned TDB type marker {character}: no model effect is declared",
                )
            elif definition.feature == "unsupported":
                raise ValueError(f"CEF has an unsupported type definition {character}")
        if "Y" in phase.options:
            if any(self.database.species[s].charge <= 0 for s in phase.constituents[0]):
                raise ValueError("CEF ionic cation sublattice requires positively charged species")
            if any(self.database.species[s].charge > 0 for s in phase.constituents[1]):
                raise ValueError(
                    "CEF ionic anion sublattice requires anions, vacancies or neutrals",
                )
        if name == phase.model_hints.get("ordered_phase"):
            if "Y" in phase.options:
                raise ValueError("CEF partitioned ionic ordering has no declared model")
            phase = self._extend_ordered(phase, eligible)
        object.__setattr__(self, "_phase", phase)
        parameters = self._select_parameters()
        object.__setattr__(self, "_parameters", parameters)
        for parameter in parameters:
            if parameter.kind == "QKT":
                validate_qkt(parameter, phase)
        self._check_endmembers()
        unused = sorted({p.kind for p in parameters} - _GIBBS_PARAMETERS)
        diagnostics = [f"parameter type {kind} retained but not a Gibbs contribution"
                       for kind in unused]
        if self.unspecified_endmembers == "zero":
            diagnostics.append(
                "unspecified endmember energies explicitly use the CEF zero convention",
            )
        if ({p.kind for p in parameters} & {"TC", "NT", "BMAGN"}
                and "magnetic_structure_factor" not in phase.model_hints):
            diagnostics.append("magnetic parameters have no declared magnetic contribution")
        if {p.kind for p in parameters} & {"VK", "VC"}:
            diagnostics.append("nonzero VK/VC only permit reference-pressure G/H/S")
        object.__setattr__(self, "_diagnostics", (*type_diagnostics, *diagnostics))

    def _active_phase(self, phase: TdbPhase, eligible: Collection[str]) -> TdbPhase:
        array = tuple(tuple(s for s in subl if s in eligible) for subl in phase.constituents)
        if any(not subl for subl in array):
            raise ValueError(f"CEF phase {phase.name} has an inactive sublattice")
        if not any(self.database.species[s].number_of_atoms for subl in array for s in subl):
            raise ValueError("CEF phase cannot consist only of vacancies/electrons")
        return TdbPhase(phase.name, phase.site_ratios, array, phase.options,
                        phase.type_characters, phase.model_hints)

    def _extend_ordered(self, phase: TdbPhase, eligible: Collection[str]) -> TdbPhase:
        disordered = self._active_phase(
            self.database.phases[cast(str, phase.model_hints["disordered_phase"])], eligible,
        )
        remaining = set(range(len(phase.constituents)))
        interstitial = []
        for constituents in disordered.constituents[1:]:
            matches = [i for i in sorted(remaining) if phase.constituents[i] == constituents]
            if len(matches) != 1:
                raise ValueError("CEF ordered/disordered interstitial sublattices are ambiguous")
            interstitial.append(matches[0])
            remaining.remove(matches[0])
        substitutional = tuple(sorted(remaining))
        if not substitutional:
            raise ValueError("CEF ordering needs a substitutional sublattice")
        sub_sites = sum(phase.site_ratios[i] for i in substitutional)
        for index, ordered_index in enumerate(interstitial, 1):
            if not isclose(phase.site_ratios[ordered_index] / sub_sites,
                           disordered.site_ratios[index] / disordered.site_ratios[0],
                           rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError("CEF ordered/disordered site-ratio normalization differs")
        array = tuple(tuple(sorted(set(subl) | set(disordered.constituents[0])))
                      if i in substitutional else subl for i, subl in enumerate(phase.constituents))
        object.__setattr__(self, "_substitutional", substitutional)
        object.__setattr__(self, "_interstitial", tuple(interstitial))
        return TdbPhase(phase.name, phase.site_ratios, array, phase.options,
                        phase.type_characters, phase.model_hints)

    def _select_parameters(self) -> tuple[_Parameter, ...]:
        phase = self._phase
        selected = []
        for parameter in self.database.parameters:
            if parameter.phase != self.phase_name:
                continue
            array = parameter.constituents
            if "Y" in phase.options and len(array) == 1 and "VA" in array[0]:
                raise ValueError("CEF short ionic parameters require real neutral species, not VA")
            targets = (phase.constituents[1],) if len(array) == 1 and "Y" in phase.options \
                else phase.constituents
            if not all(subl == ("*",) or set(subl) <= set(target)
                       for subl, target in zip(array, targets, strict=True)):
                continue
            mixing = [len(subl) for subl in array if len(subl) > 1]
            if parameter.kind in _GIBBS_PARAMETERS:
                if not mixing and (parameter.order != 0 or parameter.kind == "L"):
                    raise ValueError("CEF pure parameters use order zero and G, not L")
                if ((len(mixing) > 1 and max(mixing) > 2)
                        or (len(mixing) > 2 and parameter.order > 0)
                        or (len(mixing) == 2 and parameter.order > 2)
                        or (3 in mixing and parameter.order > 2)
                        or (any(m > 3 for m in mixing) and parameter.order > 0)):
                    raise ValueError("CEF interaction order has no declared R-K/Muggianu meaning")
            selected.append(_Parameter(parameter, array))
        explicit = {(p.kind, p.constituents, p.order, p.source.diffusing_species) for p in selected}
        generated: dict[tuple[str, tuple[tuple[str, ...], ...], int, str | None], _Parameter] = {}
        if phase.name == phase.model_hints.get("ordered_phase"):
            for option in sorted(phase.options & {"F", "B"}):
                for selected_parameter in selected:
                    for array in _symmetry_arrays(selected_parameter.constituents, option):
                        key = (selected_parameter.kind, array, selected_parameter.order,
                               selected_parameter.source.diffusing_species)
                        if key not in explicit:
                            previous = generated.get(key)
                            if (previous and previous.source.function
                                    != selected_parameter.source.function):
                                raise ValueError("contradictory CEF symmetry-generated parameters")
                            generated[key] = _Parameter(selected_parameter.source, array)
        return (*selected, *generated.values())

    def _check_endmembers(self) -> None:
        if self.unspecified_endmembers == "zero":
            return
        if prod(map(len, self.constituents)) > 100000:
            raise ValueError(
                "CEF endmember coverage is too large; declare the zero convention explicitly",
            )
        pure = [p for p in self._parameters if p.kind == "G"
                and all(len(subl) == 1 for subl in p.constituents)]
        for endmember in product(*self.constituents):
            if not any(all(subl == ("*",) or subl == (species,)
                           for subl, species in zip(p.constituents,
                               endmember[1:] if len(p.constituents) == 1
                               and len(endmember) == 2 else endmember, strict=True))
                       for p in pure):
                raise ValueError(f"CEF endmember energy is unspecified: {endmember}")

    @property
    def constituents(self) -> tuple[tuple[str, ...], ...]:
        return self._phase.constituents

    @property
    def phase_options(self) -> frozenset[str]:
        return self._phase.options

    @property
    def generated_parameter_count(self) -> int:
        return sum(p.constituents != p.source.constituents for p in self._parameters)

    def to_dict(self) -> dict[str, Any]:
        return {"model_type": "calphad_cef_v1", "database": self.database.to_dict(),
                "phase_name": self.phase_name, "components": self.components,
                "extrapolation": self.extrapolation,
                "unspecified_endmembers": self.unspecified_endmembers,
                **({"activate_suppressed": True} if self.activate_suppressed else {})}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> CefPhaseModel:
        from chemworld.data.calphad_databases import calphad_database_from_dict
        from chemworld.data.scientific_inputs import _keys

        _keys(payload, ("model_type", "database", "phase_name", "components",
                        "extrapolation", "unspecified_endmembers"),
              optional=("activate_suppressed",))
        if payload["model_type"] != "calphad_cef_v1":
            raise ValueError("unknown CEF model type")
        return cls(calphad_database_from_dict(payload["database"]), payload["phase_name"],
                   payload["components"], payload["extrapolation"],
                   payload["unspecified_endmembers"], payload.get("activate_suppressed", False))


    def _ratios(self, fractions: JetSiteFractions) -> tuple[TdbScalar, ...]:
        if "Y" not in self._phase.options:
            return self._phase.site_ratios
        cation_charge = sum(y * self.database.species[s].charge for s, y in fractions[0].items())
        anion_charge = sum(y * abs(self.database.species[s].charge)
                           for s, y in fractions[1].items())
        return anion_charge + cation_charge * fractions[1].get("VA", 0), cation_charge

    def _constraints(self, fractions: JetSiteFractions) -> tuple[TdbScalar, ...]:
        ratios = self._ratios(fractions)
        charge = sum(ratio * y * self.database.species[s].charge
                     for ratio, subl in zip(ratios, fractions, strict=True)
                     for s, y in subl.items())
        if "Y" in self._phase.options:
            # Ionic vacancies have the variable negative valence of the
            # cation sublattice. Their bookkeeping charge is not a real atom.
            charge -= ratios[1]**2 * fractions[1].get("VA", 0)
        auxiliary = tuple(sum(coefficient*prod(
            fractions[i].get(species[0], 0.) for i, species in enumerate(array))
            for array, coefficient in terms)
            for _, terms in cast(AuxiliaryBalances,
                                 self._phase.model_hints.get("auxiliary_balances", ())))
        return (*(sum(subl.values()) - 1 for subl in fractions), charge, *auxiliary)

    @property
    def constraint_labels(self) -> tuple[str, ...]:
        return (*(f"site_sum:{i}" for i in range(len(self.constituents))), "charge",
                *(f"auxiliary:{name}" for name, _ in cast(
                    AuxiliaryBalances, self._phase.model_hints.get("auxiliary_balances", ()))))

    def _composition(
        self, fractions: JetSiteFractions,
    ) -> tuple[TdbScalar, dict[str, TdbScalar]]:
        amounts: dict[str, TdbScalar] = {}
        for ratio, subl in zip(self._ratios(fractions), fractions, strict=True):
            for name, y in subl.items():
                for element, count in self.database.species[name].constituents:
                    if element not in {"VA", "/-"}:
                        amounts[element] = amounts.get(element, 0) + ratio * y * count
        total = sum(amounts.values())
        if scalar_value(total) <= 0:
            raise ValueError("CEF atom-molar normalization must be positive")
        return total, {element: amount / total for element, amount in amounts.items()}

    def _weight(self, parameter: _Parameter, fractions: JetSiteFractions) -> TdbScalar:
        if parameter.kind == "QKT":
            return qkt_weight(parameter, fractions, self._phase)
        array = parameter.constituents
        ionic = "Y" in self._phase.options
        targets = (1,) if ionic and len(array) == 1 else tuple(range(len(array)))
        mixing = [i for i, subl in enumerate(array) if len(subl) > 1]
        no_anions = ionic and not any(
            self.database.species[s].charge < 0
            for index, subl in zip(targets, array, strict=True)
            for s in (fractions[index] if subl == ("*",) else subl)
        )
        vacancy_endmember = any("VA" in subl for subl in array) and not mixing
        weight: TdbScalar = 1.0
        for index, subl in zip(targets, array, strict=True):
            if subl == ("*",):
                values = [v * fractions[1].get("VA", 0)
                          if no_anions and not vacancy_endmember
                          and self.database.species[s].charge > 0 else v
                          for s, v in fractions[index].items()]
                weight *= sum(values)
                continue
            names = list(subl)
            if ionic:
                names.sort(key=lambda s: (0 if self.database.species[s].charge else
                                         1 if s == "VA" else 2, s))
            values = [fractions[index][s] for s in names]
            if no_anions and not vacancy_endmember:
                values = [v * fractions[1].get("VA", 0)
                          if self.database.species[s].charge > 0 else v
                          for s, v in zip(names, values, strict=True)]
            weight *= prod(values)
            order = parameter.order
            if len(values) == 2 and order > 0:
                if len(mixing) == 1 or index == mixing[-order]:
                    weight *= (values[0] - values[1])**(order if len(mixing) == 1 else 1)
            elif len(values) == 3:
                others = [p for p in self._parameters if p.kind == parameter.kind
                          and p.constituents == array]
                if not (order == 0 and len(others) == 1):
                    weight *= values[order] + (1 - sum(values)) / 3
        if no_anions:
            weight *= self._ratios(fractions)[1]
        return weight

    @overload
    def _sum(
        self, kind: Collection[str], fractions: SiteFractions, temperature: float, pressure: float,
        *, pure: bool | None = None,
    ) -> float: ...

    @overload
    def _sum(
        self, kind: Collection[str], fractions: JetSiteFractions,
        temperature: TdbScalar, pressure: TdbScalar, *, pure: bool | None = None,
    ) -> TdbScalar: ...

    def _sum(
        self, kind: Collection[str], fractions: JetSiteFractions,
        temperature: TdbScalar, pressure: TdbScalar, *, pure: bool | None = None,
    ) -> TdbScalar:
        result: TdbScalar = 0.0
        for parameter in self._parameters:
            if parameter.kind not in kind:
                continue
            is_pure = all(len(subl) == 1 for subl in parameter.constituents)
            if pure is not None and pure != is_pure:
                continue
            coefficient = self.database.evaluate_parameter(
                parameter.source, temperature, pressure, extrapolation=self.extrapolation,
            )
            result += self._weight(parameter, fractions) * coefficient
        return result

    def _core(
        self, fractions: JetSiteFractions, temperature: TdbScalar, pressure: TdbScalar,
    ) -> dict[str, TdbScalar]:
        normalization, _ = self._composition(fractions)
        ratios = self._ratios(fractions)
        has = {p.kind for p in self._parameters}
        contributions: dict[str, TdbScalar] = {
            "reference": self._sum({"G"}, fractions, temperature, pressure, pure=True),
            "ideal": R * temperature * sum(
                a * _xlogx(y) for a, subl in zip(ratios, fractions, strict=True)
                for y in subl.values()),
            "excess": self._sum({"G", "L", "QKT"}, fractions, temperature, pressure, pure=False),
            "magnetic": 0.0, "two_state": 0.0, "einstein": 0.0,
        }
        hints = self._phase.model_hints
        if "magnetic_structure_factor" in hints:
            tc = self._sum({"TC"}, fractions, temperature, pressure)
            nt = self._sum({"NT"}, fractions, temperature, pressure)
            moment = self._sum({"BMAGN"}, fractions, temperature, pressure)
            contributions["magnetic"] = _magnetic_energy(
                temperature, tc, nt, moment, afm=cast(float, hints["magnetic_afm_factor"]),
                structure=cast(float, hints["magnetic_structure_factor"]),
            )
        if "GD" in has:
            gd = self._sum({"GD"}, fractions, temperature, pressure)
            contributions["two_state"] = -R * temperature * _softplus(-gd / (R * temperature))
        if "THETA" in has:
            contributions["einstein"] = _einstein_energy(
                self._sum({"THETA"}, fractions, temperature, pressure), temperature,
            )
        contributions = {key: value / normalization for key, value in contributions.items()}
        high_pressure = [self._sum({kind}, fractions, temperature, pressure)
                         for kind in ("VK", "VC")]
        if any(scalar_value(value) != 0 for value in high_pressure):
            if scalar_value(pressure) != 101325 or scalar_value(_slope(pressure)) != 0:
                raise ValueError("CEF high-pressure VK/VC closure is not implemented")
            contributions["volume"] = 0.0
        else:
            v0 = self._sum({"V0"}, fractions, temperature, pressure) / normalization
            va = self._sum({"VA"}, fractions, temperature, pressure)
            contributions["volume"] = v0 * exponential(va) * (pressure - 101325)
        return contributions

    def _response_diagnostics(
        self, fractions: SiteFractions, temperature: float, pressure: float,
    ) -> list[str]:
        diagnostics = list(self._diagnostics)
        hints = self._phase.model_hints
        if "magnetic_structure_factor" in hints:
            tc = self._sum({"TC"}, fractions, temperature, pressure)
            afm = cast(float, hints["magnetic_afm_factor"])
            criticals = ([tc / afm if tc < 0 else tc] if afm != 0 else
                         [tc, self._sum({"NT"}, fractions, temperature, pressure)])
            if any(c > 0 and isclose(temperature, c, rel_tol=1e-12) for c in criticals):
                diagnostics.append("magnetic critical point: declared one-sided branch derivative")
        return diagnostics

    def _derivative_branches(
        self, fractions: SiteFractions, temperature: float, pressure: float,
        *, include_sites: bool, include_pressure: bool,
    ) -> tuple[str, ...]:
        from chemworld.physchem.calphad_response import (
            analytic_response,
            site_labels,
            temperature_segment_diagnostics,
            unpack_sites,
        )

        local_models = [(self, fractions)]
        views = self._ordering_views(fractions)
        if views is not None:
            disordered, disordered_fractions, randomized = views
            local_models.append((disordered, disordered_fractions))
            if not self._phase.model_hints.get("never_disorder"):
                local_models.append((self, randomized))
        diagnostics: list[str] = []
        curves: list[TdbFunction] = []
        for phase, sites in local_models:
            curves.extend(p.source.function for p in phase._parameters)
            hints = phase._phase.model_hints
            if "magnetic_structure_factor" in hints:
                tc = phase._sum({"TC"}, sites, temperature, pressure)
                afm = cast(float, hints["magnetic_afm_factor"])
                criticals = ([tc / afm if tc < 0 else tc] if afm != 0 else
                             [tc, phase._sum({"NT"}, sites, temperature, pressure)])
                if any(c > 0 and isclose(temperature, c, rel_tol=1e-12) for c in criticals):
                    diagnostics.append(f"magnetic critical point: {phase.phase_name}")
                moment = phase._sum({"BMAGN"}, sites, temperature, pressure)
                if afm != 0 and moment == 0 and tc != 0:
                    n_potentials = 2 if include_pressure else 1
                    coordinates: tuple[float, ...] = (
                        (temperature, pressure) if include_pressure else (temperature,))
                    if include_sites:
                        coordinates += tuple(sites[i][s] for i, s in site_labels(phase))

                    def raw_moment(
                        z: tuple[TdbScalar, ...], phase: CefPhaseModel = phase,
                        sites: SiteFractions = sites, n_potentials: int = n_potentials,
                    ) -> tuple[TdbScalar]:
                        y = unpack_sites(phase, z[n_potentials:]) if include_sites else sites
                        p = z[1] if include_pressure else pressure
                        return (phase._sum({"BMAGN"}, y, z[0], p),)

                    _, first, second = analytic_response(raw_moment, coordinates)
                    if np.any(first != 0) or np.any(second != 0):
                        diagnostics.append(f"magnetic parameter sign boundary: {phase.phase_name}")
        diagnostics.extend(temperature_segment_diagnostics(
            curves, self.database.functions, temperature,
        ))
        return tuple(dict.fromkeys(diagnostics))

    @overload
    def _ordering_views(
        self, fractions: SiteFractions,
    ) -> tuple[CefPhaseModel, SiteFractions, SiteFractions] | None: ...

    @overload
    def _ordering_views(
        self, fractions: JetSiteFractions,
    ) -> tuple[CefPhaseModel, JetSiteFractions, JetSiteFractions] | None: ...

    def _ordering_views(
        self, fractions: JetSiteFractions,
    ) -> tuple[CefPhaseModel, JetSiteFractions, JetSiteFractions] | None:
        if not self._substitutional:
            return None
        indices = self._substitutional
        ratios = self._phase.site_ratios
        quasi = {s: sum(ratios[i] * fractions[i][s] for i in indices)
                 / sum(ratios[i] for i in indices) for s in self.constituents[indices[0]]}
        disordered_fractions = (quasi, *(fractions[i] for i in self._interstitial))
        disordered = CefPhaseModel(
            self.database, cast(str, self._phase.model_hints["disordered_phase"]), self.components,
            self.extrapolation, self.unspecified_endmembers,
        )
        randomized = tuple(quasi if i in indices else subl
                           for i, subl in enumerate(fractions))
        return disordered, disordered_fractions, randomized

    def _energies(
        self, fractions: JetSiteFractions, temperature: TdbScalar, pressure: TdbScalar,
    ) -> dict[str, TdbScalar]:
        raw = self._core(fractions, temperature, pressure)
        views = self._ordering_views(fractions)
        if views is None:
            return {**raw, "ordering": 0.0}
        disordered, disordered_fractions, randomized = views
        base = disordered._core(disordered_fractions, temperature, pressure)
        if self._phase.model_hints.get("never_disorder"):
            base["ideal"] = 0.0
            ordering = sum(raw.values())
        else:
            randomized_energy = sum(self._core(randomized, temperature, pressure).values())
            ordering = sum(raw.values()) - randomized_energy
        return {**base, "ordering": ordering}
