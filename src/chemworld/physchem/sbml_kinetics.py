"""Compile original SBML into native quantity coordinates and shared network rates.

The compiled object contains immutable configuration and initialization recipes,
never a live model clock or state. Callers own every evolving amount and value.
Unknown chemical compositions remain formal entities, not fictitious elements.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from types import MappingProxyType

import numpy as np

from chemworld.data.sbml import SBMLModel
from chemworld.data.sbml_parser import acyclic_order, walk_records
from chemworld.physchem.caloric_math import FirstDerivative, scalar_value
from chemworld.physchem.residual_system import linear_solve
from chemworld.physchem.stoichiometric_rates import stoichiometric_amount_rates


@dataclass(frozen=True)
class _Recipe:
    mode: str
    value: object
    volume: str | None = None
    scale: float = 1.
    local: tuple = ()

    @property
    def dependencies(self):
        if self.mode == "expression":
            return set(self.value.symbols)-dict(self.local).keys()-{"$time"}
        return {self.volume} if self.volume is not None else set()


@dataclass(frozen=True)
class SBMLRates:
    coordinate_rates: object
    reaction_extents_mol_s: object
    held_pool_reaction_rates_mol_s: object

    def __post_init__(self) -> None:
        for name in ("coordinate_rates", "reaction_extents_mol_s",
                     "held_pool_reaction_rates_mol_s"):
            object.__setattr__(self, name, MappingProxyType(dict(getattr(self, name))))


@dataclass(frozen=True)
class SBMLKinetics:
    model: SBMLModel
    initial_model_time: float = 0.
    controlled_parameters: tuple[str, ...] = ()
    integer_quantities: bool = False
    _initial: object = field(init=False, repr=False, compare=False)
    _initial_values: object = field(init=False, repr=False, compare=False)
    _recipes: object = field(init=False, repr=False, compare=False)
    _order: tuple = field(init=False, repr=False, compare=False)
    _functions: object = field(init=False, repr=False, compare=False)
    _species: object = field(init=False, repr=False, compare=False)
    _compartments: object = field(init=False, repr=False, compare=False)
    _parameters: object = field(init=False, repr=False, compare=False)
    _assigned: object = field(init=False, repr=False, compare=False)
    _rate_rules: object = field(init=False, repr=False, compare=False)
    _amount_scale: object = field(init=False, repr=False, compare=False)
    _volume_scale: object = field(init=False, repr=False, compare=False)
    _stoichiometries: tuple = field(init=False, repr=False, compare=False)
    _reaction_ids: tuple = field(init=False, repr=False, compare=False)
    _time_scale: float = field(init=False, repr=False, compare=False)
    _extent_scale: float | None = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if type(self.integer_quantities) is not bool:
            raise ValueError("integer source quantity views require an explicit Boolean")
        if not isinstance(self.model, SBMLModel):
            raise ValueError("SBML kinetics requires validated original SBMLModel input")
        if (isinstance(self.initial_model_time, bool)
                or not isinstance(self.initial_model_time, (int, float))
                or not isfinite(self.initial_model_time)):
            raise ValueError("initial SBML model time must be finite and explicit")
        root = self.model.declaration
        species = {r.get("id"): r for r in root.items("listOfSpecies")}
        compartments = {r.get("id"): r for r in root.items("listOfCompartments")}
        parameters = {r.get("id"): r for r in root.items("listOfParameters")}
        functions = {r.get("id"): r.mathematics for r in root.items("listOfFunctionDefinitions")}
        assigned = {r.get("variable"): r.mathematics for r in root.items("listOfRules")
                    if r.kind == "assignmentRule"}
        rate_rules = {r.get("variable"): r.mathematics for r in root.items("listOfRules")
                      if r.kind == "rateRule"}
        if (not isinstance(self.controlled_parameters, (tuple, list))
                or any(not isinstance(name, str) for name in self.controlled_parameters)
                or len(set(self.controlled_parameters)) != len(self.controlled_parameters)
                or set(self.controlled_parameters)-parameters.keys()
                or set(self.controlled_parameters) & assigned.keys()):
            raise ValueError("external controls must name unique non-assigned global parameters")
        object.__setattr__(self, "controlled_parameters", tuple(self.controlled_parameters))
        amount_scale, volume_scale = {}, {}
        for name, record in species.items():
            units = (self.model.unit_signature(record.get("substanceUnits"))
                     if record.get("substanceUnits") is not None else
                     self.model.model_unit("substanceUnits"))
            amount_scale[name] = units.amount_mol(1.)
        for name, record in compartments.items():
            if record.get("spatialDimensions", 3.) != 3.:
                raise ValueError("native SBML volume coordinates require 3D compartments")
            units = (self.model.unit_signature(record.get("units"))
                     if record.get("units") is not None else self.model.model_unit("volumeUnits"))
            volume_scale[name] = units.volume_l(1.)
        for attr, value in (("_species", species), ("_compartments", compartments),
                ("_parameters", parameters), ("_functions", functions), ("_assigned", assigned),
                ("_rate_rules", rate_rules), ("_amount_scale", amount_scale),
                ("_volume_scale", volume_scale)):
            object.__setattr__(self, attr, MappingProxyType(value))
        object.__setattr__(self, "_time_scale", self.model.model_unit("timeUnits").time_s(1.))
        reactions = tuple((r.get("id", f"@reaction:{i}"), r)
                          for i, r in enumerate(root.items("listOfReactions")))
        object.__setattr__(self, "_reaction_ids", tuple(name for name, _ in reactions))
        extent_scale = self.model.model_unit("extentUnits").amount_mol(1.) if reactions else None
        object.__setattr__(self, "_extent_scale", extent_scale)
        initial = {name: _Recipe("literal", record.get("value"))
                   for name, record in parameters.items()}
        initial.update({name: _Recipe("literal", r.get("size"))
                        for name, r in compartments.items()})
        for name, record in species.items():
            amount, concentration = record.get("initialAmount"), record.get("initialConcentration")
            only = record.get("hasOnlySubstanceUnits")
            if amount is not None:
                initial[name] = (_Recipe("literal", amount) if only else
                                 _Recipe("divide_volume", amount, record.get("compartment")))
            elif concentration is not None:
                initial[name] = (_Recipe("multiply_volume", concentration,
                                        record.get("compartment"))
                                 if only else _Recipe("literal", concentration))
            else:
                initial[name] = _Recipe("literal", None)
        for record in walk_records(root):
            if record.kind == "speciesReference" and record.get("id") is not None:
                initial[record.get("id")] = _Recipe("literal", record.get("stoichiometry"))
        for name, reaction in reactions:
            law = reaction.of("kineticLaw")[0]
            local = tuple((p.get("id"), p.get("value")) for p in law.items("listOfLocalParameters"))
            initial[name] = _Recipe("expression", law.mathematics, local=local)
        initial.update({r.get("symbol"): _Recipe("expression", r.mathematics)
                        for r in root.items("listOfInitialAssignments")})
        initial.update({name: _Recipe("expression", expression)
                        for name, expression in assigned.items()})
        order = acyclic_order({n: r.dependencies for n, r in initial.items()},
                              label="initial basis")
        initial_scope = self._evaluate(initial, order, {}, 0.)
        object.__setattr__(self, "_initial", MappingProxyType(initial_scope))
        values = {}
        for name, record in species.items():
            if name not in assigned:
                values["N:"+name] = initial_scope[name]*amount_scale[name]*(
                    1. if record.get("hasOnlySubstanceUnits") else
                    initial_scope[record.get("compartment")])
        values.update({"V:"+n: initial_scope[n]*volume_scale[n]
                       for n in compartments if n not in assigned})
        values.update({"P:"+n: initial_scope[n] for n, record in parameters.items()
                       if (not record.get("constant") or n in self.controlled_parameters)
                       and n not in assigned})
        object.__setattr__(self, "_initial_values", MappingProxyType(values))
        recipes = {}
        for name in parameters:
            recipes[name] = (_Recipe("state", "P:"+name) if "P:"+name in values else
                             _Recipe("literal", initial_scope[name]))
        for name in compartments:
            recipes[name] = _Recipe("state", "V:"+name, scale=volume_scale[name])
        for name, record in species.items():
            recipes[name] = _Recipe("state" if record.get("hasOnlySubstanceUnits") else
                "concentration", "N:"+name, None if record.get("hasOnlySubstanceUnits") else
                record.get("compartment"), amount_scale[name])
        for record in walk_records(root):
            if record.kind == "speciesReference" and record.get("id") is not None:
                recipes[record.get("id")] = _Recipe("literal", initial_scope[record.get("id")])
        for name, _reaction in reactions:
            recipes[name] = initial[name]
        recipes.update({name: _Recipe("expression", expression)
                        for name, expression in assigned.items()})
        order = acyclic_order({n: r.dependencies for n, r in recipes.items()}, label="source basis")
        object.__setattr__(self, "_recipes", MappingProxyType(recipes))
        object.__setattr__(self, "_order", order)
        coefficients = []
        for name, reaction in reactions:
            row = {}
            for group, sign in (("listOfReactants", -1.), ("listOfProducts", 1.)):
                for reference in reaction.items(group):
                    s = reference.get("species")
                    record = species[s]
                    if record.get("constant") and not record.get("boundaryCondition"):
                        raise ValueError("constant non-boundary SBML species cannot react")
                    coefficient = (initial_scope[reference.get("id")] if reference.get("id")
                                   is not None else reference.get("stoichiometry"))
                    if coefficient is None or not isfinite(coefficient) or coefficient < 0:
                        raise ValueError("SBML reaction needs finite initialized stoichiometry")
                    row[s] = row.get(s, 0.)+sign*coefficient
            coefficients.append((name, MappingProxyType(row)))
        object.__setattr__(self, "_stoichiometries", tuple(coefficients))
        self.validate_values(values, 0.)

    def _evaluate(self, recipes, order, values, time_s, targets=None):
        needed = set(order) if targets is None else set(targets)
        pending = list(needed)
        while pending:
            for name in recipes[pending.pop()].dependencies:
                if name in recipes and name not in needed:
                    needed.add(name)
                    pending.append(name)
        context = {"$time": self.initial_model_time+time_s/self._time_scale}
        for name in order:
            if name not in needed:
                continue
            recipe = recipes[name]
            mode, value, volume, scale = recipe.mode, recipe.value, recipe.volume, recipe.scale
            if mode == "expression":
                result = value.evaluate({**context, **dict(recipe.local)},
                                        functions=self._functions)
            elif mode == "state":
                result = self._source_coordinate(values, value, scale)
            elif mode == "concentration":
                result = self._source_coordinate(values, value, scale)/context[volume]
            elif mode == "divide_volume":
                result = value/context[volume]
            elif mode == "multiply_volume":
                result = value*context[volume]
            else:
                result = value
            if result is None or not isfinite(scalar_value(result)):
                raise ValueError("uninitialized or nonfinite SBML quantity "+name)
            if name in self._compartments and scalar_value(result) <= 0:
                raise ValueError("native SBML compartment volume must stay positive")
            context[name] = result
        return context

    def _source_coordinate(self, values, key, scale):
        if self.integer_quantities and key.startswith("N:"):
            from chemworld.foundation.discrete_inventory import AVOGADRO, mol_to_counts

            count = mol_to_counts({key: values[key]})[key]
            quantum = AVOGADRO*scale
            # Canonical item amounts are the exact original integer view. A
            # float mol/display round trip must not falsify A+B==N guards.
            return count if scale == 1./AVOGADRO else count/quantum
        return values[key]/scale

    @property
    def initial_values(self):
        """An initialization template, not the evolving state."""
        return self._initial_values

    def context(self, values, time_s, *, targets=None):
        if set(values) != set(self._initial_values):
            raise ValueError("SBML candidate coordinates differ from their declared basis")
        return self._evaluate(self._recipes, self._order, values, time_s, targets)

    def quantities(self, values, time_s):
        context = self.context(values, time_s)
        volumes = {n: context[n]*self._volume_scale[n] for n in self._compartments}
        amounts = {n: context[n]*self._amount_scale[n]*(1. if r.get("hasOnlySubstanceUnits") else
                   context[r.get("compartment")]) for n, r in self._species.items()}
        return amounts, volumes

    def controlled_values(self, values, time_s, assignments):
        """Explicit external parameter inputs; source constant-event rules stay intact."""
        if not assignments or set(assignments)-set(self.controlled_parameters):
            raise ValueError("parameter input is not a declared external model control")
        candidate = dict(values)
        for name, value in assignments.items():
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not isfinite(value)):
                raise ValueError("external model input must be a finite source-unit scalar")
            candidate["P:"+name] = float(value)
        self.validate_values(candidate, time_s)
        return candidate

    def assigned_values(self, values, time_s, assignments):
        """Convert source targets on one pre-execution volume view, then jump.

        Source RHS values may have been captured earlier. The implicit
        concentration-to-amount conversion uses the current pre-jump volume,
        including when that same event also assigns the compartment. See
        official SBML semantic cases 01779/01780 and delayed case 01508.
        Unassigned species retain their amounts; assignment views are then
        recomputed and the complete candidate is validated once.
        """
        for name, value in assignments.items():
            record = (self._species.get(name) or self._compartments.get(name)
                      or self._parameters.get(name))
            if (record is None or record.get("constant") or name in self._assigned
                    or isinstance(value, bool) or not isinstance(value, (int, float))
                    or not isfinite(value)):
                raise ValueError("invalid finite SBML event assignment target/value "+name)
        context = self.context(values, time_s)
        proposed = dict(values)
        for name, value in assignments.items():
            if name in self._species:
                record = self._species[name]
                proposed["N:"+name] = value*self._amount_scale[name]*(
                    1. if record.get("hasOnlySubstanceUnits") else
                    context[record.get("compartment")])
            elif name in self._compartments:
                proposed["V:"+name] = value*self._volume_scale[name]
            else:
                proposed["P:"+name] = value
        self.validate_values(proposed, time_s)
        return proposed

    def validate_values(self, values, time_s):
        if any(not isfinite(scalar_value(v)) for v in values.values()):
            raise ValueError("SBML candidate values must be finite")
        amounts, volumes = self.quantities(values, time_s)
        if any(scalar_value(v) <= 0 for v in volumes.values()):
            raise ValueError("native SBML compartment volume must stay positive")
        if any(scalar_value(n) < 0 for n in amounts.values()):
            raise ValueError("native SBML entity amount cannot be negative")
        context = self.context(values, time_s)
        for constraint in self.model.declaration.items("listOfConstraints"):
            if not bool(scalar_value(constraint.mathematics.evaluate(context,
                                                                     functions=self._functions))):
                raise ValueError("SBML model constraint failed")

    def rates(self, values, time_s):
        context = self.context(values, time_s)
        source_rates = {n: context[n] for n in self._reaction_ids}
        source_amount_rates = stoichiometric_amount_rates(self._species,
            self._stoichiometries, source_rates)
        rates, held = {}, {}
        conversion = self.model.declaration.get("conversionFactor")
        for name, record in self._species.items():
            coefficient = record.get("conversionFactor", conversion)
            rate = source_amount_rates[name]*(context[coefficient] if coefficient else 1.)
            rate *= self._amount_scale[name]/self._time_scale
            if record.get("boundaryCondition") or record.get("constant"):
                held[name] = rate
                rate = 0.
            if "N:"+name in values:
                rates["N:"+name] = rate
        for name in self._parameters:
            if "P:"+name in values:
                rates["P:"+name] = 0.
        for name in self._compartments:
            if "V:"+name in values:
                rates["V:"+name] = 0.
        unknown = []
        for name, expression in self._rate_rules.items():
            rate = expression.evaluate(context, functions=self._functions)/self._time_scale
            if name in self._species:
                if self._species[name].get("hasOnlySubstanceUnits"):
                    rates["N:"+name] = rate*self._amount_scale[name]
                else:
                    unknown.append((name, rate))
                    rates["N:"+name] = 0.
            elif name in self._compartments:
                rates["V:"+name] = rate*self._volume_scale[name]
            else:
                rates["P:"+name] = rate
        if unknown:
            if any(isinstance(v, FirstDerivative) for v in (*values.values(), time_s)):
                raise ValueError("full RHS jet through a concentration-rule solve is not declared")
            keys = tuple(values)
            seeds = np.eye(len(keys)+1)
            jets = {key: FirstDerivative(values[key], seeds[i]) for i, key in enumerate(keys)}
            scope = self.context(jets, FirstDerivative(time_s, seeds[-1]),
                                 targets=tuple(name for name, _ in unknown))
            indices = [keys.index("N:"+name) for name, _ in unknown]
            matrix, rhs = [], []
            known = np.asarray([rates[k] for k in keys]+[1.], dtype=float)
            for name, target in unknown:
                gradient = (scope[name].slope if isinstance(scope[name], FirstDerivative) else
                            np.zeros(len(keys)+1))
                matrix.append(gradient[indices])
                rhs.append(float(target)-float(gradient@known))
            answer = linear_solve(np.asarray(matrix), np.asarray(rhs))
            for (name, _), rate in zip(unknown, answer, strict=True):
                rates["N:"+name] = float(rate)
        if set(rates) != set(values):
            raise ValueError("SBML rate output does not cover exactly its active coordinates")
        return SBMLRates(rates, {name: r*self._extent_scale/self._time_scale
                                for name, r in source_rates.items()}, held)


def sbml_continuous_system(frame, *, events=(), coordinate_scales=None, policy=None,
                           numerical_form="ode"):
    """Build one continuous segment on the shared solver, not a full simulator.

    The hybrid coordinator supplies its stop surfaces and applies every source
    event. This low-level construction does not execute or silently omit them.
    """
    from chemworld.physchem.coupled_system import (
        DEFAULT_COUPLED_ODE_POLICY,
        CoupledCoordinate,
        CoupledProcess,
        CoupledSystem,
        KineticOnlyThermodynamics,
        ProcessContribution,
    )

    if not isinstance(frame, SBMLKinetics):
        raise ValueError("continuous SBML segment requires its compiled quantity basis")
    scales = {} if coordinate_scales is None else dict(coordinate_scales)
    if set(scales)-frame.initial_values.keys():
        raise ValueError("unknown SBML numerical coordinate scale")
    positive_amounts = [v for k, v in frame.initial_values.items() if k.startswith("N:") and v > 0]
    amount_scale = max(positive_amounts, default=min(frame._amount_scale.values(), default=1.))
    coordinates = []
    for name, value in frame.initial_values.items():
        if name.startswith("N:"):
            unit, scale, minimum = "mol", max(value, amount_scale), 0.
        elif name.startswith("V:"):
            unit, scale, minimum = "L", value, 0.
        else:
            unit = "SBML:"+frame._parameters[name[2:]].get("units", "unspecified")
            scale, minimum = max(abs(value), 1.), None
        coordinates.append(CoupledCoordinate(name, unit, scales.get(name, scale), minimum))

    def evaluate(state):
        return ProcessContribution(coordinate_rates=frame.rates(
            state.coordinates, state.time_s).coordinate_rates)

    def validate(state):
        frame.validate_values(state.coordinates, state.time_s)

    return CoupledSystem(KineticOnlyThermodynamics(),
        processes=(CoupledProcess("sbml_continuous_equations", evaluate, validate_state=validate),),
        coordinates=tuple(coordinates), events=tuple(events), isothermal=True,
        policy=DEFAULT_COUPLED_ODE_POLICY if policy is None else policy,
        numerical_form=numerical_form)
