"""Elementary integer channels compiled from original SBML quantity semantics.

Formal mol coordinates remain the inventory. Counts are a transient exact view;
no formula, temperature, compartment or model clock is synthesized here.
"""

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from hashlib import sha256
from math import factorial, isfinite, prod
from typing import Any, cast

from chemworld.data.sbml_parser import SBMLRecord
from chemworld.foundation.discrete_inventory import AVOGADRO, counts_to_mol, mol_to_counts
from chemworld.physchem.mathml import MathExpression, MathNode
from chemworld.physchem.sbml_kinetics import SBMLKinetics
from chemworld.physchem.stochastic_kinetics import (
    EventChannel,
    EventProcessSpec,
    StochasticAdvance,
    advance_event_channels,
    initial_continuation,
)

ALGORITHMS = frozenset(
    {"gillespie-direct-v1", "gibson-bruck-next-reaction-v1", "adaptive-explicit-tau-leap-v1"}
)
MEMORY_FORMAT = "chemworld_sbml_integer_continuation_v1"
PROCESS_FIELDS = frozenset(
    {"algorithm", "time_s", "rng_state", "pending", "signature", "events_committed"}
)


def _positive_integer(value: float) -> int:
    if (
        isinstance(value, bool)
        or not isfinite(value)
        or value <= 0
        or not float(value).is_integer()
    ):
        raise ValueError("SBML elementary stoichiometry must resolve to positive integer molecules")
    return int(value)


def _monomial(
    node: MathNode,
    scalars: Mapping[str, float],
    species: set[str],
    functions: Mapping[str, MathExpression],
    bindings: dict[str, tuple[float, dict[str, int]]] | None = None,
) -> tuple[float, dict[str, int]]:
    """Prove a source expression is one monomial; never infer orders by probes."""
    bindings = {} if bindings is None else bindings
    op, args = node.operation, node.arguments
    if op in {"number", "constant"}:
        if (
            not isinstance(node.value, (int, float))
            or isinstance(node.value, bool)
            or not isfinite(node.value)
        ):
            raise ValueError("elementary coefficients require finite real literals")
        return float(node.value), {}
    if op == "symbol":
        if not isinstance(node.value, str):
            raise ValueError("elementary law symbol must be an identifier")
        if node.value in bindings:
            return bindings[node.value]
        if node.value in scalars:
            return scalars[node.value], {}
        if node.value in species:
            return 1.0, {node.value: 1}
        raise ValueError("elementary source law has an unsupported evolving symbol: " + node.value)
    if op == "call":
        if not isinstance(node.value, str):
            raise ValueError("elementary function must name a source function")
        function = functions[node.value]._root
        inputs = [_monomial(a, scalars, species, functions, bindings) for a in args]
        return _monomial(
            function.arguments[0],
            {},
            set(),
            functions,
            dict(zip(function.bindings, inputs, strict=True)),
        )
    terms = [_monomial(a, scalars, species, functions, bindings) for a in args]
    if op == "times":
        orders: dict[str, int] = {}
        for _, powers in terms:
            for name, power in powers.items():
                orders[name] = orders.get(name, 0) + power
        return prod(value for value, _ in terms), orders
    if op == "divide" and not terms[1][1]:
        return terms[0][0] / terms[1][0], terms[0][1]
    if op == "power" and not terms[1][1]:
        value, orders = terms[0]
        exponent = terms[1][0]
        if orders and (not float(exponent).is_integer() or not 0 <= exponent <= 3):
            raise ValueError("elementary species powers must be integers from zero to three")
        return value**exponent, {
            name: int(power * exponent) for name, power in orders.items() if power * exponent
        }
    if op in {"plus", "minus"} and terms and all(t[1] == terms[0][1] for t in terms):
        values = [value for value, _ in terms]
        value = (
            sum(values)
            if op == "plus"
            else -values[0]
            if len(values) == 1
            else values[0] - values[1]
        )
        return value, terms[0][1]
    raise ValueError("SBML stochastic laws require an explicit elementary monomial")


@dataclass(frozen=True)
class SBMLIntegerKinetics:
    frame: SBMLKinetics
    _source: str = field(init=False, repr=False)
    _rules: object = field(init=False, repr=False, default=None)

    def __post_init__(self) -> None:
        root = self.frame.model.declaration
        if any(root.items(name) for name in ("listOfRules", "listOfEvents", "listOfConstraints")):
            from chemworld.physchem.sbml_integer_rules import SBMLIntegerRules

            object.__setattr__(self, "_rules", SBMLIntegerRules(self.frame))
        if not self.frame._species:
            raise ValueError("SBML integer mode needs declared source species")
        if any(
            r.get("conversionFactor", root.get("conversionFactor"))
            in self.frame.controlled_parameters
            for r in self.species.values()
        ):
            raise ValueError("controlled conversion factors cannot change molecular stoichiometry")
        source = json.dumps(self.frame.model.to_dict(), sort_keys=True, allow_nan=False)
        object.__setattr__(self, "_source", sha256(source.encode()).hexdigest())
        self.compile(self.frame.initial_values, 0.0)

    @property
    def species(self) -> Mapping[str, SBMLRecord]:
        # Legacy SBMLKinetics stores frozen declaration maps under object-typed
        # fields. Narrow that validated input at this adapter boundary.
        return cast(Mapping[str, SBMLRecord], self.frame._species)

    @property
    def species_ids(self) -> tuple[str, ...]:
        return tuple(self.species)

    @property
    def rules(self) -> Any:
        return self._rules

    @property
    def requires_boundaries(self) -> bool:
        return self._rules is not None

    def counts(self, coordinates: Mapping[str, float], time_s: float) -> dict[str, int]:
        self.frame.validate_values(coordinates, time_s)
        controls = {"P:" + p for p in self.frame.controlled_parameters}
        if self.rules is not None:
            controls.update(self.rules.mutable_coordinates)
        fixed = {
            k: v
            for k, v in self.frame.initial_values.items()
            if not k.startswith("N:") and k not in controls
        }
        if any(coordinates[k] != value for k, value in fixed.items()):
            raise ValueError("SBML integer geometry and uncontrolled inputs must remain fixed")
        # N: coordinates are the original mol inventory. Avoid a concentration
        # display round trip when testing exact molecular representability.
        return mol_to_counts({name: coordinates["N:" + name] for name in self.species_ids})

    def compile(self, coordinates: Mapping[str, float], time_s: float) -> tuple[EventChannel, ...]:
        self.counts(coordinates, time_s)
        frame, root = self.frame, self.frame.model.declaration
        scope = frame.context(coordinates, time_s)
        scalar_names = (
            *cast(Mapping[str, SBMLRecord], frame._parameters),
            *cast(Mapping[str, SBMLRecord], frame._compartments),
        )
        scalars = {name: scope[name] for name in scalar_names}
        channels = []
        reactions = zip(frame._reaction_ids, root.items("listOfReactions"), strict=True)
        for identity, reaction in reactions:
            extent_scale = frame._extent_scale
            if extent_scale is None:
                raise ValueError("SBML integer reactions require declared extent units")
            if reaction.get("reversible") or reaction.get("fast", False):
                raise ValueError("SBML integer channels require irreversible, non-fast reactions")
            reactants: dict[str, int] = {}
            changes: dict[str, int] = {}
            for group, sign in (("listOfReactants", -1), ("listOfProducts", 1)):
                for reference in reaction.items(group):
                    if not reference.get("constant"):
                        raise ValueError("SBML elementary stoichiometry must be constant")
                    name = reference.get("species")
                    record = self.species[name]
                    conversion = record.get("conversionFactor", root.get("conversionFactor"))
                    factor = scope[conversion] if conversion else 1.0
                    source_n = (
                        scope[reference.get("id")]
                        if reference.get("id")
                        else reference.get("stoichiometry")
                    )
                    if source_n is None:
                        raise ValueError("SBML elementary stoichiometry must be initialized")
                    n = _positive_integer(
                        source_n
                        * cast(Mapping[str, float], frame._amount_scale)[name]
                        * factor
                        / extent_scale
                    )
                    if sign < 0:
                        reactants[name] = reactants.get(name, 0) + n
                    if not (record.get("constant") or record.get("boundaryCondition")):
                        changes[name] = changes.get(name, 0) + sign * n
            law = reaction.of("kineticLaw")[0]
            local = {p.get("id"): p.get("value") for p in law.items("listOfLocalParameters")}
            node = law.mathematics._root
            if self.rules is not None:
                node = self.rules.expand(node, shadow=frozenset(local))
            coefficient, orders = _monomial(
                node,
                {**scalars, **local},
                set(self.species_ids),
                cast(Mapping[str, MathExpression], frame._functions),
            )
            modifiers = {r.get("species") for r in reaction.items("listOfModifiers")}
            if (
                any(orders.get(name) != n for name, n in reactants.items())
                or set(orders) - reactants.keys() - modifiers
                or sum(orders.values()) > 3
            ):
                raise ValueError("SBML elementary powers must match full reactants/modifiers")
            constant = coefficient * extent_scale * AVOGADRO / frame._time_scale
            for name, order in orders.items():
                record = self.species[name]
                quantum = 1.0 / (AVOGADRO * cast(Mapping[str, float], frame._amount_scale)[name])
                if not record.get("hasOnlySubstanceUnits"):
                    quantum /= scope[record.get("compartment")]
                constant *= quantum**order * factorial(order)
            if isinstance(constant, complex) or not isfinite(constant) or constant < 0:
                raise ValueError("SBML elementary coefficient must be finite and nonnegative")
            channels.append(
                EventChannel(
                    identity,
                    tuple(sorted((k, v) for k, v in changes.items() if v)),
                    tuple(sorted(orders.items())),
                    constant,
                )
            )
        return tuple(channels)

    def policy(
        self, coordinates: Mapping[str, float], method: str, event_budget: int, tau_policy: Any
    ) -> EventProcessSpec:
        fixed = {key: value for key, value in coordinates.items() if not key.startswith("N:")}
        context = json.dumps([self._source, fixed], sort_keys=True, allow_nan=False)
        return EventProcessSpec(method, event_budget, context, tau_policy)

    def advance(
        self,
        coordinates: Mapping[str, float],
        memory: dict[str, Any],
        time_s: float,
        duration_s: float,
        method: str,
        event_budget: int,
        tau_policy: Any,
        *, stop_after_boundary: bool = False,
        critical_reaction_ids: frozenset[str] = frozenset(),
        calendar_deadline_s: float | None = None,
    ) -> tuple[dict[str, float], dict[str, Any], StochasticAdvance]:
        if self.requires_boundaries and duration_s > 0 and not stop_after_boundary:
            raise ValueError("source rule/event profiles require original boundary coordination")
        if (
            not isinstance(memory, dict)
            or set(memory) != {"format", "process"}
            or memory["format"] != MEMORY_FORMAT
            or not isinstance(memory["process"], dict)
            or set(memory["process"]) != PROCESS_FIELDS
            or memory["process"]["time_s"] != time_s
        ):
            raise ValueError("SBML integer continuation must match the original ledger clock")
        process = memory["process"]
        rng_state, pending = process["rng_state"], process["pending"]
        if (
            type(process["time_s"]) not in {int, float}
            or not isfinite(process["time_s"])
            or process["time_s"] < 0
            or process["algorithm"] != method
            or type(process["events_committed"]) is not int
            or process["events_committed"] < 0
            or not isinstance(rng_state, dict)
            or set(rng_state) != {"bit_generator", "state", "has_uint32", "uinteger"}
            or rng_state["bit_generator"] != "PCG64"
            or not isinstance(rng_state["state"], dict)
            or set(rng_state["state"]) != {"state", "inc"}
            or any(
                type(rng_state["state"][key]) is not int
                or not 0 <= rng_state["state"][key] < 2**128
                for key in ("state", "inc")
            )
            or rng_state["state"]["inc"] % 2 != 1
            or type(rng_state["has_uint32"]) is not int
            or rng_state["has_uint32"] not in {0, 1}
            or type(rng_state["uinteger"]) is not int
            or not 0 <= rng_state["uinteger"] < 2**32
        ):
            raise ValueError("SBML integer continuation has invalid process fields or RNG")
        if pending is not None:
            fields = (
                {"time_s", "channel"}
                if method == "gillespie-direct-v1"
                else {"schema", "channel_ids", "propensities", "times_s", "dormant_hazards"}
                if method == "gibson-bruck-next-reaction-v1"
                else None
            )
            if not isinstance(pending, dict) or (fields is not None and set(pending) != fields):
                raise ValueError("SBML integer continuation has unknown pending fields")
            if method == "gillespie-direct-v1" and type(pending["time_s"]) not in {int, float}:
                raise ValueError("SBML integer pending time must be a real scalar")
        result = advance_event_channels(
            self.compile(coordinates, time_s),
            self.species_ids,
            self.policy(coordinates, method, event_budget, tau_policy),
            self.counts(coordinates, time_s),
            memory["process"],
            duration_s, stop_after_boundary=stop_after_boundary,
            critical_reaction_ids=critical_reaction_ids,
            calendar_deadline_s=calendar_deadline_s,
            fluctuation_scaled_tau=self.requires_boundaries,
        )
        values = dict(coordinates)
        values.update(
            {"N:" + name: amount for name, amount in counts_to_mol(result.counts).items()}
        )
        self.counts(values, time_s + duration_s)
        return values, {"format": MEMORY_FORMAT, "process": result.continuation}, result


def initial_memory(seed: int, method: str) -> dict[str, Any]:
    return {"format": MEMORY_FORMAT, "process": initial_continuation(seed, algorithm=method)}
