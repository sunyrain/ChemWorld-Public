"""Whole-graph task validation and expansion costs before any native execution."""

from dataclasses import dataclass, field
from types import MappingProxyType

from chemworld.data.sbml_parser import acyclic_order
from chemworld.data.sedml_declarations import SEDMLDocument, integer, mathematics, records
from chemworld.data.sedml_ranges import SEDMLRanges, _range_reference
from chemworld.data.sedml_simulations import resolve_simulations


def _calculation_inputs(node, range_ids, model_ids):
    shorthand = node.attributes.get("range")
    if shorthand is not None and shorthand not in range_ids:
        raise ValueError("task calculation refers to an unknown range")
    variables = records(node, "listOfVariables")
    for variable in variables:
        reference = _range_reference(variable)
        if reference is not None:
            if reference not in range_ids:
                raise ValueError("task variable refers to an unknown range")
        elif (variable.attributes.get("modelReference") not in model_ids
                or "taskReference" in variable.attributes):
            raise ValueError("task calculations require a known current model variable")
    scope = ({v.attributes["id"] for v in variables}
             | {p.attributes["id"] for p in records(node, "listOfParameters")})
    if shorthand is not None:
        scope.add(shorthand)
    if set(mathematics(node).symbols)-scope:
        raise ValueError("task calculation has undeclared mathematical inputs")


@dataclass(frozen=True)
class SEDMLTasks:
    document: SEDMLDocument
    maximum_invocations: int = 10000
    nodes: object = field(init=False, repr=False)
    simulations: object = field(init=False, repr=False)
    ranges: object = field(init=False, repr=False)
    sub_tasks: object = field(init=False, repr=False)
    model_ids: object = field(init=False, repr=False)
    invocations: object = field(init=False)
    samples: object = field(init=False)
    roots: tuple = field(init=False)

    def __post_init__(self):
        if type(self.maximum_invocations) is not int or not 1 <= self.maximum_invocations <= 100000:
            raise ValueError("task invocation budget must be a finite positive integer")
        root = self.document.declaration
        nodes = {n.attributes["id"]: n for n in records(root, "listOfTasks")}
        if not nodes or len(nodes) > 128:
            raise ValueError("experiment requires a bounded nonempty task graph")
        simulations = resolve_simulations(self.document)
        models = {n.attributes["id"] for n in records(root, "listOfModels")}
        graph, ranges, sub_tasks = {}, {}, {}
        for identity, node in nodes.items():
            if node.local_name == "task":
                if (node.attributes["modelReference"] not in models
                        or node.attributes["simulationReference"] not in simulations):
                    raise ValueError("simple task refers to an unknown model or simulation")
                graph[identity] = set()
            else:
                ranges[identity] = SEDMLRanges(node)
                children = records(node, "listOfSubTasks")
                graph[identity] = {n.attributes["task"] for n in children}
                if graph[identity]-nodes.keys():
                    raise ValueError("repeated task refers to an unknown subtask")
                sub_tasks[identity] = tuple(sorted(children,
                    key=lambda n: integer(n.attributes.get("order", "0"))))
        order = acyclic_order(graph, label="SED-ML task")
        model_ids, invocations, samples, depths = {}, {}, {}, {}
        for identity in order:
            node = nodes[identity]
            if node.local_name == "task":
                model_ids[identity] = frozenset((node.attributes["modelReference"],))
                invocations[identity] = 1
                simulation = simulations[node.attributes["simulationReference"]]
                samples[identity] = len(simulation.source_times)
                depths[identity] = 1
                continue
            children = sub_tasks[identity]
            depths[identity] = 1+max(depths[n.attributes["task"]] for n in children)
            if depths[identity] > 16:
                raise ValueError("nested tasks exceed the sixteen-level execution profile")
            model_ids[identity] = frozenset().union(
                *(model_ids[n.attributes["task"]] for n in children))
            # Equal/omitted order permits either ordering. Shared model state
            # would make that choice change the experiment, so refuse ambiguity.
            def effects(child):
                involved = set(model_ids[child.attributes["task"]])
                for change in records(child, "listOfChanges"):
                    involved.add(change.attributes["modelReference"])
                    involved.update(v.attributes["modelReference"]
                        for v in records(change, "listOfVariables")
                        if "modelReference" in v.attributes)
                return involved

            for i, left in enumerate(children):
                for right in children[i+1:]:
                    same_order = (left.attributes.get("order") is None
                        or right.attributes.get("order") is None
                        or integer(left.attributes["order"]) == integer(right.attributes["order"]))
                    if same_order and effects(left) & effects(right):
                        raise ValueError("unordered subtasks share evolving model state")
            changes = (*records(node, "listOfChanges"),
                       *(c for child in children for c in records(child, "listOfChanges")))
            if any(c.attributes["modelReference"] not in model_ids[identity] for c in changes):
                raise ValueError("task changes must reference an involved model")
            range_ids = {r.attributes["id"] for r in records(node, "listOfRanges")}
            for calculation_node in (*changes, *(r for r in records(node, "listOfRanges")
                                                 if r.local_name == "functionalRange")):
                _calculation_inputs(calculation_node, range_ids, model_ids[identity])
            invocations[identity] = ranges[identity].count*sum(
                invocations[n.attributes["task"]] for n in children)
            samples[identity] = ranges[identity].count*sum(
                samples[n.attributes["task"]] for n in children)
            if invocations[identity] > self.maximum_invocations or samples[identity] > 2000000:
                raise ValueError("nested tasks exceed the declared finite execution budget")
        referenced = set().union(*graph.values())
        roots = tuple(identity for identity in nodes if identity not in referenced)
        if (sum(invocations[key] for key in roots) > self.maximum_invocations
                or sum(samples[key] for key in roots) > 2000000):
            raise ValueError("all task roots exceed the declared finite execution budget")
        for name, value in (("nodes", nodes), ("simulations", simulations), ("ranges", ranges),
                            ("sub_tasks", sub_tasks), ("model_ids", model_ids),
                            ("invocations", invocations), ("samples", samples)):
            object.__setattr__(self, name, MappingProxyType(value))
        object.__setattr__(self, "roots", roots)
