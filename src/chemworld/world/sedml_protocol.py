"""Compile closed experiment declarations to existing native world/action inputs.

This immutable plan has no live state or execution clock. Every change target,
observation, numerical policy and finite task expansion is checked before a
runtime may be created by an experiment runner.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from itertools import pairwise
from math import prod
from types import MappingProxyType
from xml.etree import ElementTree as ET

from chemworld.data.combine_archive import format_kind
from chemworld.data.sbml import SBMLModel
from chemworld.data.sedml_declarations import SEDMLDocument, records
from chemworld.data.sedml_models import _derived, _model_target, resolve_models
from chemworld.data.sedml_quantities import model_quantity
from chemworld.data.sedml_ranges import _range_reference
from chemworld.data.sedml_reports import SEDMLReports
from chemworld.data.sedml_stochastic import SEDMLStochasticAlgorithm
from chemworld.data.sedml_tasks import SEDMLTasks
from chemworld.data.xml_inputs import bounded_xml
from chemworld.world.domain_modules import normalize_number_map
from chemworld.world.sbml_world import sbml_world

PROTOCOL_FORMAT = "chemworld_sedml_native_protocol_v1"


def _layouts(tasks):
    layouts = {}

    def visit(identity):
        if identity in layouts:
            return layouts[identity]
        node = tasks.nodes[identity]
        if node.local_name == "task":
            simulation = tasks.simulations[node.attributes["simulationReference"]]
            result = {node.attributes["modelReference"]:
                      ((len(simulation.source_times),), ("time",))}
        else:
            children = [visit(n.attributes["task"]) for n in tasks.sub_tasks[identity]]
            count = tasks.ranges[identity].count
            concatenate = node.attributes.get("concatenate", "false") in ("true", "1")
            result = {}
            for model_id in tasks.model_ids[identity]:
                parts = [c[model_id] for c in children if model_id in c]
                shape, axes = parts[0]
                if concatenate:
                    if any((s[1:], a[1:]) != (shape[1:], axes[1:]) for s, a in parts):
                        raise ValueError("concatenated subtasks have incompatible trailing axes")
                    result[model_id] = ((count*sum(s[0] for s, _ in parts), *shape[1:]),
                                        ("sample:"+identity, *axes[1:]))
                else:
                    if any(p != parts[0] for p in parts):
                        raise ValueError("parallel report axes require matching subtask shapes")
                    if len(parts) > 1:
                        shape, axes = (len(parts), *shape), ("subtask:"+identity, *axes)
                    result[model_id] = ((count, *shape), ("repeat:"+identity, *axes))
        if any(len(shape) > 16 for shape, _ in result.values()):
            raise ValueError("report layout exceeds sixteen dimensions")
        layouts[identity] = MappingProxyType(result)
        return layouts[identity]

    for identity in tasks.nodes:
        visit(identity)
    return MappingProxyType(layouts)


def parameter_target(model, change):
    node, attribute = _model_target(bounded_xml(model.to_xml()),
        change.attributes["target"], change.namespaces)
    if node.tag.rsplit("}", 1)[-1] != "parameter" or attribute not in (None, "value"):
        raise ValueError("live experiment changes require global parameter or value targets")
    return node.get("id")


def _validate_sequential_inputs(protocol):
    for identity, task in protocol.tasks.nodes.items():
        if task.local_name != "repeatedTask":
            continue
        boundaries = [(records(task, "listOfChanges"),
                       task.attributes["resetModel"] in ("true", "1"))]
        boundaries.extend((records(child, "listOfChanges"), False)
                           for child in protocol.tasks.sub_tasks[identity])
        for changes, initialize in boundaries:
            previous = {}
            xml_models = {change.attributes["modelReference"] for change in changes
                          if change.attributes["target"].endswith("/@value")}
            for change in changes:
                for variable in records(change, "listOfVariables"):
                    if _range_reference(variable) is not None:
                        continue
                    model_id, quantity = protocol.quantities[variable.attributes["id"]]
                    observable = quantity.observable
                    if observable is None or model_id not in previous:
                        continue
                    spec = protocol.worlds[model_id].physics.models_of("sbml")["formal_workspace"]
                    initial = initialize and model_id in xml_models
                    initial_assigned = {r.get("symbol") for r in
                        spec.model.declaration.items("listOfInitialAssignments")}
                    if (observable.kind == "parameter"
                            and observable.source_id in previous[model_id]
                            and not (initial and observable.source_id in initial_assigned)):
                        continue  # The explicit sequential batch shadow supplies this value.
                    recipe_dependencies = {name: recipe.dependencies
                                           for name, recipe in spec.frame._recipes.items()}
                    if initial:
                        recipe_dependencies.update({r.get("symbol"): set(r.mathematics.symbols)
                            for r in spec.model.declaration.items("listOfInitialAssignments")})
                    dependencies, pending = set(), [observable.source_id]
                    if observable.kind == "concentration":
                        pending.append(spec.frame._species[observable.source_id].get("compartment"))
                    while pending:
                        symbol = pending.pop()
                        if symbol in dependencies:
                            continue
                        dependencies.add(symbol)
                        pending.extend(recipe_dependencies.get(symbol, ()))
                    if dependencies & previous[model_id]:
                        raise ValueError("sequential derived reads need candidate observations")
                model_id = change.attributes["modelReference"]
                previous.setdefault(model_id, set()).add(
                    parameter_target(protocol.models[model_id], change))


@dataclass(frozen=True)
class SEDMLProtocol:
    document: SEDMLDocument
    resources: object
    document_location: str = ""
    readout_relative_sd: float = 0.
    tasks: SEDMLTasks = field(init=False, repr=False)
    models: object = field(init=False, repr=False)
    reports: SEDMLReports = field(init=False, repr=False)
    quantities: object = field(init=False, repr=False)
    worlds: object = field(init=False, repr=False)
    layouts: object = field(init=False, repr=False)

    def __post_init__(self):
        resources = dict(self.resources)
        models = resolve_models(self.document, resources, document_location=self.document_location)
        tasks = SEDMLTasks(self.document)
        reports = SEDMLReports(tasks, models)
        layouts = _layouts(tasks)
        selected = {task_id for task_id, _, _ in reports.bindings.values()} or set(tasks.roots)
        if (sum(tasks.invocations[key] for key in selected) > tasks.maximum_invocations
                or sum(tasks.samples[key] for key in selected) > 2000000):
            raise ValueError("requested output tasks exceed the finite execution budget")
        quantities, controls, policies = {}, {}, {}
        observations = {key: set() for key in models}
        for _, model_id, quantity in reports.bindings.values():
            if quantity.observable is not None:
                observations[model_id].add(quantity.observable)
        for node in tasks.nodes.values():
            if node.local_name == "task":
                model_id = node.attributes["modelReference"]
                simulation = tasks.simulations[node.attributes["simulationReference"]]
                policy = (simulation.initial_time, simulation.algorithm)
                if model_id in policies and policies[model_id] != policy:
                    raise ValueError("continuation requires one origin and numerical policy")
                policies[model_id] = policy
                scale = models[model_id].model_unit("timeUnits").time_s(1.)
                times = (simulation.initial_time, *simulation.source_times)
                for left, right in pairwise(times):
                    duration = (right-left)*scale
                    if duration and not 1e-6 <= duration <= 14400.:
                        raise ValueError("output/prewarm intervals exceed native wait limits")
                continue
            children = tasks.sub_tasks[node.attributes["id"]]
            changes = (*records(node, "listOfChanges"),
                       *(c for child in children for c in records(child, "listOfChanges")))
            for change in changes:
                model_id = change.attributes["modelReference"]
                controls.setdefault(model_id, set()).add(parameter_target(models[model_id], change))
            reset_targets = {}
            for change in records(node, "listOfChanges"):
                reset_targets.setdefault(change.attributes["modelReference"], set()).add(
                    change.attributes["target"].endswith("/@value"))
            if (node.attributes["resetModel"] in ("true", "1")
                    and any(len(modes) > 1 for modes in reset_targets.values())):
                raise ValueError("a reset boundary cannot mix XML-input and mathematical targets")
            calculations = (*changes, *(r for r in records(node, "listOfRanges")
                                        if r.local_name == "functionalRange"))
            for calculation_node in calculations:
                for variable in records(calculation_node, "listOfVariables"):
                    if _range_reference(variable) is not None:
                        continue
                    model_id = variable.attributes["modelReference"]
                    quantity = model_quantity(models[model_id], variable)
                    quantities[variable.attributes["id"]] = (model_id, quantity)
                    if quantity.observable is not None:
                        observations[model_id].add(quantity.observable)
        native_values = sum(prod(layouts[identity][model_id][0])*(len(observations[model_id])+1)
            for identity in selected for model_id in tasks.model_ids[identity])
        report_values = sum(prod(layouts[identity][model_id][0])
                            for identity, model_id, _ in reports.bindings.values())
        if max(native_values, report_values) > 2000000:
            raise ValueError("total native/readout values exceed the finite output budget")
        worlds = {}
        budget = 2*sum(tasks.samples[key] for key in tasks.roots)+3*sum(
            tasks.invocations[key] for key in tasks.roots)+10
        for model_id, (origin, algorithm) in policies.items():
            model = models[model_id]
            scale = model.model_unit("timeUnits").time_s(1.)
            numerical = algorithm.native_policy(seconds_per_source_time=scale)
            if isinstance(algorithm, SEDMLStochasticAlgorithm):
                worlds[model_id] = sbml_world(model, initial_model_time=origin, budget=budget,
                    numerical_form="stochastic", solver_method=numerical.method,
                    seed=numerical.seed, maximum_events=numerical.maximum_events,
                    tau_policy=numerical.tau_policy, readout_relative_sd=self.readout_relative_sd,
                    controlled_parameters=tuple(sorted(controls.get(model_id, ()))),
                    observables=tuple(sorted(observations[model_id], key=lambda o: o.channel_id)))
                continue
            worlds[model_id] = sbml_world(model, initial_model_time=origin, budget=budget,
                solver_method=numerical.method, rtol=numerical.rtol, atol=numerical.atol,
                maximum_step_s=numerical.max_step_s, readout_relative_sd=self.readout_relative_sd,
                controlled_parameters=tuple(sorted(controls.get(model_id, ()))),
                observables=tuple(sorted(observations[model_id], key=lambda o: o.channel_id)))
        for name, value in (("resources", resources), ("models", models),
                            ("quantities", quantities), ("worlds", worlds)):
            object.__setattr__(self, name, MappingProxyType(dict(value)))
        object.__setattr__(self, "tasks", tasks)
        object.__setattr__(self, "reports", reports)
        object.__setattr__(self, "layouts", layouts)
        _validate_sequential_inputs(self)

    def to_dict(self):
        return {"format": PROTOCOL_FORMAT, "document": self.document.to_dict(),
            "resources": {key: value.to_dict() for key, value in self.resources.items()},
            "document_location": self.document_location,
            "readout_relative_sd": self.readout_relative_sd}

    def world_with_parameters(self, model_id, values):
        """An XML-input recipe for a fresh reset, never a write to a live session."""
        original = self.models[model_id]
        spec = self.worlds[model_id].physics.models_of("sbml")["formal_workspace"]
        # A reset to the same lexical inputs still has a reset-input source recipe.
        # The checkpoint decoder must reconstruct that recipe even with no deltas.
        values = {} if isinstance(values, Mapping) and not values else normalize_number_map(values)
        if set(values)-set(spec.controlled_parameters):
            raise ValueError("reset inputs must be declared controlled parameters")
        tree = bounded_xml(original.to_xml())
        namespace = "{http://www.sbml.org/sbml/level3/version2/core}"
        for node in tree.findall(namespace+"model/"+namespace+"listOfParameters/"
                                 +namespace+"parameter"):
            if node.get("id") in values:
                node.set("value", format(values[node.get("id")], ".17g"))
        model = _derived(original, ET.tostring(tree, encoding="unicode"), self.document,
                         model_id+":reset-inputs")
        if spec.integer_kinetics is not None:
            algorithm = next(self.tasks.simulations[n.attributes["simulationReference"]].algorithm
                for n in self.tasks.nodes.values() if n.local_name == "task"
                and n.attributes["modelReference"] == model_id)
            return sbml_world(model, initial_model_time=spec.initial_model_time,
                numerical_form="stochastic", budget=self.worlds[model_id].task.budget,
                solver_method=spec.solver_method, seed=algorithm.seed,
                maximum_events=spec.maximum_events, tau_policy=spec.tau_policy,
                readout_relative_sd=spec.readout_relative_sd,
                controlled_parameters=spec.controlled_parameters, observables=spec.observables)
        return sbml_world(model, initial_model_time=spec.initial_model_time,
            numerical_form=spec.numerical_form, budget=self.worlds[model_id].task.budget,
            solver_method=spec.solver_method, rtol=spec.rtol, atol=spec.atol,
            maximum_step_s=spec.maximum_step_s, readout_relative_sd=spec.readout_relative_sd,
            controlled_parameters=spec.controlled_parameters, observables=spec.observables)

    @classmethod
    def from_dict(cls, payload):
        if (not isinstance(payload, dict) or set(payload) != {"format", "document", "resources",
                "document_location", "readout_relative_sd"} or payload["format"] != PROTOCOL_FORMAT
                or not isinstance(payload["resources"], dict)):
            raise ValueError("unknown native experiment protocol fields or format")
        return cls(SEDMLDocument.from_dict(payload["document"]),
            {key: SBMLModel.from_dict(value) for key, value in payload["resources"].items()},
            payload["document_location"], payload["readout_relative_sd"])

    @classmethod
    def from_archive(cls, archive, *, unit_defaults=None, items_per_mole=None,
                     readout_relative_sd=0.):
        resources = {entry.location: SBMLModel(entry.data.decode("utf-8"),
            archive.source_for(entry.location), {} if unit_defaults is None else unit_defaults,
            items_per_mole) for entry in archive.entries if format_kind(entry.format) == "sbml"}
        return cls(archive.document(), resources, archive.master_location, readout_relative_sd)
