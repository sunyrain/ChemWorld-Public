"""Execute declared experiment steps through the public native session API.

The adapter reads public laboratory time and declared Observation channels.
It does not read coordinates, integrate equations, or retain an evolving model
clock. Historical sample coordinates and native HistoryRecords are outputs.
"""

import hashlib
from dataclasses import dataclass, replace
from types import MappingProxyType
from uuid import UUID, uuid5

from chemworld.agents.base import HistoryRecord
from chemworld.data.sedml_models import calculation
from chemworld.data.sedml_ranges import _range_reference
from chemworld.data.sedml_reports import SEDMLArray
from chemworld.runtime import ChemWorldRuntime
from chemworld.world.sedml_protocol import SEDMLProtocol, parameter_target


class SEDMLExecutionError(RuntimeError):
    def __init__(self, message, *, history, runtime=None):
        super().__init__(message)
        self.history = tuple(history)
        self.snapshot = None if runtime is None else runtime.snapshot().to_dict()


@dataclass(frozen=True)
class SEDMLCourseResult:
    runtime: ChemWorldRuntime
    native_channels: object
    history: tuple
    initial_model_time: float

    def source_array(self, quantity):
        key = "$time" if quantity.observable is None else quantity.observable.channel_id
        array = self.native_channels[key]
        values = tuple(self.initial_model_time+value*quantity.native_to_source
                       if quantity.observable is None else value*quantity.native_to_source
                       for value in array.values)
        return SEDMLArray(values, array.shape, array.axes, array.sample_keys, quantity.source_unit)


def run_uniform_course(protocol, task_id, *, runtime=None, seed=0,
                       observation_noise_mode="sequential"):
    """Run one simple task, optionally continuing the same native model session.

The source grid specifies durations relative to this invocation's start. The
actual source/model clock continues across invocations, including warmup.
"""
    if not isinstance(protocol, SEDMLProtocol) or task_id not in protocol.tasks.nodes:
        raise ValueError("course execution requires a declared task in a compiled native protocol")
    task = protocol.tasks.nodes[task_id]
    if task.local_name != "task":
        raise ValueError("uniform course execution requires a simple task")
    model_id = task.attributes["modelReference"]
    world = protocol.worlds[model_id]
    simulation = protocol.tasks.simulations[task.attributes["simulationReference"]]
    return _run_course(world, simulation, runtime=runtime, seed=seed,
                       observation_noise_mode=observation_noise_mode)


def _run_course(world, simulation, *, runtime=None, seed=0,
                observation_noise_mode="sequential"):
    if runtime is None:
        runtime = ChemWorldRuntime(composition=world, seed=seed,
                                   observation_noise_mode=observation_noise_mode)
        runtime.reset(seed=seed)
    elif (not isinstance(runtime, ChemWorldRuntime) or runtime.compiled_composition is None
            or runtime.compiled_composition.spec.to_dict() != world.to_dict()):
        raise ValueError("continued native session does not match the compiled model declaration")
    spec = world.physics.models_of("sbml")["formal_workspace"]
    scale = spec.model.model_unit("timeUnits").time_s(1.)
    started_at = runtime.lab_status()["time_s"]
    channels = {o.channel_id: [] for o in spec.observables}
    channels["$time"] = []
    expected_units = {o.channel_id: o.unit for o in spec.observables}
    history, keys = [], []

    def execute(action):
        result = runtime.execute(action)
        reading = runtime.observe()
        history.append(HistoryRecord(len(history)+1, dict(action),
            {channel.channel_id: channel.value for channel in reading.channels},
            result.reward, dict(result.info), public_view=reading.native_record()))
        if result.info["operation_outcome"] != "success":
            raise SEDMLExecutionError("native experiment operation failed: "+str(action),
                                      history=history, runtime=runtime)

    try:
        for target in simulation.source_times:
            requested_time = started_at+(target-simulation.initial_time)*scale
            duration = requested_time-runtime.lab_status()["time_s"]
            if duration < 0:
                raise ValueError("source output grid would move the native clock backwards")
            if duration:
                execute({"operation": "wait", "duration_s": duration})
            if expected_units:
                execute({"operation": "measure", "instrument": "formal_quantity_meter"})
                reading = runtime.observe()
                if {c.channel_id: c.unit for c in reading.channels} != expected_units:
                    raise ValueError("native reading differs from declared channel units")
                for channel in reading.channels:
                    channels[channel.channel_id].append(channel.value)
            time_s = runtime.lab_status()["time_s"]
            channels["$time"].append(time_s)
            keys.append((spec.initial_model_time*scale+time_s,))
    except SEDMLExecutionError:
        raise
    except (ValueError, ArithmeticError, RuntimeError) as exc:
        raise SEDMLExecutionError(str(exc), history=history, runtime=runtime) from exc
    arrays = {key: SEDMLArray(tuple(values), (len(keys),), ("time",), tuple(keys),
                "s" if key == "$time" else expected_units[key]) for key, values in channels.items()}
    return SEDMLCourseResult(runtime, MappingProxyType(arrays), tuple(history),
                              spec.initial_model_time)


@dataclass(frozen=True)
class SEDMLExperimentResult:
    reports: dict
    history: tuple
    iterations: tuple
    snapshots: dict


class _TaskRunner:
    """Finite protocol orchestration; every physical effect is a native action."""

    def __init__(self, protocol, root_id, seed, noise_mode, identity_namespace=None):
        self.protocol, self.root_id = protocol, root_id
        self.seed, self.noise_mode = seed, noise_mode
        self.identity_namespace = identity_namespace
        self.runtimes, self.worlds, self.epochs = {}, {}, {}
        self.history, self.iterations = [], []
        self._readings = {}

    def _history(self, history, model_id, task_id):
        for record in history:
            self.history.append(replace(record, step=len(self.history)+1,
                decision_context={"root_task": self.root_id, "model_reference": model_id,
                                  "task_reference": task_id}))

    def reset(self, model_id, world=None):
        world = self.protocol.worlds[model_id] if world is None else world
        epoch = self.epochs.get(model_id, 0)
        binding = f"{self.seed}:{self.root_id}:{model_id}:{epoch}".encode()
        seed = int.from_bytes(hashlib.sha256(binding).digest()[:4], "big")
        runtime = ChemWorldRuntime(composition=world, seed=seed,
                                   observation_noise_mode=self.noise_mode)
        options = None if self.identity_namespace is None else {"replay_campaign_id":
            "episode-"+uuid5(UUID(self.identity_namespace),
                            f"{self.root_id}:{model_id}:{epoch}").hex}
        runtime.reset(seed=seed, options=options)
        self.runtimes[model_id], self.worlds[model_id] = runtime, world
        self.epochs[model_id] = epoch+1
        self._readings.pop(model_id, None)

    def session(self, model_id):
        if model_id not in self.runtimes:
            self.reset(model_id)
        return self.runtimes[model_id]

    def execute(self, model_id, task_id, action):
        runtime = self.session(model_id)
        result = runtime.execute(action)
        reading = runtime.observe()
        record = HistoryRecord(0, dict(action),
            {channel.channel_id: channel.value for channel in reading.channels},
            result.reward, dict(result.info), public_view=reading.native_record())
        self._history((record,), model_id, task_id)
        self._readings.pop(model_id, None)
        if result.info["operation_outcome"] != "success":
            raise SEDMLExecutionError("native scan operation failed: "+str(action),
                                      history=self.history, runtime=runtime)

    def model_value(self, variable, *, shadow=None):
        model_id, quantity = self.protocol.quantities[variable.attributes["id"]]
        observable = quantity.observable
        if (shadow is not None and observable is not None and observable.kind == "parameter"
                and (model_id, observable.source_id) in shadow):
            return shadow[model_id, observable.source_id]
        runtime = self.session(model_id)
        if observable is not None and model_id not in self._readings:
            self.execute(model_id, self.root_id,
                         {"operation": "measure", "instrument": "formal_quantity_meter"})
            self._readings[model_id] = {c.channel_id: c.value for c in runtime.observe().channels}
        spec = self.worlds[model_id].physics.models_of("sbml")["formal_workspace"]
        return quantity.value(self._readings.get(model_id, {}),
            time_s=runtime.lab_status()["time_s"], initial_model_time=spec.initial_model_time)

    def changes(self, nodes, ranges, task_id, *, initialize=False):
        self._readings.clear()
        shadow, targets = {}, {}

        def variable_value(variable):
            reference = _range_reference(variable)
            return (ranges[reference] if reference is not None else
                    self.model_value(variable, shadow=shadow))

        for node in nodes:
            model_id = node.attributes["modelReference"]
            target = parameter_target(self.protocol.models[model_id], node)
            reference = node.attributes.get("range")
            value = calculation(node, variable_value=variable_value,
                extra={} if reference is None else {reference: ranges[reference]})
            shadow[model_id, target] = value
            targets.setdefault(model_id, []).append(node.attributes["target"].endswith("/@value"))
        for model_id in sorted(targets):
            values = {name: value for (model, name), value in shadow.items() if model == model_id}
            if initialize and all(targets[model_id]):
                self.reset(model_id, self.protocol.world_with_parameters(model_id, values))
            else:
                self.execute(model_id, task_id,
                             {"operation": "set_model_parameters", "values": values})

def run_sedml_experiment(protocol, *, seed=0, observation_noise_mode="sequential"):
    """Execute all requested task outputs, preserving independent task roots."""
    from chemworld.eval.sedml_continuation import SEDMLExperimentSession

    session = SEDMLExperimentSession(protocol, seed=seed,
                                    observation_noise_mode=observation_noise_mode)
    session.advance()
    return session.result()
