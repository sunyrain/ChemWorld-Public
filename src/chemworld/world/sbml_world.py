"""Declare a formal multicompartment entity on the existing native world."""

from collections.abc import Iterable
from typing import Any

from chemworld.data.sbml import SBMLModel
from chemworld.foundation import WorldState
from chemworld.foundation.lab_clock import LabClockSpec
from chemworld.physchem.reaction_network import ReactionNetworkSpec
from chemworld.physchem.sbml_system import SBMLObservable, SBMLSystemSpec
from chemworld.runtime.sbml_models import install_initial_model
from chemworld.world.composition import WorldCompositionSpec
from chemworld.world.physical_world import CompositionPhysics
from chemworld.world.sbml_instruments import FORMAL_METER


def sbml_world(
    model: SBMLModel, *, maximum_step_s: float | None = None,
    initial_model_time: float = 0., budget: int | None = 40,
    observables: Iterable[SBMLObservable] = (), numerical_form: str = "ode",
    readout_relative_sd: float = .01, rtol: float = 1e-9, atol: float = 1e-12,
    controlled_parameters: tuple[str, ...] = (), solver_method: str = "BDF",
    tau_policy: Any = None, seed: int = 0, maximum_events: int = 1000,
) -> WorldCompositionSpec:
    spec = SBMLSystemSpec(model, tuple(observables), initial_model_time, numerical_form,
        maximum_step_s=maximum_step_s, readout_relative_sd=readout_relative_sd,
        rtol=rtol, atol=atol, controlled_parameters=controlled_parameters,
        solver_method=solver_method, tau_policy=tau_policy, maximum_events=maximum_events)
    vessel = "formal_workspace"
    state = WorldState({}, 0., 298.15, 101325., "formal", vessel,
        liquid_heat_capacity_J_L_K=None,
        units={"amount": "mol", "volume": "L", "temperature": "K", "pressure": "Pa", "time": "s"})
    state = install_initial_model(state, spec, seed=seed)
    network = ReactionNetworkSpec("formal-sbml:"+model.declaration.get("id", "anonymous"), (), ())
    physics = CompositionPhysics(network, {vessel: state}, lab_clock=LabClockSpec(),
        models={"sbml": {vessel: spec}}, advance_bindings={vessel: "formal-sbml-shared"})
    components = [{"kind": "sbml_model", "vessel": vessel, "parameters": {}},
                  {"kind": "lab_schedule", "vessel": vessel,
                   "parameters": {"duration_range_s": [1e-6, 14400.]}}]
    if spec.observables:
        components.append({"kind": "observation", "vessel": vessel,
                           "parameters": {"instruments": [FORMAL_METER]}})
    return WorldCompositionSpec.from_dict({
        "schema_version": "chemworld-world-composition-0.1",
        "composition_id": network.network_id,
        "vessels": [{"id": vessel, "capacity_L": .1}],
        "components": components,
        "physics": physics.to_dict(),
        "task": {"objective": "characterize a declared formal multicompartment model",
                 "budget": budget,
                 "operations": ["wait", "terminate", *(["measure"] if spec.observables else []),
                                *(["set_model_parameters"] if spec.controlled_parameters else [])],
                 "instruments": [FORMAL_METER] if spec.observables else []},
    })
