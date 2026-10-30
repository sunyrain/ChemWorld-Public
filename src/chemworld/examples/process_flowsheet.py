"""Authored ideal-gas two-recycle graph, finite stocks and user-defined design.

The synthetic A/B caloric model illustrates the API; it is not a material dataset.
All four tanks are positively preloaded and use a declared homogeneous gas branch.
"""

from __future__ import annotations

from dataclasses import replace

from chemworld.examples.molecular_process_design import two_stage_design
from chemworld.physchem.bounded_design import (
    BoundedDesignPolicy,
    DesignConstraint,
    DesignVariable,
)
from chemworld.physchem.molecular_mixture import MolecularStream
from chemworld.physchem.molecular_processes import molecular_process_state
from chemworld.physchem.process_flowsheet import (
    FlowsheetGraph,
    ProcessMixer,
    ProcessSource,
    ProcessSplitter,
    ProcessTear,
    ProcessUnit,
)
from chemworld.physchem.process_flowsheet_inventory import (
    FlowsheetTank,
    ProcessFlowsheetInventorySpec,
)
from chemworld.runtime.flowsheet_design import (
    FlowsheetCostTerm,
    FlowsheetDesignBinding,
    FlowsheetDesignSpec,
)


def recycle_flowsheet_example() -> tuple[
    ProcessFlowsheetInventorySpec, tuple[dict[str, float], ...], tuple[float, ...]
]:
    """Two coupled tears with finite feed and product tanks, in SI units."""
    domain = replace(two_stage_design(x_a=.25).path.domain, temperature_range_K=(250., 900.))
    pressure = 2e5

    def stream(temperature: float, fraction: float, rate: float) -> MolecularStream:
        state = molecular_process_state(
            domain, pressure_Pa=pressure, mole_fractions=(fraction, 1-fraction),
            coordinate="T", value=temperature, initial_temperature_K=temperature,
        )
        if not state.success or state.state is None:
            raise ValueError(state.failure_reason)
        return MolecularStream(state.state, rate)

    sources = (
        ProcessSource("feed_a", domain, stream(410., .7, .00004)),
        ProcessSource("feed_b", domain, stream(360., .2, .00002)),
    )
    units = (
        ProcessUnit("mix_a", ProcessMixer(domain, pressure, 400.),
                    ("feed_a", "return_b"), ("mixed_a",)),
        ProcessUnit("split_a", ProcessSplitter(domain, (.6, .4)),
                    ("mixed_a",), ("product_a", "return_a")),
        ProcessUnit("mix_b", ProcessMixer(domain, pressure, 390.),
                    ("feed_b", "return_a"), ("mixed_b",)),
        ProcessUnit("split_b", ProcessSplitter(domain, (.7, .3)),
                    ("mixed_b",), ("product_b", "return_b")),
    )
    graph = FlowsheetGraph(
        sources, units, {"product_a": "product_a", "product_b": "product_b"},
        tuple(ProcessTear(name, domain, stream(380., .4, .00001),
                          component_flow_scale_mol_s=.0001)
              for name in ("return_a", "return_b")),
    )
    names = ("source_a", "source_b", "receiver_a", "receiver_b")
    spec = ProcessFlowsheetInventorySpec(
        graph, tuple(FlowsheetTank(name, domain, pressure, "gas", .1, 5.) for name in names),
        {"feed_a": "source_a", "feed_b": "source_b"},
        {"product_a": "receiver_a", "product_b": "receiver_b"},
    )
    amounts = tuple({"A": total*x, "B": total*(1-x)}
                    for total, x in ((.02, .7), (.02, .2), (.01, .5), (.01, .5)))
    return spec, amounts, (410., 360., 350., 350.)


def recycle_feed_design() -> FlowsheetDesignSpec:
    """Minimize authored feed cost subject to an explicit product-rate target."""
    spec, _, _ = recycle_flowsheet_example()
    return FlowsheetDesignSpec(
        spec.graph, (DesignVariable("feed_a", "mol/s", (.00002, .00008), .00004),),
        (FlowsheetDesignBinding("feed_a", "source", "feed_a", ("molar_rate_mol_s",)),),
        "currency/s", (FlowsheetCostTerm("source:feed_a:molar_rate_mol_s", "mol/s", 1000.),),
        (DesignConstraint("outlet:product_b:molar_rate_mol_s", "mol/s", .00003,
                          1e-10, lower=.00003),),
        starts=({"feed_a": .00006},),
        policy=BoundedDesignPolicy(maximum_evaluations=60, objective_scale=.1),
    )
