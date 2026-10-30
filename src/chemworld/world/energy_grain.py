"""Assemble finite-grain physical networks into ChemWorld apparatus."""
from collections.abc import Mapping

from chemworld.physchem.energy_grain_binding import compile_energy_grain_network
from chemworld.physchem.energy_grain_specs import EnergyGrainMasterEquationSpec
from chemworld.physchem.gas_reactor_network import GasReactorNetworkSpec
from chemworld.physchem.rrkm_grain_binding import RRKMGrainSpec, rrkm_grain_network
from chemworld.world.composition import WorldCompositionSpec


def energy_grain_world(
    spec: EnergyGrainMasterEquationSpec,
    reactor_network: GasReactorNetworkSpec,
    amounts_by_vessel: Mapping[str, Mapping[str, float]],
    *,
    budget: int = 100,
) -> WorldCompositionSpec:
    """Build the original finite gas network; no new runtime or ownership layer."""
    from chemworld.world.gas_reactor_network import gas_reactor_network_world

    if (
        not spec.baseline.min_temperature_K
        <= reactor_network.temperature_K
        <= spec.baseline.max_temperature_K
    ):
        raise ValueError("reactor temperature lies outside the grained model domain")
    return gas_reactor_network_world(
        reactor_network, compile_energy_grain_network(spec), amounts_by_vessel, budget=budget
    )


def rrkm_grain_world(
    spec: RRKMGrainSpec,
    reactor_network: GasReactorNetworkSpec,
    amounts_by_vessel: Mapping[str, Mapping[str, float]],
    *,
    budget: int = 100,
) -> WorldCompositionSpec:
    from chemworld.world.gas_reactor_network import gas_reactor_network_world

    return gas_reactor_network_world(
        reactor_network, rrkm_grain_network(spec), amounts_by_vessel, budget=budget
    )
