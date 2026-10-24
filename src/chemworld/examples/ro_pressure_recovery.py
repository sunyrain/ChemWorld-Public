"""Install independent pressure-work recovery on a caller-declared aqueous RO."""

from collections.abc import Mapping
from dataclasses import replace

from chemworld.physchem.distributed_ro import DistributedROSpec
from chemworld.physchem.pressure_recovery import IsothermalLiquidPressureRecoverySpec
from chemworld.world.composition import WorldCompositionSpec
from chemworld.world.distributed_ro_world import distributed_ro_world_spec


def ro_pressure_recovery_example(
    model: DistributedROSpec,
    feed_amounts: Mapping[str, float],
    recovery: IsothermalLiquidPressureRecoverySpec,
    *,
    shared_clock: bool = True,
) -> WorldCompositionSpec:
    """Use caller-owned sourced properties; no guessed seawater calibration.

    Configure the resulting original session with set_flow_rate, then wait.
    Gross pump work remains positive; independent exported shaft work is
    negative flow_shaft_work_J. No generator or electrical credit is installed.
    """
    return distributed_ro_world_spec(
        replace(model, energy_recovery=recovery), feed_amounts, shared_clock=shared_clock
    )
