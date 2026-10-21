"""A declared static carrier stock participates in elapsed time without dynamics."""

from chemworld.runtime.advance_paths import advance_ledger
from chemworld.runtime.kernel_contracts import ModelProviderResult


class FiniteCarrierSupply:
    def __init__(self, composition):
        physics = None if composition is None else composition.physics
        self.vessels = (frozenset() if physics is None or physics.lab_clock is None
                        else frozenset(physics.finite_supply_vessels))

    def supports(self, state):
        return state.vessel_id in self.vessels

    def advance(self, state, action, *, heat=False, jacket_program=None):
        if heat or jacket_program is not None:
            return ModelProviderResult(
                {}, diagnostics={}, state=None, success=False, outcome="rejected",
                failure_reason="finite_carrier_source_is_a_static_stock",
            )
        duration = float(action["duration_s"])
        return ModelProviderResult(
            {}, diagnostics={"static_finite_carrier_source": state.vessel_id},
            state=state.replace(ledger=advance_ledger(state.ledger, duration)),
            actual_duration_s=duration,
        )
