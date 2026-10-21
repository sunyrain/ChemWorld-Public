"""Explicit selection of the existing condensed-liquid enthalpy implementation."""

LEGACY_LIQUID_STRATEGY = "legacy-liquid-enthalpy"


class LegacyLiquidAdvance:
    def __init__(self, composition=None):
        self.composition = composition
        self.thermal = None

    def bind_runtime(self, thermal):
        self.thermal = thermal

    @property
    def supported_operations(self):
        modules = self.composition.modules if self.composition is not None else self.thermal.modules
        return tuple(
            key
            for key, spec in modules.operation_specs().items()
            if spec.service_id in {"reaction_thermal", "control_program"}
        )

    def supports_legacy_input(self, state):
        """Only normalize an old plain liquid, never an unbound native model."""
        physics = self.composition.physics
        return (
            state.phase == "liquid"
            and state.vessel_id not in physics.finite_supply_vessels
            and state.process.stochastic_continuation is None
            and not any(s.spec.time_evolution == "coupled" for s in state.state_slots.slots)
            and not any(
                declaration.binding == "vessel" and state.vessel_id in physics.models_of(key)
                for key, declaration in physics.modules.model_specs().items()
            )
        )

    def supports(self, state):
        if self.composition is not None and self.composition.physics is not None:
            return (
                self.composition.physics.advance_bindings.get(state.vessel_id)
                == LEGACY_LIQUID_STRATEGY
                and self.supports_legacy_input(state)
            )
        # Legacy presets explicitly install this declaration. Crystal phases
        # belong to the separately declared finite-population strategy.
        return state.phase == "liquid" and not (
            state.phases is not None and {"mother_liquor", "solid"} <= state.phases.phases.keys()
        )

    def advance(self, state, action, *, heat, jacket_program=None):
        if self.thermal is None:
            raise ValueError("legacy liquid strategy requires its bound runtime")
        return self.thermal._integrate_legacy(
            state, action, heat=heat, jacket_program=jacket_program
        )

    def reconcile_pressure(self, state):
        return self.thermal._legacy_risk_and_pressure(state)
