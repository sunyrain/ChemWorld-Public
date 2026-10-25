"""Actual WaterTAP expressions and independently assembled finite ODE reference.

Optional reference dependencies never enter the production execution path.
Source Pyomo expressions are converted to numeric callables, not reimplemented.
Finite tanks, utilities and event definitions are the declared native extension.
"""

from importlib.metadata import version

import numpy as np
from scipy.integrate import solve_ivp

COMPONENTS = (
    "S_I",
    "S_S",
    "X_I",
    "X_S",
    "X_BH",
    "X_BA",
    "X_P",
    "S_O",
    "S_NO",
    "S_NH",
    "S_ND",
    "X_ND",
    "S_ALK",
)
REACTIONS = tuple(f"R{i}" for i in range(1, 9))


class WaterTAPReference:
    def __init__(self, parameters):
        from pyomo.core.expr.sympy_tools import sympyify_expression
        from pyomo.environ import ConcreteModel, value
        from sympy import lambdify, symbols
        from watertap.property_models.unit_specific.activated_sludge.asm1_properties import (
            ASM1ParameterBlock,
        )
        from watertap.property_models.unit_specific.activated_sludge.asm1_reactions import (
            ASM1ReactionParameterBlock,
        )

        if version("watertap") != "1.8.0":
            raise ValueError("49G requires actual WaterTAP 1.8.0")
        m = ConcreteModel()
        m.pp = ASM1ParameterBlock()
        m.rp = ASM1ReactionParameterBlock(property_package=m.pp)
        for name, number in parameters.to_dict().items():
            source = name.replace("_per_day", "").replace("_kg_m3", "")
            source = source.replace("_m3_kg_day", "")
            getattr(m.rp, source).set_value(number)
            if source in ("f_p", "i_xb", "i_xp"):
                getattr(m.pp, source).set_value(number)
        m.states = m.pp.build_state_block([0])
        m.rxns = m.rp.build_reaction_block([0], state_block=m.states)
        state, rxn = m.states[0], m.rxns[0]
        _ = rxn.reaction_rate
        inputs = symbols("c0:13")
        state_symbols = {
            id(state.conc_mass_comp[k]): inputs[i] for i, k in enumerate(COMPONENTS) if k != "S_ALK"
        }
        state_symbols[id(state.alkalinity)] = inputs[-1] / 1000
        expressions, compiled = {}, {}
        for r in REACTIONS:
            mapping, expression = sympyify_expression(rxn.rate_expression[r].expr.arg(1))
            replacements = {}
            for symbol, original in mapping.sympy2pyomo.items():
                if id(original) in state_symbols:
                    replacements[symbol] = state_symbols[id(original)]
                elif original is rxn.reaction_rate["R7"]:
                    replacements[symbol] = expressions["R7"]
                else:
                    replacements[symbol] = value(original)
            # Cancellation evaluates the unique bounded zero-biomass limit;
            # no kinetic term is added or removed.
            expressions[r] = expression.xreplace(replacements).cancel()
            compiled[r] = lambdify(inputs, expressions[r], modules="math")
        self.functions = compiled
        self.matrix = np.array(
            [
                [
                    value(m.rp.rate_reaction_stoichiometry[r, "Liq", k])
                    * (1000 / 12 if k == "S_ALK" else 1.0)
                    for r in REACTIONS
                ]
                for k in COMPONENTS
            ]
        )
        self.nitrogen_loss = -self.matrix[8, 1]
        self.model = m

    def rates(self, concentrations):
        c = tuple(concentrations[k] for k in COMPONENTS)
        return np.array(
            [
                0.0 if r in ("R7", "R8") and c[3] == c[4] == 0.0 else self.functions[r](*c)
                for r in REACTIONS
            ]
        )

    def quality(self, concentrations):
        from pyomo.environ import value

        state = self.model.states[0]
        for key in COMPONENTS[:-1]:
            state.conc_mass_comp[key].set_value(concentrations[key])
        state.alkalinity.set_value(concentrations["S_ALK"] / 1000)
        return {
            "COD_kg_O2_m3": value(state.COD),
            "TSS_kg_m3": value(state.TSS),
            "TKN_kg_N_m3": value(state.TKN),
            "total_N_kg_N_m3": value(state.Total_N),
            "BOD5_raw_kg_O2_m3": value(state.BOD5["raw"]),
            "BOD5_effluent_kg_O2_m3": value(state.BOD5["effluent"]),
            "dissolved_O2_kg_m3": concentrations["S_O"],
            "alkalinity_mol_eq_m3": value(state.alkalinity) * 1000,
        }


def independent(spec, initial, feed_flow, duration, reference):
    """Direct concentration/tank equations with source-compiled rates; Radau."""
    n, members = len(spec.reactor_volumes_m3), len(initial.water_mol)
    ri, rs = spec.internal_recycle_fraction, spec.sludge_recycle_fraction
    ew, ep = spec.clarifier_soluble_effluent_fraction, spec.clarifier_particulate_effluent_fraction
    qt = feed_flow / (1 - ri - (1 - ri) * (1 - ew) * rs)
    qc, qi = qt * (1 - ri), qt * ri
    qe, qw = qc * ew, qc * (1 - ew) * (1 - rs)
    split = np.array([ep if k.startswith("X_") else ew for k in COMPONENTS])
    volumes = np.array(initial.water_mol) * spec.water_molar_volume_m3
    dvol = np.zeros(members)
    dvol[0], dvol[-2], dvol[-1] = -feed_flow, qe, qw
    offset = members * 13
    end = offset + 8 * n
    y0 = np.r_[
        [x[k] for x in initial.components for k in COMPONENTS],
        [x[r] for x in initial.extents for r in REACTIONS],
        list(initial.utilities.values()),
        initial.work["aeration_electricity_J"],
    ]
    feed = np.array([initial.components[0][k] for k in COMPONENTS]) / volumes[0]
    rv = np.array(spec.reactor_volumes_m3)
    kla = np.array(spec.kla_per_hour)
    power = (
        float(np.sum(spec.oxygen_equilibrium_kg_m3 * kla * rv / 1.8 * 1000))
        if spec.electricity_mode == "calculated"
        else n * spec.fixed_electricity_kWh_m3 * qt * 3.6e6
        if spec.electricity_mode == "fixed"
        else 0.0
    )

    def rhs(_t, y):
        pools = y[:offset].reshape(members, 13)
        c = pools[1 : n + 1] / rv[:, None]
        dp = np.zeros_like(pools)
        dp[0] = -feed_flow * feed
        dp[1] = feed_flow * feed + qi * c[-1] + qc * (1 - split) * rs * c[-1]
        for i in range(n - 1):
            dp[i + 1] -= qt * c[i]
            dp[i + 2] += qt * c[i]
        dp[n] -= (qi + qc) * c[-1]
        dp[-2] = qc * split * c[-1]
        dp[-1] = qc * (1 - split) * (1 - rs) * c[-1]
        rates = (
            np.array(
                [
                    reference.rates(dict(zip(COMPONENTS, np.maximum(row, 0.0), strict=True)))
                    for row in c
                ]
            )
            * rv[:, None]
        )
        dp[1 : n + 1] += rates @ reference.matrix.T
        oxygen = kla / 3600 * rv * (spec.oxygen_equilibrium_kg_m3 - np.maximum(c[:, 7], 0.0))
        dp[1 : n + 1, 7] += oxygen
        return np.r_[
            dp.ravel(),
            rates.ravel(),
            -np.maximum(oxygen, 0.0).sum(),
            np.maximum(-oxygen, 0.0).sum(),
            reference.nitrogen_loss * rates[:, 1].sum(),
            power,
        ]

    events = []

    def event(name, fn):
        fn.terminal, fn.direction = True, -1
        events.append((name, fn))

    if feed_flow:
        event("asm1_feed_pickup", lambda t, y: volumes[0] - feed_flow * t - spec.feed_pickup_m3)
        event("asm1_effluent_full", lambda t, y: spec.product_capacity_m3 - volumes[-2] - qe * t)
        event("asm1_sludge_full", lambda t, y: spec.sludge_capacity_m3 - volumes[-1] - qw * t)
    if any(kla):
        event("asm1_oxygen_supply_exhausted", lambda t, y: y[end])
        event("asm1_oxygen_offgas_full", lambda t, y: spec.oxygen_offgas_capacity_kg - y[end + 1])
    event("asm1_nitrogen_offgas_full", lambda t, y: spec.nitrogen_offgas_capacity_kg - y[end + 2])
    event("asm1_electricity_budget", lambda t, y: spec.maximum_aeration_energy_J - y[end + 3])
    event("asm1_aeration_power_limit", lambda t, y: spec.aeration_power_limit_W - power)
    for i in range(n):
        for k in (9, 12):
            index = (i + 1) * 13 + k

            def available(t, y, index=index):
                return 1e-30 if y[index] == 0 and rhs(t, y)[index] >= 0 else y[index]

            event(f"asm1_{i + 1}_{COMPONENTS[k]}_exhausted", available)
    stop = next((name for name, fn in events if fn(0.0, y0) <= 0.0), None)
    if stop:
        final, dt = y0, 0.0
    else:
        sol = solve_ivp(
            rhs,
            (0.0, duration),
            y0,
            method="Radau",
            rtol=2e-10,
            atol=1e-16,
            max_step=min(60.0, spec.maximum_step_s),
            events=[fn for _, fn in events],
        )
        if not sol.success:
            raise ArithmeticError(sol.message)
        final, dt = sol.y[:, -1], sol.t[-1]
        stop = next(
            (name for (name, _), hits in zip(events, sol.t_events, strict=True) if len(hits)), None
        )
    return {
        "components": final[:offset].reshape(members, 13),
        "extents": final[offset:end].reshape(n, 8),
        "utilities": final[end : end + 3],
        "electricity_J": final[-1],
        "volumes_m3": volumes + dvol * dt,
        "duration_s": dt,
        "stop": stop,
    }
