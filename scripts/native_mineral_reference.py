"""Independent DOP853 integration over constrained Reaktoro water chemistry.

Uses no native aqueous solve, mineral rate helper, event helper or ODE backend.
The input formulas/constants define the same synthetic model on both sides.
"""

import math
from time import perf_counter

from native_aqueous_reference import ReaktoroCarbonateReference
from scipy.integrate import solve_ivp

from chemworld.physchem.aqueous_mineral import COMPOSITIONS


class ReaktoroMineralReference:
    def __init__(self, aqueous_spec, prefix=None):
        self.aqueous = ReaktoroCarbonateReference(aqueous_spec, prefix)

    def advance(self, feed, rate_constant, geometry, times):
        area0, nref = geometry["reference_area_m2"], geometry["reference_amount_mol"]
        n_initial = feed["Calcite"]
        limit = min(
            sum(n * COMPOSITIONS[k].get(e, 0) for k, n in feed.items()) for e in ("C", "Ca")
        )
        calls = 0
        heartbeat_at = perf_counter()

        def water(n):
            nonlocal calls, heartbeat_at
            calls += 1
            now = perf_counter()
            if now - heartbeat_at >= 30:
                print(
                    f"Reaktoro mineral reference alive: {calls} aqueous closures; "
                    f"candidate calcite {n:.10g} mol",
                    flush=True,
                )
                heartbeat_at = now
            return self.aqueous.solve_fixed(feed, n)

        initial = water(n_initial)
        omega0 = initial["saturation_ratio"]
        reason = (
            "no_seed"
            if n_initial == 0
            else "inactive_surface"
            if area0 == 0
            else "zero_rate"
            if rate_constant == 0
            else "saturated"
            if abs(omega0 - 1) <= 1e-10
            else None
        )
        stop, final_q, solution = (0.0 if reason else None), None, None
        if reason is None:
            q0 = math.cbrt(n_initial / nref)

            def amount(q):
                return min(limit, max(0.0, float(q)) ** 3 * nref)

            def rhs(t, q):
                props = water(amount(q[0]))
                return [-area0 * rate_constant / (3 * nref) * (1 - props["saturation_ratio"])]

            def exhausted(t, q):
                return q[0]

            direction = 1 if omega0 > 1 else -1

            def equilibrium(t, q):
                return direction * (water(amount(q[0]))["saturation_ratio"] - 1) - 1e-10

            exhausted.terminal = equilibrium.terminal = True
            exhausted.direction = equilibrium.direction = -1
            solution = solve_ivp(
                rhs,
                (0, times[-1]),
                [q0],
                method="DOP853",
                rtol=1e-10,
                atol=1e-13,
                dense_output=True,
                events=(exhausted, equilibrium),
            )
            if not solution.success:
                raise RuntimeError(
                    "independent Reaktoro/DOP853 reference failed: " + solution.message
                )
            if len(solution.t_events[0]):
                stop, final_q, reason = float(solution.t_events[0][0]), 0.0, "depleted"
            elif len(solution.t_events[1]):
                stop = float(solution.t_events[1][0])
                final_q, reason = float(solution.y_events[1][0, 0]), "saturated"
        output = []
        for t in times:
            if solution is None or t == 0:
                props = initial
            else:
                q = final_q if stop is not None and t >= stop else float(solution.sol(t)[0])
                n = 0.0 if reason == "depleted" and t >= stop else nref * q**3
                if not 0 <= n <= limit:
                    raise RuntimeError("reference mineral left its actual elemental domain")
                props = water(n)
            current = props["amounts_mol"]["Calcite"]
            area = area0 * (current / nref) ** (2 / 3) if current > 0 and area0 > 0 else 0.0
            output.append({"time_s": t, "aqueous": props, "area_m2": area})
        return {
            "trajectory": output,
            "chemical_stop_time_s": stop,
            "chemical_stop_reason": reason,
            "aqueous_closure_calls": calls,
            "reference_integrator": "scipy.DOP853 with Reaktoro constrained equilibrium",
            "nfev": solution.nfev if solution is not None else 0,
        }
