"""Independent amount-equation MOL reference; imports no native physical kernel."""

from time import monotonic

import numpy as np
from fixed_bed_cases import R, initial_data
from scipy.integrate import solve_ivp


def reference_case(case, cells):
    started = monotonic()
    volume_L, gas, bound = initial_data(case, cells)
    volume = volume_L * 0.001
    names = [*(f"cell_{i}" for i in range(cells)), "source", "collector", "purge"]
    ngas = (cells + 3) * 3
    y = np.r_[gas.ravel(), bound.ravel(), 0.0]
    site_capacity = 0.001 / cells
    masses = np.array([0.044, 0.018, 0.028])
    affinities, rates = np.array([100.0, 500.0]), np.array(case["rates"])
    offset = np.array([-20000.0, -30000.0])
    segments, time = [], 0.0
    stop, success, error = None, True, None
    # Snapshot is compared against one uninterrupted physical reference interval.
    stages = (("source", 30.0),) if case.get("snapshot") else case["stages"]
    for inlet, duration in stages:
        source, collector = names.index(inlet), names.index("collector")
        selected = [*range(cells), source, collector]
        faces = [
            (source, 0, 0.05 / cells),
            *[(i, i + 1, 0.1 / cells) for i in range(cells - 1)],
            (cells - 1, collector, 0.05 / cells),
        ]

        def velocities(amounts, faces=faces, source=source):
            pressure = amounts.sum(axis=1) * R * case["temperature"] / volume
            result = []
            for left, right, distance in faces:
                density = 0.5 * (
                    (amounts[left] @ masses) / volume[left]
                    + (amounts[right] @ masses) / volume[right]
                )
                delta = pressure[left] - pressure[right]
                a = distance * 150 * 1.8e-5 * 0.6**2 / (0.4**3 * (1e-4) ** 2)
                if left == source:
                    a += case.get("inlet_resistance_Pa_s_m", 0.0)
                b = distance * 1.75 * density * 0.6 / (0.4**3 * 1e-4)
                result.append(
                    np.sign(delta) * 2 * abs(delta) / (a + np.sqrt(a * a + 4 * b * abs(delta)))
                )
            return np.array(result)

        def rhs(t, values, faces=faces, velocities=velocities):
            n = values[:ngas].reshape(cells + 3, 3)
            q = values[ngas:-1].reshape(cells, 2)
            dn = np.zeros_like(n)
            # Gas concentration is mol/L in the authored concentration isotherm.
            activity = n[:cells, :2] / volume_L[:cells, None] * affinities
            equilibrium = site_capacity * activity / (1 + activity.sum(axis=1))[:, None]
            dq = rates * (equilibrium - q)
            positive = np.maximum(dq, 0).sum(axis=1)
            available = np.maximum(-dq, 0).sum(axis=1) + max(rates) * np.maximum(
                site_capacity - q.sum(axis=1), 0.0
            )
            for i in range(cells):
                if positive[i] > available[i]:
                    dq[i] = np.where(dq[i] > 0, dq[i] * available[i] / positive[i], dq[i])
            dn[:cells, :2] -= dq
            for (left, right, _), velocity in zip(faces, velocities(n), strict=True):
                upstream = left if velocity >= 0 else right
                flow = abs(velocity) * 1e-4 * n[upstream] / volume[upstream]
                dn[upstream] -= flow
                dn[right if velocity >= 0 else left] += flow
            return np.r_[dn.ravel(), dq.ravel(), np.sum(dq @ offset)]

        def pressure_event(index, boundary, sign):
            def event(t, values):
                n = values[:ngas].reshape(cells + 3, 3)
                return sign * (n[index].sum() * R * case["temperature"] / volume[index] - boundary)

            event.terminal = True
            event.direction = -1
            return event

        events, labels = [], []
        for i in selected:
            events.extend((pressure_event(i, 20000.0, 1), pressure_event(i, 160000.0, -1)))
            labels.extend((f"bed:minimum_pressure:{names[i]}", f"bed:maximum_pressure:{names[i]}"))

        def speed_event(t, values, velocities=velocities):
            return 0.5 - max(abs(velocities(values[:ngas].reshape(cells + 3, 3))))

        speed_event.terminal = True
        speed_event.direction = -1
        events.append(speed_event)
        labels.append("bed:maximum_face_velocity")
        initial_margins = [float(check(time, y)) for check in events]
        if min(initial_margins) < 0:
            index = int(np.argmin(initial_margins))
            stop, success = labels[index], False
            error = {
                "reason": "stage_initial_domain_violation",
                "inlet": inlet,
                "time_s": time,
                "event": stop,
                "margin": initial_margins[index],
            }
            break
        result = solve_ivp(
            rhs,
            (time, time + duration),
            y,
            method="Radau",
            rtol=2e-10,
            atol=1e-13,
            dense_output=True,
            events=events,
        )
        if not result.success:
            raise RuntimeError(result.message)
        segments.append(
            {
                "start": time,
                "end": float(result.t[-1]),
                "solution": result.sol,
                "nfev": result.nfev,
                "njev": result.njev,
            }
        )
        time, y = float(result.t[-1]), result.y[:, -1]
        if result.status == 1:
            stop = next(labels[i] for i, hits in enumerate(result.t_events) if len(hits))
            break

    def evaluate(t):
        if not 0 <= t <= time + 1e-10:
            raise ValueError("reference evaluation must stay in its accepted interval")
        if segments:
            segment = next((s for s in segments if t <= s["end"] + 1e-10), segments[-1])
            values = segment["solution"](t)
        else:
            values = y
        n, q = values[:ngas].reshape(cells + 3, 3), values[ngas:-1].reshape(cells, 2)
        return {
            "gas": n,
            "bound": q,
            "heat": float(values[-1]),
            "pressure": n.sum(axis=1) * R * case["temperature"] / volume,
        }

    return {
        "evaluate": evaluate,
        "end": time,
        "stop": stop,
        "names": names,
        "success": success,
        "error": error,
        "volumes_L": volume_L,
        "initial_gas": gas,
        "initial_bound": bound,
        "elapsed_s": monotonic() - started,
        "nfev": sum(s["nfev"] for s in segments),
        "njev": sum(s["njev"] for s in segments),
    }
