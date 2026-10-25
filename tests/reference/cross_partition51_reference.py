"""Independent closed-form flow and exact availability intervals; no native imports."""

from math import exp, expm1, inf, log, log1p

import numpy as np


def simulate(case, seed, duration=3.):
    rng = np.random.Generator(np.random.PCG64(seed))

    def draw():
        return -log1p(-float(rng.random()))

    targets, hazards = [draw(), draw()], [0., 0.]
    d, a, b = case["D"], float(case["A"]), float(case["B"])
    time, events = 0., []
    while time < duration:
        k = case["forward"] + case["catalytic"] * d + case["reverse"]
        equilibrium = case["reverse"] * (a + b) / k if k else a

        def flow(dt, a=a, equilibrium=equilibrium, k=k):
            return equilibrium + (a - equilibrium) * exp(-k * dt) if k else a

        def integral(lo, hi, a=a, equilibrium=equilibrium, k=k):
            if k == 0:
                return a * (hi - lo)
            return equilibrium * (hi - lo) + (a - equilibrium) * exp(-k * lo) * (
                -expm1(-k * (hi - lo))) / k

        crossing = inf
        if k and a == 1 and equilibrium < 1:
            crossing = 0.
        elif k and (a - 1) * (equilibrium - 1) < 0:
            crossing = -log((1 - equilibrium) / (a - equilibrium)) / k

        def capture_hazard(dt, a=a, equilibrium=equilibrium, crossing=crossing,
                           integral=integral):
            if a >= 1:
                lo, hi = 0., min(dt, crossing) if equilibrium < 1 else dt
            else:
                lo, hi = min(dt, crossing) if equilibrium > 1 else dt, dt
            return case["capture"] * integral(lo, hi)

        remaining = duration - time
        release_rate = case["release"] * d
        release_delay = (targets[0] - hazards[0]) / release_rate if release_rate else inf
        risk = targets[1] - hazards[1]
        if capture_hazard(remaining) < risk:
            capture_delay = inf
        else:
            lo, hi = 0., remaining
            for _ in range(80):
                mid = (lo + hi) / 2
                if capture_hazard(mid) < risk:
                    lo = mid
                else:
                    hi = mid
            capture_delay = hi
        channel = 0 if release_delay <= capture_delay else 1
        delay = min(release_delay, capture_delay)
        dt = min(delay, remaining)
        hazards[0] += release_rate * dt
        hazards[1] += capture_hazard(dt)
        next_a = flow(dt)
        b += a - next_a
        a = next_a
        time += dt
        if delay > remaining:
            break
        if channel == 0:
            d, a = d - 1, a + 1
        else:
            if a < 1:
                raise ArithmeticError("reference capture lacks its full quantum")
            d, a = d + 1, a - 1
        hazards[channel] = targets[channel]
        targets[channel] += draw()
        events.append({"time_s": time, "reaction_id": ("release", "capture")[channel]})
    return {"D": d, "A": a, "B": b, "events": events, "hazards": hazards,
            "targets": targets, "rng_state": rng.bit_generator.state}
