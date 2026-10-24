"""Independent closed-form flow and integrated-hazard inversion; no native imports."""

from math import exp, expm1, inf, log1p

import numpy as np


def simulate(case, seed, duration=3.):
    rng = np.random.Generator(np.random.PCG64(seed))

    def draw():
        return -log1p(-float(rng.random()))

    targets, hazards = [draw(), draw()], [0., 0.]
    time, a, enabled, events = 0., 1., 0, []
    while time < duration:
        rate = case["k0"] + case["k1"] * enabled
        channel = 1 if enabled else 0
        intensity = case["beta"] if enabled else case["alpha"] * a
        risk = targets[channel] - hazards[channel]
        if intensity == 0:
            delay = inf
        elif enabled or rate == 0:
            delay = risk / intensity
        elif risk * rate >= intensity:
            delay = inf  # A finite integrated hazard has a nonzero no-event atom.
        else:
            delay = -log1p(-risk * rate / intensity) / rate
        dt = min(delay, duration - time)
        hazards[channel] += intensity * (dt if enabled or rate == 0 else -expm1(-rate * dt) / rate)
        a *= exp(-rate * dt)
        time += dt
        if delay > dt:
            break
        hazards[channel] = targets[channel]
        targets[channel] += draw()
        enabled = 1 - enabled
        events.append({"time_s": time, "reaction_id": "deactivate" if channel else "activate"})
    return {"fraction_A": a, "E1": enabled, "events": events,
            "hazards": hazards, "targets": targets, "rng_state": rng.bit_generator.state}
