"""Independent synthetic reference formulas, without native physical evaluators."""

from math import exp, log, prod, sqrt

R = 8.31446261815324


def thermo(entry, temperature):
    if entry.species.phase == "gas":
        row = entry.gas_declaration["thermo"]["data"][0]
        assert tuple(row[1:5]) == (0, 0, 0, 0)
        cp = R * row[0]
        h = R * (row[0] * temperature + row[5])
        entropy = R * (row[0] * log(temperature) + row[6])
        return {
            "u": h - R * temperature,
            "h": h,
            "s": entropy,
            "g": h - temperature * entropy,
            "cv": cp - R,
        }
    model = entry.adsorbate
    energy = model.electronic_energy.intercept_J_mol + sum(
        model.electronic_energy.coefficients[k] * v for k, v in model.descriptors.items()
    )
    free, capacity = energy, 0.0
    for quantum in model.vibrational_energies_J_mol:
        ratio = quantum / (R * temperature)
        q = exp(-ratio)
        energy += quantum * (0.5 + q / (1 - q))
        free += 0.5 * quantum + R * temperature * log(1 - q)
        capacity += R * ratio**2 * q / (1 - q) ** 2
    return {"u": energy, "h": energy, "s": (energy - free) / temperature, "g": free, "cv": capacity}


def interaction(spec, slot, amounts):
    names = [s.species.species_id for s in spec.species if s.species.phase == "surface"]
    ns = [amounts[f"{slot.spec.slot_id}:{k}"] for k in names]
    total = sum(slot.values["capacity:" + site] for site in spec.site_types)
    gradient = [
        sum(a * n for a, n in zip(row, ns, strict=True)) / total
        for row in spec.interaction_matrix_J_mol
    ]
    return 0.5 * sum(n * g for n, g in zip(ns, gradient, strict=True)), dict(
        zip(names, gradient, strict=True)
    )


def rates(spec, slot, state):
    activities, potentials = {}, {}
    by_id = {s.species.species_id: s for s in spec.species}
    _, gradient = interaction(spec, slot, state.amounts_mol)
    for key, entry in by_id.items():
        potentials[key] = thermo(entry, state.temperature_K)["g"]
        if entry.species.phase == "gas":
            pressure = state.amounts_mol[key] * R * state.temperature_K / (state.volume_L * 1e-3)
            activities[key] = pressure / entry.gas_declaration["reference_pressure_Pa"]
        else:
            activities[key] = (
                state.amounts_mol[f"{slot.spec.slot_id}:{key}"]
                / slot.values["capacity:" + spec.adsorbate_sites[key]]
            )
            potentials[key] += gradient[key]
    for site in spec.site_types:
        vacancy = "vacancy:" + site
        activities[vacancy] = 1 - sum(
            activities[k] for k, own in spec.adsorbate_sites.items() if own == site
        )
        potentials[vacancy] = 0.0
    answer = {}
    for reaction in spec.reactions:
        delta = sum(n * potentials[k] for k, n in reaction.products.items()) - sum(
            n * potentials[k] for k, n in reaction.reactants.items()
        )
        barrier = reaction.bep.intercept_J_mol + reaction.bep.alpha * delta
        kf = reaction.attempt_frequency_s_inv * exp(-barrier / (R * state.temperature_K))
        kr = reaction.attempt_frequency_s_inv * exp(-(barrier - delta) / (R * state.temperature_K))
        scale = slot.values["capacity:" + reaction.rate_site]
        answer[reaction.reaction_id] = {
            "delta": delta,
            "forward_barrier": barrier,
            "reverse_barrier": barrier - delta,
            "kf": kf,
            "kr": kr,
            "f": scale * kf * prod(activities[k] ** n for k, n in reaction.reactants.items()),
            "r": scale * kr * prod(activities[k] ** n for k, n in reaction.products.items()),
        }
    return answer


def finite_adsorption_amount(spec, slot, state, time):
    """dx/dt=a(M-x)(S-x)-kr*x; finite gas, finite sites, no chemostat."""
    values = rates(spec, slot, state)["adsorb"]
    gas = next(s for s in spec.species if s.species.phase == "gas")
    x0 = state.amounts_mol[f"{slot.spec.slot_id}:bound_A"]
    total = state.amounts_mol[gas.species.species_id] + x0
    sites = slot.values["capacity:A"]
    concentration_activity = (
        R
        * state.temperature_K
        / (state.volume_L * 1e-3)
        / gas.gas_declaration["reference_pressure_Pa"]
    )
    a = values["kf"] * concentration_activity
    b = a * (total + sites) + values["kr"]
    c = a * total * sites
    separation = sqrt(b * b - 4 * a * c)
    stable_root = 2 * c / (b + separation)
    upper_root = (b + separation) / (2 * a)
    ratio = (x0 - stable_root) / (x0 - upper_root) * exp(-separation * time)
    return (stable_root - ratio * upper_root) / (1 - ratio)
