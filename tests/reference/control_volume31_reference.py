"""Independent scalar analytic finite mixer/CSTR reference; no ChemWorld imports."""

from math import exp, expm1


def evaluate(card, time_s):
    n = card["inventory_mol"]
    k = card["rate_constant_s"]
    total_flow, feed_fraction = 0.003, 0.4
    decay = total_flow / n + k
    steady = total_flow * feed_fraction / (total_flow + k * n)
    amplitude = 0.9 - steady
    fraction = steady + amplitude * exp(-decay * time_s)
    fraction_integral = steady * time_s - amplitude * expm1(-decay * time_s) / decay
    amount_a_rate = -n * decay * amplitude * exp(-decay * time_s)
    amounts = {name: dict(values) for name, values in card["amounts"].items()}
    rates = {name: dict.fromkeys(values, 0.0) for name, values in amounts.items()}
    transfers, flow_rates = {}, {}
    for edge, source, target, flow in card["edges"]:
        if source == "control":
            integrated = {"A": flow * fraction_integral, "B": flow * (time_s - fraction_integral)}
            instant = {"A": flow * fraction, "B": flow * (1 - fraction)}
        else:
            initial = card["amounts"][source]
            integrated = {s: flow * time_s * v / sum(initial.values()) for s, v in initial.items()}
            instant = {s: flow * v / sum(initial.values()) for s, v in initial.items()}
        transfers[edge], flow_rates[edge] = integrated, instant
        for species in ("A", "B"):
            amounts[source][species] -= integrated[species]
            amounts[target][species] += integrated[species]
            rates[source][species] -= instant[species]
            rates[target][species] += instant[species]
    extent = k * n * fraction_integral
    amounts["control"] = {"A": n * fraction, "B": n * (1 - fraction)}
    rates["control"] = {"A": amount_a_rate, "B": -amount_a_rate}
    h = {
        s: card["h0_J_mol"][s]
        + card["cp_J_mol_K"][s] * (card["temperature_K"] - card["reference_temperature_K"])
        for s in ("A", "B")
    }
    u = {s: h[s] - card["R"] * card["temperature_K"] for s in h}
    enthalpy = {
        edge: sum(h[s] * v for s, v in values.items()) for edge, values in transfers.items()
    }
    delta_u, bath = {}, {}
    for name, values in amounts.items():
        delta_u[name] = sum(u[s] * (v - card["amounts"][name][s]) for s, v in values.items())
        net_h = sum(
            enthalpy[e] * ((target == name) - (source == name))
            for e, source, target, _ in card["edges"]
        )
        bath[name] = delta_u[name] - net_h
    return {
        "time_s": time_s,
        "amounts": amounts,
        "rates_mol_s": rates,
        "transfers_mol": transfers,
        "flow_rates_mol_s": flow_rates,
        "enthalpy_J": enthalpy,
        "delta_u_J": delta_u,
        "bath_J": bath,
        "pressure_Pa": {
            name: sum(values.values())
            * card["R"]
            * card["temperature_K"]
            * 1000
            / card["volumes_L"][name]
            for name, values in amounts.items()
        },
        "temperature_K": card["temperature_K"],
        "reaction_extent_mol": extent,
        "steady_A_fraction": steady,
        "current_A_fraction": fraction,
        "integral_A_fraction_s": fraction_integral,
    }
