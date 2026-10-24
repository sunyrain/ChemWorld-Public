"""Fixed 20-unit development block after the documented canaries."""

from dataclasses import replace
from math import pi, sqrt

from examples.suspension_treatment49 import example_feed, example_spec

from chemworld.physchem.transport import PumpSpec


def fixed_cases():
    cases = []
    for temperature in (288.15, 318.15):
        for endpoint in (0.0, 10.0, 100.0):
            for flow in (2.0, 10.0):
                spec = example_spec(
                    temperature=temperature,
                    final_ntu=endpoint,
                    floc_capture=0.95 if flow == 10 else 0.7,
                )
                feed = example_feed(spec, initial_floc_mol=1e-5 if endpoint == 0 else 0)
                cases.append(
                    (
                        f"reference-{temperature}-{endpoint}-{flow}",
                        spec,
                        feed,
                        flow / 6e7,
                        20.0,
                        None,
                        True,
                        None,
                    )
                )
    base = example_spec()
    for name, spec, flow, duration, expected in (
        ("pickup", base, 10, 200, "feed_pickup_exhausted"),
        (
            "clarified-full",
            replace(base, capacities_L=(0.1, 0.001, 0.1)),
            10,
            100,
            "clarified_full",
        ),
        ("sludge-full", replace(base, capacities_L=(0.1, 0.1, 0.0001)), 10, 100, "sludge_full"),
        ("no-forward", replace(base, pump=PumpSpec(0, 10, 0.7)), 10, 20, "no_forward_flow"),
        (
            "cooling-limit",
            replace(base, bath=replace(base.bath, maximum_cooling_power_W=1e-6)),
            10,
            20,
            "bath_cooling_limit",
        ),
        ("idle", base, 0, 20, None),
    ):
        cases.append((name, spec, example_feed(spec), flow / 6e7, duration, None, False, expected))
    cases.extend(
        (
            f"world-{'shared' if shared else 'local'}",
            base,
            example_feed(base),
            10 / 6e7,
            20,
            shared,
            False,
            None,
        )
        for shared in (False, True)
    )
    return cases


def independent_flows(spec, feed, requested):
    # These fixed synthetic cases remain in the laminar regime and omit fittings/elevation.
    mw = {
        p.component.identifier: p.component.molecular_weight_g_mol / 1000
        for p in spec.material.liquid.packages
    }
    volumes = {
        p.component.identifier: mw[p.component.identifier]
        / p.by_property("liquid_density")[0].coefficients["rho_ref"]
        for p in spec.material.liquid.packages
    }
    volume = (
        sum(feed.liquid_mol[k] * v for k, v in volumes.items())
        + (feed.dispersed_mol + feed.floc_mol) * spec.material.particle.molar_volume_m3_mol
    )
    resistance = (
        128 * spec.material.viscosity_Pa_s * spec.pipe.length_m / (pi * spec.pipe.diameter_m**4)
    )
    q = min(
        requested,
        spec.pump.maximum_pressure_rise_Pa / resistance,
        sqrt(spec.pump.maximum_shaft_power_W * spec.pump.efficiency / resistance),
    )
    rho = (
        sum(n * mw[k] for k, n in feed.liquid_mol.items())
        + (feed.dispersed_mol + feed.floc_mol) * spec.material.particle.molecular_weight_kg_mol
    ) / volume
    reynolds = 4 * rho * q / (pi * spec.pipe.diameter_m * spec.material.viscosity_Pa_s)
    if reynolds > 2000 or spec.pipe.fittings_loss_coefficient or spec.pipe.elevation_change_m:
        raise ValueError("independent fixed-block hydraulic solution is laminar only")
    inlet = {
        **{"liquid:" + k: n * q / volume for k, n in feed.liquid_mol.items()},
        "dispersed": feed.dispersed_mol * q / volume,
        "floc": feed.floc_mol * q / volume,
    }
    retained = spec.jar_test.final_tss_kg_m3 * q / spec.material.particle.molecular_weight_kg_mol
    out = {**inlet, "dispersed": retained, "floc": inlet["floc"] + inlet["dispersed"] - retained}
    capture = {
        k: spec.clarifier.dispersed_capture
        if k == "dispersed"
        else spec.clarifier.floc_capture
        if k == "floc"
        else spec.clarifier.sludge_liquid_fraction
        for k in out
    }
    return (
        q,
        volume,
        inlet,
        {k: n * (1 - capture[k]) for k, n in out.items()},
        {k: n * capture[k] for k, n in out.items()},
    )
