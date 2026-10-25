"""Published ASM1 input/tear compositions in explicit native quality units."""

from dataclasses import replace

from chemworld.physchem.activated_sludge import activated_sludge_inventory
from chemworld.physchem.activated_sludge_kinetics import ASM1_COMPONENTS
from chemworld.physchem.activated_sludge_specs import ActivatedSludgeSpec

INFLUENT = dict(
    zip(
        ASM1_COMPONENTS,
        (
            0.030,
            0.0695,
            0.0512,
            0.20232,
            0.02817,
            0.0,
            0.0,
            0.0,
            0.0,
            0.03156,
            0.00695,
            0.01059,
            7.0,
        ),
        strict=True,
    )
)
MIXED = dict(
    zip(
        ASM1_COMPONENTS,
        (
            0.030,
            0.0146,
            1.149,
            0.0893,
            2.542,
            0.149,
            0.448,
            0.000393,
            0.00832,
            0.0077,
            0.00194,
            0.00562,
            5.0,
        ),
        strict=True,
    )
)
ZERO = dict.fromkeys(ASM1_COMPONENTS, 0.0)


def spec_case(reactors=5, **changes):
    spec = ActivatedSludgeSpec()
    if reactors == 1:
        spec = replace(spec, reactor_volumes_m3=(1e-5,), kla_per_hour=(10.0,))
    return replace(spec, **changes)


def inventory_case(
    spec,
    *,
    feed_volume=0.0005,
    mixed=None,
    feed=None,
    oxygen_supply=None,
    product_volume=0.0,
    sludge_volume=0.0,
):
    return activated_sludge_inventory(
        spec,
        (feed_volume, *spec.reactor_volumes_m3, product_volume, sludge_volume),
        (feed or INFLUENT, *((mixed or MIXED) for _ in spec.reactor_vessel_ids), ZERO, ZERO),
        oxygen_supply_kg=oxygen_supply,
    )


def rate_cases():
    ordinary = spec_case().parameters
    return [
        ("mixed-source", ordinary, MIXED),
        ("influent-source", ordinary, INFLUENT),
        ("aerobic", ordinary, {**MIXED, "S_O": 0.008}),
        ("anoxic", ordinary, {**MIXED, "S_O": 0.0}),
        ("no-substrate", ordinary, {**MIXED, "S_S": 0.0, "X_S": 0.0}),
        ("no-ammonia", ordinary, {**MIXED, "S_NH": 0.0}),
        ("no-nitrate", ordinary, {**MIXED, "S_NO": 0.0}),
        ("no-heterotroph", ordinary, {**MIXED, "X_BH": 0.0}),
        ("zero-biomass-hydrolysis-limit", ordinary, {**MIXED, "X_BH": 0.0, "X_S": 0.0}),
        ("R8-source-epsilon", ordinary, {**MIXED, "X_S": 1e-12}),
        (
            "modified-source-parameters",
            replace(ordinary, Y_H=0.6, Y_A=0.3, mu_H_per_day=3.0, eta_g=0.7, i_xb=0.09, i_xp=0.05),
            MIXED,
        ),
    ]


def transient_cases():
    """Frozen finite designs; flows are below every original pump limit."""
    q = 0.1281e-6 / 60
    aerobic = {**MIXED, "S_O": 0.002}
    inert = {**MIXED, "X_BH": 0.0, "X_BA": 0.0, "X_S": 0.0, "X_ND": 0.0}
    rows = [
        ("single-aerobic-batch", spec_case(1), {}, 0.0, 600.0),
        (
            "single-anoxic-batch",
            spec_case(1, kla_per_hour=(0.0,)),
            {"mixed": {**MIXED, "S_O": 0.0}},
            0.0,
            600.0,
        ),
        ("single-flow", spec_case(1), {}, q, 600.0),
        ("five-stage-recycle", spec_case(), {}, q, 600.0),
        ("five-no-internal-recycle", spec_case(internal_recycle_fraction=0.0), {}, q, 600.0),
        ("five-no-sludge-return", spec_case(sludge_recycle_fraction=0.0), {}, q, 600.0),
        (
            "five-particulate-escape",
            spec_case(clarifier_particulate_effluent_fraction=0.25),
            {},
            q,
            600.0,
        ),
        ("single-inert-carrier", spec_case(1), {"mixed": inert, "feed": inert}, q, 600.0),
        ("single-reverse-aeration", spec_case(1), {"mixed": {**inert, "S_O": 0.012}}, 0.0, 60.0),
        ("five-fixed-electricity", spec_case(electricity_mode="fixed"), {}, q, 60.0),
        ("five-zero-electricity", spec_case(electricity_mode="none"), {}, q, 60.0),
        ("feed-pickup", spec_case(1), {"feed_volume": 1e-7 + q * 10}, q, 30.0),
        ("effluent-full", spec_case(1, product_capacity_m3=1e-8), {}, q, 30.0),
        ("sludge-full", spec_case(1, sludge_capacity_m3=1e-10), {}, q, 30.0),
        ("oxygen-supply", spec_case(1), {"oxygen_supply": 1e-11}, 0.0, 30.0),
        (
            "oxygen-offgas",
            spec_case(1, oxygen_offgas_capacity_kg=1e-11),
            {"mixed": {**inert, "S_O": 0.012}},
            0.0,
            30.0,
        ),
        (
            "nitrogen-offgas",
            spec_case(1, kla_per_hour=(0.0,), nitrogen_offgas_capacity_kg=1e-11),
            {"mixed": {**MIXED, "S_O": 0.0}},
            0.0,
            30.0,
        ),
        ("NH-domain", spec_case(1), {"mixed": {**aerobic, "S_NH": 1e-8, "S_ND": 0.0}}, 0.0, 30.0),
        (
            "alkalinity-domain",
            spec_case(1),
            {"mixed": {**aerobic, "S_ALK": 1e-6, "S_ND": 0.0}},
            0.0,
            30.0,
        ),
        ("electricity-budget", spec_case(1, maximum_aeration_energy_J=0.0001), {}, 0.0, 30.0),
        ("electricity-power", spec_case(1, aeration_power_limit_W=0.0), {}, 0.0, 30.0),
        (
            "zero-reactive-pools",
            spec_case(1, kla_per_hour=(0.0,)),
            {"mixed": ZERO, "feed": ZERO},
            0.0,
            30.0,
        ),
    ]
    return rows
