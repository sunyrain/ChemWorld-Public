"""All 63 scientific public chemicals.utils names, using public documentation only."""

from functools import partial
from itertools import product

from chemworld.physchem import composition_math as c
from chemworld.physchem import fluid_response as f

EXCLUDED = {"to_num": "legacy string parser", "none_and_length_check": "input shape helper"}


def phase_adapter(d2P_dVdT, V=None, dP_dT=None, dP_dV=None, d2P_dV2=None):
    return f.phase_from_response(d2P_dVdT, molar_volume=V, dP_dT=dP_dT,
                                dP_dV=dP_dV, d2P_dV2=d2P_dV2)


ADAPTERS = {
    "property_mass_to_molar": f.mass_molar_property,
    "property_molar_to_mass": partial(f.mass_molar_property, to="mass"),
    "rho_to_Vm": f.density_molar_volume,
    "Vm_to_rho": f.density_molar_volume,
    "velocity_to_molar_velocity": f.velocity_basis,
    "v_to_v_molar": f.velocity_basis,
    "molar_velocity_to_velocity": partial(f.velocity_basis, to="mass"),
    "v_molar_to_v": partial(f.velocity_basis, to="mass"),
    "zs_to_ws": c.fraction_basis,
    "ws_to_zs": partial(c.fraction_basis, inverse=True),
    "zs_to_Vfs": c.fraction_basis,
    "Vfs_to_zs": partial(c.fraction_basis, inverse=True),
    "normalize": c.normalized_fractions,
    "remove_zeros": c.regularized_fractions,
    "SG": lambda rho, rho_ref=f.REFERENCE_WATER_DENSITY: f.gravity_scale(
        rho, reference_density=rho_ref
    ),
    "Watson_K": f.watson_characterization,
    "Parachor": f.parachor,
    "radius_of_gyration": lambda MW, A, B, C, planar=False: f.molecular_radius_of_gyration(
        MW, A, B, C, planar=planar
    ),
    "vapor_mass_quality": f.vapor_mass_fraction,
    "Z": f.compressibility_factor,
    "Cp_minus_Cv": f.heat_capacity_difference,
    "isobaric_expansion": f.isobaric_expansion,
    "isothermal_compressibility": f.isothermal_compressibility,
    "Joule_Thomson": lambda T, V, Cp, dV_dT=None, beta=None: f.joule_thomson(
        T, V, Cp, dV_dT=dV_dT, expansion=beta
    ),
    "speed_of_sound": lambda V, dP_dV, Cp, Cv, MW=None: f.acoustic_velocity(
        V, dP_dV, Cp, Cv, molecular_weight=MW
    ),
    "isentropic_exponent": f.isentropic_exponent,
    "isentropic_exponent_PV": f.isentropic_exponent_pv,
    "isentropic_exponent_PT": f.isentropic_exponent_pt,
    "isentropic_exponent_TV": f.isentropic_exponent_tv,
    "phase_identification_parameter": f.phase_identification_parameter,
    "phase_identification_parameter_phase": phase_adapter,
    "mixing_simple": c.weighted_property,
    "mixing_logarithmic": partial(c.weighted_property, rule="logarithmic"),
    "mixing_power": lambda xs, ps, r: c.weighted_property(xs, ps, rule="power", exponent=r),
    "dxs_to_dxsn1": c.constrained_composition_gradient,
    "d2xs_to_d2xsn1": c.constrained_composition_hessian,
    "dxs_to_dns": c.composition_gradient_to_moles,
    "dns_to_dn_partials": lambda dns, F: c.partial_molar_from_mole_gradient(F, dns),
    "dxs_to_dn_partials": lambda dxs, xs, F: c.partial_molar_properties(F, dxs, xs),
    "d2xs_to_dxdn_partials": c.partial_molar_composition_jacobian,
    "d2ns_to_dn2_partials": c.extensive_hessian_from_moles,
    "mix_component_flows": lambda ids1, ids2, f1, f2, xs1, xs2: c.combine_component_flows(
        (ids1, ids2), totals=(f1, f2), fractions=(xs1, xs2)
    ),
    "mix_component_partial_flows": lambda ids1, ids2, ns1=None, ns2=None: c.combine_component_flows(
        (ids1, ids2), partial_flows=(ns1, ns2)
    ),
    "mix_multiple_component_flows": lambda ids, flows, fractions: c.combine_component_flows(
        ids, totals=flows, fractions=fractions
    ),
    "solve_flow_composition_mix": c.solve_stream_composition,
    "ms_to_ns": lambda ms, MWs: c.convert_component_flows(
        ms, source="mass", target="molar", molecular_weights=MWs
    ),
    "ns_to_ms": lambda ns, MWs: c.convert_component_flows(
        ns, source="molar", target="mass", molecular_weights=MWs
    ),
    "ns_to_Qls": lambda ns, Vmls: c.convert_component_flows(
        ns, source="molar", target="reference_volume", reference_molar_volumes=Vmls
    ),
    "Qls_to_ns": lambda Qls, Vmls: c.convert_component_flows(
        Qls, source="reference_volume", target="molar", reference_molar_volumes=Vmls
    ),
    "ms_to_Qls": lambda ms, MWs, Vmls: c.convert_component_flows(
        ms,
        source="mass",
        target="reference_volume",
        molecular_weights=MWs,
        reference_molar_volumes=Vmls,
    ),
    "Qls_to_ms": lambda Qls, MWs, Vmls: c.convert_component_flows(
        Qls,
        source="reference_volume",
        target="mass",
        molecular_weights=MWs,
        reference_molar_volumes=Vmls,
    ),
}

for name, source, target in (
    ("API_to_SG", "API", "specific_gravity"),
    ("SG_to_API", "specific_gravity", "API"),
    ("API_to_rho", "API", "density"),
    ("rho_to_API", "density", "API"),
    ("Baume_heavy_to_SG", "Baume_heavy", "specific_gravity"),
    ("SG_to_Baume_heavy", "specific_gravity", "Baume_heavy"),
    ("Baume_light_to_SG", "Baume_light", "specific_gravity"),
    ("SG_to_Baume_light", "specific_gravity", "Baume_light"),
    ("Baume_heavy_to_rho", "Baume_heavy", "density"),
    ("rho_to_Baume_heavy", "density", "Baume_heavy"),
    ("Baume_light_to_rho", "Baume_light", "density"),
    ("rho_to_Baume_light", "density", "Baume_light"),
):
    ADAPTERS[name] = partial(f.gravity_scale, source=source, target=target)


def cases():
    """Fixed fixture design; no outputs or reference calculations choose coverage."""
    rows = []

    def add(name, *args, **kwargs):
        rows.append({"name": name, "args": args, "kwargs": kwargs})

    for value, mw in ((4.2, 18.01528), (-123.0, 100.0)):
        for name in (
            "property_mass_to_molar",
            "property_molar_to_mass",
            "velocity_to_molar_velocity",
            "molar_velocity_to_velocity",
            "v_to_v_molar",
            "v_molar_to_v",
        ):
            add(name, value, mw)
    for mw, rho, vm in ((18.01528, 997.0, 18e-6), (100.0, 800.0, 0.01)):
        add("rho_to_Vm", rho, mw)
        add("Vm_to_rho", vm, mw)
    for xs, mws, vms in (
        ((0.7, 0.2, 0.1), (18.0, 44.0, 70.0), (2e-5, 5e-5, 9e-5)),
        ((0.0, 0.4, 0.6), (2.0, 16.0, 30.0), (1e-4, 2e-4, 7e-4)),
    ):
        for name in ("zs_to_ws", "ws_to_zs"):
            add(name, xs, mws)
        for name in ("zs_to_Vfs", "Vfs_to_zs"):
            add(name, xs, vms)
        for name in ("ns_to_ms", "ms_to_ns"):
            add(name, (2.0, 3.0, 5.0), mws)
        for name in ("ns_to_Qls", "Qls_to_ns"):
            add(name, (2.0, 3.0, 5.0), vms)
        for name in ("ms_to_Qls", "Qls_to_ms"):
            add(name, (2.0, 3.0, 5.0), mws, vms)
        add("mixing_simple", xs, (-4.0, 7.0, 100.0))
        add("mixing_logarithmic", xs, (0.1, 0.3, 20.0))
        for exponent in (-4.0, -3.0, -2.0, -1.0, -0.5, 1 / 3, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 0.123):
            add("mixing_power", xs, (0.1, 0.3, 20.0), exponent)
        g, h = (
            (-0.0028, -0.00719, -0.00859),
            ((0.152, 0.08, 0.547), (0.08, 0.674, 0.729), (0.547, 0.729, 0.131)),
        )
        add("dxs_to_dxsn1", g)
        add("d2xs_to_d2xsn1", h)
        add("dxs_to_dns", g, xs)
        add("dns_to_dn_partials", g, 12.0)
        add("dxs_to_dn_partials", g, xs, 12.0)
        add("d2xs_to_dxdn_partials", h, xs)
        add("d2ns_to_dn2_partials", h, g)
    for values in ((3.0, 2.0, 1.0), (0.0, 1e-9, 1.0)):
        add("normalize", values)
        add("remove_zeros", list(values), 1e-12)
    for sg in (0.75, 1.2):
        rho = sg * f.REFERENCE_WATER_DENSITY
        add("SG", rho)
        for name in ADAPTERS:
            if name.startswith("SG_to_"):
                add(name, sg)
            elif name.startswith("rho_to_") and name != "rho_to_Vm":
                add(name, rho)
            elif name.startswith("API_to_"):
                add(name, 141.5 / sg - 131.5)
            elif name.startswith("Baume_heavy_to_"):
                add(name, 145 - 145 / sg)
            elif name.startswith("Baume_light_to_"):
                add(name, 140 / sg - 130)
        add("Watson_K", 400.0, sg)
    for vf in (0.0, 0.2, 0.8, 1.0):
        add("vapor_mass_quality", vf, 100.0, 20.0)
    for sigma in (0.0, 0.02672166960656005):
        add("Parachor", 100.15888, 800.8088185536124, 4.97865317223119, sigma)
    for planar in (False, True):
        add("radius_of_gyration", 32.00452, 5.692e-47, 3.367e-46, 3.681e-46, planar=planar)
    for t, p, v, pt, pv, cp, cv in (
        (300.0, 1e5, 0.025, 333.0, -4e6, 30.0, 21.7),
        (
            300.0,
            1e8,
            4.730885141495376e-5,
            509689.2959155567,
            -5417785576072.434,
            38.36583283578205,
            23.98081290153672,
        ),
    ):
        vt, vp = -pt / pv, 1 / pv
        add("Z", t, p, v)
        add("Cp_minus_Cv", t, pt, pv)
        add("isobaric_expansion", v, vt)
        add("isothermal_compressibility", v, vp)
        add("Joule_Thomson", t, v, cp, dV_dT=vt)
        add("Joule_Thomson", t, v, cp, beta=vt / v)
        add("speed_of_sound", v, pv, cp, cv)
        add("speed_of_sound", v, pv, cp, cv, MW=28.97)
        add("isentropic_exponent", cp, cv)
        add("isentropic_exponent_PV", cp, cv, v, p, pv)
        add("isentropic_exponent_PT", cp, p, vt)
        add("isentropic_exponent_TV", cv, v, pt)
    add("phase_identification_parameter", 0.025, 333.0, -4e6, 3.2e8, -13333.0)
    add(
        "phase_identification_parameter",
        0.000130229900874,
        582169.397484,
        -3.66431747236e12,
        4.48067893805e17,
        -20518995218.2,
    )
    add("phase_identification_parameter_phase", 1.0)
    add("phase_identification_parameter_phase", -13333.0, 0.025, 333.0, -4e6, 3.2e8)
    add(
        "phase_identification_parameter_phase",
        -20518995218.2,
        0.000130229900874,
        582169.397484,
        -3.66431747236e12,
        4.48067893805e17,
    )
    for ids2 in (("A", "B"), ("B", "C"), ("B", "A")):
        add("mix_component_flows", ("A", "B"), ids2, 3.0, 2.0, (0.2, 0.8), (0.7, 0.3))
        for ns1, ns2 in product((None, (1.0, 2.0)), repeat=2):
            add("mix_component_partial_flows", ("A", "B"), ids2, ns1, ns2)
        add(
            "mix_multiple_component_flows",
            (("A", "B"), ids2, ("C",)),
            (3.0, 2.0, 1.0),
            ((0.2, 0.8), (0.7, 0.3), (1.0,)),
        )
    # All 19 patterns among F/z/w with at least one F; two flow states.
    mws = (18.0, 44.0, 70.0)
    for flow in ((2.0, 3.0, 5.0), (0.1, 4.0, 100.0)):
        total, mass = sum(flow), sum(v * mw for v, mw in zip(flow, mws, strict=True))
        for pattern in product(range(3), repeat=3):
            if 0 not in pattern:
                continue
            fs = [v if k == 0 else None for k, v in zip(pattern, flow, strict=True)]
            zs = [v / total if k == 1 else None for k, v in zip(pattern, flow, strict=True)]
            ws = [
                v * mw / mass if k == 2 else None
                for k, v, mw in zip(pattern, flow, mws, strict=True)
            ]
            add("solve_flow_composition_mix", fs, zs, ws, mws)
    return rows
