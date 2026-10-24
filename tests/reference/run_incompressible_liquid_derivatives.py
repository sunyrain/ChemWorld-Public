"""Supplementary fixed 20-liquid-state / 140-comparison response block."""

import argparse
import json
from collections import Counter
from pathlib import Path

from tests.reference.run_incompressible import compare

from chemworld.data.incompressible import incompressible_fluid


def point(fluid, T, p, x):
    import CoolProp.CoolProp as CP  # noqa: N817

    inputs = {"fluid": fluid.fluid_id, "T_K": T, "p_Pa": p, "x": x}

    def state():
        return fluid.state(T, p, x)

    def analytic(axis, field):
        s = state()
        return s.cp_J_kg_K/T if field is None else getattr(s, field)

    def difference(axis, field):
        if axis == "x" and fluid.fraction_basis == "pure":
            return 0.0
        h = .001 if axis == "T" else 1e-5
        values = [getattr(fluid.state(T+i*h if axis == "T" else T, p,
                                     x+i*h if axis == "x" else x), field) for i in (1, -1)]
        return (values[0]-values[1])/(2*h)

    rows = []
    for field, tf, xf, atol in (
        ("mass_density_kg_m3", "drho_dT_kg_m3_K", "drho_dx_kg_m3", 1e-7),
        ("enthalpy_J_kg", "cp_J_kg_K", "dh_dx_J_kg", .02),
        ("entropy_J_kg_K", None, "ds_dx_J_kg_K", 1e-4),
    ):
        for axis, derivative in (("T", tf), ("x", xf)):
            rows.append(compare(inputs, field+"_"+axis,
                                lambda axis=axis, d=derivative: analytic(axis, d),
                                lambda axis=axis, field=field: difference(axis, field), atol, 2e-5))

    def reference_derivative():
        ref = CP.AbstractState("INCOMP", fluid.fluid_id)
        if fluid.fraction_basis == "volume":
            ref.set_volu_fractions([x])
        else:
            ref.set_mass_fractions([x])
        values = []
        for offset in (.001, -.001):
            ref.update(CP.PT_INPUTS, p, T+offset)
            values.append(ref.hmass())
        return (values[0]-values[1])/.002

    rows.append(compare(inputs, "reference_h_T", lambda: state().cp_J_kg_K,
                        reference_derivative, .02, 2e-5))
    return rows


def main():
    import CoolProp

    if CoolProp.__version__ != "8.0.0":
        raise ValueError("qualification requires CoolProp 8.0.0")
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("never overwrite earlier derivative evidence")
    rows = []
    for name in ("MEG", "MCA", "APG", "DowQ"):
        fluid = incompressible_fluid(name)
        xmin, xmax = fluid.fraction_bounds
        xs = (1.,) if fluid.fraction_basis == "pure" else tuple(
            xmin+q*(xmax-xmin) for q in (.25, .5, .75))
        for x in xs:
            for p in (1e5, 2e6):
                lower, upper = fluid.temperature_bounds_K
                if dict(fluid.correlations)["T_freeze"] is not None:
                    lower = max(lower, fluid.freezing_temperature(p, x))
                T = lower+.75*(upper-lower)
                rows.extend(point(fluid, T, p, x))
                print(f"supplementary liquid responses {len(rows)}/140", flush=True)
    assert len(rows) == 140
    result = {"mode": "development_supplement; original_120_not_replaced",
              "reference": "CoolProp 8.0.0", "comparisons": rows,
              "counts": dict(Counter(r["status"] for r in rows))}
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(result["counts"])


if __name__ == "__main__":
    main()
