"""Reference-only adapters; production never imports the upstream library."""

from fluids import __version__
from fluids import fittings as f

from chemworld.data.scientific_inputs import (
    InputSource,
    InputValue,
    QuantityColumn,
    ScientificRow,
    ScientificTable,
)
from chemworld.physchem.data_conflicts import DataSourceProvenance
from chemworld.physchem.fitting_curves import FittingCurve, FittingCurves


def specifications():
    polynomials = {
        "entrance_distance45_Miller": "entrance_distance_45_Miller_coeffs",
        "entrance_distance_Miller": "entrance_distance_Miller_coeffs",
        "entrance_rounded_Miller": "entrance_rounded_Miller_coeffs",
        "bend_rounded_Crane": "bend_rounded_Crane_coeffs",
        "bend_miter_Miller": "bend_miter_Miller_coeffs",
        "bend_reynolds_limit_Miller": "bend_rounded_Miller_C_Re_limit_1",
    }
    splines = {
        "entrance_distance_Harris": "entrance_distance_Harris_tck",
        "entrance_distance_Idelchik": "entrance_distance_Idelchik_tck",
        "entrance_beveled_Idelchik": "entrance_beveled_Idelchik_tck",
        "entrance_rounded_Harris": "entrance_rounded_Harris_tck",
        "entrance_rounded_Idelchik": "entrance_rounded_Idelchik_tck",
        "contraction_rounded_Miller": "tck_contraction_abrupt_Miller",
        "contraction_conical_Miller": "contraction_conical_Miller_tck",
        "contraction_conical_Blevins": "contraction_conical_Blevins_tck",
        "transition_friction_Idelchik": "contraction_conical_friction_Idelchik_tck",
        "diffuser_conical_Miller": "tck_diffuser_conical_Miller",
        "diffuser_conical_Idelchik": "diffuser_conical_Idelchik_tck",
        "bend_rounded_Miller": "tck_bend_rounded_Miller",
        "bend_reynolds_Miller": "tck_bend_rounded_Miller_C_Re",
    }
    linear = {
        "entrance_rounded_Crane": ("entrance_rounded_ratios_Crane", "entrance_rounded_Ks_Crane"),
        "contraction_rounded_Idelchik": (
            "contraction_round_Idelchik_ratios",
            "contraction_round_Idelchik_factors",
        ),
        "bend_miter_Crane": ("bend_miter_Crane_angles", "bend_miter_Crane_fds"),
        "bend_miter_Blevins": ("bend_miter_Blevins_angles", "bend_miter_Blevins_Ks"),
        "bend_standard_Crane": ("crane_standard_bend_angles", "crane_standard_bend_losses"),
        "miller_outlet_limit": ("bend_rounded_Miller_C_o_Kbs", "bend_rounded_Miller_C_o_limits"),
        "miller_outlet_low": ("bend_rounded_Miller_C_o_Kbs", "bend_rounded_Miller_C_o_limit_0_01"),
    }
    for name, attribute in polynomials.items():
        yield name, "polynomial", {"coefficients": list(getattr(f, attribute))}, (), False
    for name, attribute in splines.items():
        spec = getattr(f, attribute)
        if len(spec) == 3:
            yield (
                name,
                "spline1d",
                {"knots_x": list(spec[0]), "coefficients": list(spec[1])},
                (spec[2],),
                True,
            )
        else:
            yield (
                name,
                "spline2d",
                {"knots_x": list(spec[0]), "knots_y": list(spec[1]), "coefficients": list(spec[2])},
                tuple(spec[3:]),
                False,
            )
    for name, (xs, ys) in linear.items():
        yield (
            name,
            "linear",
            {"x": list(getattr(f, xs)), "value": list(getattr(f, ys))},
            (),
            name == "bend_standard_Crane",
        )
    for index, spec in enumerate(f.tck_bend_rounded_Miller_C_os):
        yield (
            f"miller_outlet_{index}",
            "spline1d",
            {"knots_x": list(spec[0]), "coefficients": list(spec[1])},
            (spec[2],),
            True,
        )


def source_curves():
    curves = {}
    source = InputSource(
        DataSourceProvenance(
            "fluids",
            0,
            "Published fitting curves",
            __version__,
            "Reference-only source data adapter",
        ),
        "literature",
    )
    for name, kind, arrays, degrees, extrapolate in specifications():
        coordinate = "x" if kind == "linear" else "index"
        columns = tuple(
            QuantityColumn(n, "dimensionless", "parameter", (-1e100, 1e100), "Source fit")
            for n in (coordinate, "value")
        )
        raw = (
            [("points", x, y) for x, y in zip(arrays["x"], arrays["value"], strict=True)]
            if kind == "linear"
            else [(group, i, y) for group, ys in arrays.items() for i, y in enumerate(ys)]
        )
        rows = tuple(
            ScientificRow(
                str(i),
                "fluids",
                group,
                str(i),
                "parameters",
                {
                    coordinate: InputValue(float(x), "dimensionless", None, "Source fit"),
                    "value": InputValue(float(y), "dimensionless", None, "Source fit"),
                },
            )
            for i, (group, x, y) in enumerate(raw)
        )
        curves[name] = FittingCurve(
            ScientificTable(name, "parameters", columns, (source,), rows),
            kind=kind,
            degrees=degrees,
            extrapolate=extrapolate,
        )
    return FittingCurves(curves)


if __name__ == "__main__":
    specs = list(specifications())
    print("curves", len(specs))
    print(
        "source_numeric_leaves",
        sum(sum(len(a) for a in arrays.values()) for _, _, arrays, _, _ in specs),
    )
    print(
        "source_rows",
        sum(
            len(arrays["x"]) if kind == "linear" else sum(len(a) for a in arrays.values())
            for _, kind, arrays, _, _ in specs
        ),
    )
