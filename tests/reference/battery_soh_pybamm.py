"""Independent released SOH equations and fixed synthetic/literature inputs."""

from dataclasses import replace
from itertools import pairwise

import numpy as np
from battery_ocv_cases import released_functions, source_curve

from chemworld.data.scientific_inputs import InputSource
from chemworld.physchem.battery_materials import ElectrodeOCV
from chemworld.physchem.battery_soh import ElectrodeSOHSpec, HalfCellSOHSpec
from chemworld.physchem.data_conflicts import DataSourceProvenance


def synthetic(coefficients, identity):
    return ElectrodeOCV(
        identity,
        InputSource(
            DataSourceProvenance(
                "synthetic-soh", 0, "Self-authored fixed electrode SOH polynomial", "1", "CC0-1.0"
            ),
            "synthetic",
        ),
        polynomial_V=tuple(coefficients),
    )


def case(index):
    n, p = synthetic((0.2, -0.1), "linear-negative"), synthetic((4.2, -0.8), "linear-positive")
    un, up = lambda x: 0.2 - 0.1 * x, lambda y: 4.2 - 0.8 * y
    spec = ElectrodeSOHSpec(n, p, 3600, 3600, 3.38, 3.92, 3600)
    if index == 2:
        n, p = synthetic((0.3, -0.2), "asymmetric-n"), synthetic((4.5, -0.6), "asymmetric-p")
        un, up = lambda x: 0.3 - 0.2 * x, lambda y: 4.5 - 0.6 * y
        spec = ElectrodeSOHSpec(n, p, 3600, 5400, 3.76, 4.24, 4500)
    elif index in (3, 4, 6, 7):
        n, p = source_curve("negative")[0], source_curve("positive")[0]
        functions = released_functions()
        un, up = functions["negative"], functions["positive"]
        if index in (3, 4):
            spec = ElectrodeSOHSpec(
                n,
                p,
                5000,
                4800,
                3.2 if index == 3 else 3.1,
                4.2 if index == 3 else 4.1,
                5000 if index == 3 else 4200,
            )
        elif index == 6:
            spec = ElectrodeSOHSpec(
                n,
                p,
                5000,
                4800,
                float(up(0.1 + 4100 / 4800) - un(0.08)),
                float(up(0.1) - un(0.9)),
                4100,
                "cell_capacity",
            )
        else:
            spec = HalfCellSOHSpec(p, 4800, float(up(0.85)), float(up(0.2)))
    elif index == 5:
        n, p = (
            synthetic((0.3, -0.2, -0.1), "quadratic-n"),
            synthetic((4.4, -0.6, -0.2), "quadratic-p"),
        )
        un, up = lambda x: 0.3 - 0.2 * x - 0.1 * x * x, lambda y: 4.4 - 0.6 * y - 0.2 * y * y
        spec = ElectrodeSOHSpec(n, p, 3600, 5400, 3.7535, 4.2295, 2160, "cell_capacity")
    elif index == 8:
        p, up = synthetic((0.3, -0.2), "half-linear"), lambda y: 0.3 - 0.2 * y
        spec = HalfCellSOHSpec(p, 3600, 0.12, 0.26)
    elif index in (9, 10):
        coefficients = np.polynomial.polynomial.polymul([0.0625, -0.5, 1], [0.5625, -1.5, 1]) * 16
        coefficients[0] += 3
        n, p = synthetic((0.0,), "quartic-counter"), synthetic(coefficients, "quartic-working")
        un, up = lambda x: 0 * x, lambda y: 3 + 16 * (y - 0.25) ** 2 * (y - 0.75) ** 2
        spec = (
            ElectrodeSOHSpec(n, p, 3600, 3600, 3.01, 3.04, 3600)
            if index == 9
            else HalfCellSOHSpec(p, 3600, 3.01, 3.04)
        )
    elif index == 11:
        spec = replace(spec, known_capacity_C=1e6)
    elif index == 12:
        spec = replace(
            spec,
            positive_capacity_C=5400,
            known_capacity_C=2160,
            known="cell_capacity",
            minimum_voltage_V=3.54,
            maximum_voltage_V=3.92,
        )
    elif index != 1:
        raise ValueError("fixed SOH block has twelve cases")
    return spec, un, up


def released_solution(spec, un, up):
    """Execute released full or half SOH with explicitly equal net OCV functions."""
    import pybamm
    from pybamm.models.full_battery_models.lithium_ion.electrode_soh import _ElectrodeSOH

    params = pybamm.ParameterValues("Chen2020")

    # U() in the released parameter class adds this artificial boundary term.
    # Cancel it in the supplied parameter FUNCTION, not in the solved equations.
    def supplied(function):
        return lambda theta: function(theta) - 1e-6 * (1 / theta + 1 / (theta - 1))

    params.update(
        {
            "Negative electrode OCP [V]": supplied(un),
            "Positive electrode OCP [V]": supplied(up),
            "Open-circuit voltage at 0% SOC [V]": spec.minimum_voltage_V,
            "Open-circuit voltage at 100% SOC [V]": spec.maximum_voltage_V,
        },
        check_already_exists=False,
    )
    half = isinstance(spec, HalfCellSOHSpec)
    if half:
        model = pybamm.lithium_ion.ElectrodeSOHHalfCell()
        inputs = {"Q_w": spec.working_capacity_C / 3600}
        solver = pybamm.AlgebraicSolver(method="lsq__trf", tol=1e-12)
    else:
        model = _ElectrodeSOH(
            known_value="cell capacity"
            if spec.known == "cell_capacity"
            else "cyclable lithium capacity"
        )
        inputs = {
            "Q_n": spec.negative_capacity_C / 3600,
            "Q_p": spec.positive_capacity_C / 3600,
            "Q" if spec.known == "cell_capacity" else "Q_Li": spec.known_capacity_C / 3600,
        }
        if spec.known == "cyclable_lithium":
            # Feasible midpoint initial guesses use only declared inventory/domain,
            # never a native root. The released model equations stay unchanged.
            low = max(
                0.01,
                (spec.known_capacity_C - 0.99 * spec.positive_capacity_C)
                / spec.negative_capacity_C,
            )
            high = min(
                0.99,
                (spec.known_capacity_C - 0.01 * spec.positive_capacity_C)
                / spec.negative_capacity_C,
            )
            for variable in model.initial_conditions:
                model.initial_conditions[variable] = (
                    low + 0.2 * (high - low) if variable.name == "x_0" else low + 0.8 * (high - low)
                )
        solver = pybamm.AlgebraicSolver(tol=1e-12)
    params.process_model(model)
    solution = solver.solve(model, [0], inputs=inputs)
    if half:
        return {
            "y_0": float(solution["x_0"].data[0]),
            "y_100": float(solution["x_100"].data[0]),
            "cell_capacity_C": 3600 * float(solution["Q"].data[0]),
        }
    return {
        **{k: float(solution[k].data[0]) for k in ("x_0", "x_100", "y_0", "y_100")},
        "cell_capacity_C": 3600 * float(solution["Q"].data[0]),
        "cyclable_lithium_C": 3600 * float(solution["Q_Li"].data[0]),
    }


def reference(spec, un, up, index):
    if index in (1, 2):
        an, bn = (0.2, -0.1) if index == 1 else (0.3, -0.2)
        ap, bp = (4.2, -0.8) if index == 1 else (4.5, -0.6)
        qn, qp, li = spec.negative_capacity_C, spec.positive_capacity_C, spec.known_capacity_C
        slope = -bp * qn / qp - bn
        x0 = (spec.minimum_voltage_V - ap + an - bp * li / qp) / slope
        xh = (spec.maximum_voltage_V - ap + an - bp * li / qp) / slope
        return [
            {
                "x_0": x0,
                "x_100": xh,
                "y_0": (li - qn * x0) / qp,
                "y_100": (li - qn * xh) / qp,
                "cell_capacity_C": qn * (xh - x0),
                "cyclable_lithium_C": li,
            }
        ], "independent linear algebra"
    if index == 8:
        return [
            {
                "y_0": (0.3 - spec.minimum_voltage_V) / 0.2,
                "y_100": (0.3 - spec.maximum_voltage_V) / 0.2,
                "cell_capacity_C": spec.working_capacity_C
                * (spec.maximum_voltage_V - spec.minimum_voltage_V)
                / 0.2,
            }
        ], "hand linear"
    if index in (9, 10):
        poly = (
            3
            + 16
            * np.polynomial.Polynomial([-0.25, 1]) ** 2
            * np.polynomial.Polynomial([-0.75, 1]) ** 2
        )
        roots = sorted(
            (float(z.real), kind)
            for kind, v in ((0, 3.01), (1, 3.04))
            for z in (poly - v).roots()
            if abs(z.imag) < 1e-10 and 0.01 < z.real < 0.99
        )
        answer = []
        for left, right in pairwise(roots):
            if (left[1], right[1]) == (1, 0) and poly.deriv()(left[0]) < 0:
                yh, y0 = left[0], right[0]
                item = {"y_0": y0, "y_100": yh, "cell_capacity_C": 3600 * (y0 - yh)}
                if index == 9:
                    item.update(x_0=1 - y0, x_100=1 - yh, cyclable_lithium_C=3600.0)
                answer.append(item)
        return answer, "independent quartic polynomial roots and derivative sign"
    return [released_solution(spec, un, up)], "PyBaMM 25.4.2 released SOH model"
