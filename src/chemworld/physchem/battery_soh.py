"""Read-only equilibrium electrode SOH, not a capacity-fitting or ageing model.

Capacities use C, voltages V, energy J. Nonlinear OCV branches are isolated on
their declared domains. Incomplete search is never reported as uniqueness.
"""

from dataclasses import dataclass, replace
from itertools import pairwise
from math import isfinite
from numbers import Real

import numpy as np

from chemworld.data.scientific_inputs import InputSource, InputValue
from chemworld.physchem.battery_materials import ElectrodeOCV
from chemworld.physchem.caloric_math import Interval
from chemworld.physchem.residual_system import (
    ResidualDerivatives,
    ResidualEquation,
    ResidualSystem,
    ResidualVariable,
    solve_steady_residual,
)
from chemworld.physchem.root_isolation import (
    DEFAULT_ISOLATION_POLICY,
    IsolatedRoot,
    IsolationPolicy,
    isolate_roots,
)

SOH_VERSION = "equilibrium-electrode-soh-v1"


def _finite(value):
    if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value):
        raise ValueError("SOH inputs must be finite real numbers")
    return float(value)


def _validate(spec, capacity_names):
    for name in (*capacity_names, "minimum_voltage_V", "maximum_voltage_V"):
        object.__setattr__(spec, name, _finite(getattr(spec, name)))
    if any(getattr(spec, name) <= 0 for name in capacity_names):
        raise ValueError("SOH capacities must be strictly positive")
    if spec.minimum_voltage_V >= spec.maximum_voltage_V:
        raise ValueError("SOH needs an ordered voltage window")
    inputs = tuple(spec.parameter_inputs)
    if (
        any(not isinstance(k, str) or not k or not isinstance(v, InputValue) for k, v in inputs)
        or len(dict(inputs)) != len(inputs)
        or (inputs and (not isinstance(spec.source, InputSource) or not spec.source_row_id))
    ):
        raise ValueError("SOH parameter inputs require a source and unique row bindings")
    if spec.source is not None and not isinstance(spec.source, InputSource):
        raise ValueError("SOH source must use the common InputSource")
    object.__setattr__(spec, "parameter_inputs", inputs)


def _parameter_row(cls, table, row_id, bindings, **kwargs):
    names = (
        {"negative_capacity_C", "positive_capacity_C", "known_capacity_C"}
        if cls is ElectrodeSOHSpec
        else {"working_capacity_C"}
    )
    names |= {"minimum_voltage_V", "maximum_voltage_V"}
    if table.kind != "parameters" or set(bindings) != names:
        raise ValueError("SOH needs explicit complete parameter-column bindings")
    rows = [r for r in table.rows if r.row_id == row_id]
    if len(rows) != 1:
        raise ValueError("SOH requires one unambiguous parameter row")
    row = rows[0]
    values, inputs = {}, []
    for name, column in bindings.items():
        cell = row.values[column]
        if cell.quality != "valid":
            raise ValueError("SOH requires valid input parameters")
        values[name] = cell.to("V" if name.endswith("_V") else "C").value
        inputs.append((name, cell))
    return cls(
        **kwargs,
        **values,
        source_row_id=row_id,
        parameter_inputs=tuple(inputs),
        source=next(s for s in table.sources if s.provenance.source_id == row.source_id),
    )


def _to_dict(spec):
    curve_names = (
        ("negative_ocv", "positive_ocv") if isinstance(spec, ElectrodeSOHSpec) else ("working_ocv",)
    )
    result = {
        "model_version": SOH_VERSION,
        "kind": "full" if isinstance(spec, ElectrodeSOHSpec) else "positive-working-half",
    }
    for name in spec.__dataclass_fields__:
        value = getattr(spec, name)
        result[name] = (
            value.to_dict()
            if name in curve_names or (name == "source" and value)
            else {k: v.to_dict() for k, v in value}
            if name == "parameter_inputs"
            else value
        )
    return result


def soh_spec_from_dict(value):
    if value.get("model_version") != SOH_VERSION:
        raise ValueError("unsupported SOH version")
    kind = value.get("kind")
    cls = {"full": ElectrodeSOHSpec, "positive-working-half": HalfCellSOHSpec}.get(kind)
    if cls is None or set(value) != set(cls.__dataclass_fields__) | {"kind", "model_version"}:
        raise ValueError("unsupported SOH kind or fields")
    kwargs = {k: v for k, v in value.items() if k in cls.__dataclass_fields__}
    for name in ("negative_ocv", "positive_ocv", "working_ocv"):
        if name in kwargs:
            kwargs[name] = ElectrodeOCV.from_dict(kwargs[name])
    kwargs["source"] = InputSource.from_dict(kwargs["source"]) if kwargs["source"] else None
    kwargs["parameter_inputs"] = tuple(
        (k, InputValue.from_dict(v)) for k, v in kwargs["parameter_inputs"].items()
    )
    return cls(**kwargs)


@dataclass(frozen=True)
class ElectrodeSOHSpec:
    negative_ocv: ElectrodeOCV
    positive_ocv: ElectrodeOCV
    negative_capacity_C: float
    positive_capacity_C: float
    minimum_voltage_V: float
    maximum_voltage_V: float
    known_capacity_C: float
    known: str = "cyclable_lithium"
    source: InputSource | None = None
    source_row_id: str = ""
    parameter_inputs: tuple[tuple[str, InputValue], ...] = ()

    def __post_init__(self) -> None:
        _validate(self, ("negative_capacity_C", "positive_capacity_C", "known_capacity_C"))
        if self.known not in {"cyclable_lithium", "cell_capacity"}:
            raise ValueError("SOH known quantity must be cyclable_lithium or cell_capacity")
        if not all(isinstance(c, ElectrodeOCV) for c in (self.negative_ocv, self.positive_ocv)):
            raise ValueError(
                "SOH requires two explicit single-valued equilibrium electrode OCV curves"
            )

    def to_dict(self):
        return _to_dict(self)

    @classmethod
    def from_parameter_row(cls, table, row_id, *, bindings, **kwargs):
        return _parameter_row(cls, table, row_id, bindings, **kwargs)


@dataclass(frozen=True)
class HalfCellSOHSpec:
    working_ocv: ElectrodeOCV
    working_capacity_C: float
    minimum_voltage_V: float
    maximum_voltage_V: float
    source: InputSource | None = None
    source_row_id: str = ""
    parameter_inputs: tuple[tuple[str, InputValue], ...] = ()

    def __post_init__(self) -> None:
        _validate(self, ("working_capacity_C",))
        if not isinstance(self.working_ocv, ElectrodeOCV):
            raise ValueError(
                "half-cell SOH requires a single-valued equilibrium working electrode OCV "
                "versus Li/Li+"
            )

    def to_dict(self):
        return _to_dict(self)

    @classmethod
    def from_parameter_row(cls, table, row_id, *, bindings, **kwargs):
        return _parameter_row(cls, table, row_id, bindings, **kwargs)


@dataclass(frozen=True)
class SOHResult:
    spec: ElectrodeSOHSpec | HalfCellSOHSpec
    solutions: tuple[dict, ...]
    searches: tuple[dict, ...]
    unresolved: tuple[dict, ...] = ()
    infeasible_reason: str | None = None

    @property
    def status(self):
        return (
            "infeasible"
            if self.infeasible_reason
            else "unresolved"
            if self.unresolved
            else "no_solution"
            if not self.solutions
            else "unique"
            if len(self.solutions) == 1
            else "multiple"
        )

    def branch(self, index=None):
        if index is None:
            if self.status != "unique":
                raise ValueError("SOH requires a unique result or an explicit branch index")
            index = 0
        if type(index) is not int or not 0 <= index < len(self.solutions):
            raise ValueError("SOH branch index is outside the available solutions")
        return dict(self.solutions[index])

    def at_soc(self, soc, *, branch=None):
        soc = _finite(soc)
        if not 0 <= soc <= 1:
            raise ValueError("SOC must be inside [0, 1]")
        solution = self.branch(branch)
        y = solution["y_0"] + soc * (solution["y_100"] - solution["y_0"])
        if isinstance(self.spec, HalfCellSOHSpec):
            return {"soc": soc, "y": y, "voltage_V": self.spec.working_ocv.value(y)}
        x = solution["x_0"] + soc * (solution["x_100"] - solution["x_0"])
        return {
            "soc": soc,
            "x": x,
            "y": y,
            "voltage_V": self.spec.positive_ocv.value(y) - self.spec.negative_ocv.value(x),
        }

    def soc_at_voltage(self, voltage_V, *, branch=None, policy=DEFAULT_ISOLATION_POLICY):
        voltage_V = _finite(voltage_V)
        s = self.branch(branch)
        dy = s["y_100"] - s["y_0"]
        dx = 0.0 if isinstance(self.spec, HalfCellSOHSpec) else s["x_100"] - s["x_0"]
        positive = (
            self.spec.working_ocv
            if isinstance(self.spec, HalfCellSOHSpec)
            else self.spec.positive_ocv
        )

        def enclosure(a, b):
            y = s["y_0"] + dy * Interval(a[0], b[0])
            f = positive.enclosure(y.lower, y.upper, trial=True) - voltage_V
            j = dy * positive.enclosure(y.lower, y.upper, derivative=True, trial=True)
            if isinstance(self.spec, ElectrodeSOHSpec):
                x = s["x_0"] + dx * Interval(a[0], b[0])
                f -= self.spec.negative_ocv.enclosure(x.lower, x.upper, trial=True)
                j -= dx * self.spec.negative_ocv.enclosure(
                    x.lower, x.upper, derivative=True, trial=True
                )
            return (f,), ((j,),)

        def derivative(z):
            y = s["y_0"] + dy * z
            d = dy * positive.derivatives(y, trial=True)[1]
            if isinstance(self.spec, ElectrodeSOHSpec):
                d -= dx * self.spec.negative_ocv.derivatives(s["x_0"] + dx * z, trial=True)[1]
            return d

        return isolate_roots(
            lambda z: [self.at_soc(z[0], branch=branch)["voltage_V"] - voltage_V],
            lambda z: [[derivative(z[0])]],
            enclosure,
            variables=(_variable("soc", 0.0, 1.0),),
            residual_scales=(1.0,),
            policy=policy,
        )

    def theoretical_energy_j(self, initial_soc=1.0, final_soc=0.0, *, branch=None):
        """Positive for discharge. Reversible OCV energy, not loaded delivered work."""
        a, b = self.at_soc(initial_soc, branch=branch), self.at_soc(final_soc, branch=branch)
        if isinstance(self.spec, HalfCellSOHSpec):
            return self.spec.working_capacity_C * self.spec.working_ocv.integral(a["y"], b["y"])
        return self.spec.positive_capacity_C * self.spec.positive_ocv.integral(
            a["y"], b["y"]
        ) - self.spec.negative_capacity_C * self.spec.negative_ocv.integral(b["x"], a["x"])

    def to_dict(self):
        return {
            "status": self.status,
            "inputs": self.spec.to_dict(),
            "solutions": list(self.solutions),
            "searches": list(self.searches),
            "unresolved": list(self.unresolved),
            "infeasible_reason": self.infeasible_reason,
            "scope": "equilibrium voltage-window geometry; no fitted ageing or wet-cell validation",
        }


def _variable(name, lower, upper):
    return ResidualVariable(name, "dimensionless", role="algebraic", lower=lower, upper=upper)


def _cutoff_search(value, slope, enclosure, domain, target, policy):
    def bounds(a, b):
        f, d = enclosure(a[0], b[0])
        return (f - target,), ((d,),)

    return isolate_roots(
        lambda z: [value(z[0]) - target],
        lambda z: [[slope(z[0])]],
        bounds,
        variables=(_variable("stoichiometry", *domain),),
        residual_scales=(1.0,),
        policy=policy,
    )


def _lithium_geometry(spec):
    qn, qp, li = spec.negative_capacity_C, spec.positive_capacity_C, spec.known_capacity_C
    n, p = spec.negative_ocv, spec.positive_ocv
    low = max(n.stoichiometry_domain[0], (li - qp * p.stoichiometry_domain[1]) / qn)
    high = min(n.stoichiometry_domain[1], (li - qp * p.stoichiometry_domain[0]) / qn)

    def positive(x):
        return (li - qn * x) / qp

    def voltage(x):
        return p.value(positive(x), trial=True) - n.value(x, trial=True)

    def slope(x):
        return (
            -qn / qp * p.derivatives(positive(x), trial=True)[1] - n.derivatives(x, trial=True)[1]
        )

    def enclosure(a, b):
        y = (li - qn * Interval(a, b)) / qp
        return (
            p.enclosure(y.lower, y.upper, trial=True) - n.enclosure(a, b, trial=True),
            -qn / qp * p.enclosure(y.lower, y.upper, derivative=True, trial=True)
            - n.enclosure(a, b, derivative=True, trial=True),
        )

    return (low, high), positive, voltage, slope, enclosure


def _window_pairs(low_search, high_search, slope, *, increasing):
    """Adjacent cutoff roots delimit the entire window, not merely its endpoints."""
    roots = sorted(
        [
            (r["value"][0], kind)
            for kind, search in ((0, low_search), (1, high_search))
            for r in search.roots
        ]
    )
    pairs = []
    for left, right in pairwise(roots):
        if (left[1], right[1]) != ((0, 1) if increasing else (1, 0)):
            continue
        a, b = left[0], right[0]
        if b > a and (
            (slope(a) > 0 and slope(b) > 0) if increasing else (slope(a) < 0 and slope(b) < 0)
        ):
            pairs.append((a, b))
    return pairs


def _known_lithium(spec, policy):
    domain, positive, voltage, slope, enclosure = _lithium_geometry(spec)
    if domain[0] >= domain[1]:
        return SOHResult(
            spec, (), (), infeasible_reason="lithium inventory leaves no working interval"
        )
    low = _cutoff_search(voltage, slope, enclosure, domain, spec.minimum_voltage_V, policy)
    high = _cutoff_search(voltage, slope, enclosure, domain, spec.maximum_voltage_V, policy)
    searches = ({"cutoff": "minimum", **low.to_dict()}, {"cutoff": "maximum", **high.to_dict()})
    unresolved = tuple(
        {"cutoff": label, **box}
        for label, search in (("minimum", low), ("maximum", high))
        for box in search.unresolved
    )
    solutions = []
    if not unresolved:
        for x0, x100 in _window_pairs(low, high, slope, increasing=True):
            solutions.append(
                {
                    "x_0": x0,
                    "x_100": x100,
                    "y_0": positive(x0),
                    "y_100": positive(x100),
                    "cell_capacity_C": spec.negative_capacity_C * (x100 - x0),
                    "cyclable_lithium_C": spec.known_capacity_C,
                    "path_check": "adjacent complete cutoff roots, positive charging slope",
                }
            )
    return SOHResult(spec, tuple(solutions), searches, unresolved)


def _capacity_equations(spec, values, parameters=None):
    qn, qp, q, vmin, vmax = _parameters(spec)[1] if parameters is None else parameters
    xh, yh = values
    x0, y0 = xh - q / qn, yh + q / qp
    n, p = spec.negative_ocv, spec.positive_ocv
    return np.array(
        [
            p.value(yh, trial=True) - n.value(xh, trial=True) - vmax,
            p.value(y0, trial=True) - n.value(x0, trial=True) - vmin,
        ]
    )


def _capacity_jacobian(spec, values, parameters=None):
    qn, qp, q, _, _ = _parameters(spec)[1] if parameters is None else parameters
    xh, yh = values
    n, p = spec.negative_ocv, spec.positive_ocv
    return np.array(
        [
            [-n.derivatives(xh, trial=True)[1], p.derivatives(yh, trial=True)[1]],
            [-n.derivatives(xh - q / qn, trial=True)[1], p.derivatives(yh + q / qp, trial=True)[1]],
        ]
    )


def _known_cell_capacity(spec, policy):
    n, p = spec.negative_ocv, spec.positive_ocv
    qn, qp, q = spec.negative_capacity_C, spec.positive_capacity_C, spec.known_capacity_C
    a = (n.stoichiometry_domain[0] + q / qn, p.stoichiometry_domain[0])
    b = (n.stoichiometry_domain[1], p.stoichiometry_domain[1] - q / qp)
    if any(low > high for low, high in zip(a, b, strict=True)):
        return SOHResult(spec, (), (), infeasible_reason="cell capacity leaves no working interval")

    def enclosure(lower, upper):
        xh, yh = (Interval(low, high) for low, high in zip(lower, upper, strict=True))
        x0, y0 = xh - q / qn, yh + q / qp

        def u(curve, theta, derivative=False):
            return curve.enclosure(theta.lower, theta.upper, derivative=derivative, trial=True)

        return (
            (
                u(p, yh) - u(n, xh) - spec.maximum_voltage_V,
                u(p, y0) - u(n, x0) - spec.minimum_voltage_V,
            ),
            ((-u(n, xh, True), u(p, yh, True)), (-u(n, x0, True), u(p, y0, True))),
        )

    candidates: tuple[IsolatedRoot | dict[str, list[float]], ...]
    if any(low == high for low, high in zip(a, b, strict=True)):
        # A known capacity can use exactly an electrode's complete declared
        # domain. Eliminate fixed coordinates rather than widening/nudging it.
        free = [i for i in range(2) if a[i] < b[i]]
        if not free:
            candidates = (
                ({"value": list(a)},)
                if np.max(np.abs(_capacity_equations(spec, a))) <= policy.root.tolerance
                else ()
            )
            searches = [
                {
                    "fixed_endpoint_pair": list(a),
                    "residual_V": _capacity_equations(spec, a).tolist(),
                }
            ]
            unresolved = []
        else:
            coordinate = free[0]

            def expand(z):
                point = list(a)
                point[coordinate] = z
                return point

            def reduced_enclosure(low, high):
                f, j = enclosure(expand(low), expand(high))
                return f[0], j[0][coordinate]

            reduced = _cutoff_search(
                lambda z: _capacity_equations(spec, expand(z))[0],
                lambda z: _capacity_jacobian(spec, expand(z))[0, coordinate],
                reduced_enclosure,
                (a[coordinate], b[coordinate]),
                0.0,
                policy,
            )
            candidates = tuple(
                {"value": expand(r["value"][0])}
                for r in reduced.roots
                if np.max(np.abs(_capacity_equations(spec, expand(r["value"][0]))))
                <= policy.root.tolerance
            )
            searches, unresolved = [reduced.to_dict()], list(reduced.unresolved)
    else:
        roots = isolate_roots(
            lambda x: _capacity_equations(spec, x),
            lambda x: _capacity_jacobian(spec, x),
            enclosure,
            variables=tuple(
                _variable(name, low, high)
                for name, low, high in zip(("x_100", "y_100"), a, b, strict=True)
            ),
            residual_scales=(1.0, 1.0),
            policy=policy,
        )
        candidates = roots.roots
        searches, unresolved = [roots.to_dict()], list(roots.unresolved)
    solutions = []
    for root in candidates:
        xh, yh = root["value"]
        li = qn * xh + qp * yh
        path = _known_lithium(replace(spec, known="cyclable_lithium", known_capacity_C=li), policy)
        searches.append({"candidate": root["value"], "path_search": path.to_dict()})
        unresolved.extend({"candidate": root["value"], **r} for r in path.unresolved)
        for solution in path.solutions:
            if max(
                abs(solution["x_100"] - xh),
                abs(solution["x_0"] - (xh - q / qn)),
                abs(solution["y_100"] - yh),
                abs(solution["y_0"] - (yh + q / qp)),
            ) <= (4 * policy.coordinate_tolerance):
                solutions.append({**solution, "cell_capacity_C": q, "cyclable_lithium_C": li})
    return SOHResult(spec, tuple(solutions), tuple(searches), tuple(unresolved))


def solve_electrode_soh(spec, *, policy=DEFAULT_ISOLATION_POLICY, sensitivity=True):
    """All resolved equilibrium working intervals; never selects the first branch."""
    if not isinstance(policy, IsolationPolicy):
        raise ValueError("SOH requires the common isolation policy")
    if isinstance(spec, HalfCellSOHSpec):
        c = spec.working_ocv

        def enclosure(a, b):
            return c.enclosure(a, b), c.enclosure(a, b, derivative=True)

        low = _cutoff_search(
            c.value,
            lambda x: c.derivatives(x)[1],
            enclosure,
            c.stoichiometry_domain,
            spec.minimum_voltage_V,
            policy,
        )
        high = _cutoff_search(
            c.value,
            lambda x: c.derivatives(x)[1],
            enclosure,
            c.stoichiometry_domain,
            spec.maximum_voltage_V,
            policy,
        )
        unresolved = tuple(
            {"cutoff": label, **box}
            for label, search in (("minimum", low), ("maximum", high))
            for box in search.unresolved
        )
        solutions = []
        if not unresolved:
            for yh, y0 in _window_pairs(low, high, lambda x: c.derivatives(x)[1], increasing=False):
                solutions.append(
                    {
                        "y_0": y0,
                        "y_100": yh,
                        "cell_capacity_C": spec.working_capacity_C * (y0 - yh),
                        "reference": "positive working electrode versus Li/Li+ at 0 V",
                        "path_check": (
                            "adjacent complete cutoff roots, negative working-electrode slope"
                        ),
                    }
                )
        result = SOHResult(
            spec,
            tuple(solutions),
            ({"cutoff": "minimum", **low.to_dict()}, {"cutoff": "maximum", **high.to_dict()}),
            unresolved,
        )
    elif isinstance(spec, ElectrodeSOHSpec):
        result = (_known_lithium if spec.known == "cyclable_lithium" else _known_cell_capacity)(
            spec, policy
        )
    else:
        raise ValueError("SOH requires an explicit full or positive-working half-cell spec")
    if sensitivity and result.solutions:
        enriched = tuple(
            {**s, "local_parameter_response": soh_parameter_response(spec, s, policy=policy)}
            for s in result.solutions
        )
        result = replace(result, solutions=enriched)
    return result


def _parameters(spec):
    names = (
        ("working_capacity_C",)
        if isinstance(spec, HalfCellSOHSpec)
        else ("negative_capacity_C", "positive_capacity_C", "known_capacity_C")
    )
    names += ("minimum_voltage_V", "maximum_voltage_V")
    return names, tuple(getattr(spec, name) for name in names)


def soh_parameter_response(spec, solution, *, policy=DEFAULT_ISOLATION_POLICY):
    """Local implicit derivatives via the shared residual solver, not an estimator."""
    names, values = _parameters(spec)
    half = isinstance(spec, HalfCellSOHSpec)
    capacity_mode = not half and spec.known == "cell_capacity"
    coordinates = (
        ("y_0", "y_100") if half else ("x_100", "y_100") if capacity_mode else ("x_0", "x_100")
    )
    initial = np.array([solution[name] for name in coordinates])

    def residual(z, params):
        if capacity_mode:
            return _capacity_equations(spec, z, params)
        if half:
            return np.array(
                [
                    spec.working_ocv.value(z[0], trial=True) - params[1],
                    spec.working_ocv.value(z[1], trial=True) - params[2],
                ]
            )
        qn, qp, li, vmin, vmax = params
        return np.array(
            [
                spec.positive_ocv.value((li - qn * x) / qp, trial=True)
                - spec.negative_ocv.value(x, trial=True)
                - v
                for x, v in zip(z, (vmin, vmax), strict=True)
            ]
        )

    def state(z, params):
        if capacity_mode:
            return _capacity_jacobian(spec, z, params)
        if half:
            return np.diag([spec.working_ocv.derivatives(y, trial=True)[1] for y in z])
        qn, qp, li, _, _ = params
        return np.diag(
            [
                -qn / qp * spec.positive_ocv.derivatives((li - qn * x) / qp, trial=True)[1]
                - spec.negative_ocv.derivatives(x, trial=True)[1]
                for x in z
            ]
        )

    def parameter(z, params):
        if half:
            return np.array([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
        qn, qp, q, _, _ = params
        if capacity_mode:
            x0, y0 = z[0] - q / qn, z[1] + q / qp
            un = spec.negative_ocv.derivatives(x0, trial=True)[1]
            up = spec.positive_ocv.derivatives(y0, trial=True)[1]
            return np.array(
                [
                    [0.0, 0.0, 0.0, 0.0, -1.0],
                    [-un * q / qn**2, -up * q / qp**2, up / qp + un / qn, -1.0, 0.0],
                ]
            )
        rows = []
        for i, x in enumerate(z):
            y = (q - qn * x) / qp
            up = spec.positive_ocv.derivatives(y, trial=True)[1]
            rows.append([-up * x / qp, -up * y / qp, up / qp, -float(i == 0), -float(i == 1)])
        return np.array(rows)

    if np.max(np.abs(residual(initial, values))) > 1e-8:
        raise ValueError("SOH parameter response requires an actual endpoint solution")
    for name in ("x_0", "x_100", "y_0", "y_100"):
        if name in solution:
            curve = (
                spec.working_ocv
                if half
                else spec.negative_ocv
                if name.startswith("x")
                else spec.positive_ocv
            )
            curve.value(solution[name])  # strict source-domain validation
    j = state(initial, values)
    singular = np.linalg.svd(j, compute_uv=False)
    rank = int(np.sum(singular > 1e-12 * max(1.0, singular[0])))
    if rank < 2:
        return {"status": "rank_deficient", "rank": rank, "singular_values": singular.tolist()}
    system = ResidualSystem(
        "equilibrium-electrode-soh",
        tuple(ResidualVariable(name, "dimensionless", role="algebraic") for name in coordinates),
        tuple(
            ResidualEquation(label + "-cutoff", "V")
            for label in (("maximum", "minimum") if capacity_mode else ("minimum", "maximum"))
        ),
        lambda t, z, rate, params: residual(z, params),
        ResidualDerivatives(
            "analytic",
            state=lambda t, z, rate, p: state(z, p),
            rate=lambda t, z, rate, p: np.zeros((2, 2)),
            parameter=lambda t, z, rate, p: parameter(z, p),
        ),
        parameters=tuple(
            ResidualVariable(
                name, "V" if name.endswith("_V") else "C", max(1.0, abs(v)), role="parameter"
            )
            for name, v in zip(names, values, strict=True)
        ),
        parameter_values=values,
    )
    response = solve_steady_residual(system, initial, policy=policy.root, sensitivity=True)
    if not response.root.success:
        return {
            "status": "point_root_failed",
            "rank": rank,
            "failure_reason": response.root.failure_reason,
        }
    derivatives = response.parameter_response
    if half:
        dq = spec.working_capacity_C * (derivatives[0] - derivatives[1])
        dq[0] += solution["y_0"] - solution["y_100"]
        output = {"cell_capacity_C": dq.tolist()}
    elif capacity_mode:
        dli = spec.negative_capacity_C * derivatives[0] + spec.positive_capacity_C * derivatives[1]
        dli[0] += solution["x_100"]
        dli[1] += solution["y_100"]
        output = {"cyclable_lithium_C": dli.tolist()}
    else:
        dq = spec.negative_capacity_C * (derivatives[1] - derivatives[0])
        dq[0] += solution["x_100"] - solution["x_0"]
        output = {"cell_capacity_C": dq.tolist()}
    return {
        "status": "local_full_rank",
        "rank": rank,
        "parameters": list(names),
        "coordinates": list(coordinates),
        "singular_values": singular.tolist(),
        "coordinate_derivatives": derivatives.tolist(),
        "output_derivatives": output,
        "scope": "local implicit response with OCV curves fixed; not parameter identifiability",
    }


def _battery_soh_inputs(spec, state=None):
    from chemworld.physchem.battery_models import F, _energy_bins
    from chemworld.physchem.data_conflicts import DataSourceProvenance

    if getattr(spec, "plating", None) is not None:
        raise ValueError("lithium plating automatic equilibrium-SOH/SOC mapping is unsupported")
    if getattr(spec, "model_family", None) == "split-ocvr-isothermal-v1":
        state = spec.initial_state() if state is None else state
        curves = []
        lithium_C = 0.0
        for electrode in ("negative", "positive"):
            key = electrode + ":Li"
            if key not in state.amounts_mol:
                raise ValueError("active electrode inventory is missing")
            amount = _finite(state.amounts_mol[key])
            capacity = getattr(spec, electrode + "_capacity_C")
            low = getattr(spec, electrode + "_minimum_fraction")
            high = getattr(spec, electrode + "_maximum_fraction")
            if not low * (capacity / F) <= amount <= high * (capacity / F):
                raise ValueError("active electrode inventory is outside its physical capacity")
            lithium_C += F * amount
            ocv = getattr(spec, electrode + "_ocv")
            curves.append(
                replace(
                    ocv,
                    stoichiometry_domain=(
                        max(low, ocv.stoichiometry_domain[0]),
                        min(high, ocv.stoichiometry_domain[1]),
                    ),
                )
            )
        return ElectrodeSOHSpec(
            *curves,
            spec.negative_capacity_C,
            spec.positive_capacity_C,
            spec.minimum_voltage_V,
            spec.maximum_voltage_V,
            lithium_C,
        )
    if getattr(spec, "model_family", None) == "lead-acid-loqs-isothermal-v1":
        raise ValueError("lead-acid LOQS equilibrium-SOH/SOC mapping is unsupported")
    if getattr(spec, "model_family", None) == "composite-dfn-isothermal-v1":
        raise ValueError("composite DFN automatic equilibrium-SOH/SOC mapping is unsupported")
    if getattr(spec, "model_family", None) == "newman-tobias-isothermal-v1":
        raise ValueError("Newman-Tobias equilibrium-SOH mapping is unsupported")
    if getattr(spec, "model_family", None) == "msmr-spm-isothermal-v1":
        raise ValueError("MSMR equilibrium-SOH mapping is unsupported")
    if getattr(spec, "model_family", None) == "spm-spatial-thermal-v1":
        raise ValueError("spatial thermal automatic equilibrium-SOH mapping is unsupported")
    if getattr(spec, "model_family", None) == "composite-spm-isothermal-v1":
        raise ValueError("composite automatic equilibrium-SOH mapping is unsupported")
    if getattr(spec, "model_family", None) == "half-cell-dfn-isothermal-v1":
        raise ValueError("DFN half-cell automatic equilibrium-SOH mapping is unsupported")
    if getattr(spec, "model_family", None) == "half-cell-spm-isothermal-v1":
        raise ValueError("finite half-cell dynamics have no automatic equilibrium-SOH mapping")
    for electrode in ("negative", "positive"):
        ocp = getattr(spec, electrode + "_ocp", None)
        if ocp is not None and ocp.mode != "single":
            raise ValueError(
                "equilibrium SOH does not accept operational OCP branches; "
                "supply explicit equilibrium ElectrodeOCV references to ElectrodeSOHSpec"
            )
    state = spec.initial_state() if state is None else state
    li = 0.0
    for _, particle, keys in _energy_bins(spec):
        for key, volume in zip(keys, particle.mesh.cell_volumes_m3, strict=True):
            amount = _finite(state.amounts_mol[key])
            if not 0 <= amount <= volume * particle.maximum_concentration_mol_m3:
                raise ValueError("active electrode inventory is outside its physical capacity")
            li += F * amount
    source = InputSource(
        DataSourceProvenance(
            "native-explicit-linear-ocv", 0, "Explicit linear native battery option", "1", "CC0-1.0"
        ),
        "synthetic",
    )

    def curve(name):
        existing = getattr(spec, name + "_ocv")
        return (
            existing
            if existing is not None
            else ElectrodeOCV(
                "native-" + name,
                source,
                (getattr(spec, name + "_ocv_intercept_V"), getattr(spec, name + "_ocv_slope_V")),
                stoichiometry_domain=(spec.surface_margin, 1 - spec.surface_margin),
            )
        )

    return ElectrodeSOHSpec(
        curve("negative"),
        curve("positive"),
        F * spec.negative.capacity_mol,
        F * spec.positive.capacity_mol,
        spec.minimum_voltage_V,
        spec.maximum_voltage_V,
        li,
    )


def soh_from_battery(spec, state=None, *, policy=DEFAULT_ISOLATION_POLICY, sensitivity=True):
    """Read active Li totals for a single-valued equilibrium battery declaration.

    Operational OCP branches are not implicitly replaced by their midpoint or a
    chosen history. A separate explicit equilibrium-reference SOH declaration
    remains available through :class:`ElectrodeSOHSpec`.
    """
    return solve_electrode_soh(
        _battery_soh_inputs(spec, state), policy=policy, sensitivity=sensitivity
    )


def battery_spec_at_soc(spec, result, soc, *, branch=None):
    """Prepare a NEW initial config. Never resets a live inventory, clock or device."""
    from chemworld.physchem.battery_models import F

    if getattr(spec, "plating", None) is not None:
        raise ValueError("lithium plating automatic equilibrium-SOH/SOC mapping is unsupported")
    if getattr(spec, "model_family", None) == "lead-acid-loqs-isothermal-v1":
        raise ValueError("lead-acid LOQS equilibrium-SOH/SOC mapping is unsupported")
    if getattr(spec, "model_family", None) == "composite-dfn-isothermal-v1":
        raise ValueError("composite DFN automatic equilibrium-SOH/SOC mapping is unsupported")
    if getattr(spec, "model_family", None) == "newman-tobias-isothermal-v1":
        raise ValueError("Newman-Tobias equilibrium-SOH mapping is unsupported")
    if getattr(spec, "model_family", None) == "msmr-spm-isothermal-v1":
        raise ValueError("MSMR equilibrium-SOH mapping is unsupported")
    if getattr(spec, "model_family", None) == "spm-spatial-thermal-v1":
        raise ValueError("spatial thermal automatic SOC configuration mapping is unsupported")
    if getattr(spec, "model_family", None) == "composite-spm-isothermal-v1":
        raise ValueError("composite automatic SOC configuration mapping is unsupported")
    if getattr(spec, "model_family", None) == "half-cell-dfn-isothermal-v1":
        raise ValueError("DFN half-cell automatic equilibrium-SOH mapping is unsupported")
    if getattr(spec, "model_family", None) == "half-cell-spm-isothermal-v1":
        raise ValueError("finite half-cell dynamics have no automatic SOC configuration mapping")
    split = getattr(spec, "model_family", None) == "split-ocvr-isothermal-v1"
    negative_capacity = spec.negative_capacity_C if split else F * spec.negative.capacity_mol
    positive_capacity = spec.positive_capacity_C if split else F * spec.positive.capacity_mol
    if (
        not isinstance(result.spec, ElectrodeSOHSpec)
        or not np.isclose(
            result.spec.negative_capacity_C, negative_capacity, rtol=1e-12, atol=0.0
        )
        or not np.isclose(
            result.spec.positive_capacity_C, positive_capacity, rtol=1e-12, atol=0.0
        )
    ):
        raise ValueError("SOH capacities do not match this full battery")
    # Capacity equality alone is insufficient: the same two OCV functions and
    # voltage limits must describe the battery being prepared.
    actual = _battery_soh_inputs(spec)
    if (
        actual.negative_ocv != result.spec.negative_ocv
        or actual.positive_ocv != result.spec.positive_ocv
        or actual.minimum_voltage_V != result.spec.minimum_voltage_V
        or actual.maximum_voltage_V != result.spec.maximum_voltage_V
    ):
        raise ValueError("SOH OCV/window does not match this battery")
    state = result.at_soc(soc, branch=branch)
    return replace(spec, initial_negative_fraction=state["x"], initial_positive_fraction=state["y"])
