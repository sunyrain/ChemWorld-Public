"""Independent one-dimensional collector integration and static BV oracle.

No production Green map, particle helper, residual assembly or field reader is
imported. The inputs are SI geometry, integrated cell currents and explicit
electrode observations. With q positive on discharge, continuity gives
di_n/dz=-q/dz and di_p/dz=+q/dz; Ohm's law is dphi/dz=-i/K,
where K=sigma*collector_thickness*width [S m].

For piecewise-constant q/dz the currents are piecewise linear and the potentials
are quadratic. ``continuous_centre_V`` and ``continuous_mean_V`` are their actual
point and volume-average observations. ``fv_centre_V`` instead integrates the
piecewise-constant face current on each half cell (source lumped at its centre).
It is the ordinary cell-centred FV potential, not a continuous cell average.
For the negative sheet, mean-FV=q*dz/(6K), centre-FV=q*dz/(8K); the positive
sheet has the opposite signs. The continuous and FV dissipation identities use
their respective mean and centre potentials, so their difference is retained.

The static closure does not time-integrate particles. A supplied outer-shell
fraction can have an explicit flux coefficient theta_s=theta_outer-b*q_electrode,
or b=0 can denote a directly supplied surface observation. These are distinct
input contracts, not interchangeable source/numerical observations. All original
voltage observations remain available; source-style linear boundary observation
can be selected for CV without replacing the physical tab voltage.

Supplying observed surface fractions is a conditional, fixed-state closure
check, not an independent prediction of the particle time evolution. The
material scope here is the declared affine OCP and symmetric one-electron BV
at fixed temperature and electrolyte reference concentration.
"""

from copy import deepcopy
from math import isfinite

import numpy as np
from scipy.optimize import least_squares

F = 96485.33212
R = 8.314462618
OBSERVATIONS = ("fv_centre_V", "continuous_centre_V", "continuous_mean_V")


def _positive(value, name):
    value = float(value)
    if not isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def _vector(value, count, name):
    array = np.asarray(value, dtype=float)
    if array.ndim == 0:
        array = np.full(count, float(array))
    if array.shape != (count,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite and have one value per cell")
    return array.copy()


def _edges(z_edges_m):
    edges = np.asarray(z_edges_m, dtype=float)
    if (
        edges.ndim != 1
        or len(edges) < 3
        or not np.all(np.isfinite(edges))
        or edges[0] != 0
        or np.any(np.diff(edges) <= 0)
    ):
        raise ValueError("z edges must start at zero and increase, with at least two cells")
    return edges


def _integrate_sheet(widths, face_current, conductance, *, boundary, voltage):
    # Both integrations use the exact same face values. The continuous current
    # is linear within a cell; the FV current switches at its centre.
    left, right = face_current[:-1], face_current[1:]
    drops = -widths * (left + right) / (2 * conductance)
    potential = np.r_[0.0, np.cumsum(drops)]
    potential += voltage - potential[0 if boundary == "left" else -1]
    gradient = (right - left) / widths
    centre = potential[:-1] - (left * widths / 2 + gradient * widths**2 / 8) / conductance
    average = potential[:-1] - (left * widths / 2 + gradient * widths**2 / 6) / conductance
    fv_centre = potential[:-1] - left * widths / (2 * conductance)
    continuous_loss = np.sum(widths * (left**2 + left * right + right**2) / (3 * conductance))
    fv_loss = np.sum(widths * (left**2 + right**2) / (2 * conductance))
    return {
        "face_current_A": face_current,
        "face_potential_V": potential,
        "continuous_centre_V": centre,
        "continuous_mean_V": average,
        "fv_centre_V": fv_centre,
        "continuous_joule_W": float(continuous_loss),
        "fv_joule_W": float(fv_loss),
    }


def integrate_collectors(
    q_A,
    *,
    z_edges_m,
    negative_conductance_S_m,
    positive_conductance_S_m,
    negative_tab="left",
    physical_tab_voltage_V=0.0,
):
    """Integrate continuity and Ohm's law for signed, piecewise-constant q/dz.

    q has units A per z cell, not A/m2 or A/m. All face currents are oriented
    left-to-right. The positive tab is fixed on the right; the negative tab is
    a zero-potential gauge on either endpoint. Nonuniform z grids are accepted.
    """
    edges = _edges(z_edges_m)
    widths = np.diff(edges)
    q = _vector(q_A, len(widths), "q_A")
    kn = _positive(negative_conductance_S_m, "negative conductance [S m]")
    kp = _positive(positive_conductance_S_m, "positive conductance [S m]")
    voltage = float(physical_tab_voltage_V)
    if negative_tab not in ("left", "right") or not isfinite(voltage):
        raise ValueError("a finite tab voltage and negative tab left/right are required")
    integral = np.r_[0.0, np.cumsum(q)]
    current = float(integral[-1])
    negative_faces = (current if negative_tab == "left" else 0.0) - integral
    negative = _integrate_sheet(widths, negative_faces, kn, boundary=negative_tab, voltage=0.0)
    positive = _integrate_sheet(widths, integral, kp, boundary="right", voltage=voltage)
    centres = (edges[:-1] + edges[1:]) / 2
    source_voltage = positive["fv_centre_V"][-1] + widths[-1] / 2 * (
        positive["fv_centre_V"][-1] - positive["fv_centre_V"][-2]
    ) / (centres[-1] - centres[-2])
    negative_values = negative["fv_centre_V"]
    if negative_tab == "left":
        negative_extrapolated_tab = negative_values[0] - widths[0] / 2 * (
            negative_values[1] - negative_values[0]
        ) / (centres[1] - centres[0])
    else:
        negative_extrapolated_tab = negative_values[-1] + widths[-1] / 2 * (
            negative_values[-1] - negative_values[-2]
        ) / (centres[-1] - centres[-2])
    return {
        "z_edges_m": edges.copy(),
        "z_centres_m": centres,
        "through_current_A": q,
        "current_A": current,
        "negative": negative,
        "positive": positive,
        "local_voltage_V": {key: positive[key] - negative[key] for key in OBSERVATIONS},
        "physical_tab_voltage_V": voltage,
        "source_boundary_voltage_V": float(source_voltage),
        # PyBaMM's Voltage is the positive-tab boundary value alone. This
        # negative extrapolation is a diagnostic, never subtracted or fitted.
        "negative_extrapolated_tab_voltage_V": float(negative_extrapolated_tab),
        "continuous_joule_W": negative["continuous_joule_W"] + positive["continuous_joule_W"],
        "fv_joule_W": negative["fv_joule_W"] + positive["fv_joule_W"],
    }


def _electrode(data, count, label):
    required = (
        "outer_fraction",
        "interface_area_m2",
        "exchange_current_scale_A_m2",
        "ocv_intercept_V",
        "ocv_slope_V",
    )
    if not set(required) <= set(data):
        raise ValueError(f"{label} electrode lacks explicit static material inputs")
    result = {key: _vector(data[key], count, f"{label}:{key}") for key in required}
    result["surface_flux_coefficient_A_inv"] = _vector(
        data.get("surface_flux_coefficient_A_inv", 0.0), count, f"{label}:surface flux coefficient"
    )
    if (
        np.any(result["outer_fraction"] <= 0)
        or np.any(result["outer_fraction"] >= 1)
        or np.any(result["interface_area_m2"] <= 0)
        or np.any(result["exchange_current_scale_A_m2"] <= 0)
        or np.any(result["surface_flux_coefficient_A_inv"] < 0)
        or np.any(result["ocv_slope_V"] > 0)
    ):
        raise ValueError(
            "static oracle requires physical interiors, positive kinetics and monotone OCV"
        )
    return result


class StaticClosureError(RuntimeError):
    """A failed independent solve retains its raw diagnostic; never a success fallback."""

    def __init__(self, diagnostic):
        super().__init__("independent collector static closure failed")
        self.diagnostic = diagnostic


def solve_static_collector(
    *,
    z_edges_m,
    negative_conductance_S_m,
    positive_conductance_S_m,
    negative,
    positive,
    negative_tab="left",
    temperature_K=298.15,
    control_mode="A",
    target=0.0,
    current_limit_A=1.0,
    series_resistance_ohm=0.0,
    surface_margin=1e-8,
    collector_observation="fv_centre_V",
    voltage_observation="physical-tab",
    residual_tolerance=1e-10,
):
    """Independently solve all local BV equations plus CC/finite-CV closure.

        SciPy least_squares solves N cell currents and one physical terminal voltage.
        q is not clipped to the terminal source limit: zero terminal current can
        support real internal redistribution. Bounds only enforce the explicitly
        declared surface domain. The returned q/V/observations are unmodified solver
        values. A failed residual check raises StaticClosureError with diagnostics.

    Each electrode dictionary has ``outer_fraction``, ``interface_area_m2``,
    ``exchange_current_scale_A_m2``, ``ocv_intercept_V``, ``ocv_slope_V`` and optional
    ``surface_flux_coefficient_A_inv``. Scalars broadcast to all z cells. For a
    directly supplied surface observation the last field must be zero. Positive
    tab potential includes I*series_resistance above the external terminal value;
    the chosen voltage observation also subtracts this same explicit contact drop.
    Solver residuals retain local voltage in V and source current divided by its
    declared limit (dimensionless); neither is a time-integration error estimate.
    """
    edges = _edges(z_edges_m)
    count = len(edges) - 1
    kn = _positive(negative_conductance_S_m, "negative conductance")
    kp = _positive(positive_conductance_S_m, "positive conductance")
    temperature = _positive(temperature_K, "temperature")
    limit = _positive(current_limit_A, "source current limit")
    tolerance = _positive(residual_tolerance, "residual tolerance")
    if (
        control_mode not in ("A", "V")
        or negative_tab not in ("left", "right")
        or collector_observation not in OBSERVATIONS
        or voltage_observation not in ("physical-tab", "source-boundary")
        or not isfinite(target)
        or not isfinite(series_resistance_ohm)
        or series_resistance_ohm < 0
        or not 0 < surface_margin < 0.5
    ):
        raise ValueError("invalid static closure convention or control/domain input")
    if control_mode == "A" and abs(target) > limit:
        raise ValueError("CC target exceeds the declared source current limit")
    electrodes = (_electrode(negative, count, "negative"), _electrode(positive, count, "positive"))
    lower, upper = np.full(count, -np.inf), np.full(count, np.inf)
    for sign, material in zip((1.0, -1.0), electrodes, strict=True):
        theta, coefficient = material["outer_fraction"], material["surface_flux_coefficient_A_inv"]
        for i in range(count):
            if coefficient[i] == 0:
                if not surface_margin <= theta[i] <= 1 - surface_margin:
                    raise ValueError("supplied fixed surface is outside the declared margin")
                continue
            ends = (theta[i] - np.array([surface_margin, 1 - surface_margin])) / (
                sign * coefficient[i]
            )
            lower[i], upper[i] = max(lower[i], min(ends)), min(upper[i], max(ends))
    if np.any(lower >= upper):
        raise ValueError("no joint admissible surface-current interval")

    def evaluate(y):
        q, voltage = y[:-1], float(y[-1])
        current = float(np.sum(q))
        fields = integrate_collectors(
            q,
            z_edges_m=edges,
            negative_conductance_S_m=kn,
            positive_conductance_S_m=kp,
            negative_tab=negative_tab,
            physical_tab_voltage_V=voltage + current * series_resistance_ohm,
        )
        material_fields = {}
        local_equilibrium = np.zeros(count)
        for name, sign, material in zip(
            ("negative", "positive"), (1.0, -1.0), electrodes, strict=True
        ):
            theta = (
                material["outer_fraction"] - sign * q * material["surface_flux_coefficient_A_inv"]
            )
            if np.any(theta <= 0) or np.any(theta >= 1):
                raise ValueError("static trial left the bounded material domain")
            j0 = material["exchange_current_scale_A_m2"] * np.sqrt(theta * (1 - theta))
            eta = (
                2
                * R
                * temperature
                / F
                * np.arcsinh(sign * q / (2 * material["interface_area_m2"] * j0))
            )
            ocv = material["ocv_intercept_V"] + material["ocv_slope_V"] * theta
            local_equilibrium -= sign * (ocv + eta)
            material_fields[name] = {
                "surface_fraction": theta,
                "j0_A_m2": j0,
                "kinetic_overpotential_V": eta,
                "ocv_V": ocv,
            }
        controlled_voltage = (
            voltage
            if voltage_observation == "physical-tab"
            else fields["source_boundary_voltage_V"] - current * series_resistance_ohm
        )
        residual = fields["local_voltage_V"][collector_observation] - local_equilibrium
        return residual, fields, material_fields, controlled_voltage

    # The CV complementarity can be solved more robustly as a fixed active-set
    # enumeration. Every candidate solves a smooth full-cell system independently;
    # only a candidate satisfying its physical inequalities is accepted.
    initial_current = target if control_mode == "A" else 0.0
    initial_q = np.minimum(np.maximum(np.full(count, initial_current / count), lower), upper)
    initial_voltage = float(
        np.mean(
            electrodes[1]["ocv_intercept_V"]
            + electrodes[1]["ocv_slope_V"] * electrodes[1]["outer_fraction"]
            - electrodes[0]["ocv_intercept_V"]
            - electrodes[0]["ocv_slope_V"] * electrodes[0]["outer_fraction"]
        )
    )
    branches = (
        [("CC", target)]
        if control_mode == "A"
        else [("CV", None), ("positive-limit", limit), ("negative-limit", -limit)]
    )
    attempts = []
    for branch, fixed_current in branches:

        def smooth_residual(y, fixed_current=fixed_current):
            residual, _fields, _materials, controlled_voltage = evaluate(y)
            control = (
                (np.sum(y[:-1]) - fixed_current) / limit
                if fixed_current is not None
                else controlled_voltage - target
            )
            return np.r_[residual, control]

        result = least_squares(
            smooth_residual,
            np.r_[initial_q, initial_voltage],
            bounds=(np.r_[lower, -np.inf], np.r_[upper, np.inf]),
            x_scale=np.r_[np.full(count, limit / count), 1.0],
            ftol=1e-13,
            xtol=1e-13,
            gtol=1e-13,
            max_nfev=2000,
        )
        smooth = smooth_residual(result.x)
        _unused, fields, materials, controlled = evaluate(result.x)
        current = float(np.sum(result.x[:-1]))
        admissible = (
            abs(current) <= limit + tolerance
            if branch == "CV"
            else controlled >= target - tolerance
            if branch == "positive-limit"
            else controlled <= target + tolerance
            if branch == "negative-limit"
            else True
        )
        norm = float(np.max(np.abs(smooth)))
        attempts.append(
            {
                "branch": branch,
                "success": bool(result.success),
                "status": int(result.status),
                "message": result.message,
                "nfev": int(result.nfev),
                "residual_max": norm,
                "admissible": bool(admissible),
                "q_A": result.x[:-1].tolist(),
                "physical_terminal_voltage_V": float(result.x[-1]),
            }
        )
        if result.success and norm <= tolerance and admissible:
            return {
                "current_A": current,
                "voltage_V": float(result.x[-1]),
                "controlled_voltage_V": float(controlled),
                "mode": branch,
                "through_current_A": result.x[:-1].copy(),
                "collectors": fields,
                "negative": materials["negative"],
                "positive": materials["positive"],
                "local_residual_V": smooth[:-1].copy(),
                "control_residual": float(smooth[-1]),
                "solver_attempts": attempts,
                "collector_observation": collector_observation,
                "voltage_observation": voltage_observation,
            }
    raise StaticClosureError({"attempts": attempts, "residual_tolerance": tolerance})


def static_inputs_from_case(
    data,
    grid,
    layout,
    *,
    surface_model="shell-flux",
    negative_outer_fraction=None,
    positive_outer_fraction=None,
):
    """Convert authored inputs, without importing production or its case adapter.

    ``shell-flux`` explicitly reconstructs a uniform spherical outer-shell mean
    using the quadratic shell moment. ``given-surface`` uses the supplied surface
    directly (b=0), as needed for raw source surface observations. Neither option
    changes q, the supplied fraction, the source output, or its OCP.
    """
    if surface_model not in ("shell-flux", "given-surface"):
        raise ValueError("declare shell-flux or given-surface")
    nz, nr = grid["z"], grid["r"]
    if type(nz) is not int or type(nr) is not int or nz < 2 or nr < 1:
        raise ValueError("positive radial cells and at least two z cells are required")
    if layout.get("positive_tab") != "right":
        raise ValueError("the positive collector tab must be right")
    result = {
        "z_edges_m": np.linspace(0.0, data["height_m"], nz + 1),
        "negative_conductance_S_m": data["sigma_cc_S_m"]
        * data["collector_thickness_m"]
        * data["width_m"],
        "positive_conductance_S_m": data["sigma_cc_S_m"]
        * data["collector_thickness_m"]
        * data["width_m"],
        "negative_tab": layout["negative_tab"],
        "temperature_K": data["temperature_K"],
        "current_limit_A": data["maximum_current_A"],
        "series_resistance_ohm": data["series_resistance_ohm"],
        "surface_margin": data["surface_margin"],
    }
    for name, supplied in (
        ("negative", negative_outer_fraction),
        ("positive", positive_outer_fraction),
    ):
        particle = deepcopy(data[name])
        if supplied is None:
            authored = np.asarray(data["initial_z_fractions"][name], dtype=float)
            supplied = (
                authored.copy()
                if len(authored) == nz
                else np.interp(
                    (np.arange(nz) + 0.5) / nz, np.linspace(0.0, 1.0, len(authored)), authored
                )
            )
        radius = _positive(particle["radius_m"], "particle radius")
        volume = _positive(particle["active_volume_m3"], "particle active volume")
        diffusivity = _positive(particle["diffusivity_m2_s"], "particle diffusivity")
        cmax = _positive(particle["maximum_concentration_mol_m3"], "particle cmax")
        area = 3 * volume / (radius * nz)
        inner = radius * (1 - 1 / nr)
        mean_r2 = 3 / 5 * (radius**5 - inner**5) / (radius**3 - inner**3)
        distance = (radius**2 - mean_r2) / (2 * radius)
        result[name] = {
            "outer_fraction": supplied,
            "interface_area_m2": area,
            "exchange_current_scale_A_m2": particle["exchange_current_scale_A_m2"],
            "ocv_intercept_V": particle["ocv"][0],
            "ocv_slope_V": particle["ocv"][1],
            "surface_flux_coefficient_A_inv": (
                distance / (F * area * diffusivity * cmax) if surface_model == "shell-flux" else 0.0
            ),
        }
    return result
