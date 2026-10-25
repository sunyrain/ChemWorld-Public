"""Actual released modular DFN, independently mapped synthetic two-phase materials."""

import numpy as np
from battery_dfn_composite_cases import ATOL, RTOL, UNITS, declared_phases


def phase_prefixes(data, limit=None):
    phases = declared_phases(data, limit)
    if limit == "merged-dfn":
        return [(phases[0], "", ""), (phases[1], "", "")]
    return [
        (phases[0], "Primary: ", "primary "),
        (phases[1], "Secondary: ", "secondary "),
        (phases[2], "", ""),
    ]


def build_reference(data, grid, *, limit=None):
    import pybamm

    if pybamm.__version__ != "25.4.2":
        raise RuntimeError("fixed actual composite DFN source requires PyBaMM25.4.2")
    single = limit == "merged-dfn"
    spm = limit == "composite-spm"

    def finite_source(variables):
        current = variables["Current [A]"]
        requested = current + data["maximum_current_A"] * (
            variables["Voltage [V]"] - pybamm.InputParameter("composite DFN voltage target")
        )
        return current - pybamm.maximum(
            -data["maximum_current_A"], pybamm.minimum(data["maximum_current_A"], requested)
        )

    cv = data["waveform"][0][2] == "V"
    options = {
        "particle phases": ("1", "1") if single else ("2", "1"),
        "surface form": "algebraic",
        "open-circuit potential": "single",
        "particle": "Fickian diffusion",
        "thermal": "isothermal",
        "SEI": "none",
        "lithium plating": "none",
        "contact resistance": "false",
        "calculate discharge energy": "true",
        **({"operating mode": finite_source} if cv else {}),
    }
    model = (pybamm.lithium_ion.SPM if spm else pybamm.lithium_ion.DFN)(options)
    parameters = pybamm.ParameterValues("Chen2020" if single else "Chen2020_composite")
    updates = {
        "Electrode height [m]": data["area_m2"],
        "Electrode width [m]": 1.0,
        "Number of electrodes connected in parallel to make a cell": 1.0,
        "Negative electrode thickness [m]": data["lengths_m"][0],
        "Separator thickness [m]": data["lengths_m"][1],
        "Positive electrode thickness [m]": data["lengths_m"][2],
        "Initial concentration in electrolyte [mol.m-3]": data["initial_concentration_mol_m3"],
        "Electrolyte diffusivity [m2.s-1]": data["diffusivity_m2_s"],
        "Electrolyte conductivity [S.m-1]": data["conductivity_S_m"],
        "Cation transference number": data["cation_transference"],
        "Thermodynamic factor": data["thermodynamic_factor"],
        "Initial temperature [K]": data["temperature_K"],
        "Ambient temperature [K]": data["temperature_K"],
        "Reference temperature [K]": data["temperature_K"],
        "Lower voltage cut-off [V]": data["minimum_voltage_V"],
        "Upper voltage cut-off [V]": data["maximum_voltage_V"],
        "Current function [A]": "[input]",
    }
    for region, name in enumerate(("Negative", "Separator", "Positive")):
        stem = name if name == "Separator" else name + " electrode"
        updates[stem + " porosity"] = data["porosities"][region]
        updates[stem + " Bruggeman coefficient (electrolyte)"] = data["bruggeman"][region]
        if region != 1:
            updates[stem + " conductivity [S.m-1]"] = data["solid_conductivity_S_m"][region // 2]
            updates[stem + " Bruggeman coefficient (electrode)"] = 0.0
    for value, prefix, label in phase_prefixes(data, limit):
        domain = value["electrode"]
        title = domain.capitalize()
        region = 0 if domain == "negative" else 2

        def compensated_ocv(theta, coefficients=value["polynomial"]):
            return sum(c * theta**i for i, c in enumerate(coefficients)) - 1e-6 * (
                1 / theta + 1 / (theta - 1)
            )

        def exchange(
            c_e,
            c_s,
            c_max,
            temperature,
            scale=value["exchange_current_scale_A_m2"],
            reference=data["initial_concentration_mol_m3"],
        ):
            theta = c_s / c_max
            return scale * pybamm.sqrt(c_e / reference * theta * (1 - theta))

        updates.update(
            {
                prefix + f"{title} particle radius [m]": value["radius_m"],
                prefix + f"{title} particle diffusivity [m2.s-1]": value["diffusivity_m2_s"],
                prefix + f"Maximum concentration in {domain} electrode [mol.m-3]": value[
                    "maximum_concentration_mol_m3"
                ],
                prefix + f"Initial concentration in {domain} electrode [mol.m-3]": value[
                    "maximum_concentration_mol_m3"
                ]
                * value["initial_fraction"],
                prefix + f"{title} electrode active material volume fraction": value[
                    "active_volume_m3"
                ]
                / (data["area_m2"] * data["lengths_m"][region]),
                prefix + f"{title} electrode exchange-current density [A.m-2]": exchange,
                prefix + f"{title} electrode OCP [V]": compensated_ocv,
                prefix + f"{title} electrode OCP entropic change [V.K-1]": 0.0,
                prefix + f"{title} electrode charge transfer coefficient": 0.5,
            }
        )
        key = f"qualification:{value['phase_id']}:"
        local = f"{title} {label}particle "
        surface = model.variables[
            (f"X-averaged {domain} {label}particle " if spm else local) + "surface concentration"
        ]
        average = model.variables[
            (
                f"Average {domain} {label}particle concentration"
                if spm
                else f"R-averaged {domain} {label}particle concentration"
            )
        ]
        jname = (f"X-averaged {domain}" if spm else title) + f" electrode {label}"
        j = model.variables[jname + "interfacial current density [A.m-2]"]
        model.variables.update(
            {
                key + "surface": surface,
                key + "average": average,
                key + "profile": model.variables[
                    (f"X-averaged {domain} {label}particle " if spm else local)
                    + "concentration [mol.m-3]"
                ],
                key + "j": j,
                key + "q": j
                * 3
                * value["active_volume_m3"]
                / value["radius_m"]
                / (1 if spm else grid["x"][region]),
                key + "j0": model.variables[jname + "exchange current density [A.m-2]"],
                key + "eta": model.variables[jname + "reaction overpotential [V]"],
            }
        )
        model.events.extend(
            [
                pybamm.Event(
                    f"qualification:phase:{value['phase_id']}:depleted",
                    pybamm.min(surface) - data["surface_margin"],
                ),
                pybamm.Event(
                    f"qualification:phase:{value['phase_id']}:saturated",
                    1 - data["surface_margin"] - pybamm.max(surface),
                ),
            ]
        )
    if not spm:
        model.events.append(
            pybamm.Event(
                "qualification:salt-floor",
                pybamm.min(model.variables["Electrolyte concentration [mol.m-3]"])
                - data["minimum_concentration_mol_m3"],
            )
        )
    parameters.update(updates, check_already_exists=False)
    points = {}
    for variable in model.default_var_pts:
        name = str(variable)
        if name.startswith("r_"):
            points[variable] = grid["r"]
        elif name in ("x_n", "x_s", "x_p"):
            points[variable] = grid["x"][("x_n", "x_s", "x_p").index(name)]
        else:
            points[variable] = 3
    sim = pybamm.Simulation(
        model,
        parameter_values=parameters,
        var_pts=points,
        solver=pybamm.IDAKLUSolver(rtol=RTOL, atol=ATOL, root_tol=1e-6, root_method="lm"),
    )
    return sim


def build_metadata(sim):
    import pybamm

    sim.build()
    model = sim.built_model
    layout = []
    for variable, slices in model.y_slices.items():
        if isinstance(variable, pybamm.Concatenation):
            continue
        layout.append(
            {
                "name": variable.name,
                "indices": [i for sl in slices for i in range(sl.start, sl.stop)],
                "scale": float(sim.parameter_values.process_symbol(variable.scale).evaluate()),
                "reference": float(
                    sim.parameter_values.process_symbol(variable.reference).evaluate()
                ),
                "domains": variable.domains,
            }
        )
    return {
        "state_count": int(model.len_rhs_and_alg),
        "rhs_size": int(model.len_rhs),
        "algebraic_size": int(model.len_alg),
        "termination_event_names": [
            e.name for e in model.events if e.event_type == pybamm.EventType.TERMINATION
        ],
        "rhs": [v.name for v in model.rhs],
        "algebraic": [v.name for v in model.algebraic],
        "state_layout": layout,
        "default_var_pts": {str(k): v for k, v in sim._var_pts.items()},
        "qualified_fields": {
            k: {"size": int(v.size), "domains": v.domains}
            for k, v in model.variables.items()
            if k.startswith("qualification:")
        },
    }


def source_time(data, grid, *, limit=None):
    progress = {"completed_segments": [], "stage": "build", "current_time_s": 0.0}
    try:
        return _source_time(data, grid, limit, progress)
    except Exception as exc:
        exc.partial_history = progress
        raise


def _source_time(data, grid, limit, progress):
    import pybamm

    sim = build_reference(data, grid, limit=limit)
    metadata = build_metadata(sim)
    segments = progress["completed_segments"]
    spatial = limit != "composite-spm"
    for index, (duration, target, unit) in enumerate(data["waveform"]):
        start = 0.0 if sim.solution is None else float(sim.solution.t[-1])
        times = np.linspace(0.0, duration, data["samples_per_segment"])
        inputs = {"composite DFN voltage target" if unit == "V" else "Current function [A]": target}
        progress.update(
            stage="integrate",
            active_segment={
                "index": index,
                "start_time_s": start,
                "duration_s": duration,
                "target": target,
                "unit": unit,
                "requested_end_time_s": start + duration,
            },
        )
        solution = sim.step(duration, save=False, t_eval=times, t_interp=times, inputs=inputs)
        returned = np.asarray(solution.t)
        progress.update(stage="extract", current_time_s=float(returned[-1]))
        record = {
            "times_s": returned.tolist(),
            "end_time_s": float(returned[-1]),
            "termination": solution.termination,
            "raw_profiles": {},
            "regions": {},
        }
        progress["active_segment"].update(returned_end_time_s=float(returned[-1]), record=record)

        count = len(returned)

        def entries(name, solution=solution, count=count):
            values = np.asarray(solution[name].entries)
            if values.shape[-1] != count:
                raise ValueError("source output time dimension mismatch: " + name)
            return np.moveaxis(values, -1, 0)

        def scalar(name, entries=entries, count=count):
            result = entries(name)
            if result.size != count:
                raise ValueError("source scalar has unresolved spatial shape: " + name)
            return result.reshape(count)

        record["values"] = np.column_stack(
            [
                scalar("Voltage [V]"),
                scalar("Current [A]"),
                3600 * scalar("Discharge capacity [A.h]"),
                3600 * scalar("Discharge energy [W.h]"),
            ]
        ).tolist()
        for phase, _, label in phase_prefixes(data, limit):
            name = phase["phase_id"]
            domain = phase["electrode"]
            key = f"qualification:{name}:"
            nx = grid["x"][0 if domain == "negative" else 2] if spatial else 1
            nr = grid["r"]
            mesh = sim.mesh[domain + " " + label + "particle"]
            radial = entries(key + "profile")
            if radial.shape != ((len(returned), nr, nx) if spatial else (len(returned), nr)):
                raise ValueError(
                    f"source original radial/x field shape mismatch {name}: {radial.shape}"
                )
            radial = radial.transpose(0, 2, 1) if spatial else radial[:, None, :]
            values = {
                k: entries(key + k).reshape(len(returned), nx)
                for k in ("surface", "average", "j", "q", "j0", "eta")
            }
            edges = np.asarray(mesh.edges) / phase["radius_m"]
            shell_fraction = np.diff(edges**3)
            values["shell_amounts_mol"] = (
                radial * phase["active_volume_m3"] / nx * shell_fraction[None, None, :]
            )
            values["theta"] = radial / phase["maximum_concentration_mol_m3"]
            xmesh = sim.mesh[domain + " electrode"]
            record["raw_profiles"][name] = {
                "electrode": domain,
                "phase": phase,
                "x_nodes_m": np.asarray(xmesh.nodes).tolist()
                if spatial
                else [float(np.mean(xmesh.nodes))],
                "r_nodes_m": np.asarray(mesh.nodes).tolist(),
                "r_edges_m": np.asarray(mesh.edges).tolist(),
                **{k: v.tolist() for k, v in values.items()},
            }
        if spatial:
            salt = entries("Electrolyte concentration [mol.m-3]")
            offset = 0
            for region, (name, count) in enumerate(
                zip(("negative", "separator", "positive"), grid["x"], strict=True)
            ):
                domain = name + " electrode" if name != "separator" else name
                mesh = sim.mesh[domain]
                portion = salt[:, offset : offset + count]
                view = {
                    "nodes_m": np.asarray(mesh.nodes).tolist(),
                    "edges_m": np.asarray(mesh.edges).tolist(),
                    "salt_mol_m3": portion.tolist(),
                    "salt_amounts_mol": (
                        portion
                        * np.diff(mesh.edges)[None, :]
                        * data["area_m2"]
                        * data["porosities"][region]
                    ).tolist(),
                    "electrolyte_potential_V": entries(
                        name.capitalize() + " electrolyte potential [V]"
                    ).tolist(),
                    "electrolyte_face_current_A": (
                        entries(name.capitalize() + " electrolyte current density [A.m-2]")
                        * data["area_m2"]
                    ).tolist(),
                }
                if name != "separator":
                    view["solid_potential_V"] = entries(
                        name.capitalize() + " electrode potential [V]"
                    ).tolist()
                    view["solid_face_current_A"] = (
                        entries(name.capitalize() + " electrode current density [A.m-2]")
                        * data["area_m2"]
                    ).tolist()
                record["regions"][name] = view
                offset += count
        record["source_state_vectors"] = np.asarray(solution.y).T.tolist()
        record["source_initial_rhs"] = (
            np.asarray(
                sim.built_model.concatenated_rhs.evaluate(
                    t=float(returned[0]), y=np.asarray(solution.y[:, 0]), inputs=inputs
                )
            )
            .reshape(-1)
            .tolist()
        )
        record["source_initial_algebraic_residual"] = (
            np.asarray(
                sim.built_model.concatenated_algebraic.evaluate(
                    t=float(returned[0]), y=np.asarray(solution.y[:, 0]), inputs=inputs
                )
            )
            .reshape(-1)
            .tolist()
        )

        record["source_state_layout"] = metadata["state_layout"]
        record["solver"] = {
            "name": "PyBaMM25.4.2 IDAKLUSolver",
            "rtol": RTOL,
            "atol": ATOL,
            "root_tol": 1e-6,
            "integration_time_s": float(solution.integration_time.value),
        }
        record["source_event_values_at_endpoint"] = {}
        for event in sim.built_model.events:
            if event.event_type == pybamm.EventType.TERMINATION:
                value = event.expression.evaluate(
                    t=float(returned[-1]), y=np.asarray(solution.y[:, -1]), inputs=inputs
                )
                record["source_event_values_at_endpoint"][event.name] = (
                    np.asarray(value).reshape(-1).tolist()
                )
        segments.append(record)
        if solution.termination != "final time":
            break
    return {
        "implementation": "reference",
        "family": "SPM" if not spatial else "DFN",
        "unit": data["id"],
        "inputs": data,
        "grid": grid,
        "limit": limit,
        "source_version": "25.4.2",
        "source_build": metadata,
        "segments": segments,
        "response_units": UNITS,
        "faraday_C_mol": float(pybamm.constants.F.evaluate()),
        "gas_constant_J_mol_K": float(pybamm.constants.R.evaluate()),
        "tolerances": {"rtol": RTOL, "atol": ATOL, "root_tol": 1e-6},
    }
