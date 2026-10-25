"""Actual teqp source models/critical equations; no native solution initializes a reference."""

import json
from pathlib import Path

import numpy as np
import teqp
from scipy.optimize import least_squares

R = 8.31446261815324


def source_declaration(names):
    directory = Path(teqp.get_datapath()) / "dev"
    filenames = {"Propane": "n-Propane"}
    fluids = [
        json.loads((directory / "fluids" / (filenames.get(name, name) + ".json")).read_text())
        for name in names
    ]
    cas = {f["INFO"]["CAS"] for f in fluids}
    pairs = [
        p
        for p in json.loads((directory / "mixtures/mixture_binary_pairs.json").read_text())
        if p["CAS1"] in cas and p["CAS2"] in cas
    ]
    functions = {p.get("function", "") for p in pairs}
    departures = [
        p
        for p in json.loads((directory / "mixtures/mixture_departure_functions.json").read_text())
        if p["Name"] in functions
    ]
    return {
        "fluids": fluids,
        "binary_pairs": pairs,
        "departure_functions": departures,
        "gas_constant_J_mol_K": R,
        "gas_constant_convention": "common_constant_override",
    }


def source_model(names):
    return teqp.build_multifluid_model(list(names), teqp.get_datapath(), flags={"Rmodel": "CODATA"})


def source_critical(model, x, bounds):
    x = np.asarray(x)
    tr, rhor = model.get_Tr(x), model.get_rhor(x)
    calls = []

    def residual(y):
        t, rho = np.exp(y)
        conditions = np.array(model.get_criticality_conditions(t, rho * x))
        result = conditions * np.array([rho / (R * t), rho * rho / (R * t)])
        calls.append({"T": float(t), "rho": float(rho), "residual": result.tolist()})
        return result

    root = least_squares(
        residual,
        np.log([tr, rhor]),
        bounds=np.log(np.asarray(bounds).T),
        xtol=1e-12,
        ftol=1e-12,
        gtol=1e-12,
        max_nfev=300,
    )
    return {
        "success": bool(root.success and max(abs(root.fun)) <= 2e-8),
        "temperature_K": float(np.exp(root.x[0])),
        "density_mol_m3": float(np.exp(root.x[1])),
        "residual": root.fun.tolist(),
        "evaluations": calls,
        "message": root.message,
    }


def source_response(model, t, c):
    c = np.asarray(c, dtype=float)
    rho = float(sum(c))
    x = c / rho
    h = np.asarray(model.build_Psi_Hessian_autodiff(t, c))
    eigen, vectors = np.linalg.eigh(h * rho / (R * t))
    mode = vectors[:, 0]
    if mode[np.argmax(abs(mode))] < 0:
        mode = -mode
    # Independent finite differences of actual source AD Hessians. Two steps
    # retain numerical consistency; this is not the native analytic derivative.
    fourth = []
    relaxed = []
    for step in (1e-4, 5e-5):

        def hessian(s):
            return (
                np.asarray(model.build_Psi_Hessian_autodiff(t, c + rho * s * mode)) * rho / (R * t)
            )

        plus, minus = hessian(step), hessian(-step)
        raw = float(mode @ (plus - 2 * h * rho / (R * t) + minus) @ mode / step**2)
        b = (plus - minus) @ mode / (2 * step)
        projection = vectors[:, 1:].T @ b
        eff = raw - 3 * float(sum(projection**2 / eigen[1:])) if len(c) > 1 else raw
        fourth.append(raw)
        relaxed.append(eff)
    return {
        "hessian": h.tolist(),
        "scaled_eigenvalues": eigen.tolist(),
        "mode": mode.tolist(),
        "conditions": np.asarray(model.get_criticality_conditions(t, c)).tolist(),
        "pressure_Pa": float(rho * R * t * (1 + model.get_Ar01(t, rho, x))),
        "fourth_steps": fourth,
        "relaxed_fourth_steps": relaxed,
    }
