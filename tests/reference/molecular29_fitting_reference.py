"""Independent PhasePy PR coexistence for the prospectively selected Rolo data.

This module imports no ChemWorld scientific implementation. The small constants
are factual HEOS entries distributed by chemicals; measured labels are from the
original Rolo tables and are not used to determine these bulk endpoints.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import monotonic

import numpy as np
import phasepy
from phasepy.equilibrium import bubblePy
from scipy.optimize import brentq

DATA = Path(__file__).with_name("data") / "rolo2002_surface_tension.json"
CONSTANTS = {
    "n-heptane": {"CAS": "142-82-5", "Tc_K": 540.2, "Pc_Pa": 2735730., "omega": .349},
    "n-decane": {"CAS": "124-18-5", "Tc_K": 617.7, "Pc_Pa": 2103000., "omega": .4884},
    "n-hexadecane": {"CAS": "544-76-3", "Tc_K": 722.1, "Pc_Pa": 1479850., "omega": .749},
}


def source_model(ids):
    components = [phasepy.component(name=name, Tc=CONSTANTS[name]["Tc_K"],
        Pc=CONSTANTS[name]["Pc_Pa"]/1e5, w=CONSTANTS[name]["omega"]) for name in ids]
    mixture = components[0]
    for component in components[1:]:
        mixture = mixture+component
    return phasepy.preos(mixture)


def endpoints(observation):
    ids = observation["component_ids"]
    x = np.array(observation["liquid_mole_fractions"])
    temperature = observation["temperature_K"]
    model = source_model(ids)
    if len(ids) == 1:
        pressure, vl, vv = (float(np.asarray(v).ravel()[0]) for v in model.psat(temperature))
        # psat returns volumes from an earlier pressure iterate for these low-P
        # alkanes. Refine on source pressure and fugacity, never native output.
        initial = {"pressure_bar": pressure, "vl_cm3_mol": vl, "vv_cm3_mol": vv}
        b = float(np.asarray(model.b).ravel()[0])
        a = float(np.asarray(model.a_eos(temperature)).ravel()[0])
        rt = 83.14*temperature

        def pair(logp):
            p = float(np.exp(logp))
            volumes = [brentq(lambda v: float(np.asarray(
                model.pressure(temperature, v)).ravel()[0])-p, low, high,
                xtol=5e-14, rtol=1e-15) for low, high in
                ((max(b*1.00001, .9*vl), 1.1*vl), (.5*rt/p, 2*rt/p))]
            fugacities = [float(np.asarray(model._logfug_aux(
                p*v/rt, a*p/rt**2, b*p/rt)).ravel()[0]) for v in volumes]
            return volumes, fugacities

        logp = brentq(lambda v: float(np.diff(pair(v)[1])[0]),
                       np.log(.5*pressure), np.log(2*pressure), xtol=1e-12)
        (vl, vv), (liquid_mu, vapor_mu) = pair(logp)
        pressure = float(np.exp(logp))
        y = x
        source_error = None
        liquid_pressure = float(np.asarray(model.pressure(temperature, vl)).ravel()[0])
        vapor_pressure = float(np.asarray(model.pressure(temperature, vv)).ravel()[0])
        mu_error = abs(liquid_mu-vapor_mu)
    else:
        initial = None
        psats = np.array([float(np.asarray(source_model([name]).psat(temperature)[0]).ravel()[0])
                          for name in ids])
        p_guess = float(x@psats)
        y_guess = x*psats/p_guess
        result = bubblePy(y_guess, p_guess, x, temperature, model, full_output=True)
        pressure, vl, vv = float(result.P), float(result.v1), float(result.v2)
        y = np.array(result.Y)
        source_error = float(result.error)
        liquid_mu = model.logfugef(x, temperature, pressure, "L", v0=vl)[0]+np.log(x)
        vapor_mu = model.logfugef(y, temperature, pressure, "V", v0=vv)[0]+np.log(y)
        mu_error = float(np.max(abs(liquid_mu-vapor_mu)))
        liquid_pressure, vapor_pressure = (float(model.pressure(z, v, temperature))
                                            for z, v in ((x, vl), (y, vv)))
    pressure_error = max(abs(liquid_pressure-pressure), abs(vapor_pressure-pressure))
    return {"observation_id": observation["observation_id"], "temperature_K": temperature,
            "component_ids": ids, "pressure_Pa": pressure*1e5,
            "liquid_mole_fractions": x.tolist(), "vapor_mole_fractions": y.tolist(),
            "liquid_density_mol_m3": 1e6/vl, "vapor_density_mol_m3": 1e6/vv,
            "source_mu_error_RT": mu_error, "source_pressure_error_Pa": pressure_error*1e5,
            "source_reported_error": source_error,
            "source_initial_unpolished": initial,
            "success": bool(mu_error <= 2e-7
                            and pressure_error*1e5 <= max(1e-5, pressure*1e5*1e-7)),
            "oma": float(model.oma), "omb": float(model.omb),
            "kappa": np.asarray(model.k).ravel().tolist(), "R_J_mol_K": 8.314}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("source outputs are create-only")
    args.output.mkdir(parents=True)
    dataset = json.loads(DATA.read_text())
    started = monotonic()
    rows = []
    for index, observation in enumerate(dataset["observations"], 1):
        try:
            row = endpoints(observation)
        except Exception as exc:
            row = {"observation_id": observation["observation_id"], "success": False,
                   "exception": f"{type(exc).__name__}: {exc}"}
        rows.append(row)
        elapsed = monotonic()-started
        print(f"F source endpoints {index}/60 success={row['success']} "
              f"rate={index/max(elapsed,1e-9):.2f}/s ETA={elapsed/index*(60-index):.1f}s",
              flush=True)
    summary = {"scope": "source-only bulk endpoints; no native SGT or measured-label fitting",
               "constants_source": "chemicals HEOS Tc/Pc/omega factual entries",
               "constants": CONSTANTS, "EOS": "PhasePy unshifted PR; classical qmr EOS kij=0",
               "units": rows, "total": len(rows), "passed": sum(r["success"] for r in rows),
               "failed": sum(not r["success"] for r in rows), "elapsed_s": monotonic()-started}
    (args.output/"summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")


if __name__ == "__main__":
    main()
