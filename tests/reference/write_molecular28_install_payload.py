"""Development-only JSON fixtures; installed probe imports no reference engines."""

import argparse
import json
from pathlib import Path

import numpy as np
from phasepy.equilibrium import vlle

from reference.molecular28_general_cases import native_spec
from reference.molecular28_general_reference import (
    phasepy_oracle,
    thermo_boundary,
    thermo_fixed_volume,
    thermo_state,
    vlle_state,
)
from reference.molecular28_solid_reference import cases, reference
from reference.run_molecular28_general import boundary_seed, boundary_spec
from reference.run_molecular28_solid import declaration


def payload():
    source = vlle_state(300.0, 3e6)
    eos = phasepy_oracle()
    potentials = []
    for amounts, density, state in zip(
        source["amounts"], source["densities"], ("L", "L", "V"), strict=True
    ):
        x = np.asarray(amounts) / sum(amounts)
        phi, _ = eos.logfugef(x, 300.0, 30.0, state, v0=1e6 / density)
        potentials.append(np.log(x) + phi)
    original_potentials = np.asarray(potentials)
    refined = vlle(
        np.array([0.005, 0.005, 0.99]),
        np.array([0.1, 0.89, 0.01]),
        np.array([0.98, 0.01, 0.01]),
        np.array([0.3, 0.3, 0.4]),
        300.0,
        30.0,
        eos,
        K_tol=1e-20,
        full_output=True,
    )
    potentials = []
    for x, volume, state in zip(refined.X, refined.v, refined.states, strict=True):
        phi, _ = eos.logfugef(x, 300.0, 30.0, state, v0=volume)
        potentials.append(np.log(x) + phi)
    anchor = potentials[-1]
    points = []
    for x in (
        (0.3, 0.3, 0.4),
        (1 / 3,) * 3,
        (0.99, 0.005, 0.005),
        (0.005, 0.99, 0.005),
        (0.005, 0.005, 0.99),
    ):
        x = np.asarray(x)
        for state in ("L", "V"):
            phi, volume = eos.logfugef(x, 300.0, 30.0, state)
            distance = float(x @ (np.log(x) + phi - anchor))
            points.append(
                {
                    "composition": x.tolist(),
                    "branch": state,
                    "rho": float(1e6 / volume),
                    "tpd_RT": distance,
                }
            )
    source["common_mu_RT_without_logP"] = anchor.tolist()
    source["source_common_mu_error_RT"] = float(np.max(np.abs(potentials - anchor)))
    source["source_precision_refinement"] = {
        "original_K_tol": 1e-10,
        "refined_K_tol": 1e-20,
        "original_common_mu_RT": original_potentials.tolist(),
        "original_common_mu_error_RT": float(
            np.max(abs(original_potentials - original_potentials[-1]))
        ),
        "refined_compositions": refined.X.tolist(),
        "refined_beta": refined.beta.tolist(),
        "refined_volumes_cm3_mol": list(map(float, refined.v)),
        "source_error_inner": float(refined.error_inner),
        "source_error_outer": float(refined.error_outer),
    }
    source["finite_tpd_points"] = points
    cold = thermo_state((0.6, 0.4), total=0.01, T=300.0, P=6e5)
    bubble = thermo_boundary((0.6, 0.4), "bubble", T=290.0)
    solid_case = next(c for c in cases() if c["id"] == "S-CONSUMER-2")
    return {
        "schema": "chemworld-parent28-installed-probe-v1",
        "vlle": {
            "spec": native_spec("vlle").to_dict(),
            "source": source,
            "feed": {"methane": 0.3, "butane": 0.3, "water": 0.4},
        },
        "thermal": {
            "spec": native_spec("vle").to_dict(),
            "feed": {"propane": 0.006, "butane": 0.004},
            "cold": cold,
            "hot_P": thermo_state((0.6, 0.4), total=0.01, T=320.0, P=6e5),
            "hot_V": thermo_fixed_volume((0.6, 0.4), 320.0, cold["V"], total=0.01),
        },
        "bubble": {
            "spec": boundary_spec("vle", "bubble", (0.6, 0.4)).to_dict(),
            "seed": boundary_seed(bubble, "bubble").to_dict(),
            "source": bubble,
        },
        "solid": {
            "spec": declaration(solid_case).to_dict(),
            "case": solid_case,
            "source": reference(solid_case),
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = payload()
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
    print(
        json.dumps(
            {
                "payload": str(args.output),
                "units": 5,
                "independent_tpd_points": len(data["vlle"]["source"]["finite_tpd_points"]),
            }
        )
    )
