"""Import the pure-fluid EOS and its explicitly separated phase initial guesses."""

from collections.abc import Mapping
from typing import Any

from chemworld.data.helmholtz import load_helmholtz_fluid
from chemworld.physchem.helmholtz_solvers import HelmholtzAncillary, HelmholtzPhaseModel


def load_helmholtz_phase_model(data: Mapping[str, Any], *, source: str,
                               eos_index: int | None = None) -> HelmholtzPhaseModel:
    fluid = load_helmholtz_fluid(data, source=source, eos_index=eos_index)
    try:
        critical = data["STATES"]["critical"]
        for key, unit in (("T_units", "K"), ("p_units", "Pa"), ("rhomolar_units", "mol/m^3")):
            if critical.get(key, unit) != unit:
                raise ValueError("Helmholtz critical units must use native SI")
        ancillaries = {}
        for name in ("rhoL", "rhoV", "pS", "pL", "pV"):
            if name not in data["ANCILLARIES"]:
                continue
            row = data["ANCILLARIES"][name]
            ancillaries[name] = HelmholtzAncillary(
                row["type"], row["T_r"], row["reducing_value"], row["Tmin"], row["Tmax"],
                tuple(row["n"]), tuple(row["t"]), row.get("using_tau_r", False))
        return HelmholtzPhaseModel(fluid, critical["T"], critical["rhomolar"], critical["p"],
                                    ancillaries)
    except KeyError as exc:
        raise ValueError(f"incomplete Helmholtz phase declaration: {exc.args[0]}") from exc
