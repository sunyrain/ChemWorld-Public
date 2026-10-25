"""Complete ASM1 quality-basis kinetics: empirical quantities, not molecules.

Source: Henze et al. ASM1 (1987), as implemented by WaterTAP 1.8.0. All eight
processes and thirteen components are present. Native molar chemistry and
thermochemistry are not inferred from COD, nitrogen mass or alkalinity.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
from math import isfinite
from typing import Any

COD_COMPONENTS = ("S_I", "S_S", "X_I", "X_S", "X_BH", "X_BA", "X_P")
N_COMPONENTS = ("S_NO", "S_NH", "S_ND", "X_ND")
ASM1_COMPONENTS = (*COD_COMPONENTS, "S_O", *N_COMPONENTS, "S_ALK")
ASM1_REACTIONS = tuple(f"R{i}" for i in range(1, 9))
PARTICULATE_COMPONENTS = frozenset(("X_I", "X_S", "X_BH", "X_BA", "X_P", "X_ND"))
COMPONENT_UNITS = {k: "mol_eq" if k == "S_ALK" else "kg" for k in ASM1_COMPONENTS}
COMPONENT_BASES = {
    **dict.fromkeys(COD_COMPONENTS, "COD"),
    **dict.fromkeys(N_COMPONENTS, "N"),
    "S_O": "O2",
    "S_ALK": "alkalinity_charge_equivalent",
}
EXTENT_BASES = {r: "N" if r in ("R6", "R8") else "COD" for r in ASM1_REACTIONS}


@dataclass(frozen=True)
class ASM1Parameters:
    Y_A: float = 0.24
    Y_H: float = 0.67
    f_p: float = 0.08
    i_xb: float = 0.08
    i_xp: float = 0.06
    mu_A_per_day: float = 0.5
    mu_H_per_day: float = 4.0
    K_S_kg_m3: float = 0.01
    K_OH_kg_m3: float = 0.0002
    K_OA_kg_m3: float = 0.0004
    K_NO_kg_m3: float = 0.0005
    b_H_per_day: float = 0.3
    b_A_per_day: float = 0.05
    eta_g: float = 0.8
    eta_h: float = 0.8
    k_h_per_day: float = 3.0
    K_X: float = 0.1
    K_NH_kg_m3: float = 0.001
    k_a_m3_kg_day: float = 50.0

    def __post_init__(self) -> None:
        if any(isinstance(v, bool) or not isfinite(v) or v < 0 for v in asdict(self).values()):
            raise ValueError("ASM1 parameters require finite nonnegative source values")
        if (
            not 0 < self.Y_H < 1
            or not 0 < self.Y_A < 4.57
            or not 0 <= self.f_p <= 1
            or self.i_xb < self.f_p * self.i_xp
            or any(
                getattr(self, k) <= 0
                for k in (
                    "K_S_kg_m3",
                    "K_OH_kg_m3",
                    "K_OA_kg_m3",
                    "K_NO_kg_m3",
                    "K_NH_kg_m3",
                    "K_X",
                )
            )
        ):
            raise ValueError("ASM1 yields, nitrogen fractions and saturation constants invalid")

    def to_dict(self) -> dict[str, float]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ASM1Parameters:
        if set(value) != {f.name for f in fields(cls)}:
            raise ValueError("ASM1 source parameters require their complete exact declaration")
        return cls(**value)


def asm1_stoichiometry(p: ASM1Parameters) -> dict[str, dict[str, float]]:
    """Rows are original component units per declared reaction extent unit.

    Mass components use kg (COD, O2, or N); alkalinity uses mol-equivalent.
    R6/R8 extents are kg-N; all other extents are kg-COD. The source nitrogen
    conversion is exactly 14 kg/kmol; it is not a fictitious species MW.
    """
    h, a, fp, nb, np_ = p.Y_H, p.Y_A, p.f_p, p.i_xb, p.i_xp
    eq_per_kg_N = 1000 / 14
    return {
        "R1": {
            "S_S": -1 / h,
            "X_BH": 1.0,
            "S_O": -(1 - h) / h,
            "S_NH": -nb,
            "S_ALK": -nb * eq_per_kg_N,
        },
        "R2": {
            "S_S": -1 / h,
            "X_BH": 1.0,
            "S_NO": -(1 - h) / (2.86 * h),
            "S_NH": -nb,
            "S_ALK": ((1 - h) / (2.86 * h) - nb) * eq_per_kg_N,
        },
        "R3": {
            "X_BA": 1.0,
            "S_O": -(4.57 - a) / a,
            "S_NO": 1 / a,
            "S_NH": -nb - 1 / a,
            "S_ALK": (-nb - 2 / a) * eq_per_kg_N,
        },
        "R4": {"X_BH": -1.0, "X_S": 1 - fp, "X_P": fp, "X_ND": nb - fp * np_},
        "R5": {"X_BA": -1.0, "X_S": 1 - fp, "X_P": fp, "X_ND": nb - fp * np_},
        "R6": {"S_NH": 1.0, "S_ND": -1.0, "S_ALK": eq_per_kg_N},
        "R7": {"S_S": 1.0, "X_S": -1.0},
        "R8": {"S_ND": 1.0, "X_ND": -1.0},
    }


def asm1_rates(p: ASM1Parameters, concentrations: Mapping[str, float]) -> dict[str, float]:
    """Eight rates in extent kg/(m3 s), on declared COD or N basis.

    At X_S=X_BH=0 the bounded continuous limit of hydrolysis is zero. This
    explicitly removes a 0/0 expression, not any source process. Ammonia and
    alkalinity do not get extra limitation terms absent from the source model.
    """
    if set(concentrations) != set(ASM1_COMPONENTS) or any(
        isinstance(v, bool) or not isfinite(v) or v < 0 for v in concentrations.values()
    ):
        raise ValueError("ASM1 needs every nonnegative source-basis concentration")
    c = concentrations
    h, a, s, o, no, nh = (c[k] for k in ("X_BH", "X_BA", "S_S", "S_O", "S_NO", "S_NH"))
    oxygen = o / (p.K_OH_kg_m3 + o)
    anoxic = p.K_OH_kg_m3 / (p.K_OH_kg_m3 + o) * no / (p.K_NO_kg_m3 + no)
    denominator = p.K_X * h + c["X_S"]
    hydrolysis = (
        p.k_h_per_day * c["X_S"] * h / denominator * (oxygen + p.eta_h * anoxic)
        if denominator
        else 0.0
    )
    per_day = {
        "R1": p.mu_H_per_day * s / (p.K_S_kg_m3 + s) * oxygen * h,
        "R2": p.mu_H_per_day * s / (p.K_S_kg_m3 + s) * p.eta_g * anoxic * h,
        "R3": p.mu_A_per_day * nh / (p.K_NH_kg_m3 + nh) * o / (p.K_OA_kg_m3 + o) * a,
        "R4": p.b_H_per_day * h,
        "R5": p.b_A_per_day * a,
        "R6": p.k_a_m3_kg_day * c["S_ND"] * h,
        "R7": hydrolysis,
        "R8": hydrolysis * c["X_ND"] / (c["X_S"] + 1e-10),
    }
    rates = {key: value / 86400 for key, value in per_day.items()}
    if any(not isfinite(v) or v < 0 for v in rates.values()):
        raise ArithmeticError("ASM1 source rate is outside its finite nonnegative domain")
    return rates


def asm1_quality(p: ASM1Parameters, concentrations: Mapping[str, float]) -> dict[str, float]:
    """Read-only source projections; X_ND is not a second particulate mass."""
    c = concentrations
    asm1_rates(p, c)
    tkn = (
        sum(c[k] for k in ("S_NH", "S_ND", "X_ND"))
        + p.i_xb * (c["X_BH"] + c["X_BA"])
        + p.i_xp * (c["X_P"] + c["X_I"])
    )
    biodegradable = c["S_S"] + c["X_S"] + (1 - p.f_p) * (c["X_BH"] + c["X_BA"])
    return {
        "COD_kg_O2_m3": sum(c[k] for k in COD_COMPONENTS),
        "TSS_kg_m3": 0.75 * sum(c[k] for k in ("X_I", "X_S", "X_P", "X_BH", "X_BA")),
        "TKN_kg_N_m3": tkn,
        "total_N_kg_N_m3": tkn + c["S_NO"],
        "BOD5_raw_kg_O2_m3": 0.65 * biodegradable,
        "BOD5_effluent_kg_O2_m3": 0.25 * biodegradable,
        "dissolved_O2_kg_m3": c["S_O"],
        "alkalinity_mol_eq_m3": c["S_ALK"],
    }
