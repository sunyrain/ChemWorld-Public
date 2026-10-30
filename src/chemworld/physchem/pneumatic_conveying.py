"""Explicit pneumatic saltation correlations using current native gas properties."""

from dataclasses import asdict, dataclass
from math import exp, isfinite, log, pi, sqrt
from typing import Any

from chemworld.physchem.transport import GRAVITY_M_S2, FluidState

SALTATION_METHODS = (
    "Rizk", "Schade", "Geldart_Ling", "Matsumoto_1974", "Matsumoto_1975",
    "Matsumoto_1977", "Weber_saltation",
)


def _positive(value: float | None, name: str) -> float:
    if value is None or isinstance(value, bool) or not isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be explicitly finite and positive")
    return value


@dataclass(frozen=True)
class SaltationResult:
    method: str
    saltation_velocity_m_s: float
    gas_volume_flow_m3_s: float
    gas_mass_flow_kg_s: float
    solids_mass_flux_kg_m2_s: float
    branch: str
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def saltation_velocity(
    method: str, gas: FluidState, *, solids_mass_flow_kg_s: float, pipe_diameter_m: float,
    particle_diameter_m: float | None = None, particle_density_kg_m3: float | None = None,
    terminal_velocity_m_s: float | None = None,
) -> SaltationResult:
    """Return a selected empirical transport threshold, not a dynamic conveying state.

Terminal velocity is a caller-supplied property at the declared gas state.
No terminal-velocity or particle-density default is inferred. All dimensions SI.
"""
    if method not in SALTATION_METHODS:
        raise ValueError("unknown saltation method")
    mass = _positive(solids_mass_flow_kg_s, "solids mass flow")
    diameter = _positive(pipe_diameter_m, "pipe diameter")
    area = pi*diameter*diameter/4
    flux = mass/area
    gas_scale = flux/gas.density_kg_m3
    speed_scale = sqrt(GRAVITY_M_S2*diameter)
    notes = ["empirical saltation threshold; source parity is not material qualification"]
    dp = None
    if method != "Geldart_Ling":
        dp = _positive(particle_diameter_m, "particle diameter")
        if dp >= diameter:
            raise ValueError("particle diameter must be below pipe diameter")
    if particle_density_kg_m3 is not None:
        _positive(particle_density_kg_m3, "particle density")
    density_ratio = None
    if method.startswith("Matsumoto") or method == "Schade":
        density_ratio = _positive(particle_density_kg_m3, "particle density")/gas.density_kg_m3
        if density_ratio <= 1:
            raise ValueError("saltation particles must be denser than carrier gas")
    branch = "single correlation"
    if method == "Geldart_Ling":
        low = flux/diameter <= 47000
        a, b, c = (1.5, .465, -.01) if low else (8.7, .302, .153)
        speed = a*flux**b*diameter**c*gas.viscosity_Pa_s**.055*gas.density_kg_m3**-.42
        branch = "Gs/D <= 47000" if low else "Gs/D > 47000"
    elif method == "Rizk":
        if dp is None:
            raise AssertionError("invariant violated: dp is not None")
        alpha, beta = 1440*dp+1.96, 1100*dp+2.5
        # Log form avoids forming a tiny empirical power before taking its root.
        speed = exp((log(gas_scale)+alpha*log(10)+beta*log(speed_scale))/(1+beta))
    elif method == "Schade":
        if dp is None or density_ratio is None:
            raise AssertionError("invariant violated: dp or density_ratio is None")
        speed = (gas_scale**.11*(diameter/dp)**.025*density_ratio**.34*speed_scale)**(1/1.11)
    elif method == "Weber_saltation":
        if dp is None:
            raise AssertionError("invariant violated: dp is not None")
        terminal = _positive(terminal_velocity_m_s, "terminal velocity")
        factor = 7+8*terminal/3 if terminal <= 3 else 15.
        speed = (factor*(dp/diameter)**.1*speed_scale*gas_scale**.25)**.8
        branch = "terminal <= 3 m/s" if terminal <= 3 else "terminal > 3 m/s"
    else:
        if dp is None or density_ratio is None:
            raise AssertionError("invariant violated: dp or density_ratio is None")
        terminal = _positive(terminal_velocity_m_s, "terminal velocity")
        fine = method == "Matsumoto_1977" and dp <= 1.39*diameter*density_ratio**-.74
        if fine:
            coefficient = 5560*(dp/diameter)**1.43/(10*speed_scale)**4
            speed = (gas_scale/coefficient)**.2
            branch = "fine particles"
            notes.append("fine-particle branch is independent of supplied terminal velocity")
        else:
            a, b, c, d = {"Matsumoto_1974": (.448, .5, -1.75, 3.),
                           "Matsumoto_1975": (1.11, .55, -2.3, 3.),
                           "Matsumoto_1977": (.373, 1.06, -3.7, 3.61)}[method]
            particle_froude = terminal/sqrt(GRAVITY_M_S2*dp)
            coefficient = a*density_ratio**b*(particle_froude/10)**c/(10*speed_scale)**d
            speed = (gas_scale/coefficient)**(1/(1+d))
            branch = "coarse particles"
        if method in {"Matsumoto_1974", "Matsumoto_1975"}:
            notes.append("reference notes high results; original-source accuracy unresolved")
            if dp <= .0003:
                notes.append("particle diameter outside source recommendation >0.3 mm")
    _positive(speed, "predicted saltation velocity")
    volume_flow = speed*area
    return SaltationResult(method, speed, volume_flow, volume_flow*gas.density_kg_m3,
                            flux, branch, tuple(notes))
