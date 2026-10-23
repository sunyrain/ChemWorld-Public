"""Differential-meter geometry, Miller/nozzle equations and gas expansion.

These are constitutive equations. A standard name does not certify the actual
installation, straight-run lengths, surface condition or empirical validity.
"""

from math import acos, exp, expm1, isfinite, log, pi, sqrt

from chemworld.physchem.flow_meters import _positive, orifice_discharge_coefficient

METER_ALIASES = {
    "orifice": "ISO 5167 orifice",
    "eccentric orifice": "ISO 15377 eccentric orifice",
    "conical orifice": "ISO 15377 conical orifice",
    "quarter circle orifice": "ISO 15377 quarter-circle orifice",
    "segmental orifice": "Miller segmental orifice",
}
METER_TYPES = (
    "ISO 5167 orifice",
    "ISO 15377 eccentric orifice",
    "ISO 15377 conical orifice",
    "ISO 15377 quarter-circle orifice",
    "Miller orifice",
    "Miller eccentric orifice",
    "Miller conical orifice",
    "Miller quarter circle orifice",
    "Miller segmental orifice",
    "long radius nozzle",
    "ISA 1932 nozzle",
    "venturi nozzle",
    "as cast convergent venturi tube",
    "machined convergent venturi tube",
    "rough welded convergent venturi tube",
    "cone meter",
    "wedge meter",
    "Hollingshead orifice",
    "Hollingshead venturi smooth",
    "Hollingshead venturi sharp",
    "Hollingshead v cone",
    "Hollingshead wedge",
    "unspecified meter",
)


def meter_beta(pipe_diameter_m: float, secondary_dimension_m: float, meter_type: str) -> float:
    _positive(pipe_diameter_m, "pipe diameter")
    _positive(secondary_dimension_m, "meter secondary dimension")
    ratio = secondary_dimension_m / pipe_diameter_m
    meter_type = METER_ALIASES.get(meter_type, meter_type)
    if not 0 < ratio < 1 or meter_type not in METER_TYPES:
        raise ValueError("known meter type and secondary dimension below pipe diameter required")
    if meter_type in {"cone meter", "Hollingshead v cone"}:
        return sqrt(1 - ratio**2)
    if meter_type in {"wedge meter", "Hollingshead wedge"}:
        return sqrt((acos(1 - 2 * ratio) - 2 * (1 - 2 * ratio) * sqrt(ratio * (1 - ratio))) / pi)
    return ratio


def meter_expansibility(
    beta: float,
    pressure_ratio: float,
    heat_capacity_ratio: float,
    *,
    method: str = "orifice",
    enforce_pressure_domain: bool = True,
) -> float:
    """Subcritical gas expansion; pressure ratio defaults to the >=.75 fit domain.

    Nozzle k=1 and equal-pressure limits use the same supplied effective beta,
    including noncircular wedge passages. No choking claim is made.
    """
    if method not in {"orifice", "orifice_1989", "nozzle", "cone", "conical_orifice"}:
        raise ValueError("unknown meter expansion method")
    if not 0 < beta < 1 or not 0 < pressure_ratio <= 1:
        raise ValueError("physical effective beta and downstream/upstream pressure ratio required")
    if not isfinite(heat_capacity_ratio) or heat_capacity_ratio < 1:
        raise ValueError("heat capacity ratio must be at least one")
    if enforce_pressure_domain and pressure_ratio < 0.75:
        raise ValueError("meter expansion outside declared pressure-ratio domain")
    k, ratio = heat_capacity_ratio, pressure_ratio
    if ratio == 1:
        return 1.0
    b4 = beta**4
    if method == "orifice":
        return 1 + (0.351 + 0.256 * b4 + 0.93 * b4**2) * expm1(log(ratio) / k)
    if method == "orifice_1989":
        return 1 - (0.41 + 0.35 * b4) * (1 - ratio) / k
    if method == "cone":
        return 1 - (0.649 + 0.696 * b4) * (1 - ratio) / k
    if method == "conical_orifice":
        return 0.5 * (
            meter_expansibility(beta, ratio, k, method="orifice", enforce_pressure_domain=False)
            + meter_expansibility(beta, ratio, k, method="nozzle", enforce_pressure_domain=False)
        )
    if method != "nozzle":
        raise ValueError("unknown meter expansion method")
    power = (k - 1) / k
    logarithmic_factor = -log(ratio) if power == 0 else -expm1(power * log(ratio)) / power
    r2 = exp(2 * log(ratio) / k)
    return sqrt(r2 * (1 - b4) / (1 - b4 * r2) * logarithmic_factor / (1 - ratio))


def miller_orifice_coefficient(
    beta: float,
    pipe_diameter_m: float,
    reynolds: float,
    *,
    subtype: str = "orifice",
    taps: str = "corner",
    tap_position: str = "180 degree",
) -> float:
    """Five Miller shapes; fitted coefficients are equation parameters."""
    _positive(reynolds, "Reynolds number")
    _positive(pipe_diameter_m, "pipe diameter")
    if not 0 < beta < 1:
        raise ValueError("physical orifice beta required")
    subtype = subtype.removeprefix("Miller ")
    b, d = beta, pipe_diameter_m
    taps = "D-D/2" if taps in {"D", "D and D/2"} else taps
    if subtype == "conical orifice":
        return 0.734 if 250 * b <= reynolds <= 500 * b else 0.73
    if subtype == "quarter circle orifice":
        return 0.7746 - 0.1334 * b**2.1 + 1.4098 * b**8 + 0.0675 * b**4 / (1 - b**4) + 0.3865 * b**3
    if subtype == "orifice":
        c = 0.5959 + 0.0312 * b**2.1 - 0.184 * b**8
        if taps == "flange":
            c += (2.286 / (d * 1000) if d * 1000 >= 58.4 else 0.039) * b**4 / (
                1 - b**4
            ) - 0.856 * b**3 / (d * 1000)
        elif taps == "D-D/2":
            c += 0.039 * b**4 / (1 - b**4) - 0.01584
        elif taps == "pipe":
            c = 0.5959 + 0.461 * b**2.1 + 0.48 * b**8 + 0.039 * b**4 / (1 - b**4)
        elif taps != "corner":
            raise ValueError("Miller concentric supports corner/flange/D-D/2/pipe taps")
        return c + 91.706 * b**2.5 / reynolds**0.75
    if taps not in {"flange", "vena contracta"}:
        raise ValueError("Miller eccentric/segmental supports flange or vena-contracta taps")
    if subtype == "segmental orifice":
        values = (
            (0.6284, 0.1462, -0.8464, 0.2603, -0.2886)
            if taps == "flange"
            else (0.6261, 0.1851, -0.2879, 0.117, -0.2845)
        )
        if d >= 0.1:
            values = (0.6276, 0.0828, 0.2739, -0.0934, -0.1132)
        return sum(
            coefficient * term
            for coefficient, term in zip(
                values, (1, b**2.1, b**8, b**4 / (1 - b**4), b**3), strict=True
            )
        )
    if subtype != "eccentric orifice" or tap_position not in {"180 degree", "90 degree"}:
        raise ValueError("known Miller shape and eccentric tap position required")
    # Entries are polynomial and asymptotic fit coefficients from the named
    # equations, not tabulated experimental observations.
    fits = {
        ("flange", "180 degree", True): (
            (7.3, -15.7, 170.8, -399.7, 332.2),
            (0.5917, 0.3061, 0.3406, -0.1019, -0.2715),
        ),
        ("flange", "180 degree", False): (
            (-139.7, 1328.8, -4228.2, 5691.9, -2710.4),
            (0.6016, 0.3312, -1.5581, 0.651, -0.7308),
        ),
        ("flange", "90 degree", True): (
            (69.1, -469.4, 1245.6, -1287.5, 486.2),
            (0.5866, 0.3917, 0.7586, -0.2273, -0.3343),
        ),
        ("flange", "90 degree", False): (
            (-103.2, 898.3, -2557.3, 2977, -1131.3),
            (0.6037, 0.1598, -0.2918, 0.0244, -0.079),
        ),
        ("vena contracta", "180 degree", True): (
            (23.3, -207, 821.5, -1388.6, 900.3),
            (0.5925, 0.338, 0.4016, -0.1046, -0.3212),
        ),
        ("vena contracta", "180 degree", False): (
            (55.7, -471.4, 1721.8, -2722.6, 1569.4),
            (0.5922, 0.3932, 0.3412, -0.0569, -0.4628),
        ),
        ("vena contracta", "90 degree", True): (
            (-69.3, 556.9, -1332.2, 1303.7, -394.8),
            (0.5875, 0.3813, 0.6898, -0.1963, -0.3366),
        ),
        ("vena contracta", "90 degree", False): (
            (52.8, -434.2, 1571.2, -2460.9, 1420.2),
            (0.5949, 0.4078, 0.0547, 0.0955, -0.5608),
        ),
    }
    polynomial, asymptotic = fits[taps, tap_position, d < 0.1]
    correction = sum(coefficient * b**power for power, coefficient in enumerate(polynomial))
    constant = sum(
        coefficient * term
        for coefficient, term in zip(
            asymptotic, (1, b**2.1, b**8, b**4 / (1 - b**4), b**3), strict=True
        )
    )
    return constant + correction / reynolds**0.75


def meter_discharge_coefficient(
    beta: float,
    pipe_diameter_m: float,
    reynolds: float,
    *,
    meter_type: str,
    taps: str = "corner",
    tap_position: str = "180 degree",
) -> float:
    _positive(reynolds, "Reynolds number")
    _positive(pipe_diameter_m, "pipe diameter")
    if not 0 < beta < 1:
        raise ValueError("physical meter beta required")
    method = METER_ALIASES.get(meter_type, meter_type)
    b, d, re = beta, pipe_diameter_m, reynolds
    if method.startswith("Miller ") and method != "Miller wedge meter":
        return miller_orifice_coefficient(
            b, d, re, subtype=method, taps=taps, tap_position=tap_position
        )
    if method == "ISO 5167 orifice":
        return orifice_discharge_coefficient(
            re, pipe_diameter_m=d, beta=b, taps="D-D/2" if taps in {"D", "D and D/2"} else taps
        )
    if method == "ISO 15377 eccentric orifice":
        return 0.9355 - 1.6889 * b + 3.0428 * b * b - 1.7989 * b**3
    if method == "ISO 15377 quarter-circle orifice":
        return 0.73823 + 0.3309 * b - 1.16158 * b * b + 1.5084 * b**3
    constants = {
        "ISO 15377 conical orifice": 0.734,
        "as cast convergent venturi tube": 0.984,
        "machined convergent venturi tube": 0.995,
        "rough welded convergent venturi tube": 0.985,
        "cone meter": 0.82,
    }
    if method in constants:
        return constants[method]
    if method == "ISA 1932 nozzle":
        return 0.99 - 0.2262 * b**4.1 - (0.00175 * b * b - 0.0033 * b**4.15) * (1e6 / re) ** 1.15
    if method == "long radius nozzle":
        return 0.9965 - 0.00653 * sqrt(b * 1e6 / re)
    if method == "venturi nozzle":
        return 0.9858 - 0.196 * b**4.5
    if method == "wedge meter":
        return 0.77 - 0.09 * b
    if method == "Miller wedge meter":
        a, coefficient = (
            (0.7883, 0.107)
            if d <= 0.7 * 0.0254
            else (0.6143, 0.718)
            if d <= 1.4 * 0.0254
            else (0.5433, 0.2453)
        )
        return a + coefficient * (1 - b * b)
    raise ValueError(
        "this meter needs an explicit calibration coefficient or supplied source table"
    )


def meter_loss_coefficient(beta: float, discharge_coefficient: float) -> float:
    _positive(discharge_coefficient, "discharge coefficient")
    if not 0 < beta < 1:
        raise ValueError("physical beta required")
    return (
        sqrt(1 - beta**4 * (1 - discharge_coefficient**2)) / (discharge_coefficient * beta**2) - 1
    ) ** 2


def meter_coefficient_from_loss(beta: float, loss_coefficient: float) -> float:
    _positive(loss_coefficient, "loss coefficient")
    if not 0 < beta < 1:
        raise ValueError("physical beta required")
    return sqrt((1 - beta**4) / ((2 * sqrt(loss_coefficient) + loss_coefficient) * beta**4))


def wet_venturi_correction(
    *,
    pipe_diameter_m: float,
    throat_diameter_m: float,
    gas_mass_flow_kg_s: float,
    liquid_mass_flow_kg_s: float,
    gas_density_kg_m3: float,
    liquid_density_kg_m3: float,
    pressure_difference_Pa: float,
    liquid_factor: float = 1.0,
) -> tuple[float, float]:
    """Reader-Harris wet-gas coefficient and permanent pressure loss (no inventory)."""
    beta = meter_beta(pipe_diameter_m, throat_diameter_m, "venturi nozzle")
    for value in (gas_mass_flow_kg_s, gas_density_kg_m3, liquid_density_kg_m3, liquid_factor):
        _positive(value, "wet venturi parameter")
    if (
        liquid_density_kg_m3 <= gas_density_kg_m3
        or not isfinite(liquid_mass_flow_kg_s)
        or liquid_mass_flow_kg_s < 0
        or not isfinite(pressure_difference_Pa)
        or pressure_difference_Pa < 0
    ):
        raise ValueError(
            "positive liquid/gas density contrast and nonnegative flows/pressure loss required"
        )
    velocity = 4 * gas_mass_flow_kg_s / (pi * pipe_diameter_m**2 * gas_density_kg_m3)
    froude = velocity / sqrt(
        9.80665 * pipe_diameter_m * (liquid_density_kg_m3 - gas_density_kg_m3) / gas_density_kg_m3
    )
    x = liquid_mass_flow_kg_s / gas_mass_flow_kg_s * sqrt(gas_density_kg_m3 / liquid_density_kg_m3)
    coefficient = 1 - 0.0463 * exp(-0.05 * froude / beta**2.5) * min(1, sqrt(x / 0.016))
    wet_loss = 0.61 * exp(
        -11 * gas_density_kg_m3 / liquid_density_kg_m3 - 0.045 * froude / liquid_factor
    )
    wet_loss *= -expm1(-35 * x**0.75 * exp(-0.28 * froude / liquid_factor))
    return coefficient, pressure_difference_Pa * (wet_loss + 0.0896 + 0.48 * beta**9)
