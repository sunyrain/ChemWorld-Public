"""Shared dimensionless crossover shape for water/air heat conduction.

IAPWS R15-11 eq. 19; Lemmon-Jacobsen (2004) Omega-Omega0 divided by y.
Model-specific length, heat-capacity units and small-y cutoffs stay with
their physical models. This helper has no EOS or susceptibility policy.
"""

from math import atan, expm1, pi


def critical_conductivity_shape(y, density_reduced, cv_over_cp):
    """Z(y), using series only to avoid cancellation of elementary terms."""
    if y == 0 or density_reduced == 0:
        return 0.0
    q = y**3 / (3 * density_reduced**2)
    s = y / (1 + q)
    atan_minus_y = -(y**3) / 3 + y**5 / 5 - y**7 / 7 if y < 1e-4 else atan(y) - y
    exponential_remainder = (
        s**2 / 2 - s**3 / 6 + s**4 / 24 - s**5 / 120 + s**6 / 720 if s < 1e-4 else expm1(-s) + s
    )
    bracket = (1 - cv_over_cp) * atan_minus_y + y * q / (1 + q) + exponential_remainder
    return 2 / (pi * y) * bracket
