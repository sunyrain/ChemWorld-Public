"""Reference-only isolation of teqp0.23.2's finite BH integration rule.

This callback never enters a serialized/native model. Use only to attribute
source comparison differences; the default model keeps its independent128nodes.
"""

from math import log

import numpy as np

from chemworld.physchem.caloric_math import exponential, logarithm, scalar_value
from chemworld.physchem.saftvrmie import mie_prefactor

NODES, WEIGHTS = np.polynomial.legendre.leggauss(10)


def source_diameter(component, temperature):
    lr, la = component.repulsive_exponent, component.attractive_exponent
    kappa = mie_prefactor(lr, la)*component.well_depth_K
    eps = np.finfo(float).eps
    j = exponential(logarithm(-log(eps)*temperature/kappa)/lr)
    for _ in range(4):
        u = kappa*(j**lr-j**la)
        up = kappa*(lr*j**lr-la*j**la)/j
        upp = kappa*(lr*(lr-1)*j**lr-la*(la-1)*j**la)/j**2
        f, fp, fpp = -u/temperature-log(eps), -up/temperature, -upp/temperature
        denominator = 2*fp*fp-f*fpp
        if scalar_value(denominator) < eps:
            break
        j -= 2*f*fp/denominator
    low = 1/j
    values = []
    for node, weight in zip(NODES, WEIGHTS, strict=True):
        r = (1+low)/2+(1-low)/2*float(node)
        values.append(float(weight)*(1-exponential(-kappa*(r**-lr-r**-la)/temperature)))
    return component.diameter_m*(low+(1-low)/2*sum(values))
