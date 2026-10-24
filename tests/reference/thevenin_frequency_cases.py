"""Predeclared stationary authored inputs, shared as data, not as model equations."""

import numpy as np
from battery_thevenin_cases import law, table

from chemworld.physchem.battery_thevenin import CONSTANT_UNITS, TheveninSpec
from chemworld.physchem.reactor_shared import ScalarControlProgram

FREQUENCIES = np.array([10.**(-4+.5*k) for k in range(13)])


def case_data(index):
    if not 1 <= index <= 12:
        raise ValueError("the fixed frequency block has exactly twelve units")
    count = (0, 1, 2, 4, 1, 2, 1, 1, 1, 0, 4, 2)[index-1]
    return {"rc_count": count, "capacity_C": 100., "soc": .3 if index == 7 else .7,
        "temperature_K": 310. if index == 12 else 298.15,
        "ambient_K": 298.15, "maximum_current_A": .1 if index == 12 else 10.,
        "cell_heat_capacity_J_K": 100., "jig_heat_capacity_J_K": 500.,
        "cell_jig_conductance_W_K": 0. if index == 12 else .5,
        "jig_ambient_conductance_W_K": 0. if index == 12 else .2,
        "r0_ohm": 0. if index == 10 else .1,
        "resistances_ohm": [.2*(k+1) for k in range(count)],
        "capacitances_F": [3.+2*k for k in range(count)],
        "entropy_V_K": -1e-4 if index == 6 else 0. if index == 11 else 1e-4,
        "ocv_nodes": ([[0., 3.], [.5, 3.2], [1., 4.2]] if index in (7, 8)
                      else [[0., 3.6], [1., 3.6]] if index == 11 else [[0., 3.], [1., 4.]]),
        "feedback": "affine" if index == 5 else "current_cusp" if index == 9 else "constant"}


def native_case(index):
    data = case_data(index)
    count = data["rc_count"]
    p = {"capacity_C": data["capacity_C"], "maximum_current_A": data["maximum_current_A"],
        "initial_soc": data["soc"], "initial_cell_temperature_K": data["temperature_K"],
        "initial_jig_temperature_K": data["temperature_K"], "minimum_soc": 0., "maximum_soc": 1.,
        "minimum_voltage_V": .1, "maximum_voltage_V": 10., "minimum_temperature_K": 250.,
        "maximum_temperature_K": 400., "reference_frame_temperature_K": 298.15,
        "housing_volume_L": .01, **{k: data[k] for k in (
            "cell_heat_capacity_J_K", "jig_heat_capacity_J_K", "cell_jig_conductance_W_K",
            "jig_ambient_conductance_W_K")}, **{f"initial_rc_{k}_V": 0. for k in range(count)}}

    def constant(identity, value, unit):
        return law(identity, {}, unit, lambda: value)

    r0 = constant("series-r", data["r0_ohm"], "Ohm")
    pairs = tuple((constant(f"r{k}", r, "Ohm"), constant(f"c{k}", c, "F"))
                  for k, (r, c) in enumerate(zip(data["resistances_ohm"],
                                                data["capacitances_F"], strict=True)))
    if data["feedback"] == "affine":
        axes = {"soc": (0., 1.), "temperature_K": (250., 400.), "current_A": (-10., 0., 10.)}
        r0 = law("affine-r0", axes, "Ohm",
                 lambda z, T, i: .08+.02*z+.0001*(T-298.15)+.002*i)
        pairs = ((law("affine-r1", axes, "Ohm",
                      lambda z, T, i: .15+.05*z+.0002*(T-298.15)+.001*i),
                  law("affine-c1", axes, "F",
                      lambda z, T, i: 4.+z+.002*(T-298.15)+.01*i)),)
    elif data["feedback"] == "current_cusp":
        r0 = law("cusp-r0", {"current_A": (-10., 0., 10.)}, "Ohm", lambda i: .1+.02*abs(i))
        pairs = ((law("cusp-r1", {"current_A": (-10., 0., 10.)}, "Ohm",
                      lambda i: .2+.01*abs(i)), pairs[0][1]),)
    nodes = dict(data["ocv_nodes"])
    return TheveninSpec(table(f"frequency-F{index:02d}", {
        **CONSTANT_UNITS, **{f"initial_rc_{k}_V": "V" for k in range(count)}}, [p]), "0",
        law("ocv", {"soc": tuple(nodes)}, "V", lambda z: nodes[z]), r0, pairs,
        constant("entropy", data["entropy_V_K"], "V/K"),
        ScalarControlProgram(((0., data["ambient_K"]),), "K"))


def response_order(index):
    return ["Z_ohm", "soc_per_A", "cell_K_per_A", "jig_K_per_A",
            *(f"rc{k}_V_per_A" for k in range(case_data(index)["rc_count"]))]


def response_tolerances(index):
    count = case_data(index)["rc_count"]
    return np.array([2e-6, 1e-7, 1e-6, 1e-6, *([1e-7]*count)]), np.array(
        [2e-6, 1e-6, 1e-5, 1e-5, *([1e-6]*count)])
