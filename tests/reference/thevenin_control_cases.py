"""Fixed authored inputs for sixteen named-controller qualification units."""

from copy import deepcopy

import numpy as np

BASE = {
    "capacity_C": 100., "initial_soc": .6, "initial_temperature_K": 298.15,
    "ambient_K": 298.15, "r0_ohm": .1, "r0_law": "constant", "ocv_intercept_V": 3.,
    "rc_resistances_ohm": [], "rc_capacitances_F": [], "entropy_V_K": 0.,
    "cell_heat_capacity_J_K": 100., "jig_heat_capacity_J_K": 500.,
    "cell_jig_conductance_W_K": .5, "jig_ambient_conductance_W_K": .2,
    "maximum_current_A": 2., "minimum_voltage_V": .1, "maximum_voltage_V": 10.,
    "diffusion_cells": 0, "diffusion_tau_s": 30., "samples_per_segment": 41,
    "current_scale_A": 1., "power_scale_W": 1., "resistance_scale_Ohm": 1.,
    "voltage_scale_V": 1., "power_response_time_s": 100., "resistance_response_time_s": 100.,
    "voltage_response_time_s": 1., "antiwindup_time_s": 100/3600.,
}


def variant(name, mode, target, duration, *, q0=.1, branch=None, current_limit=None,
            expected="trajectory", **changes):
    return {"name": name, "mode": mode, "segments": [{"target": target, "duration_s": duration}],
            "initial_current_A": q0, "branch_A": branch, "current_limit_A": current_limit,
            "expected": expected, "changes": changes}


UNITS = (
    (variant("discharge", "power", .72, 20., branch=(-2., 2.)),),
    (variant("charge", "power", -.72, 20., branch=(-2., 2.)),),
    (variant("low-root", "power", 3., 1., q0=.5, branch=(0., 1.4),
             r0_ohm=1., maximum_current_A=4.),
     variant("high-root", "power", 3., 1., q0=3., branch=(2.1, 3.5),
             r0_ohm=1., maximum_current_A=4.)),
    (variant("fold", "power", 3.24, 0., q0=1.8, branch=(0., 4.), expected="fold_rejection",
             r0_ohm=1., maximum_current_A=4.),),
    (variant("no-root", "power", 3.4, 0., branch=(0., 1.4), expected="no_root_rejection",
             r0_ohm=1., maximum_current_A=4.),),
    (variant("zero-series-R", "power", .72, 10., branch=(-2., 2.), r0_ohm=0.),),
    (variant("certified-current-R", "power", .72, 20., branch=(-2., 2.), r0_law="linear"),
     variant("uncertified-power-branch", "power", 3.7, 0., branch=(0., 2.),
             expected="nonmonotone_power_rejection", r0_law="absolute")),
    (variant("load-two-RC", "resistance", 9., 20., r0_ohm=0.,
             rc_resistances_ohm=[.2, .4], rc_capacitances_F=[3., 5.]),),
    (variant("dynamic-power", "differential_power", .72, 20., q0=.1),),
    ({**variant("power-reversal", "differential_power", .72, 10., q0=.2),
      "segments": [{"target": .72, "duration_s": 10.}, {"target": -.72, "duration_s": 40.}]},),
    (variant("upstream-unstable-resistance", "differential_resistance", 18., 5.,
             q0=.21, branch=(.05, 2.)),),
    (variant("zero-current-singularity", "differential_resistance", 18., 0.,
             q0=0., branch=(.05, 2.), expected="singular_initial_current_rejection"),),
    (variant("CCCV-transition", "cccv", 3.65, 40., q0=-.2, current_limit=-.2),),
    (variant("CCCV-antiwindup", "cccv", 3.65, .5, q0=-.8, current_limit=-.2),),
    (variant("power-diffusion-RC", "power", .72, 20., branch=(-2., 2.), diffusion_cells=64,
             rc_resistances_ohm=[.02, .04], rc_capacitances_F=[5., 10.], entropy_V_K=1e-4),),
    (variant("CCCV-diffusion", "cccv", 3.65, 40., q0=-.5, current_limit=-.2,
             diffusion_cells=64),),
)


def case_data(index, variant_index=0):
    if index not in range(1, 17):
        raise ValueError("named control block fixes exactly sixteen units")
    selected = deepcopy(UNITS[index-1][variant_index])
    return {**deepcopy(BASE), **selected.pop("changes"), **selected}


def dynamic(data):
    return data["mode"] in {"differential_power", "differential_resistance", "cccv"}


def target_unit(mode):
    return {"power": "W", "resistance": "Ohm", "differential_power": "W",
            "differential_resistance": "Ohm", "cccv": "V"}[mode]


def response_names(data):
    names = ["voltage_V", "current_A", "soc", "cell_temperature_K", "jig_temperature_K",
             "charge_C", "work_J", "throughput_charge_C", "throughput_work_J",
             "model_heat_J", "ambient_loss_J"]
    names += [f"rc{k}_V" for k in range(len(data["rc_resistances_ohm"]))]
    if dynamic(data):
        names += ["controller_current_A"]
    if data["diffusion_cells"]:
        names += ["surface_soc", "diffusion_overpotential_V", "diffusion_heat_W"]
    return names


def response_atol(data):
    values = [1e-5, 1e-5, 1e-6, 1e-4, 1e-4, 1e-4, 2e-3, 1e-4, 2e-3, 2e-3, 2e-3]
    values += [1e-5]*len(data["rc_resistances_ohm"])
    if dynamic(data):
        values += [1e-5]
    if data["diffusion_cells"]:
        values += [2e-5, 1e-5, 1e-5]
    return np.asarray(values)


def native_spec(data):
    from battery_thevenin_cases import law, table

    from chemworld.physchem.battery_thevenin import CONSTANT_UNITS, TheveninSpec
    from chemworld.physchem.battery_thevenin_control import CONTROL_UNITS, TheveninControlSpec
    from chemworld.physchem.battery_thevenin_diffusion import TheveninDiffusionSpec
    from chemworld.physchem.reactor_shared import ScalarControlProgram

    n = len(data["rc_resistances_ohm"])
    p = {key: data[key] for key in (
        "capacity_C", "maximum_current_A", "minimum_voltage_V", "maximum_voltage_V",
        "initial_soc", "cell_heat_capacity_J_K", "jig_heat_capacity_J_K",
        "cell_jig_conductance_W_K", "jig_ambient_conductance_W_K")}
    p.update(minimum_soc=0., maximum_soc=1.,
        initial_cell_temperature_K=data["initial_temperature_K"],
        initial_jig_temperature_K=data["initial_temperature_K"], minimum_temperature_K=250.,
        maximum_temperature_K=400., reference_frame_temperature_K=298.15, housing_volume_L=.01)
    p.update({f"initial_rc_{k}_V": 0. for k in range(n)})

    def constant(identity, value, unit):
        return law(identity, {}, unit, lambda: value)

    r0 = constant("control-series-r", data["r0_ohm"], "Ohm")
    if data["r0_law"] != "constant":
        limit = data["maximum_current_A"]
        r0 = law("control-current-dependent-r", {"current_A": (-limit, 0., limit)}, "Ohm",
                 (lambda current: .1+.01*current) if data["r0_law"] == "linear"
                 else (lambda current: .1+.4*abs(current)))
    rc = tuple((constant(f"control-R{k}", r, "Ohm"), constant(f"control-C{k}", c, "F"))
               for k, (r, c) in enumerate(zip(data["rc_resistances_ohm"],
                                             data["rc_capacitances_F"], strict=True)))
    control = TheveninControlSpec(table("named-controller", CONTROL_UNITS,
                                       [{key: data[key] for key in CONTROL_UNITS}]), "0")
    diffusion = None
    if data["diffusion_cells"]:
        diffusion = TheveninDiffusionSpec(table("control-diffusion",
            {"diffusion_time_constant_s": "s"},
            [{"diffusion_time_constant_s": data["diffusion_tau_s"]}]),
            "0", tuple(np.linspace(0., 1., data["diffusion_cells"]+1)))
    return TheveninSpec(table("control-circuit-constants", {
        **CONSTANT_UNITS, **{f"initial_rc_{k}_V": "V" for k in range(n)}}, [p]), "0",
        law("control-OCV", {"soc": (0., .5, 1.)}, "V", lambda z: data["ocv_intercept_V"]+z),
        r0, rc, constant("control-entropy", data["entropy_V_K"], "V/K"),
        ScalarControlProgram(((0., data["ambient_K"]),), "K"), diffusion=diffusion, control=control)


def native_program(data, part):
    from chemworld.physchem.battery_thevenin_control import TheveninControlProgram
    from chemworld.physchem.reactor_shared import ScalarControlProgram

    return TheveninControlProgram(data["mode"],
        ScalarControlProgram(((0., part["target"]),), target_unit(data["mode"])),
        ScalarControlProgram(((0., data["current_limit_A"]),), "A")
        if data["mode"] == "cccv" else None,
        data["branch_A"])
