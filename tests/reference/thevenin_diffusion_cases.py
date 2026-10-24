"""Fixed authored inputs for the twelve-unit Thevenin diffusion development block."""

import numpy as np

GRIDS = (32, 64, 128)
FREQUENCIES = np.array([10.**(-4+k/3) for k in range(13)])
CORRECTION_URL = "https://github.com/pybamm-team/PyBaMM/pull/5869"
CORRECTION_LABEL = "PyBaMM 25.4.2 plus explicitly attributed PR5869 two-expression correction"


def case_data(index):
    if not 1 <= index <= 12:
        raise ValueError("the fixed diffusion block has exactly twelve units")
    data = {
        "id": f"D{index:02d}", "kind": "frequency" if index >= 11 else "time",
        "grids": list(GRIDS), "capacity_C": 100., "initial_soc": .6,
        "initial_temperature_K": 298.15, "ambient_K": 298.15,
        "r0_ohm": .1, "rc_resistances_ohm": [], "rc_capacitances_F": [],
        "entropy_V_K": 0., "cell_heat_capacity_J_K": 100.,
        "jig_heat_capacity_J_K": 500., "cell_jig_conductance_W_K": .5,
        "jig_ambient_conductance_W_K": .2, "maximum_current_A": 2.,
        "minimum_voltage_V": .1, "maximum_voltage_V": 10.,
        "minimum_temperature_K": 250., "maximum_temperature_K": 400.,
        "tau_s": 30., "ocv_nodes": [[0., 3.], [1., 4.]],
        "segments": [{"duration_s": 20., "value": .2, "unit": "A"}],
        "samples_per_segment": 21, "expected_surface_event": False,
    }
    if index == 2:
        data["segments"][0]["value"] = -.2
    if index in (3, 4):
        data["tau_s"] = 3. if index == 3 else 300.
    if index in (5, 12):
        data.update(rc_resistances_ohm=[.2, .4], rc_capacitances_F=[3., 5.])
    if index == 6:
        data["segments"] = [{"duration_s": duration, "value": value, "unit": "A"}
                            for duration, value in ((10., .2), (20., 0.), (10., -.2), (30., 0.))]
    if index == 7:
        data["maximum_current_A"] = .2
        data["segments"] = [{"duration_s": 30., "value": 3.57, "unit": "V"}]
    if index == 8:
        data.update(initial_soc=.08, tau_s=300., expected_surface_event=True)
        data["segments"][0]["value"] = .5
    if index == 9:
        data.update(initial_soc=.55, ocv_nodes=[[0., 3.], [.5, 3.2], [1., 4.2]])
    if index == 10:
        data["r0_ohm"] = 0.
        data["segments"] = [{"duration_s": duration, "value": value, "unit": "A"}
                            for duration, value in ((20., .2), (20., -.2), (300., 0.))]
    if index >= 11:
        data["segments"] = []
        data["frequencies_Hz"] = FREQUENCIES.tolist()
    if index == 12:
        data["entropy_V_K"] = 1e-4
    return data


TIME_RESPONSES = (
    "voltage_V", "current_A", "soc", "surface_soc", "cell_temperature_K",
    "jig_temperature_K", "charge_C", "work_J", "model_heat_J", "ambient_loss_J",
    "diffusion_overpotential_V", "diffusion_heat_W",
)
TIME_ATOL = (1e-5, 1e-5, 1e-6, 2e-5, 1e-4, 1e-4, 1e-4, 2e-3, 2e-3, 2e-3, 1e-5, 1e-5)
MESH_ATOL = (5e-4, 5e-4, 2e-5, 5e-4, 5e-4, 5e-4, 2e-3, 2e-2, 2e-2, 2e-2, 5e-4, 5e-4)


def time_names(index):
    return [*TIME_RESPONSES,
            *(f"rc{k}_V" for k in range(len(case_data(index)["rc_resistances_ohm"])))]


def time_atol(index):
    return np.array([*TIME_ATOL, *([1e-6]*len(case_data(index)["rc_resistances_ohm"]))])


def mesh_atol(index):
    return np.array([*MESH_ATOL, *([5e-4]*len(case_data(index)["rc_resistances_ohm"]))])


def frequency_names(index):
    return ["Z_ohm", "soc_per_A", "surface_soc_per_A", "cell_K_per_A", "jig_K_per_A",
            *(f"rc{k}_V_per_A" for k in range(len(case_data(index)["rc_resistances_ohm"])))]


def frequency_atol(index):
    return np.array([2e-6, 1e-7, 1e-7, 1e-6, 1e-6,
                     *([1e-7]*len(case_data(index)["rc_resistances_ohm"]))])


def native_case(index, cells):
    from battery_thevenin_cases import law, table

    from chemworld.physchem.battery_thevenin import CONSTANT_UNITS, TheveninSpec
    from chemworld.physchem.battery_thevenin_diffusion import TheveninDiffusionSpec
    from chemworld.physchem.reactor_shared import ScalarControlProgram

    data = case_data(index)
    if cells not in GRIDS:
        raise ValueError("qualification grid must be one of the three fixed meshes")
    count = len(data["rc_resistances_ohm"])
    p = {name: data[name] for name in (
        "capacity_C", "maximum_current_A", "initial_soc", "minimum_voltage_V",
        "maximum_voltage_V", "minimum_temperature_K", "maximum_temperature_K",
        "cell_heat_capacity_J_K", "jig_heat_capacity_J_K", "cell_jig_conductance_W_K",
        "jig_ambient_conductance_W_K")}
    p.update(initial_cell_temperature_K=data["initial_temperature_K"],
             initial_jig_temperature_K=data["initial_temperature_K"],
             reference_frame_temperature_K=298.15, housing_volume_L=.01,
             minimum_soc=0., maximum_soc=1.)
    p.update({f"initial_rc_{k}_V": 0. for k in range(count)})

    def constant(identity, value, unit):
        return law(identity, {}, unit, lambda: value)

    nodes = dict(data["ocv_nodes"])
    diffusion = TheveninDiffusionSpec(table(
        f"D{index:02d}-diffusion", {"diffusion_time_constant_s": "s"},
        [{"diffusion_time_constant_s": data["tau_s"]}]), "0",
        tuple(np.linspace(0., 1., cells+1)))
    return TheveninSpec(table(f"D{index:02d}-constants", {
        **CONSTANT_UNITS, **{f"initial_rc_{k}_V": "V" for k in range(count)}}, [p]), "0",
        law("ocv", {"soc": tuple(nodes)}, "V", lambda z: nodes[z]),
        constant("series-r", data["r0_ohm"], "Ohm"),
        tuple((constant(f"r{k}", r, "Ohm"), constant(f"c{k}", c, "F"))
              for k, (r, c) in enumerate(zip(data["rc_resistances_ohm"],
                                            data["rc_capacitances_F"], strict=True))),
        constant("entropy", data["entropy_V_K"], "V/K"),
        ScalarControlProgram(((0., data["ambient_K"]),), "K"), diffusion=diffusion)
