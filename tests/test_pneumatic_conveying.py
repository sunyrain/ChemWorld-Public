from math import pi

import pytest

from chemworld.physchem.pneumatic_conveying import saltation_velocity
from chemworld.physchem.transport import FluidState


def test_current_gas_flow_basis_and_explicit_terminal_velocity():
    gas = FluidState(1.2, 1.8e-5)
    result = saltation_velocity("Rizk", gas, solids_mass_flow_kg_s=.1,
                                pipe_diameter_m=.05, particle_diameter_m=.0005)
    assert result.gas_volume_flow_m3_s == pytest.approx(result.saltation_velocity_m_s*pi*.05**2/4)
    assert result.gas_mass_flow_kg_s == 1.2*result.gas_volume_flow_m3_s
    with pytest.raises(ValueError, match="terminal"):
        saltation_velocity("Weber_saltation", gas, solids_mass_flow_kg_s=.1,
                            pipe_diameter_m=.05, particle_diameter_m=.0005)
    with pytest.raises(ValueError, match="below pipe"):
        saltation_velocity("Rizk", gas, solids_mass_flow_kg_s=.1,
                            pipe_diameter_m=.05, particle_diameter_m=.1)


def test_particle_branches_and_retained_source_warning():
    args = {"solids_mass_flow_kg_s": .1, "pipe_diameter_m": .05,
            "particle_density_kg_m3": 2500, "terminal_velocity_m_s": 2}
    gas = FluidState(1.2, 1.8e-5)
    fine = saltation_velocity("Matsumoto_1977", gas, particle_diameter_m=1e-5, **args)
    coarse = saltation_velocity("Matsumoto_1977", gas, particle_diameter_m=.001, **args)
    assert fine.branch == "fine particles" and coarse.branch == "coarse particles"
    older = saltation_velocity("Matsumoto_1974", gas, particle_diameter_m=.0001, **args)
    assert len(older.warnings) == 3
