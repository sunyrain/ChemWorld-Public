"""One independent mapping of ThermoPack TV responses to the shared EOS report."""


def thermopack_residual_targets(reference, temperature, density, fractions):
    p, pt, pv = reference.pressure_tv(temperature, 1/density, fractions, dpdt=True, dpdv=True)
    u, cv = reference.internal_energy_tv(temperature, 1/density, fractions,
                                        dedt=True, property_flag='R')
    a, at, an = reference.helmholtz_tv(temperature, 1/density, fractions,
                                      dadt=True, dadn=True, property_flag='R')
    R = reference.Rgas
    return {'pressure_Pa': p, 'compressibility_factor': p/(density*R*temperature),
            'residual_helmholtz_J_mol': a, 'residual_internal_energy_J_mol': u,
            'residual_enthalpy_J_mol': u+p/density-R*temperature,
            'residual_entropy_J_mol_K': -at, 'residual_cv_J_mol_K': cv,
            'residual_cp_J_mol_K': cv-temperature*pt*pt/pv-R,
            'dP_dT_rho_Pa_K': pt, 'dP_drho_T_Pa_m3_mol': -pv/density**2}, an
