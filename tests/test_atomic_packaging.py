"""Check resources and actual installation of the sdist-built wheel."""

import importlib.util
import json
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def installed_atomic_resources(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("atomic-install")
    root = Path(__file__).resolve().parents[1]
    # uv builds the wheel from the sdist, exercising the generated source index.
    subprocess.run(
        ["uv", "build", "--out-dir", str(tmp_path)],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    resources = ("elements.csv", "isotopes.csv", "provenance.json")
    wheel = next(tmp_path.glob("*.whl"))
    with zipfile.ZipFile(wheel) as archive:
        for name in resources:
            package_path = "chemworld/physchem/atomic_data/" + name
            assert archive.read(package_path) == (root / "src" / package_path).read_bytes()
    with tarfile.open(next(tmp_path.glob("*.tar.gz"))) as archive:
        for name in resources:
            member = next(item for item in archive if item.name.endswith("/atomic_data/" + name))
            assert (
                archive.extractfile(member).read()
                == (root / "src/chemworld/physchem/atomic_data" / name).read_bytes()
            )
    program = """
import sys
sys.path.insert(0, sys.argv[1])
import chemworld
assert sys.argv[1] in chemworld.__file__, chemworld.__file__
from chemworld.physchem import ComponentSpec, get_element, get_isotope, atomic_data_provenance
from chemworld.physchem import (
    PropertyCorrelation, PiecewiseHeatCapacity, temperature_from_caloric_change,
    vibrational_temperature,
)
from chemworld.data.identifiers import IdentifierDatabase
from chemworld.data.constants import ConstantDatabase
from chemworld.data.caloric import CaloricDatabase
from chemworld.data import TdbDatabase
from chemworld.data.scientific_inputs import InputSource
from chemworld.physchem.data_conflicts import DataSourceProvenance
from chemworld.data.pressure import PressureDatabase
from chemworld.data.phase_change import PhaseChangeDatabase
from chemworld.data.volume import VolumeDatabase
from pathlib import Path
packed_volume_root = Path(sys.argv[1] + '.volume-data')
packed_volume_db = VolumeDatabase(packed_volume_root)
assert len(packed_volume_db.catalog) == 8 and packed_volume_db.loaded_sources == ()
packed_volume_path = packed_volume_root / packed_volume_db.source('PERRY_VOLUME').relative_path
packed_volume_path.parent.mkdir(parents=True)
packed_volume_path.write_text('CAS\\tChemical\\tC1\\tC2\\tC3\\tC4\\tTmin\\tTmax\\n'
    '7732-18-5\\twater\\t2000\\t0.25\\t600\\t0.3\\t200\\t500\\n')
packed_volume_card = packed_volume_db.property_input('7732-18-5',
    ComponentSpec('packed-water','H2O',cas_number='7732-18-5'),method='PERRY_VOLUME')
assert packed_volume_card.evaluate_state(temperature_K=300,pressure_Pa=101325,
    phase='liquid',quantities=('density',)).properties['density'].value > 100
from chemworld.data.viscosity import ViscosityDatabase
from chemworld.physchem import viscosity_value, water_viscosity_iapws, wilke_viscosity
viscosity_data = ViscosityDatabase(sys.argv[1] + '.external-data')
assert len(viscosity_data.catalog) == 8 and viscosity_data.loaded_sources == ()
assert abs(water_viscosity_iapws(298.15, 998)-.000889735100149808) < 1e-14
assert viscosity_value('mu_air_lemmon', 300, rho=40) > 1.8e-5
assert wilke_viscosity([1], [1e-5], [28]) == 1e-5
from chemworld.physchem import (
    VISCOSITY_SCALES, viscosity_converter, viscosity_index, twu_1985_viscosity,
    przedziecki_sridhar_temperature_viscosity, lucas_liquid_viscosity,
    gas_mixture_viscosity_ledger, dense_mixture_viscosity_ledger,
)
assert len(VISCOSITY_SCALES) == 47
assert viscosity_index(6.394e-6, 2e-6, rounding=True) == 100
assert 3300e-6 < viscosity_converter((1.4*2.5)**.5, 'barbey', 'kinematic viscosity') < 4400e-6
assert abs(twu_1985_viscosity(338.7055, 672.3166, 895.5189)-.008235009644854494) < 1e-14
assert viscosity_value('Przedziecki_Sridhar', 383, Tm=178, Tc=591.8, Pc=41e5,
                       Vc=316e-6, Vf=95e-6, omega=.263, MW=92.14) > 0
assert lucas_liquid_viscosity(300, 5e7, critical_temperature_K=572.2,
    critical_pressure_Pa=3.47e6, acentric_factor=.236,
    saturation_pressure_Pa=2e5, saturation_viscosity_Pa_s=.00068) > .00068
from chemworld.physchem import conductivity_value, conductivity_derivatives
from chemworld.data.conductivity import ConductivityDatabase
conductivity_data = ConductivityDatabase(sys.argv[1] + '.external-data')
assert len(conductivity_data.catalog) == 4 and conductivity_data.loaded_sources == ()
assert abs(conductivity_value('Sheffy_Johnson', 300, MW=47, Tm=280)-.17740150413112193) < 1e-12
assert conductivity_derivatives('Nicola', 300,
    dict(MW=142.3, Tc=611.7, Pc=2110000, omega=.49))[1] < 0
from chemworld.physchem import filippov_conductivity, compressed_liquid_conductivity
assert abs(filippov_conductivity((.258, .742), (.1692, .1528))-.159291676288) < 1e-12
assert abs(compressed_liquid_conductivity(304, 633e6, method='Missenard',
    critical_temperature_K=591.8, critical_pressure_Pa=4.1e6,
    reference_conductivity_W_m_K=.129)-.2198375777069657) < 1e-12
from chemworld.physchem import water_conductivity_iapws, air_conductivity_lemmon
assert abs(water_conductivity_iapws(647.35, 750)-.5976194153179502) < 1e-12
assert abs(air_conductivity_lemmon(300, 40)-.0263839695044) < 1e-12
assert air_conductivity_lemmon(132.64, 10400, Cp=2000, Cv=35, mu=1.8e-5,
    drho_dP=0, drho_dP_reference=0) > 0
assert abs(conductivity_value('Eli_Hanley', 373.15, MW=72.151, Tc=460.4,
    Vc=3.06e-4, Zc=.267, omega=.227, Cvm=135.9)-.02247951724513664) < 1e-12
from chemworld.physchem import chung_dense_conductivity
assert abs(chung_dense_conductivity(473, MW=42.081, Tc=364.9, Vc=184.6e-6,
    omega=.142, Cvm=82.67, Vm=172.1e-6, mu=1.34e-5, dipole=.4)-.06160569232570781) < 1e-12
from chemworld.physchem import ely_hanley_dense_conductivity, stiel_thodos_dense_conductivity
assert abs(conductivity_value('Gharagheizi_gas',580,MW=16.04246,Tb=111.66,
    Pc=4599000,omega=.0115478)-.09594861261873211) < 1e-12
assert abs(stiel_thodos_dense_conductivity(378.15,MW=44.013,Tc=309.6,Pc=7240000,
    Vc=97.4e-6,Zc=.274,Vm=144e-6,kg=.0234)-.041245574404863684) < 1e-12
packed_ely=dict(MW=42.081,Tc=364.9,Vc=1.81e-4,Zc=.274,omega=.144,Cvm=82.70,Vm=1.721e-4)
packed_compatible=conductivity_value('Eli_Hanley_dense',473,**packed_ely)
assert abs(packed_compatible-.06038475754109959) < 1e-12
assert ely_hanley_dense_conductivity(473,**packed_ely) > packed_compatible
assert conductivity_value('Ely_Hanley_dense_translational',473,**packed_ely) == (
    ely_hanley_dense_conductivity(473,**packed_ely))
from pathlib import Path
import json
from chemworld.physchem import (
    volume_value, volume_derivatives, volume_correlation, ComponentPropertyPackage,
    evaluate_correlation, correlation_temperature_derivative,
)
assert abs(volume_value('Goodman',200,Tt=243.225,Vml=.00023585)-.0002053665090860923) < 1e-15
assert volume_derivatives('Bhirud_normal',500,{'Tc':500.,'Pc':4e6,'omega':.2},order=3)[0] > 0
packed_volume = volume_correlation('PPDS17',
    parameters={'Tc':562.05,'a0':.0115508,'a1':.281004,'a2':-.00635447},
    temperature_range_K=(280,500),source_note='NIST TDE example',output='density')
packed_volume = PropertyCorrelation.from_dict(json.loads(json.dumps(packed_volume.to_dict())))
packed_volume_package = ComponentPropertyPackage(ComponentSpec('benzene','C6H6'),(packed_volume,))
assert abs(packed_volume_package.evaluate('liquid_density',temperature_K=300).value
    -871.520087707) < 1e-8
assert correlation_temperature_derivative(packed_volume,temperature_K=300).to('g/(mL*K)').value < 0
packed_gas_volume = volume_correlation('ideal_gas',parameters={},
    temperature_range_K=(200,600),pressure_range_Pa=(1e5,1e6),source_note='ideal gas')
assert evaluate_correlation(packed_gas_volume,temperature_K=300,pressure_Pa=2e5).value == (
    evaluate_correlation(packed_gas_volume,temperature_K=300,pressure_Pa=4e5).value * 2)
from chemworld.physchem import (
    MixtureSpec, mixture_volume_correlation, mixture_volume_evaluation, evaluate_property_state,
    volume_pressure_derivatives,
)
packed_liquids = (ComponentSpec('methanol','CH4O'), ComponentSpec('water','H2O'))
packed_mix = MixtureSpec.from_mole_fractions(packed_liquids,(.4576,.5424),
    phase_label='liquid',temperature_K=298,pressure_Pa=1e7)
packed_mix_data = {'methanol':{'Tcs':512.58,'Vcs':.000117,'omegas':.559},
    'water':{'Tcs':647.29,'Vcs':5.6e-5,'omegas':.344}}
packed_mix_curve = mixture_volume_correlation(packed_mix,method='COSTALD_mixture_compressed',
    component_parameters=packed_mix_data,saturation_pressure_model='generalized_riedel',
    temperature_range_K=(280,350),pressure_range_Pa=(1e7,1e8),source_note='synthetic equation')
packed_mix_curve = PropertyCorrelation.from_dict(json.loads(json.dumps(packed_mix_curve.to_dict())))
packed_mix_reports = [evaluate_property_state(
    tuple(ComponentPropertyPackage(c,()) for c in packed_liquids),
    {'methanol':.4576,'water':.5424},temperature_K=298,pressure_Pa=p,phase='liquid',
    quantities=('density','molar_volume'),volume_model=packed_mix_curve) for p in (1e7,1e8)]
assert packed_mix_reports[1].properties['density'].value > (
    packed_mix_reports[0].properties['density'].value)
assert volume_pressure_derivatives('Tait_molar',300,
    {'P':1e7,'P_ref':1e5,'V_ref':1e-4,'B':1e8,'C':.1},order=3)[1] < 0
assert not any(name in sys.modules for name in ('chemicals','mpmath'))
from chemworld.data.interface import InterfaceDatabase
from chemworld.physchem import (
    interface_value, interface_derivatives, winterfeld_scriven_davis,
    water_oil_interfacial_tension, laplace_pressure, capillary_force,
)
from chemworld.physchem.property_inputs import PropertyInputCard
interface_data = InterfaceDatabase(sys.argv[1] + '.external-data')
assert len(interface_data.catalog) == 5 and interface_data.loaded_sources == ()
assert abs(interface_value('sigma_IAPWS', 300)-.0716859625271) < 1e-12
assert interface_derivatives('sigma_IAPWS', 647.096, {}) == (0, 0)
assert winterfeld_scriven_davis((1,), (0,), (10000,)) == 0
assert abs(water_oil_interfacial_tension(580, water_density_kg_m3=980,
    oil_density_kg_m3=760, oil_critical_temperature_K=914)-.02893598143089256) < 1e-12
interface_file = Path(sys.argv[1] + '.external-data') / 'Interface/Jasper-Lange.tsv'
interface_file.parent.mkdir(parents=True, exist_ok=True)
interface_file.write_text(chr(10).join((
    chr(9).join(('CAS','Name','a','b','Tmin','Tmax')),
    chr(9).join(('71-36-3','Synthetic','24','.0773','250','400')),
)), encoding='utf8')
interface_card = interface_data.property_input('71-36-3',
    ComponentSpec('packed-butanol','C4H10O',cas_number='71-36-3'), method='JASPER_LANGE')
interface_card = PropertyInputCard.from_dict(json.loads(json.dumps(interface_card.to_dict())))
interface_pressures = []
for t in (280,350):
    sigma = interface_card.evaluate_state(temperature_K=t, pressure_Pa=101325,
        phase='liquid', quantities=('surface_tension',)).properties['surface_tension']
    interface_pressures.append(laplace_pressure(sigma.to('mN/m'),
        curvature_1_per_m=1000, curvature_2_per_m=1000).value)
    assert capillary_force(sigma, wetted_perimeter_m=.02, contact_angle_rad=0).value > 0
assert interface_pressures[0] > interface_pressures[1] > 0
from chemworld.physchem import boiling_vaporization_enthalpy, watson_exponent, evaluate_correlation
from chemworld.physchem import (
    CorrelationSpeciesThermo, CaloricReference, stoichiometric_matrix, balance_stoichiometry,
    vapor_pressure_temperature_derivative, vapor_pressure_parameter_jacobian,
    water_saturation_pressure, water_saturation_temperature,
)
from chemworld.foundation.dimensions import dimension_definition, unmapped_supported_unit_dimensions
assert get_element('Og').atomic_number == 118
assert get_isotope('D').mass.value == 2.01410177784
assert ComponentSpec('salt', 'NH4+').charge == 1
assert atomic_data_provenance()['isotopes'] == 2939
assert IdentifierDatabase(sys.argv[1] + '.external-data').loaded_sources == ()
constants = ConstantDatabase(sys.argv[1] + '.external-data')
assert len(constants.catalog) == 43 and constants.loaded_sources == ()
from chemworld.data.electro_optical import ElectroOpticalDatabase
from chemworld.data.assessments import AssessmentDatabase, AssessmentValue
from chemworld.data.fuels import FuelDatabase, FUEL_SOURCES, source_octane_index
from chemworld.physchem import (
    combustion_data, CombustionData, mixture_combustion_spec, combustion_stoichiometry,
)
from chemworld.foundation.units import Quantity
from chemworld.physchem.specs import MixtureSpec
packed_fuel = combustion_data('CH4',Hf=-74520,method='Stoichiometry')
packed_fuel = CombustionData.from_dict(json.loads(json.dumps(packed_fuel.to_dict())))
assert packed_fuel.heat(Quantity(2,'mol')).value == -1781208
from chemworld.physchem import (
    temperature_quantity, temperature_input, temperature_observation, temperature_convert,
    its90_68_difference, henry_pressure, water_refractive_index_iapws,
    suzuki_flammability_limit,
)
from chemworld.data.scientific_inputs import InputValue
from chemworld.foundation.observations import Observation, ObservationChannel, ObservationTime
packed_t = temperature_quantity(Quantity(226.85,'degC'),'ITS-68','ITS-90')
assert abs(temperature_convert(packed_t.value,'ITS-90','ITS-68')-500) < 2e-10
packed_input = temperature_input(InputValue(300,'K',.02,'Packaged measurement'),
    'ITS-68','ITS-90')
packed_raw = ObservationChannel('temperature','K',300,uncertainty=.02,
    sample_id='packed',sampled_at=ObservationTime('world',4),available_at=ObservationTime('world',7))
packed_observation = temperature_observation(packed_raw,'ITS-68','ITS-90')
packed_observation = Observation.from_native_record(json.loads(json.dumps(
    packed_observation.native_record(),allow_nan=False)))
assert not packed_observation.errors()
packed_calibrated = packed_observation.channel('converted_temperature')
assert packed_calibrated.value == packed_input.value
assert packed_observation.channel('temperature') == packed_raw
assert henry_pressure(packed_calibrated.value,13,-300) != henry_pressure(300,13,-300)
assert water_refractive_index_iapws(packed_calibrated.value,997) != (
    water_refractive_index_iapws(300,997))
assert suzuki_flammability_limit(packed_fuel.heat(Quantity(1,'mol')).value,limit='LFL') > 0
assert its90_68_difference(77,convention='chemicals_1_5_2') == 0
assert its90_68_difference(77) > 0
assert combustion_stoichiometry('CH3Cl')['H2O'] == 1
fuel_root = Path(sys.argv[1] + '.fuel-data')
fuel_source = next(s for s in FUEL_SOURCES if s.source_id == 'FLORIAN_LIMING_ANN')
fuel_path = fuel_root / fuel_source.relative_path
fuel_path.parent.mkdir(parents=True,exist_ok=True)
fuel_path.write_text(chr(10).join((chr(9).join(('CAS','RON','MON')),
    chr(9).join(('64-17-5','100','90')))),encoding='utf8')
fuel_db = FuelDatabase(fuel_root)
assert len(fuel_db.catalog)==9 and fuel_db.loaded_sources==()
octanes = [fuel_db.fuel_value('64-17-5',p,method='FLORIAN_LIMING_ANN') for p in ('RON','MON')]
octanes = [AssessmentValue.from_dict(json.loads(json.dumps(v.to_dict()))) for v in octanes]
assert source_octane_index(*octanes).value == 95
burn_components = [ComponentSpec(f,f,default_phase='gas',cas_number=c) for f,c in (
    ('N2','7727-37-9'),('O2','7782-44-7'),('CH4','74-82-8'),
    ('H2O','7732-18-5'),('CO2','124-38-9'))]
burn_air = MixtureSpec.from_mole_fractions(burn_components,[.79,.21,0,0,0],
    phase_label='gas',temperature_K=300,pressure_Pa=101325)
burn_feed = MixtureSpec.from_mole_fractions(burn_components,[0,0,1,0,0],
    phase_label='gas',temperature_K=300,pressure_Pa=101325)
burn_result = mixture_combustion_spec(burn_air,burn_feed,burn_components,
    n_out=21,ratio=20,ratio_basis='mole')
assert burn_result['n_air']==20 and burn_result['n_fuel']==1
from chemworld.data.assessment_catalog import ASSESSMENT_SOURCE_MAP
from chemworld.foundation.units import Quantity
from chemworld.physchem import (
    iso_10156_lower_limit, emission_impact, octanol_water_partition,
    ppmv_to_mass_concentration, mass_concentration_to_ppmv, gas_exposure_comparison,
)
from chemworld.physchem.specs import MixtureSpec
import gzip
assessment_root = Path(sys.argv[1] + '.assessment-data')
assessment_data = AssessmentDatabase(assessment_root)
assert len(assessment_data.catalog) == 14 and assessment_data.loaded_sources == ()
for source_id, rows in (
    ('IPCC_2021', (('CAS','20yr GWP'),('74-82-8','80'))),
    ('SYRRES', (('CAS','logP'),('64-17-5','-1'))),
):
    source = ASSESSMENT_SOURCE_MAP[source_id]
    path = assessment_root / source.relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if source.kind == 'tsv-gzip' else open
    with opener(path, 'wt', encoding='utf8') as stream:
        stream.write(chr(10).join(chr(9).join(row) for row in rows))
gwp = assessment_data.assessment('74-82-8','GWP',method='IPCC (2021) 20yr')
gwp = AssessmentValue.from_dict(json.loads(json.dumps(gwp.to_dict())))
assert emission_impact({'74-82-8': Quantity(2,'kg')}, {'74-82-8':gwp}).value == 160
logp = assessment_data.assessment('64-17-5','logP',method='SYRRES')
logp = AssessmentValue.from_dict(json.loads(json.dumps(logp.to_dict())))
split = octanol_water_partition({'64-17-5':1}, {'64-17-5':logp},
    water_volume_L=1,octanol_volume_L=.25,temperature_K=300)
assert abs(split.organic_amounts_mol['64-17-5']-.025/1.025) < 1e-12
assert split.material_balance_error_mol < 1e-12
ontario_path = assessment_root / ASSESSMENT_SOURCE_MAP['ONTARIO'].relative_path
ontario_path.parent.mkdir(parents=True, exist_ok=True)
ontario_path.write_text(json.dumps({'74-82-8':{
    'Name':'Synthetic methane','TWA (ppm)':10,'TWA (mg/m^3)':None,
    'Skin':False,'MW':16.043}}), encoding='utf8')
exposure = assessment_data.assessment('74-82-8','TWA',method='Ontario Limits')
exposure = AssessmentValue.from_dict(json.loads(json.dumps(exposure.to_dict())))
gas = MixtureSpec.from_mole_fractions((
    ComponentSpec('methane','CH4',default_phase='gas'),
    ComponentSpec('nitrogen','N2',default_phase='gas')), (5e-6,1-5e-6),
    phase_label='gas',temperature_K=300,pressure_Pa=101325)
assert gas_exposure_comparison(gas,'methane',exposure,source_identity='74-82-8',
    averaging_window='TWA').value == .5
assert iso_10156_lower_limit([.5,.5],[.05,None],['74-82-8','7727-37-9']) == .1
assert abs(Quantity(10,'ppmv').to('gas_volume_fraction').value-1e-5) < 1e-18
concentration = ppmv_to_mass_concentration(100,40,temperature_K=300,pressure_Pa=101325)
assert abs(mass_concentration_to_ppmv(concentration,40,
    temperature_K=300,pressure_Pa=101325)-100) < 1e-10
from chemworld.physchem import (
    water_permittivity_iapws, water_refractive_index_iapws,
    brix_to_refractive_index, brix_reading, parallel_plate_capacitance,
)
assert abs(water_permittivity_iapws(300,1000)-78.03317817364899) < 1e-11
assert water_refractive_index_iapws(300,1000) > 1
assert brix_to_refractive_index(95) > brix_to_refractive_index(85)
optical_data = ElectroOpticalDatabase(sys.argv[1] + '.external-data')
assert len(optical_data.catalog) == 7 and optical_data.loaded_sources == ()
for filename, rows in (
    ('Electrolytes/Permittivity (Dielectric Constant) of Liquids.tsv', (
        ('CAS','Chemical','T','Permittivity','A','B','C','D','Tmin','Tmax'),
        ('64-17-5','Synthetic','300','25','55','-.1','','','280','350'))),
    ('Misc/CRC Handbook Organic RI.csv', (
        ('CAS','RI','RIT'),('64-17-5','1.3611','293.15'))),
):
    path = Path(sys.argv[1] + '.external-data') / filename
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(chr(10).join(chr(9).join(row) for row in rows),encoding='utf8')
for method, prop, temperature in (
    ('CRC_PERMITTIVITY','relative_permittivity',300),
    ('CRC_RI','refractive_index',293.15),
):
    card = optical_data.property_input('64-17-5',
        ComponentSpec('packed-ethanol','C2H6O',cas_number='64-17-5'),
        method=method,source_point=(method=='CRC_RI'))
    card = PropertyInputCard.from_dict(json.loads(json.dumps(card.to_dict())))
    result = card.package.evaluate(prop,temperature_K=temperature)
    if prop == 'relative_permittivity':
        assert parallel_plate_capacitance(result,area_m2=.01,separation_m=.001).value > 0
    else:
        assert brix_reading(result).quantity().to('degBx').value > 0
from chemworld.physchem import (
    henry_convert, henry_pressure, henry_pressure_mixture, henry_raoult_flash,
    HansenParameters, hansen_distance, solid_solubility,
)
from chemworld.physchem.specs import MixtureSpec, PropertyCorrelation
from chemworld.physchem.property_reports import PropertyEvaluation
assert henry_convert(101325, 'SI', 'KHpx') == 1
assert henry_convert(2, 'Hbp', 'mol/(kg*atm)') == 2
assert abs(henry_pressure(300, 15, 300, .04, .001, 100, .00001)-37105004.47898146) < 1e-7
assert 0 < solid_solubility(300,350,12000) < 1
hansen_path = Path(sys.argv[1] + '.external-data') / 'Misc/hspipy_hansen_solubility_parameters.tsv'
hansen_path.parent.mkdir(parents=True,exist_ok=True)
hansen_path.write_text(chr(10).join((
    chr(9).join(('CAS','HANSEN_DELTA_D','HANSEN_DELTA_P','HANSEN_DELTA_H')),
    chr(9).join(('64-17-5','15000','8000','19000')),
    chr(9).join(('71-36-3','16000','6000','16000')),
)),encoding='utf8')
hansen_vectors = [HansenParameters.from_record(r) for r in constants.records('HSPIPY_HANSEN')]
hansen_vectors = [HansenParameters.from_dict(json.loads(json.dumps(v.to_dict())))
                  for v in hansen_vectors]
assert 4000 < hansen_distance(*hansen_vectors).value < 4200
packed_henry = PropertyCorrelation('synthetic-gas-in-water','henry_pressure',
    {'A':12.206072645530174},{'temperature':'K'},'Pa','henry_pressure',
    {'temperature':(250,400)},'Synthetic source',{'solvent_id':'water'})
packed_henry = PropertyCorrelation.from_dict(json.loads(json.dumps(packed_henry.to_dict())))
packed_h = evaluate_correlation(packed_henry,temperature_K=300)
packed_mixture = MixtureSpec.from_mole_fractions((ComponentSpec('water','H2O'),
    ComponentSpec('gas','CO2')),(.7,.3),phase_label='liquid',temperature_K=300,pressure_Pa=50000)
packed_flash = henry_raoult_flash(packed_mixture,henry_evaluations={'gas':packed_h},
    solvent_vapor_pressures={'water':PropertyEvaluation('vapor_pressure','synthetic','constant',
        20000,'Pa',{'temperature':300})})
assert 0 < packed_flash.vapor_fraction < 1
for identity,z in zip(packed_mixture.component_ids,packed_mixture.mole_fractions):
    assert abs((1-packed_flash.vapor_fraction)*packed_flash.liquid_composition[identity]
        +packed_flash.vapor_fraction*packed_flash.vapor_composition[identity]-z) < 1e-9
assert dimension_definition('parachor').vector.mass == .25
assert unmapped_supported_unit_dimensions() == ()
cp = PropertyCorrelation('packed-cp', 'poling_cp', dict(a=4, b=0, c=0, d=0, e=0),
    {'temperature': 'K'}, 'J/(mol*K)', 'ideal_gas_heat_capacity',
    {'temperature': (200, 600)}, 'Synthetic packaged formula')
curve = PiecewiseHeatCapacity('packed-ranges', (cp,), 'Synthetic packaged caloric input')
root = temperature_from_caloric_change(curve, 4*8.31446261815324*100,
    reference_temperature_K=300)
assert abs(root-400) < 1e-8
assert vibrational_temperature(667) > 900
caloric = CaloricDatabase(sys.argv[1] + '.external-data')
assert len(caloric.catalog) == 21 and caloric.loaded_sources == ()
assert balance_stoichiometry(stoichiometric_matrix(
    [{'H': 2}, {'O': 2}, {'H': 2, 'O': 1}], [True, True, False])) == (2, 1, 2)
thermo = CorrelationSpeciesThermo('packed', curve,
    CaloricReference(phase='gas', energy_J_mol=-1000, entropy_J_mol_K=100))
assert thermo.evaluate(400).enthalpy_J_mol > -1000
tdb = TdbDatabase.from_string('FUNCTION F 100 42; 1000 N !', source=InputSource(
    DataSourceProvenance('packed-tdb', 0, 'Synthetic packaged TDB', '1', 'CC0-1.0'),
    'synthetic'))
assert tdb.evaluate_function('F', 300) == 42
assert TdbDatabase.from_dict(tdb.to_dict()).to_dict() == tdb.to_dict()
from chemworld.physchem.calphad import CefPhaseModel
cef_input = TdbDatabase.from_string(chr(10).join([
    'ELEMENT FE BCC_A2 55.845 0 0 !', 'TYPE_DEF % SEQ * !',
    'PHASE ALPHA % 1 1 !', 'CONSTITUENT ALPHA :FE: !',
    'PARAMETER G(ALPHA,FE;0) 100 1000+T; 1000 N !',
]), source=tdb.source)
cef = CefPhaseModel(cef_input, 'ALPHA')
assert cef.evaluate([{'FE': 1}], 500).enthalpy_J_mol == 1000
assert CefPhaseModel.from_dict(cef.to_dict()).to_dict() == cef.to_dict()
from chemworld.data import ChemSageDatabase
cs_text = chr(10).join(('Synthetic packaged ChemSage Cp', '1 1 0 1', 'FE', '55.845',
    '6 1 2 3 4 5 6', '1 1', 'ALPHA', '10 1 1', '1234 32.1',
    '1800 30 0 0 0', '1 .3 .5'))+chr(10)
cs_input = ChemSageDatabase.from_string(cs_text, source=tdb.source)
cs_model = CefPhaseModel(cs_input, 'ALPHA')
assert abs(cs_model.temperature_response([{'FE': 1}], 600
    ).fixed_constitution_heat_capacity_J_mol_K-(30+.3*600**.5)) < 1e-9
assert CefPhaseModel.from_dict(cs_model.to_dict()).database.text == cs_text
from chemworld.physchem import MQMQAPhaseModel
mq_text = chr(10).join(('Synthetic packaged MQMQA', '2 2 0 1 0', 'FE O', '55.845 15.999',
    '6 1 2 3 4 5 6', '1 1', 'SLAG', 'SUBQ', '1 1', 'FEO', '1 1 1 1',
    '1800 0 0 -60 0 0 0', '1 1 0 0 0', '2.4', '1 1', 'FE', 'O', '2', '1',
    '2', '1', '1', '1', '1 1 2 2 4 4 4 4', '0'))+chr(10)
mq_input = ChemSageDatabase.from_string(mq_text, source=InputSource(DataSourceProvenance(
    'packed-mqmqa', 0, 'Synthetic packaged MQMQA', '1', 'CC0-1.0'), 'synthetic'))
mq_model = MQMQAPhaseModel(mq_input, 'SLAG')
assert abs(mq_model.temperature_response([{'Q1_1_2_2': 1}], 600
    ).fixed_constitution_heat_capacity_J_mol_K-30) < 1e-9
assert MQMQAPhaseModel.from_dict(mq_model.to_dict()).database.text == mq_text
assert cef.temperature_response([{'FE': 1}], 500).fixed_constitution_heat_capacity_J_mol_K == 0
response = cef.derivatives([{'FE': 1}], 500)
assert response.hessian.shape == (4, 3, 3)
mu = cef.chemical_potentials([{'FE': 1}], 500)
assert mu.status == 'determined' and abs(mu.values_J_mol['FE']-1500) < 1e-8
constitution = cef.stationary_constitution({'FE': 1}, [{'FE': 1}], 500)
assert constitution.success and abs(constitution.relaxed_heat_capacity_J_mol_K) < 1e-9
from chemworld.physchem import CefMaterialSpec
from chemworld.world.solid_world import calphad_world_spec
from chemworld.runtime import ChemWorldRuntime
cs_material = CefMaterialSpec(cs_model, ({'FE': 1},), 500, 700, 600, .01)
cs_world = calphad_world_spec(cs_material,
    element_amounts_mol={'FE': .001}, temperature_K=600)
cs_runtime = ChemWorldRuntime(composition=cs_world)
cs_runtime.reset(seed=40)
cs_runtime = ChemWorldRuntime.from_snapshot(cs_runtime.snapshot())
cs_result = cs_runtime.execute({'operation': 'heat', 'target_temperature_K': 610,
    'duration_s': 1})
assert cs_result.info['transaction_status'] == 'committed'
assert 600 < cs_runtime._state.temperature_K < 610
mq_material = CefMaterialSpec(mq_model, ({'Q1_1_2_2': 1},), 500, 700, 600, .01,
    phase_type='liquid')
mq_world = calphad_world_spec(mq_material,
    element_amounts_mol={'FE': .0005, 'O': .0005}, temperature_K=600)
mq_runtime = ChemWorldRuntime(composition=mq_world)
mq_runtime.reset(seed=40)
mq_runtime = ChemWorldRuntime.from_snapshot(mq_runtime.snapshot())
mq_step = mq_runtime.execute({'operation': 'heat', 'target_temperature_K': 610, 'duration_s': 1})
assert mq_step.info['transaction_status'] == 'committed', mq_step.info
assert 600 < mq_runtime._state.temperature_K < 610
from chemworld.runtime.solid_solutions import solid_enthalpy
packed_material = CefMaterialSpec(cef, ({'FE': 1},), 400, 800, 500, .01)
packed_world = calphad_world_spec(packed_material,
    element_amounts_mol={'FE': .001}, temperature_K=500)
packed_runtime = ChemWorldRuntime(composition=packed_world)
packed_runtime.reset(seed=40)
packed_before_H = solid_enthalpy(packed_runtime._state, packed_material)
packed_step = packed_runtime.execute(
    {'operation': 'heat', 'target_temperature_K': 510, 'duration_s': 1})
assert packed_step.info['transaction_status'] == 'committed', packed_step.info
assert packed_runtime._state.species_amounts == {'FE': .001}
assert abs(solid_enthalpy(packed_runtime._state, packed_material) - packed_before_H
    - packed_runtime._state.ledger.energy_jacket_J) < 1e-7
assert ChemWorldRuntime.from_snapshot(packed_runtime.snapshot())._state.to_dict() == (
    packed_runtime._state.to_dict())
pressure_data = PressureDatabase(sys.argv[1] + '.external-data')
assert len(pressure_data.catalog) == 10 and pressure_data.loaded_sources == ()
pressure = PropertyCorrelation('packed-pressure', 'wagner25',
    dict(Tc=600, Pc=3e6, a=-7, b=1, c=-1, d=-2), {'temperature': 'K'}, 'Pa',
    'vapor_pressure', {'temperature': (250, 600)}, 'Synthetic packaged pressure input')
assert vapor_pressure_temperature_derivative(pressure, temperature_K=350, order=2).valid
jacobian = vapor_pressure_parameter_jacobian(pressure, (350,), parameters=('a', 'b'))
assert len(jacobian.values[0]) == 2
assert abs(water_saturation_temperature(water_saturation_pressure(373.15))-373.15) < 2e-8
phase_data = PhaseChangeDatabase(sys.argv[1] + '.external-data')
assert len(phase_data.catalog) == 10 and phase_data.loaded_sources == ()
latent = PropertyCorrelation('packed-latent', 'velasco_hvap', dict(Tc=600, omega=.2),
    {'temperature': 'K'}, 'J/mol', 'heat_of_vaporization')
assert evaluate_correlation(latent, temperature_K=350).value > 0
assert boiling_vaporization_enthalpy(method='Chen', boiling_temperature_K=300,
    critical_temperature_K=600, critical_pressure_Pa=3e6) > 0
assert abs(watson_exponent(temperature_1_K=300, temperature_2_K=400,
    enthalpy_1_J_mol=30000, enthalpy_2_J_mol=30000*(2/3)**.38,
    critical_temperature_K=600)-.38) < 1e-12
from chemworld.physchem import (
    CriticalEstimate, ComponentSpec, estimate_critical_property,
    eos_component_from_critical_inputs, mixture_critical_property,
)
from chemworld.data.scientific_inputs import InputValue
packed_component = ComponentSpec('packed-decane', 'C10H22')
packed_Tc = estimate_critical_property(packed_component, method='Tb_Tc_relationship',
    parameters={'Tb': 447.3, 'fit': 'Perry8E'})
packed_Vc = estimate_critical_property(packed_component, method='Mersmann_Kind', parameters={})
packed_Pc = estimate_critical_property(packed_component, method='Ihmels',
    parameters={'Tc': packed_Tc, 'Vc': packed_Vc})
assert CriticalEstimate.from_dict(packed_Tc.to_dict()) == packed_Tc
assert eos_component_from_critical_inputs(packed_component, Tc=packed_Tc, Pc=packed_Pc,
    omega=InputValue(.2, 'dimensionless', None, 'Explicit synthetic packed input')
    ).critical_pressure_Pa > 0
assert mixture_critical_property('Li', (.5, .5), Tcs=(400, 500), Vcs=(.0002, .0002)) == 450
from chemworld.physchem import hekayati_raeissi_critical, CubicEOSSpec, evaluate_cubic_eos
packed_hr_component = ComponentSpec('packed-toluene', 'C7H8')
packed_hr_inputs = {'MW':InputValue(.09213842,'kg/mol',None,'Explicit molecular weight'),
    'V_sat':InputValue(.10686,'L/mol',None,'Saturated liquid at 298.15 K'),
    'Pc':InputValue(41.08,'bar',None,'Explicit critical pressure')}
packed_hr_Tc = estimate_critical_property(packed_hr_component, method='Hekayati_Raeissi',
    parameters=packed_hr_inputs, output_property='Tc')
assert abs(packed_hr_Tc.value-599.7965819136947) < 1e-9
assert CriticalEstimate.from_dict(packed_hr_Tc.to_dict()) == packed_hr_Tc
packed_hr_spec = eos_component_from_critical_inputs(packed_hr_component,Tc=packed_hr_Tc,
    Pc=packed_hr_inputs['Pc'],omega=InputValue(.2,'dimensionless',None,'Explicit omega'))
packed_hr_state = evaluate_cubic_eos(CubicEOSSpec('packed-hr','peng_robinson',(packed_hr_spec,)),
    {'packed-toluene':1},temperature_K=700,pressure_Pa=1e6)
assert packed_hr_state.molar_volume_m3_mol > 0
assert hekayati_raeissi_critical(MW=92.13842,Vc=.000316,Pc=4108000)[0] > 0
from chemworld.physchem import LennardJonesSpec, lennard_jones_collision_integral
from chemworld.data.lennard_jones import LennardJonesDatabase
assert len(LennardJonesDatabase(sys.argv[1] + '.external-data').catalog) == 2
packed_lj = LennardJonesSpec('packed-decane', 300, 5e-10, 'Synthetic packaged LJ parameters')
packed_viscosity = packed_lj.viscosity_correlation(packed_component,
    collision_method='Kim_Monroe', temperature_range_K=(200, 1000))
assert evaluate_correlation(packed_viscosity, temperature_K=500).value > 0
assert lennard_jones_collision_integral(99, method='Neufeld_Janzen_Aziz', l=1, s=7) > 0
assert 'chemicals' not in sys.modules and 'periodictable' not in sys.modules
from chemworld.data import SBMLModel
sbml_text = '''<sbml xmlns="http://www.sbml.org/sbml/level3/version2/core" level="3"
version="2"><model timeUnits="second" substanceUnits="mole" volumeUnits="litre">
<listOfParameters><parameter id="x" value="2" units="dimensionless" constant="true"/>
<parameter id="y" units="dimensionless" constant="false"/></listOfParameters><listOfRules>
<assignmentRule variable="y"><math xmlns="http://www.w3.org/1998/Math/MathML"><apply>
<power/><ci>x</ci><cn>3</cn></apply></math></assignmentRule></listOfRules></model></sbml>'''
sbml_source = InputSource(DataSourceProvenance('packaged-sbml', 0,
    'Self-authored packaged declarative model', '1', 'CC0-1.0'), 'synthetic')
sbml_model = SBMLModel(sbml_text, sbml_source)
sbml_rule = sbml_model.declaration.items('listOfRules')[0].mathematics
assert sbml_rule.evaluate({'x': 2.}) == 8.
assert SBMLModel.from_dict(sbml_model.to_dict()).to_xml() == sbml_text
assert sbml_model.model_unit('volumeUnits').volume_l(.1) == .1
from chemworld.physchem.sbml_kinetics import SBMLKinetics
sbml_equations = SBMLKinetics(sbml_model)
assert dict(sbml_equations.initial_values) == {}
assert sbml_equations.context({}, 0)['y'] == 8.
from chemworld.physchem.coordinate_hybrid import CoordinateHybridMemory, advance_coordinate_events
from chemworld.physchem.coupled_system import CoupledState
from chemworld.physchem.sbml_events import SBMLTriggerRoots, sbml_coordinate_events
from chemworld.physchem.sbml_kinetics import sbml_continuous_system
packed_event_xml = sbml_text.replace('</listOfParameters>', '''<parameter id="z" value="0"
constant="false" units="dimensionless"/></listOfParameters>''').replace('</model>', '''
<listOfEvents><event id="queued" useValuesFromTriggerTime="true"><trigger initialValue="false"
persistent="true"><math xmlns="http://www.w3.org/1998/Math/MathML"><true/></math></trigger>
<delay><math xmlns="http://www.w3.org/1998/Math/MathML"><cn>0.2</cn></math></delay>
<listOfEventAssignments><eventAssignment variable="z"><math
xmlns="http://www.w3.org/1998/Math/MathML"><cn>3</cn></math></eventAssignment>
</listOfEventAssignments></event></listOfEvents></model>''')
packed_frame = SBMLKinetics(SBMLModel(packed_event_xml, sbml_source))
packed_events = sbml_coordinate_events(packed_frame)
packed_roots = SBMLTriggerRoots(packed_frame)
packed_continuous = sbml_continuous_system(packed_frame)
packed_before = advance_coordinate_events(packed_continuous,
    CoupledState({}, 298.15, 1., coordinates=packed_frame.initial_values), .1,
    CoordinateHybridMemory.initialize(packed_events), packed_events,
    roots=packed_roots.roots, trigger_values=packed_roots.trigger_values)
assert len(packed_before.memory.events.pending) == 1
packed_resume = CoordinateHybridMemory.from_dict(json.loads(json.dumps(
    packed_before.memory.to_dict(), allow_nan=False)))
packed_after = advance_coordinate_events(packed_continuous, packed_before.final_state, .1,
    packed_resume, packed_events, roots=packed_roots.roots,
    trigger_values=packed_roots.trigger_values)
assert packed_after.final_state.coordinates['P:z'] == 3.
assert packed_after.memory.events.pending == () and len(packed_after.events) == 1
from chemworld.world.sbml_world import sbml_world
from chemworld.physchem.sbml_system import SBMLObservable, parameter_observable
from chemworld.runtime import ChemWorldRuntime
from chemworld.runtime.sbml_models import model_record
packed_world_xml = packed_event_xml.replace('<listOfParameters>', '''<listOfCompartments>
<compartment id="cell" size="1" spatialDimensions="3" units="litre" constant="true"/>
</listOfCompartments><listOfParameters>''')
packed_source_model = SBMLModel(packed_world_xml,sbml_source)
packed_native = ChemWorldRuntime(composition=sbml_world(packed_source_model,
    maximum_step_s=.02, controlled_parameters=('x',), readout_relative_sd=0.,
    observables=(SBMLObservable('volume','cell'),parameter_observable(packed_source_model,'x'))))
packed_native.reset(seed=3)
assert packed_native.execute({'operation':'wait','duration_s':.1}).info[
    'operation_outcome']=='success'
assert len(model_record(packed_native._state).settings['continuation']['events']['pending']) == 1
assert packed_native.execute({'operation':'set_model_parameters','values':{'x':4.}}).info[
    'operation_outcome']=='success'
packed_native = ChemWorldRuntime.from_snapshot(json.loads(json.dumps(
    packed_native.snapshot().to_dict(),allow_nan=False)))
assert packed_native.execute({'operation':'wait','duration_s':.1}).info[
    'operation_outcome']=='success'
assert model_record(packed_native._state).settings['coordinates']['P:z'] == 3.
assert packed_native.execute({'operation':'measure','instrument':'formal_quantity_meter'}).info[
    'operation_outcome']=='success'
assert packed_native.observe().channels[0].unit == 'L'
assert packed_native.observe().channels[1].unit == 'dimensionless'
assert packed_native.observe().channels[1].value == 4.
assert 'coordinates' not in packed_native._state.to_dict(include_hidden=False)['equipment'][
    'formal_model']['settings']
from chemworld.data.sedml_declarations import SEDMLDocument
from chemworld.data.sedml_models import resolve_models
from chemworld.data.sedml_simulations import resolve_simulations
from chemworld.data.combine_archive import ArchiveEntry, CombineArchive, COMBINE_URI
packed_sedml_text = '''<sedML xmlns="http://sed-ml.org/sed-ml/level1/version4"
level="1" version="4" xmlns:s="http://www.sbml.org/sbml/level3/version2/core">
<listOfModels><model id="base" language="urn:sedml:language:sbml" source="model.xml"/>
<model id="derived" language="urn:sedml:language:sbml" source="#base"><listOfChanges>
<changeAttribute target="/s:sbml/s:model/s:listOfParameters/s:parameter[@id='x']/@value"
newValue="3"/></listOfChanges></model></listOfModels><listOfSimulations>
<uniformTimeCourse id="course" initialTime="0" outputStartTime="0" outputEndTime="0.4"
numberOfSteps="2"><algorithm kisaoID="KISAO:0000288"><listOfAlgorithmParameters>
<algorithmParameter kisaoID="KISAO:0000467" value="0.02"/></listOfAlgorithmParameters>
</algorithm></uniformTimeCourse></listOfSimulations><listOfTasks>
<task id="task" modelReference="derived" simulationReference="course"/></listOfTasks></sedML>'''
packed_sedml = SEDMLDocument(packed_sedml_text, sbml_source)
packed_models = resolve_models(packed_sedml, {'model.xml':SBMLModel(packed_world_xml,sbml_source)})
packed_derived = SBMLKinetics(packed_models['derived'])
assert packed_derived.context(packed_derived.initial_values,0.)['y'] == 27.
assert resolve_simulations(packed_sedml)['course'].source_times == (0.,.2,.4)
packed_archive = CombineArchive((
    ArchiveEntry('model.xml',COMBINE_URI+'sbml',packed_world_xml.encode('utf-8')),
    ArchiveEntry('experiment.sedml',COMBINE_URI+'sed-ml',packed_sedml_text.encode('utf-8'),True),
),sbml_source)
packed_archive = CombineArchive.from_bytes(packed_archive.to_bytes(),source=sbml_source)
assert packed_archive.document().to_xml() == packed_sedml_text
assert packed_archive.entry('model.xml').data.decode('utf-8') == packed_world_xml
from chemworld.world.sedml_protocol import SEDMLProtocol
from chemworld.data.sedml_reports import SEDMLArray
packed_scan = '''<repeatedTask id="scan" range="grid" resetModel="true" concatenate="false">
<listOfRanges><vectorRange id="grid"><value>2</value><value>4</value></vectorRange></listOfRanges>
<listOfChanges><setValue modelReference="derived" range="grid"
target="/s:sbml/s:model/s:listOfParameters/s:parameter[@id='x']/@value"><math
xmlns="http://www.w3.org/1998/Math/MathML"><ci>grid</ci></math></setValue></listOfChanges>
<listOfSubTasks><subTask task="task"/></listOfSubTasks></repeatedTask>'''
packed_outputs = '''<listOfDataGenerators><dataGenerator id="read_x"><listOfVariables>
<variable id="vx" taskReference="scan"
target="/s:sbml/s:model/s:listOfParameters/s:parameter[@id='x']"/></listOfVariables>
<math xmlns="http://www.w3.org/1998/Math/MathML"><ci>vx</ci></math></dataGenerator>
</listOfDataGenerators><listOfOutputs><report id="report"><listOfDataSets>
<dataSet id="column" label="x" dataReference="read_x"/></listOfDataSets></report></listOfOutputs>'''
packed_protocol_text = packed_sedml_text.replace('</listOfTasks>',packed_scan+'</listOfTasks>'
    ).replace('</sedML>',packed_outputs+'</sedML>')
packed_plan = SEDMLProtocol(SEDMLDocument(packed_protocol_text,sbml_source),
    {'model.xml':SBMLModel(packed_world_xml,sbml_source)})
packed_plan = SEDMLProtocol.from_dict(json.loads(json.dumps(packed_plan.to_dict(),allow_nan=False)))
assert packed_plan.tasks.invocations['scan'] == 2 and packed_plan.tasks.samples['scan'] == 6
assert packed_plan.worlds['derived'].physics.models_of('sbml')[
    'formal_workspace'].controlled_parameters == ('x',)
packed_x = SEDMLArray((2.,2.,2.,4.,4.,4.),(2,3),('scan:grid','time'),
    tuple((i,t) for i in range(2) for t in (0.,.2,.4)),'dimensionless')
assert packed_plan.reports.evaluate({('read_x','vx'):packed_x})['report'][0]['data'] == (
    packed_x.to_dict())
from chemworld.eval.sedml_experiment import run_sedml_experiment
packed_experiment = run_sedml_experiment(packed_plan,seed=5)
assert packed_experiment.reports['report'][0]['data']['values'] == list(packed_x.values)
assert packed_experiment.reports['report'][0]['data']['shape'] == [2,3]
assert all(record.info['operation_outcome']=='success' for record in packed_experiment.history)
assert ChemWorldRuntime.from_snapshot(packed_experiment.snapshots['scan']['derived']).lab_status()[
    'time_s']==.4
assert all(name not in sys.modules for name in ('COPASI', 'libsbml', 'mpmath'))
assert 'pycalphad' not in sys.modules
from chemworld.physchem.battery_models import BatterySPMSpec, ParticleSpec
from chemworld.physchem.battery_dfn import BatteryDFNSpec
from chemworld.physchem.battery_electrolyte import BatteryElectrolyteSpec
from chemworld.physchem.battery_impedance import battery_impedance
from chemworld.physchem.battery_materials import ElectrodeOCV
from chemworld.physchem import soh_from_battery, battery_spec_at_soc, soh_spec_from_dict
from dataclasses import replace
packed_soh_cell = BatterySPMSpec(minimum_voltage_V=3.38, maximum_voltage_V=3.92)
packed_soh_state = packed_soh_cell.initial_state()
packed_soh = soh_from_battery(packed_soh_cell, packed_soh_state)
assert packed_soh.status == 'unique'
assert soh_spec_from_dict(json.loads(json.dumps(packed_soh.spec.to_dict()))) == packed_soh.spec
assert abs(packed_soh.theoretical_energy_j() - packed_soh.branch()['cell_capacity_C']*3.65) < 1e-8
assert abs(battery_spec_at_soc(packed_soh_cell, packed_soh, .5).initial_negative_fraction-.5
    ) < 1e-10
assert packed_soh_state == packed_soh_cell.initial_state()
assert packed_soh.soc_at_voltage(3.65).status == 'unique'
from chemworld.physchem import BatteryMPMSpec, ParticleSizeDistribution, mpm_fields
from chemworld.physchem.battery_models import advance_battery, battery_spec_from_dict
from chemworld.physchem.reactor_shared import ScalarControlProgram
packed_psd = ParticleSizeDistribution((5e-6,15e-6),(.5,.5),sbml_source)
packed_mpm = BatteryMPMSpec(negative=ParticleSpec(cells=4),positive=ParticleSpec(cells=4),
    negative_distribution=packed_psd, positive_distribution=packed_psd)
assert battery_spec_from_dict(json.loads(json.dumps(packed_mpm.to_dict()))) == packed_mpm
packed_mpm_fields = mpm_fields(packed_mpm, packed_mpm.initial_state(), current_A=.2)
assert max(abs(x) for x in packed_mpm_fields['bv_residual_V']) < 1e-10
packed_mpm_step = advance_battery(packed_mpm, packed_mpm.initial_state(),
    ScalarControlProgram(((0.,.2),),'A'), .1)
assert packed_mpm_step.outcome == 'success'
assert abs(packed_mpm_step.final_state.coordinates['discharged_charge_C']-.02) < 1e-8
from chemworld.physchem import TheveninSpec, TabulatedScalar, thevenin_system, thevenin_fields
from chemworld.physchem.battery_thevenin import CONSTANT_UNITS
from chemworld.data.scientific_inputs import (
    InputValue, QuantityColumn, ScientificRow, ScientificTable,
)
def packed_constant(identity, value, unit):
    table = ScientificTable(identity, 'parameters',
        (QuantityColumn('value',unit,'parameter',(-1e9,1e9),'Synthetic package fixture'),),
        (sbml_source,), (ScientificRow('value',sbml_source.provenance.source_id,
        'model','constants','parameters',
        {'value':InputValue(value,unit,None,'Uncertainty not supplied')}),))
    return TabulatedScalar(table,(),'value',unit)
packed_ecm_values = dict(capacity_C=100., maximum_current_A=1., minimum_voltage_V=.1,
    maximum_voltage_V=5., initial_soc=.7, minimum_soc=0., maximum_soc=1.,
    initial_cell_temperature_K=298.15, initial_jig_temperature_K=298.15,
    minimum_temperature_K=250., maximum_temperature_K=400., cell_heat_capacity_J_K=100.,
    jig_heat_capacity_J_K=500., cell_jig_conductance_W_K=.5,
    jig_ambient_conductance_W_K=.2, reference_frame_temperature_K=298.15, housing_volume_L=.01)
packed_ecm_parameters = ScientificTable('packed-ecm', 'parameters', tuple(
    QuantityColumn(key,unit,'parameter',(-1e9,1e9),'Synthetic package fixture')
    for key,unit in CONSTANT_UNITS.items()), (sbml_source,),
    (ScientificRow('cell',sbml_source.provenance.source_id,'model','constants','parameters',
    {key:InputValue(value,CONSTANT_UNITS[key],None,'Uncertainty not supplied')
    for key,value in packed_ecm_values.items()}),))
packed_ecm = TheveninSpec(packed_ecm_parameters,'cell',packed_constant('ocv',3.7,'V'),
    packed_constant('r0',.1,'Ohm'),(),packed_constant('entropy',0.,'V/K'),
    ScalarControlProgram(((0.,298.15),),'K'))
packed_ecm = TheveninSpec.from_dict(json.loads(json.dumps(packed_ecm.to_dict(),allow_nan=False)))
packed_ecm_control = ScalarControlProgram(((0.,.2),),'A')
packed_ecm_step = thevenin_system(packed_ecm,packed_ecm_control).advance(
    packed_ecm.initial_state(),.1)
assert packed_ecm_step.outcome == 'success'
assert abs(packed_ecm_step.final_state.coordinates['thevenin:soc']-.6998) < 1e-9
assert abs(thevenin_fields(packed_ecm,packed_ecm_step.final_state,
    packed_ecm_control)['voltage_V']-3.68) < 1e-9
assert packed_ecm_step.energy_balance_residual_J is None
from chemworld.examples.native_thevenin import default_spec as circuit_fixture, run as run_circuit
from chemworld.world.thevenin_world import thevenin_world_spec
from chemworld.physchem.battery_thevenin import SOC as circuit_soc
from chemworld.runtime.thevenin import unpack_thevenin
packed_circuit = ChemWorldRuntime(composition=thevenin_world_spec(circuit_fixture()))
packed_circuit.reset(seed=9)
assert packed_circuit.execute({'operation':'configure_electrical_control',
    'current_A':.2}).info['transaction_status'] == 'committed'
assert packed_circuit.execute({'operation':'measure',
    'instrument':'thevenin_meter'}).info['actual_duration_s'] == 1.
assert abs(unpack_thevenin(packed_circuit._state)[3].coordinates[circuit_soc]-.698) < 1e-9
packed_circuit = ChemWorldRuntime.from_snapshot(json.loads(json.dumps(
    packed_circuit.snapshot().to_dict(),sort_keys=True,allow_nan=False)))
assert packed_circuit.execute({'operation':'wait','duration_s':.1}).info[
    'transaction_status'] == 'committed'
assert packed_circuit._state.ledger.time_s == 1.1
assert run_circuit(sys.argv[1]+'.thevenin.jsonl')['replay_steps'] == 9
assert run_circuit(sys.argv[1]+'.thevenin-analysis.jsonl',
    include_impedance_analysis=True)['replay_steps'] == 10
from chemworld.physchem import TheveninDiffusionSpec, DIFFUSION_CONVENTION
from chemworld.examples.native_thevenin import diffusion_spec
packed_diffusive_spec = diffusion_spec(cells=4)
assert isinstance(packed_diffusive_spec.diffusion, TheveninDiffusionSpec)
assert packed_diffusive_spec.diffusion.convention == DIFFUSION_CONVENTION
packed_diffusive_spec = TheveninSpec.from_dict(json.loads(json.dumps(
    packed_diffusive_spec.to_dict(), sort_keys=True, allow_nan=False)))
packed_diffusive = ChemWorldRuntime(composition=thevenin_world_spec(packed_diffusive_spec))
packed_diffusive.reset(seed=9)
assert packed_diffusive.execute({'operation':'configure_electrical_control',
    'current_A':.2}).info['transaction_status'] == 'committed'
assert packed_diffusive.execute({'operation':'wait','duration_s':1.}).info[
    'transaction_status'] == 'committed'
packed_spatial_state = unpack_thevenin(packed_diffusive._state)[3]
assert packed_diffusive_spec.diffusion.surface_soc(packed_spatial_state.coordinates[circuit_soc],
    packed_spatial_state.coordinates) < packed_spatial_state.coordinates[circuit_soc]
packed_diffusive = ChemWorldRuntime.from_snapshot(json.loads(json.dumps(
    packed_diffusive.snapshot().to_dict(), sort_keys=True, allow_nan=False)))
assert unpack_thevenin(packed_diffusive._state)[3] == packed_spatial_state
assert run_circuit(sys.argv[1]+'.thevenin-diffusion.jsonl', spec=packed_diffusive_spec,
    include_impedance_analysis=True)['replay_steps'] == 10
from chemworld.physchem import TheveninControlSpec, TheveninControlProgram, thevenin_control_program
from chemworld.examples.native_thevenin import control_spec
packed_control_spec = control_spec(packed_diffusive_spec)
assert isinstance(packed_control_spec.control, TheveninControlSpec)
packed_named = TheveninControlProgram('power', ScalarControlProgram(((0.,.5),),'W'),
    current_branch_A=(0.,1.))
assert thevenin_control_program(packed_named.to_dict()) == packed_named
assert run_circuit(sys.argv[1]+'.thevenin-named-control.jsonl', spec=packed_control_spec,
    include_named_controls=True, include_impedance_analysis=True)['replay_steps'] == 18
from chemworld.physchem import ElectrodeOCPSpec
from chemworld.examples.battery_hysteresis import (
    default_spec as hysteresis_spec, run as run_hysteresis)
for packed_many_particle in (False, True):
    packed_hysteresis = hysteresis_spec(many_particle=packed_many_particle, cells=4)
    assert isinstance(packed_hysteresis.negative_ocp, ElectrodeOCPSpec)
    assert len(packed_hysteresis.hysteresis_keys) == 2
    assert run_hysteresis(sys.argv[1]+f'.hysteresis-{packed_many_particle}.jsonl',
        spec=packed_hysteresis)['replay_steps'] == 11
from chemworld.physchem import BatteryHalfCellSPMSpec, LithiumMetalCounterElectrodeSpec
from chemworld.examples.battery_half_cell import (
    default_spec as half_cell_spec, run as run_half_cell)
packed_half_cell = half_cell_spec(cells=4)
assert isinstance(packed_half_cell, BatteryHalfCellSPMSpec)
assert isinstance(packed_half_cell.metal, LithiumMetalCounterElectrodeSpec)
assert not hasattr(packed_half_cell, 'negative')
assert run_half_cell(sys.argv[1]+'.half-cell.jsonl',
    spec=packed_half_cell)['replay_steps'] == 11
from chemworld.examples.battery_half_cell_impedance import run as run_half_cell_eis
packed_half_eis = run_half_cell_eis(sys.argv[1]+'.half-cell-eis.json',
    spec=battery_spec_from_dict(packed_half_cell.to_dict()))
assert packed_half_eis['spectrum']['success'] and packed_half_eis['state_unchanged']
for packed_inventory in packed_half_eis['inventory_responses']:
    packed_values = {k: complex(v['real'], v['imag'])
                     for k,v in packed_inventory['responses'].items()}
    assert abs(packed_values['metal_Li_mol_per_A']
               +packed_values['working_Li_mol_per_A']) < 1e-10
assert len(packed_half_eis['full_response']['variables']) == len(packed_half_cell.inventory_keys)+4
from chemworld.physchem import ActiveMaterialPhaseSpec, BatteryCompositeSPMSpec
from chemworld.examples.battery_composite import (
    default_spec as composite_spec, run as run_composite)
packed_composite = composite_spec(cells=4)
assert isinstance(packed_composite, BatteryCompositeSPMSpec)
assert len(packed_composite.negative_phases) == 2
assert all(isinstance(phase, ActiveMaterialPhaseSpec) for phase in packed_composite.negative_phases)
assert not hasattr(packed_composite, 'negative')
assert run_composite(sys.argv[1]+'.composite.jsonl',
    spec=packed_composite)['replay_steps'] == 11
from chemworld.physchem.battery_impedance import BatteryImpedance, battery_impedance_system
from chemworld.examples.battery_composite_impedance import (
    default_spec as composite_eis_spec, run as run_composite_eis)
packed_composite_eis = battery_spec_from_dict(composite_eis_spec(cells=4).to_dict())
packed_eis_state = packed_composite_eis.initial_state()
packed_eis_system, _ = battery_impedance_system(packed_composite_eis, packed_eis_state)
packed_eis = battery_impedance(packed_composite_eis, [.001,.01], state=packed_eis_state)
assert isinstance(packed_eis, BatteryImpedance) and packed_eis.success
assert len(packed_eis_system.variables) == len(packed_eis.response.variables)
assert packed_eis_state == packed_composite_eis.initial_state()
packed_eis_report = run_composite_eis(sys.argv[1]+'.composite-eis.json', spec=packed_composite_eis)
assert packed_eis_report['spectrum']['success'] and packed_eis_report['state_unchanged']
for packed_phase in packed_eis_report['phase_responses']:
    packed_q = {k: complex(v['real'],v['imag'])
                for k,v in packed_phase['phase_current_response_A_per_A'].items()}
    assert abs(packed_q['negative:phase:alpha']+packed_q['negative:phase:beta']-1) < 1e-10
    assert abs(packed_q['positive:phase:positive']+1) < 1e-10
assert json.loads(json.dumps(packed_eis.response.to_dict(),allow_nan=False))['success']
from chemworld.physchem import BatterySpatialThermalSPMSpec, BatteryHeatStackSpec
from chemworld.examples.battery_spatial_thermal import (
    default_spec as spatial_thermal_spec, run as run_spatial_thermal)
packed_spatial_thermal = spatial_thermal_spec(cells=4, thermal_cells=2)
assert isinstance(packed_spatial_thermal, BatterySpatialThermalSPMSpec)
assert isinstance(packed_spatial_thermal.heat_stack, BatteryHeatStackSpec)
assert len(packed_spatial_thermal.thermal_coordinate_keys) == 7
assert run_spatial_thermal(sys.argv[1]+'.spatial-thermal.jsonl',
    spec=packed_spatial_thermal)['replay_steps'] == 11
from chemworld.physchem import BatteryMSMRSPMSpec, MSMRMaterialSpec
from chemworld.examples.battery_msmr import default_spec as msmr_spec, run as run_msmr
packed_msmr = msmr_spec(cells=4)
assert isinstance(packed_msmr, BatteryMSMRSPMSpec)
assert isinstance(packed_msmr.negative_material, MSMRMaterialSpec)
assert len(packed_msmr.inventory_keys) == 8
assert run_msmr(sys.argv[1]+'.msmr.jsonl', spec=packed_msmr)['replay_steps'] == 11
from chemworld.examples.battery_msmr_impedance import run as run_msmr_eis
packed_msmr_eis = run_msmr_eis(sys.argv[1]+'.msmr-eis.json',
    spec=battery_spec_from_dict(packed_msmr.to_dict()))
assert packed_msmr_eis['spectrum']['success'] and packed_msmr_eis['state_unchanged']
assert packed_msmr_eis['reaction_rows']['negative'] == list(
    packed_msmr.negative_material.reaction_row_ids)
for packed_inventory in packed_msmr_eis['inventory_responses']:
    packed_values = {k: complex(v['real'], v['imag'])
                     for k,v in packed_inventory['responses'].items()}
    assert abs(packed_values['negative_Li_mol_per_A']
               +packed_values['positive_Li_mol_per_A']) < 1e-10
from chemworld.physchem import (AveragedElectrodeSpec, BatteryNewmanTobiasSpec,
    PorousElectrodeGeometrySpec)
from chemworld.examples.battery_newman_tobias import (
    default_spec as newman_tobias_spec, run as run_newman_tobias)
packed_nt = newman_tobias_spec(cells=(3,2,3))
assert isinstance(packed_nt, BatteryNewmanTobiasSpec)
assert isinstance(packed_nt.negative, AveragedElectrodeSpec)
assert isinstance(packed_nt.geometry, PorousElectrodeGeometrySpec)
assert set(packed_nt.inventory_keys) == {'negative:Li','positive:Li'}
assert not hasattr(packed_nt.negative, 'diffusivity_m2_s')
assert run_newman_tobias(sys.argv[1]+'.newman-tobias.jsonl',
    spec=packed_nt)['replay_steps'] == 11
from chemworld.examples.battery_newman_tobias_impedance import run as run_nt_eis
packed_nt_eis = run_nt_eis(sys.argv[1]+'.newman-tobias-eis.json', spec=packed_nt)
assert packed_nt_eis['spectrum']['success'] and packed_nt_eis['state_unchanged']
assert packed_nt_eis['reaction_current_variables'] == list(packed_nt.reaction_keys)
assert json.loads(json.dumps(packed_nt_eis['full_response'],allow_nan=False))['success']
for packed_inventory in packed_nt_eis['inventory_responses']:
    packed_values = {k: complex(v['real'], v['imag'])
                     for k,v in packed_inventory['responses'].items()}
    assert abs(packed_values['negative_Li_mol_per_A']
               +packed_values['positive_Li_mol_per_A']) < 1e-10
for packed_spatial in packed_nt_eis['cell_current_responses']:
    packed_currents = {k: complex(v['real'], v['imag'])
                       for k,v in packed_spatial['cell_current_A_per_A'].items()}
    assert len(packed_currents) == 6
    for packed_electrode, packed_expected in (('negative',1.),('positive',-1.)):
        assert abs(sum(v for k,v in packed_currents.items()
                       if k.startswith(packed_electrode+':'))-packed_expected) < 1e-10
from chemworld.physchem import (BatteryLeadAcidSpec, LeadAcidGeometrySpec,
    LeadAcidMaterialSpec, LeadAcidReadout, lead_acid_readout)
from chemworld.examples.battery_lead_acid import (
    default_spec as lead_acid_spec, run as run_lead_acid,
)
packed_lead_acid = lead_acid_spec()
assert isinstance(packed_lead_acid, BatteryLeadAcidSpec)
assert isinstance(packed_lead_acid.geometry, LeadAcidGeometrySpec)
assert isinstance(packed_lead_acid.material, LeadAcidMaterialSpec)
assert len(packed_lead_acid.inventory_keys) == 6
assert 'Li' not in packed_lead_acid.inventory_species.values()
assert isinstance(lead_acid_readout(packed_lead_acid,
    packed_lead_acid.initial_state(), 0), LeadAcidReadout)
assert run_lead_acid(sys.argv[1]+'.lead-acid.jsonl',
    spec=packed_lead_acid)['replay_steps'] == 11
from chemworld.physchem import BatterySplitOCVRSpec, SplitOCVRReadout, split_ocvr_readout
from chemworld.examples.battery_split_ocvr import (
    default_spec as split_ocvr_spec, run as run_split_ocvr,
)
packed_split_ocvr = split_ocvr_spec()
assert isinstance(packed_split_ocvr, BatterySplitOCVRSpec)
assert not hasattr(packed_split_ocvr, 'negative')
assert isinstance(split_ocvr_readout(packed_split_ocvr,
    packed_split_ocvr.initial_state(), 0.), SplitOCVRReadout)
assert run_split_ocvr(sys.argv[1]+'.split-ocvr.jsonl',
    spec=packed_split_ocvr)['replay_steps'] == 11
from chemworld.physchem import LithiumPlatingSpec, PlatingReadout, plating_readout
from chemworld.examples.battery_plating import (
    default_spec as plating_spec, run as run_plating,
)
packed_plating = plating_spec()
assert isinstance(packed_plating.plating, LithiumPlatingSpec)
assert {'plating:Li','dead:Li'} <= set(packed_plating.inventory_keys)
assert isinstance(plating_readout(packed_plating,
    packed_plating.initial_state(), 0.), PlatingReadout)
assert BatterySPMSpec(**json.loads(json.dumps(packed_plating.to_dict()))) == packed_plating
assert run_plating(sys.argv[1]+'.plating.jsonl',
    spec=packed_plating)['replay_steps'] == 11
from chemworld.examples.battery_mpm_plating import (
    default_spec as mpm_plating_spec, run as run_mpm_plating,
)
from chemworld.physchem.battery_models import battery_spec_from_dict
packed_mpm_plating = mpm_plating_spec()
assert isinstance(packed_mpm_plating, BatteryMPMSpec)
assert packed_mpm_plating.counts == (3,2)
assert battery_spec_from_dict(json.loads(json.dumps(
    packed_mpm_plating.to_dict()))) == packed_mpm_plating
assert run_mpm_plating(sys.argv[1]+'.mpm-plating.jsonl',
    spec=packed_mpm_plating)['replay_steps'] == 11
from chemworld.examples.battery_partial_plating import (
    default_spec as partial_plating_spec, run as run_partial_plating,
)
packed_partial_plating = partial_plating_spec()
assert packed_partial_plating.plating.mode == 'partially reversible'
assert packed_partial_plating.plating.decay_rate_s_inv == .02
assert battery_spec_from_dict(json.loads(json.dumps(
    packed_partial_plating.to_dict()))) == packed_partial_plating
assert run_partial_plating(sys.argv[1]+'.partial-plating.jsonl',
    spec=packed_partial_plating)['replay_steps'] == 11
from chemworld.examples.battery_axen_plating import (
    default_spec as axen_plating_spec, run as run_axen_plating,
)
packed_axen_plating = axen_plating_spec()
assert packed_axen_plating.plating.mode == 'reversible'
assert len(packed_axen_plating.hysteresis_keys) == 2
assert battery_spec_from_dict(json.loads(json.dumps(
    packed_axen_plating.to_dict()))) == packed_axen_plating
assert run_axen_plating(sys.argv[1]+'.axen-plating.jsonl',
    spec=packed_axen_plating)['replay_steps'] == 11

packed_ocv = ElectrodeOCV('packaged-electrode', sbml_source, (.2,-.1),
    tanh_terms=((-0.02,5.,.4),))
assert ElectrodeOCV.from_dict(packed_ocv.to_dict()) == packed_ocv
packed_particles = dict(negative=ParticleSpec(cells=8), positive=ParticleSpec(cells=8))
packed_electrolyte = BatteryElectrolyteSpec(cells=(3,2,3))
for packed_cell in (
    BatterySPMSpec(**packed_particles, negative_ocv=packed_ocv),
    BatterySPMSpec(**packed_particles, electrolyte=packed_electrolyte),
    BatteryDFNSpec(**packed_particles, electrolyte=packed_electrolyte),
):
    packed_cell_state = packed_cell.initial_state()
    packed_spectrum = battery_impedance(packed_cell, [.001,.01], state=packed_cell_state)
    assert packed_spectrum.success
    assert all(z.real > 0 and z.imag < 0 for z in packed_spectrum.impedance_ohm)
    assert packed_cell_state == packed_cell.initial_state()
    assert json.loads(json.dumps(packed_spectrum.to_dict(), allow_nan=False))['unit'] == 'ohm'
assert 'pybamm' not in sys.modules
import math
from chemworld.physchem import (
    VIRIAL_METHODS, virial_correlation, virial_temperature_report,
    virial_value, tsonopoulos_polar_parameters, ComponentUncertainty,
)
from chemworld.physchem.property_inputs import PropertyInputCard
for virial_method, virial_spec in VIRIAL_METHODS.items():
    virial_all = dict(Tc=500.,Pc=5e6,Vc=.0002,omega=.2,a=.03,b=.01)
    virial_p = {k:virial_all[k] for k in virial_spec.required.split()}
    virial_curve = virial_correlation(virial_method,parameters=virial_p,
        temperature_range_K=(300,1500),source_note='Synthetic installed qualification')
    virial_card = PropertyInputCard(
        ComponentPropertyPackage(ComponentSpec('packed-virial','CH4'),(virial_curve,)),
        (sbml_source,),
        {virial_curve.correlation_id: ComponentUncertainty(
            virial_curve.correlation_id,unit=virial_curve.output_unit,
            source_id=sbml_source.provenance.source_id,coverage='unspecified',
            note='Synthetic installed qualification')},
        None,'Virial coefficients only, no caloric data')
    virial_restored = PropertyInputCard.from_json(json.dumps(virial_card.to_dict()))
    virial_result = virial_restored.package.evaluate(virial_spec.property_id,temperature_K=600)
    assert abs(virial_result.value-virial_value(virial_method,600,**virial_p)) < 1e-15
    virial_reports = virial_temperature_report(
        virial_restored.package.correlations[0],temperature_K=600)
    assert len(virial_reports)==4 and all(math.isfinite(r.to(r.unit).value) for r in virial_reports)
assert tsonopoulos_polar_parameters(Tc=500,Pc=5e6,species_type='methanol')['a']==.0878
from chemworld.physchem import (
    virial_cross_parameters, virial_coefficient_matrices, virial_mixture_response,
    density_virial_state, virial_pressure, density_to_pressure_virial,
    pressure_virial_volume, pressure_virial_state,
)
cross = virial_cross_parameters([200.,400.],[4e6,5e6],[1e-4,2e-4],[.01,.2])
cross_b = [[{k:v for k,v in cell.items() if k != 'Vc'} for cell in row] for row in cross]
mixed_b = virial_mixture_response([.4,.6],virial_coefficient_matrices('Tsonopoulos',500.,cross_b))
gas = density_virial_state(500.,1e5,(mixed_b.value,1e-8))
assert abs(virial_pressure(500.,gas.molar_volume_m3_mol,(mixed_b.value,1e-8))-1e5)<1e-5
pressure_coeffs = density_to_pressure_virial(500.,(mixed_b.value,1e-8))
pressure_volume = pressure_virial_volume(500.,1e5,pressure_coeffs)
assert abs(pressure_virial_state(500.,pressure_volume,pressure_coeffs).pressure_Pa-1e5)<1e-5
assert json.loads(json.dumps(gas.to_dict(),allow_nan=False))['expansion']=='density'
from chemworld.physchem import VirialGasThermodynamics, virial_gas_batch_system
from chemworld.physchem.property_packages import SpeciesCaloricModel
from chemworld.physchem.property_reports import CaloricReference
from chemworld.physchem.coupled_system import CoupledState, HeatFlux
packed_crc_path = packed_volume_root / packed_volume_db.source('CRC_VIRIAL').relative_path
packed_crc_path.write_text('CAS\\tChemical\\ta1\\ta2\\ta3\\ta4\\ta5\\n'
    '74-82-8\\tmethane\\t-42\\t-123\\t-20\\t0\\t0\\n')
packed_gas_component = ComponentSpec('packed-methane','CH4',cas_number='74-82-8')
packed_crc_card = packed_volume_db.property_input('74-82-8', packed_gas_component,
    method='CRC_VIRIAL',temperature_bounds_K=(200,800))
packed_crc_card = PropertyInputCard.from_json(json.dumps(packed_crc_card.to_dict()))
packed_crc_state = packed_crc_card.evaluate_state(temperature_K=300,pressure_Pa=1e5,
    phase='gas',quantities=('molar_volume','density'),gas_model='virial')
assert packed_crc_state.model_details['root']['candidates']
packed_critical_root = Path(sys.argv[1]+'.virial-constants')
packed_critical_db = ConstantDatabase(packed_critical_root)
packed_critical_path = packed_critical_root / packed_critical_db.source('HEOS').relative_path
packed_critical_path.parent.mkdir(parents=True)
packed_critical_path.write_text('CAS\\tname\\tTc\\tPc\\tVc\\tomega\\n'
    '74-82-8\\tmethane\\t190.564\\t4599200\\t0.00009863\\t0.01142\\n')
packed_c_card = packed_critical_db.virial_input('74-82-8',packed_gas_component,
    method='Liu_Xiang',constant_source='HEOS',temperature_range_K=(200,800))
packed_gas_package = ComponentPropertyPackage(packed_gas_component,
    (packed_crc_card.package.correlations[0],packed_c_card.package.correlations[0]))
packed_gas_ideal = SpeciesCaloricModel(PropertyCorrelation(correlation_id='packed-ideal-cv',
    equation_id='cp_polynomial',property_id='isochoric_heat_capacity',coefficients={'a':25},
    input_units={'temperature':'K'},output_unit='J/(mol*K)',
    validity_ranges={'temperature':(200,800)},source_note='Synthetic installed Cv'),
    CaloricReference(phase='gas',temperature_K=300,energy_basis='internal_energy'))
packed_thermal = VirialGasThermodynamics(packed_gas_package,packed_gas_ideal)
packed_thermal = VirialGasThermodynamics.from_dict(json.loads(json.dumps(packed_thermal.to_dict())))
packed_heating = virial_gas_batch_system(packed_thermal,heat_flux=lambda _:HeatFlux(heater_W=5))
packed_gas_run = packed_heating.advance(CoupledState({'packed-methane':2},300,10),1)
assert packed_gas_run.final_state.time_s == 1
assert packed_gas_run.final_state.temperature_K > 300
assert abs(packed_gas_run.energy_balance_residual_J) < 1e-4
from dataclasses import replace as packed_replace
from chemworld.physchem.property_inputs import PropertyInputCard
from chemworld.physchem.specs import ComponentUncertainty
from chemworld.physchem.transport_devices import LiquidTransportProperties, FiniteTransportSpec
from chemworld.world.transport_world import synthetic_transport_properties, transport_world_spec
from chemworld.physchem import ideal_gas_transport_state
packed_transport_base = synthetic_transport_properties().packages[1]
packed_transport_component = packed_replace(packed_transport_base.component, cas_number='64-17-5')
packed_mu_path = Path(sys.argv[1] + '.external-data') / viscosity_data.source(
    'DUTT_PRASAD_VISCOSITY').relative_path
packed_mu_path.parent.mkdir(parents=True, exist_ok=True)
packed_mu_path.write_text('CAS\\tChemical\\tA\\tB\\tC\\tTmin\\tTmax\\n'
    '64-17-5\\tsynthetic source record\\t-2.8857\\t-1032.53\\t-55.95\\t248\\t348\\n')
packed_mu_input = viscosity_data.property_input('64-17-5', packed_transport_component,
    method='DUTT_PRASAD_VISCOSITY')
packed_source = InputSource(DataSourceProvenance('packed-transport-companion', 0,
    'Synthetic transport companions, not material measurements', version='1',
    license='synthetic fixture'), 'synthetic')
packed_curves = tuple(c for c in packed_transport_base.correlations
    if c.property_id != 'liquid_viscosity') + (PropertyCorrelation(
    'packed-psat','dippr101',{'A':9.903487552536127,'B':0},{'temperature':'K'},'Pa',
    'vapor_pressure',{'temperature':(280,340)},'Synthetic saturation-pressure baseline'),)
packed_companion = PropertyInputCard(ComponentPropertyPackage(packed_transport_component,
    packed_curves),(packed_source,),{c.correlation_id:ComponentUncertainty(c.correlation_id,
    unit=c.output_unit,source_id=packed_source.provenance.source_id,coverage='synthetic',
    note='Declared companion, not material accuracy') for c in packed_curves},
    CaloricReference(phase='liquid',pressure_Pa=2e5),'Explicit companion caloric datum')
packed_complete = PropertyInputCard.combine((packed_mu_input,packed_companion),
    reference_note='Explicit source-selected installed transport input')
packed_complete = PropertyInputCard.from_json(json.dumps(packed_complete.to_dict()))
packed_transport = LiquidTransportProperties((packed_complete.package,),(280,340),2e5,
    {'ethanol':packed_complete.reference},{'ethanol':{'critical_temperature_K':572.2,
    'critical_pressure_Pa':3.47e6,'acentric_factor':.236}})
packed_flow = ChemWorldRuntime(composition=transport_world_spec(
    FiniteTransportSpec(packed_transport)))
packed_flow.reset(seed=24)
assert packed_flow.execute({'operation':'set_flow_rate','flow_rate_mL_min':20}).info[
    'transaction_status'] == 'committed'
assert packed_flow.execute({'operation':'run_flow','duration_s':1}).info[
    'transaction_status'] == 'committed'
packed_flow_copy = ChemWorldRuntime.from_snapshot(json.loads(json.dumps(
    packed_flow.snapshot().to_dict(),allow_nan=False)))
for packed_session in (packed_flow,packed_flow_copy):
    assert packed_session.execute({'operation':'run_flow','duration_s':1}).info[
        'transaction_status'] == 'committed'
assert packed_flow._state.to_dict() == packed_flow_copy._state.to_dict()
packed_flow.close()
packed_flow_copy.close()
packed_gas_component = ComponentSpec('packed-transport-gas','N2',default_phase='gas')
packed_gas_curves = tuple(PropertyCorrelation('packed-gas-'+prop,'dippr100',{'A':value},
    {'temperature':'K'},unit,prop,{'temperature':(280,460)},'Synthetic installed gas',
    {'phase':'gas'}) for prop,value,unit in (
    ('ideal_gas_heat_capacity',30,'J/(mol*K)'),('gas_viscosity',2e-5,'Pa*s'),
    ('thermal_conductivity',.03,'W/(m*K)')))
packed_fluid,_,packed_volume = ideal_gas_transport_state(
    (ComponentPropertyPackage(packed_gas_component,packed_gas_curves),),
    {'packed-transport-gas':.01},400,1e5,
    {'packed-transport-gas':CaloricReference(phase='gas')})
assert packed_volume > 0 and packed_fluid.viscosity_Pa_s == 2e-5
from chemworld.physchem import critical_correlation_input, critical_compressibility
packed_zc = estimate_critical_property(packed_component, method='Zc_definition',
    parameters={'Tc':packed_Tc,'Pc':packed_Pc,'Vc':packed_Vc})
assert packed_zc.value == critical_compressibility(packed_Tc.value,packed_Pc.value,packed_Vc.value)
packed_derived_card = critical_correlation_input(packed_component, family='volume',
    method='Rackett', parameters={'Tc':packed_Tc,'Pc':packed_Pc,'Zc':packed_zc},
    temperature_range_K=(280,340), source=packed_source)
packed_derived_copy = PropertyInputCard.from_json(json.dumps(packed_derived_card.to_dict()))
assert packed_derived_copy.package.correlations[0].metadata['parameter_sources'][
    'Zc']['method'] == 'Zc_definition'
assert packed_derived_copy.package.evaluate('liquid_molar_volume',temperature_K=300).value > 0
packed_mixture_base = synthetic_transport_properties()
packed_mixture = packed_replace(packed_mixture_base, volume_mixing_method='COSTALD_mixture',
    volume_mixing_parameters={k:{'Tcs':600+50*i,'Vcs':.0002+.0001*i,'omegas':.2+.1*i}
        for i,k in enumerate(packed_mixture_base.component_ids)},
    volume_mixing_note='Explicit synthetic installed volume law; additive enthalpy')
packed_mixture_copy = LiquidTransportProperties.from_dict(json.loads(json.dumps(
    packed_mixture.to_dict())))
assert packed_mixture_copy.evaluate({'water':.2,'ethanol':.8},300) == packed_mixture.evaluate(
    {'water':.2,'ethanol':.8},300)
packed_flow = ChemWorldRuntime(composition=transport_world_spec(FiniteTransportSpec(
    packed_mixture_copy),feed_mole_fractions={'water':.8,'ethanol':.2},
    receiver_mole_fractions={'water':.2,'ethanol':.8}))
packed_flow.reset(seed=24)
for packed_action in ({'operation':'set_flow_rate','flow_rate_mL_min':6},
                      {'operation':'run_flow','duration_s':1}):
    assert packed_flow.execute(packed_action).info['transaction_status'] == 'committed'
packed_copy = ChemWorldRuntime.from_snapshot(json.loads(json.dumps(
    packed_flow.snapshot().to_dict())))
for packed_session in (packed_flow,packed_copy):
    assert packed_session.execute({'operation':'run_flow','duration_s':1}).info[
        'transaction_status'] == 'committed'
assert packed_flow._state.to_dict() == packed_copy._state.to_dict()
packed_flow.close()
packed_copy.close()
from chemworld.physchem import PropertySelectionPolicy
from chemworld.examples.native_flash_research import synthetic_vle_properties
from chemworld.physchem.flash_units import VLEPropertyPackage, tp_flash
packed_vle = synthetic_vle_properties()
packed_light = packed_vle.packages[0]
packed_original_psat = packed_light.by_property('vapor_pressure')[0]
packed_selected_psat = packed_replace(packed_original_psat, correlation_id='explicit-packed-psat',
    coefficients={**packed_original_psat.coefficients,
                  'A':math.log(1e5)+30000/8.31446261815324/345},
    metadata={'source':{'source_id':'packed-selected-source','record_id':'explicit-row'}})
packed_light = packed_replace(packed_light,
    correlations=(*packed_light.correlations,packed_selected_psat)).with_selection({
    'vapor_pressure':PropertySelectionPolicy('source_priority',
                                            source_priority=('packed-selected-source',))})
packed_selected_vle = packed_replace(packed_vle,packages=(packed_light,packed_vle.packages[1]))
packed_selected_vle = VLEPropertyPackage.from_dict(json.loads(json.dumps(
    packed_selected_vle.to_dict())))
packed_selected_flash = tp_flash(packed_selected_vle,{'light':.5,'heavy':.5},
    temperature_K=350,pressure_Pa=1e5,feed_enthalpy_J=1000)
packed_default_flash = tp_flash(packed_vle,{'light':.5,'heavy':.5},
    temperature_K=350,pressure_Pa=1e5)
assert packed_default_flash.vapor_fraction > 0
assert packed_selected_flash.vapor_fraction == 0
assert abs(packed_selected_flash.total_enthalpy_j-(90*(350-298.15)-35000)) < 1e-6
assert packed_selected_vle.packages[0].selection_report('vapor_pressure',temperature_K=350
    ).selected_id == 'explicit-packed-psat'
packed_query = fuel_db.inspect_value('64-17-5','RON',methods=('FLORIAN_LIMING_ANN',))
assert packed_query['methods'][0]['records'][0]['value']['value'] == 100
assert packed_query['selected_record_id'] is None
packed_query = IdentifierDatabase(sys.argv[1] + '.missing-identities').inspect_search(
    '7732-18-5',kind='cas',source_ids=('pubchem-small','absent'))
assert [s['status'] for s in packed_query['sources']] == ['source_unavailable','unknown_method']
assert packed_query['candidates'] == []
assert 'chemicals' not in sys.modules
from chemworld.physchem import (
    dippr_temperature_response, dippr_parameter_jacobian, dippr_mixed_derivative,
    dippr106_coefficients_from_point,
)
packed_dippr = PropertyCorrelation('packed-derivatives', 'dippr100',
    {'A':25., 'B':.01, 'C':.0002}, {'temperature':'K'}, 'J/(mol*K)',
    'ideal_gas_heat_capacity', {'temperature':(200,500)}, 'Synthetic installed test')
packed_response = dippr_temperature_response(packed_dippr, temperature_K=300)
assert all(abs(a-e) < 1e-12 for a,e in zip(packed_response.values, (46.,.13,.0004,0.)))
assert dippr_parameter_jacobian(packed_dippr, [300], parameters=('B','C')).values == ((300,90000),)
assert dippr_mixed_derivative(packed_dippr, temperature_K=300,
    coordinates=('temperature','temperature','C')).value == 2
packed_local = dippr106_coefficients_from_point(temperature_K=300,
    critical_temperature_K=650, value=50000*(1-300/650)**.4,
    first_derivative=-50000*.4/650*(1-300/650)**(-.6))
assert abs(packed_local['A']-50000) < 1e-6 and abs(packed_local['B']-.4) < 1e-12
assert all(name not in sys.modules for name in ('chemicals','mpmath'))
from chemworld.data.correlation_calibration import CorrelationCalibration, CorrelationFitParameter
from chemworld.data.scientific_inputs import ScientificRow, ScientificTable, QuantityColumn
packed_fit_source = InputSource(DataSourceProvenance('packed-fit', 0, 'Synthetic installed Cp',
    version='1', license='MIT'), 'synthetic')
packed_fit_columns = (
    QuantityColumn('temperature','K','condition',(200,500),'absolute temperature'),
    QuantityColumn('value','J/(mol*K)','observation',(0,100),'synthetic Cp'),
)
packed_fit_rows = tuple(ScientificRow(str(i),'packed-fit',str(i),str(i),
    'train' if i < 4 else 'validation', {
        'temperature': InputValue(T,'K',0.,'exact synthetic input'),
        'value': InputValue(30+.02*T,'J/(mol*K)',.1,'declared synthetic sigma'),
    }) for i,T in enumerate((250.,300.,350.,400.,275.,375.)))
packed_fit_table = ScientificTable('packed-fit','measurements',packed_fit_columns,
    (packed_fit_source,), packed_fit_rows)
packed_fit_curve = PropertyCorrelation('packed-fit-seed','dippr100',{'A':32.,'B':.01},
    {'temperature':'K'},'J/(mol*K)','ideal_gas_heat_capacity',{'temperature':(240,450)})
packed_fit = CorrelationCalibration(packed_fit_table, packed_fit_curve,
    (CorrelationFitParameter('A',(10,60),32), CorrelationFitParameter('B',(.001,.1),.01)),
    fitted_correlation_id='packed-fitted').fit(max_nfev=100)
assert packed_fit['status'] == 'fitted' and packed_fit['validation']['weighted_rmse'] < 1e-8
assert abs(packed_fit['estimate']['A']-30) < 1e-8 and abs(packed_fit['estimate']['B']-.02) < 1e-8
assert packed_fit['identifiability']['rank'] == 2
assert all(name not in sys.modules for name in ('chemicals','mpmath'))
from chemworld.physchem import (
    solve_stream_composition, partial_molar_properties, weighted_property,
    heat_capacity_difference, acoustic_velocity, gravity_scale,
)
packed_flows, packed_zs, packed_ws = solve_stream_composition(
    (2.,None,None),(None,.3,None),(None,None,350/518),(18.,44.,70.))
assert all(abs(a-e)<1e-10 for a,e in zip(packed_flows,(2.,3.,5.)))
assert abs(sum(x*p for x,p in zip(packed_zs,
    partial_molar_properties(13.,(-5.,3.,9.),packed_zs)))-13) < 1e-12
assert abs(weighted_property((.25,.75),(1.,9.),rule='power',exponent=1.5)-20.5**(2/3))<1e-12
assert abs(heat_capacity_difference(300.,333.,-4e6)-8.316675)<1e-12
assert acoustic_velocity(.025,-4e6,30.,21.7,molecular_weight=28.97)>300
assert abs(gravity_scale(gravity_scale(850.,target='API'),source='API',target='density')-850)<1e-10
assert all(name not in sys.modules for name in ('chemicals','mpmath'))
import json
from math import isclose
from chemworld.data.helmholtz_phase import load_helmholtz_phase_model
from chemworld.physchem import (
    HelmholtzPhaseModel, HelmholtzTwoPhaseState,
    HelmholtzEquilibriumIsochoricThermodynamics, solve_two_phase_state,
    CoupledState, CoupledSystem, HeatFlux,
)
from chemworld.physchem.solver_backend import ODESolverPolicy
# Self-authored polynomial alpha_r=delta**4/60-(2/3)*delta*tau.
# It has analytic critical T=300 K, rho=1000 mol/m3; no external data.
packed_heos_source = {
    'INFO': {'NAME': 'packed-polynomial'},
    'EOS': [{'gas_constant': 8.3, 'molar_mass': .04, 'T_max': 1500, 'p_max': 1e9,
        'pseudo_pure': False,
        'STATES': {'reducing': {'T': 300, 'rhomolar': 1000}, 'sat_min_liquid': {'T': 100}},
        'alpha0': [{'type': 'IdealGasHelmholtzLead', 'a1': 0, 'a2': 0},
                   {'type': 'IdealGasHelmholtzLogTau', 'a': 1.5}],
        'alphar': [{'type': 'ResidualHelmholtzPower', 'n': [1/60, -2/3],
                    'd': [4, 1], 't': [0, 1], 'l': [0, 0]}]}],
    'STATES': {'critical': {'T': 300, 'rhomolar': 1000, 'p': 996000}},
    'ANCILLARIES': {k: {'type': kind, 'T_r': 300, 'reducing_value': reducing,
                        'Tmin': 100, 'Tmax': 300, 'n': [n], 't': [t], 'using_tau_r': tau}
        for k, kind, reducing, n, t, tau in (
            ('rhoL', 'rhoLnoexp', 1000, 2, .5, False),
            ('rhoV', 'rhoV', 1000, -4, .5, False),
            ('pS', 'pV', 996000, -4, 1, True))},
}
packed_heos = load_helmholtz_phase_model(packed_heos_source,
    source='self-authored installed fixture')
packed_heos = HelmholtzPhaseModel.from_dict(json.loads(json.dumps(packed_heos.to_dict())))
packed_flash = solve_two_phase_state(packed_heos, {'T': 240, 'Q': .35})
assert packed_flash.success and isinstance(packed_flash.state, HelmholtzTwoPhaseState)
packed_mix = packed_flash.state
assert isclose(packed_mix.enthalpy_J_mol, packed_mix.internal_energy_J_mol
    + packed_mix.pressure_Pa/packed_mix.molar_density_mol_m3, rel_tol=1e-9)
packed_volume = 2000/packed_mix.molar_density_mol_m3
packed_thermal = HelmholtzEquilibriumIsochoricThermodynamics(packed_heos, 'fluid', packed_volume)
packed_thermal = HelmholtzEquilibriumIsochoricThermodynamics.from_dict(
    json.loads(json.dumps(packed_thermal.to_dict())))
packed_initial = CoupledState({'fluid': 2}, 240, packed_volume)
packed_process = CoupledSystem(packed_thermal, heat_flux=lambda _: HeatFlux(10),
    policy=ODESolverPolicy('installed-two-phase', method='BDF', rtol=1e-11, atol=1e-13))
packed_result = packed_process.advance(packed_initial, 1)
assert abs(packed_thermal.internal_energy_j(packed_result.final_state)
    -packed_thermal.internal_energy_j(packed_initial)-10) < 1e-4
assert abs(packed_result.energy_balance_residual_J) < 1e-4
# Explicit multifluid import consumes installed pure terms, common caloric
# codecs and the existing finite-inventory energy advance; no upstream backend.
from copy import deepcopy
from chemworld.data import load_multifluid_model
from chemworld.physchem import (
    MultifluidModel, MolecularCaloricModel, MolecularIsochoricThermodynamics,
)
packed_mf_pure = [deepcopy(packed_heos_source) for _ in range(2)]
for i, node in enumerate(packed_mf_pure):
    node['INFO'].update(NAME='packed-mf-'+str(i), CAS=str(i+1))
packed_mf = load_multifluid_model({
    'fluids': packed_mf_pure,
    'binary_pairs': [{'Name1':'packed-mf-0','Name2':'packed-mf-1','CAS1':'1','CAS2':'2',
        'betaT':1.,'gammaT':1.,'betaV':1.,'gammaV':1.,'F':0.}],
    'departure_functions': [],
    'gas_constant_J_mol_K': packed_mf_pure[0]['EOS'][0]['gas_constant'],
    'gas_constant_convention': 'source_equal',
}, source='self-authored installed multifluid fixture')
assert MultifluidModel.from_dict(json.loads(json.dumps(packed_mf.to_dict()))) == packed_mf
# Revisioned local catalogue chooses the same physical model; no network or
# upstream package is required by an installed source-to-inventory consumer.
from chemworld.data import MultifluidSourceCatalog, load_multifluid_catalog, read_multifluid_catalog
packed_mf_catalog = MultifluidSourceCatalog(
    catalog_id='installed-synthetic-mixture', revision='1', source='self-authored fixture',
    fluids=packed_mf_pure,
    binary_pairs=[{'Name1':'packed-mf-0','Name2':'packed-mf-1','CAS1':'1','CAS2':'2',
        'betaT':1.,'gammaT':1.,'betaV':1.,'gammaV':1.,'F':0.}], departure_functions=[])
packed_mf_catalog_path = Path(sys.argv[1]+'.multifluid-catalog.json')
packed_mf_catalog_path.write_text(json.dumps(packed_mf_catalog.to_dict()))
assert read_multifluid_catalog(packed_mf_catalog_path) == packed_mf_catalog
assert load_multifluid_catalog(packed_mf_catalog.to_dict()) == packed_mf_catalog
packed_mf_selected = packed_mf_catalog.select(('1','2'),
    gas_constant_J_mol_K=packed_mf_pure[0]['EOS'][0]['gas_constant'],
    gas_constant_convention='source_equal')
assert packed_mf_selected.component_ids == packed_mf.component_ids
assert packed_mf_selected.state(400.,20.,(.3,.7)).pressure_Pa == (
    packed_mf.state(400.,20.,(.3,.7)).pressure_Pa)
packed_mf_caloric = MolecularCaloricModel.from_dict(
    json.loads(json.dumps(packed_mf.caloric_model().to_dict())))
assert packed_mf_caloric.residual_model == packed_mf
packed_mf_state = packed_mf_caloric.state(400., 20., (.3, .7))
assert isclose(packed_mf_state.enthalpy_J_mol, packed_mf_state.internal_energy_J_mol
    + packed_mf_state.pressure_Pa/20., rel_tol=1e-12)
packed_mf_thermal = MolecularIsochoricThermodynamics(packed_mf_caloric, 100.)
packed_mf_initial = CoupledState({'packed-mf-0':.6,'packed-mf-1':1.4}, 400., 100.)
packed_mf_process = CoupledSystem(packed_mf_thermal, heat_flux=lambda _: HeatFlux(5.),
    policy=ODESolverPolicy('installed-multifluid', method='BDF', rtol=1e-10, atol=1e-12))
packed_mf_result = packed_mf_process.advance(packed_mf_initial, .5)
assert packed_mf_result.final_state.amounts_mol == packed_mf_initial.amounts_mol
assert abs(packed_mf_thermal.internal_energy_j(packed_mf_result.final_state)
    - packed_mf_thermal.internal_energy_j(packed_mf_initial)-2.5) < 1e-5
assert abs(packed_mf_result.energy_balance_residual_J) < 1e-5
assert all(name not in sys.modules for name in ('CoolProp','thermo','fluids'))
print('packaged atomic resources OK')
"""
    # Native workflows use filesystem-backed bundled protocols. Importing the
    # .whl as a ZIP is not an installation and does not exercise that contract.
    target = tmp_path / "installed"
    subprocess.run(
        ["uv", "pip", "install", "--python", sys.executable, "--target", str(target),
         "--no-deps", str(wheel)],
        cwd=tmp_path, check=True, capture_output=True, text=True,
    )
    probe = tmp_path / "isolated_wheel_probe.py"
    probe.write_text(program, encoding="utf-8")
    result = subprocess.run(
        [sys.executable, "-I", str(probe), str(target)],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "packaged atomic resources OK" in result.stdout
    phase_probe = tmp_path / "installed_phase_probe.py"
    phase_probe.write_text(
        "import sys\nsys.path.insert(0, sys.argv[1])\n"
        "import chemworld\nassert sys.argv[1] in chemworld.__file__\n"
        + (root / "tests/reference/probe_conv23_install.py").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    phase_result = subprocess.run(
        [sys.executable, "-I", str(phase_probe), str(target)],
        cwd=tmp_path, check=False, capture_output=True, text=True,
    )
    assert phase_result.returncode == 0, phase_result.stdout + phase_result.stderr
    engineering_probe = tmp_path / "installed_engineering_probe.py"
    engineering_probe.write_text(
        "import sys\nsys.path.insert(0, sys.argv[1])\n"
        "import chemworld\nassert sys.argv[1] in chemworld.__file__\n"
        + (root / "tests/reference/probe_conv30_install.py").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    return root, tmp_path, target, engineering_probe


def test_atomic_resources_survive_sdist_wheel_and_isolated_import(installed_atomic_resources):
    _root, tmp_path, target, engineering_probe = installed_atomic_resources
    engineering_result = subprocess.run(
        [sys.executable, "-I", str(engineering_probe), str(target)],
        cwd=tmp_path, check=False, capture_output=True, text=True,
    )
    assert engineering_result.returncode == 0, engineering_result.stdout + engineering_result.stderr
    print(engineering_result.stdout)


@pytest.mark.reference
def test_reference_environment_arrays_in_installed_wheel(installed_atomic_resources):
    pytest.importorskip("fluids", reason="optional empirical MSIS/SPA reference arrays")
    root, tmp_path, target, engineering_probe = installed_atomic_resources
    # Reference parameter arrays are caller data generated before isolation;
    # the installed child evaluates its own model without importing references.
    engineering_data = {}
    for family in ("msis", "solar"):
        path = root / f"tests/reference/{family}_source_parameters.py"
        loader = importlib.util.spec_from_file_location(f"install_{family}_inputs", path)
        module = importlib.util.module_from_spec(loader)
        loader.loader.exec_module(module)
        engineering_data[family] = module.source_parameters().to_dict()
    parameter_path = tmp_path / "caller_environment_parameters.json"
    parameter_path.write_text(json.dumps(engineering_data), encoding="utf-8")
    engineering_result = subprocess.run(
        [sys.executable, "-I", str(engineering_probe), str(target), str(parameter_path)],
        cwd=tmp_path, check=False, capture_output=True, text=True,
    )
    assert engineering_result.returncode == 0, engineering_result.stdout + engineering_result.stderr
    print(engineering_result.stdout)
