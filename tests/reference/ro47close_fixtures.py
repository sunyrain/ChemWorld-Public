"""Authored neutral multi-salt RO fixtures; no empirical membrane-data claim."""

from dataclasses import replace

from chemworld.physchem.aqueous_transport import AqueousTransportLaw
from chemworld.physchem.elements import molecular_weight, parse_formula
from chemworld.physchem.multisolute_ro_properties import (
    MultisoluteROProperties,
    NeutralROChannel,
    ROFlatSheetChannel,
)
from chemworld.physchem.seawater_ro import SeawaterROSpec
from chemworld.physchem.transport import PipeSpec, PumpSpec
from reference.nanofiltration48_fixtures import spec as nf_spec
from reference.ro47_fixtures import spec as binary_spec


def multisolute_spec():
    props = nf_spec(ions=4).properties

    def law(value):
        return AqueousTransportLaw(
            value, "m/s", props.viscosity.source_id, (280.0, 340.0), (1e5, 2e6), 500.0
        )

    multi = MultisoluteROProperties(
        props.database,
        props.viscosity,
        (
            NeutralROChannel("NaCl", (("Na+", 1), ("Cl-", 1)), law(1e-7), law(1e-5)),
            NeutralROChannel("CaSO4", (("Ca2+", 1), ("SO4--", 1)), law(5e-8), law(8e-6)),
        ),
    )
    old = binary_spec()
    source = props.viscosity.source_id
    return replace(
        old,
        properties=multi,
        salt_permeability=None,
        water_permeability=replace(old.water_permeability, source_id=source),
        mass_transfer=replace(old.mass_transfer, source_id=source),
    )


def multisolute_feed(spec):
    return spec.properties.amounts(20.0, {"NaCl": 0.01, "CaSO4": 0.002})


def seawater_spec(*, finite_elements=3):
    args = ("source-ro47-input", (283.15, 393.15), (1e5, 12e6), 5000.0)
    return SeawaterROSpec(
        AqueousTransportLaw(4.2e-12, "m/(Pa*s)", *args),
        AqueousTransportLaw(3.5e-8, "m/s", *args),
        PumpSpec(10e6, 20000.0, 0.75),
        10.0,
        5.0,
        298.15,
        5e6,
        finite_elements=finite_elements,
    )


def mcas_spec(*, neutral=True, finite_elements=3):
    model = multisolute_spec()
    prop = model.properties
    rows = list(prop.database.species)
    if neutral:
        template = rows[1]
        composition = parse_formula("CH4N2O")
        thermo = replace(template.standard_state.thermo, species_id="urea", composition=composition)
        rows.append(
            replace(
                template,
                species_id="urea",
                composition=composition,
                charge=0,
                standard_state=replace(template.standard_state, thermo=thermo),
            )
        )
    rows = tuple(
        replace(
            s,
            standard_state=replace(
                s.standard_state,
                pressure_bounds_Pa=(1e5, 12e6),
                molar_volume_m3_mol=molecular_weight(dict(s.composition)) / 1e6,
            ),
        )
        for s in rows
    )
    database = replace(
        prop.database,
        species=rows,
        activity_model=replace(
            prop.database.activity_model,
            pressure_bounds_Pa=(1e5, 12e6),
            charges=tuple((s.species_id, s.charge) for s in rows if s.phase == "aqueous"),
        ),
    )

    def law(value, unit="m/s"):
        return AqueousTransportLaw(
            value, unit, prop.viscosity.source_id, (280.0, 340.0), (1e5, 12e6), 5000.0
        )

    channels = tuple(
        replace(c, permeability=law(3.5e-8), mass_transfer=law(1e-5)) for c in prop.channels
    )
    if neutral:
        channels += (NeutralROChannel("urea", (("urea", 1),), law(3.5e-8), law(1e-5)),)
    properties = MultisoluteROProperties(
        database, law(0.001, "Pa*s"), channels, "mcas-ideal-volume"
    )
    return replace(
        model,
        properties=properties,
        water_permeability=law(4.2e-12, "m/(Pa*s)"),
        mass_transfer=law(1e-5),
        pump=PumpSpec(10e6, 20000.0, 0.75),
        membrane_area_m2=10.0,
        temperature_K=298.15,
        channel=PipeSpec(5.0, 0.0015),
        inlet_pressure_Pa=5e6,
        reservoir_pressure_Pa=101325.0,
        maximum_interface_concentration_mol_m3=2000.0,
        flat_sheet_channel=ROFlatSheetChannel(
            0.001,
            0.9,
            tuple((c.channel_id, law(1.5e-9, "m^2/s")) for c in channels),
            finite_elements,
        ),
    )


def mcas_feed(model):
    basis = {"NaCl": 0.25, "CaSO4": 0.01}
    if "urea" in model.properties.component_ids:
        basis["urea"] = 0.02
    return model.properties.amounts(50.0, basis)
