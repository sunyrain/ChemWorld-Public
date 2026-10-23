"""Synthetic finite liquid apparatus with explicit source and receiver vessels."""

from chemworld.foundation import WorldState
from chemworld.foundation.state_ledgers import (
    EquipmentLedger,
    EquipmentRecord,
    VesselLedger,
    VesselRecord,
)
from chemworld.physchem.property_packages import ComponentPropertyPackage
from chemworld.physchem.property_reports import _validated_fraction_mapping
from chemworld.physchem.reaction_network import ReactionNetworkSpec, SpeciesSpec
from chemworld.physchem.specs import ComponentSpec, PropertyCorrelation
from chemworld.physchem.transport_devices import FiniteTransportSpec, LiquidTransportProperties
from chemworld.runtime.transport_devices import TRANSPORT_TYPE, install_tank
from chemworld.world.composition import WorldCompositionSpec
from chemworld.world.physical_world import CompositionPhysics


def synthetic_transport_properties():
    packages = []
    for name, formula, cp, rho, mu, k in (
        ("water", "H2O", 75.0, 1000.0, 0.001, 0.6),
        ("ethanol", "C2H6O", 110.0, 800.0, 0.0012, 0.2),
    ):
        declarations = (
            ("liquid_heat_capacity", "cp_polynomial", {"a": cp}, "J/(mol*K)"),
            (
                "liquid_density",
                "linear_liquid_density",
                {"rho_ref": rho, "T_ref": 298.15, "alpha": 0.0003},
                "kg/m^3",
            ),
            (
                "liquid_viscosity",
                "andrade_viscosity",
                {"A": mu / 2.718281828459045, "B": 298.15},
                "Pa*s",
            ),
            (
                "thermal_conductivity",
                "linear_thermal_conductivity",
                {"k_ref": k, "T_ref": 298.15, "slope": 0},
                "W/(m*K)",
            ),
        )
        packages.append(
            ComponentPropertyPackage(
                ComponentSpec(name, formula),
                tuple(
                    PropertyCorrelation(
                        correlation_id=f"synthetic-{name}-{prop}",
                        property_id=prop,
                        equation_id=equation,
                        coefficients=coeff,
                        input_units={"temperature": "K"},
                        output_unit=unit,
                        validity_ranges={"temperature": (280.0, 340.0)},
                        metadata={"phase": "liquid"},
                        source_note="Synthetic transport fixture; not fitted material data",
                    )
                    for prop, equation, coeff, unit in declarations
                ),
            )
        )
    return LiquidTransportProperties(tuple(packages))


def transport_world_spec(
    spec=None, *, feed_temperature_K=298.15, feed_volume_L=0.02, receiver_volume_L=0.005, budget=60,
    feed_mole_fractions=None, receiver_mole_fractions=None, receiver_temperature_K=298.15,
):
    """Build the original two-reservoir apparatus from explicit material inputs.

    The historical binary 80:20 example remains the default. Other mixtures
    require explicit fractions; a pure component has its unique composition.
    """
    spec = spec or FiniteTransportSpec(synthetic_transport_properties())
    ids = spec.properties.component_ids
    if feed_mole_fractions is None:
        if len(ids) not in (1, 2):
            raise ValueError("declare feed fractions for a multicomponent transport world")
        feed_mole_fractions = dict(zip(ids, (1.0,) if len(ids) == 1 else (0.8, 0.2), strict=True))
    if receiver_mole_fractions is None:
        receiver_mole_fractions = feed_mole_fractions
    compositions = tuple(_validated_fraction_mapping(x, "mole") for x in (
        feed_mole_fractions, receiver_mole_fractions,
    ))
    if any(set(x) != set(ids) for x in compositions):
        raise ValueError("transport compositions must match the declared components")
    states = {}
    for index, (name, volume, temperature) in enumerate(
        (
            (spec.source_vessel_id, feed_volume_L, feed_temperature_K),
            ("reactor", receiver_volume_L, receiver_temperature_K),
        )
    ):
        fractions = compositions[index]
        per_mole = spec.properties.evaluate(fractions, temperature)[2] * 1000
        amounts = {key: x * volume / per_mole for key, x in fractions.items()}
        equipment = (
            {}
            if index == 0
            else {
                "pump": EquipmentRecord(
                    "pump",
                    TRANSPORT_TYPE,
                    name,
                    batch_reset="preserve",
                    settings={"physical_model": spec.to_dict()},
                )
            }
        )
        state = WorldState(
            amounts,
            volume,
            temperature,
            spec.properties.pressure_Pa,
            "liquid",
            name,
            equipment=EquipmentLedger(equipment),
            vessels=VesselLedger(
                {
                    name: VesselRecord(
                        name,
                        "finite-transport-tank",
                        spec.receiver_capacity_L if index else max(0.1, volume),
                        spec.properties.temperature_bounds_K[1],
                        max(5e5, spec.properties.pressure_Pa),
                        ("liquid",),
                        temperature,
                        spec.properties.pressure_Pa,
                    )
                }
            ),
            units={
                "amount": "mol",
                "volume": "L",
                "temperature": "K",
                "pressure": "Pa",
                "time": "s",
                "cost": "currency",
                "risk": "risk",
            },
        )
        states[name] = install_tank(
            state, spec, index, amounts, spec.enthalpy(index, amounts, temperature)
        )
    states = {"reactor": states["reactor"], spec.source_vessel_id: states[spec.source_vessel_id]}
    physics = CompositionPhysics(
        ReactionNetworkSpec(
            "finite-liquid-transport",
            tuple(
                SpeciesSpec(p.component.identifier, p.component.formula, phase="liquid")
                for p in spec.properties.packages
            ),
            (),
        ),
        states,
    )
    return WorldCompositionSpec.from_dict(
        {
            "schema_version": "chemworld-world-composition-0.1",
            "composition_id": "finite-liquid-transport",
            "components": [
                {
                    "id": "feed-material",
                    "kind": "material_input",
                    "vessel": spec.source_vessel_id,
                    "parameters": {},
                },
                {
                    "id": "pump-path",
                    "kind": "finite_transport",
                    "vessel": "reactor",
                    "parameters": {},
                },
                {
                    "id": "gauge",
                    "kind": "observation",
                    "vessel": "reactor",
                    "parameters": {"instruments": ["flow_meter"]},
                },
            ],
            "vessels": [
                {"id": v, "capacity_L": s.vessels.vessels[v].max_volume_L}
                for v, s in states.items()
            ],
            "connections": [
                {
                    "id": "feed-path",
                    "unit": "mol",
                    "source": {"component": "feed-material", "port": "out"},
                    "target": {"component": "pump-path", "port": "in"},
                }
            ],
            "physics": physics.to_dict(),
            "task": {
                "objective": "study finite liquid transport",
                "budget": budget,
                "operations": ["set_flow_rate", "run_flow", "wait", "measure", "terminate"],
                "instruments": ["flow_meter"],
                "termination": "explicit-stop-or-budget",
            },
        }
    )
