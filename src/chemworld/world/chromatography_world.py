"""Finite synthetic chromatograph, source bottles and real sample/waste routing."""

from chemworld.foundation import WorldState, instrument_equipment_id
from chemworld.foundation.solvents import SolventInventory
from chemworld.foundation.state_ledgers import (
    EquipmentLedger,
    EquipmentRecord,
    PhaseLedger,
    PhaseRecord,
    VesselLedger,
    VesselRecord,
)
from chemworld.foundation.state_ownership import StateSlots
from chemworld.physchem.chromatography_methods import (
    ChromatographyProgram,
    EmpiricalChromatographyAnalyteSpec,
)
from chemworld.physchem.reaction_network import ReactionNetworkSpec, SpeciesSpec
from chemworld.physchem.reactor_shared import ScalarControlProgram
from chemworld.runtime.chromatography import initial_memory
from chemworld.world.composition import WorldCompositionSpec
from chemworld.world.physical_world import CompositionPhysics


def chromatography_world_spec(instrument="hplc", *, noise_sd=0.00001, model_overrides=None):
    """Three synthetic analytes; no property is inferred from a chemical name.

    Fixed 60 s offline analyses do not evolve the host. The programme changes
    instrument conditions, not the sample bottle's thermodynamic state. Thermal
    utilities and a continuous mobile-phase reservoir are not modelled here.
    """
    if instrument not in {"hplc", "gc"}:
        raise ValueError("Choose hplc or gc")
    name = "programmed_" + instrument
    analytes = tuple(
        EmpiricalChromatographyAnalyteSpec(
            f"reporter_{i}",
            instrument,
            k,
            298.15,
            0.2 if instrument == "hplc" else None,
            hplc_log10_k_mobile_phase_slope=2,
            gc_retention_enthalpy_J_mol=-20000,
            detector_response_slope=1,
            theoretical_plates=400,
            provenance_id="NATIVE-EXT-25-synthetic-not-material-fit",
        )
        for i, k in enumerate((2, 4, 8), 1)
    )
    programs = []
    for knots in (((0, 0),), ((0, 0), (30, 1)), ((0, 0), (15, 0.2), (30, 1))):
        temperature = ScalarControlProgram(
            tuple((t, 298.15 + (40 * f if instrument == "gc" else 0)) for t, f in knots), "K"
        )
        fraction = (
            None
            if instrument == "gc"
            else ScalarControlProgram(tuple((t, 0.2 + 0.6 * f) for t, f in knots), "dimensionless")
        )
        programs.append(ChromatographyProgram(temperature, fraction).to_dict())
    parameters = {
        "analytes": [a.to_dict() for a in analytes],
        "programs": programs,
        "carryover_fraction": 0.001,
        "wash_rate_s": 0.1,
        "initial_gain": 1.0,
        "gain_per_analysis": 0.001,
        "baseline": 0.005,
        "baseline_drift": 0.002,
        "baseline_per_analysis": 0.0001,
        "noise_sd": noise_sd,
        "saturation_AU": 2.0,
    } | (model_overrides or {})
    analytes = tuple(EmpiricalChromatographyAnalyteSpec(**a) for a in parameters["analytes"])
    recipes = {
        "reactor": (0.002, (0, 0, 0)),
        "blank": (0.05, (0, 0, 0)),
        "standard": (0.02, (0.01, 0.02, 0.03)),
        "standard_low": (0.02, (0.005, 0.01, 0.015)),
        "standard_high": (0.02, (0.02, 0.04, 0.06)),
        "sample": (0.02, (0.03, 0.015, 0.01)),
        "waste": (0.0, (0, 0, 0)),
    }
    states, components = {}, []
    for vessel, (volume, concentrations) in recipes.items():
        amounts = {a.analyte_id: c * volume for a, c in zip(analytes, concentrations, strict=True)}
        record = EquipmentRecord(
            instrument_equipment_id(name),
            "instrument",
            vessel,
            batch_reset="preserve",
            settings={"chromatography": parameters, "program_index": 0},
        )
        states[vessel] = WorldState(
            amounts,
            volume,
            298.15,
            101325,
            "liquid",
            vessel,
            units={
                "amount": "mol",
                "volume": "L",
                "temperature": "K",
                "pressure": "Pa",
                "time": "s",
                "cost": "currency",
                "risk": "risk",
            },
            phases=PhaseLedger(
                {
                    "liquid": PhaseRecord(
                        "liquid",
                        vessel,
                        "liquid",
                        volume,
                        amounts,
                        solvents=SolventInventory.pure(0, volume),
                    )
                }
            ),
            equipment=EquipmentLedger({record.equipment_id: record} if vessel == "reactor" else {}),
            state_slots=StateSlots(
                (initial_memory(name, vessel, analytes),) if vessel == "reactor" else ()
            ),
            vessels=VesselLedger(
                {
                    vessel: VesselRecord(
                        vessel,
                        "sealed-liquid-bottle",
                        0.1,
                        400,
                        2e5,
                        (),
                        298.15,
                        101325,
                    )
                }
            ),
        )
        components.append(
            {
                "id": vessel + ":material",
                "vessel": vessel,
                "kind": "material_input",
                "parameters": {},
            }
        )
    components.extend(
        (
            {
                "id": "chromatograph",
                "vessel": "reactor",
                "kind": "programmed_chromatography",
                "parameters": {},
            },
            {
                "id": "detector",
                "vessel": "reactor",
                "kind": "observation",
                "parameters": {"instruments": [name]},
            },
        )
    )
    network = ReactionNetworkSpec(
        "finite-chromatography-reporters",
        tuple(
            SpeciesSpec(a.analyte_id, formula)
            for a, formula in zip(analytes, ("C2H6O", "C3H6O", "C7H8"), strict=True)
        ),
        (),
    )
    connections = []
    for source, target in [(s, "reactor") for s in recipes if s not in {"reactor", "waste"}] + [
        ("reactor", "waste")
    ]:
        connections.append(
            {
                "id": source + "-to-" + target,
                "unit": "mol",
                "source": {"component": source + ":material", "port": "out"},
                "target": {"component": target + ":material", "port": "in"},
            }
        )
    return WorldCompositionSpec.from_dict(
        {
            "schema_version": "chemworld-world-composition-0.1",
            "composition_id": "native-programmed-" + instrument,
            "components": components,
            "connections": connections,
            "vessels": [{"id": v, "capacity_L": 0.1} for v in states],
            "physics": CompositionPhysics(network, states).to_dict(),
            "task": {
                "objective": "characterize programmed retention, calibration and carryover",
                "budget": 160,
                "operations": [
                    "select_vessel",
                    "route_material",
                    "measure",
                    "configure_chromatography",
                    "calibrate_chromatography",
                    "terminate",
                ],
                "instruments": [name],
                "termination": "explicit-stop-or-budget",
            },
        }
    )
