"""Authored ideal-gas two-stage design, then the original finite receiver.

The synthetic constant-Cv parameters are analytic examples, not material data.
Currency is one caller-declared utility cost scale, not an equipment economy.
"""

from __future__ import annotations

from chemworld.foundation import WorldState
from chemworld.foundation.state_ledgers import PhaseLedger, PhaseRecord, VesselLedger, VesselRecord
from chemworld.physchem.bounded_design import DesignConstraint, DesignVariable
from chemworld.physchem.helmholtz import HelmholtzFluid
from chemworld.physchem.helmholtz_terms import HelmholtzTerm
from chemworld.physchem.molecular_flash import MolecularDensityBasin
from chemworld.physchem.molecular_mixture import MolecularCaloricModel, MolecularStream
from chemworld.physchem.molecular_processes import (
    MolecularHeatDuty,
    MolecularPressureStage,
    MolecularProcessDomain,
)
from chemworld.physchem.multifluid import MultifluidBinaryPair, MultifluidModel
from chemworld.runtime.molecular_process_design import (
    MolecularDesignControl,
    MolecularProcessDesignSpec,
    MolecularUtilityPrices,
)
from chemworld.runtime.molecular_process_path import MolecularProcessPath
from chemworld.runtime.molecular_process_runtime import MolecularProcessReceiverSpec

R = 8.31446261815324


def two_stage_design(
    *,
    x_a: float = 1.0,
    temperature_K: float = 320.0,
    final_pressure_Pa: float = 9e5,
    efficiencies: tuple[float, float] = (0.85, 0.85),
    cooling_capacity_W: float = 50.0,
    cooling_price_per_J: float = 0.0,
    maximum_pressure_Pa: float = 6.5e5,
    stage1_maximum_temperature_K: float | None = None,
    start_coordinate: float = 0.2,
) -> MolecularProcessDesignSpec:
    references = tuple(
        HelmholtzFluid(
            key,
            "",
            0.04 + 0.02 * i,
            R,
            300.0,
            1000.0,
            100.0,
            2000.0,
            1e8,
            (
                HelmholtzTerm("IdealGasHelmholtzLead", {"a1": 2.0 + i, "a2": -3.5 - i}),
                HelmholtzTerm("IdealGasHelmholtzLogTau", {"a": 2.5 + i}),
            ),
            (),
            "CC0 authored constant-Cv analytic example",
        )
        for i, key in enumerate(("A", "B"))
    )
    residual = MultifluidModel(
        references,
        (MultifluidBinaryPair(("A", "B"), 1.0, 1.0, 1.0, 1.0, 0.0, (), "CC0 zero departure"),),
        R,
        "source_equal",
        "CC0 authored zero-residual model",
    )
    domain = MolecularProcessDomain(
        MolecularCaloricModel(residual, references),
        (150.0, 1400.0),
        MolecularDensityBasin("gas", (1.0, 5000.0), 100.0),
    )
    inlet = MolecularStream(
        domain.state(temperature_K, 1e5 / (R * temperature_K), (x_a, 1 - x_a)), 0.01
    )
    pressure = DesignVariable("interstage_pressure", "Pa", (1.5e5, maximum_pressure_Pa), 2e5, "log")
    pressure_start = pressure.decode(start_coordinate)
    path = MolecularProcessPath(
        domain,
        "external-feed",
        inlet,
        (
            MolecularPressureStage(domain, "compressor", pressure_start, efficiencies[0]),
            MolecularHeatDuty(domain, 0.0, boundary_temperature_K=150.0),
            MolecularPressureStage(domain, "compressor", final_pressure_Pa, efficiencies[1]),
        ),
    )
    constraints: tuple[DesignConstraint, ...] = (
        DesignConstraint("device:1:temperature_K", "K", 300.0, 2e-5, lower=temperature_K),
        DesignConstraint("cooling_W", "W", 50.0, 2e-7, upper=cooling_capacity_W),
    )
    if stage1_maximum_temperature_K is not None:
        constraints += (
            DesignConstraint(
                "device:0:temperature_K", "K", 300.0, 2e-5, upper=stage1_maximum_temperature_K
            ),
        )
    return MolecularProcessDesignSpec(
        path,
        (
            MolecularDesignControl(pressure, "outlet_pressure_Pa", 0),
            MolecularDesignControl(
                DesignVariable("cooling_duty", "W", (-50.0, 0.0), 0.0), "heat_duty_W", 1
            ),
        ),
        constraints,
        MolecularUtilityPrices(cooling_per_J=cooling_price_per_J),
        ({"interstage_pressure": pressure_start, "cooling_duty": 0.0},),
    )


def design_receiver(
    specification: MolecularProcessDesignSpec,
    *,
    maximum_amount_mol: float = 0.3,
) -> tuple[MolecularProcessReceiverSpec, WorldState]:
    """Finite tank declaration; optimization itself owns no inventory or clock."""
    path = specification.path
    amounts = {
        key: 0.05 * x
        for key, x in zip(
            path.domain.model.component_ids, path.inlet.state.mole_fractions, strict=True
        )
    }
    pressure = path.domain.model.state(320.0, 2.5, path.inlet.state.mole_fractions).pressure_Pa
    phase = PhaseRecord(
        "bulk",
        "receiver",
        "gas",
        20.0,
        amounts,
        metadata={"carrier_representation": "species_inventory"},
    )
    vessel = VesselRecord("receiver", "tank", 20.0, 1200.0, 2e6, ("gas",), 320.0, pressure)
    state = WorldState(
        amounts,
        20.0,
        320.0,
        pressure,
        "gas",
        "receiver",
        phases=PhaseLedger({"bulk": phase}),
        vessels=VesselLedger({"receiver": vessel}),
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
    return MolecularProcessReceiverSpec(path, "bulk", maximum_amount_mol), state
