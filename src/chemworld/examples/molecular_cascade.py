"""Explicit PR propane/butane demonstration, not a physical source catalog."""

from collections.abc import Sequence
from typing import Any

from chemworld.physchem.cpa import CPAComponent, CPAModel
from chemworld.physchem.heat_transfer_units import ThermalDeviceSpec
from chemworld.physchem.helmholtz import HelmholtzFluid
from chemworld.physchem.helmholtz_terms import HelmholtzTerm
from chemworld.physchem.molecular_cascade import MolecularCascadeSpec
from chemworld.physchem.molecular_cascade_inventory import MolecularCascadeInventorySpec
from chemworld.physchem.molecular_flash import MolecularDensityBasin
from chemworld.physchem.molecular_mixture import MolecularCaloricModel, MolecularStream


def molecular_cascade_example(
    stages: int = 4, *, temperatures_K: Sequence[float] | None = None, flow_ratio: float = 0.75
) -> tuple[Any, ...]:
    gas = 8.31446261815324
    ids = ("propane", "butane")
    critical = ((369.83, 4.248e6, 0.152), (425.12, 3.796e6, 0.2))
    model = CPAModel(
        tuple(
            CPAComponent(
                k,
                0.4572355289213822 * gas * gas * tc * tc / pc,
                0.07779607390388846 * gas * tc / pc,
                tc,
                0.37464 + 1.54226 * w - 0.26992 * w * w,
                "authored PR critical constants",
            )
            for k, (tc, pc, w) in zip(ids, critical, strict=True)
        ),
        ((0.0, 0.0), (0.0, 0.0)),
        (200.0, 500.0),
        "explicit PR binary",
        cubic_model="peng_robinson",
    )
    refs = tuple(
        HelmholtzFluid(
            k,
            "",
            mass,
            gas,
            298.15,
            101325 / (gas * 298.15),
            200.0,
            500.0,
            1e9,
            (
                HelmholtzTerm("IdealGasHelmholtzLead", {"a1": cp / gas - 1, "a2": -cp / gas}),
                HelmholtzTerm("IdealGasHelmholtzLogTau", {"a": cp / gas - 1}),
            ),
            (),
            "authored constant ideal Cp; H=S=0 at 298.15 K, 101325 Pa",
        )
        for k, cp, mass in zip(ids, (80.0, 100.0), (0.04409562, 0.0581222), strict=True)
    )
    temperatures = tuple(temperatures_K) if temperatures_K is not None else (300.0,) * stages
    spec = MolecularCascadeSpec(
        MolecularCaloricModel(model, refs),
        MolecularDensityBasin("vapor", (1.0, 2000.0), 250.0),
        MolecularDensityBasin("liquid", (7500.0, 13500.0), 10000.0),
        6e5,
        tuple(
            ThermalDeviceSpec(
                maximum_power_W=100.0,
                maximum_cooling_power_W=100.0,
                mode="isothermal",
                setpoint_K=t,
            )
            for t in temperatures
        ),
    )
    finite = MolecularCascadeInventorySpec(spec, feed_rates_mol_s=(0.001, 0.001 * flow_ratio))
    amounts = tuple(
        {"propane": 0.02 * z, "butane": 0.02 * (1 - z)} for z in (0.95, 0.05, 0.95, 0.05)
    )
    tank_t = (340.0, 280.0, 340.0, 280.0)
    feeds = tuple(
        MolecularStream(finite.at_tp(i, amounts[i], tank_t[i]), finite.feed_rates_mol_s[i])
        for i in (0, 1)
    )
    return finite, amounts, tank_t, feeds


def molecular_cascade_demo_world() -> Any:
    from chemworld.world.molecular_cascade_world import build_molecular_cascade_world

    spec, amounts, temperatures, _ = molecular_cascade_example()
    return build_molecular_cascade_world(
        spec, amounts, temperatures, formulas={"propane": "C3H8", "butane": "C4H10"}
    )
