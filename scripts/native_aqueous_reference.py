"""Optional Reaktoro 2.13 reference with explicitly matched synthetic chemistry.

The Python API and activity callback are documented in the upstream custom
database tutorial and Reaktoro/Core/ActivityModel.py.cxx (tag v2.13.0).
No native equilibrium answer or initial guess is used by this reference.
"""

import math
import os
import sys
from pathlib import Path

import numpy as np

from chemworld.physchem.aqueous_mineral import CHARGES, COMPOSITIONS, FORMULAS
from chemworld.physchem.elements import molecular_weight


def load_reaktoro(prefix=None):
    """Load an optional compatible binary distribution into the locked Python."""
    handles = []
    if prefix is not None:
        prefix = Path(prefix)
        if sys.platform == "win32":
            handles.append(os.add_dll_directory(str(prefix / "Library" / "bin")))
        sys.path.append(
            str(
                prefix
                / (
                    "Lib/site-packages"
                    if sys.platform == "win32"
                    else "lib/python3.12/site-packages"
                )
            )
        )
    import autodiff
    import reaktoro

    return reaktoro, autodiff, handles


class ReaktoroCarbonateReference:
    def __init__(self, spec, prefix=None):
        self.r, self.ad, self.dll_handles = load_reaktoro(prefix)
        r, ad = self.r, self.ad
        if r.__version__ != "2.13.0":
            raise ValueError("this independently authored reference is pinned to Reaktoro 2.13.0")
        r.Memoization.disable()
        self.spec = spec
        elements = {
            e: r.Element(e, molecular_weight({e: 1}) / 1000)
            for e in ("H", "O", "C", "Ca", "Na", "Cl")
        }
        # Reaktoro v2.13.0 Common/Constants.hpp defines this R. Only reaction
        # differences matter; arbitrary primary chemical potentials are zero.
        rtln10 = 8.3144621 * spec.temperature_K * math.log(10)
        gibbs = dict.fromkeys(FORMULAS, 0.0)
        gibbs.update(
            {
                "OH-": rtln10 * spec.pKw,
                "HCO3-": rtln10 * spec.pKa1,
                "CO3--": rtln10 * (spec.pKa1 + spec.pKa2),
                "Calcite": rtln10 * (spec.pKa1 + spec.pKa2 + spec.log10_Ksp),
            }
        )
        species = []
        for name, composition in COMPOSITIONS.items():
            species.append(
                r.Species()
                .withName(name)
                .withFormula(FORMULAS[name])
                .withCharge(CHARGES[name])
                .withElements(
                    r.ElementalComposition(
                        [(elements[e], float(n)) for e, n in composition.items()]
                    )
                )
                .withAggregateState(
                    r.AggregateState.Solid if name == "Calcite" else r.AggregateState.Aqueous
                )
                .withStandardGibbsEnergy(gibbs[name])
            )
        self.database = r.Database(list(elements.values()), species)

        def generate(phase_species):
            keys = [s.name() for s in phase_species]
            iw = keys.index("H2O")
            mw = float(phase_species[iw].molarMass())
            charges = [CHARGES[name] for name in keys]

            def activity(args):
                # Keep autodiff numbers through every operation, so the
                # independent reference supplies its own exact derivatives.
                x = args.x
                molality = [x[i] / (mw * x[iw]) for i in range(len(keys))]
                ionic = sum(m * z * z / 2 for m, z in zip(molality, charges, strict=True))
                root = ad.sqrt(ionic)
                common = -math.log(10) * spec.davies_A * (root / (1 + root) - spec.davies_b * ionic)
                lng = [common * z * z for z in charges]
                lna = [ad.log(m) + g for m, g in zip(molality, lng, strict=True)]
                lna[iw] = ad.real(0)
                lng[iw] = ad.real(0)
                props = r.ActivityProps.create(len(keys))
                ga, aa = ad.ArrayXreal(len(keys)), ad.ArrayXreal(len(keys))
                for i in range(len(keys)):
                    ga[i], aa[i] = lng[i], lna[i]
                props.ln_g = ga
                props.ln_a = aa
                return props

            return r.ActivityModel(activity)

        aqueous = r.AqueousPhase(list(FORMULAS)[:-1]).setActivityModel(generate)
        phases = [aqueous]
        if spec.allow_calcite:
            phases.append(r.MineralPhase("Calcite"))
        self.system = r.ChemicalSystem(self.database, *phases)
        self.solver = r.EquilibriumSolver(self.system)
        options = r.EquilibriumOptions()
        options.epsilon = 1e-24
        self.solver.setOptions(options)

    def solve(self, amounts):
        state = self.initial_state(amounts)
        result = self.solver.solve(state)
        if not result.succeeded():
            raise RuntimeError("independent Reaktoro equilibrium did not converge")
        return self.readout(state)

    def initial_state(self, amounts):
        r = self.r
        state = r.ChemicalState(self.system)
        state.temperature(self.spec.temperature_K, "K")
        state.pressure(101325, "Pa")
        for name, amount in amounts.items():
            if name != "Calcite" or self.spec.allow_calcite:
                state.set(name, amount, "mol")
        return state

    def solve_fixed(self, amounts, mineral_mol):
        """Fast Reaktoro chemistry at prescribed stock and original total atoms.

        Initial component amounts, including charge, are independent of the
        nonnegative initial guess. No counterion or solution species is repaired.
        """
        if not self.spec.allow_calcite or not math.isfinite(mineral_mol) or mineral_mol < 0:
            raise ValueError("fixed reference requires an allowed finite calcite amount")
        r = self.r
        original = np.array([amounts.get(s.name(), 0) for s in self.system.species()])
        components = np.asarray(self.system.formulaMatrix(), dtype=float) @ original
        state = self.initial_state({**amounts, "Calcite": mineral_mol})
        conditions = r.EquilibriumConditions(self.system)
        conditions.temperature(self.spec.temperature_K, "K")
        conditions.pressure(101325, "Pa")
        conditions.setInitialComponentAmounts(components)
        restrictions = r.EquilibriumRestrictions(self.system)
        restrictions.cannotReact("Calcite")
        result = self.solver.solve(state, conditions, restrictions)
        if not result.succeeded():
            raise RuntimeError("independent fixed-mineral Reaktoro chemistry did not converge")
        return self.readout(state)

    def readout(self, state):
        output = {
            name: float(state.speciesAmount(name))
            if name != "Calcite" or self.spec.allow_calcite
            else 0.0
            for name in FORMULAS
        }
        water = output["H2O"] * molecular_weight(COMPOSITIONS["H2O"]) / 1000
        props = state.props()
        return {
            "amounts_mol": output,
            "solvent_mass_kg": water,
            "pH": -math.log10(float(props.speciesActivity("H+"))),
            "ionic_strength_mol_kg": sum(
                n * CHARGES[k] ** 2 / (2 * water) for k, n in output.items()
            ),
            "saturation_ratio": float(props.speciesActivity("Ca++"))
            * float(props.speciesActivity("CO3--"))
            / 10**self.spec.log10_Ksp,
            "version": self.r.__version__,
        }
