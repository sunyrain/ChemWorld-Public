"""Fixed-composition dry air and total H2O on the original n/T/V clock."""

from dataclasses import dataclass
from math import fsum, isclose, isfinite

from chemworld.physchem.humid_air_inventory import _positive, humid_air_volume
from chemworld.physchem.specs import ComponentSpec

# Lemmon et al. (2000), Table 3, https://trc.nist.gov/refprop/Documents/Air.pdf.
# Inventory mass uses the declared real species. The pseudo-pure G8 potential
# retains its own published reference molar mass/gas constants without tuning.
DRY_AIR_MOLE_FRACTIONS = (.7812, .2096, .0092)


@dataclass(frozen=True)
class HumidAirIsochoricThermodynamics:
    nitrogen: ComponentSpec
    oxygen: ComponentSpec
    argon: ComponentSpec
    water: ComponentSpec
    volume_L: float
    condensed_phase: str
    support_heat_capacity_J_K: float = 0.
    reference_temperature_K: float = 298.15
    material_outlet_phase: str | None = None

    def __post_init__(self):
        for component, composition, cas in zip(self.components,
            ({"N": 2}, {"O": 2}, {"Ar": 1}, {"H": 2, "O": 1}),
            ("7727-37-9", "7782-44-7", "7440-37-1", "7732-18-5"), strict=True):
            if (not isinstance(component, ComponentSpec) or component.charge != 0
                    or component.composition != composition
                    or component.cas_number not in ("", cas)):
                raise ValueError("humid-air stock requires neutral N2/O2/Ar/H2O identities")
            _positive(component.molecular_weight_g_mol, "component molecular weight")
        if len(set(self.inventory_keys)) != 4:
            raise ValueError("humid-air components need four distinct inventory keys")
        if self.condensed_phase not in ("liquid", "ice"):
            raise ValueError("humid-air thermal model requires explicit liquid or ice")
        _positive(self.volume_L, "volume_L")
        _positive(self.reference_temperature_K, "reference_temperature_K")
        _positive(self.support_heat_capacity_J_K, "support heat capacity", zero=True)
        if self.material_outlet_phase not in (None, "gas", "condensate"):
            raise ValueError("humid-air material outlet must select gas or condensate")

    @property
    def components(self):
        return (self.nitrogen, self.oxygen, self.argon, self.water)

    @property
    def inventory_keys(self):
        return tuple(c.identifier for c in self.components)

    def validate_temperature_interval(self, low, high):
        lower, upper = (270., 473.) if self.condensed_phase == "liquid" else (193., 273.16)
        if not lower <= low < high <= upper:
            raise ValueError("thermal bounds leave the humid-air condensed-surface domain")

    def _masses(self, amounts, *, rates=False, enforce_ratio=True):
        if set(amounts) - set(self.inventory_keys) or (not rates and set(amounts) != set(
            self.inventory_keys
        )):
            raise ValueError("humid-air stock requires exactly the four declared components")
        numbers = [amounts.get(key, 0.) for key in self.inventory_keys]
        if any(isinstance(n, bool) or not isfinite(n) or (not rates and n < 0) for n in numbers):
            raise ValueError("humid-air amounts/rates must be finite; stocks must be nonnegative")
        dry = fsum(numbers[:3])
        if not rates and dry <= 0:
            raise ValueError("humid-air thermal inventory requires positive dry air")
        if enforce_ratio and any(not isclose(n, dry*x, rel_tol=1e-10, abs_tol=0.)
                                 for n, x in zip(numbers[:3], DRY_AIR_MOLE_FRACTIONS, strict=True)):
            raise ValueError("humid-air inventory/rate breaks the fixed N2/O2/Ar mole ratio")
        return (fsum(n*c.molecular_weight_g_mol/1000
                     for n, c in zip(numbers[:3], self.components[:3], strict=True)),
                numbers[3]*self.water.molecular_weight_g_mol/1000)

    def equilibrium_state(self, state):
        self._masses(state.amounts_mol)
        return self._mass_coordinate_state(state)

    def _mass_coordinate_state(self, state):
        # The shared ODE forms its Jacobian by perturbing individual n axes.
        # Extend U through total dry mass for these trial evaluations only.
        # validate_state checks every initial/accepted state on the fixed-ratio
        # manifold, and chemical_energy_rate_w checks every physical tangent.
        # No changed composition is accepted or normalized into another stock.
        if state.volume_L != self.volume_L:
            raise ValueError("humid-air isochoric thermal model rejects changing volume")
        dry, water = self._masses(state.amounts_mol, enforce_ratio=False)
        return humid_air_volume(state.temperature_K, dry, water, self.volume_L*1e-3,
                                condensed_phase=self.condensed_phase)

    def validate_state(self, state):
        self.equilibrium_state(state)

    def heat_capacity_j_k(self, state):
        return self._mass_coordinate_state(state).heat_capacity_J_K+self.support_heat_capacity_J_K

    def internal_energy_j(self, state):
        return (self._mass_coordinate_state(state).internal_energy_J+self.support_heat_capacity_J_K
                * (state.temperature_K-self.reference_temperature_K))

    def pressure_pa(self, state):
        # Solver trial scalars can survive an equal-valued equilibrium cache
        # lookup. World pressure and its constitution predicates must be plain
        # JSON scalars even when the cached response came from a NumPy trial.
        return float(self.equilibrium_state(state).gas.pressure_Pa)

    def chemical_energy_rate_w(self, state, derivatives, contributions):
        if any(c.legacy_reaction_energy_W is not None for c in contributions):
            raise ValueError("same-EOS humid-air internal energy rejects duplicate reaction heat")
        dry, water = self._masses(derivatives, rates=True)
        if dry == 0 and water == 0:
            return 0.
        response = self._mass_coordinate_state(state)
        return dry*response.dair_internal_energy_J_kg+water*response.dwater_internal_energy_J_kg

    def material_outlet(self, state, fraction):
        """Constitutive finite packet at the current state, without a second stock.

        This explicitly selected phase is sampled once. It is not continuous
        pressure-driven flow or an integrated adiabatic discharge trajectory.
        """
        _positive(fraction, "material fraction")
        if fraction > 1 or self.material_outlet_phase is None:
            raise ValueError("humid-air material requires a declared phase and fraction <= 1")
        if self.material_outlet_phase == "gas" and fraction == 1:
            raise ValueError("gas withdrawal must retain positive dry-air inventory")
        current = self.equilibrium_state(state)
        amounts = dict.fromkeys(self.inventory_keys, 0.)
        water_mass = self.water.molecular_weight_g_mol/1000
        if self.material_outlet_phase == "gas":
            for key in self.inventory_keys[:3]:
                amounts[key] = state.amounts_mol[key]*fraction
            amounts[self.water.identifier] = (
                current.gas_mass_kg*current.gas.water_mass_fraction*fraction/water_mass)
            phase, mass, h = "gas", current.gas_mass_kg, current.gas.enthalpy_J_kg
        else:
            if current.condensate is None or current.condensed_water_mass_kg <= 0:
                raise ValueError("declared condensate outlet has no condensed water")
            phase, mass = self.condensed_phase, current.condensed_water_mass_kg
            amounts[self.water.identifier] = mass*fraction/water_mass
            h = current.condensate.enthalpy_J_kg
        return {"amounts_mol": amounts, "enthalpy_J": mass*fraction*h,
                "material_phase": phase, "withdrawn_mass_kg": mass*fraction,
                "source_temperature_K": state.temperature_K,
                "source_pressure_Pa": current.gas.pressure_Pa,
                "fraction_of_selected_phase": fraction}

    def to_dict(self):
        return {"model": "g8-humid-air-isochoric-v1",
                **{name: getattr(self, name).to_dict()
                   for name in ("nitrogen", "oxygen", "argon", "water")},
                **{name: getattr(self, name) for name in (
                    "volume_L", "condensed_phase", "support_heat_capacity_J_K",
                    "reference_temperature_K")},
                **({"material_outlet_phase": self.material_outlet_phase}
                   if self.material_outlet_phase is not None else {})}

    @classmethod
    def from_dict(cls, payload):
        values = dict(payload)
        if values.pop("model", None) != "g8-humid-air-isochoric-v1":
            raise ValueError("unknown humid-air thermal declaration")
        for key in ("nitrogen", "oxygen", "argon", "water"):
            values[key] = ComponentSpec.from_dict(values[key])
        return cls(**values)
