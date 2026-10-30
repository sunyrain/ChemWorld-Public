"""Native single-particle battery, using the common finite-volume solver.

Equation reference: PyBaMM BasicSPM (BSD-3-Clause), commit
a4ea370d9ff844e586907cea7109c38f36ce3660. This independent implementation uses
ChemWorld inventories, diffusion, BV constants and event integration. Default
parameters are synthetic; explicit source projections retain their input binding.
Optional caloric closure derives from the same OCV state function.
"""

from dataclasses import asdict, dataclass, replace
from functools import cached_property
from math import asinh, isfinite, pi, sqrt
from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import brentq

from chemworld.physchem.battery_degradation import SEI_SPECIES, SEISpec
from chemworld.physchem.battery_electrolyte import ELECTROLYTE_SPECIES, BatteryElectrolyteSpec
from chemworld.physchem.battery_materials import ElectrodeOCV
from chemworld.physchem.battery_ocp import (
    ElectrodeOCPSpec,
    hysteresis_coordinates,
    hysteresis_key,
    hysteresis_rates,
    validate_hysteresis,
    validate_ocp_cv,
)
from chemworld.physchem.battery_plating import PLATING_SPECIES, LithiumPlatingSpec
from chemworld.physchem.coupled_system import (
    CoupledAlgebraic,
    CoupledCoordinate,
    CoupledEvent,
    CoupledProcess,
    CoupledState,
    CoupledSystem,
    EnergyPort,
    HeatFlux,
    InternalTransfer,
    ProcessContribution,
)
from chemworld.physchem.electrochemistry import FARADAY_C_PER_MOL, R_J_PER_MOL_K
from chemworld.physchem.finite_volume import ConservativeDiffusion, DiffusionMesh
from chemworld.physchem.residual_system import (
    ResidualDerivatives,
    ResidualEquation,
    ResidualNumericalError,
    ResidualSystem,
    ResidualVariable,
    RootPolicy,
    solve_steady_residual,
)

if TYPE_CHECKING:
    from chemworld.physchem.battery_parameter_sets import BatteryParameterBinding

F = FARADAY_C_PER_MOL
R = R_J_PER_MOL_K
BATTERY_COORDINATES = ("discharged_charge_C", "delivered_work_J")
BATTERY_ALGEBRAIC = ("battery:current_A", "battery:voltage_V")


@dataclass(frozen=True)
class BatteryThermalSpec:
    heat_capacity_J_K: float = 100.0
    environment_temperature_K: float = 298.15
    loss_conductance_W_K: float = 0.0
    minimum_temperature_K: float = 270.0
    maximum_temperature_K: float = 360.0

    def __post_init__(self) -> None:
        if (
            not isfinite(self.heat_capacity_J_K)
            or self.heat_capacity_J_K <= 0
            or not isfinite(self.loss_conductance_W_K)
            or self.loss_conductance_W_K < 0
            or not 0
            < self.minimum_temperature_K
            < self.environment_temperature_K
            < self.maximum_temperature_K
            or not isfinite(self.maximum_temperature_K)
        ):
            raise ValueError("invalid battery caloric or ambient domain")


@dataclass(frozen=True)
class ParticleSpec:
    radius_m: float = 1e-5
    diffusivity_m2_s: float = 1e-14
    maximum_concentration_mol_m3: float = 30000.0
    active_volume_m3: float = 1e-6
    cells: int = 40
    mesh_exponent: float = 2.0
    exchange_current_scale_A_m2: float = 1.0

    def __post_init__(self) -> None:
        for key in (
            "radius_m",
            "diffusivity_m2_s",
            "maximum_concentration_mol_m3",
            "active_volume_m3",
            "exchange_current_scale_A_m2",
        ):
            if not isfinite(getattr(self, key)) or getattr(self, key) <= 0:
                raise ValueError(f"{key} must be positive and finite")
        if not isfinite(self.mesh_exponent) or not 1 <= self.mesh_exponent <= 3:
            raise ValueError("radial mesh exponent must be in [1, 3]")
        _ = self.mesh

    @cached_property
    def mesh(self):
        count = self.active_volume_m3 / (4 * pi / 3 * self.radius_m**3)
        uniform = DiffusionMesh.spherical(self.radius_m, self.cells, particle_count=count)
        faces = self.radius_m * (
            1 - (1 - np.array(uniform.faces_m) / self.radius_m) ** self.mesh_exponent
        )
        return DiffusionMesh("spherical", tuple(faces), particle_count=count)

    @property
    def area_m2(self):
        return 3 * self.active_volume_m3 / self.radius_m

    @property
    def capacity_mol(self):
        return self.maximum_concentration_mol_m3 * self.active_volume_m3

    def keys(self, electrode):
        return tuple(f"{electrode}:Li:{i}" for i in range(self.cells))


@dataclass(frozen=True)
class BatterySPMSpec:
    negative: ParticleSpec = ParticleSpec()
    positive: ParticleSpec = ParticleSpec()
    initial_negative_fraction: float = 0.7
    initial_positive_fraction: float = 0.3
    negative_ocv_intercept_V: float = 0.2
    negative_ocv_slope_V: float = -0.1
    positive_ocv_intercept_V: float = 4.2
    positive_ocv_slope_V: float = -0.8
    temperature_K: float = 298.15
    surface_margin: float = 0.01
    minimum_voltage_V: float = 2.5
    maximum_voltage_V: float = 4.5
    maximum_current_A: float = 2.0
    series_resistance_ohm: float = 0.0
    thermal: BatteryThermalSpec | None = None
    sei: SEISpec | None = None
    provenance: str = "synthetic-SPM-v1"
    numerical_form: str = "residual"
    electrolyte: BatteryElectrolyteSpec | None = None
    negative_ocv: ElectrodeOCV | None = None
    positive_ocv: ElectrodeOCV | None = None
    negative_ocp: ElectrodeOCPSpec | None = None
    positive_ocp: ElectrodeOCPSpec | None = None
    plating: LithiumPlatingSpec | None = None
    parameter_binding: "BatteryParameterBinding | None" = None
    cell_envelope_volume_m3: float | None = None
    kinetics_stoichiometry_clip: float | None = None

    def __post_init__(self) -> None:
        if self.numerical_form not in {"ode", "residual"}:
            raise ValueError("battery numerical form must be ode or residual")
        if isinstance(self.sei, dict):
            object.__setattr__(self, "sei", SEISpec(**self.sei))
        if isinstance(self.thermal, dict):
            object.__setattr__(self, "thermal", BatteryThermalSpec(**self.thermal))
        if not isfinite(self.series_resistance_ohm) or self.series_resistance_ohm < 0:
            raise ValueError("series resistance must be finite and nonnegative")
        for name in ("negative", "positive"):
            if isinstance(getattr(self, name), dict):
                object.__setattr__(self, name, ParticleSpec(**getattr(self, name)))
        if isinstance(self.electrolyte, dict):
            object.__setattr__(self, "electrolyte", BatteryElectrolyteSpec(**self.electrolyte))
        family = ("MPM" if getattr(self, "model_family", None) == "mpm-isothermal-v1"
                  else "DFN" if getattr(self, "model_family", None) == "dfn-isothermal-v1"
                  else "SPM" if self.electrolyte is None else "SPMe")
        for name in ("negative_ocp", "positive_ocp"):
            ocp = getattr(self, name)
            if isinstance(ocp, dict):
                ocp = ElectrodeOCPSpec.from_dict(ocp)
                object.__setattr__(self, name, ocp)
            if ocp is not None:
                if not isinstance(ocp, ElectrodeOCPSpec):
                    raise ValueError("operational OCP requires an explicit electrode declaration")
                if ocp.mode != "single":
                    if (family not in {"SPM", "MPM"} or self.thermal is not None
                            or self.sei is not None or self.numerical_form != "residual"):
                        raise ValueError(
                            "directional OCP requires isothermal no-side SPM/MPM residual physics")
                    for curve in (ocp.lithiation, ocp.delithiation):
                        if (family not in curve.supported_models
                                or curve.stoichiometry_domain[0] > self.surface_margin
                                or curve.stoichiometry_domain[1] < 1 - self.surface_margin):
                            raise ValueError(
                                "operational OCP branch must cover this model "
                                "and stopping surfaces")
        for name in ("negative_ocv", "positive_ocv"):
            curve = getattr(self, name)
            if isinstance(curve, dict):
                curve = ElectrodeOCV.from_dict(curve)
                object.__setattr__(self, name, curve)
            if curve is not None:
                if not isinstance(curve, ElectrodeOCV) or family not in curve.supported_models:
                    raise ValueError("OCV curve is incompatible with the battery model")
                if (curve.stoichiometry_domain[0] > self.surface_margin
                        or curve.stoichiometry_domain[1] < 1 - self.surface_margin):
                    raise ValueError("OCV domain must cover both battery stopping surfaces")
                if self.thermal is not None or self.sei is not None:
                    raise ValueError(
                        "nonlinear OCV currently requires isothermal, no-side-reaction physics")
        if self.electrolyte is not None:
            if not isinstance(self.electrolyte, BatteryElectrolyteSpec):
                raise ValueError("electrolyte requires an explicit SPMe specification")
            if self.thermal is not None or self.sei is not None:
                raise ValueError(
                    "SPMe-v1 is isothermal without SEI; thermal/aging coupling is separate")
            for particle, index in ((self.negative, 0), (self.positive, 2)):
                solid_volume = (self.electrolyte.area_m2 * self.electrolyte.lengths_m[index]
                                * (1 - self.electrolyte.porosities[index]))
                if particle.active_volume_m3 > solid_volume * (1 + 1e-12):
                    raise ValueError("active particle volume exceeds non-pore electrode volume")
        if not 0 < self.surface_margin < 0.5:
            raise ValueError("surface margin must be strictly between zero and one half")
        for value in (self.initial_negative_fraction, self.initial_positive_fraction):
            if not self.surface_margin < value < 1 - self.surface_margin:
                raise ValueError("initial stoichiometry must be inside the supported domain")
        for name in (
            "negative_ocv_intercept_V",
            "negative_ocv_slope_V",
            "positive_ocv_intercept_V",
            "positive_ocv_slope_V",
            "minimum_voltage_V",
            "maximum_voltage_V",
        ):
            if not isfinite(getattr(self, name)):
                raise ValueError("finite OCV and voltage bounds are required")
        if (
            not isfinite(self.temperature_K)
            or self.temperature_K <= 0
            or not isfinite(self.maximum_current_A)
            or self.maximum_current_A <= 0
            or self.minimum_voltage_V >= self.maximum_voltage_V
            or not self.provenance
        ):
            raise ValueError("invalid battery operating domain")
        if self.thermal is not None and (
            not self.thermal.minimum_temperature_K
            < self.temperature_K
            < self.thermal.maximum_temperature_K
            or self.negative_ocv_slope_V > 0
            or self.positive_ocv_slope_V > 0
        ):
            raise ValueError(
                "thermal SPM requires a valid initial temperature and convex chemical energy"
            )

        if isinstance(self.plating, dict):
            object.__setattr__(self, "plating", LithiumPlatingSpec.from_dict(self.plating))
        if self.plating is not None:
            from chemworld.physchem.battery_plating import validate_plating_configuration

            validate_plating_configuration(self)

        if self.cell_envelope_volume_m3 is not None:
            volume = self.cell_envelope_volume_m3
            if (isinstance(volume, bool) or not isfinite(volume)
                    or volume < self.negative.active_volume_m3 + self.positive.active_volume_m3):
                raise ValueError("cell envelope volume must contain both active material volumes")
        if self.kinetics_stoichiometry_clip is not None:
            clip = self.kinetics_stoichiometry_clip
            if (isinstance(clip, bool) or not isfinite(clip)
                    or not 0 < clip < self.surface_margin):
                raise ValueError("kinetics trial clip must be inside the declared stopping margin")
        source_options = (self.cell_envelope_volume_m3 is not None
                          or self.kinetics_stoichiometry_clip is not None
                          or self.parameter_binding is not None)
        if source_options and (
                family != "SPM" or self.thermal is not None or self.sei is not None
                or self.plating is not None or self.has_directional_ocp
                or self.numerical_form != "residual"):
            raise ValueError(
                "source projection options require single-valued no-side isothermal SPM")
        if self.parameter_binding is not None:
            from chemworld.physchem.battery_parameter_sets import BatteryParameterBinding

            binding = self.parameter_binding
            if isinstance(binding, dict):
                binding = BatteryParameterBinding.from_dict(binding)
                object.__setattr__(self, "parameter_binding", binding)
            if not isinstance(binding, BatteryParameterBinding):
                raise ValueError("battery parameters require an explicit source binding")
            binding.validate_projection(self)

    @property
    def volume_l(self):
        if self.cell_envelope_volume_m3 is not None:
            return self.cell_envelope_volume_m3 * 1000
        if self.plating is not None:
            return (self.plating.native_parameters["negative_electrode_volume_m3"]
                    + self.positive.active_volume_m3) * 1000
        if self.electrolyte is not None:
            return self.electrolyte.area_m2 * sum(self.electrolyte.lengths_m) * 1000
        return (self.negative.active_volume_m3 + self.positive.active_volume_m3) * 1000

    @property
    def model_id(self):
        if self.parameter_binding is not None:
            binding = self.parameter_binding
            return (f"source-SPM:{binding.parameter_set_id}@{binding.source_version}:"
                    f"{binding.projection_id}")
        if self.plating is not None:
            return "synthetic-SPM-plating-v1"
        return "synthetic-SPM-v1" if self.electrolyte is None else self.electrolyte.model_version

    @property
    def inventory_keys(self):
        return tuple(self.inventory_species)

    @property
    def inventory_species(self):
        return {
            **dict.fromkeys(
                (*self.negative.keys("negative"), *self.positive.keys("positive")), "Li"
            ),
            **({} if self.sei is None else SEI_SPECIES),
            **({} if self.plating is None else PLATING_SPECIES),
            **({} if self.electrolyte is None else dict.fromkeys(
                self.electrolyte.keys, ELECTROLYTE_SPECIES)),
        }

    @property
    def film_chemical_energy_j_mol(self):
        return (
            2
            * -F
            * (
                self.negative_ocv_intercept_V
                + self.negative_ocv_slope_V * self.initial_negative_fraction
            )
            - self.sei.reference_driving_energy_J_mol
        )

    @property
    def usable_charge_c(self):
        return F * min(
            self.negative.capacity_mol * self.initial_negative_fraction,
            self.positive.capacity_mol * (1 - self.initial_positive_fraction),
        )

    def current_at_c_rate(self, rate):
        if not isfinite(rate):
            raise ValueError("C-rate must be finite")
        return rate * self.usable_charge_c / 3600

    def to_dict(self):
        # OCP tables contain immutable mappings; serialize their declared codec,
        # not dataclasses' private implementation caches.
        result = asdict(replace(
            self, negative_ocp=None, positive_ocp=None, plating=None, parameter_binding=None
        ))
        for name in ("cell_envelope_volume_m3", "kinetics_stoichiometry_clip"):
            if getattr(self, name) is None:
                result.pop(name)
        if self.parameter_binding is None:
            result.pop("parameter_binding")
        else:
            result["parameter_binding"] = self.parameter_binding.to_dict()
        if self.plating is None:
            result.pop("plating")
        else:
            result["plating"] = self.plating.to_dict()
        if self.electrolyte is None:
            result.pop("electrolyte")
        else:
            result["electrolyte"] = self.electrolyte.to_dict()
        for name in ("negative_ocv", "positive_ocv"):
            curve = getattr(self, name)
            if curve is None:
                result.pop(name)
            else:
                result[name] = curve.to_dict()
        for name in ("negative_ocp", "positive_ocp"):
            value = getattr(self, name)
            if value is None:
                result.pop(name)
            else:
                result[name] = value.to_dict()
        return result

    @property
    def has_directional_ocp(self):
        return any(c is not None and c.mode != "single"
                   for c in (self.negative_ocp, self.positive_ocp))

    @property
    def hysteresis_keys(self):
        return tuple(hysteresis_key(name) for name in ("negative", "positive")
                     if (c := getattr(self, name + "_ocp")) is not None and c.mode == "axen")

    @property
    def coordinate_keys(self):
        return (*BATTERY_COORDINATES, *self.hysteresis_keys)

    def initial_coordinates(self):
        return {**dict.fromkeys(BATTERY_COORDINATES, 0.0),
                **{hysteresis_key(name): c.native_parameters["initial_h"]
                   for name in ("negative", "positive")
                   if (c := getattr(self, name + "_ocp")) is not None and c.mode == "axen"}}

    def ocp_derivatives(self, electrode, theta, current_A, state):
        if electrode not in {"negative", "positive"}:
            raise ValueError("unknown electrode")
        ocp = getattr(self, electrode + "_ocp")
        if ocp is None or ocp.mode == "single":
            return self.ocv(electrode, theta), self.ocv(electrode, theta, derivative=True), 0.0, 0.0
        return ocp.derivatives(electrode, theta, current_A,
                               state.coordinates.get(hysteresis_key(electrode)))

    def operational_ocp(self, electrode, theta, current_A, state, *, derivative=False):
        return self.ocp_derivatives(electrode, theta, current_A, state)[int(derivative)]

    def ocv(self, electrode, theta, *, derivative=False):
        if electrode not in {"negative", "positive"}:
            raise ValueError("unknown electrode")
        curve = getattr(self, electrode + "_ocv")
        if curve is not None:
            # The same analytic law may be evaluated during Newton/event trials.
            # Accepted inventories and surfaces remain bounded by the real events.
            return (curve.derivatives(theta, trial=True)[1] if derivative
                    else curve.value(theta, trial=True))
        slope = getattr(self, electrode + "_ocv_slope_V")
        return (slope if derivative else
                getattr(self, electrode + "_ocv_intercept_V") + slope * theta)

    def initial_state(self):
        amounts = {}
        for name, particle, theta in (
            ("negative", self.negative, self.initial_negative_fraction),
            ("positive", self.positive, self.initial_positive_fraction),
        ):
            amounts.update(
                zip(
                    particle.keys(name),
                    particle.mesh.cell_volumes_m3 * particle.maximum_concentration_mol_m3 * theta,
                    strict=True,
                )
            )
        if self.sei is not None:
            amounts.update(self.sei.initial_amounts(self.negative.area_m2))
        if self.plating is not None:
            amounts.update(self.plating.initial_amounts())
        if self.electrolyte is not None:
            amounts.update(self.electrolyte.initial_amounts())
        return CoupledState(
            amounts,
            self.temperature_K,
            self.volume_l,
            coordinates=self.initial_coordinates(),
        )


@dataclass(frozen=True)
class BatteryReadout:
    voltage_V: float
    current_A: float
    negative_surface_fraction: float
    positive_surface_fraction: float
    negative_average_fraction: float
    positive_average_fraction: float
    negative_overpotential_V: float
    positive_overpotential_V: float
    negative_main_current_A: float
    sei_current_A: float
    film_resistance_ohm: float
    concentration_overpotential_V: float = 0.0
    electrolyte_ohmic_V: float = 0.0
    solid_ohmic_V: float = 0.0


def particle_surface_fraction(particle, name, state, outward_mol_s):
    cell = state.amounts_mol[particle.keys(name)[-1]] / particle.mesh.cell_volumes_m3[-1]
    half_cell = particle.mesh.spherical_surface_distance_m
    surface = cell - outward_mol_s * half_cell / (particle.area_m2 * particle.diffusivity_m2_s)
    return float(surface / particle.maximum_concentration_mol_m3)


def battery_side_rate(spec, state):
    return 0.0 if spec.sei is None else spec.sei.reaction_rate_mol_s(state, spec.negative.area_m2)


def battery_particle_flux(spec, state, name, current_A):
    if getattr(spec, "model_family", None) == "half-cell-dfn-isothermal-v1":
        raise ValueError("DFN half-cell uses per-position reaction currents, not one particle flux")
    if getattr(spec, "model_family", None) == "collector-spm-isothermal-v1":
        raise ValueError(
            "collector SPM uses per-position reaction currents; legacy particle flux is unsupported"
        )
    if getattr(spec, "plating", None) is not None:
        from chemworld.physchem.battery_plating import plating_particle_flux

        return plating_particle_flux(spec, state, name, current_A)
    if getattr(spec, "model_family", None) == "split-ocvr-isothermal-v1":
        raise ValueError("Split OCV-R has no particle flux")
    if getattr(spec, "model_family", None) == "lead-acid-loqs-isothermal-v1":
        raise ValueError("lead-acid LOQS has no lithium particle flux")
    return (
        current_A / F + 2 * battery_side_rate(spec, state) if name == "negative" else -current_A / F
    )


def battery_spec_from_dict(value):
    """Decode explicit physical families; never fall back on a failed model."""
    family = value.get("model_family")
    if family is None:
        return BatterySPMSpec(**value)
    if family == "half-cell-dfn-isothermal-v1":
        from chemworld.physchem.battery_half_cell_dfn import BatteryHalfCellDFNSpec

        return BatteryHalfCellDFNSpec.from_dict(value)
    if family == "split-ocvr-isothermal-v1":
        from chemworld.physchem.battery_split_ocvr import BatterySplitOCVRSpec

        return BatterySplitOCVRSpec.from_dict(value)
    if family == "lead-acid-loqs-isothermal-v1":
        from chemworld.physchem.battery_lead_acid import BatteryLeadAcidSpec

        return BatteryLeadAcidSpec.from_dict(value)
    if family == "composite-dfn-isothermal-v1":
        from chemworld.physchem.battery_dfn_composite import BatteryCompositeDFNSpec

        return BatteryCompositeDFNSpec.from_dict(value)
    if family == "collector-spm-isothermal-v1":
        from chemworld.physchem.battery_collector_spm import BatteryCollectorSPMSpec

        return BatteryCollectorSPMSpec.from_dict(value)
    if family == "newman-tobias-isothermal-v1":
        from chemworld.physchem.battery_newman_tobias import BatteryNewmanTobiasSpec

        return BatteryNewmanTobiasSpec.from_dict(value)
    if family == "msmr-spm-isothermal-v1":
        from chemworld.physchem.battery_msmr import BatteryMSMRSPMSpec

        return BatteryMSMRSPMSpec.from_dict(value)
    from chemworld.physchem.battery_composite import COMPOSITE_FAMILY, BatteryCompositeSPMSpec
    from chemworld.physchem.battery_dfn import DFN_ID, BatteryDFNSpec
    from chemworld.physchem.battery_half_cell import HALF_CELL_FAMILY, BatteryHalfCellSPMSpec
    from chemworld.physchem.battery_mpm import MPM_ID, BatteryMPMSpec
    from chemworld.physchem.battery_spatial_thermal import BatterySpatialThermalSPMSpec

    if family == "spm-spatial-thermal-v1":
        return BatterySpatialThermalSPMSpec.from_dict(value)
    if family == COMPOSITE_FAMILY:
        return BatteryCompositeSPMSpec.from_dict(value)
    if family == HALF_CELL_FAMILY:
        return BatteryHalfCellSPMSpec.from_dict(value)
    if family == MPM_ID:
        return BatteryMPMSpec(**value)
    if family != DFN_ID:
        raise ValueError("unknown battery model family")
    return BatteryDFNSpec(**value)


def battery_readout(spec, state, current_A):
    if getattr(spec, "parameter_binding", None) is not None and (
            abs(state.temperature_K - spec.temperature_K) > 1e-8
            or abs(state.volume_L - spec.volume_l) > 1e-12):
        raise ValueError("source SPM readout requires its declared fixed temperature and volume")
    if getattr(spec, "plating", None) is not None:
        from chemworld.physchem.battery_plating import plating_readout

        return plating_readout(spec, state, current_A)
    if getattr(spec, "model_family", None) == "split-ocvr-isothermal-v1":
        from chemworld.physchem.battery_split_ocvr import split_ocvr_readout

        return split_ocvr_readout(spec, state, current_A)
    if getattr(spec, "model_family", None) == "lead-acid-loqs-isothermal-v1":
        from chemworld.physchem.battery_lead_acid import lead_acid_readout

        return lead_acid_readout(spec, state, current_A)
    if getattr(spec, "model_family", None) == "composite-dfn-isothermal-v1":
        from chemworld.physchem.battery_dfn_composite import composite_dfn_readout

        return composite_dfn_readout(spec, state, current_A)
    if getattr(spec, "model_family", None) == "collector-spm-isothermal-v1":
        from chemworld.physchem.battery_collector_spm import collector_spm_readout

        return collector_spm_readout(spec, state, current_A)
    if getattr(spec, "model_family", None) == "newman-tobias-isothermal-v1":
        from chemworld.physchem.battery_newman_tobias import newman_tobias_readout

        return newman_tobias_readout(spec, state, current_A)
    if getattr(spec, "model_family", None) == "msmr-spm-isothermal-v1":
        from chemworld.physchem.battery_msmr import msmr_readout

        return msmr_readout(spec, state, current_A)
    if getattr(spec, "model_family", None) == "spm-spatial-thermal-v1":
        from chemworld.physchem.battery_spatial_thermal import spatial_thermal_readout

        return spatial_thermal_readout(spec, state, current_A)
    if getattr(spec, "model_family", None) == "composite-spm-isothermal-v1":
        from chemworld.physchem.battery_composite import composite_readout

        return composite_readout(spec, state, current_A)
    if getattr(spec, "model_family", None) == "half-cell-dfn-isothermal-v1":
        from chemworld.physchem.battery_half_cell_dfn import half_cell_dfn_readout

        return half_cell_dfn_readout(spec, state, current_A)
    if getattr(spec, "model_family", None) == "half-cell-spm-isothermal-v1":
        from chemworld.physchem.battery_half_cell import half_cell_readout

        return half_cell_readout(spec, state, current_A)
    if isinstance(state, CoupledState):
        validate_hysteresis(spec, state)
    if getattr(spec, "model_family", None) == "mpm-isothermal-v1":
        from chemworld.physchem.battery_mpm import mpm_readout

        return mpm_readout(spec, state, current_A)
    if getattr(spec, "model_family", None) == "dfn-isothermal-v1":
        from chemworld.physchem.battery_dfn import dfn_readout

        return dfn_readout(spec, state, current_A)
    if not isfinite(current_A) or abs(current_A) > spec.maximum_current_A:
        raise ValueError("current is outside battery capability")
    values, eta, averages = {}, {}, {}
    for name, particle in (("negative", spec.negative), ("positive", spec.positive)):
        flux = battery_particle_flux(spec, state, name, current_A)
        theta = particle_surface_fraction(particle, name, state, flux)
        values[name] = theta
        # Smooth finite trial extension beyond the stopping surfaces. Accepted
        # states/events use the actual, unclipped theta; no missing Li is repaired.
        clip = (spec.surface_margin / 2 if spec.kinetics_stoichiometry_clip is None
                else spec.kinetics_stoichiometry_clip)
        trial_theta = float(np.clip(theta, clip, 1 - clip))
        j0 = particle.exchange_current_scale_A_m2 * sqrt(trial_theta * (1 - trial_theta))
        if spec.electrolyte is not None:
            j0 *= spec.electrolyte.kinetics_factor(state, name)
        eta[name] = 2 * R * state.temperature_K / F * asinh(F * flux / particle.area_m2 / (2 * j0))
        averages[name] = (
            sum(state.amounts_mol[k] for k in particle.keys(name)) / particle.capacity_mol
        )
    un = spec.operational_ocp("negative", values["negative"], current_A, state)
    up = spec.operational_ocp("positive", values["positive"], current_A, state)
    side_current = -2 * F * battery_side_rate(spec, state)
    film_resistance = (
        0.0 if spec.sei is None else spec.sei.resistance_ohm(state, spec.negative.area_m2)
    )
    electrolyte = ({} if spec.electrolyte is None
                   else spec.electrolyte.potential(state, current_A))
    concentration = electrolyte.get("concentration_overpotential_V", 0.0)
    ionic_ohmic = electrolyte.get("electrolyte_ohmic_V", 0.0)
    solid_ohmic = electrolyte.get("solid_ohmic_V", 0.0)
    return BatteryReadout(
        up
        - un
        + eta["positive"]
        - eta["negative"]
        - current_A * (spec.series_resistance_ohm + film_resistance)
        + concentration + ionic_ohmic + solid_ohmic,
        current_A,
        values["negative"],
        values["positive"],
        averages["negative"],
        averages["positive"],
        eta["negative"],
        eta["positive"],
        current_A - side_current,
        side_current,
        film_resistance,
        concentration,
        ionic_ohmic,
        solid_ohmic,
    )


def battery_control_mode(spec, state, program):
    if getattr(spec, "plating", None) is not None:
        from chemworld.physchem.battery_plating import plating_control_mode

        return plating_control_mode(spec, state, program)
    if getattr(spec, "model_family", None) == "split-ocvr-isothermal-v1":
        from chemworld.physchem.battery_split_ocvr import split_ocvr_control_mode

        return split_ocvr_control_mode(spec, state, program)
    if getattr(spec, "model_family", None) == "lead-acid-loqs-isothermal-v1":
        from chemworld.physchem.battery_lead_acid import lead_acid_control_mode

        return lead_acid_control_mode(spec, state, program)
    if getattr(spec, "model_family", None) == "composite-dfn-isothermal-v1":
        from chemworld.physchem.battery_dfn_composite import composite_dfn_control_mode

        return composite_dfn_control_mode(spec, state, program)
    if getattr(spec, "model_family", None) == "collector-spm-isothermal-v1":
        from chemworld.physchem.battery_collector_spm import collector_spm_control_mode

        return collector_spm_control_mode(spec, state, program)
    if getattr(spec, "model_family", None) == "newman-tobias-isothermal-v1":
        from chemworld.physchem.battery_newman_tobias import newman_tobias_control_mode

        return newman_tobias_control_mode(spec, state, program)
    if getattr(spec, "model_family", None) == "msmr-spm-isothermal-v1":
        from chemworld.physchem.battery_msmr import msmr_control_mode

        return msmr_control_mode(spec, state, program)
    if getattr(spec, "model_family", None) == "spm-spatial-thermal-v1":
        from chemworld.physchem.battery_spatial_thermal import spatial_thermal_control_mode

        return spatial_thermal_control_mode(spec, state, program)
    if getattr(spec, "model_family", None) == "composite-spm-isothermal-v1":
        from chemworld.physchem.battery_composite import composite_control_mode

        return composite_control_mode(spec, state, program)
    if getattr(spec, "model_family", None) == "half-cell-dfn-isothermal-v1":
        from chemworld.physchem.battery_half_cell_dfn import half_cell_dfn_control_mode

        return half_cell_dfn_control_mode(spec, state, program)
    if getattr(spec, "model_family", None) == "half-cell-spm-isothermal-v1":
        from chemworld.physchem.battery_half_cell import half_cell_control_mode

        return half_cell_control_mode(spec, state, program)
    target = program.value_at(state.time_s)
    if program.unit == "A":
        return "constant_current", target
    if program.unit != "V":
        raise ValueError("battery control uses A or V")
    validate_ocp_cv(spec)
    if getattr(spec, "model_family", None) == "mpm-isothermal-v1":
        from chemworld.physchem.battery_mpm import mpm_fields

        fields = mpm_fields(spec, state, program=program)
        if fields["voltage_V"] < target - 1e-9:
            return "negative_current_limit", -spec.maximum_current_A
        if fields["voltage_V"] > target + 1e-9:
            return "positive_current_limit", spec.maximum_current_A
        return "constant_voltage", target
    if getattr(spec, "model_family", None) == "dfn-isothermal-v1":
        from chemworld.physchem.battery_dfn import dfn_fields

        fields = dfn_fields(spec, state, program=program)
        if fields["voltage_V"] < target - 1e-9:
            return "negative_current_limit", -spec.maximum_current_A
        if fields["voltage_V"] > target + 1e-9:
            return "positive_current_limit", spec.maximum_current_A
        return "constant_voltage", target
    limit = spec.maximum_current_A
    if battery_readout(spec, state, -limit).voltage_V < target:
        return "negative_current_limit", -limit
    if battery_readout(spec, state, limit).voltage_V > target:
        return "positive_current_limit", limit
    return "constant_voltage", target


def battery_algebraic(spec, program):
    """Dimensionless control constraint plus the native terminal-voltage law."""
    if getattr(spec, "plating", None) is not None:
        from chemworld.physchem.battery_plating import plating_algebraic

        return plating_algebraic(spec, program)
    if getattr(spec, "model_family", None) == "split-ocvr-isothermal-v1":
        from chemworld.physchem.battery_split_ocvr import split_ocvr_algebraic

        return split_ocvr_algebraic(spec, program)
    if getattr(spec, "model_family", None) == "lead-acid-loqs-isothermal-v1":
        from chemworld.physchem.battery_lead_acid import lead_acid_algebraic

        return lead_acid_algebraic(spec, program)
    if getattr(spec, "model_family", None) == "composite-dfn-isothermal-v1":
        from chemworld.physchem.battery_dfn_composite import composite_dfn_algebraic

        return composite_dfn_algebraic(spec, program)
    if getattr(spec, "model_family", None) == "collector-spm-isothermal-v1":
        from chemworld.physchem.battery_collector_spm import collector_spm_algebraic

        return collector_spm_algebraic(spec, program)
    if getattr(spec, "model_family", None) == "newman-tobias-isothermal-v1":
        from chemworld.physchem.battery_newman_tobias import newman_tobias_algebraic

        return newman_tobias_algebraic(spec, program)
    if getattr(spec, "model_family", None) == "msmr-spm-isothermal-v1":
        from chemworld.physchem.battery_msmr import msmr_algebraic

        return msmr_algebraic(spec, program)
    if getattr(spec, "model_family", None) == "spm-spatial-thermal-v1":
        from chemworld.physchem.battery_spatial_thermal import spatial_thermal_algebraic

        return spatial_thermal_algebraic(spec, program)
    if getattr(spec, "model_family", None) == "composite-spm-isothermal-v1":
        from chemworld.physchem.battery_composite import composite_algebraic

        return composite_algebraic(spec, program)
    if getattr(spec, "model_family", None) == "half-cell-dfn-isothermal-v1":
        from chemworld.physchem.battery_half_cell_dfn import half_cell_dfn_algebraic

        return half_cell_dfn_algebraic(spec, program)
    if getattr(spec, "model_family", None) == "half-cell-spm-isothermal-v1":
        from chemworld.physchem.battery_half_cell import half_cell_algebraic

        return half_cell_algebraic(spec, program)
    if getattr(spec, "model_family", None) == "mpm-isothermal-v1":
        from chemworld.physchem.battery_mpm import mpm_algebraic

        return mpm_algebraic(spec, program)
    if program.unit == "V":
        validate_ocp_cv(spec)
    current_key, voltage_key = BATTERY_ALGEBRAIC
    variables = (
        ResidualVariable(current_key, "A", spec.maximum_current_A, "algebraic",
                         -spec.maximum_current_A, spec.maximum_current_A),
        ResidualVariable(voltage_key, "V", 1, "algebraic"),
    )
    equations = (ResidualEquation("battery:control", "1"),
                 ResidualEquation("battery:terminal-voltage", "V"))

    def initial_values(state):
        current = program.value_at(state.time_s) if program.unit == "A" else 0.0
        return {current_key: current,
                voltage_key: battery_readout(spec, state, current).voltage_V}

    def residual(state):
        current, voltage = (state.algebraic_values[key] for key in BATTERY_ALGEBRAIC)
        mode, target = battery_control_mode(spec, state, program)
        control = voltage - target if mode == "constant_voltage" else (
            (current - target) / spec.maximum_current_A
        )
        return [control, voltage - battery_readout(spec, state, current).voltage_V]

    return CoupledAlgebraic(variables, equations, initial_values, residual)


def battery_control_current(spec, state, program):
    if getattr(spec, "plating", None) is not None:
        from chemworld.physchem.battery_plating import plating_control_current

        return plating_control_current(spec, state, program)
    if getattr(spec, "model_family", None) == "split-ocvr-isothermal-v1":
        from chemworld.physchem.battery_split_ocvr import split_ocvr_control_current

        return split_ocvr_control_current(spec, state, program)
    if getattr(spec, "model_family", None) == "lead-acid-loqs-isothermal-v1":
        from chemworld.physchem.battery_lead_acid import lead_acid_control_current

        return lead_acid_control_current(spec, state, program)
    if getattr(spec, "model_family", None) == "composite-dfn-isothermal-v1":
        from chemworld.physchem.battery_dfn_composite import composite_dfn_control_current

        return composite_dfn_control_current(spec, state, program)
    if getattr(spec, "model_family", None) == "collector-spm-isothermal-v1":
        from chemworld.physchem.battery_collector_spm import collector_spm_control_current

        return collector_spm_control_current(spec, state, program)
    if getattr(spec, "model_family", None) == "newman-tobias-isothermal-v1":
        from chemworld.physchem.battery_newman_tobias import newman_tobias_control_current

        return newman_tobias_control_current(spec, state, program)
    if getattr(spec, "model_family", None) == "msmr-spm-isothermal-v1":
        from chemworld.physchem.battery_msmr import msmr_control_current

        return msmr_control_current(spec, state, program)
    if getattr(spec, "model_family", None) == "spm-spatial-thermal-v1":
        from chemworld.physchem.battery_spatial_thermal import spatial_thermal_control_current

        return spatial_thermal_control_current(spec, state, program)
    if getattr(spec, "model_family", None) == "composite-spm-isothermal-v1":
        from chemworld.physchem.battery_composite import composite_control_current

        return composite_control_current(spec, state, program)
    if getattr(spec, "model_family", None) == "half-cell-dfn-isothermal-v1":
        from chemworld.physchem.battery_half_cell_dfn import half_cell_dfn_control_current

        return half_cell_dfn_control_current(spec, state, program)
    if getattr(spec, "model_family", None) == "half-cell-spm-isothermal-v1":
        from chemworld.physchem.battery_half_cell import half_cell_control_current

        return half_cell_control_current(spec, state, program)
    if BATTERY_ALGEBRAIC[0] in state.algebraic_values:
        return state.algebraic_values[BATTERY_ALGEBRAIC[0]]
    target = program.value_at(state.time_s)
    if program.unit == "A":
        return target
    if program.unit != "V":
        raise ValueError("battery control uses A or V")
    if getattr(spec, "model_family", None) == "mpm-isothermal-v1":
        from chemworld.physchem.battery_mpm import mpm_fields

        return float(mpm_fields(spec, state, program=program)["current_A"])
    if getattr(spec, "model_family", None) == "dfn-isothermal-v1":
        from chemworld.physchem.battery_dfn import dfn_fields

        return float(dfn_fields(spec, state, program=program)["current_A"])
    if spec.numerical_form == "residual":
        from chemworld.physchem.coupled_system import CoupledStateView

        closure = battery_algebraic(spec, program)
        guess = closure.initial_values(state)

        def residual(t, y, rate, p):
            view = CoupledStateView(
                state.amounts_mol, state.temperature_K, state.volume_L, state.time_s,
                coordinates=state.coordinates,
                algebraic_values=dict(zip(BATTERY_ALGEBRAIC, y, strict=True)),
            )
            return closure.residual(view)

        system = ResidualSystem(
            "battery:electrical-closure", closure.variables, closure.equations, residual,
            ResidualDerivatives("finite_difference"),
        )
        answer = solve_steady_residual(system, [guess[key] for key in BATTERY_ALGEBRAIC],
                                       policy=RootPolicy(tolerance=1e-12))
        if not answer.root.success:
            raise ResidualNumericalError("battery control:" + answer.root.failure_reason)
        return float(answer.root.value[0])
    low, high = -spec.maximum_current_A, spec.maximum_current_A

    def residual(current):
        return battery_readout(spec, state, current).voltage_V - target

    # A finite source remains at its declared current limit when the requested
    # voltage cannot yet be reached. The measured voltage remains the actual V.
    if residual(low) < 0:
        return low
    if residual(high) > 0:
        return high
    return float(brentq(residual, low, high, xtol=1e-12))


def _energy_bins(spec):
    if getattr(spec, "model_family", None) == "half-cell-dfn-isothermal-v1":
        raise ValueError("DFN half-cell chemical-energy and equilibrium-SOH closure is unsupported")
    if getattr(spec, "plating", None) is not None:
        raise ValueError("plating complete chemical-energy closure is unsupported")
    if getattr(spec, "model_family", None) == "split-ocvr-isothermal-v1":
        raise ValueError("Split OCV-R chemical-energy closure is unsupported")
    if getattr(spec, "model_family", None) == "lead-acid-loqs-isothermal-v1":
        raise ValueError(
            "lead-acid LOQS chemical-energy and equilibrium-SOH closure is unsupported")
    if getattr(spec, "model_family", None) == "composite-dfn-isothermal-v1":
        raise ValueError("composite DFN chemical-energy and equilibrium-SOH closure is unsupported")
    if getattr(spec, "model_family", None) == "collector-spm-isothermal-v1":
        raise ValueError("collector SPM chemical-energy and equilibrium-SOH closure is unsupported")
    if getattr(spec, "model_family", None) == "newman-tobias-isothermal-v1":
        raise ValueError("Newman-Tobias chemical-energy and equilibrium-SOH closure is unsupported")
    if getattr(spec, "model_family", None) == "msmr-spm-isothermal-v1":
        raise ValueError("MSMR chemical-energy and equilibrium-SOH closure is unsupported")
    if getattr(spec, "model_family", None) == "composite-spm-isothermal-v1":
        raise ValueError("composite chemical-energy and equilibrium-SOH closure is unsupported")
    if getattr(spec, "model_family", None) == "half-cell-spm-isothermal-v1":
        raise ValueError(
            "finite half-cell chemical-energy and equilibrium-SOH closure is unsupported")
    if getattr(spec, "model_family", None) in {"dfn-isothermal-v1", "mpm-isothermal-v1"}:
        return tuple((name, particle, keys) for name, _, particle, keys, _ in spec.particle_bins)
    return tuple((name, getattr(spec, name), getattr(spec, name).keys(name))
                 for name in ("negative", "positive"))


def battery_chemical_potentials(spec, state):
    if getattr(spec, "model_family", None) == "collector-spm-isothermal-v1":
        raise ValueError("collector SPM chemical potentials are unsupported")
    if getattr(spec, "model_family", None) == "spm-spatial-thermal-v1":
        from chemworld.physchem.battery_spatial_thermal import spatial_thermal_chemical_potentials

        return spatial_thermal_chemical_potentials(spec, state)
    result = {}
    for name, particle, keys in _energy_bins(spec):
        for key, volume in zip(keys, particle.mesh.cell_volumes_m3, strict=True):
            theta = state.amounts_mol[key] / (volume * particle.maximum_concentration_mol_m3)
            result[key] = -F * spec.ocv(name, theta)
    if spec.sei is not None:
        result.update(
            {"sei:EC": 0.0, "sei:carbonate": spec.film_chemical_energy_j_mol, "sei:ethylene": 0.0}
        )
    return result


def battery_chemical_energy_j(spec, state):
    if getattr(spec, "model_family", None) == "collector-spm-isothermal-v1":
        raise ValueError("collector SPM chemical-energy closure is unsupported")
    if getattr(spec, "model_family", None) == "spm-spatial-thermal-v1":
        from chemworld.physchem.battery_spatial_thermal import spatial_thermal_chemical_energy_j

        return spatial_thermal_chemical_energy_j(spec, state)
    energy = 0.0
    for name, particle, keys in _energy_bins(spec):
        a, b = getattr(spec, f"{name}_ocv_intercept_V"), getattr(spec, f"{name}_ocv_slope_V")
        curve = getattr(spec, name + "_ocv")
        for key, volume in zip(keys, particle.mesh.cell_volumes_m3, strict=True):
            n = state.amounts_mol[key]
            if curve is None:
                energy -= F * (
                    a * n + b * n * n / (2 * volume * particle.maximum_concentration_mol_m3))
            else:
                capacity = volume * particle.maximum_concentration_mol_m3
                energy -= F * capacity * curve.integral(curve.stoichiometry_domain[0],
                                                        n / capacity, trial=True)
    if spec.sei is not None:
        energy += state.amounts_mol["sei:carbonate"] * spec.film_chemical_energy_j_mol
    return energy


def battery_dissipation_w(spec, state, current):
    if getattr(spec, "model_family", None) == "half-cell-dfn-isothermal-v1":
        raise ValueError("DFN half-cell complete chemical-energy closure is unsupported")
    """Derive four dissipative contributions independently of endpoint residuals."""
    if getattr(spec, "plating", None) is not None:
        raise ValueError("plating complete dissipation closure is unsupported")
    if getattr(spec, "model_family", None) == "split-ocvr-isothermal-v1":
        raise ValueError("Split OCV-R chemical-energy dissipation closure is unsupported")
    if getattr(spec, "model_family", None) == "lead-acid-loqs-isothermal-v1":
        raise ValueError("lead-acid LOQS dissipation closure is unsupported")
    if getattr(spec, "model_family", None) == "composite-dfn-isothermal-v1":
        raise ValueError("composite DFN complete dissipation closure is unsupported")
    if getattr(spec, "model_family", None) == "collector-spm-isothermal-v1":
        raise ValueError("collector SPM dissipation closure is unsupported")
    if getattr(spec, "model_family", None) == "newman-tobias-isothermal-v1":
        raise ValueError("Newman-Tobias dissipation closure is unsupported")
    if getattr(spec, "model_family", None) == "msmr-spm-isothermal-v1":
        raise ValueError("MSMR dissipation closure is unsupported")
    if getattr(spec, "model_family", None) == "spm-spatial-thermal-v1":
        raise ValueError("spatial thermal dissipation decomposition is unsupported")
    if getattr(spec, "model_family", None) == "composite-spm-isothermal-v1":
        raise ValueError("composite dissipation closure is unsupported")
    if getattr(spec, "model_family", None) == "half-cell-spm-isothermal-v1":
        raise ValueError("finite half-cell dissipation closure is unsupported")
    if spec.has_directional_ocp:
        raise ValueError(
            "directional OCP does not declare complete hysteresis energy or dissipation")
    if getattr(spec, "model_family", None) == "mpm-isothermal-v1":
        from chemworld.physchem.battery_mpm import mpm_dissipation_w

        return mpm_dissipation_w(spec, state, current)
    potentials = battery_chemical_potentials(spec, state)
    reading = battery_readout(spec, state, current)
    interior = boundary = 0.0
    for name, particle in (("negative", spec.negative), ("positive", spec.positive)):
        keys = particle.keys(name)
        rates = ConservativeDiffusion(particle.mesh, particle.diffusivity_m2_s).evaluate(
            [state.amounts_mol[key] for key in keys]
        )
        interior += sum(
            float(rate) * (potentials[left] - potentials[right])
            for rate, left, right in zip(
                rates.face_rates_mol_s[1:-1], keys[:-1], keys[1:], strict=True
            )
        )
        flux = battery_particle_flux(spec, state, name, current)
        theta = particle_surface_fraction(particle, name, state, flux)
        surface_mu = -F * spec.ocv(name, theta)
        boundary += flux * (potentials[keys[-1]] - surface_mu)
    sei_heat = 0.0
    if spec.sei is not None:
        sei_heat = battery_side_rate(spec, state) * battery_sei_affinity_j_mol(spec, state, current)
    return {
        "kinetic_W": reading.negative_main_current_A * reading.negative_overpotential_V
        - current * reading.positive_overpotential_V,
        "ohmic_W": current**2 * (spec.series_resistance_ohm + reading.film_resistance_ohm),
        "interior_diffusion_W": interior,
        "surface_diffusion_W": boundary,
        "sei_reaction_W": sei_heat,
    }


def battery_sei_affinity_j_mol(spec, state, current):
    if getattr(spec, "model_family", None) == "half-cell-dfn-isothermal-v1":
        raise ValueError("DFN half-cell side-reaction chemical affinity is unsupported")
    """Electrochemical driving force for the declared irreversible side process."""
    if getattr(spec, "plating", None) is not None:
        raise ValueError("plating and SEI affinity coupling is unsupported")
    if getattr(spec, "model_family", None) == "split-ocvr-isothermal-v1":
        raise ValueError("Split OCV-R SEI chemistry is unsupported")
    if getattr(spec, "model_family", None) == "lead-acid-loqs-isothermal-v1":
        raise ValueError("lead-acid LOQS SEI chemistry is unsupported")
    if getattr(spec, "model_family", None) == "composite-dfn-isothermal-v1":
        raise ValueError("composite DFN SEI affinity is unsupported")
    if getattr(spec, "model_family", None) == "collector-spm-isothermal-v1":
        raise ValueError("collector SPM SEI affinity is unsupported")
    if getattr(spec, "model_family", None) == "newman-tobias-isothermal-v1":
        raise ValueError("Newman-Tobias SEI affinity is unsupported")
    if getattr(spec, "model_family", None) == "msmr-spm-isothermal-v1":
        raise ValueError("MSMR SEI affinity is unsupported")
    if getattr(spec, "model_family", None) == "spm-spatial-thermal-v1":
        raise ValueError("spatial thermal SEI chemistry is unsupported")
    if getattr(spec, "model_family", None) == "composite-spm-isothermal-v1":
        raise ValueError("composite SEI chemistry is unsupported")
    if getattr(spec, "model_family", None) == "half-cell-spm-isothermal-v1":
        raise ValueError("finite half-cell SEI chemistry is unsupported")
    reading = battery_readout(spec, state, current)
    mu = -F * (
        spec.negative_ocv_intercept_V
        + spec.negative_ocv_slope_V * reading.negative_surface_fraction
    )
    return 2 * mu - spec.film_chemical_energy_j_mol - 2 * F * reading.negative_overpotential_V


class BatteryThermodynamics:
    def __init__(self, spec):
        self.spec = spec
        self.inventory_keys = spec.inventory_keys

    def heat_capacity_j_k(self, state):
        return self.spec.thermal.heat_capacity_J_K if self.spec.thermal else 1.0

    def internal_energy_j(self, state):
        if self.spec.thermal is None:
            return None
        return battery_chemical_energy_j(self.spec, state) + self.spec.thermal.heat_capacity_J_K * (
            state.temperature_K - self.spec.temperature_K
        )

    def chemical_energy_rate_w(self, state, amount_rates, contributions):
        if self.spec.thermal is None:
            return 0.0
        if any(c.legacy_reaction_energy_W is not None for c in contributions):
            raise ValueError("battery chemical energy cannot also receive legacy reaction heat")
        mu = battery_chemical_potentials(self.spec, state)
        return sum(mu[key] * rate for key, rate in amount_rates.items())


def _ocp_spm_rhs_jacobian(spec, state, program, identities):
    """Exact shared-residual derivatives for the declared isothermal OCP scope."""
    from scipy import sparse

    index = {key: i for i, key in enumerate(identities)}
    jac = np.zeros((len(identities), len(identities)))
    ik, vk = BATTERY_ALGEBRAIC
    current = state.algebraic_values[ik]
    dv = np.zeros(len(identities))
    dv[index[ik]] = -spec.series_resistance_ohm
    for name, qsign, vsign in (("negative", 1, -1), ("positive", -1, 1)):
        particle = getattr(spec, name)
        keys = particle.keys(name)
        columns = [index[k] for k in keys]
        jac[np.ix_(columns, columns)] = ConservativeDiffusion(
            particle.mesh, particle.diffusivity_m2_s).inventory_jacobian_s_inv.toarray()
        jac[index[keys[-1]], index[ik]] = -qsign / F
        theta = particle_surface_fraction(particle, name, state, qsign * current / F)
        clip = (spec.surface_margin / 2 if spec.kinetics_stoichiometry_clip is None
                else spec.kinetics_stoichiometry_clip)
        trial = float(np.clip(theta, clip, 1 - clip))
        j0 = particle.exchange_current_scale_A_m2 * sqrt(trial * (1 - trial))
        z = qsign * current / (2 * particle.area_m2 * j0)
        factor = 2 * R * state.temperature_K / F / sqrt(1 + z * z)
        eta_q = factor / (2 * particle.area_m2 * j0)
        eta_theta = (-factor * z * (1 - 2 * trial) / (2 * trial * (1 - trial))
                     if theta == trial else 0.0)
        _, slope, di, dh = spec.ocp_derivatives(name, theta, current, state)
        theta_i = -qsign * particle.mesh.spherical_surface_distance_m / (
            F * particle.area_m2 * particle.diffusivity_m2_s
            * particle.maximum_concentration_mol_m3)
        dv[index[ik]] += vsign * (di + (slope + eta_theta) * theta_i + eta_q * qsign)
        dv[index[keys[-1]]] += vsign * (slope + eta_theta) / (
            particle.mesh.cell_volumes_m3[-1] * particle.maximum_concentration_mol_m3)
        if "temperature" in index:
            dv[index["temperature"]] += vsign * 2 * R / F * asinh(z)
        key = hysteresis_key(name)
        if key in index:
            dv[index[key]] += vsign * dh
            ocp = getattr(spec, name + "_ocp")
            _, dri, drh = ocp.rate_derivatives(
                name, current, state.coordinates[key], particle.capacity_mol)
            jac[index[key], index[ik]] = dri
            jac[index[key], index[key]] = drh
    jac[index["discharged_charge_C"], index[ik]] = 1
    jac[index["delivered_work_J"]] = current * dv
    jac[index["delivered_work_J"], index[ik]] += battery_readout(spec, state, current).voltage_V
    mode, _ = battery_control_mode(spec, state, program)
    if mode == "constant_voltage":
        jac[index[ik], index[vk]] = -1
    else:
        jac[index[ik], index[ik]] = -1 / spec.maximum_current_A
    jac[index[vk]] = dv
    jac[index[vk], index[vk]] -= 1
    return sparse.csc_matrix(jac)


def battery_system(spec, program, *, thermal_device=None, control_events=()):
    if getattr(spec, "parameter_binding", None) is not None and thermal_device is not None:
        raise ValueError("source SPM thermal-device coupling is unsupported")
    if getattr(spec, "plating", None) is not None:
        from chemworld.physchem.battery_plating import plating_system

        return plating_system(spec, program, thermal_device=thermal_device,
                              control_events=control_events)
    if getattr(spec, "model_family", None) == "split-ocvr-isothermal-v1":
        from chemworld.physchem.battery_split_ocvr import split_ocvr_system

        return split_ocvr_system(spec, program, thermal_device=thermal_device,
                                 control_events=control_events)
    if getattr(spec, "model_family", None) == "lead-acid-loqs-isothermal-v1":
        from chemworld.physchem.battery_lead_acid import lead_acid_system

        return lead_acid_system(spec, program, thermal_device=thermal_device,
                                control_events=control_events)
    if getattr(spec, "model_family", None) == "composite-dfn-isothermal-v1":
        from chemworld.physchem.battery_dfn_composite import composite_dfn_system

        return composite_dfn_system(spec, program, thermal_device=thermal_device,
                                    control_events=control_events)
    if getattr(spec, "model_family", None) == "collector-spm-isothermal-v1":
        from chemworld.physchem.battery_collector_spm import collector_spm_system

        return collector_spm_system(spec, program, thermal_device=thermal_device,
                                    control_events=control_events)
    if getattr(spec, "model_family", None) == "newman-tobias-isothermal-v1":
        from chemworld.physchem.battery_newman_tobias import newman_tobias_system

        return newman_tobias_system(spec, program, thermal_device=thermal_device,
                                    control_events=control_events)
    if getattr(spec, "model_family", None) == "msmr-spm-isothermal-v1":
        from chemworld.physchem.battery_msmr import msmr_system

        return msmr_system(spec, program, thermal_device=thermal_device,
                           control_events=control_events)
    if getattr(spec, "model_family", None) == "spm-spatial-thermal-v1":
        from chemworld.physchem.battery_spatial_thermal import spatial_thermal_system

        return spatial_thermal_system(spec, program, thermal_device=thermal_device,
                                      control_events=control_events)
    if getattr(spec, "model_family", None) == "composite-spm-isothermal-v1":
        from chemworld.physchem.battery_composite import composite_system

        return composite_system(spec, program, thermal_device=thermal_device,
                                control_events=control_events)
    if getattr(spec, "model_family", None) == "half-cell-dfn-isothermal-v1":
        from chemworld.physchem.battery_half_cell_dfn import half_cell_dfn_system

        return half_cell_dfn_system(
            spec, program, thermal_device=thermal_device, control_events=control_events)
    if getattr(spec, "model_family", None) == "half-cell-spm-isothermal-v1":
        from chemworld.physchem.battery_half_cell import half_cell_system

        return half_cell_system(spec, program, thermal_device=thermal_device,
                                control_events=control_events)
    if getattr(spec, "model_family", None) == "mpm-isothermal-v1":
        from chemworld.physchem.battery_mpm import mpm_system

        return mpm_system(spec, program, thermal_device=thermal_device,
                          control_events=control_events)
    if getattr(spec, "model_family", None) == "dfn-isothermal-v1":
        from chemworld.physchem.battery_dfn import dfn_system

        return dfn_system(spec, program, thermal_device=thermal_device,
                          control_events=control_events)
    if program.unit not in {"A", "V"} or any(
        (
            abs(value) > spec.maximum_current_A
            if program.unit == "A"
            else not spec.minimum_voltage_V <= value <= spec.maximum_voltage_V
        )
        for _, value in program.setpoints
    ):
        raise ValueError(
            "SPM requires a current programme or voltage programme inside the device domain"
        )
    if spec.thermal is None and thermal_device is not None:
        raise ValueError("a thermal device requires the explicit battery energy model")
    processes = [
        ConservativeDiffusion(particle.mesh, particle.diffusivity_m2_s)
        .couple(particle.keys(name), process_id=f"battery:{name}", owner_id=name)
        .process
        for name, particle in (("negative", spec.negative), ("positive", spec.positive))
    ]
    nk, pk = spec.negative.keys("negative"), spec.positive.keys("positive")

    def transfer(state):
        current = battery_control_current(spec, state, program)
        reading = battery_readout(spec, state, current)
        source, target = (nk[-1], pk[-1]) if current >= 0 else (pk[-1], nk[-1])
        return ProcessContribution(
            internal_transfers=(InternalTransfer("battery:Li", source, target, abs(current) / F),),
            coordinate_rates={
                "discharged_charge_C": current,
                "delivered_work_J": current * reading.voltage_V,
                **hysteresis_rates(spec, state, current),
            },
            energy_port_rates_W={"battery:terminal": -current * reading.voltage_V}
            if spec.thermal
            else {},
        )

    def validate(state):
        validate_hysteresis(spec, state)
        if (spec.thermal is None and abs(state.temperature_K - spec.temperature_K) > 1e-8) or abs(
            state.volume_L - spec.volume_l
        ) > 1e-12:
            raise ValueError("SPM requires its declared isothermal volume")
        if (
            spec.thermal
            and not spec.thermal.minimum_temperature_K - 1e-8
            <= state.temperature_K
            <= spec.thermal.maximum_temperature_K + 1e-8
        ):
            raise ValueError("battery temperature is outside the caloric domain")
        for name, particle in (("negative", spec.negative), ("positive", spec.positive)):
            for key, volume in zip(particle.keys(name), particle.mesh.cell_volumes_m3, strict=True):
                if state.amounts_mol[key] > volume * particle.maximum_concentration_mol_m3 * (
                    1 + 1e-8
                ):
                    raise ValueError("particle inventory exceeds its finite capacity")
        if spec.electrolyte is not None and np.min(spec.electrolyte.concentrations(state)) < (
            spec.electrolyte.minimum_concentration_mol_m3 * (1 - 1e-7)
        ):
            raise ValueError("electrolyte is outside its declared positive-concentration domain")

    processes.append(CoupledProcess("battery:intercalation", transfer, validate_state=validate))
    if spec.sei is not None:

        def grow_sei(state):
            rate = battery_side_rate(spec, state)
            return ProcessContribution(
                amount_rates_mol_s={
                    nk[-1]: -2 * rate,
                    "sei:EC": -rate,
                    "sei:carbonate": rate,
                    "sei:ethylene": rate,
                }
            )

        processes.append(CoupledProcess("battery:solvent-limited-sei", grow_sei))
    events = []
    if spec.electrolyte is not None:
        electrolyte = spec.electrolyte
        processes.extend(electrolyte.processes(
            lambda state: battery_control_current(spec, state, program)))
        events.append(CoupledEvent(
            "battery:electrolyte:depleted",
            lambda state: float(np.min(electrolyte.concentrations(state)))
            - electrolyte.minimum_concentration_mol_m3,
        ))
    if program.unit == "V" and spec.numerical_form == "residual":
        events.extend(
            CoupledEvent(
                f"battery:cv:{label}-limit",
                lambda state, current=current: (
                    battery_readout(spec, state, current).voltage_V - program.value_at(state.time_s)
                ),
                direction=0, terminal=False,
            )
            for label, current in (("negative", -spec.maximum_current_A),
                                   ("positive", spec.maximum_current_A))
        )
    if spec.sei is not None:
        events.append(
            CoupledEvent(
                "battery:sei:affinity-limit",
                lambda state: battery_sei_affinity_j_mol(
                    spec, state, battery_control_current(spec, state, program)
                ),
            )
        )
    for name, particle in (("negative", spec.negative), ("positive", spec.positive)):

        def surface(state, p=particle, n=name):
            return particle_surface_fraction(
                p,
                n,
                state,
                battery_particle_flux(
                    spec, state, n, battery_control_current(spec, state, program)
                ),
            )

        events.extend(
            (
                CoupledEvent(
                    f"battery:{name}:depleted",
                    lambda s, read=surface: read(s) - spec.surface_margin,
                ),
                CoupledEvent(
                    f"battery:{name}:saturated",
                    lambda s, read=surface: 1 - spec.surface_margin - read(s),
                ),
            )
        )
    events.extend(
        (
            CoupledEvent(
                "battery:minimum-voltage",
                lambda s: (
                    battery_readout(spec, s, battery_control_current(spec, s, program)).voltage_V
                    - spec.minimum_voltage_V
                ),
            ),
            CoupledEvent(
                "battery:maximum-voltage",
                lambda s: (
                    spec.maximum_voltage_V
                    - battery_readout(spec, s, battery_control_current(spec, s, program)).voltage_V
                ),
            ),
        )
    )
    isothermal = spec.thermal is None or (
        thermal_device is not None
        and thermal_device.enabled
        and thermal_device.mode == "isothermal"
    )
    if spec.thermal:
        events.extend(
            (
                CoupledEvent(
                    "battery:minimum-temperature",
                    lambda s: s.temperature_K - spec.thermal.minimum_temperature_K,
                ),
                CoupledEvent(
                    "battery:maximum-temperature",
                    lambda s: spec.thermal.maximum_temperature_K - s.temperature_K,
                ),
            )
        )

    def heat(state):
        return HeatFlux(
            0 if thermal_device is None else thermal_device.delivered_power_w(state.temperature_K),
            0
            if spec.thermal is None
            else spec.thermal.loss_conductance_W_K
            * (state.temperature_K - spec.thermal.environment_temperature_K),
        )

    laws = {
        "lithium_mol": dict.fromkeys((*nk, *pk), 1),
        "negative_faradaic_C": {**dict.fromkeys(nk, F), "discharged_charge_C": 1},
        "positive_faradaic_C": {**dict.fromkeys(pk, F), "discharged_charge_C": -1},
    }
    scales = {
        key: particle.capacity_mol / particle.cells
        for name, particle in (("negative", spec.negative), ("positive", spec.positive))
        for key in particle.keys(name)
    }
    if spec.sei is not None:
        laws["lithium_mol"]["sei:carbonate"] = 2
        laws["negative_faradaic_C"]["sei:carbonate"] = 2 * F
        laws.update(
            {
                "carbon_mol": {"sei:EC": 3, "sei:carbonate": 1, "sei:ethylene": 2},
                "hydrogen_mol": {"sei:EC": 4, "sei:ethylene": 4},
                "oxygen_mol": {"sei:EC": 3, "sei:carbonate": 3},
            }
        )
        scales.update(
            dict.fromkeys(
                SEI_SPECIES,
                max(
                    1e-10,
                    spec.sei.solvent_volume_m3 * spec.sei.initial_solvent_concentration_mol_m3,
                ),
            )
        )
    if spec.electrolyte is not None:
        laws["lithium_mol"].update(dict.fromkeys(spec.electrolyte.keys, 1))
        laws["electrolyte_salt_mol"] = dict.fromkeys(spec.electrolyte.keys, 1)
        scales.update(spec.electrolyte.initial_amounts())
    return CoupledSystem(
        BatteryThermodynamics(spec),
        tuple(processes),
        isothermal=isothermal,
        events=(*events, *control_events),
        heat_flux=heat,
        energy_ports=(EnergyPort("battery:terminal", "external_source", "battery", "in"),)
        if spec.thermal
        else (),
        coordinates=(
            CoupledCoordinate("discharged_charge_C", "C", spec.usable_charge_c),
            CoupledCoordinate("delivered_work_J", "J", spec.usable_charge_c * 4),
            *hysteresis_coordinates(spec),
        ),
        inventory_scales_mol=scales,
        conservation_laws=laws,
        numerical_form=spec.numerical_form,
        algebraic=battery_algebraic(spec, program) if spec.numerical_form == "residual" else None,
        residual_rhs_jacobian=(
            (lambda state, identities: _ocp_spm_rhs_jacobian(spec, state, program, identities))
            if spec.has_directional_ocp else None),
    )


def advance_battery(
    spec,
    state,
    program,
    duration_s,
    *,
    evaluation_times_s=None,
    thermal_device=None,
    control_events=(),
):
    if getattr(spec, "plating", None) is not None:
        from chemworld.physchem.battery_plating import advance_plating

        return advance_plating(spec, state, program, duration_s,
                               evaluation_times_s=evaluation_times_s,
                               thermal_device=thermal_device, control_events=control_events)
    if getattr(spec, "model_family", None) == "split-ocvr-isothermal-v1":
        from chemworld.physchem.battery_split_ocvr import advance_split_ocvr

        return advance_split_ocvr(spec, state, program, duration_s,
                                  evaluation_times_s=evaluation_times_s,
                                  thermal_device=thermal_device, control_events=control_events)
    if getattr(spec, "model_family", None) == "lead-acid-loqs-isothermal-v1":
        from chemworld.physchem.battery_lead_acid import advance_lead_acid

        return advance_lead_acid(spec, state, program, duration_s,
                                evaluation_times_s=evaluation_times_s,
                                thermal_device=thermal_device, control_events=control_events)
    if getattr(spec, "model_family", None) == "composite-dfn-isothermal-v1":
        from chemworld.physchem.battery_dfn_composite import advance_composite_dfn

        return advance_composite_dfn(spec, state, program, duration_s,
                                     evaluation_times_s=evaluation_times_s,
                                     thermal_device=thermal_device, control_events=control_events)
    if getattr(spec, "model_family", None) == "collector-spm-isothermal-v1":
        from chemworld.physchem.battery_collector_spm import advance_collector_spm

        return advance_collector_spm(spec, state, program, duration_s,
                                     evaluation_times_s=evaluation_times_s,
                                     thermal_device=thermal_device, control_events=control_events)
    if getattr(spec, "model_family", None) == "newman-tobias-isothermal-v1":
        from chemworld.physchem.battery_newman_tobias import advance_newman_tobias

        return advance_newman_tobias(spec, state, program, duration_s,
                                      evaluation_times_s=evaluation_times_s,
                                      thermal_device=thermal_device, control_events=control_events)
    if getattr(spec, "model_family", None) == "msmr-spm-isothermal-v1":
        from chemworld.physchem.battery_msmr import advance_msmr

        return advance_msmr(spec, state, program, duration_s,
                            evaluation_times_s=evaluation_times_s,
                            thermal_device=thermal_device, control_events=control_events)
    if getattr(spec, "model_family", None) == "spm-spatial-thermal-v1":
        from chemworld.physchem.battery_spatial_thermal import advance_spatial_thermal

        return advance_spatial_thermal(spec, state, program, duration_s,
                                       evaluation_times_s=evaluation_times_s,
                                       thermal_device=thermal_device, control_events=control_events)
    if getattr(spec, "model_family", None) == "composite-spm-isothermal-v1":
        from chemworld.physchem.battery_composite import advance_composite

        return advance_composite(spec, state, program, duration_s,
                                 evaluation_times_s=evaluation_times_s,
                                 thermal_device=thermal_device, control_events=control_events)
    if getattr(spec, "model_family", None) == "half-cell-dfn-isothermal-v1":
        from chemworld.physchem.battery_half_cell_dfn import advance_half_cell_dfn

        return advance_half_cell_dfn(
            spec, state, program, duration_s, evaluation_times_s=evaluation_times_s,
            thermal_device=thermal_device, control_events=control_events)
    if getattr(spec, "model_family", None) == "half-cell-spm-isothermal-v1":
        from chemworld.physchem.battery_half_cell import advance_half_cell

        return advance_half_cell(spec, state, program, duration_s,
                                 evaluation_times_s=evaluation_times_s,
                                 thermal_device=thermal_device, control_events=control_events)
    if (
        thermal_device is not None
        and thermal_device.enabled
        and thermal_device.mode == "isothermal"
        and abs(state.temperature_K - thermal_device.setpoint_K) > 1e-8
    ):
        raise ValueError("isothermal bath cannot jump battery temperature")
    result = battery_system(
        spec, program, thermal_device=thermal_device, control_events=control_events
    ).advance(state, duration_s, evaluation_times_s=evaluation_times_s)
    return replace(result, solver_diagnostic={
        **result.solver_diagnostic,
        "battery_control_mode": battery_control_mode(spec, result.final_state, program)[0],
    })
