"""Equipment heat transfer with fouling and phase-change energy ledgers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import exp, isfinite, log, pi
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from chemworld.physchem.gas_mechanism_batch import NASA7GasThermodynamics
    from chemworld.physchem.helmholtz_equilibrium_thermal import (
        HelmholtzEquilibriumIsochoricThermodynamics,
    )
    from chemworld.physchem.helmholtz_thermal import HelmholtzIsochoricThermodynamics

from chemworld.physchem.coupled_system import ConstantCvThermodynamics, SpeciesInternalEnergy
from chemworld.physchem.property_packages import PropertyPackageThermodynamics

HeatTransferSurface = Literal["jacket", "coil", "shell"]
PhaseChangeMode = Literal["none", "boiling", "condensation"]


@dataclass(frozen=True)
class ThermalDeviceSpec:
    """Reusable heater boundary; the connected material owns its temperature.

    The device is an ideal utility with a finite delivered-power limit, not an
    independently integrated heater body. Calibration and control belong to
    the device and remain unchanged when its thermal port is reconnected.
    """

    conductance_W_K: float = 0.2
    maximum_power_W: float = 5.0
    setpoint_K: float = 350.0
    calibration_factor: float = 1.0
    enabled: bool = True
    mode: Literal["finite_power", "isothermal"] = "finite_power"
    minimum_setpoint_K: float = 300.0
    maximum_setpoint_K: float = 430.0
    maximum_cooling_power_W: float = 0.0

    def __post_init__(self) -> None:
        for name in ("conductance_W_K", "maximum_power_W", "maximum_cooling_power_W"):
            _nonnegative(getattr(self, name), name)
        _positive(self.setpoint_K, "setpoint_K")
        _positive(self.calibration_factor, "calibration_factor")
        if not (
            0 < self.minimum_setpoint_K <= self.setpoint_K <= self.maximum_setpoint_K
            and isfinite(self.maximum_setpoint_K)
        ):
            raise ValueError("setpoint must lie within the device's finite operating range")
        if not isinstance(self.enabled, bool) or self.mode not in {"finite_power", "isothermal"}:
            raise ValueError("thermal device requires boolean enable and a supported mode")

    @property
    def supported_operations(self) -> tuple[str, ...]:
        return ("heat", "wait", "pause_control", "resume_control")

    def delivered_power_w(self, sample_temperature_K: float) -> float:
        _positive(sample_temperature_K, "sample_temperature_K")
        if not self.enabled or self.mode == "isothermal":
            # Isothermal compensation is computed by the common energy balance.
            return 0.0
        return min(
            self.maximum_power_W,
            max(
                -self.maximum_cooling_power_W,
                self.conductance_W_K
                * self.calibration_factor
                * (self.setpoint_K - sample_temperature_K),
            ),
        )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload) -> ThermalDeviceSpec:
        return cls(**payload)


@dataclass(frozen=True)
class CondensedBackgroundSpec:
    """Declared volume carrier, without an invented molecular composition.

    The authored volumetric Cv defines U(T). Condensed PV/Poynting work is
    neglected and port H is approximated by U, without a Cp-to-Cv conversion.
    Volume remains an explicit L stock.
    """

    volumetric_cv_J_L_K: float
    feed_temperature_K: float = 298.15
    solvent_index: int = 0
    source_vessel_id: str | None = None

    def __post_init__(self):
        _positive(self.volumetric_cv_J_L_K, "background heat capacity")
        _positive(self.feed_temperature_K, "background feed temperature")
        if self.source_vessel_id is not None and (
            not isinstance(self.source_vessel_id, str) or not self.source_vessel_id
        ):
            raise ValueError("carrier source must name an explicitly declared vessel")
        if (
            isinstance(self.solvent_index, bool)
            or not isinstance(self.solvent_index, int)
            or self.solvent_index not in range(4)
        ):
            raise ValueError("background carrier requires one declared solvent stock")


@dataclass(frozen=True)
class ThermalSystemSpec:
    """Constitutive data for one local system; owns no evolving inventory."""

    thermodynamics: (
        ConstantCvThermodynamics | PropertyPackageThermodynamics | NASA7GasThermodynamics
        | HelmholtzIsochoricThermodynamics | HelmholtzEquilibriumIsochoricThermodynamics
    )
    pressure_law: Literal["ideal_gas", "fixed_pressure", "helmholtz"] = "ideal_gas"
    environment_temperature_K: float = 300.0
    loss_conductance_W_K: float = 0.01
    minimum_temperature_K: float = 300.0
    maximum_temperature_K: float = 450.0
    stop_temperature_K: float = 440.0
    maximum_pressure_Pa: float = 500000.0
    condensed_background: CondensedBackgroundSpec | None = None

    def __post_init__(self) -> None:
        from chemworld.physchem.gas_mechanism_batch import NASA7GasThermodynamics
        from chemworld.physchem.helmholtz_equilibrium_thermal import (
            HelmholtzEquilibriumIsochoricThermodynamics,
        )
        from chemworld.physchem.helmholtz_thermal import HelmholtzIsochoricThermodynamics

        is_helmholtz = isinstance(self.thermodynamics, (
            HelmholtzIsochoricThermodynamics, HelmholtzEquilibriumIsochoricThermodynamics))
        if is_helmholtz != (self.pressure_law == "helmholtz"):
            raise ValueError("Helmholtz calorics and pressure must use the same EOS")
        if isinstance(self.thermodynamics, (
            HelmholtzIsochoricThermodynamics, HelmholtzEquilibriumIsochoricThermodynamics
        )):
            fluid = (self.thermodynamics.fluid
                     if isinstance(self.thermodynamics, HelmholtzIsochoricThermodynamics)
                     else self.thermodynamics.phase_model.fluid)
            if not (fluid.minimum_temperature_K <= self.minimum_temperature_K
                    < self.maximum_temperature_K <= fluid.maximum_temperature_K):
                raise ValueError("thermal bounds leave the Helmholtz source temperature domain")

        if (isinstance(self.thermodynamics, NASA7GasThermodynamics)
                and self.pressure_law != "ideal_gas"):
            raise ValueError("NASA7 gas caloric data require ideal-gas pressure closure")
        _positive(self.environment_temperature_K, "environment_temperature_K")
        _positive(self.maximum_pressure_Pa, "maximum_pressure_Pa")
        _nonnegative(self.loss_conductance_W_K, "loss_conductance_W_K")
        if not (
            0 < self.minimum_temperature_K < self.stop_temperature_K < self.maximum_temperature_K
        ) or not isfinite(self.maximum_temperature_K):
            raise ValueError("thermal limits must be finite, positive and ordered")
        if self.pressure_law not in {"ideal_gas", "fixed_pressure", "helmholtz"}:
            raise ValueError("unsupported local pressure law")
        if self.condensed_background is not None and (
            not isinstance(self.condensed_background, CondensedBackgroundSpec)
            or self.pressure_law != "fixed_pressure"
        ):
            raise ValueError("a condensed background requires an explicit fixed-pressure model")
        validate = getattr(self.thermodynamics, "validate_temperature_interval", None)
        if validate is not None:
            validate(self.minimum_temperature_K, self.maximum_temperature_K)
        if (
            isinstance(self.thermodynamics, PropertyPackageThermodynamics)
            and any(
                value.reference.phase == "gas" for value in self.thermodynamics.species.values()
            )
            and self.pressure_law != "ideal_gas"
        ):
            raise ValueError("native gas caloric data require the ideal-gas pressure closure")

    def to_dict(self) -> dict[str, object]:
        return {
            "thermodynamics": self.thermodynamics.to_dict()
            if hasattr(self.thermodynamics, "to_dict")
            else {
                "species": {
                    key: asdict(value) for key, value in self.thermodynamics.species.items()
                },
                "support_heat_capacity_J_K": self.thermodynamics.support_heat_capacity_J_K,
                "reference_temperature_K": self.thermodynamics.reference_temperature_K,
            },
            **{
                key: getattr(self, key)
                for key in (
                    "pressure_law",
                    "environment_temperature_K",
                    "loss_conductance_W_K",
                    "minimum_temperature_K",
                    "maximum_temperature_K",
                    "stop_temperature_K",
                    "maximum_pressure_Pa",
                )
            },
            **(
                {"condensed_background": asdict(self.condensed_background)}
                if self.condensed_background is not None
                else {}
            ),
        }

    @classmethod
    def from_dict(cls, payload) -> ThermalSystemSpec:
        values = dict(payload)
        if values.get("condensed_background") is not None:
            values["condensed_background"] = CondensedBackgroundSpec(
                **values["condensed_background"]
            )
        thermal = dict(values.pop("thermodynamics"))
        if thermal.get("model") == "helmholtz-equilibrium-isochoric-v1":
            from chemworld.physchem.helmholtz_equilibrium_thermal import (
                HelmholtzEquilibriumIsochoricThermodynamics,
            )

            return cls(HelmholtzEquilibriumIsochoricThermodynamics.from_dict(thermal), **values)
        if thermal.get("model") == "helmholtz-isochoric-thermal-v1":
            from chemworld.physchem.helmholtz_thermal import HelmholtzIsochoricThermodynamics

            return cls(HelmholtzIsochoricThermodynamics.from_dict(thermal), **values)
        if thermal.get("model") == "nasa7-ideal-gas-v1":
            from chemworld.physchem.gas_mechanism_batch import NASA7GasThermodynamics

            return cls(NASA7GasThermodynamics.from_dict(thermal), **values)
        if "model" in thermal:
            return cls(PropertyPackageThermodynamics.from_dict(thermal), **values)
        thermal["species"] = {
            key: SpeciesInternalEnergy(**value) for key, value in thermal["species"].items()
        }
        return cls(ConstantCvThermodynamics(**thermal), **values)


@dataclass(frozen=True)
class TubularHeatTransferBoundarySpec:
    """Geometry-resolved single-phase tube thermal boundary.

    It converts an overall heat-transfer coefficient and wetted tube perimeter
    into the distributed ``UA/L`` contract consumed by the PFR solver.
    """

    inner_diameter_m: float
    overall_u_W_m2_K: float
    boundary_temperature_K: float
    effectiveness_factor: float = 1.0
    provenance_id: str = ""

    def __post_init__(self) -> None:
        _positive(self.inner_diameter_m, "inner_diameter_m")
        _positive(self.overall_u_W_m2_K, "overall_u_W_m2_K")
        _positive(self.boundary_temperature_K, "boundary_temperature_K")
        if not 0.0 < self.effectiveness_factor <= 1.0 or not isfinite(self.effectiveness_factor):
            raise ValueError("effectiveness_factor must be finite and in (0, 1]")
        if not self.provenance_id.strip():
            raise ValueError("provenance_id cannot be empty")

    @property
    def wetted_perimeter_m(self) -> float:
        return pi * self.inner_diameter_m

    @property
    def conductance_per_length_W_m_K(self) -> float:  # noqa: N802 - unit suffix
        return self.overall_u_W_m2_K * self.wetted_perimeter_m * self.effectiveness_factor

    def to_dict(self) -> dict[str, float | str]:
        return {
            "inner_diameter_m": self.inner_diameter_m,
            "overall_u_W_m2_K": self.overall_u_W_m2_K,
            "boundary_temperature_K": self.boundary_temperature_K,
            "effectiveness_factor": self.effectiveness_factor,
            "wetted_perimeter_m": self.wetted_perimeter_m,
            "conductance_per_length_W_m_K": self.conductance_per_length_W_m_K,
            "provenance_id": self.provenance_id,
        }


@dataclass(frozen=True)
class HeatTransferEquipmentSpec:
    equipment_id: str
    surface_type: HeatTransferSurface
    area_m2: float
    clean_overall_u_W_m2_K: float
    geometry_correction_factor: float = 1.0
    jacket_coverage_fraction: float = 1.0
    provenance_id: str = ""

    def __post_init__(self) -> None:
        if not self.equipment_id or not self.provenance_id:
            raise ValueError("equipment_id and provenance_id cannot be empty")
        if self.surface_type not in {"jacket", "coil", "shell"}:
            raise ValueError("surface_type must be jacket, coil, or shell")
        _positive(self.area_m2, "area_m2")
        _positive(self.clean_overall_u_W_m2_K, "clean_overall_u_W_m2_K")
        if not 0.25 <= self.geometry_correction_factor <= 1.5:
            raise ValueError("geometry_correction_factor must be in [0.25, 1.5]")
        if not 0.0 < self.jacket_coverage_fraction <= 1.0:
            raise ValueError("jacket_coverage_fraction must be in (0, 1]")
        if self.surface_type != "jacket" and self.jacket_coverage_fraction != 1.0:
            raise ValueError("jacket_coverage_fraction only applies to jacket surfaces")

    @property
    def surface_correction_factor(self) -> float:
        coverage = self.jacket_coverage_fraction if self.surface_type == "jacket" else 1.0
        return coverage * self.geometry_correction_factor

    def to_dict(self) -> dict[str, object]:
        return {
            "equipment_id": self.equipment_id,
            "surface_type": self.surface_type,
            "area_m2": self.area_m2,
            "clean_overall_u_W_m2_K": self.clean_overall_u_W_m2_K,
            "geometry_correction_factor": self.geometry_correction_factor,
            "jacket_coverage_fraction": self.jacket_coverage_fraction,
            "surface_correction_factor": self.surface_correction_factor,
            "provenance_id": self.provenance_id,
        }


@dataclass(frozen=True)
class FoulingEvolutionSpec:
    model_id: str
    initial_resistance_m2_K_W: float
    asymptotic_resistance_m2_K_W: float
    rate_constant_per_s: float
    provenance_id: str

    def __post_init__(self) -> None:
        for name, value in (
            ("initial_resistance_m2_K_W", self.initial_resistance_m2_K_W),
            ("asymptotic_resistance_m2_K_W", self.asymptotic_resistance_m2_K_W),
            ("rate_constant_per_s", self.rate_constant_per_s),
        ):
            _nonnegative(value, name)
        if self.asymptotic_resistance_m2_K_W < self.initial_resistance_m2_K_W:
            raise ValueError("asymptotic fouling resistance cannot be below initial")
        if not self.model_id or not self.provenance_id:
            raise ValueError("model_id and provenance_id cannot be empty")

    def resistance_at(self, elapsed_time_s: float) -> float:
        _nonnegative(elapsed_time_s, "elapsed_time_s")
        span = self.asymptotic_resistance_m2_K_W - self.initial_resistance_m2_K_W
        return self.initial_resistance_m2_K_W + span * (
            1.0 - exp(-self.rate_constant_per_s * elapsed_time_s)
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "initial_resistance_m2_K_W": self.initial_resistance_m2_K_W,
            "asymptotic_resistance_m2_K_W": self.asymptotic_resistance_m2_K_W,
            "rate_constant_per_s": self.rate_constant_per_s,
            "provenance_id": self.provenance_id,
        }


@dataclass(frozen=True)
class PhaseChangeBoundarySpec:
    mode: PhaseChangeMode
    saturation_temperature_K: float
    latent_heat_J_mol: float
    available_phase_change_mol: float
    provenance_id: str

    def __post_init__(self) -> None:
        if self.mode not in {"none", "boiling", "condensation"}:
            raise ValueError("unsupported phase-change mode")
        _positive(self.saturation_temperature_K, "saturation_temperature_K")
        _positive(self.latent_heat_J_mol, "latent_heat_J_mol")
        _nonnegative(self.available_phase_change_mol, "available_phase_change_mol")
        if not self.provenance_id:
            raise ValueError("provenance_id cannot be empty")

    def to_dict(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "saturation_temperature_K": self.saturation_temperature_K,
            "latent_heat_J_mol": self.latent_heat_J_mol,
            "available_phase_change_mol": self.available_phase_change_mol,
            "provenance_id": self.provenance_id,
        }


@dataclass(frozen=True)
class EquipmentHeatTransferResult:
    model_id: str
    equipment_id: str
    surface_type: HeatTransferSurface
    initial_temperature_K: float
    final_temperature_K: float
    utility_temperature_K: float
    duration_s: float
    clean_overall_u_W_m2_K: float
    effective_overall_u_W_m2_K: float
    fouling_resistance_m2_K_W: float
    surface_correction_factor: float
    conductance_W_K: float
    average_heat_transfer_W: float
    heat_energy_J: float
    sensible_energy_J: float
    latent_energy_J: float
    phase_change_mode: PhaseChangeMode
    phase_changed_mol: float
    phase_change_fraction: float
    energy_balance_residual_J: float
    warnings: tuple[str, ...]
    provenance: dict[str, str]

    def to_dict(self) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "equipment_id": self.equipment_id,
            "surface_type": self.surface_type,
            "initial_temperature_K": self.initial_temperature_K,
            "final_temperature_K": self.final_temperature_K,
            "utility_temperature_K": self.utility_temperature_K,
            "duration_s": self.duration_s,
            "clean_overall_u_W_m2_K": self.clean_overall_u_W_m2_K,
            "effective_overall_u_W_m2_K": self.effective_overall_u_W_m2_K,
            "fouling_resistance_m2_K_W": self.fouling_resistance_m2_K_W,
            "surface_correction_factor": self.surface_correction_factor,
            "conductance_W_K": self.conductance_W_K,
            "average_heat_transfer_W": self.average_heat_transfer_W,
            "heat_energy_J": self.heat_energy_J,
            "sensible_energy_J": self.sensible_energy_J,
            "latent_energy_J": self.latent_energy_J,
            "phase_change_mode": self.phase_change_mode,
            "phase_changed_mol": self.phase_changed_mol,
            "phase_change_fraction": self.phase_change_fraction,
            "energy_balance_residual_J": self.energy_balance_residual_J,
            "warnings": list(self.warnings),
            "provenance": dict(self.provenance),
        }


def equipment_heat_transfer(
    equipment: HeatTransferEquipmentSpec,
    *,
    process_temperature_K: float,
    utility_temperature_K: float,
    process_heat_capacity_J_K: float | None = None,
    duration_s: float,
    elapsed_fouling_time_s: float = 0.0,
    fouling: FoulingEvolutionSpec | None = None,
    phase_change: PhaseChangeBoundarySpec | None = None,
    caloric_model: PropertyPackageThermodynamics | ConstantCvThermodynamics | None = None,
    caloric_state=None,
) -> EquipmentHeatTransferResult:
    """Integrate lumped equipment heat transfer with optional phase plateau."""

    _positive(process_temperature_K, "process_temperature_K")
    _positive(utility_temperature_K, "utility_temperature_K")
    if caloric_model is None:
        if caloric_state is not None or process_heat_capacity_J_K is None:
            raise ValueError("select a declared capacity or a caloric model/state pair")
        _positive(process_heat_capacity_J_K, "process_heat_capacity_J_K")
    elif caloric_state is None or process_heat_capacity_J_K is not None:
        raise ValueError("caloric equipment requires one model/state pair and one energy basis")
    _positive(duration_s, "duration_s")
    _nonnegative(elapsed_fouling_time_s, "elapsed_fouling_time_s")
    fouling_resistance = 0.0 if fouling is None else fouling.resistance_at(elapsed_fouling_time_s)
    effective_u = 1.0 / (1.0 / equipment.clean_overall_u_W_m2_K + fouling_resistance)
    conductance = effective_u * equipment.area_m2 * equipment.surface_correction_factor
    boundary = phase_change or PhaseChangeBoundarySpec(
        mode="none",
        saturation_temperature_K=process_temperature_K,
        latent_heat_J_mol=1.0,
        available_phase_change_mol=0.0,
        provenance_id="no-phase-change-boundary",
    )
    warnings: list[str] = []
    caloric_result = None
    if caloric_model is not None:
        from chemworld.physchem.coupled_system import CoupledState, CoupledSystem, HeatFlux

        if (
            not isinstance(caloric_state, CoupledState)
            or caloric_state.temperature_K != process_temperature_K
        ):
            raise ValueError("equipment temperature must match its owned caloric inventory")
        if boundary.mode != "none":
            raise ValueError("variable-caloric phase change needs an explicit flash/latent model")
        system = CoupledSystem(
            caloric_model,
            heat_flux=lambda state: HeatFlux(
                heater_W=conductance * (utility_temperature_K - state.temperature_K)
            ),
        )
        caloric_result = system.advance(caloric_state, duration_s)
        final_temperature = caloric_result.final_state.temperature_K
        sensible = caloric_model.internal_energy_j(
            caloric_result.final_state
        ) - caloric_model.internal_energy_j(caloric_state)
        latent, phase_changed = 0.0, 0.0
        warnings.extend(caloric_result.solver_diagnostic.get("property_trial_warnings", ()))
    elif boundary.mode == "boiling":
        final_temperature, sensible, latent, phase_changed = _boiling_path(
            initial_temperature_K=process_temperature_K,
            utility_temperature_K=utility_temperature_K,
            heat_capacity_J_K=process_heat_capacity_J_K,
            conductance_W_K=conductance,
            duration_s=duration_s,
            boundary=boundary,
            warnings=warnings,
        )
    elif boundary.mode == "condensation":
        final_temperature, sensible, latent, phase_changed = _condensation_path(
            initial_temperature_K=process_temperature_K,
            utility_temperature_K=utility_temperature_K,
            heat_capacity_J_K=process_heat_capacity_J_K,
            conductance_W_K=conductance,
            duration_s=duration_s,
            boundary=boundary,
            warnings=warnings,
        )
    else:
        final_temperature = _lumped_sensible_temperature(
            initial_temperature_K=process_temperature_K,
            utility_temperature_K=utility_temperature_K,
            heat_capacity_J_K=process_heat_capacity_J_K,
            conductance_W_K=conductance,
            duration_s=duration_s,
        )
        sensible = process_heat_capacity_J_K * (final_temperature - process_temperature_K)
        latent = 0.0
        phase_changed = 0.0
        saturation = boundary.saturation_temperature_K
        if (
            process_temperature_K < saturation < final_temperature
            and utility_temperature_K > saturation
        ):
            warnings.append("boiling_possible_but_phase_change_not_enabled")
        if (
            final_temperature < saturation < process_temperature_K
            and utility_temperature_K < saturation
        ):
            warnings.append("condensation_possible_but_phase_change_not_enabled")
    heat_energy = (
        sensible + latent
        if caloric_result is None
        else caloric_result.final_state.boundary_heat_J - caloric_state.boundary_heat_J
    )
    phase_fraction = (
        0.0
        if boundary.available_phase_change_mol <= 0.0
        else phase_changed / boundary.available_phase_change_mol
    )
    return EquipmentHeatTransferResult(
        model_id=(
            "equipment_phase_change_heat_transfer_v1"
            if caloric_result is None
            else "equipment_caloric_heat_transfer_v1"
        ),
        equipment_id=equipment.equipment_id,
        surface_type=equipment.surface_type,
        initial_temperature_K=process_temperature_K,
        final_temperature_K=final_temperature,
        utility_temperature_K=utility_temperature_K,
        duration_s=duration_s,
        clean_overall_u_W_m2_K=equipment.clean_overall_u_W_m2_K,
        effective_overall_u_W_m2_K=effective_u,
        fouling_resistance_m2_K_W=fouling_resistance,
        surface_correction_factor=equipment.surface_correction_factor,
        conductance_W_K=conductance,
        average_heat_transfer_W=heat_energy / duration_s,
        heat_energy_J=heat_energy,
        sensible_energy_J=sensible,
        latent_energy_J=latent,
        phase_change_mode=boundary.mode,
        phase_changed_mol=phase_changed,
        phase_change_fraction=phase_fraction,
        energy_balance_residual_J=heat_energy - sensible - latent,
        warnings=tuple(warnings),
        provenance={
            "equipment": equipment.provenance_id,
            "fouling": "none" if fouling is None else fouling.provenance_id,
            "phase_change": boundary.provenance_id,
        },
    )


def _boiling_path(
    *,
    initial_temperature_K: float,
    utility_temperature_K: float,
    heat_capacity_J_K: float,
    conductance_W_K: float,
    duration_s: float,
    boundary: PhaseChangeBoundarySpec,
    warnings: list[str],
) -> tuple[float, float, float, float]:
    saturation = boundary.saturation_temperature_K
    if utility_temperature_K <= saturation:
        raise ValueError("boiling requires utility_temperature_K above saturation")
    if initial_temperature_K > saturation:
        raise ValueError("boiling path requires initial temperature at or below saturation")
    time_to_saturation = _time_to_temperature(
        initial_temperature_K,
        utility_temperature_K,
        saturation,
        heat_capacity_J_K,
        conductance_W_K,
    )
    if duration_s <= time_to_saturation:
        final = _lumped_sensible_temperature(
            initial_temperature_K=initial_temperature_K,
            utility_temperature_K=utility_temperature_K,
            heat_capacity_J_K=heat_capacity_J_K,
            conductance_W_K=conductance_W_K,
            duration_s=duration_s,
        )
        return final, heat_capacity_J_K * (final - initial_temperature_K), 0.0, 0.0
    sensible_to_saturation = heat_capacity_J_K * (saturation - initial_temperature_K)
    remaining = duration_s - time_to_saturation
    plateau_rate = conductance_W_K * (utility_temperature_K - saturation)
    latent_capacity = boundary.available_phase_change_mol * boundary.latent_heat_J_mol
    latent = min(plateau_rate * remaining, latent_capacity)
    phase_changed = latent / boundary.latent_heat_J_mol
    if latent < latent_capacity:
        warnings.append("boiling_incomplete_at_end_of_contact")
        return saturation, sensible_to_saturation, latent, phase_changed
    warnings.append("boiling_inventory_exhausted")
    time_after_phase = remaining - latent_capacity / plateau_rate
    final = _lumped_sensible_temperature(
        initial_temperature_K=saturation,
        utility_temperature_K=utility_temperature_K,
        heat_capacity_J_K=heat_capacity_J_K,
        conductance_W_K=conductance_W_K,
        duration_s=time_after_phase,
    )
    sensible = sensible_to_saturation + heat_capacity_J_K * (final - saturation)
    return final, sensible, latent, phase_changed


def _condensation_path(
    *,
    initial_temperature_K: float,
    utility_temperature_K: float,
    heat_capacity_J_K: float,
    conductance_W_K: float,
    duration_s: float,
    boundary: PhaseChangeBoundarySpec,
    warnings: list[str],
) -> tuple[float, float, float, float]:
    saturation = boundary.saturation_temperature_K
    if utility_temperature_K >= saturation:
        raise ValueError("condensation requires utility_temperature_K below saturation")
    if initial_temperature_K < saturation:
        raise ValueError("condensation path requires initial temperature at or above saturation")
    time_to_saturation = _time_to_temperature(
        initial_temperature_K,
        utility_temperature_K,
        saturation,
        heat_capacity_J_K,
        conductance_W_K,
    )
    if duration_s <= time_to_saturation:
        final = _lumped_sensible_temperature(
            initial_temperature_K=initial_temperature_K,
            utility_temperature_K=utility_temperature_K,
            heat_capacity_J_K=heat_capacity_J_K,
            conductance_W_K=conductance_W_K,
            duration_s=duration_s,
        )
        return final, heat_capacity_J_K * (final - initial_temperature_K), 0.0, 0.0
    sensible_to_saturation = heat_capacity_J_K * (saturation - initial_temperature_K)
    remaining = duration_s - time_to_saturation
    plateau_rate = conductance_W_K * (saturation - utility_temperature_K)
    latent_capacity = boundary.available_phase_change_mol * boundary.latent_heat_J_mol
    latent_magnitude = min(plateau_rate * remaining, latent_capacity)
    phase_changed = latent_magnitude / boundary.latent_heat_J_mol
    latent = -latent_magnitude
    if latent_magnitude < latent_capacity:
        warnings.append("condensation_incomplete_at_end_of_contact")
        return saturation, sensible_to_saturation, latent, phase_changed
    warnings.append("condensable_inventory_exhausted")
    time_after_phase = remaining - latent_capacity / plateau_rate
    final = _lumped_sensible_temperature(
        initial_temperature_K=saturation,
        utility_temperature_K=utility_temperature_K,
        heat_capacity_J_K=heat_capacity_J_K,
        conductance_W_K=conductance_W_K,
        duration_s=time_after_phase,
    )
    sensible = sensible_to_saturation + heat_capacity_J_K * (final - saturation)
    return final, sensible, latent, phase_changed


def _lumped_sensible_temperature(
    *,
    initial_temperature_K: float,
    utility_temperature_K: float,
    heat_capacity_J_K: float,
    conductance_W_K: float,
    duration_s: float,
) -> float:
    return utility_temperature_K + (initial_temperature_K - utility_temperature_K) * exp(
        -conductance_W_K * duration_s / heat_capacity_J_K
    )


def _time_to_temperature(
    initial_temperature_K: float,
    utility_temperature_K: float,
    target_temperature_K: float,
    heat_capacity_J_K: float,
    conductance_W_K: float,
) -> float:
    if initial_temperature_K == target_temperature_K:
        return 0.0
    ratio = (target_temperature_K - utility_temperature_K) / (
        initial_temperature_K - utility_temperature_K
    )
    if not 0.0 < ratio < 1.0:
        raise ValueError("target temperature is not reachable between process and utility")
    return -heat_capacity_J_K / conductance_W_K * log(ratio)


def _positive(value: float, field_name: str) -> None:
    if value <= 0.0 or not isfinite(value):
        raise ValueError(f"{field_name} must be positive and finite")


def _nonnegative(value: float, field_name: str) -> None:
    if value < 0.0 or not isfinite(value):
        raise ValueError(f"{field_name} must be nonnegative and finite")


__all__ = [
    "EquipmentHeatTransferResult",
    "FoulingEvolutionSpec",
    "HeatTransferEquipmentSpec",
    "PhaseChangeBoundarySpec",
    "TubularHeatTransferBoundarySpec",
    "equipment_heat_transfer",
]
