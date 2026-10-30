"""Released battery data and a checked projection onto the original SPM.

This module owns immutable declarations, not inventories or an integrator.
The six released functions are finite named mathematical laws. No external
Python, callback, or reference simulator is loaded by the production reader.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from functools import cached_property, lru_cache
from hashlib import sha256
from importlib.resources import files
from math import exp, isfinite, sqrt
from numbers import Real
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Self, cast

from chemworld.data.scientific_inputs import (
    InputSource,
    InputValue,
    ScientificTable,
    _keys,
    strict_json_loads,
)
from chemworld.physchem.battery_materials import ElectrodeOCV, ElectrodeOCVEndpointPolicy
from chemworld.physchem.data_conflicts import DataSourceProvenance

if TYPE_CHECKING:
    from chemworld.physchem.battery_models import BatterySPMSpec

PARAMETER_SET_VERSION = "chemworld-battery-parameter-set-1"
BINDING_VERSION = "chemworld-battery-parameter-binding-1"
SOURCE_VERSION = "PyBaMM-25.4.2"
PROJECTION_ID = "SPM-fixed-T-v1"
DATA_FILENAME = "battery_chen2020_pybamm_25_4_2.json"
CATALOG_REFERENCE = "chemworld.data/parameters/" + DATA_FILENAME
_FUNCTIONS = {
    "graphite_LGM50_ocp_Chen2020": ("polynomial-exp-tanh", ("sto",)),
    "nmc_LGM50_ocp_Chen2020": ("polynomial-exp-tanh", ("sto",)),
    "graphite_LGM50_electrolyte_exchange_current_density_Chen2020": (
        "symmetric-arrhenius-bv",
        ("c_e", "c_s_surf", "c_s_max", "T"),
    ),
    "nmc_LGM50_electrolyte_exchange_current_density_Chen2020": (
        "symmetric-arrhenius-bv",
        ("c_e", "c_s_surf", "c_s_max", "T"),
    ),
    "electrolyte_diffusivity_Nyman2008": ("nyman-electrolyte-diffusivity", ("c_e", "T")),
    "electrolyte_conductivity_Nyman2008": ("nyman-electrolyte-conductivity", ("c_e", "T")),
}


def _json(value: object) -> str:
    import json

    return json.dumps(
        value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
    )


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value):
        raise ValueError(name + " requires a finite real number")
    return float(value)


@lru_cache(maxsize=1)
def _released_json() -> str:
    text = files("chemworld.data").joinpath("parameters", DATA_FILENAME).read_text("utf-8")
    payload = strict_json_loads(text)
    _keys(
        payload,
        (
            "schema_version",
            "name",
            "version",
            "material_id",
            "sources",
            "parameters",
            "functions",
            "metadata",
            "fields",
            "notes",
            "license_notice",
        ),
    )
    if (
        payload["schema_version"] != PARAMETER_SET_VERSION
        or payload["name"] != "Chen2020"
        or payload["version"] != SOURCE_VERSION
    ):
        raise ValueError("unsupported released battery parameter catalog")
    return _json(payload)


def pybamm_endpoint_source() -> InputSource:
    """The numerical endpoint convention is independently attributed."""
    return InputSource(
        DataSourceProvenance(
            "pybamm-25.4.2-lithium-ion-endpoints",
            0,
            "PyBaMM 25.4.2 parameters/lithium_ion_parameters.py::ParticleLithiumIonParameters.U "
            "and settings.py: OCP barrier 1e-6 V, OCP clip 1e-10, j0 stoichiometry clip 1e-8",
            SOURCE_VERSION,
            "BSD-3-Clause",
        ),
        "literature",
    )


@dataclass(frozen=True)
class BatteryParameterField:
    source_field: str
    kind: str
    usage: str
    scope: str

    def __post_init__(self) -> None:
        if (
            self.kind not in {"number", "function", "metadata"}
            or self.usage not in {"consumed", "retained-not-consumed"}
            or not self.source_field
            or not self.scope
        ):
            raise ValueError("invalid battery source-field classification")

    def to_dict(self) -> dict[str, str]:
        return {k: getattr(self, k) for k in ("source_field", "kind", "usage", "scope")}


@dataclass(frozen=True)
class BatteryParameterFunction:
    source_field: str
    function_id: str
    kind: str
    arguments: tuple[str, ...]
    parameters: ScientificTable
    validity_note: str

    def __post_init__(self) -> None:
        if _FUNCTIONS.get(self.function_id) != (self.kind, tuple(self.arguments)):
            raise ValueError("unknown battery source function or argument declaration")
        object.__setattr__(self, "arguments", tuple(self.arguments))
        if (
            not isinstance(self.parameters, ScientificTable)
            or self.parameters.kind != "parameters"
            or len(self.parameters.rows) != 1
            or not self.source_field
            or not self.validity_note
        ):
            raise ValueError("battery function requires one sourced parameter row")

    @property
    def inputs(self) -> Mapping[str, InputValue]:
        return self.parameters.rows[0].values

    def value(self, name: str, unit: str) -> float:
        # ScientificTable requires valid, nonmissing cells in a parameter table.
        return cast(float, self.inputs[name].to(unit).value)

    def as_ocv(self, *, endpoint_policy: ElectrodeOCVEndpointPolicy | None = None) -> ElectrodeOCV:
        if self.kind != "polynomial-exp-tanh":
            raise ValueError("this battery source function is not an OCV")
        cells = self.inputs
        polynomial = tuple(k for k in cells if k.startswith("p") and k[1:].isdigit())

        def terms(prefix: str) -> tuple[tuple[str, str, str], ...]:
            count = sum(k.startswith(prefix) and k.endswith("_amplitude") for k in cells)
            return tuple(
                (f"{prefix}{i}_amplitude", f"{prefix}{i}_scale", f"{prefix}{i}_shift")
                for i in range(count)
            )

        options = {} if endpoint_policy is None else {"endpoint_policy": endpoint_policy}
        electrode = "graphite" if self.function_id.startswith("graphite") else "NMC"
        curve = ElectrodeOCV.from_parameter_row(
            self.parameters,
            self.parameters.rows[0].row_id,
            material_id="Chen2020-LGM50-" + electrode,
            polynomial=polynomial,
            exponential_terms=terms("exp"),
            tanh_terms=terms("tanh"),
            stoichiometry_domain=(0.0, 1.0),
            supported_models=("SPM",),
            **options,
        )
        # Source references are a mapping in JSON. Canonicalize this new
        # source object's tuple so sorted JSON retains exact dataclass equality.
        return replace(curve, parameter_inputs=tuple(sorted(curve.parameter_inputs)))

    def evaluate(self, **arguments: float) -> float:
        """Evaluate the declared source mathematics, without a validity claim."""
        if set(arguments) != set(self.arguments):
            raise ValueError("battery source function requires its exact declared arguments")
        values = {k: _number(v, k) for k, v in arguments.items()}
        if self.kind == "polynomial-exp-tanh":
            return self.as_ocv().value(values["sto"])
        if values["T"] <= 0 or values["c_e"] < 0:
            raise ValueError("source mathematics requires T > 0 and c_e >= 0")
        if self.kind == "symmetric-arrhenius-bv":
            from chemworld.physchem.electrochemistry import R_J_PER_MOL_K

            cs, maximum = values["c_s_surf"], values["c_s_max"]
            if maximum <= 0 or not 0 <= cs <= maximum:
                raise ValueError("source exchange-current concentration is outside [0, cmax]")
            arrhenius = exp(
                self.value("activation_energy", "J/mol")
                / R_J_PER_MOL_K
                * (1 / self.value("reference_temperature", "K") - 1 / values["T"])
            )
            result = (
                self.value("m_ref", "A*m^(5/2)/mol^(3/2)")
                * arrhenius
                * values["c_e"] ** 0.5
                * cs**0.5
                * (maximum - cs) ** 0.5
            )
        else:
            z = values["c_e"] / self.value("concentration_reference", "mol/m^3")
            if self.kind == "nyman-electrolyte-diffusivity":
                result = (
                    self.value("quadratic", "m^2/s") * z**2
                    + self.value("linear", "m^2/s") * z
                    + self.value("constant", "m^2/s")
                )
            else:
                result = (
                    self.value("cubic", "S/m") * z**3
                    + self.value("three_halves", "S/m") * z**1.5
                    + self.value("linear", "S/m") * z
                )
        if not isfinite(result):
            raise ValueError("nonfinite battery source function result")
        return result

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_field": self.source_field,
            "function_id": self.function_id,
            "kind": self.kind,
            "arguments": list(self.arguments),
            "parameters": self.parameters.to_dict(),
            "validity_note": self.validity_note,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Self:
        _keys(
            payload,
            ("source_field", "function_id", "kind", "arguments", "parameters", "validity_note"),
        )
        return cls(
            **{
                **payload,
                "arguments": tuple(payload["arguments"]),
                "parameters": ScientificTable.from_dict(payload["parameters"]),
            }
        )


@dataclass(frozen=True)
class BatteryParameterSet:
    """The complete, pinned released catalog; all returned containers are immutable."""

    _payload_json: str

    def __post_init__(self) -> None:
        canonical = _json(strict_json_loads(self._payload_json))
        if canonical != _released_json():
            raise ValueError("unknown or modified released battery parameter set")
        object.__setattr__(self, "_payload_json", canonical)
        fields = self.fields
        names = {item.source_field for item in fields}
        if (
            len(fields) != 94
            or len(names) != 94
            or len(self.parameters.columns) != 86
            or len(self.functions) != 6
            or len(self.parameters.rows) != 1
            or {f.function_id for f in self.functions} != set(_FUNCTIONS)
            or set(self.metadata) != {"chemistry", "citations"}
            or names
            != {*self.scalar_inputs, *(f.source_field for f in self.functions), *self.metadata}
        ):
            raise ValueError("incomplete released battery parameter catalog")

    def to_dict(self) -> dict[str, Any]:
        return strict_json_loads(self._payload_json)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Self:
        return cls(_json(payload))

    @property
    def name(self) -> str:
        return "Chen2020"

    @property
    def version(self) -> str:
        return SOURCE_VERSION

    @property
    def catalog_reference(self) -> str:
        return CATALOG_REFERENCE

    @property
    def catalog_sha256(self) -> str:
        return sha256(self._payload_json.encode("utf-8")).hexdigest()

    @cached_property
    def parameters(self) -> ScientificTable:
        return ScientificTable.from_dict(self.to_dict()["parameters"])

    @property
    def scalar_inputs(self) -> Mapping[str, InputValue]:
        return self.parameters.rows[0].values

    @cached_property
    def functions(self) -> tuple[BatteryParameterFunction, ...]:
        return tuple(BatteryParameterFunction.from_dict(p) for p in self.to_dict()["functions"])

    @cached_property
    def fields(self) -> tuple[BatteryParameterField, ...]:
        return tuple(BatteryParameterField(**p) for p in self.to_dict()["fields"])

    @property
    def metadata(self) -> Mapping[str, str | tuple[str, ...]]:
        raw = self.to_dict()["metadata"]
        return MappingProxyType(
            {"chemistry": raw["chemistry"], "citations": tuple(raw["citations"])}
        )

    @property
    def sources(self) -> tuple[InputSource, ...]:
        return tuple(InputSource.from_dict(s) for s in self.to_dict()["sources"])

    def scalar(self, source_field: str, unit: str) -> float:
        if source_field not in self.scalar_inputs:
            raise ValueError("unknown battery source scalar: " + source_field)
        return cast(float, self.scalar_inputs[source_field].to(unit).value)

    def function(self, source_field: str) -> BatteryParameterFunction:
        matches = [f for f in self.functions if f.source_field == source_field]
        if len(matches) != 1:
            raise ValueError("unknown battery source function: " + source_field)
        return matches[0]

    @property
    def nominal_capacity_Ah(self) -> float:  # noqa: N802 - explicit physical unit in public API
        return self.scalar("Nominal cell capacity [A.h]", "Ah")

    @property
    def source_nominal_capacity_Ah(self) -> float:  # noqa: N802 - explicit physical unit in public API
        return self.nominal_capacity_Ah

    def current_at_nominal_c_rate(self, rate: float) -> float:
        return _number(rate, "nominal C-rate") * self.nominal_capacity_Ah

    def electrode_curve(self, electrode: str) -> ElectrodeOCV:
        if electrode not in {"negative", "positive"}:
            raise ValueError("unknown source electrode")
        return self.function(electrode.capitalize() + " electrode OCP [V]").as_ocv(
            endpoint_policy=ElectrodeOCVEndpointPolicy(source=pybamm_endpoint_source())
        )

    def build_spm(
        self,
        *,
        maximum_current_A: float,
        radial_cells: int = 64,
        surface_margin: float = 0.01,
        numerical_form: str = "residual",
        temperature_K: float = 298.15,
        model: str = "SPM",
        electrolyte_concentration_mol_m3: float = 1000.0,
    ) -> BatterySPMSpec:
        """Project source physics; grid, stopping margin and Imax are explicit policy."""
        from chemworld.physchem.battery_models import BatterySPMSpec, ParticleSpec

        if (
            model != "SPM"
            or _number(temperature_K, "temperature") != 298.15
            or _number(electrolyte_concentration_mol_m3, "fixed electrolyte") != 1000.0
        ):
            raise ValueError("Chen2020 projection requires SPM at fixed 298.15 K and ce=1000")
        if type(radial_cells) is not int or radial_cells < 2:
            raise ValueError("source SPM radial_cells must be an integer >= 2")
        if _number(maximum_current_A, "maximum current") <= 0:
            raise ValueError("source SPM equipment maximum current must be positive")
        if _number(surface_margin, "surface margin") <= 1e-8:
            raise ValueError("source stopping surfaces must be inside the kinetics clip")
        # Each derivation validates a nonmissing numeric value at construction.
        projection = {d.target: cast(float, d.value.value) for d in self._derivations()}
        particles: dict[str, ParticleSpec] = {}
        for electrode in ("negative", "positive"):
            particles[electrode] = ParticleSpec(
                cells=radial_cells,
                radius_m=projection[f"{electrode}.radius_m"],
                diffusivity_m2_s=projection[f"{electrode}.diffusivity_m2_s"],
                maximum_concentration_mol_m3=projection[f"{electrode}.maximum_concentration_mol_m3"],
                active_volume_m3=projection[f"{electrode}.active_volume_m3"],
                exchange_current_scale_A_m2=projection[f"{electrode}.exchange_current_scale_A_m2"],
            )
        return BatterySPMSpec(
            negative=particles["negative"],
            positive=particles["positive"],
            initial_negative_fraction=projection["initial_negative_fraction"],
            initial_positive_fraction=projection["initial_positive_fraction"],
            temperature_K=projection["temperature_K"],
            minimum_voltage_V=projection["minimum_voltage_V"],
            maximum_voltage_V=projection["maximum_voltage_V"],
            series_resistance_ohm=projection["series_resistance_ohm"],
            cell_envelope_volume_m3=projection["cell_envelope_volume_m3"],
            kinetics_stoichiometry_clip=projection["kinetics_stoichiometry_clip"],
            negative_ocv=self.electrode_curve("negative"),
            positive_ocv=self.electrode_curve("positive"),
            maximum_current_A=maximum_current_A,
            surface_margin=surface_margin,
            numerical_form=numerical_form,
            provenance="Chen2020/PyBaMM-25.4.2:SPM-fixed-T-v1",
            parameter_binding=BatteryParameterBinding.from_parameter_set(self),
        )

    def _derivations(self) -> tuple[BatteryParameterDerivation, ...]:
        result: list[BatteryParameterDerivation] = []

        def add(
            target: str, unit: str, names: Sequence[str], formula: str, value: float,
            extra: Sequence[tuple[str, InputValue]] = (),
        ) -> None:
            inputs = tuple((name, self.scalar_inputs[name]) for name in names) + tuple(extra)
            result.append(
                BatteryParameterDerivation(
                    target,
                    tuple(names),
                    formula,
                    InputValue(
                        value, unit, None, "Derived from released inputs; uncertainty not supplied"
                    ),
                    inputs,
                )
            )

        for electrode in ("negative", "positive"):
            title = electrode.capitalize()
            for field, source, unit in (
                ("radius_m", f"{title} particle radius [m]", "m"),
                ("diffusivity_m2_s", f"{title} particle diffusivity [m2.s-1]", "m^2/s"),
                (
                    "maximum_concentration_mol_m3",
                    f"Maximum concentration in {electrode} electrode [mol.m-3]",
                    "mol/m^3",
                ),
            ):
                add(f"{electrode}.{field}", unit, (source,), "identity", self.scalar(source, unit))
            geometry = (
                "Electrode height [m]",
                "Electrode width [m]",
                f"{title} electrode thickness [m]",
                f"{title} electrode active material volume fraction",
            )
            active = (
                self.scalar(geometry[0], "m")
                * self.scalar(geometry[1], "m")
                * self.scalar(geometry[2], "m")
                * self.scalar(geometry[3], "dimensionless")
            )
            add(
                f"{electrode}.active_volume_m3",
                "m^3",
                geometry,
                "height*width*thickness*epsilon_s",
                active,
            )
            cmax = f"Maximum concentration in {electrode} electrode [mol.m-3]"
            c0 = f"Initial concentration in {electrode} electrode [mol.m-3]"
            add(
                f"initial_{electrode}_fraction",
                "dimensionless",
                (c0, cmax),
                "c0/cmax",
                self.scalar(c0, "mol/m^3") / self.scalar(cmax, "mol/m^3"),
            )
            law = self.function(f"{title} electrode exchange-current density [A.m-2]")
            ce = "Initial concentration in electrolyte [mol.m-3]"
            tref, temperature = "Reference temperature [K]", "Initial temperature [K]"
            scale = (
                law.value("m_ref", "A*m^(5/2)/mol^(3/2)")
                * sqrt(self.scalar(ce, "mol/m^3"))
                * self.scalar(cmax, "mol/m^3")
            )
            add(
                f"{electrode}.exchange_current_scale_A_m2",
                "A/m^2",
                (ce, cmax, tref, temperature),
                "m_ref*sqrt(ce_fixed)*cmax; T=Tref=298.15 K, Arrhenius factor=1",
                scale,
                tuple((law.source_field + "::" + k, v) for k, v in law.inputs.items()),
            )
        for target, source, unit in (
            ("temperature_K", "Initial temperature [K]", "K"),
            ("minimum_voltage_V", "Lower voltage cut-off [V]", "V"),
            ("maximum_voltage_V", "Upper voltage cut-off [V]", "V"),
            ("series_resistance_ohm", "Contact resistance [Ohm]", "Ohm"),
            ("cell_envelope_volume_m3", "Cell volume [m3]", "m^3"),
        ):
            add(target, unit, (source,), "identity", self.scalar(source, unit))
        add(
            "kinetics_stoichiometry_clip",
            "dimensionless",
            (),
            "PyBaMM25.4.2/settings.py::tolerances[j0__c_s]",
            1e-8,
        )
        return tuple(result)


@lru_cache(maxsize=1)
def battery_parameter_set(
    name: str = "Chen2020", *, version: str = SOURCE_VERSION,
) -> BatteryParameterSet:
    if name != "Chen2020" or version != SOURCE_VERSION:
        raise ValueError("unknown battery parameter set or source version")
    return BatteryParameterSet(_released_json())


@dataclass(frozen=True)
class BatteryParameterDerivation:
    target: str
    source_fields: tuple[str, ...]
    formula: str
    value: InputValue
    inputs: tuple[tuple[str, InputValue], ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_fields", tuple(self.source_fields))
        object.__setattr__(self, "inputs", tuple(sorted(tuple(v) for v in self.inputs)))
        if (
            not self.target
            or not self.formula
            or not isinstance(self.value, InputValue)
            or self.value.quality != "valid"
            or self.value.value is None
            or len(dict(self.inputs)) != len(self.inputs)
            or any(not isinstance(v, InputValue) or v.quality != "valid" for _, v in self.inputs)
        ):
            raise ValueError("invalid sourced battery parameter derivation")

    def to_dict(self) -> dict[str, Any]:
        return {
            "target": self.target,
            "source_fields": list(self.source_fields),
            "formula": self.formula,
            "value": self.value.to_dict(),
            "inputs": {k: v.to_dict() for k, v in self.inputs},
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Self:
        _keys(payload, ("target", "source_fields", "formula", "value", "inputs"))
        return cls(
            payload["target"],
            tuple(payload["source_fields"]),
            payload["formula"],
            InputValue.from_dict(payload["value"]),
            tuple((k, InputValue.from_dict(v)) for k, v in payload["inputs"].items()),
        )


@dataclass(frozen=True)
class BatteryParameterBinding:
    parameter_set_id: str
    source_version: str
    catalog_reference: str
    catalog_sha256: str
    projection_id: str
    derivations: tuple[BatteryParameterDerivation, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "derivations", tuple(self.derivations))
        source = battery_parameter_set(self.parameter_set_id, version=self.source_version)
        if (
            self.catalog_reference != source.catalog_reference
            or self.catalog_sha256 != source.catalog_sha256
            or self.projection_id != PROJECTION_ID
            or self.derivations != source._derivations()
        ):
            raise ValueError(
                "battery parameter binding differs from its released source projection"
            )

    @classmethod
    def from_parameter_set(cls, source: BatteryParameterSet) -> Self:
        if not isinstance(source, BatteryParameterSet):
            raise ValueError("binding requires a released BatteryParameterSet")
        return cls(
            source.name,
            source.version,
            source.catalog_reference,
            source.catalog_sha256,
            PROJECTION_ID,
            source._derivations(),
        )

    @property
    def source_nominal_capacity_Ah(self) -> float:  # noqa: N802 - explicit physical unit in public API
        return battery_parameter_set(self.parameter_set_id).nominal_capacity_Ah

    def current_at_nominal_c_rate(self, rate: float) -> float:
        return _number(rate, "nominal C-rate") * self.source_nominal_capacity_Ah

    def validate_projection(self, spec: object) -> bool:
        from chemworld.physchem.battery_models import BatterySPMSpec

        if type(spec) is not BatterySPMSpec:
            raise ValueError("source binding supports only the actual single-material SPM")
        if any(
            getattr(spec, k) is not None
            for k in ("thermal", "sei", "electrolyte", "plating", "negative_ocp", "positive_ocp")
        ):
            raise ValueError(
                "source SPM binding rejects thermal, side, electrolyte or directional options"
            )
        if spec.surface_margin <= 1e-8:
            raise ValueError("source stopping surfaces must be inside the kinetics clip")
        if _number(spec.maximum_current_A, "maximum current") <= 0:
            raise ValueError("source SPM equipment maximum current must be positive")
        for item in self.derivations:
            value: object = spec
            for part in item.target.split("."):
                value = getattr(value, part)
            if _number(value, item.target) != item.value.value:
                raise ValueError("source battery projection mismatch: " + item.target)
        source = battery_parameter_set(self.parameter_set_id, version=self.source_version)
        for electrode in ("negative", "positive"):
            curve = getattr(spec, electrode + "_ocv")
            if (
                not isinstance(curve, ElectrodeOCV)
                or curve.to_dict() != source.electrode_curve(electrode).to_dict()
            ):
                raise ValueError("source battery projection mismatch: " + electrode + "_ocv")
        for title in ("Negative", "Positive"):
            if (
                source.scalar(title + " electrode charge transfer coefficient", "dimensionless")
                != 0.5
            ):
                raise ValueError("source SPM requires symmetric Butler-Volmer kinetics")
            porosity = source.scalar(title + " electrode porosity", "dimensionless")
            active = source.scalar(
                title + " electrode active material volume fraction", "dimensionless"
            )
            if porosity + active > 1:
                raise ValueError("source electrode volumes are incompatible")
        return True

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": BINDING_VERSION,
            "parameter_set_id": self.parameter_set_id,
            "source_version": self.source_version,
            "catalog_reference": self.catalog_reference,
            "catalog_sha256": self.catalog_sha256,
            "projection_id": self.projection_id,
            "derivations": [d.to_dict() for d in self.derivations],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Self:
        _keys(
            payload,
            (
                "schema_version",
                "parameter_set_id",
                "source_version",
                "catalog_reference",
                "catalog_sha256",
                "projection_id",
                "derivations",
            ),
        )
        if payload["schema_version"] != BINDING_VERSION:
            raise ValueError("unsupported battery parameter binding version")
        return cls(
            **{k: v for k, v in payload.items() if k not in {"schema_version", "derivations"}},
            derivations=tuple(
                BatteryParameterDerivation.from_dict(d) for d in payload["derivations"]
            ),
        )
