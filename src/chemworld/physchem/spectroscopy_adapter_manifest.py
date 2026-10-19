"""WF-20 provider and adapter proposal for spectral identifiability audits."""

from __future__ import annotations

from collections.abc import Mapping
from math import isfinite
from typing import Any

from chemworld.physchem.maturity import (
    MaturityLevel,
    ModelAdapterManifest,
    ModelCard,
    ModelExecutionRole,
    ModelProviderContract,
    ValidationEvidence,
)
from chemworld.physchem.spectroscopy import SpectralMeasurement
from chemworld.physchem.spectroscopy_identifiability import (
    CHEMICALS_COMMIT,
    RMG_PY_COMMIT,
    SpectralIdentifiabilitySpec,
    evaluate_spectral_identifiability,
)
from chemworld.runtime.kernel_contracts import ModelProviderResult
from chemworld.world.instruments import (
    INSTRUMENT_RUNTIME_MODEL_ID,
    INSTRUMENT_RUNTIME_PROVENANCE,
    INSTRUMENT_RUNTIME_PROVIDER_PATH,
    instrument_contracts,
)
from chemworld.world.observation_kernel import raw_signal

OWNED_PATHS = (
    "src/chemworld/physchem/spectroscopy_identifiability.py",
    "src/chemworld/physchem/spectroscopy_adapter_manifest.py",
    "tests/test_spectroscopy_identifiability.py",
)
INTEGRATION_OPERATIONS = ("measure",)
PUBLIC_SPECIES_CHANNELS = frozenset(
    {
        "reactant_public",
        "target_public",
        "impurity_public",
        "degradation_public",
    }
)
FORBIDDEN_PACKET_KEYS = frozenset(
    {
        "debug",
        "hidden_parameters",
        "model_id",
        "model_ids",
        "private_seed",
        "provider",
        "provider_parameters",
        "provider_path",
        "world_provider",
    }
)


def instrument_runtime_provider_contract() -> ModelProviderContract:
    """Return the provider contract consumed by the formal measure runtime."""

    return ModelProviderContract(
        model_id=INSTRUMENT_RUNTIME_MODEL_ID,
        module_id="spectroscopy_instruments",
        maturity=MaturityLevel.REFERENCE_VALIDATED,
        role=ModelExecutionRole.RUNTIME,
        provider_path=INSTRUMENT_RUNTIME_PROVIDER_PATH,
        input_fields=(
            "instrument_id",
            "public_values",
            "public_species_amounts_mol",
            "sample_basis_volume_L",
            "seed",
            "replicate_count",
        ),
        output_fields=("packet", "processed_estimates", "uncertainty"),
        units={
            "instrument_id": "dimensionless id",
            "public_values": "normalized public estimates",
            "public_species_amounts_mol": "mol",
            "sample_basis_volume_L": "L",
            "seed": "dimensionless integer",
            "replicate_count": "count",
            "packet": "instrument-specific public signal",
            "processed_estimates": "instrument-specific public estimates",
            "uncertainty": "instrument-specific public uncertainty",
        },
        validity_checks=(
            "instrument id has a declared bounded synthetic contract",
            "sample volume, public values, and public aggregate amounts are finite",
            "only anonymous task-public aggregate species channels enter the provider",
            "packet exposes raw signal, peaks, anonymous assignments, processed estimates, "
            "uncertainty, calibration, and missingness",
            "packet contains no hidden identity, private seed, provider path, or model id",
        ),
        diagnostic_fields=(
            "instrument_id",
            "packet_schema_valid",
            "layered_packet",
            "identity_safe",
            "finite_signal",
            "calibration_profile",
            "replicate_count",
            "missingness_count",
            "saturation_count",
        ),
        failure_policy=(
            "fail closed before returning an observation when the instrument, numerical "
            "domain, public identity boundary, or layered packet contract is invalid"
        ),
        provenance=INSTRUMENT_RUNTIME_PROVENANCE,
        intended_operations=INTEGRATION_OPERATIONS,
    )


class ValidatedInstrumentRuntimeProvider:
    """Validated public-signal provider used by every formal measurement."""

    @property
    def model_contract(self) -> ModelProviderContract:
        return instrument_runtime_provider_contract()

    def validate_domain(self, inputs: Mapping[str, Any]) -> tuple[str, ...]:
        violations: list[str] = []
        instrument_id = inputs.get("instrument_id")
        if instrument_id not in {"hplc", "gc", "uvvis", "ph_meter", "final_assay"}:
            violations.append("instrument_id has no declared runtime contract")
        public_values = inputs.get("public_values")
        if not isinstance(public_values, Mapping):
            violations.append("public_values must be a mapping")
        elif not _finite_optional_mapping(public_values):
            violations.append("public_values must contain only finite numbers or null")
        public_amounts = inputs.get("public_species_amounts_mol")
        if public_amounts is not None:
            if not isinstance(public_amounts, Mapping):
                violations.append("public_species_amounts_mol must be a mapping or null")
            else:
                unexpected = sorted(set(map(str, public_amounts)) - PUBLIC_SPECIES_CHANNELS)
                if unexpected:
                    violations.append(f"non-public species channels: {unexpected}")
                if not _finite_nonnegative_mapping(public_amounts):
                    violations.append("public species amounts must be finite and nonnegative")
        volume = inputs.get("sample_basis_volume_L")
        if not isinstance(volume, int | float) or not isfinite(float(volume)) or volume <= 0:
            violations.append("sample_basis_volume_L must be positive and finite")
        seed = inputs.get("seed")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            violations.append("seed must be a nonnegative integer")
        replicates = inputs.get("replicate_count")
        if isinstance(replicates, bool) or not isinstance(replicates, int) or replicates < 1:
            violations.append("replicate_count must be a positive integer")
        return tuple(violations)

    def evaluate(self, inputs: Mapping[str, Any]) -> ModelProviderResult:
        violations = self.validate_domain(inputs)
        if violations:
            return _runtime_failed_result("; ".join(violations))
        instrument_id = str(inputs["instrument_id"])
        public_values = dict(inputs["public_values"])
        public_amounts_value = inputs.get("public_species_amounts_mol")
        public_amounts = None if public_amounts_value is None else dict(public_amounts_value)
        packet = raw_signal(
            instrument_id,
            public_values,
            species_amounts_mol=public_amounts,
            volume_L=float(inputs["sample_basis_volume_L"]),
            seed=int(inputs["seed"]),
            replicate_count=int(inputs["replicate_count"]),
        )
        layered = _is_layered_packet(instrument_id, packet)
        identity_safe = not (_nested_keys(packet) & FORBIDDEN_PACKET_KEYS)
        finite_signal = _all_numeric_values_finite(packet)
        if not packet or not layered or not identity_safe or not finite_signal:
            failures = []
            if not packet:
                failures.append("empty signal packet")
            if not layered:
                failures.append("layered public packet contract failed")
            if not identity_safe:
                failures.append("private identity field detected")
            if not finite_signal:
                failures.append("non-finite public signal detected")
            return _runtime_failed_result("; ".join(failures))
        contract = instrument_contracts()[instrument_id]
        processed = packet.get("processed_estimates", {})
        uncertainty = packet.get("uncertainty", {})
        diagnostics = {
            "instrument_id": instrument_id,
            "packet_schema_valid": True,
            "layered_packet": layered,
            "identity_safe": identity_safe,
            "finite_signal": finite_signal,
            "calibration_profile": contract.calibration_profile,
            "replicate_count": int(inputs["replicate_count"]),
            "missingness_count": _missingness_count(packet),
            "saturation_count": _saturation_count(packet),
        }
        return ModelProviderResult(
            outputs={
                "packet": packet,
                "processed_estimates": dict(processed) if isinstance(processed, Mapping) else {},
                "uncertainty": dict(uncertainty) if isinstance(uncertainty, Mapping) else {},
            },
            diagnostics=diagnostics,
            provenance=self.model_contract.provenance,
        )


def instrument_runtime_adapter_manifest() -> ModelAdapterManifest:
    """Return the formal runtime adapter manifest for public instruments."""

    return ModelAdapterManifest(
        adapter_id="foundation-instrument-runtime-integration",
        adapter_version="1.0",
        owner_workstream="foundation-instrument-runtime-integration",
        provider_contract=instrument_runtime_provider_contract(),
        owned_paths=(
            "src/chemworld/runtime/observation_services.py",
            "src/chemworld/runtime/instrument_cost_services.py",
            "src/chemworld/world/instruments.py",
            "src/chemworld/world/spectra.py",
            "src/chemworld/world/observation_kernel.py",
            "src/chemworld/physchem/spectroscopy_adapter_manifest.py",
            "tests/test_instrument_runtime_integration.py",
        ),
        integration_operations=INTEGRATION_OPERATIONS,
        target_world_law="chemworld-physical-chemistry-vnext",
        status="integrated",
    )


def instrument_runtime_model_card() -> ModelCard:
    """Return the evidence-bound card for the formal public instrument runtime."""

    return ModelCard(
        model_id=INSTRUMENT_RUNTIME_MODEL_ID,
        module_id="spectroscopy_instruments",
        title="Validated Synthetic Instrument Runtime",
        maturity=MaturityLevel.REFERENCE_VALIDATED,
        summary=(
            "A bounded synthetic measurement runtime that dispatches UV-vis, HPLC, "
            "GC, pH, and final-assay observations through explicit public packet, "
            "calibration, uncertainty, sample, and transaction contracts."
        ),
        equations=(
            "public packet = calibrated_signal(public aggregate sample, instrument, seed)",
            "sample_out = sample_in - exact_declared_consumption",
            "measurement failure => atomic rollback(cost, sample, signal, history)",
        ),
        assumptions=(
            "only anonymous task-public aggregate species channels enter the provider",
            "each instrument uses its declared bounded synthetic reference model",
            "replicate noise is deterministic under the declared measurement seed",
        ),
        validity_limits=(
            "virtual liquid samples and declared instrument contracts only",
            "not a predictor of real samples or a physical-device emulator",
            "identity-bearing hidden state and provider metadata never enter public packets",
        ),
        failure_modes=(
            "unknown instruments, nonfinite inputs, or private species channels fail closed",
            "insufficient sample volume fails before charging or consuming sample",
            "invalid packets and provider exceptions atomically roll back the measurement",
        ),
        units={
            "sample volume": "L",
            "species amount": "mol",
            "axis and signal": "instrument contract specific",
            "uncertainty": "instrument contract specific",
        },
        reference_reading=INSTRUMENT_RUNTIME_PROVENANCE,
        validation_evidence=(
            ValidationEvidence(
                evidence_id="validated-instrument-runtime-dynamic-dispatch",
                evidence_type="runtime_integration_test",
                description=(
                    "All five formal instruments dynamically execute the provider, record "
                    "contract-bound provenance, and preserve layered anonymous packets."
                ),
                status="implemented",
                command_or_path="tests/test_instrument_runtime_integration.py",
                tolerance="exact seeded replay and strict public packet schema",
            ),
            ValidationEvidence(
                evidence_id="validated-instrument-runtime-reference-closures",
                evidence_type="analytical_and_integration_test",
                description=(
                    "Beer-Lambert, retention/plate, potentiometric, calibration, "
                    "identifiability, sample consumption, and failure rollback checks."
                ),
                status="implemented",
                command_or_path=(
                    "tests/test_instruments_reference.py; "
                    "tests/test_instrument_runtime_integration.py"
                ),
                tolerance="declared analytical tolerances and atomic state equality",
            ),
        ),
        model_limit_notes=(
            "Current measurements are instantaneous virtual snapshots: sampling and return "
            "share world time; no queue or concurrent chemical advance is simulated.",
            "Sampling debits the selected inventory (or representative crystallizer slurry); "
            "unselected receivers are unchanged. Clock time and chromatogram axes differ.",
            "Reference-validated applies to the synthetic observation contract only.",
            "No real compound identity or empirical instrument accuracy is claimed.",
        ),
        intended_use=(
            "formal ChemWorld measure operations",
            "agent reasoning over bounded spectra and calibrated estimates",
            "deterministic replay and public/private boundary testing",
        ),
    )


def spectroscopy_identifiability_provider_contract() -> ModelProviderContract:
    """Return the WF-00-compatible diagnostic provider contract."""

    return ModelProviderContract(
        model_id="chemworld_spectral_identifiability_audit_vnext",
        module_id="spectroscopy_instruments",
        maturity=MaturityLevel.REFERENCE_VALIDATED,
        role=ModelExecutionRole.REFERENCE,
        provider_path=(
            "chemworld.physchem.spectroscopy_adapter_manifest.SpectralIdentifiabilityProvider"
        ),
        input_fields=("reference", "alternative", "audit_spec"),
        output_fields=("report",),
        units={
            "reference": "SpectralMeasurement",
            "alternative": "SpectralMeasurement",
            "audit_spec": "dimensionless policy",
            "report": "JSON signal-level audit",
        },
        validity_checks=(
            "measurements use the same instrument and public signal contract",
            "measurement axes are identical",
            "each state supplies the configured minimum replicate count",
            "replicate arrays are finite and match the public axis",
        ),
        diagnostic_fields=(
            "identifiable",
            "replicate_stable",
            "states_distinct",
            "between_state_rmse",
            "separation_ratio",
            "warnings",
        ),
        failure_policy=(
            "reject invalid signal pairs with an explicit unsuccessful result; "
            "valid but indistinguishable pairs return a successful diagnostic "
            "whose identifiable field is false"
        ),
        provenance=(
            "ChemWorld public replicate-signal RMSE identities",
            f"chemicals {CHEMICALS_COMMIT}: non-instrument reference boundary",
            f"RMG-Py {RMG_PY_COMMIT}: non-instrument reference boundary",
        ),
        intended_operations=INTEGRATION_OPERATIONS,
    )


class SpectralIdentifiabilityProvider:
    """Diagnostic provider for pairwise public instrument signals."""

    @property
    def model_contract(self) -> ModelProviderContract:
        return spectroscopy_identifiability_provider_contract()

    def validate_domain(self, inputs: Mapping[str, Any]) -> tuple[str, ...]:
        violations: list[str] = []
        if not isinstance(inputs.get("reference"), SpectralMeasurement):
            violations.append("reference must be a SpectralMeasurement")
        if not isinstance(inputs.get("alternative"), SpectralMeasurement):
            violations.append("alternative must be a SpectralMeasurement")
        audit_spec = inputs.get("audit_spec")
        if audit_spec is not None and not isinstance(
            audit_spec,
            SpectralIdentifiabilitySpec,
        ):
            violations.append("audit_spec must be a SpectralIdentifiabilitySpec or None")
        return tuple(violations)

    def evaluate(self, inputs: Mapping[str, Any]) -> ModelProviderResult:
        domain_violations = self.validate_domain(inputs)
        if domain_violations:
            return _failed_result("; ".join(domain_violations), self.model_contract.provenance)
        reference = inputs["reference"]
        alternative = inputs["alternative"]
        audit_spec = inputs.get("audit_spec")
        try:
            report = evaluate_spectral_identifiability(
                reference,
                alternative,
                spec=audit_spec,
            )
        except ValueError as error:
            return _failed_result(str(error), self.model_contract.provenance)
        diagnostics = {
            "identifiable": report.identifiable,
            "replicate_stable": report.replicate_stable,
            "states_distinct": report.states_distinct,
            "between_state_rmse": report.between_state_rmse,
            "separation_ratio": report.separation_ratio,
            "warnings": list(report.warnings),
        }
        return ModelProviderResult(
            outputs={"report": report.to_dict()},
            diagnostics=diagnostics,
            warnings=report.warnings,
            provenance=report.provenance,
        )


def spectroscopy_identifiability_adapter_manifest() -> ModelAdapterManifest:
    """Return the claim-bound proposal for later WF-110 intake."""

    return ModelAdapterManifest(
        adapter_id="wf-20-spectral-identifiability",
        adapter_version="0.1",
        owner_workstream="wf-20-spectral-identifiability",
        provider_contract=spectroscopy_identifiability_provider_contract(),
        owned_paths=OWNED_PATHS,
        integration_operations=INTEGRATION_OPERATIONS,
        target_world_law="chemworld-physical-chemistry-vnext",
        status="proposal",
    )


def _failed_result(
    failure_reason: str,
    provenance: tuple[str, ...],
) -> ModelProviderResult:
    return ModelProviderResult(
        outputs={},
        diagnostics={
            "identifiable": False,
            "replicate_stable": False,
            "states_distinct": False,
            "between_state_rmse": None,
            "separation_ratio": None,
            "warnings": [failure_reason],
        },
        warnings=(failure_reason,),
        success=False,
        failure_reason=failure_reason,
        provenance=provenance,
    )


def _runtime_failed_result(failure_reason: str) -> ModelProviderResult:
    return ModelProviderResult(
        outputs={},
        diagnostics={
            "instrument_id": None,
            "packet_schema_valid": False,
            "layered_packet": False,
            "identity_safe": False,
            "finite_signal": False,
            "calibration_profile": None,
            "replicate_count": 0,
            "missingness_count": 0,
            "saturation_count": 0,
        },
        warnings=(failure_reason,),
        success=False,
        failure_reason=failure_reason,
        provenance=INSTRUMENT_RUNTIME_PROVENANCE,
    )


def _finite_optional_mapping(values: Mapping[Any, Any]) -> bool:
    return all(
        value is None
        or (
            isinstance(value, int | float)
            and not isinstance(value, bool)
            and isfinite(float(value))
        )
        for value in values.values()
    )


def _finite_nonnegative_mapping(values: Mapping[Any, Any]) -> bool:
    return all(
        isinstance(value, int | float)
        and not isinstance(value, bool)
        and isfinite(float(value))
        and float(value) >= 0.0
        for value in values.values()
    )


def _nested_keys(payload: Any) -> set[str]:
    keys: set[str] = set()
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            keys.add(str(key).lower())
            keys.update(_nested_keys(value))
    elif isinstance(payload, list | tuple):
        for value in payload:
            keys.update(_nested_keys(value))
    return keys


def _all_numeric_values_finite(payload: Any) -> bool:
    if isinstance(payload, Mapping):
        return all(_all_numeric_values_finite(value) for value in payload.values())
    if isinstance(payload, list | tuple):
        return all(_all_numeric_values_finite(value) for value in payload)
    if isinstance(payload, float):
        return isfinite(payload)
    return True


def _layered_signal(packet: Mapping[str, Any]) -> bool:
    return {
        "sample_state",
        "raw_signal",
        "peaks",
        "assignments",
        "processed_estimates",
        "uncertainty",
        "calibration",
        "missingness",
        "metadata",
    } <= set(packet)


def _is_layered_packet(instrument_id: str, packet: Mapping[str, Any]) -> bool:
    if instrument_id != "final_assay":
        return _layered_signal(packet)
    spectra = packet.get("spectra")
    if not isinstance(spectra, Mapping):
        return False
    required = {"hplc", "gc", "uvvis", "ph_meter"}
    return required <= set(spectra) and all(
        isinstance(spectra[channel], Mapping) and _layered_signal(spectra[channel])
        for channel in required
    )


def _missingness_count(packet: Mapping[str, Any]) -> int:
    missingness = packet.get("missingness")
    count = 0
    if isinstance(missingness, Mapping):
        entries = missingness.get("entries", ())
        if isinstance(entries, list | tuple):
            count += len(entries)
    spectra = packet.get("spectra")
    if isinstance(spectra, Mapping):
        count += sum(
            _missingness_count(value) for value in spectra.values() if isinstance(value, Mapping)
        )
    return count


def _saturation_count(packet: Mapping[str, Any]) -> int:
    count = 0
    peaks = packet.get("peaks", ())
    if isinstance(peaks, list | tuple):
        count += sum(
            bool(peak.get("saturated", False)) for peak in peaks if isinstance(peak, Mapping)
        )
    spectra = packet.get("spectra")
    if isinstance(spectra, Mapping):
        count += sum(
            _saturation_count(value) for value in spectra.values() if isinstance(value, Mapping)
        )
    return count


__all__ = [
    "FORBIDDEN_PACKET_KEYS",
    "INTEGRATION_OPERATIONS",
    "OWNED_PATHS",
    "PUBLIC_SPECIES_CHANNELS",
    "SpectralIdentifiabilityProvider",
    "ValidatedInstrumentRuntimeProvider",
    "instrument_runtime_adapter_manifest",
    "instrument_runtime_model_card",
    "instrument_runtime_provider_contract",
    "spectroscopy_identifiability_adapter_manifest",
    "spectroscopy_identifiability_provider_contract",
]
