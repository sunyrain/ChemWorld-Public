"""Public-payload leakage audit helpers.

These checks are intentionally conservative for agent-facing payloads. Public
views may expose stable protocol hashes and aggregate public labels, but they
must not expose hidden mechanism internals such as true species ids, rate laws,
stoichiometric matrices, or rate constants.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

SENSITIVE_KEY_NAMES = frozenset(
    {
        "compiled_mechanism",
        "debug_mechanism",
        "hidden_parameters",
        "initial_amount_policy",
        "mechanism_manifest",
        "mechanism_observable_mapping",
        "observable_mapping",
        "private_salt",
        "rate_constants",
        "rate_law",
        "rate_law_equation_ids",
        "rate_law_evaluators",
        "reaction_enthalpies",
        "reactions",
        "score_spec",
        "species_amounts",
        "species_amounts_mol",
        "species_index",
        "species_roles",
        "stoichiometric_matrix",
        "theta",
        "truth",
    }
)

SPECIES_VALUE_KEYS = frozenset(
    {
        "species_id",
        "species_ids",
        "target_species",
        "impurity_species",
        "initial_limiting_species",
        "reactant_species",
        "product_species",
    }
)

PUBLIC_SPECIES_SUFFIX = "_public"


@dataclass(frozen=True)
class PublicLeakageFinding:
    path: str
    reason: str
    key: str
    value_preview: str

    def to_dict(self) -> dict[str, str]:
        return {
            "path": self.path,
            "reason": self.reason,
            "key": self.key,
            "value_preview": self.value_preview,
        }


def audit_public_payload(
    payload: Any,
    *,
    hidden_species_ids: set[str] | frozenset[str] | None = None,
    allow_debug_truth: bool = False,
) -> list[PublicLeakageFinding]:
    """Return findings for hidden mechanism data in a public payload."""

    findings: list[PublicLeakageFinding] = []
    hidden_species = frozenset(hidden_species_ids or ())

    def visit(value: Any, path: str, parent_key: str = "") -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                key_text = str(key)
                key_norm = key_text.lower()
                child_path = f"{path}.{key_text}" if path else key_text
                if allow_debug_truth and (key_norm == "truth" or key_norm.startswith("debug_")):
                    continue
                if key_norm in SENSITIVE_KEY_NAMES:
                    findings.append(
                        PublicLeakageFinding(
                            path=child_path,
                            reason="sensitive_key",
                            key=key_text,
                            value_preview=_preview(child),
                        )
                    )
                    continue
                visit(child, child_path, key_norm)
        elif isinstance(value, list | tuple):
            for index, child in enumerate(value):
                visit(child, f"{path}[{index}]", parent_key)
        elif isinstance(value, str):
            _check_string_value(value, path, parent_key, hidden_species, findings)

    visit(payload, "$")
    return findings


def public_leakage_passed(
    payload: Any,
    *,
    hidden_species_ids: set[str] | frozenset[str] | None = None,
    allow_debug_truth: bool = False,
) -> bool:
    return not audit_public_payload(
        payload,
        hidden_species_ids=hidden_species_ids,
        allow_debug_truth=allow_debug_truth,
    )


def _check_string_value(
    value: str,
    path: str,
    parent_key: str,
    hidden_species: frozenset[str],
    findings: list[PublicLeakageFinding],
) -> None:
    if parent_key in {"channel_id", "derived_from", "sample_id", "clock_id"}:
        if (value.lower() in SENSITIVE_KEY_NAMES
                or value.lower().startswith(("true_", "internal_", "solver_", "diagnostic"))):
            findings.append(PublicLeakageFinding(
                path, "private_channel_identity", parent_key, _preview(value),
            ))
        if value in hidden_species:
            findings.append(PublicLeakageFinding(
                path, "hidden_species_id", parent_key, _preview(value),
            ))
    if parent_key == "species_id" and not value.endswith(PUBLIC_SPECIES_SUFFIX):
        findings.append(
            PublicLeakageFinding(
                path=path,
                reason="non_public_species_label",
                key=parent_key,
                value_preview=_preview(value),
            )
        )
        return
    if parent_key in SPECIES_VALUE_KEYS or parent_key in {
        "assignment",
        "group",
        "role",
        "public_role",
    }:
        values = [value]
    else:
        values = []
    for item in values:
        if item in hidden_species:
            findings.append(
                PublicLeakageFinding(
                    path=path,
                    reason="hidden_species_id",
                    key=parent_key,
                    value_preview=_preview(value),
                )
            )
            return


def _preview(value: Any, max_len: int = 120) -> str:
    text = repr(value)
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


__all__ = [
    "PUBLIC_SPECIES_SUFFIX",
    "PublicLeakageFinding",
    "audit_public_payload",
    "public_leakage_passed",
]
