"""Consume the existing complete Bell2018 records as native EOS alpha inputs."""

from __future__ import annotations

from chemworld.data.phase_change import PhaseChangeRecord
from chemworld.physchem.eos import EOSComponentSpec
from chemworld.physchem.specs import ComponentSpec


def twu_eos_component_from_record(
    record: PhaseChangeRecord,
    *,
    component: ComponentSpec | None = None,
) -> EOSComponentSpec:
    """Bind Tc/Pc/L/M/N and exact source identity, without inventing omega.

    Bell's source contains no acentric factor. The original component's legacy
    zero field is unused by the explicitly selected Twu alpha, and this role is
    recorded rather than presenting it as a measured property.
    """
    if not isinstance(record, PhaseChangeRecord) or record.source_id != "BELL_TWU":
        raise ValueError("Twu EOS import requires an original Bell2018 PhaseChangeRecord")
    identity = record.source_key
    if not identity:
        raise ValueError("Bell Twu record requires its InChIKey identity")
    if component is not None and identity not in {component.identifier, component.inchi_key}:
        raise ValueError("Bell Twu InChIKey does not match the selected native component")
    values = {key: record.value(key) for key in ("Tc", "Pc", "c0", "c1", "c2")}
    return EOSComponentSpec(
        identity if component is None else component.identifier,
        values["Tc"].quantity("K").value,
        values["Pc"].quantity("Pa").value,
        alpha_model="twu91",
        alpha_coefficients=tuple(
            values[key].quantity("dimensionless").value for key in ("c0", "c1", "c2")
        ),
        alpha_provenance={
            "record_id": record.record_id,
            "InChIKey": identity,
            "source": record.source.provenance.to_dict(),
            "parameters": {k: v.to_dict() for k, v in values.items()},
            "coefficient_order": "L=c0, M=c1, N=c2",
            "acentric_factor_role": (
                "legacy zero field unused by selected Twu91; not a measured omega"),
            "application_basis": (
                "selected cubic EOS; source fit does not establish transferable material accuracy"),
        },
    )
