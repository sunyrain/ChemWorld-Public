"""Apply an explicit finite parameter mapping through installed model codecs."""

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, replace

from chemworld.data.scientific_inputs import QuantityColumn, ScientificTable
from chemworld.world.physical_world import CompositionPhysics


@dataclass(frozen=True)
class ModelParameterTarget:
    column: QuantityColumn
    model_id: str
    model_version: str
    binding: str
    field_path: tuple[str, ...]

    def __post_init__(self):
        if self.column.role != "parameter":
            raise ValueError("model bindings require parameter columns")
        if not all(isinstance(v, str) and v.strip() == v and v for v in (
            self.model_id, self.model_version, self.binding, *self.field_path,
        )) or not self.field_path:
            raise ValueError("parameter targets require explicit model, binding and field path")
        object.__setattr__(self, "field_path", tuple(self.field_path))


@dataclass(frozen=True)
class ParameterImport:
    physics: CompositionPhysics
    receipt: dict


def bind_parameter_row(
    physics: CompositionPhysics,
    table: ScientificTable,
    row_id: str,
    targets: tuple[ModelParameterTarget, ...],
) -> ParameterImport:
    """Return new configuration and provenance; never mutate a running session.

    The caller supplies the finite whitelist, independently of untrusted input.
    Only numeric leaves of installed model payloads can be assigned. Initial
    states, network structure, ownership and execution objects are not targets.
    """
    if table.kind != "parameters":
        raise ValueError("measurement data cannot be bound to physical parameters")
    rows = [row for row in table.rows if row.row_id == row_id]
    if len(rows) != 1:
        raise ValueError("select exactly one existing parameter row")
    row = rows[0]
    columns = {col.quantity_id: col for col in table.columns}
    if len(targets) != len(columns) or {t.column.quantity_id for t in targets} != set(columns):
        raise ValueError("targets must cover the finite parameter set exactly once")
    paths = {(t.model_id, t.binding, t.field_path) for t in targets}
    if len(paths) != len(targets):
        raise ValueError("multiple parameters cannot write the same model field")
    declarations = physics.modules.model_specs()
    payloads, applied = {}, []
    for target in targets:
        name = target.column.quantity_id
        if columns[name].reference_state != target.column.reference_state:
            raise ValueError(f"parameter {name} uses a different reference state")
        value = target.column.normalize(row.values[name])
        declaration = declarations.get(target.model_id)
        if declaration is None or declaration.version != target.model_version:
            raise ValueError("unknown or incompatible parameter target model/version")
        models = physics.models_of(target.model_id)
        if target.binding not in models:
            raise ValueError("parameter target binding is not present")
        key = (target.model_id, target.binding)
        if key not in payloads:
            payloads[key] = deepcopy(declaration.encode(models[target.binding]))
        node = payloads[key]
        for part in target.field_path[:-1]:
            if not isinstance(node, Mapping) or part not in node:
                raise ValueError("parameter field path does not exist in the declared model")
            node = node[part]
        leaf = target.field_path[-1]
        if (not isinstance(node, dict) or leaf not in node or isinstance(node[leaf], bool)
                or not isinstance(node[leaf], (int, float))):
            raise ValueError("parameter target must be an existing numeric model leaf")
        if isinstance(node[leaf], int):
            if not value.value.is_integer():
                raise ValueError("integer model parameters cannot be rounded during import")
            node[leaf] = int(value.value)
        else:
            node[leaf] = value.value
        applied.append({
            "quantity_id": name, "model_id": target.model_id, "model_version": target.model_version,
            "binding": target.binding, "field_path": list(target.field_path),
            "reference_state": target.column.reference_state, **value.to_dict(),
        })
    models = {key: dict(bindings) for key, bindings in physics.models.items()}
    for (model_id, binding), payload in payloads.items():
        models[model_id][binding] = declarations[model_id].decode(payload)
    return ParameterImport(
        replace(physics, models=models),
        {
            "dataset_id": table.dataset_id, "row_id": row_id, "group_id": row.group_id,
            "source": next(
                source.to_dict() for source in table.sources
                if source.provenance.source_id == row.source_id
            ),
            "parameters": applied,
        },
    )
