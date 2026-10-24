"""Render the installed module declarations; never maintain a second registry."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from chemworld.world.domain_modules import DomainModuleRegistry


def capability_rows(registry=None):
    registry = registry or DomainModuleRegistry.default()
    return [
        {
            "module": module.module_id,
            "dependencies": list(module.dependencies),
            "scope": list(module.scope_notes),
            "advance_strategies": [spec.strategy_id for spec in module.advance_strategies],
            "components": {
                name: {
                    "operations": list(component.operations),
                    "interfaces": list(component.interfaces),
                    "state_owners": list(component.state_owners),
                    "processes": list(component.processes),
                    "ports": {
                        "input": list(component.input_ports),
                        "output": list(component.outputs),
                        "unit": component.port_unit,
                    },
                    "parameters": {key: asdict(rule) for key, rule in component.parameters},
                }
                for name, component in module.components
            },
            "operations": {
                operation.operation.id: {
                    "service": operation.service_id,
                    "fields": [f.public_schema() for f in operation.fields],
                }
                for operation in module.operations
            },
            "instruments": {
                instrument.definition.id: instrument.contract.to_dict()
                for instrument in module.instruments
            },
        }
        for module in registry.modules
    ]


def bound_capabilities(session):
    """Reuse the bound runtime's public contracts, without another range registry."""
    from chemworld.runtime.component_network import public_network

    return {
        "scope": "Installed, task-permitted and executable capabilities in the active vessel. "
        "Module declarations describe installed software, not apparatus in every world.",
        "active_vessel": session._state.vessel_id,
        "actions": session.available_actions(include_invalid=True),
        "instruments": session.action_schema("measure").get("fields", [{}])[0].get(
            "choice_status", []
        ),
        "connections": public_network(session._state),
        "modules": capability_rows(session.domain_modules),
    }


def execution_inventory(registry=None):
    """Project installed declarations without inferring compiled or thread-safe support.

    An exported function or a model codec is not a scientific qualification result.
    Task-specific evidence belongs to the existing workstream records. This view also
    works in an installed package, without reading repository task documents.
    """
    from chemworld.physchem import _MODULE_EXPORTS

    registry = registry or DomainModuleRegistry.default()

    def address(function):
        return f"{function.__module__}:{function.__qualname__}"

    declarations = {module.module_id: module for module in registry.modules}
    rows = capability_rows(registry)
    for row in rows:
        module = declarations[row["module"]]
        row["execution"] = {
            "python": "installed_declaration; validity_depends_on_bound_world",
            "cpu_lowering": "not_declared",
            "thread_safety": "not_declared",
            "continuation": "runtime_session_v2; model_specific_qualification_required",
            "model_codecs": [
                {
                    "id": model.model_id,
                    "version": model.version,
                    "binding": model.binding,
                    "decode": address(model.decode),
                    "encode": address(model.encode),
                }
                for model in module.models
            ],
            "advance_bindings": [
                {
                    "id": strategy.strategy_id,
                    "factory": address(strategy.factory),
                    "shared_clock": strategy.shared_clock,
                    "timed_measurements": strategy.timed_measurements,
                }
                for strategy in module.advance_strategies
            ],
        }
    return {
        "schema": "chemworld-execution-inventory-v1",
        "scope": "Generated installed entry points, not scientific qualification or CPU support.",
        "modules": rows,
        "public_physchem": [
            {"module": "chemworld.physchem." + module, "symbols": list(symbols)}
            for module, symbols in sorted(_MODULE_EXPORTS.items())
        ],
    }


def markdown():
    rows = [
        "# Installed native capability declarations",
        "",
        "Generated from `DomainModuleRegistry`; not a second module registry. "
        "Operation availability and numeric domains are further narrowed by the "
        "actual composition, physical specifications, device and task. A module "
        "declaration does not guarantee arbitrary cross-domain coupling.",
        "",
        "| Module | Components / strategies | Scope |",
        "| --- | --- | --- |",
    ]
    for row in capability_rows():
        names = ", ".join([*row["components"], *row["advance_strategies"]]) or "Shared operations"
        scope = (
            " ".join(row["scope"])
            or "See the authored physical spec and runtime contract; no universal validity claim."
        )
        rows.append(f"| `{row['module']}` | {names} | {scope} |")
    rows += [
        "",
        "For declared action units, parameter rules, state ownership and dependencies:",
        "`uv run --no-sync python -m chemworld.examples.native_capabilities --format json`",
        "",
    ]
    return "\n".join(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--format", choices=("markdown", "json", "execution-json"), default="markdown"
    )
    parser.add_argument("--composition", type=Path, help="Export bound public contracts as JSON")
    args = parser.parse_args()
    if args.composition:
        from chemworld.runtime import ChemWorldRuntime

        payload = json.loads(args.composition.read_text())
        session = ChemWorldRuntime(composition=payload.get("composition", payload))
        try:
            session.reset(seed=7)
            print(json.dumps(bound_capabilities(session), indent=2))
        finally:
            session.close()
        return
    if args.format == "execution-json":
        print(json.dumps(execution_inventory(), indent=2))
    else:
        print(markdown() if args.format == "markdown" else json.dumps(capability_rows(), indent=2))


if __name__ == "__main__":
    main()
