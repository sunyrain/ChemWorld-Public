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
    parser.add_argument("--format", choices=("markdown", "json"), default="markdown")
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
    print(markdown() if args.format == "markdown" else json.dumps(capability_rows(), indent=2))


if __name__ == "__main__":
    main()
