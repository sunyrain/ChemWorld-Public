"""Protect CLI output compatibility, concise discovery and output-file preflight."""

import json

import pytest

from chemworld.cli import _load_json_file, main
from chemworld.examples import native_capabilities
from chemworld.tasks import list_tasks
from chemworld.world.domain_modules import DomainModuleRegistry
from chemworld.world.scenario import list_scenarios


@pytest.mark.parametrize("existing", [False, True])
def test_render_never_overwrites_or_aliases_outputs(tmp_path, existing):
    path = tmp_path / "result.json"
    if existing:
        path.write_text("retained", encoding="utf-8")
    args = ["render", "--task", "reaction-to-assay", "--trajectory", str(path)]
    if not existing:
        args += ["--snapshot", str(path)]
    with pytest.raises(SystemExit) as exc:
        main(args)
    assert exc.value.code == 2
    assert path.read_text(encoding="utf-8") == "retained" if existing else not path.exists()


@pytest.mark.parametrize(
    "arguments, identity, omitted",
    [
        (["tasks", "list"], "task_id", {"kernel_maturity", "allowed_operations"}),
        (["scenarios", "list"], "scenario_id", {"hidden_parameter_seed", "initial_state_seed"}),
        (["capabilities"], "module", {"scope", "dependencies"}),
    ],
)
def test_summary_commands_preserve_every_entry_without_full_schemas(
    arguments, identity, omitted, capsys,
):
    main(arguments)
    full_text = capsys.readouterr().out
    full = json.loads(full_text)
    main([*arguments, "--format", "summary"])
    summary_text = capsys.readouterr().out
    summary = json.loads(summary_text)
    assert summary and len(summary_text.encode("utf-8")) < len(full_text.encode("utf-8"))
    assert [row[identity] for row in summary] == [row[identity] for row in full]
    for row, original in zip(summary, full, strict=True):
        assert omitted.isdisjoint(row)
        if identity != "module":
            assert row == {key: original[key] for key in row}
            continue
        assert row["components"] == sorted(original["components"])
        assert row["instruments"] == sorted(original["instruments"])
        assert row["advance_strategies"] == original["advance_strategies"]
        expected_operations = set(original["operations"])
        for component in original["components"].values():
            expected_operations.update(component["operations"])
        assert row["operations"] == sorted(expected_operations)


@pytest.mark.parametrize("arguments", [["tasks", "list"], ["scenarios", "list"], ["capabilities"]])
def test_default_and_explicit_json_keep_complete_catalog(arguments, capsys):
    main(arguments)
    default = capsys.readouterr().out
    main([*arguments, "--format", "json"])
    assert capsys.readouterr().out == default
    if arguments[0] == "tasks":
        expected = [task.to_dict() for task in list_tasks()]
    elif arguments[0] == "scenarios":
        expected = [scenario.to_dict() for scenario in list_scenarios()]
    else:
        expected = native_capabilities.capability_rows()
    # Declarations may contain tuples; compare their actual JSON representation.
    assert json.loads(default) == json.loads(json.dumps(expected))


@pytest.mark.parametrize("empty", [False, True])
def test_capability_summary_does_not_build_full_schemas(monkeypatch, empty):
    registry = DomainModuleRegistry(()) if empty else DomainModuleRegistry.default()

    def unexpected(*args, **kwargs):
        raise AssertionError("Summary must not build or serialize the full schemas")

    monkeypatch.setattr(native_capabilities, "capability_rows", unexpected)
    monkeypatch.setattr(native_capabilities, "asdict", unexpected)
    summary = native_capabilities.summary_rows(registry)
    assert [row["module"] for row in summary] == [module.module_id for module in registry.modules]
def test_json_cli_inputs_accept_inline_objects(capsys):
    main(
        [
            "validate-action",
            "--task",
            "reaction-to-assay",
            "--action",
            '{"operation":"add_reagent","amount_mol":0.012}',
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["valid"] is True


def test_json_cli_input_error_explains_file_or_inline_form():
    with pytest.raises(ValueError, match="JSON object file path or an inline JSON object"):
        _load_json_file("not-json")
