from scripts.smoke_test_wheel import PROBE, clean_environment


def test_failed_installed_command_retains_both_streams(tmp_path):
    import sys

    import pytest
    from scripts.smoke_test_wheel import captured_run

    prefix = tmp_path / "failed"
    with pytest.raises(RuntimeError, match="Installed command failed"):
        captured_run([sys.executable, "-c",
                      "import sys; print('partial result'); print('reason', file=sys.stderr); "
                      "sys.exit(7)"], log_prefix=prefix)
    assert prefix.with_suffix(".stdout.log").read_text().strip() == "partial result"
    assert prefix.with_suffix(".stderr.log").read_text().strip() == "reason"


def test_child_progress_is_visible_before_process_completion(tmp_path, monkeypatch):
    import io
    import sys

    from scripts.smoke_test_wheel import captured_run

    prefix = tmp_path / "live"
    acknowledgement = tmp_path / "progress-ack"
    observed = []

    class Output(io.StringIO):
        def write(self, text):
            if text.startswith("CHEMWORLD_PROGRESS"):
                observed.append("finished" not in prefix.with_suffix(".stdout.log").read_text())
                acknowledgement.touch()
            return super().write(text)

    monkeypatch.setattr(sys, "stdout", Output())
    result = captured_run([
        sys.executable, "-c", "import sys,time\nfrom pathlib import Path\n"
        "ack=Path(sys.argv[1])\n"
        "print('CHEMWORLD_PROGRESS live',file=sys.stderr,flush=True)\n"
        "deadline=time.monotonic()+5\n"
        "while not ack.exists() and time.monotonic()<deadline: time.sleep(.01)\n"
        "assert ack.exists(), 'parent did not forward progress'\n"
        "print('finished',flush=True)\n", str(acknowledgement),
    ], log_prefix=prefix, poll_interval_s=.02)
    assert result.stdout.strip() == "finished"
    assert any(observed), "progress was withheld until the child exited"


def test_failure_of_one_archive_does_not_hide_the_other_unit(tmp_path, monkeypatch):
    import json
    import sys

    from scripts import smoke_test_wheel as smoke

    archives = [tmp_path / "first.whl", tmp_path / "second.tar.gz"]
    for archive in archives:
        archive.write_bytes(b"fixture archive; no installation attempted")
    calls = []

    def install(archive, **kwargs):
        calls.append(archive)
        if archive == archives[0]:
            raise RuntimeError("injected installation failure")
        return {"installed_check": "passed"}

    output = tmp_path / "report.json"
    monkeypatch.setattr(smoke, "check_install", install)
    monkeypatch.setattr(sys, "argv", ["smoke", "--archive", str(archives[0]),
                                     "--archive", str(archives[1]), "--output", str(output)])
    assert smoke.main() == 1
    assert calls == archives
    rows = json.loads(output.read_text())
    assert [row["installed_check"] for row in rows] == ["failed", "passed"]
    assert "injected installation failure" in rows[0]["failure"]
    assert all(row["archive_sha256"] for row in rows)


def test_build_failure_reports_both_unexecuted_formats(tmp_path, monkeypatch):
    import json
    import sys

    from scripts import smoke_test_wheel as smoke

    def fail(*args, **kwargs):
        raise RuntimeError("injected build failure")

    output = tmp_path / "report.json"
    monkeypatch.setattr(smoke, "captured_run", fail)
    monkeypatch.setattr(sys, "argv", ["smoke", "--output", str(output)])
    assert smoke.main() == 1
    rows = json.loads(output.read_text())
    assert [row["archive_kind"] for row in rows] == ["wheel", "sdist"]
    assert all(row["installed_check"] == "not_executed" for row in rows)
    assert all("injected build failure" in row["failure"] for row in rows)


def test_installed_smoke_does_not_inherit_editable_imports(monkeypatch):
    monkeypatch.setenv("PYTHONPATH", "/private/checkout/src")
    monkeypatch.setenv("PYTHONHOME", "/other/python")
    monkeypatch.setenv("VIRTUAL_ENV", "/editable/venv")
    env = clean_environment()
    assert "PYTHONPATH" not in env
    assert "PYTHONHOME" not in env
    assert "VIRTUAL_ENV" not in env
    assert env["PYTHONNOUSERSITE"] == "1"


def test_current_installation_probe_does_not_require_frozen_research_evidence():
    assert "mechanism_adaptation_execution" not in PROBE
    assert "is_relative_to(Path(sys.prefix)" in PROBE
    assert "for task in list_tasks()" in PROBE
    assert "platform.python_version()" in PROBE
    assert "platform.system()" in PROBE
    compile(PROBE, "installed-package-probe", "exec")
