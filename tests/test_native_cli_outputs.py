"""Output preflight must finish before a CLI session writes user files."""

import pytest

from chemworld.cli import main


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
