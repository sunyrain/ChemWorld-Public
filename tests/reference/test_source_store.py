import json
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from source_store import SourceStore


def test_workers_share_bytes_and_keep_independent_original_bindings(tmp_path):
    block = tmp_path / "block"
    SourceStore(block)
    source = tmp_path / "module.py"
    source.write_bytes(b"VALUE = 36\n")

    def capture(index):
        unit = block / str(index)
        unit.mkdir()
        return SourceStore(unit).capture(source)[0]

    with ThreadPoolExecutor(max_workers=4) as pool:
        targets = list(pool.map(capture, range(12)))
    assert len(set(targets)) == 1
    assert list((block / ".source-store").glob("*/*")) == [targets[0]]
    assert len(list(block.glob("*/source-references.json"))) == 12
    timestamp = targets[0].stat().st_mtime_ns
    source.write_bytes(b"VALUE = 37\n")
    with pytest.raises(ValueError, match="loaded source changed"):
        SourceStore(block / "0").capture(source)
    assert targets[0].read_bytes() == b"VALUE = 36\n"
    assert targets[0].stat().st_mtime_ns == timestamp
    # A different unit may legitimately start from the revised development source.
    assert capture(12) != targets[0]


@pytest.mark.parametrize("damage", ["missing", "changed"])
def test_missing_or_corrupted_objects_are_not_silently_replaced(tmp_path, damage):
    source = tmp_path / "source.py"
    source.write_bytes(b"first")
    directory = tmp_path / "unit"
    target, _ = SourceStore(directory).capture(source)
    binding = (directory / "source-references.json").read_bytes()
    if damage == "missing":
        target.unlink()
    else:
        target.write_bytes(b"damaged")
    with pytest.raises(ValueError, match=f"snapshot {damage}"):
        SourceStore(directory).capture(source)
    assert (directory / "source-references.json").read_bytes() == binding
    assert not target.exists() if damage == "missing" else target.read_bytes() == b"damaged"


def test_whole_block_can_move_without_losing_references(tmp_path):
    block = tmp_path / "block"
    SourceStore(block)
    unit = block / "case"
    unit.mkdir()
    source = tmp_path / "original.py"
    source.write_bytes(b"preserved bytes")
    SourceStore(unit).capture(source)
    moved = tmp_path / "retained-block"
    shutil.move(str(block), moved)
    store = SourceStore(moved / "case")
    reference = json.loads(store.index.read_text(encoding="utf-8"))
    assert not Path(reference["store"]).is_absolute()
    assert store.object_path(store.bindings[str(source)]).read_bytes() == b"preserved bytes"
