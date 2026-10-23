"""Source selection and validity are meaningful model inputs."""

import json
import sqlite3

import pytest

from chemworld.data.unifac_sources import (
    iter_ddbst_sqlite,
    iter_ddbst_text,
    read_unifac_interactions,
    read_unifac_subgroups,
    select_unifac_interactions,
    unifac_model_from_assignments,
)
from chemworld.physchem.data_conflicts import DataSourceProvenance
from chemworld.physchem.equilibrium import activity_coefficients

SOURCE = DataSourceProvenance("test", 0, "authored source data", version="1")


def test_explicit_revisions_retain_overridden_values(tmp_path):
    path = tmp_path / "source.tsv"
    path.write_text("1\t2\t100\n2\t1\t-50\n", encoding="utf-8")
    first = read_unifac_interactions(path, source=SOURCE)
    path.write_text("1\t2\t200\t.1\t.001\n", encoding="utf-8")
    second = read_unifac_interactions(path, source=SOURCE)
    with pytest.raises(ValueError, match="precedence"):
        select_unifac_interactions((first, second))
    parameters, selection = select_unifac_interactions(
        (first, second), later_revisions_override=True
    )
    assert parameters[1, 2] == (200, 0.1, 0.001)
    assert selection["superseded"] == [first[0].to_dict()]
    assert parameters[2, 1] == (-50, 0, 0)


def test_text_flags_and_sqlite_cas_consume_explicit_groups(tmp_path):
    path = tmp_path / "groups.tsv"
    path.write_text("AAAA\t1 0 1\t1 2\t2 3\t1 2\nBBBB\t1 1 0\t2 1\t2 1\t\n", encoding="utf-8")
    rows = list(iter_ddbst_text(path, source=SOURCE))
    assert rows[0][1].groups == {2: 3}
    assert not rows[0][1].valid
    assert not rows[1][2].groups
    with pytest.raises(TypeError):
        rows[0][0].groups[1] = 3
    db = tmp_path / "assignments.sqlite"
    with sqlite3.connect(db) as c:
        c.execute(
            'CREATE TABLE DDBST ("index" INTEGER, UNIFAC TEXT, MODIFIED_UNIFAC TEXT, PSRK TEXT)'
        )
        c.execute("INSERT INTO DDBST VALUES (64175, ?, ?, ?)", ("1:2", "", "1:2"))
        c.execute("INSERT INTO DDBST VALUES (110543, ?, ?, ?)", ("2:1", "2:1", ""))
    native = list(iter_ddbst_sqlite(db, source=SOURCE, cas_number="64-17-5"))
    assert len(native) == 1 and native[0][0].groups == rows[0][0].groups
    assert native[0][0].identity == "64-17-5"
    assert not native[0][1].valid
    card = tmp_path / "subgroups.json"
    card.write_text(
        json.dumps(
            {
                "source": SOURCE.to_dict(),
                "subgroups": [
                    {"group_id": 1, "main_group_id": 1, "R": 0.9, "Q": 0.8},
                    {"group_id": 2, "main_group_id": 2, "R": 1.5, "Q": 1.2},
                ],
            }
        ),
        encoding="utf-8",
    )
    subgroups, source = read_unifac_subgroups(card)
    ip = tmp_path / "interactions.tsv"
    ip.write_text("1\t2\t200\n2\t1\t50\n", encoding="utf-8")
    revisions = (read_unifac_interactions(ip, source=SOURCE),)
    assignments = {"A": rows[0][0], "B": rows[1][0]}
    spec = unifac_model_from_assignments(
        "from-source",
        assignments,
        subgroups=subgroups,
        subgroup_source=source,
        revisions=revisions,
        version=0,
    )
    assert activity_coefficients(spec, {"A": 0.3, "B": 0.7}, temperature_K=320)["A"] > 1
    assert spec.provenance["sources"]["molecular_sources"]["A"]["identity"] == "AAAA"
    with pytest.raises(ValueError, match="valid molecular"):
        unifac_model_from_assignments(
            "bad",
            {"A": rows[0][1]},
            subgroups=subgroups,
            subgroup_source=source,
            revisions=revisions,
            version=1,
        )
    with pytest.raises(ValueError, match="only original"):
        unifac_model_from_assignments(
            "wrong-topology",
            assignments,
            subgroups=subgroups,
            subgroup_source=source,
            revisions=revisions,
            version=3,
        )


@pytest.mark.parametrize("text", ["1\t2\t2\n1\t2\t3\n", "1\t2\n", "1\t2\tnan\n"])
def test_bad_interaction_tables_rejected(tmp_path, text):
    path = tmp_path / "bad.tsv"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError):
        read_unifac_interactions(path, source=SOURCE)


def test_nist_temperature_range_and_invalid_ddbst_sentinel_preserved(tmp_path):
    path = tmp_path / "range.tsv"
    path.write_text("1\t2\t257.05\t-.1011\t0\t277.59\t576.96\n", encoding="utf-8")
    record = read_unifac_interactions(path, source=SOURCE)[0]
    assert record.temperature_range_K == (277.59, 576.96)
    assert record.to_dict()["temperature_range_K"] == [277.59, 576.96]
    path.write_text("AAAA\t0 1 0\t-1 1\t2 1\t-1 2\n", encoding="utf-8")
    row = next(iter_ddbst_text(path, source=SOURCE))
    assert not row[0].valid and row[0].groups == {-1: 1}
    assert row[1].valid and row[1].groups == {2: 1}
