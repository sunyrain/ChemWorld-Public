"""Real bounded ZIP import/export; archive content is never extracted or run."""

import stat
from io import BytesIO
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

import pytest

from chemworld.data.combine_archive import (
    COMBINE_URI,
    MANIFEST_NAMESPACE,
    ArchiveEntry,
    CombineArchive,
)
from test_sbml import model
from test_sedml_declarations import SEDML


def example():
    source = model().source
    entries = (ArchiveEntry("model.xml", COMBINE_URI+"sbml", model().to_xml().encode()),
        ArchiveEntry("experiment.sedml", COMBINE_URI+"sed-ml.level-1.version-4",
                     SEDML.encode(), True),
        ArchiveEntry("notes.txt", "http://purl.org/NET/mediatypes/text/plain", b"Data-only notes"))
    return CombineArchive(entries, source)


def test_manifest_zip_model_experiment_metadata_and_source_roundtrip_is_deterministic(tmp_path):
    original = example()
    payload = original.to_bytes()
    restored = CombineArchive.from_bytes(payload, source=original.source)
    assert restored.entries == original.entries
    assert restored.source == original.source
    assert restored.to_bytes() == payload
    assert restored.master_location == "experiment.sedml"
    document = restored.document()
    assert document.to_xml() == SEDML
    assert document.source.provenance.license == original.source.provenance.license
    assert document.source.provenance.checksum
    assert restored.entry("./model.xml", relative_to="experiment.sedml").data == model().to_xml(
        ).encode()
    assert list(tmp_path.iterdir()) == []


def raw_archive(files, declaration, *, special=None, compression=ZIP_DEFLATED):
    output = BytesIO()
    manifest = ('<omexManifest xmlns="'+MANIFEST_NAMESPACE+'">'
        '<content location="." format="'+COMBINE_URI+'omex"/>'+declaration+'</omexManifest>')
    with ZipFile(output, "w", compression=compression) as archive:
        archive.writestr("manifest.xml", manifest)
        for name, content in files:
            if special:
                info = ZipInfo(name)
                info.create_system = 3
                info.external_attr = special << 16
                archive.writestr(info, content)
            else:
                archive.writestr(name, content)
    return output.getvalue()


def declaration(location="experiment.sedml", format="sed-ml", *, extra=""):
    return '<content location="'+location+'" format="'+COMBINE_URI+format+'" '+extra+'/>'


@pytest.mark.parametrize("mutation", ["traversal", "unlisted", "missing", "case_alias",
    "duplicate_declaration", "unknown_manifest_field", "unknown_format", "executable",
    "symlink", "compression_bomb", "ambiguous_master", "external_path"])
def test_archive_rejects_a_whole_bad_container_before_experiment_execution(mutation):
    files = [("experiment.sedml", SEDML)]
    declarations = declaration()
    special = None
    if mutation == "traversal":
        files = [("../experiment.sedml", SEDML)]
        declarations = declaration("../experiment.sedml")
    elif mutation == "unlisted":
        files.append(("unlisted.xml", "data"))
    elif mutation == "missing":
        declarations += declaration("missing.xml", "sbml")
    elif mutation == "case_alias":
        files.append(("EXPERIMENT.sedml", SEDML))
        declarations += declaration("EXPERIMENT.sedml")
    elif mutation == "duplicate_declaration":
        declarations += declaration("./experiment.sedml")
    elif mutation == "unknown_manifest_field":
        declarations = declaration(extra='execute="true"')
    elif mutation == "unknown_format":
        declarations = declaration(format="sed-ml.level-1.version-5")
    elif mutation == "executable":
        files.append(("run.py", "raise AssertionError('must never run')"))
        declarations += '<content location="run.py" format="text/plain"/>'
    elif mutation == "symlink":
        special = stat.S_IFLNK | 0o777
    elif mutation == "compression_bomb":
        files = [("experiment.sedml", "a"*100000)]
    elif mutation == "ambiguous_master":
        files.append(("second.sedml", SEDML))
        declarations += declaration("second.sedml")
    elif mutation == "external_path":
        declarations = declaration("https://example.org/experiment.sedml")
    payload = raw_archive(files, declarations, special=special)
    with pytest.raises(ValueError):
        CombineArchive.from_bytes(payload, source=model().source)


def test_a_nested_experiment_can_reference_an_internal_parent_directory():
    source = model().source
    archive = CombineArchive((
        ArchiveEntry("models/model.xml", COMBINE_URI+"sbml", model().to_xml().encode()),
        ArchiveEntry("experiments/main.sedml", COMBINE_URI+"sed-ml", SEDML.encode(), True),
    ), source)
    restored = CombineArchive.from_bytes(archive.to_bytes(), source=source)
    entry = restored.entry("../models/model.xml", relative_to="experiments/main.sedml")
    assert entry.location == "models/model.xml"
    with pytest.raises(ValueError):
        restored.entry("../../escape.xml", relative_to="experiments/main.sedml")


def test_export_uses_stored_bytes_when_inert_metadata_would_exceed_the_ratio_limit():
    original = example()
    metadata = ArchiveEntry("repeated.txt", "text/plain", b"a"*100000)
    original = CombineArchive((*original.entries, metadata), original.source)
    payload = original.to_bytes()
    restored = CombineArchive.from_bytes(payload, source=original.source)
    assert restored.entry("repeated.txt").data == metadata.data
