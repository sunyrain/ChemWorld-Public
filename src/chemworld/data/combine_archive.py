"""Data-only bounded OMEX v1 reader/writer; never extracts or executes entries."""

import hashlib
import stat
from dataclasses import dataclass
from io import BytesIO
from pathlib import PurePosixPath
from xml.etree import ElementTree as ET
from zipfile import ZIP_DEFLATED, ZIP_STORED, BadZipFile, ZipFile, ZipInfo
from zlib import compress

from chemworld.data.scientific_inputs import InputSource
from chemworld.data.sedml_declarations import SEDMLDocument, boolean
from chemworld.data.sedml_xml import local_archive_path
from chemworld.data.xml_inputs import bounded_xml
from chemworld.physchem.data_conflicts import DataSourceProvenance

MANIFEST_NAMESPACE = "http://identifiers.org/combine.specifications/omex-manifest"
COMBINE_URI = "http://identifiers.org/combine.specifications/"
MAX_ENTRY_BYTES = 20*1024*1024
MAX_TOTAL_BYTES = 64*1024*1024
MAX_ARCHIVE_BYTES = 32*1024*1024
MAX_ENTRIES = 256
MAX_COMPRESSION_RATIO = 200
_EXECUTABLE = {".py", ".pyc", ".exe", ".dll", ".js", ".bat", ".cmd", ".ps1", ".com",
               ".scr", ".sh", ".so", ".dylib", ".vbs", ".jar", ".ipynb", ".wasm"}
_FORMATS = {
    "sbml": "sbml", "sbml.level-3.version-2": "sbml", "sed-ml": "sedml",
    "sed-ml.level-1.version-4": "sedml", "omex-metadata": "metadata",
}
_MEDIA = {"application/pdf", "application/rdf+xml", "image/png", "image/jpeg", "text/csv",
          "text/plain"}


def format_kind(value):
    if not isinstance(value, str):
        raise ValueError("COMBINE content format must be a declared string URI")
    for prefix in (COMBINE_URI, COMBINE_URI.replace("http:", "https:")):
        if value.startswith(prefix):
            if value[len(prefix):] in _FORMATS:
                return _FORMATS[value[len(prefix):]]
            raise ValueError("unsupported COMBINE format/version")
    prefix = "http://purl.org/NET/mediatypes/"
    media = value[len(prefix):] if value.startswith(prefix) else value
    if media in _MEDIA:
        return "metadata"
    raise ValueError("unknown or executable COMBINE content format")


@dataclass(frozen=True)
class ArchiveEntry:
    location: str
    format: str
    data: bytes
    master: bool = False

    def __post_init__(self):
        if (local_archive_path(self.location) != self.location or self.location == "manifest.xml"
                or len(self.location) > 1024
                or PurePosixPath(self.location).suffix.casefold() in _EXECUTABLE
                or not isinstance(self.data, bytes) or len(self.data) > MAX_ENTRY_BYTES
                or type(self.master) is not bool or not isinstance(self.format, str)):
            raise ValueError("invalid or executable COMBINE entry")
        kind = format_kind(self.format)
        if self.master and kind != "sedml":
            raise ValueError("this execution profile requires a SED-ML master experiment")


@dataclass(frozen=True)
class CombineArchive:
    entries: tuple[ArchiveEntry, ...]
    source: InputSource

    def __post_init__(self):
        entries = tuple(self.entries)
        if (not isinstance(self.source, InputSource) or not 1 <= len(entries) < MAX_ENTRIES
                or any(not isinstance(e, ArchiveEntry) for e in entries)
                or len({e.location.casefold() for e in entries}) != len(entries)
                or sum(len(e.data) for e in entries) > MAX_TOTAL_BYTES):
            raise ValueError("COMBINE entries/provenance exceed the closed archive contract")
        masters = [e for e in entries if e.master]
        experiments = [e for e in entries if format_kind(e.format) == "sedml"]
        if len(masters) > 1 or (not masters and len(experiments) != 1):
            raise ValueError("COMBINE needs one unambiguous SED-ML master")
        object.__setattr__(self, "entries", entries)

    @property
    def master_location(self):
        masters = [e.location for e in self.entries if e.master]
        return masters[0] if masters else next(e.location for e in self.entries
                                              if format_kind(e.format) == "sedml")

    def source_for(self, location):
        entry = self.entry(location)
        parent = self.source.provenance
        return InputSource(DataSourceProvenance(parent.source_id+":"+location, parent.priority,
            parent.citation+"; archive entry "+location, parent.version, parent.license,
            hashlib.sha256(entry.data).hexdigest()), self.source.data_kind)

    def entry(self, location, *, relative_to=None):
        base = "" if relative_to is None else str(PurePosixPath(relative_to).parent)
        if base == ".":
            base = ""
        path = local_archive_path(location, base=base)
        matches = [e for e in self.entries if e.location == path]
        if len(matches) != 1:
            raise ValueError("archive does not contain the declared local resource")
        return matches[0]

    def document(self, location=None):
        location = self.master_location if location is None else location
        entry = self.entry(location)
        if format_kind(entry.format) != "sedml":
            raise ValueError("archive resource is not a declared SED-ML experiment")
        return SEDMLDocument(entry.data.decode("utf-8"), self.source_for(entry.location))

    @classmethod
    def from_bytes(cls, data, *, source):
        if (not isinstance(data, bytes) or not data.startswith(b"PK\x03\x04")
                or len(data) > MAX_ARCHIVE_BYTES):
            raise ValueError("COMBINE requires bounded ZIP bytes")
        try:
            with ZipFile(BytesIO(data)) as archive:
                infos = archive.infolist()
                if len(infos) > MAX_ENTRIES:
                    raise ValueError("COMBINE exceeds its ZIP entry budget")
                files, aliases, total = {}, set(), 0
                for info in infos:
                    path = local_archive_path(info.filename)
                    mode = info.external_attr >> 16
                    if (path.casefold() in aliases or info.flag_bits & 1 or stat.S_ISLNK(mode)
                            or (stat.S_IFMT(mode) not in {0, stat.S_IFREG, stat.S_IFDIR})
                            or info.file_size > MAX_ENTRY_BYTES or info.compress_size < 0
                            or info.file_size > MAX_COMPRESSION_RATIO*max(1, info.compress_size)):
                        raise ValueError(
                            "duplicate, encrypted, special or overcompressed ZIP entry")
                    aliases.add(path.casefold())
                    total += info.file_size
                    if total > MAX_TOTAL_BYTES:
                        raise ValueError("COMBINE exceeds its total decoded byte budget")
                    if info.is_dir():
                        continue
                    with archive.open(info) as stream:
                        decoded = stream.read(MAX_ENTRY_BYTES+1)
                    if len(decoded) != info.file_size or len(decoded) > MAX_ENTRY_BYTES:
                        raise ValueError("ZIP entry size differs from its bounded declaration")
                    files[path] = decoded
            if "manifest.xml" not in files:
                raise ValueError("COMBINE requires a root manifest.xml")
            manifest = bounded_xml(files["manifest.xml"].decode("utf-8"),
                max_bytes=262144, max_nodes=MAX_ENTRIES+4, max_depth=4)
            if manifest.tag != "{"+MANIFEST_NAMESPACE+"}omexManifest" or manifest.attrib:
                raise ValueError("unknown COMBINE manifest root or attributes")
            if (manifest.text or "").strip():
                raise ValueError("unexpected manifest text")
            declared, entries = {}, []
            for item in manifest:
                if (item.tag != "{"+MANIFEST_NAMESPACE+"}content" or len(item)
                        or set(item.attrib)-{"location", "format", "master"}
                        or not {"location", "format"} <= item.attrib.keys()
                        or (item.text or "").strip() or (item.tail or "").strip()):
                    raise ValueError("unknown COMBINE content declaration")
                path = local_archive_path(item.get("location"), archive_self=True)
                if path.casefold() in declared:
                    raise ValueError("duplicate manifest location or portable-path alias")
                declared[path.casefold()] = path
                master = boolean(item.get("master", "false"))
                expected = COMBINE_URI+("omex" if path == "." else "omex-manifest")
                if path in {".", "manifest.xml"}:
                    if item.get("format") != expected or master:
                        raise ValueError("archive/manifest self declaration differs from OMEX")
                    continue
                if path not in files:
                    raise ValueError("manifest refers to a missing archive entry")
                entries.append(ArchiveEntry(path, item.get("format"), files[path], master))
            if "." not in declared or set(files)-{"manifest.xml"} != {e.location for e in entries}:
                raise ValueError("manifest must describe the archive and all actual data files")
            return cls(tuple(entries), source)
        except (BadZipFile, UnicodeError, RuntimeError, NotImplementedError) as exc:
            raise ValueError("invalid or unsupported COMBINE ZIP payload") from exc

    def to_bytes(self):
        manifest = ET.Element("{"+MANIFEST_NAMESPACE+"}omexManifest")
        ET.SubElement(manifest, "{"+MANIFEST_NAMESPACE+"}content",
                      location=".", format=COMBINE_URI+"omex")
        payload = {}
        for entry in self.entries:
            attributes = {"location": entry.location, "format": entry.format}
            if entry.master:
                attributes["master"] = "true"
            ET.SubElement(manifest, "{"+MANIFEST_NAMESPACE+"}content", attributes)
            payload[entry.location] = entry.data
        payload["manifest.xml"] = ET.tostring(manifest, encoding="utf-8", xml_declaration=True)
        output = BytesIO()
        with ZipFile(output, "w", compression=ZIP_DEFLATED, compresslevel=9) as archive:
            for path, content in sorted(payload.items()):
                info = ZipInfo(path, date_time=(1980, 1, 1, 0, 0, 0))
                compressed_size = len(compress(content, level=9, wbits=-15))
                info.compress_type = (ZIP_STORED if len(content) >
                    MAX_COMPRESSION_RATIO*max(1, compressed_size) else ZIP_DEFLATED)
                info.create_system, info.external_attr = 3, (stat.S_IFREG | 0o644) << 16
                archive.writestr(info, content)
        data = output.getvalue()
        if len(data) > MAX_ARCHIVE_BYTES:
            raise ValueError("exported COMBINE exceeds the compressed archive budget")
        # Enforce the same decoder limits, including compression ratio, on export.
        self.from_bytes(data, source=self.source)
        return data
