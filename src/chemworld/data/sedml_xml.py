"""Bounded declarative experiment XML, scoped prefixes and closed local paths."""

import re
import unicodedata
from dataclasses import dataclass, field
from io import StringIO
from types import MappingProxyType
from urllib.parse import unquote, urlsplit
from xml.etree import ElementTree as ET

from chemworld.data.xml_inputs import bounded_xml

_NAME = r"[A-Za-z_][A-Za-z0-9_.-]*"
_STEP = re.compile(r"("+_NAME+r"):("+_NAME+r")(?:\[@id=(['\"])("+_NAME+r")\3\])?\Z")
_ATTRIBUTE = re.compile(r"@("+_NAME+r")\Z")
_RESERVED = {"con", "prn", "aux", "nul", *(f"com{i}" for i in range(1, 10)),
             *(f"lpt{i}" for i in range(1, 10))}


@dataclass(frozen=True)
class ExperimentXMLNode:
    tag: str
    attributes: object
    namespaces: object
    children: tuple = ()
    text: str = ""
    tail: str = field(default="", repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "attributes", MappingProxyType(dict(self.attributes)))
        object.__setattr__(self, "namespaces", MappingProxyType(dict(self.namespaces)))
        object.__setattr__(self, "children", tuple(self.children))

    @property
    def local_name(self):
        return self.tag.rsplit("}", 1)[-1]

    def element(self):
        """A fresh XML view for the existing MathML parser, never owned state."""
        result = ET.Element(self.tag, dict(self.attributes))
        result.text, result.tail = self.text, self.tail
        result.extend(child.element() for child in self.children)
        return result


def experiment_xml(text):
    # Preflight precedes a second, namespace-aware parse. Both passes are pure;
    # the common SBML/MathML reader and its original default path stay unchanged.
    bounded_xml(text, max_bytes=4*1024*1024, max_nodes=20000, max_depth=64)
    pending, stack, declarations = {}, [], 0
    for event, item in ET.iterparse(StringIO(text), events=("start-ns", "start", "end")):
        if event == "start-ns":
            prefix, namespace = item
            pending[prefix] = namespace
            declarations += 1
            if declarations > 4096:
                raise ValueError("experiment XML exceeds its namespace declaration budget")
        elif event == "start":
            scope = {"xml": "http://www.w3.org/XML/1998/namespace"} if not stack else dict(
                stack[-1][1])
            scope.update(pending)
            pending.clear()
            if len(scope) > 32:
                raise ValueError("experiment XML exceeds its namespace prefix budget")
            stack.append((item, scope, []))
        else:
            element, scope, children = stack.pop()
            node = ExperimentXMLNode(element.tag, element.attrib, scope, tuple(children),
                                     element.text or "", element.tail or "")
            if stack:
                stack[-1][2].append(node)
            else:
                return node
    raise ValueError("empty declarative experiment")


def local_archive_path(location, *, base="", archive_self=False):
    """Resolve a local URI without filesystem access or platform-dependent aliases.

    ``base`` is the directory of an already validated archive document. Relative
    parent references may stay inside that archive; ZIP entry names use no base.
    """
    if (not isinstance(location, str) or not location or location != location.strip()
            or "\\" in location or any(ord(c) < 32 or ord(c) == 127 for c in location)):
        raise ValueError("archive location must be a safe local URI")
    uri = urlsplit(location)
    if uri.scheme or uri.netloc or uri.query or uri.fragment:
        raise ValueError("external or qualified archive resources are not supported")
    path = unquote(uri.path, errors="strict")
    if (path.startswith("/") or "\\" in path or ":" in path or "%" in path
            or unicodedata.normalize("NFC", path) != path
            or any(ord(c) < 32 or ord(c) == 127 for c in path)):
        raise ValueError("unsafe or aliased archive path")
    if base and local_archive_path(base) != base:
        raise ValueError("archive base must be a canonical local directory")
    parts = base.split("/") if base else []
    for part in path.split("/"):
        if part in {"", "."}:
            continue
        if part == "..":
            if not parts:
                raise ValueError("archive reference escapes its root")
            parts.pop()
        else:
            if part.rstrip(" .") != part or part.split(".")[0].casefold() in _RESERVED:
                raise ValueError("archive path contains a platform-dependent alias")
            parts.append(part)
    if not parts:
        if archive_self and location in {".", "./"}:
            return "."
        raise ValueError("archive resource must name a file")
    return "/".join(parts)


def closed_xpath(root, target, namespaces, *, require_attribute=False):
    """Unique absolute child XPath, optional SId predicate and final attribute.

    No wildcard, descendant axis, function, arbitrary predicate or XPath engine
    is installed. Caller decides which matched model quantities may be changed.
    """
    if not isinstance(target, str) or len(target) > 2048 or not target.startswith("/"):
        raise ValueError("model target needs a bounded absolute child XPath")
    steps = target[1:].split("/")
    attribute = _ATTRIBUTE.fullmatch(steps[-1])
    attribute = attribute[1] if attribute is not None else None
    if attribute is not None:
        steps.pop()
    if not steps or len(steps) > 8 or (require_attribute and attribute is None):
        raise ValueError("unsupported model target shape")
    selected = [root]
    for index, step in enumerate(steps):
        parsed = _STEP.fullmatch(step)
        if parsed is None or parsed[1] not in namespaces:
            raise ValueError("unsupported XPath step or unbound namespace prefix")
        name = "{"+namespaces[parsed[1]]+"}"+parsed[2]
        candidates = selected if index == 0 else [child for parent in selected for child in parent]
        selected = [node for node in candidates if node.tag == name
                    and (parsed[4] is None or node.get("id") == parsed[4])]
        if len(selected) > 256:
            raise ValueError("model target exceeds its match budget")
    if len(selected) != 1 or (attribute is not None and attribute not in selected[0].attrib):
        raise ValueError("model target must select exactly one existing element or attribute")
    return selected[0], attribute
