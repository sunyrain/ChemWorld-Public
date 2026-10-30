"""Bounded, local declarative XML; no DTD, entity or external resource loading."""

import re
from io import StringIO
from typing import Protocol, cast
from xml.etree import ElementTree as ET


class _CompletedXML(Protocol):
    root: ET.Element


def bounded_xml(
    text: str, *, max_bytes: int = 20*1024*1024, max_nodes: int = 100000, max_depth: int = 128,
) -> ET.Element:
    if not isinstance(text, str) or not text or len(text.encode("utf-8")) > max_bytes:
        raise ValueError("XML requires nonempty text within the declared byte limit")
    if re.search(r"<!\s*(DOCTYPE|ENTITY)\b", text, re.IGNORECASE):
        raise ValueError("XML DTD and entity declarations are forbidden")
    declaration = re.match(r"\ufeff?\s*<\?xml\s+([^?]*)\?>", text)
    if declaration:
        encoding = re.search(r"encoding\s*=\s*['\"]([^'\"]+)['\"]", declaration[1])
        if encoding and encoding[1].lower() != "utf-8":
            raise ValueError("declarative XML must use UTF-8")
    try:
        parser = ET.iterparse(StringIO(text), events=("start", "end"))
        count, depth = 0, -1
        for event, _node in parser:
            if event == "start":
                count += 1
                depth += 1
                if count > max_nodes or depth > max_depth:
                    raise ValueError("XML exceeds declared node/depth bounds")
            else:
                depth -= 1
    except (ET.ParseError, RecursionError) as exc:
        raise ValueError("invalid declarative XML") from exc
    # iterparse supplies root after exhaustion; some supported stubs omit it.
    return cast(_CompletedXML, parser).root
