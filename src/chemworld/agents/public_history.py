"""Read a bounded page of the host-published public operation stream."""

import json
from pathlib import Path


def public_history_page(
    public: Path, *, limit: int = 5, offset: int | None = None, max_bytes: int = 32_640
) -> dict[str, object]:
    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes < 768:
        raise ValueError("max_bytes must be an integer of at least 768")
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise ValueError("limit must be a positive integer")
    if offset is not None and (
        isinstance(offset, bool) or not isinstance(offset, int) or offset < 0
    ):
        raise ValueError("offset must be a non-negative integer")
    limit = min(limit, 10)
    path = public / "operation_history.jsonl"
    if not path.is_file():
        raise ValueError(
            "complete public history is unavailable; use the original workspace runtime"
        )
    events = []
    total = 0
    if path.exists():
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                if offset is None or offset <= total < offset + limit:
                    events.append(json.loads(line))
                    if len(events) > limit:
                        events.pop(0)
                total += 1
    start = max(0, total - len(events)) if offset is None else min(offset, total)
    cache_path = public / "history.jsonl"
    cached_count = 0
    if cache_path.exists():
        with cache_path.open(encoding="utf-8") as handle:
            cached_count = sum(bool(line.strip()) for line in handle)
    result: dict[str, object] = {
        "authoritative": False,
        "source": "public/operation_history.jsonl",
        "events": events,
        "offset": start,
        "next_offset": None,
        "total_event_count": total,
        "omitted_event_count": total - len(events),
        "truncated": len(events) < total,
        "cache_retained_event_count": cached_count,
        "cache_truncated": cached_count < total,
    }
    while True:
        next_offset = start + len(events)
        result.update(
            offset=start,
            next_offset=next_offset if next_offset < total else None,
            omitted_event_count=total - len(events),
            truncated=len(events) < total,
        )
        size = len(json.dumps(result, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
        if size <= max_bytes:
            return result
        if not events:
            raise ValueError("history metadata exceeds the tool response byte budget")
        if len(events) <= 1:
            # Preserve the event on disk and give a precise direct-read reference.
            result["oversized_event"] = {
                "offset": start,
                "message": "Read this event from source; it exceeds the tool response byte budget.",
            }
            events.clear()
        elif offset is None:
            events.pop(0)
            start += 1
        else:
            events.pop()
