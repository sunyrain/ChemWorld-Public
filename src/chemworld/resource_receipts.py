"""Owned immutable receipts and an exact canonical-JSON hash prefix.

Only an unresolved receipt may be completed. Readers cannot mutate either
the event mapping, order view or nested receipt data. Scratch hash state is
reconstructed at restore and never enters a scientific snapshot.
"""

import hashlib
from collections.abc import Mapping, Sequence


class _Order(Sequence):
    def __init__(self, values):
        self.__values = values

    def __getitem__(self, key):
        result = self.__values[key]
        return tuple(result) if isinstance(key, slice) else result

    def __len__(self):
        return len(self.__values)


class ResourceReceipts(Mapping):
    def __init__(self, card, *, encode, freeze, thaw):
        self.__records = {}
        self.__ids = []
        self.order = _Order(self.__ids)
        self.__encode, self.__freeze, self.__thaw = encode, freeze, thaw
        self.__prefix = hashlib.sha256(b'{"card":' + encode(card) + b',"events":[')
        self.__sealed = 0

    def reset_card(self, card):
        self.__prefix = hashlib.sha256(b'{"card":' + self.__encode(card) + b',"events":[')
        self.__sealed = 0

    def __getitem__(self, key):
        return self.__records[key]

    def __iter__(self):
        return iter(self.__ids)

    def __len__(self):
        return len(self.__ids)

    def append(self, event):
        key = event["event_id"]
        if key in self.__records:
            raise ValueError("duplicate receipt")
        self.__records[key] = self.__freeze(event)
        self.__ids.append(key)

    def complete(self, key, outcome):
        event = self.__records[key]
        if event["outcome"] is not None:
            raise ValueError("completed receipt is immutable")
        self.__records[key] = self.__freeze({**event, "outcome": outcome})

    def canonical_hash(self, tail):
        # Incorporate completed leading records once. Pending/interleaved
        # records remain in the re-evaluated tail until they are immutable.
        while self.__sealed < len(self.__ids):
            event = self.__records[self.__ids[self.__sealed]]
            if event["outcome"] is None:
                break
            if self.__sealed:
                self.__prefix.update(b",")
            self.__prefix.update(self.__encode(self.__thaw(event)))
            self.__sealed += 1
        result = self.__prefix.copy()
        for index in range(self.__sealed, len(self.__ids)):
            if index:
                result.update(b",")
            result.update(self.__encode(self.__thaw(self.__records[self.__ids[index]])))
        result.update(b"]," + self.__encode(tail)[1:])
        return result.hexdigest()
