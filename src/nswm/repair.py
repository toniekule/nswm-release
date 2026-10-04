"""Exhaustive inclusion-minimal repair search for nonmonotone oracles."""
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations, product
from typing import Callable

from .schema import Commitment, Edit, Query, Verdict, canonical


@dataclass(frozen=True)
class RepairResult:
    repairs: tuple[tuple[Edit, ...], ...]
    queries: int
    exhaustive: bool
    unknown_queries: int
    complete_cardinalities: tuple[int, ...] = ()
    unresolved_cardinalities: tuple[int, ...] = ()

    @property
    def minimum_cardinality(self):
        if not self.repairs:
            return None
        smallest = min(map(len, self.repairs))
        if all(n in self.complete_cardinalities and n not in self.unresolved_cardinalities
               for n in range(smallest)):
            return smallest
        return None

    @property
    def status(self):
        if self.repairs:
            return "verified_minimal"
        return "repair_unavailable" if self.exhaustive and not self.unknown_queries else "unknown"


def search_repairs(query: Query, commitment: Commitment, catalogue: tuple[Edit, ...],
                   oracle: Callable[[Query, Commitment], Verdict], max_edits: int = 6,
                   budget: int = 4096) -> RepairResult:
    if not 1 <= max_edits <= 6 or budget < 1:
        raise ValueError("repair search allows 1..6 edits and a positive query budget")
    # Canonicalize before querying. Duplicate slots in a set are not executable.
    catalogue = tuple(sorted(set(catalogue), key=lambda e: (e.slot, e.value.lo, e.value.hi)))
    for edit in catalogue:
        if edit.slot not in commitment.slots:
            raise ValueError("catalogue references an unknown slot")
    catalogue = tuple(e for e in catalogue if e.value != commitment.slots[e.slot])
    cache = {}
    exhausted = False

    def check(edits):
        nonlocal exhausted
        edited = commitment.edited(tuple(edits))
        key = canonical({k: [v.lo, v.hi] for k, v in edited.slots.items()})
        if key not in cache:
            if len(cache) >= budget:
                exhausted = True
                return Verdict.UNKNOWN
            cache[key] = Verdict(oracle(query, edited))
        return cache[key]

    original = check(())
    if original == Verdict.VALID:
        return RepairResult(((),), len(cache), True, 0, (0,), ())
    if original == Verdict.UNKNOWN:
        return RepairResult((), len(cache), False, 1)
    found = []
    complete, unresolved = [0], []
    by_slot = defaultdict(list)
    for edit in catalogue:
        by_slot[edit.slot].append(edit)
    groups = tuple(by_slot.values())
    for size in range(1, min(max_edits, len(groups)) + 1):
        candidates = (edits for selected in combinations(groups, size) for edits in product(*selected))
        for edits in candidates:
            verdict = check(edits)
            if verdict == Verdict.UNKNOWN and size not in unresolved:
                unresolved.append(size)
            if verdict != Verdict.VALID:
                if exhausted:
                    break
                continue
            # Test ALL proper subsets, not just removing one edit. With
            # nonmonotone validity, a distant smaller subset can also be valid.
            minimal = True
            for n in range(size):
                for subset in combinations(edits, n):
                    if check(subset) != Verdict.INVALID:
                        minimal = False
            if minimal:
                found.append(tuple(edits))
            if exhausted:
                break
        if exhausted:
            break
        complete.append(size)
    return RepairResult(tuple(found), len(cache), not exhausted,
                        sum(v == Verdict.UNKNOWN for v in cache.values()), tuple(complete), tuple(unresolved))
