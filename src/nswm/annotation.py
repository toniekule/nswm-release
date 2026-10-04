"""Fixed-context labels, verified repairs and observation-scoped Gate targets."""
from copy import deepcopy
from dataclasses import asdict

from .repair import search_repairs
from .schema import Certificate, Query, Request, Verdict, commitment_from_dict, fingerprint
from .verify import verify


def annotate_event(event, critic_oracle, edit_catalogue, critic_checks=(), gate_checks=(),
                   gate_oracle=None, repair_budget=4096, max_edits=6, verifier_budget=64,
                   repair_cache=None, oracle_bindings=None):
    query = Query(**event["query"])
    commitment = commitment_from_dict(event["commitment"])
    request = Request(event["id"], query, commitment, commitment)
    if repair_cache is not None and sum(r.queries for r in repair_cache.values()) > repair_budget:
        raise ValueError("family repair cache exceeds its query budget")
    result = deepcopy(event)
    result["targets"], result["annotation"] = {}, {}
    for mode, oracle, checks in (("critic", critic_oracle, critic_checks), ("gate", gate_oracle, gate_checks)):
        checks = tuple(checks)
        proof = verify(request, checks, None, budget=verifier_budget)
        verdict = Verdict(oracle(query, commitment)) if oracle else Verdict.UNKNOWN
        if proof.cancelled:
            if verdict == Verdict.VALID:
                raise ValueError("oracle and independent proof disagree")
            verdict = Verdict.INVALID
        repair = None
        search = None
        if oracle is not None and verdict == Verdict.INVALID:
            if repair_cache is None:
                search = search_repairs(query, commitment, tuple(edit_catalogue), oracle,
                                        max_edits=max_edits, budget=repair_budget)
            else:
                from .repair import RepairResult
                if not oracle_bindings or not oracle_bindings.get(mode):
                    raise ValueError("repair memoization requires independent oracle source bindings")
                key = repair_key(query, commitment, edit_catalogue, mode, oracle_bindings[mode], max_edits)
                if key not in repair_cache:
                    remaining = repair_budget - sum(r.queries for r in repair_cache.values())
                    repair_cache[key] = (search_repairs(query, commitment, tuple(edit_catalogue), oracle,
                        max_edits=max_edits, budget=remaining) if remaining > 0 else RepairResult((), 0, False, 0))
                search = repair_cache[key]
            repair = search.repairs[0] if search.repairs else None
        elif verdict == Verdict.VALID:
            repair = ()
        if proof.cancelled:
            check = next(c for c in checks if c.check_id == proof.checked[-1])
            bound = proof.bound
            certificate = Certificate(verdict, dict(query.premises), check.constraint, check.entities,
                    check.time, {"residual": asdict(bound.residual), "unit": bound.unit}, bound.covered_scope, repair)
        else:
            certificate = Certificate(verdict, dict(query.premises) if verdict != Verdict.UNKNOWN else None, repair=repair)
        result["targets"][mode] = certificate.target()
        result["annotation"][mode] = {"query_binding": query.binding, "verification": proof.status,
            "checked": proof.checked, "repair_status": search.status if search else None,
            "repair_queries": search.queries if search else 0,
            "catalogue_exhaustive": search.exhaustive if search else None,
            "minimum_cardinality": search.minimum_cardinality if search else None,
            "complete_cardinalities": search.complete_cardinalities if search else (),
            "unresolved_cardinalities": search.unresolved_cardinalities if search else (),
            "minimal_repairs": [[asdict(e) for e in repair] for repair in search.repairs] if search else []}
    return result


def repair_key(query, commitment, catalogue, mode, source_binding, max_edits=6):
    return fingerprint({"query": query.binding, "commitment": asdict(commitment),
        "catalogue": [asdict(e) for e in catalogue], "mode": mode, "source": source_binding, "max_edits": max_edits})


def check_repair(query, commitment, edits, oracle):
    from itertools import combinations
    edits = tuple(edits)
    if len(edits) > 6 or len({e.slot for e in edits}) != len(edits):
        return "malformed"
    if Verdict(oracle(query, commitment.edited(edits))) != Verdict.VALID:
        return "not_validated"
    for size in range(len(edits)):
        for subset in combinations(edits, size):
            verdict = Verdict(oracle(query, commitment.edited(subset)))
            if verdict == Verdict.UNKNOWN:
                return "unresolved_minimality"
            if verdict == Verdict.VALID:
                return "redundant"
    return "verified_minimal"
