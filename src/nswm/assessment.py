"""Independent physical witness and fixed-context repair assessment."""
from .annotation import check_repair
from .schema import Edit, Interval, Query, Request, Verdict, commitment_from_dict
from .targets import parse_certificate
from .verify import verify


def temporal_iou(left, right):
    if left is None or right is None:
        return 0.0
    intersection = max(0.0, min(left[1], right[1]) - max(left[0], right[0]))
    union = max(left[1], right[1]) - min(left[0], right[0])
    return intersection / union if union > 0 else float(tuple(left) == tuple(right))


def assess_prediction(row, mode, certificate, root, context=None):
    from .sdg import load_context
    from .evidence import FutureOracle, HistoryOracle, load_future
    if row.get("evidence_kind") not in {"mujoco_controlled_domain", "mujoco_mechanisms_v3"}:
        return {"status": "unavailable", "reason": "no registered physical replay assessor"}
    version = row.get("construction", {}).get("evidence_version")
    if version not in {2, 3}:
        return {"status": "unavailable", "reason": "physical evidence has no separate history and future sources"}
    context = context or load_context(row, root)
    query = Query(**row["query"])
    commitment = commitment_from_dict(row["commitment"])
    if version == 3:
        from .counterfactual import MechanismHistoryOracle, MechanismFutureOracle
        from .corpus_v3 import load_evidence
        from .schema import fingerprint
        loaded, future = load_evidence(row, root, replay=False)
        if fingerprint(loaded) != fingerprint(context):
            raise ValueError("cached replay context differs from event context")
        HistoryOracle = lambda q, category: MechanismHistoryOracle(q)
        FutureOracle = MechanismFutureOracle
        load_future = lambda row, root, context: future
    if mode == "gate":
        oracle = HistoryOracle(query, row["category"])
        source = "permitted_history"
    elif mode == "critic":
        oracle = FutureOracle(context, commitment, load_future(row, root, context))
        source = "measured_candidate_future"
    else:
        raise ValueError("unknown assessment mode")
    truth = oracle(query, commitment)
    inventory = oracle.checks(query, commitment)
    request = Request(row["id"], query, commitment, commitment)
    proof = verify(request, inventory, None, 16)
    eligible = truth == Verdict.INVALID and any(c.evaluate().residual is not None and c.evaluate().residual.lower > 0 for c in inventory)
    witness = False
    constraint = entity = time = False
    if certificate.judgment == Verdict.INVALID:
        for check in inventory:
            bound = check.evaluate()
            constraint = certificate.constraint == check.constraint
            entity = (len(certificate.entities or ()) == len(check.entities)
                      and set(certificate.entities or ()) == set(check.entities))
            time = temporal_iou(certificate.time, check.time) >= 0.5
            try:
                residual = certificate.witness["residual"]
                claimed = Interval(**residual) if isinstance(residual, dict) else Interval.point(residual)
                valid = (certificate.witness["unit"] == bound.unit and claimed.lower > 0
                         and bound.residual is not None and claimed.contains(bound.residual))
            except (ValueError, KeyError, TypeError):
                valid = False
            witness = constraint and entity and time and valid and bound.premises_valid
            if witness:
                break
    status = "not_emitted"
    if certificate.judgment == Verdict.INVALID and certificate.repair is not None:
        catalogue = {Edit(e["slot"], Interval(**e["value"])) for e in row["construction"]["edit_catalogue"]}
        if any(edit not in catalogue for edit in certificate.repair):
            status = "inadmissible"
        else:
            try:
                status = check_repair(query, commitment, certificate.repair, oracle)
            except (ValueError, KeyError):
                status = "malformed"
    repaired = status in {"verified_minimal", "redundant", "unresolved_minimality"}
    return {"status": "assessed", "assessor": "mujoco_mechanisms_v3" if version == 3 else "mujoco_controlled_domain_v2", "label_source": source,
        "context_sha256": row["construction"]["context_sha256"], "truth": truth.value,
        "eligible": eligible, "constraint_valid": constraint, "entities_valid": entity,
        "time_valid": time, "witness_valid": witness, "repair_status": status,
        "repair_valid": repaired, "minimal": status == "verified_minimal",
        "verification": proof.status, "cancelled": proof.cancelled,
        "verification_role": "arm_invariant_inventory_diagnostic", "verification_budget": 16}
