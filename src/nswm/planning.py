"""Request admission, scope-aware cancellation, and complete episode accounting."""
from dataclasses import asdict, dataclass, field
import math
import random
import time
from typing import Callable

from .schema import Certificate, Commitment, Request, scope_contains
from .verify import Check, verify


@dataclass(frozen=True)
class Cost:
    compute: float = 0.0
    wall_seconds: float = 0.0
    nfe: int = 0
    unit: str = "work_units"

    def __post_init__(self):
        if (not math.isfinite(self.compute) or not math.isfinite(self.wall_seconds) or self.compute < 0
                or self.wall_seconds < 0 or not isinstance(self.nfe, int) or self.nfe < 0
                or not isinstance(self.unit, str) or not self.unit):
            raise ValueError("cost must be finite and nonnegative")


class BackendFailure(RuntimeError):
    def __init__(self, message, cost=None, wall_seconds=0.0, job_id=""):
        super().__init__(message)
        self.cost, self.wall_seconds, self.job_id = cost, wall_seconds, job_id


@dataclass
class Ledger:
    budget: float
    unit: str = "work_units"
    compute: float = 0.0
    wall_seconds: float = 0.0
    nfe: int = 0
    entries: list[dict] = field(default_factory=list)
    accounting_complete: bool = True

    def fits(self, amount):
        return amount >= 0 and self.compute + amount <= self.budget

    def charge(self, stage, cost, **details):
        if cost.unit != self.unit:
            raise ValueError("ledger cost units differ")
        self.compute += cost.compute
        self.wall_seconds += cost.wall_seconds
        self.nfe += cost.nfe
        self.entries.append({"stage": stage, **asdict(cost), **details,
                             "cumulative_compute": self.compute,
                             "overrun": max(0.0, self.compute - self.budget)})


@dataclass(frozen=True)
class Prediction:
    certificate: Certificate
    risk: float
    cost: Cost


@dataclass(frozen=True)
class Future:
    job_id: str
    request_binding: str
    commitment: Commitment | None
    payload: object
    cost: Cost
    completed: bool = True


@dataclass(frozen=True)
class Episode:
    status: str
    rounds: int
    history: object
    ledger: Ledger


POLICIES = {"full", "random", "score", "verdict", "discard", "search", "certificate", "preview"}


def run_episode(history, propose, predict, inventory, generate, evaluate, execute, terminal,
                policy="certificate", budget=100.0, unit="work_units", verifier_budget=16,
                max_rounds=64, proposal_reserve=0.1, prediction_reserve=0.1,
                check_reserve=0.01, generation_reserve=1.0, threshold=0.5,
                random_rate=0.3, seed=0, preview=None, verification_cost=None):
    if policy not in POLICIES or not math.isfinite(budget) or budget < 0 or not 0 <= random_rate <= 1:
        raise ValueError("invalid policy or budget")
    reserves = (proposal_reserve, prediction_reserve, check_reserve, generation_reserve)
    if any(not math.isfinite(r) or r < 0 for r in reserves) or verifier_budget < 0 or max_rounds < 1:
        raise ValueError("invalid reservations or iteration limits")
    if policy == "preview" and preview is None:
        raise ValueError("preview policy requires a preview backend")
    ledger = Ledger(budget, unit)
    rng = random.Random(seed)
    if terminal(history) == "success":
        return Episode("success", 0, history, ledger)
    for round_index in range(max_rounds):
        if not ledger.fits(proposal_reserve):
            return Episode("timeout", round_index, history, ledger)
        requests, cost = propose(history)
        ledger.charge("proposal", cost, round=round_index)
        if len({r.request_id for r in requests}) != len(requests):
            raise ValueError("duplicate request IDs")
        retained, cancellation_count = [], 0
        random_cancel = set(rng.sample(range(len(requests)), round(len(requests) * random_rate))) if policy == "random" else set()
        for i, request in enumerate(requests):
            prediction = None
            if policy in {"score", "verdict", "discard", "certificate"} and ledger.fits(prediction_reserve):
                prediction = predict(request, "judgment" if policy == "verdict" else "complete")
                ledger.charge("prediction", prediction.cost, request=request.request_id, binding=request.binding)
            cancelled = i in random_cancel
            reason = "random" if cancelled else "retained"
            if policy in {"score", "verdict", "discard"} and prediction and prediction.risk >= threshold:
                cancelled, reason = True, "learned_threshold"
            elif policy == "preview" and preview and ledger.fits(prediction_reserve):
                risk, preview_cost = preview(request)
                ledger.charge("preview", preview_cost, request=request.request_id, binding=request.binding)
                cancelled, reason = risk >= threshold, "preview_threshold" if risk >= threshold else "retained"
            elif policy in {"certificate", "search"}:
                checks = inventory(request)
                available = verifier_budget if check_reserve == 0 else min(verifier_budget,
                        max(0, int((budget - ledger.compute) / check_reserve)))
                t0 = time.perf_counter()
                result = verify(request, checks, prediction.certificate if prediction else None, available)
                elapsed = time.perf_counter() - t0
                measured = verification_cost(elapsed, len(result.checked)) if verification_cost else Cost(
                        elapsed if unit == "wall_seconds" else len(result.checked) * check_reserve
                        if unit == "work_units" else 0.0, elapsed, unit=unit)
                ledger.charge("verification", measured, request=request.request_id,
                               binding=request.binding, status=result.status, checked=result.checked,
                               scope=asdict(result.bound.covered_scope) if result.bound and result.bound.covered_scope else None)
                cancelled, reason = result.cancelled, result.status
            ledger.entries.append({"stage": "request", "round": round_index, "request": request.request_id,
                                   "parent_id": request.parent_id, "binding": request.binding,
                                   "cancelled": cancelled, "reason": reason})
            if cancelled:
                cancellation_count += 1
            else:
                retained.append(request)
        completed = []
        unfinished = False
        for request in retained:
            if not ledger.fits(generation_reserve):
                unfinished = True
                break
            try:
                future = generate(request)
            except BackendFailure as failure:
                if failure.cost is None:
                    ledger.accounting_complete = False
                    ledger.charge("generation", Cost(wall_seconds=failure.wall_seconds, unit=unit),
                                  request=request.request_id, binding=request.binding, job=failure.job_id,
                                  completed=False, compute_known=False, error=str(failure))
                    return Episode("backend_error", round_index + 1, history, ledger)
                ledger.charge("generation", failure.cost, request=request.request_id,
                              binding=request.binding, job=failure.job_id, completed=False,
                              compute_known=True, error=str(failure))
                unfinished = True
                continue
            ledger.charge("generation", future.cost, request=request.request_id, binding=request.binding,
                          job=future.job_id, completed=future.completed)
            if future.request_binding != request.binding:
                raise ValueError("generator returned a different request identity")
            if not future.completed:
                unfinished = True
                continue
            accepted = future.commitment is not None and scope_contains(request.acceptance, future.commitment)
            utility, admissible, eval_cost = evaluate(request, future) if accepted else (-math.inf, False, Cost(unit=unit))
            ledger.charge("acceptance_scoring", eval_cost, request=request.request_id, accepted=accepted, admissible=admissible)
            if accepted and admissible and math.isfinite(utility):
                completed.append((utility, len(completed), request, future))
            if ledger.compute > budget:
                unfinished = True
                break
        if completed:
            _, _, request, future = max(completed, key=lambda x: (x[0], -x[1]))
            history, exec_cost = execute(history, request.query.actions[0], future)
            ledger.charge("execution", exec_cost, request=request.request_id)
            state = terminal(history)
            if state in {"success", "failure"}:
                return Episode(state, round_index + 1, history, ledger)
        elif cancellation_count == len(requests) or (not unfinished and retained):
            return Episode("failure", round_index + 1, history, ledger)
        else:
            return Episode("timeout", round_index + 1, history, ledger)
    return Episode("timeout", max_rounds, history, ledger)
