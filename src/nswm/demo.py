"""CPU episode exercising exclusion, branch coverage, dispatch and costs."""
from dataclasses import asdict

from .backends import AnalyticGenerator
from .planning import Cost, Prediction, run_episode
from .schema import Certificate, Commitment, Interval, Query, Request, Verdict
from .verify import scalar_inventory


def planning_demo(policy="certificate", budget=10.0):
    def propose(history):
        q = Query(f"history-{history}", ((1.0,),), {"capacity": "1", "complete_scalar_domain": True}, 16)
        requests = []
        for label, value in [("crossing", Interval.point("1.3")), ("stopping", Interval.point("0.8")),
                              ("residual", Interval("-10", "10"))]:
            z = Commitment({"demand": value})
            requests.append(Request(f"{history}-{label}", q, z, z, f"native-{history}"))
        return requests, Cost(0.1)

    def predict(request, prefix):
        violation = request.commitment.slots["demand"].lower > 1
        cert = Certificate(Verdict.INVALID if violation else Verdict.UNKNOWN,
                           constraint="contact_solidity", entities=("A", "B"), time=(0, 1))
        return Prediction(cert, float(violation), Cost(0.05 if prefix == "judgment" else 0.1))

    def inventory(request):
        return scalar_inventory(request.query, request.commitment, {"capacity": Interval.point(1)},
               [{"constraint": "contact_solidity", "slot": "demand", "limit": "capacity", "premises_valid": True}])

    episode = run_episode(0, propose, predict, inventory, AnalyticGenerator(),
                          lambda request, future: (1.0, True, Cost(0.02)),
                          lambda history, action, future: (history + 1, Cost()),
                          lambda history: "success" if history >= 2 else "running",
                          policy=policy, budget=budget, verifier_budget=16,
                          preview=lambda request: (float(request.commitment.slots["demand"].lower > 1), Cost(0.2, nfe=4)))
    return asdict(episode)
