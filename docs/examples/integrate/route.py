"""Route a support message with Jeb, then act, confirm or escalate on the model's confidence."""
import requests

JEB = "http://localhost:8100/v1/systemone"   # jeb serve; for AINode use http://<node>:3000/v1/systemone
TEAMS = {"billing": "charges, invoices, refunds", "technical": "bugs, errors, outages",
         "account": "login, password, cancelling", "sales": "pricing, upgrades"}


def route(message: str) -> tuple[str, str, float]:
    r = requests.post(JEB, timeout=30, json={
        "state": {"message": message},
        "questions": {"team": {"type": "choice", "instructions": "Which team should handle this message?",
                               "criteria": TEAMS}}})
    r.raise_for_status()
    answer = r.json()["answers"]["team"]
    team, p = answer["choice"], answer["probabilities"][answer["choice"]]
    if p >= 0.90:
        return "act", team, p          # route it automatically
    if p >= 0.60:
        return "confirm", team, p      # route it, and flag it for a quick human check
    return "escalate", team, p         # too close to call: a person picks the team


if __name__ == "__main__":
    for msg in ["Hi, I was charged twice for my March subscription. Can you refund the duplicate?",
                "The export button does nothing and I can't log in on the mobile app either."]:
        action, team, p = route(msg)
        print(f"{action:9} {team:10} {p:.3f}  {msg[:60]}")
