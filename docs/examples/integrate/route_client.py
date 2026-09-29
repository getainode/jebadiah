"""The same routing with the jebadiah-decide package and no server: it talks to Ollama directly."""
from jebadiah_decide import Jeb

jeb = Jeb("ollama")      # Jebadiah 9B v2; Jeb("lmstudio"), Jeb("llama-server"), Jeb("mlx") work the same way
jeb.prepare()            # checks Ollama and pulls the model the first time
out = jeb.decide(
    {"message": "Hi, I was charged twice for my March subscription. Can you refund the duplicate?"},
    {"team": {"type": "choice", "instructions": "Which team should handle this message?",
              "criteria": {"billing": "charges, invoices, refunds", "technical": "bugs, errors, outages",
                           "account": "login, password, cancelling", "sales": "pricing, upgrades"}}})
answer = out["answers"]["team"]
p = answer["probabilities"][answer["choice"]]
action = "act" if p >= 0.90 else "confirm" if p >= 0.60 else "escalate"
print(action, answer["choice"], round(p, 3))
