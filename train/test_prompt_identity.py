"""Proof that the copied renderer is AINode's: render a sample of records through both and
compare bytes. Runs on the Mac against the checked-out repo (pip install not needed)."""
import json, os, sys, random
sys.path.insert(0, os.environ.get("AINODE_SRC", "ainode-src"))  # a checkout of getainode/ainode at the pinned commit
from ainode.api import decide as D, systemone as S
import ainode_prompt_verbatim as V
from jebadiah_prompt import flatten_text

def render_real(state, q):
    t = S.translate_one("q", q)
    return D.build_messages(D.serialize_state(state), None, t.question, t.options), t.names
def render_copy(state, q):
    t = V.translate_one("q", q)
    return V.build_messages(V.serialize_state(state), None, t.question, t.options), t.names

paths = sys.argv[1:]
n = 0
for p in paths:
    rows = [json.loads(l) for l in open(p)]
    random.Random(0).shuffle(rows)
    for r in rows[:300]:
        for qid, q in r["questions"].items():
            a = render_real(r["state"], q); b = render_copy(r["state"], q)
            assert a == b, (p, r["id"], qid)
            n += 1
print("identical for", n, "questions across", len(paths), "files; commit", V.PROMPT_SOURCE_COMMIT[:12], "sha", V.PROMPT_SOURCE_SHA256[:16])
