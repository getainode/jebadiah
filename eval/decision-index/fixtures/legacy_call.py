"""Original __call__ from 8f5fd57, frozen to verify default output compatibility."""
def __call__(self, state, questions):
    renderer = self.scorer.renderer
    rendered, raw = {}, {}
    for qid, q in questions.items():
        if q["type"] not in ("choice", "noul"):
            raise Unsupported(f"question type {q['type']!r} is not one Jebadiah answers")
        q2, notes = self._translate(q)
        try:
            r = self.scorer.render(state, q2)
        except ValueError as exc:
            if "single-token labels" in str(exc):
                raise Unsupported(f"{len(q['criteria'])} options exceed the model's {renderer.max_options_extended} single-token labels") from exc
            raise
        if r.truncated:
            raise Unsupported(f"prompt exceeds the declared context of {self.context} tokens; refused, not truncated")
        n = len(self.tok.encode(r.prompt, add_special_tokens=False))
        rendered[qid] = (r, q2["type"], n)
        raw[qid] = {"labels": r.letters, "label_scheme": r.label_scheme, "prompt_tokens": n}
        if notes:
            raw[qid]["translation"] = notes
    # batches by padded token budget, longest first, so memory is bounded and a long prompt runs alone
    order = sorted(rendered, key=lambda k: -rendered[k][2])
    probs = {}
    i = 0
    while i < len(order):
        batch = [order[i]]
        while i + len(batch) < len(order) and len(batch) < self.max_batch and rendered[order[i]][2] * (len(batch) + 1) <= self.token_budget:
            batch.append(order[i + len(batch)])
        out = self.scorer.score_rendered([(rendered[k][0], rendered[k][1]) for k in batch])
        for k, p in zip(batch, out):
            probs[k] = p
        i += len(batch)
    answers = {}
    for qid, q in questions.items():
        r = rendered[qid][0]
        a = self.answer_from_probs(q, r.keys, probs[qid])
        if q["type"] == "choice":
            # the published helper rounds to 6 decimals for the route; the kit ranks ToolRet and
            # BRIGHT candidates by probability across requests, where rounding would create ties,
            # so the answer carries the unrounded softmax (same argmax, same values to 1e-6)
            a["probabilities"] = dict(zip(r.keys, probs[qid]))
        else:
            a["noul"] = float(probs[qid][r.keys.index("true")])
        answers[qid] = a
        raw[qid]["probabilities"] = dict(zip(r.keys, probs[qid]))
    usage = {"input_tokens": sum(v[2] for v in rendered.values())}
    return {"model": self.model_id, "answers": answers, "usage": usage}, raw
