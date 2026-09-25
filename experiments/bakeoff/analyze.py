"""Metrics for the bake-off report. Reads questions.jsonl, answers-*.jsonl (fleet + API path) and
answers-h-*.jsonl (harness path); writes metrics.json and metrics.md (the tables pasted into REPORT.md)."""
import itertools
import json
import os
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
Q = {}
for l in open(os.path.join(HERE, "questions.jsonl")):
    r = json.loads(l)
    Q[r["id"]] = r
SOURCES = ["condaqa", "contractnli", "sharc", "unfairtos", "mhs-clear", "mhs-close"]


def load(path, key="p"):
    out = {}
    if not os.path.exists(path):
        return out
    for l in open(path):
        e = json.loads(l)
        if e.get("kind", "main") != "main" or e.get("ok") is False or e.get(key) is None:
            continue
        out[e["id"]] = float(e[key])
    return out


T = {}
T["deepseek"] = load(os.path.join(HERE, "answers-deepseek.jsonl"))
T["qwen"] = load(os.path.join(HERE, "answers-qwen.jsonl"))
HARN = ["fable", "opus", "gpt6astra", "grok", "gemflash", "gempro"]
for t in HARN:
    d = load(os.path.join(HERE, f"answers-h-{t}.jsonl"))
    if d:
        T["h-" + t] = d
API = {t: load(os.path.join(HERE, f"answers-{t}.jsonl")) for t in ["fable", "gpt6astra", "gemflash", "opus", "gpt6luna", "gempro"]}
for t in ["fable", "gpt6astra", "gemflash"]:
    T["api-" + t] = API[t]


def ids_of(src=None, pool=None):
    ids = [i for i, r in Q.items() if src is None or r["source_key"] == src
           or (src == "human" and r["slice"] == "human") or (src == "moderation" and r["slice"] == "moderation")]
    return [i for i in ids if pool is None or i in pool]


def label(i):
    return bool(Q[i]["label"]["answer"])


def ece(ps, ys, bins=10):
    n = len(ps)
    tot = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [k for k, p in enumerate(ps) if (lo <= p < hi) or (b == bins - 1 and p == 1.0)]
        if idx:
            tot += len(idx) / n * abs(st.mean(ps[k] for k in idx) - st.mean(ys[k] for k in idx))
    return tot


def binmetrics(pred, ids):
    ids = [i for i in ids if i in pred]
    if not ids:
        return None
    ps = [pred[i] for i in ids]
    ys = [1.0 if label(i) else 0.0 for i in ids]
    yhat = [1.0 if p >= 0.5 else 0.0 for p in ps]
    n = len(ids)
    fa = sum(1 for a, y in zip(yhat, ys) if a == 1 and y == 0) / n
    miss = sum(1 for a, y in zip(yhat, ys) if a == 0 and y == 1) / n
    return {"n": n, "acc": sum(a == y for a, y in zip(yhat, ys)) / n,
            "maj": max(st.mean(ys), 1 - st.mean(ys)),
            "pred_rate": st.mean(yhat), "human_rate": st.mean(ys), "line": st.mean(yhat) - st.mean(ys),
            "false_alarm": fa, "miss": miss,
            "brier": st.mean((p - y) ** 2 for p, y in zip(ps, ys)), "ece": ece(ps, ys)}


def modmetrics(pred, ids):
    ids = [i for i in ids if i in pred]
    if not ids:
        return None
    ps = [pred[i] for i in ids]
    sh = [Q[i]["human_share"] for i in ids]
    close = [i for i in ids if Q[i]["source_key"] == "mhs-close"]
    out = {"n": len(ids), "mae_share": st.mean(abs(p - s) for p, s in zip(ps, sh)),
           "brier_share": st.mean((p - s) ** 2 for p, s in zip(ps, sh))}
    if close:
        out["n_close"] = len(close)
        out["acc_close"] = st.mean(((pred[i] >= 0.5) == (Q[i]["human_share"] >= 0.5)) for i in close)
    return out


def mean_of(names):
    common = set.intersection(*[set(T[n]) for n in names])
    return {i: st.mean(T[n][i] for n in names) for i in common}


def run():
    M = {"teachers": {t: len(v) for t, v in T.items()}, "bin": {}, "mod": {}}
    cols = ["all-human"] + SOURCES[:4] + ["mhs-clear", "mhs-close", "all"]
    for t, pred in T.items():
        M["bin"][t] = {}
        for c in cols:
            ids = ids_of("human") if c == "all-human" else (ids_of() if c == "all" else ids_of(c))
            M["bin"][t][c] = binmetrics(pred, ids)
        M["mod"][t] = modmetrics(pred, ids_of("moderation"))
    # common-subset comparison: every harness teacher + fleet on the ids all of them answered
    allrows = [t for t in T if not t.startswith("api-")]
    full = [t for t in allrows if len(T[t]) == len(Q)]
    common = set.intersection(*[set(T[t]) for t in full])
    gsub = set.intersection(*[set(T[t]) for t in allrows])
    M["gsub_n"] = len(gsub)
    M["gsub"] = {t: {"human": binmetrics(T[t], [i for i in gsub if Q[i]["slice"] == "human"]),
                     "moderation_bin": binmetrics(T[t], [i for i in gsub if Q[i]["slice"] == "moderation"]),
                     "mod": modmetrics(T[t], [i for i in gsub if Q[i]["slice"] == "moderation"])} for t in allrows}
    M["common_n"] = len(common)
    M["common"] = {t: {"human": binmetrics(T[t], [i for i in common if Q[i]["slice"] == "human"]),
                       "moderation_bin": binmetrics(T[t], [i for i in common if Q[i]["slice"] == "moderation"]),
                       "mod": modmetrics(T[t], [i for i in common if Q[i]["slice"] == "moderation"])} for t in full}
    # pairwise argmax agreement
    M["agree"] = {}
    for a, b in itertools.combinations(allrows, 2):
        ids = set(T[a]) & set(T[b])
        M["agree"][f"{a}|{b}"] = {"n": len(ids), "agree": st.mean((T[a][i] >= .5) == (T[b][i] >= .5) for i in ids)}
    # panels
    single = [t for t in full]
    hum = [i for i in common if Q[i]["slice"] == "human"]
    rank = sorted(single, key=lambda t: -M["common"][t]["human"]["acc"])
    M["rank_by_human_acc"] = rank
    panels = {"pool (deepseek+qwen)": ["deepseek", "qwen"]}
    frontier = [t for t in rank if t.startswith("h-")]
    panels["best 2 frontier"] = frontier[:2]
    panels["best 3 frontier"] = frontier[:3]
    panels["all frontier"] = frontier
    panels["best 3 overall"] = rank[:3]
    M["panels"] = {}
    for name, names in panels.items():
        if len(names) < 2:
            continue
        pred = {i: p for i, p in mean_of(names).items() if i in common}
        M["panels"][name] = {"members": names,
                             "human": binmetrics(pred, hum),
                             "moderation_bin": binmetrics(pred, [i for i in common if Q[i]["slice"] == "moderation"]),
                             "mod": modmetrics(pred, [i for i in common if Q[i]["slice"] == "moderation"])}
    # API path vs harness path
    M["path"] = {}
    for t in ["fable", "gpt6astra", "gemflash"]:
        a, h = API[t], T.get("h-" + t, {})
        ids = sorted(set(a) & set(h))
        if ids:
            M["path"][t] = {"n": len(ids), "argmax_agree": st.mean((a[i] >= .5) == (h[i] >= .5) for i in ids),
                            "mean_abs_diff": st.mean(abs(a[i] - h[i]) for i in ids),
                            "api_acc": binmetrics(a, [i for i in ids if Q[i]["slice"] == "human"])["acc"],
                            "harness_acc": binmetrics(h, [i for i in ids if Q[i]["slice"] == "human"])["acc"],
                            "api_pred_rate": st.mean(a[i] >= .5 for i in ids),
                            "harness_pred_rate": st.mean(h[i] >= .5 for i in ids)}
    # harness call stats
    M["calls"] = {}
    for t in HARN:
        p = os.path.join(HERE, f"calls-h-{t}.jsonl")
        if not os.path.exists(p):
            continue
        c = [json.loads(l) for l in open(p)]
        ok = [x for x in c if x["ok"]]
        M["calls"][t] = {"calls": len(c), "ok": len(ok), "failed_or_partial": len(c) - len(ok),
                         "timeouts": sum(x.get("error") == "timeout" for x in c),
                         "median_secs": st.median(x["secs"] for x in c),
                         "median_secs_ok": st.median(x["secs"] for x in ok) if ok else None,
                         "answered": len(T.get("h-" + t, {}))}
    json.dump(M, open(os.path.join(HERE, "metrics.json"), "w"), indent=1)
    write_md(M, cols)


def f(x, pct=False, d=3):
    if x is None:
        return "-"
    return f"{100 * x:.1f}" if pct else f"{x:.{d}f}"


def write_md(M, cols):
    L = []
    L.append(f"Answer counts: " + ", ".join(f"{t} {n}" for t, n in M["teachers"].items()))
    L.append("")
    L.append("### Accuracy (argmax at 0.5, %) against the human label, all answered items per teacher")
    L.append("| teacher | " + " | ".join(cols) + " |")
    L.append("|---" * (len(cols) + 1) + "|")
    L.append("| majority baseline | " + " | ".join(f(M["bin"]["deepseek"][c]["maj"], True) for c in cols) + " |")
    for t in M["bin"]:
        L.append(f"| {t} | " + " | ".join(
            (f(M['bin'][t][c]['acc'], True) + (f" (n={M['bin'][t][c]['n']})" if M['bin'][t][c] and M['bin'][t][c]['n'] < len(ids_of('human') if c == 'all-human' else ids_of() if c == 'all' else ids_of(c)) else "")) if M["bin"][t][c] else "-"
            for c in cols) + " |")
    L.append("")
    L.append("### The line: predicted-true rate minus human-true rate (percentage points), with false alarm / miss (% of items)")
    L.append("| teacher | " + " | ".join(cols) + " |")
    L.append("|---" * (len(cols) + 1) + "|")
    for t in M["bin"]:
        L.append(f"| {t} | " + " | ".join(
            (f"{100 * M['bin'][t][c]['line']:+.1f} ({f(M['bin'][t][c]['false_alarm'], True)}/{f(M['bin'][t][c]['miss'], True)})") if M["bin"][t][c] else "-"
            for c in cols) + " |")
    L.append("")
    L.append("### Brier / ECE (10 bins) against the human label")
    L.append("| teacher | " + " | ".join(cols) + " |")
    L.append("|---" * (len(cols) + 1) + "|")
    for t in M["bin"]:
        L.append(f"| {t} | " + " | ".join(
            (f"{f(M['bin'][t][c]['brier'])} / {f(M['bin'][t][c]['ece'])}") if M["bin"][t][c] else "-" for c in cols) + " |")
    L.append("")
    L.append("### Moderation slice against the annotator share")
    L.append("| teacher | n | MAE vs share | Brier vs share | close calls n | accuracy on close calls (%) |")
    L.append("|---|---|---|---|---|---|")
    for t, m in M["mod"].items():
        if m:
            L.append(f"| {t} | {m['n']} | {f(m['mae_share'])} | {f(m['brier_share'])} | {m.get('n_close', '-')} | {f(m.get('acc_close'), True)} |")
    L.append("")
    L.append(f"### Common subset: the {M['common_n']} items every fleet and harness teacher answered")
    L.append("| teacher | human acc % | human line pp (FA/miss %) | human Brier | human ECE | moderation acc % | moderation line pp | MAE vs share | close-call acc % |")
    L.append("|---|---|---|---|---|---|---|---|---|")

    def row(name, m):
        h, mb, md = m["human"], m["moderation_bin"], m["mod"]
        return (f"| {name} | {f(h['acc'], True)} | {100 * h['line']:+.1f} ({f(h['false_alarm'], True)}/{f(h['miss'], True)}) | "
                f"{f(h['brier'])} | {f(h['ece'])} | {f(mb['acc'], True)} | {100 * mb['line']:+.1f} | {f(md['mae_share'])} | "
                f"{f(md.get('acc_close'), True)} (n={md.get('n_close', 0)}) |")
    for t in M["rank_by_human_acc"]:
        L.append(row(t, M["common"][t]))
    for name, m in M["panels"].items():
        L.append(row(f"panel: {name} = mean of {'+'.join(m['members'])}", m))
    L.append("")
    L.append(f"### Gemini subset: the {M['gsub_n']} items every teacher answered, including both Gemini harness teachers")
    L.append("| teacher | human acc % | human line pp (FA/miss %) | human Brier | human ECE | moderation acc % | moderation line pp | MAE vs share | close-call acc % |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for t in sorted(M["gsub"], key=lambda t: -M["gsub"][t]["human"]["acc"]):
        L.append(row(t, M["gsub"][t]))
    L.append("")
    L.append("### Pairwise argmax agreement (%), on the items both answered")
    names = [t for t in M["bin"] if not t.startswith("api-")]
    L.append("| | " + " | ".join(names) + " |")
    L.append("|---" * (len(names) + 1) + "|")
    for a in names:
        cells = []
        for b in names:
            if a == b:
                cells.append("-")
            else:
                k = f"{a}|{b}" if f"{a}|{b}" in M["agree"] else f"{b}|{a}"
                cells.append(f(M["agree"][k]["agree"], True))
        L.append(f"| {a} | " + " | ".join(cells) + " |")
    L.append("")
    L.append("### API path vs harness path, same model, same items")
    L.append("| model | items both | argmax agreement % | mean abs prob diff | human acc API % | human acc harness % | predicted-true rate API % | harness % |")
    L.append("|---|---|---|---|---|---|---|---|")
    for t, m in M["path"].items():
        L.append(f"| {t} | {m['n']} | {f(m['argmax_agree'], True)} | {f(m['mean_abs_diff'])} | {f(m['api_acc'], True)} | "
                 f"{f(m['harness_acc'], True)} | {f(m['api_pred_rate'], True)} | {f(m['harness_pred_rate'], True)} |")
    L.append("")
    L.append("### Harness calls")
    L.append("| teacher | calls | fully ok | failed or partial | of which timeouts | median s/call (all) | median s/call (ok) | items answered |")
    L.append("|---|---|---|---|---|---|---|---|")
    for t, m in M["calls"].items():
        L.append(f"| {t} | {m['calls']} | {m['ok']} | {m['failed_or_partial']} | {m['timeouts']} | {m['median_secs']:.1f} | "
                 f"{m['median_secs_ok']:.1f} | {m['answered']} |")
    open(os.path.join(HERE, "metrics.md"), "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    run()
