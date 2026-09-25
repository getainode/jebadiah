# 11. Decision Index entry

The Decision Index (apolinario/decision-index) scores decision models on 40 public benchmarks; its maintainer runs
every entrant himself on one RTX PRO 6000 with the author's own inference code. Most entrants, including Kev and
Nimble, run in-process through their authors' Python packages rather than over HTTP.

`eval/decision-index/jebadiah_engine.py` is our entry: an `Engine` subclass for the kit that downloads a Jebadiah
model at a pinned Hub revision and calls its published `scripts/` (prompt renderer, label-token logit read,
temperatures) unchanged. Choice questions return a probability for every option, unrounded; yes/no questions
return P(true). Questions are batched by padded token budget so a long prompt runs alone. `mps_delta.py` is a
workaround used only on Apple silicon: it replaces one triangular solve in the Gated DeltaNet layers that MPS does
not support with an equivalent block recursion.

The submission is pull request apolinario/decision-index#8 (Jebadiah 9B v1), open at the time of writing. The
JevBench shim in `eval/jevbench/` reuses this engine and enables `score` questions, which the Decision Index suite
does not have.
