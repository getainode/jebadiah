# 01. The v0 recipe and the first sweep

**Question.** Does the v0 recipe (one epoch, LoRA r16 on every linear layer, candidate-label cross-entropy, public
data only) carry from 4B to 9B, and are one epoch and the teacher's score spread the right choices?

**Runs.** `smoke`, `4b-v0`, `9b-v0`, `4b-2ep`, `4b-scorefix`, `9b-base-eval`, one rented H100 PCIe 80 GB.

| run | what changed | headline | Nimble 324 | DSj HelpSteer2 | Jevals HelpSteer2 ECE |
|---|---|---:|---:|---:|---:|
| `9b-base-eval` | no adapter | 68.70 | 69.1 | 9.6 | |
| `4b-v0` | the recipe (published as 4B v0) | 70.94 | 77.5 | -20.2 | |
| `9b-v0` | same recipe on the 9B (published as 9B v0) | 72.53 | 76.9 | -21.4 | |
| `4b-2ep` | two epochs | 69.67 | 75.6 | -23.4 | |
| `4b-scorefix` | ordinal score targets, temperature fitted on the training target | 70.05 | 74.4 | -3.5 | |

**What we learned.**

- Training adds about 4 headline points over the untrained 9B base.
- Two epochs lost to one everywhere except the in-distribution set. One epoch stays.
- v0's HelpSteer2 Decision Score is below zero: the model was confidently wrong on the human rubric. The v0 score
  temperature (0.39) came from fitting hard labels to a model trained on soft ones, which sharpened it. The score fix
  (a unimodal ordinal target around the human label, adjacent level 0.2, and the temperature fitted on that same
  target) moves it most of the way back to honest, at a small cost in sharpness elsewhere.

**Records.** `results/runs/{smoke,4b-v0,9b-v0,4b-2ep,4b-scorefix,9b-base-eval}/`, nonce passes in
`results/nonce/{4b-v0,9b-v0,4b-2ep,4b-scorefix,base-4b,base-9b}/`.
