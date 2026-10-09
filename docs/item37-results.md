# Item 37: three isolated A3 ladder rungs

Runs are in progress. Models remain private and are never submitted.
Each rung is kept only if its frozen-proxy paired 95% interval versus A3 is above zero.

G1 appends only 1,350 item33 grounding training questions to pinned A3.
Its 150 grounding holdout questions remain diagnostic; the original 512-question
calibration is byte-identical. D0 changes only LoRA dropout from 0.05 to 0.0.
L5 changes only peak learning rate from 1e-4 to 5e-5, retaining cosine and warmup 30.
All use seed 17, one full epoch, rank/alpha 16/32, 2,048 tokens, batch 8,
item27 speed flags, and backbone autocast off.

Training code: ebd3dfb. G1 data: 98356116e9fdd9db8de8759bfab9eac5e83f0c03.
D0/L5 A3 data: bb4ce7b51ec3d2a8b8a0f3057afe25a9c3c9a5ed.

| Run | Training job | Initial status |
|---|---|---|
| G1 | https://huggingface.co/jobs/jbrashear/6ac845b0fee2c900701721fb | launched |
| D0 | https://huggingface.co/jobs/jbrashear/6ac845b0095c578089300c3f | launched |
| L5 | https://huggingface.co/jobs/jbrashear/6ac845b0fee2c900701721fd | launched |

Each training timeout is 140 minutes; each proxy is 35 minutes.
The aggregate timeout allocation is $24.0625 at $2.75/hour, below $25.
Failed stages and any recovery count against the same cap.

Local checks: regenerated item33 hashes match, full-source overlap scan passes,
G1 composition ID/family/state collision checks pass, all splits lint,
20 data tests pass, dropout parser and isolated config comparisons pass,
and compilation and whitespace checks pass. All checks run on local hardware.

Scores, paired intervals versus A3 and v2, area deltas, worst-five deltas,
merge results, measured spend and cleanup will be recorded after completion.
