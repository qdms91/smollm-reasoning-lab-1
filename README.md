# Arithmetic comparisons with SmolLM2 and reinforcement learning

Can a small language model learn to decide whether the sum of two integers exceeds a threshold? We ran full-parameter REINFORCE on SmolLM2-360M-Instruct on a laptop CPU. All twelve evaluated trained checkpoints scored **50/100**, the same as the frozen direct model. One always answered No. The others always answered Yes. We also found many correct sums in replies with a wrong or missing final decision.

[Scientific report (PDF)](docs/report.pdf)

## Experiments

| Method | What we varied and why |
| --- | --- |
| Frozen direct | Establish the untrained reference. The ten development probes in `01_baseline_inference.py` isolate Yes/No option order. The final direct control uses D’s order-neutral prompt. |
| B · Frozen external loop | `02_baseline_inference.py` tests whether inviting or requesting a self-question elicits one. The final B control uses C’s later prompt and routing rule so B and C differ in training history under the same loop. |
| D · Direct-answer RL | `03_train_d.py` rewards a correct final Boolean. Compare interleaved, five-update single-class blocks and same-threshold opposite-label pairs at learning rates of 2e-6 and 1e-5. |
| A · Single-response RL | `04_train_a.py` asks for a question, addition equation and decision within one completion. Compare plain versus scaffold prompts and raise the scaffold’s question bonus from 0.10 to 0.40 at fixed 2e-6. |
| C · External-loop RL | `05_train_c.py` trains an optional second turn at 2e-6. Each call generates its own tokens. The first call earns downstream reward. Frozen and trained policies use the same 64-token total budget. |

The arithmetic file contains exactly 200 training, 50 validation and 100 test questions. Each split is balanced by label and disjoint by unordered operand pair. The ten development probes are separate. The original training runs used seed 73, one epoch, two questions per update, stochastic training and greedy evaluation. We selected training checkpoints on validation before opening the final test.

## What we found on the held-out questions

| Observation | Result |
| --- | --- |
| Frozen B versus trained C | B scored 10/100 with 79 invalid finals. All 98 second turns began with an exact question copy. C took zero second turns and scored 50/100 by always answering Yes. |
| Interleaved versus blocked D | At both learning rates the final answer was always Yes. At 1e-5, every interleaved reply used 2 tokens and every blocked reply used all 64. |
| Threshold-paired D at 1e-5 | All Yes at update 25, all No at update 100. At update 25, 78 correct equations followed Yes, including 39 wrong final decisions. |
| Scaffolded A | Increasing the question reward coincided with 6 → 53 replies containing a correct equation. All of those equations followed Yes. Both runs scored 50/100. |

The original 16 rows include four frozen prompts, eleven recorded training winners and one additional checkpoint identified by the validation probability diagnostic. All conditions saw the same 100 items. The raw replies allow inspection of individual arithmetic and scoring failures. [Summary](results/final_test/summary.csv) · [Raw replies](results/final_test/responses.jsonl) · [Evaluation manifest](results/final_test/manifest.json)

A follow-up with training seed 74 is saved in [`followups/`](followups/). At the higher learning rate, blocked training again reached the 64-token limit on every validation question, without improving accuracy. The longer outputs differed in content across seeds.
