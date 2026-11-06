# Sprint 5.18 — PPO stability and optimization diagnostics

This is a retrospective diagnostic study of the three Sprint 5.17 runs, not a
training experiment. No training, new evaluation episodes, policy inference or
reserved final-test board generation is performed. The frozen eight-comparison
Sprint 5.17 analysis is unchanged. Recommendations below require separate approval.

## Findings and interpretation

**Observed:** seed 5918 regressed on the shared development boards between 75k
and 100k. Paired outcomes support a policy-performance difference on these boards;
they do not identify its cause. Optimization remained finite. There is no common
entropy, KL, critic or gradient pattern that explains all three seeds' performance
changes. Global gradient clipping was almost always active, but that is not proof
of exploding gradients or a reason to relax clipping.

**Recommendation:** test one late-training learning-rate schedule against the
unchanged constant-rate control, with independent paired replications. This is a
low-complexity, interpretable probe of late policy drift, not a diagnosis that the
current learning rate is defective. It may also impair continued learning, as
seed 5920's improvement illustrates. No optimization change is implemented here.

## Inputs, provenance and boundaries

Repository root: `/Users/zehraatalay/Desktop/match3-ai-lab`. `AGENTS.md`, the frozen
protocol, Sprint 5.17 report, trainer, loss and optimizer were inspected. Existing
tracked changes and untracked files were preserved. Only the new analysis module,
its tests, this document and new Sprint 5.18 artifacts are added.

Inputs per seed 5918/5919/5920: 100000 transition records, 6252 optimizer records,
391 rollout records, training-episode JSON, 1000 evaluation episodes with move
records, and six checkpoints. Across runs there are 18 checkpoints and 3000
recorded evaluation episodes on **200 shared boards**, not 3000 independent boards.
The 15 milestone first-action audits already completed in Sprint 5.17 remain
hash-protected; they are not rerun. This sprint loads checkpoint payloads only to
verify configurations, without constructing policies or environments.

The protocol digest remains
`d386db9e6b1de0d3aeb188dc1b9a6b29c1d5853af1f191706c2a7c7c153564c6`.
The pre-analysis inventory hashes 626 historical JSON/JSONL/CSV/NPZ/checkpoint
artifacts, including Sprint 5.17. Additional protocol, report, patch and figure
hashes bring the protected artifact set to 655 files. Frozen production-source
hashes are independently checked by `load_protocol`; no production trainer,
agent, game or environment implementation is edited.

Available: sampled-action probability ratios, approximate KL, clip fraction,
policy/value losses, masked entropy, combined gradient norm, learning rate,
rollout pre-update explained variance and raw advantage summaries, per-transition
values/next-values/rewards/termination flags, and episode/move gameplay statistics.
Not recorded: separate actor/critic gradient norms, full-policy KL over a fixed
state set, logits or action masks for every training state, actual Adam parameter
step norms, or dense validation checkpoints between the five milestones.

## Verified definitions

### Gradient clipping

In `ppo_update.optimize_ppo`, `loss.backward()` precedes
`norm = torch.nn.utils.clip_grad_norm_(agent.network.parameters(), gradient_clip,
error_if_nonfinite=True)`, and the returned `norm` is logged. Inspection of the
installed Torch 2.15.0 implementation confirms it computes `total_norm`, clips
gradients in place, then returns the original `total_norm`. A focused test with a
gradient `[3,4]` verifies that the logged value is 5 while the post-clip norm is .5.
The [PyTorch API reference](https://docs.pytorch.org/docs/2.15/generated/torch.nn.utils.clip_grad_norm_.html)
documents the total-norm calculation and in-place clipping.

All frozen run configurations and all 18 saved checkpoint configurations specify
**global L2 threshold .5**. Frequency is `count(recorded_norm > .5) / optimizer_steps`,
unweighted by minibatch size. Torch's scale is `min(1, .5/(norm+1e-6))`; the
threshold calculation ignores epsilon-only boundary scaling. No production norm
equals .5, so this boundary convention does not change these frequencies.
The norm combines policy, critic and entropy gradients through the shared network.
Neither the norm nor its clipping scale measures the eventual Adam parameter step.

### Entropy and loss

The coefficient is **.01 in every frozen configuration and all 18 checkpoints**.
The frozen dataclass is passed unchanged from `train_ppo` to `optimize_ppo` and
`clipped_objective`; the inspected call path has no coefficient reassignment or
schedule. Thus positive entropy bonus is `.01 * recorded_entropy`, with
**negative** contribution to the minimized loss. It is derived, not an additional
recorded metric. Learning rate was independently recorded as exactly `.0003` on
all 18756 optimizer steps.

The loss is `policy_loss + .5 * value_loss - .01 * entropy`, and recorded
`value_loss` is already half MSE. Comparing loss scalars alone does not establish
which component dominates shared-network gradients. Entropy is over masked legal
actions in changing on-policy states, not normalized by each state's legal-action
count. Validation uses deterministic argmax, not policy sampling; high training
entropy therefore does not mean evaluation actions were randomly sampled.

### Alignment and aggregation

All horizontal axes are **collected transitions**. Intermediate policies have
optimized through 24832/49920/74752 at 25k/50k/75k, with 168/80/248 pending samples;
the final 100k policy includes its 160-sample tail. Optimizer/rollout summaries use
`low < completion_step <= high`, assigning only updates completed between recorded
checkpoint policies. Transition distributions use collected indices. A rollout
can cross a checkpoint boundary, so these are intentionally different windows.

Optimizer means are per minibatch update; rollout means are per rollout. The
32-sample final minibatch and 160-sample final rollout retain equal record weights
in those summaries; transition distributions retain per-transition weights.
There are four optimization epochs, normally 16 minibatch steps per 256-transition
rollout, with 12 for the tail: 391 rollouts and 6252 Adam steps per run. Each run
presents 400000 reused samples to optimization; it collects 100000 transitions.

GAE targets and raw advantages are reconstructed using the original gamma .99,
lambda .95, rollout boundaries and separate termination/truncation flags. Every
reconstructed rollout EV/advantage summary must match its saved value. These are
the original pre-update on-policy bootstrapped targets, not validation returns
and not Monte Carlo ground truth. Undefined constant-target EV remains missing.

## Per-seed checkpoint evidence

| Seed | 0 | 25k | 50k | 75k | 100k |
|---|---:|---:|---:|---:|---:|
| 5918 | 29.5% | 45.5% | 51.0% | 63.0% | 49.0% |
| 5919 | 49.5% | 52.5% | 62.0% | 54.5% | 57.0% |
| 5920 | 40.5% | 51.0% | 51.0% | 56.5% | 63.0% |

Each score uses 200 completed deterministic development episodes. Machine-readable
checkpoint summaries include objective completion, remaining HP, episode moves,
reward counts, tiles cleared, cascades and special activations. All validation
rewards are -1 or +1; consequently mean reward is exactly `2*win_rate-1`, and is
not independent performance evidence.

| Seed / interval | Mean KL | Clip fraction | Entropy | Value loss | EV | Pre-clip norm |
|---|---:|---:|---:|---:|---:|---:|
| 5918 0–25k | .00566 | .08165 | 2.388 | .23423 | .070 | 1.709 |
| 5918 25–50k | .00584 | .08004 | 1.247 | .23322 | .128 | 2.387 |
| 5918 50–75k | .00451 | .06038 | .724 | .22794 | .172 | 2.579 |
| 5918 75–100k | .00563 | .07150 | 1.409 | .22708 | .180 | 1.847 |
| 5919 0–25k | .00562 | .08016 | 2.667 | .24431 | .033 | 1.542 |
| 5919 25–50k | .00485 | .06347 | 1.361 | .24149 | .081 | 2.071 |
| 5919 50–75k | .00558 | .07238 | 1.108 | .23260 | .133 | 2.017 |
| 5919 75–100k | .00597 | .07873 | 1.360 | .22926 | .151 | 1.691 |
| 5920 0–25k | .00645 | .09294 | 2.452 | .23874 | .047 | 1.681 |
| 5920 25–50k | .00670 | .09260 | 2.116 | .23336 | .098 | 1.697 |
| 5920 50–75k | .00603 | .08247 | 1.710 | .23397 | .124 | 1.765 |
| 5920 75–100k | .00528 | .06785 | 1.365 | .23096 | .134 | 1.650 |

**5918, 75k→100k:** 126→98 wins; 70 boards won by both policies, 56 earlier-only
wins, 28 later-only wins and 46 losses for both. Difference **−14 pp**; pointwise
paired board-bootstrap 95% CI **[−22.5, −5.0] pp**; exact two-sided McNemar
`p=.002985`, exploratory 12-comparison Holm `p=.032839`. This is evidence of a
performance change on these boards, conditional on this run. It does not prove a
general training-budget effect, exclude development-board selection effects or
identify an optimization cause.

Objective completion fell 96.46%→94.83%; remaining HP rose .440→.685; moves rose
11.705→12.390. Mean clears fell by 60.455, cascades by 3.795 and special activations
by 2.690. These are descriptive policy-behavior changes; fewer clears do not by
themselves mean worse strategy because move count and objective targeting matter.

Entropy bonus rose .00724→.01409 across the last two intervals. KL rose and its
largest minibatch estimate (.04709) occurred at step 76032 in epoch four. In 5k
windows entropy rises to 1.895 during 85k–90k and then falls to 1.132 during
95k–100k. This is not a single permanent jump. Without intervening evaluations we
cannot locate when win rate declined or say a KL peak immediately caused it.
Value loss slightly decreases, EV improves .172→.180, raw advantage SD stays
about .674→.677, and mean gradient norm decreases. Training terminal-episode wins
are essentially unchanged, 1204/2093 (57.52%) versus 1209/2101 (57.54%). One final
budget-truncated episode is separately retained as TIME_LIMIT. The stochastic
training policy, changing boards and deterministic validation policy differ;
their scores are not interchangeable.

**5919:** the 50k→75k decline is −7.5 pp, paired CI [−16.5,+1.5] pp,
exploratory Holm p=.9616. Entropy decreases while performance decreases, unlike
5918's late decline. Value loss decreases and EV rises. At 100k performance
partially recovers to 57% while entropy increases. This contradicts a simple rule
that increasing entropy always reduces validation win rate.

**5920:** the last two changes are +5.5 and +6.5 pp, with paired CIs [−4.5,+15.5]
and [−3,+16] pp; both exploratory Holm p=1. Entropy decreases throughout, and
late KL/clip fraction decrease. Its largest KL (.05333 at 55040, epoch four) is
larger than either other run's maximum, despite subsequent improvement. A large
single minibatch estimate alone does not classify a run as unstable.

## Statistical evidence and its limits

The new analysis examines **all 12 adjacent within-run comparisons**, not only
the conspicuous decline. One separate Holm family covers those 12 McNemar tests.
Each bootstrap replicate uses one common vector of 200 board indices across all
seeds, checkpoints and continuous metrics (20000 replicates, seed 51818).
Intervals are pointwise percentile intervals, not Holm-adjusted simultaneous CIs.
Continuous gameplay metrics have descriptive paired differences and CIs, without
additional significance claims.

Only 5918's initial improvement (+16 pp, Holm .03222) and late decline (−14 pp,
Holm .03284) meet .05 in this exploratory family. These are **post-hoc** tests;
this correction does not turn a data-selected investigation into a preregistration
or account for every possible analysis the observed curves could motivate.
Some adjacent contrasts also appear in Sprint 5.17, under its different frozen
family; the original results are neither replaced nor reinterpreted.

There are three independent training runs and four adjacent changes per run.
Changes share checkpoints, all policies share boards, and thousands of minibatch
records are autocorrelated rather than independent training replications.
Interval-alignment plots are descriptive within-run associations. No pooled
correlation significance tests or causal regressions are reported. Board CIs are
conditional on the observed models; they do not quantify population uncertainty
over training seeds. The 70% development objective was not reached.

## Policy, critic and numerical stability

Mean approximate KL is .00541/.00550/.00611; 99th percentiles are
.02338/.02307/.02381. Mean clipped sample fraction is .0734/.0737/.0839; maxima
are .46875/.484375/.640625. There are tails worth monitoring, not pervasive
majority clipping. These estimates use sampled actions under the old rollout
policy and the current minibatch policy, not full-distribution KL between saved
milestone policies. PPO ratio clipping is not a hard bound on all policy changes.
Mean policy loss is −.01569/−.01627/−.01749; changing normalized advantages make
its scalar magnitude an unsuitable standalone measure of policy quality.

Clipping is active on **6242/6252 (99.840%), 6243/6252 (99.856%), and 6246/6252
(99.904%)** optimizer steps. Mean raw norms are 2.130/1.830/1.698, maxima
10.024/6.471/6.176; mean clipping scales are .270/.304/.319. Nearly constant
clipping frequency cannot distinguish successful and declining intervals. The
gradient threshold remains unchanged; heavy clipping warrants measurement, not
an unsupported recommendation to increase it.

Pre-update EV generally rises but remains modest. Raw rollout advantage SD is
roughly .674–.689 across interval means, without a late surge in 5918. State values
occasionally lie outside [-1,1] (5918 late range −1.307 to +1.543). Completed
sparse discounted returns are bounded, but this network is unconstrained and
bootstrapped GAE targets need not share that bound. This suggests imperfect value
calibration worth monitoring, not proof of a faulty critic. EV compares changing
on-policy trajectories and cannot rank the seeds' critics on a common test set.
Reconstructed values, targets and advantages retain distributions in JSON.

All recorded optimizer and defined rollout diagnostics and all reconstructed
targets are finite. Saved evaluation records have no invalid actions or error
fields. The original completion logs show all three runs finishing without
numerical warnings. One budget-ending truncated training episode in 5918 and
5920 is expected behavior, not a numerical failure. Neither NaN/Inf failures nor
unexpected learning-rate/optimizer-step anomalies were found. Finite metrics do
not establish successful optimization or rule out all implementation defects.

## Ranked hypotheses and intervention trade-offs

Ranking weighs evidence first, then complexity and experimental information.
It is not a posterior probability or a causal attribution. Expected win-rate
benefits are unknown for every proposed intervention.

| Rank | Hypothesis | Supporting and opposing evidence | Intervention / cost / experimental value |
|---|---|---|---|
| 1 | H5: training variation, finite boards and policy drift | Strong seed heterogeneity; many wide paired intervals; 5918 late difference is supported on these boards, so sampling uncertainty alone is insufficient | Independent paired replications; no algorithm complexity, substantial compute, highest value for generalization |
| 2 | H1: constant LR permits unhelpful late drift | LR is constant; 5918 reverses and 5919 peaks early; 5920 keeps improving and no universal KL failure appears | One late LR schedule; low implementation cost and unchanged update budget; high value for a clean controlled probe, uncertain benefit |
| 3 | H3: imperfect critic increases advantage noise | Modest EV and occasional value overshoot; EV rises even during declines and advantage SD stays similar | Separate critic LR/epochs or architecture would complicate shared-network attribution and may add compute; defer until controlled critic evidence |
| 4 | H2: unfavorable exploration balance | 5918 late entropy rebound; 5919 rebounds while recovering and its earlier decline occurs with falling entropy; masked state distributions vary | One entropy coefficient/schedule is simple and cheap but directly changes the objective; weaker cross-seed support than a late LR probe |
| 5 | H4: excessive PPO update magnitude | Some KL/ratio tails; no broad saturation and largest KL belongs to improving 5920 | KL early stopping adds a threshold and changes optimizer/critic exposure; greater confounding and implementation complexity than LR scheduling |
| 6 | H6: shared-loss gradients plus global clipping limit learning | Clipping is almost universal, but successful and unsuccessful periods share it; no component gradients | Component-gradient instrumentation could be useful later; changing clip threshold or splitting optimizers now has low diagnostic specificity |

No hypothesis establishes an implementation defect. Replication is a design
requirement, not a second optimization change. We recommend H1 as the **one
controlled intervention**, embedded in independent replications to address H5.

The original [PPO paper](https://arxiv.org/abs/1707.06347) motivates bounded
surrogate optimization with repeated minibatch epochs. The authors' implementation
study documents LR annealing and KL-based stopping as distinct choices
([Huang et al.](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/)).
Those other-task observations motivate candidates; they do not predict an effect
on Blocker Mayhem or justify transferring the reported gains quantitatively.

## Proposed experiment — approval required, not implemented or run

**Hypothesis:** reducing LR only after 50k improves the fixed 100k development
endpoint, and may reduce late policy-performance regressions. Improvement and
stability are distinct outcomes; lower KL alone is not a successful result.

| Item | Proposed specification |
|---|---|
| Control | Frozen Sprint 5.17 configuration, constant Adam LR .0003 |
| Treatment | Only LR schedule changes: constant .0003 through 50k, then linear decay to .00003 at 100k |
| Exact schedule | At rollout completion c, before its first optimizer step: `lr(c)=.0003*(1-.9*min(1,max(0,(c-50000)/50000)))`; same rate for all four epochs/minibatches of that rollout |
| Initialization | Fresh initialization; do not continue any existing checkpoint |
| Replications | 12 independent master-seed pairs, proposed 5930–5941, shared seed within each control/treatment pair; validate freshness against all local artifacts before registration |
| Budget | 100000 collected transitions per arm per pair; 24 runs, 2400000 transitions, 9384 rollouts and 150048 Adam steps total |
| Other settings | Same level, sparse reward, architecture, masks, gamma/lambda, clipping, coefficients, minibatches, epochs, guards and deterministic evaluation |
| Development boards | Proposed 300800–300998 inclusive: 199 locally unexcluded boards, same boards for every arm, seed and checkpoint |
| Final-test boards | Never access 400000–400999; independently approved held-out confirmation is a later stage |
| Checkpoints | 0, 25k, 50k, 75k, 100k collected transitions, unchanged pending-rollout accounting; fixed 100k primary endpoint |
| Primary metric | Mean paired treatment-minus-control win-rate difference at 100k over 12 master-seed pairs and 199 boards |
| Primary inference | Two-sided paired sign-flip test over the 12 seed-pair mean effects (4096 sign patterns), alpha .05; validity requires prespecified pairwise sign symmetry/exchangeability under the null, not an exact board-level randomization claim |
| Uncertainty | 20000 crossed bootstrap replicates: resample seed pairs, and one common board-index vector across all resampled pairs/arms/checkpoints; report 95% CI, all pair effects and leave-one-pair-out sensitivity |
| Secondary diagnostics | Prespecified 100k–75k signed win change and negative-part regression per arm; objective/HP/moves, KL tails, entropy, EV, GAE/values, pre-clip norms/frequency, optimization and elapsed time |
| Multiplicity | One primary hypothesis; secondary endpoints descriptive, with no additional confirmatory p-values; any later tests explicitly exploratory with a declared family |
| Provisional runtime | About 4.88 hours in eight sequential waves of three concurrent CPU processes at Sprint 5.17 contention/thread settings; reserve 5–7 hours including analysis and variability |
| Artifacts | Approximately 1.1 GiB at previous per-run size; reserve at least 2 GiB and recheck disk before execution |

The exact schedule is specified here as a proposed formula only; no trainer
support or scheduler has been added. At c=50176 the treatment first differs;
all updates through 49920 and the 50k policy should match the control exactly.
The final 160-sample rollout uses .00003. Optimizer state, file format and
inference remain compatible in principle, but a future schedule must be stored
in new protocol/config metadata. Existing checkpoints cannot provide exact
active-engine/rollout continuation and must not be reused as resumed arms.

Within each pair, use identical spawned initialization, environment, policy and
shuffle streams; pairs use distinct master seeds. After policies diverge,
action-dependent episode lengths can change which board occurs at a given
transition. Common seeds are variance control, not identical later trajectories.
Use the same process/thread settings, balance arm order and contention, and
retain every run. No replacement seeds, silent restarts or endpoint selection.
Check identical hashes/decisions/records through the 50k prefix in future runs.

**Seed limitation:** the conservative local scan leaves exactly 199 values
300800–300998 unexcluded in the existing validation partition. 300999 appears as
metadata and is conservatively excluded; that does not prove a board was played.
Do not silently expand the partition to obtain 200 boards. Recheck provenance
before freezing the next protocol, stop on conflicts, and request approval for
any changed seed allocation. Proposed boards have not been generated or evaluated.
Local provenance cannot rule out external or unrecorded use.

**Success:** prespecified positive primary effect, two-sided p<.05, crossed 95%
CI lower bound >0, and observed gain at least **5 pp**, with valid runs and all
pair results reported. To claim a gain of at least 5 pp, the lower CI must itself
exceed 5 pp. Stability improvement additionally needs the prespecified regression
diagnostic to improve; endpoint gain alone is not proof of stability.

**Failure/inconclusive:** a negative effect is a failed intervention; an interval
spanning zero or a gain below 5 pp is insufficient evidence for adoption. Smaller
KL without endpoint benefit is not success. Numerical/provenance failures stop
the study without replacement. Nonsignificance is not equivalence.

Twelve pairs balance compute and training-seed uncertainty; this is not a powered
guarantee. For a 5 pp effect, a normal approximation gives about 80% power at .05
only if paired SD is approximately 6.2 pp or less; at SD 10 pp the rough requirement
is 32 pairs. No paired-treatment SD is available from the three historical runs.
Use this as explicit planning uncertainty, not a formal sign-flip/bootstrap power
calculation, and do not increase sample size after looking at outcomes. If a
stronger power guarantee is required, revise and approve the design before training.

The runtime estimate uses mean observed process duration 2195.2 s per run, which
includes evaluation/checkpoint work and contention. Twenty-four runs total about
14.63 process-hours; dividing by three concurrency slots gives 4.88 hours elapsed.
This is not measured CPU time, and 199 rather than 200 evaluations is not assumed
to materially change runtime. There will be 23880 development episodes if this
future design is approved; none is executed during Sprint 5.18.

## Outputs, reproduction and validation

`experiments/results/sprint5_18_diagnostics/` contains `summary.json`,
`history_inventory.json`, `derived_updates.csv`, `checkpoint_config_audit.json`,
`hypotheses.json`, `experiment_proposal.json`, `validation.json`, and four PNG/PDF
figure pairs. The raw input CSVs remain in the three original Sprint 5.17 run
directories. `summary.json` includes full distributions, all 12 paired outcome
tables/CIs, 5k local windows, KL peaks and protected hashes.

Figures: `checkpoint_performance` (five metrics by seed), `policy_optimization`
(seven metrics by seed), `critic_diagnostics` (four metrics by seed), and
`interval_alignment` (three descriptive relationships). No duplicate standalone
plots are generated. Optimizer figures show faint raw data and trailing means of
50 optimizer steps, with shorter prefix windows. The constant learning-rate trace
is unsmoothed with a nonzero y-axis margin to avoid magnifying rounding noise.
Critic/checkpoint plots are
unsmoothed; value loss in the critic plot is a per-rollout optimizer mean.
The derived CSV retains each optimizer step's entropy bonus and clipping flag;
all original raw values remain available. Orange shading marks 75k–100k.

To reproduce numeric analysis and figures into a **new** output directory, from
the repository root in the existing `match3-ai` environment:

```sh
MPLCONFIGDIR=/private/tmp/match3-ai-matplotlib python -m match3_ai.training.ppo_stability \
  --protocol experiments/results/sprint5_17_preregistration/protocol.json \
  --history-inventory experiments/results/sprint5_18_diagnostics/history_inventory.json \
  --output-dir /private/tmp/match3-sprint5-18-new-output
```

The saved inventory is reused, not rescanned with new findings included as
historical inputs. Existing output directories are rejected. Interpretation,
hypothesis ranking and the experiment proposal are authored deliverables rather
than algorithmically generated conclusions. PDF creation metadata may differ;
numerical summaries and CSVs are deterministic.

Validation results and exact commands are recorded in `validation.json` after
final checks. Focused tests use synthetic records and direct gradient clipping;
no training or environment episode is invoked. Existing statistics tests also
operate on synthetic records. The full suite is not repeated for this isolated
analysis addition because it includes training/environment tests outside this
scope; the verified Sprint 5.17 full-suite result remains 796 passed. This is a
coverage limitation, not a new full-suite pass claim. Final checks include
repository-wide Ruff, formatting for the added Python files, Git whitespace,
numeric reproduction and preservation of every pre-existing workspace file.
