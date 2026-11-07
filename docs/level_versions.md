# Original and calibrated level versions

The original six level JSON files and IDs remain unchanged. The packaged registry
also exposes three explicit `-v2` IDs; choosing an original ID still chooses the
original configuration. No level is silently replaced. Existing benchmarks that
list the original IDs retain their scope. New versions use the same padded
observation and 144-action interface.

| New ID | Frozen Stage 2 candidate | Colors | Moves | Blocker HP | Random | Greedy | Confirmation target result |
|---|---|---:|---:|---:|---:|---:|---|
| wooden-trouble-v2 | wooden-trouble__colors5-double-moves6 | 5 | 6 | 8 | 38.5% | 31.0% | Failed: Greedy below 40–80% |
| double-trouble-v2 | double-trouble__colors5-moves6 | 5 | 6 | 8 | 56.5% | 56.0% | Passed |
| frozen-garden-v2 | frozen-garden__colors5-moves10 | 5 | 10 | 12 | 50.5% | 59.5% | Passed |

Each configuration/agent used 200 development boards. Stage 2 completed 2,400
Random/Greedy episodes without execution failures. These are baseline confirmation
results, not DQN results or independent final-test performance. Selection was
frozen from baseline screening before confirmation; Wooden Trouble was retained
as explicitly requested despite failing its Greedy target. Its v2 label denotes
a configuration version, not a claim that every calibration target passed.

Preregistered targets were Random 20–60%, Greedy 40–80%, first-move wins ≤10%,
no frequent long cascades (>1% of moves exceeding 20 rounds), and no 100+ round
chain. Uncertainty and all diagnostics remain in the Stage 2 summary/report.

## Frozen provenance

Sources: `experiments/results/level_calibration_stage2_confirmation/`.
The corresponding `configs/<candidate>.json` artifacts were verified against
both the Stage 2 manifest and its frozen Stage 1B selection hashes. Only
`level_id` changes in the new packaged JSON; names, seeds, masks, objectives,
blockers, colors, and move limits are copied unchanged. No new JSON schema or
mechanics were introduced.

| Candidate | Confirmed source SHA-256 |
|---|---|
| wooden-trouble__colors5-double-moves6 | `2cf5a4a71fbf2b0f173fd5cd97bad2e9650496a937356828e8838a483111db48` |
| double-trouble__colors5-moves6 | `99f8686fdd3507221e6096c3eabff1b8838afa9ce4d8b386a00167684bfa5a50` |
| frozen-garden__colors5-moves10 | `322aba2c05b16dddd9e8bfe6710187b7632929921733cb915b47477ca346f8ed` |

Stage 2 manifest SHA-256:
`6f2ebcfc436cf24c92f9508574f0331eee35205fd0c2a739fc4276ca9850eb4d`.
Stage 2 summary SHA-256:
`eb51db140462144e66bfd433d541d1576595732c33c25862f0c2cb136c9f5737`.

## Seed compatibility

The six original registry positions are fixed in their historical order; v2 IDs
are appended. Each v2 explicitly inherits its original level's seed namespace.
Consequently, explicit environment seeds reproduce the confirmed candidate's
board/engine seeds when comparing v2 with the original-ID configuration override.
Changing colors can still change the accepted board relative to the original
configuration. New IDs/config hashes distinguish records even when namespaces
are shared. Original configuration hashes and RNG behavior remain unchanged.

When adding another version, append its registry entry and explicitly decide its
seed namespace. Do not alphabetically insert entries into the historical order.
The `level_ids` registry now contains nine entries; clients intentionally
requesting every packaged level will discover the versions too. Historical
experiment files and checkpoint metadata are not rewritten.
