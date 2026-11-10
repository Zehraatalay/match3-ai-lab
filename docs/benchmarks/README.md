# Published benchmark evidence

These small snapshots support the README tables. Original local results and all
checkpoints remain under ignored `results/`; no experiment was rerun or edited
for publication. `publication_sources.json` records original and published file
SHA256 digests. Two classic-MCTS JSON manifests had home-directory prefixes
redacted. Their nested provenance digests still refer to the original artifacts;
these publication copies are display/evidence records, not interchangeable input
for the strict historical-reuse validator.

- `mcts_baselines_main_100/`: paired Random/Greedy/RT/GO development benchmark
- `mcts_ablation_100/`: four-way 30-simulation development ablation
- `mcts_ablation_analysis_100/`: exploratory paired contrasts
- `az4/`: 500 episodes on 50 profile/seed boards, six trained checkpoints plus
  untrained Neural MCTS, Random, Greedy, and classic MCTS

The AZ-4 report includes the exact profile, seed, budget, timing, and score-proxy
limitations. Trained checkpoint files are intentionally not included. Publication
snapshots contain logical local result/checkpoint paths for provenance; these are
not download links or promises that weights ship in a fresh clone.
