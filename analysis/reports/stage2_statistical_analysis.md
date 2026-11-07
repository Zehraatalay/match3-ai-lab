# Stage 2: Paired Agent Comparison

## Method

- Dataset: Stage 2 calibration confirmation.
- Agents: Greedy and Random.
- Sample size: 200 matched boards per level.
- Test: exact two-sided McNemar.
- Multiple comparisons: Holm correction.
- Effect size: Greedy minus Random win rate.
- Confidence intervals: paired percentile bootstrap (95%).
- Bootstrap resamples: 10,000.
- Evaluation: development data, not independent final test.

## Results

| Level | Difference (pp) | 95% bootstrap CI (pp) | Exact p | Holm-adjusted p | Significant? |
|---|---:|---:|---:|---:|---|
| Wooden Trouble v2 | -7.5 | [-16.5, +1.5] | 0.1284 | 0.2568 | No |
| Double Trouble v2 | -0.5 | [-9.5, +8.5] | 1.0000 | 1.0000 | No |
| Frozen Garden v2 | +9.0 | [-0.5, +18.5] | 0.0822 | 0.2466 | No |

## Interpretation notes

- A non-significant result does not establish equivalence.
- Confidence intervals quantify uncertainty in observed differences.
- The paired bootstrap resamples boards, not individual agents.
- Percentile bootstrap CIs are approximate and may not align exactly with exact-test decisions.
- Comparisons are within each fixed level configuration.
- These results do not establish causal effects of gameplay mechanics.
