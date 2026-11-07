# Results & Analysis

This report summarizes the main experiments from Match-3 AI Lab: reinforcement learning agent evaluation and gameplay data analysis.

The two analyses use different datasets and evaluation protocols.

## 1. Reinforcement Learning

### Experimental Setup

I compared four agents: Random, Greedy, Double DQN (DDQN), and Proximal Policy Optimization (PPO).

The evaluation covered three level configurations:
- Double Trouble
- Wooden Trouble
- Frozen Garden

Each agent played 100 episodes per level. Evaluation boards were paired across agents to make comparisons more consistent.

The reported DDQN and PPO results use checkpoints selected during validation. Evaluation results were not used to select those checkpoints.

### Win Rates

| Level | Random | Greedy | DDQN | PPO |
|---|---:|---:|---:|---:|
| Double Trouble | 62% | 64% | **77%** | 58% |
| Wooden Trouble | 43% | 32% | **63%** | 51% |
| Frozen Garden | 48% | 54% | **76%** | 66% |
| **Macro average** | **51.0%** | **50.0%** | **72.0%** | **58.3%** |

DDQN had the highest observed win rate on all three levels.

The largest observed difference between DDQN and Greedy was on Wooden Trouble: 63% versus 32%.

PPO performed better than both baseline agents on Wooden Trouble and Frozen Garden, but not on Double Trouble.

### Limitations

- Each agent was evaluated on 100 episodes per level.
- Trained policies were selected using validation performance.
- Training experiments used limited random seeds and computational budgets.
- The final differences between agents were not formally tested for statistical significance.
- Results may change with different levels, training budgets, or random seeds.

These results suggest that DDQN learned useful strategies for the evaluated configurations, but they do not establish general superiority across Match-3 games.

---

## 2. Gameplay Data Analysis

I analyzed gameplay data separately from the reinforcement learning evaluation.

The goal was to understand which board characteristics and gameplay metrics were associated with winning and whether they could help predict episode outcomes.

### Dataset

The analysis used 2,400 gameplay episodes from Random and Greedy agents.

The paired statistical comparison used 200 matched boards per level.

The dataset covers three level configurations and includes episode outcomes, gameplay metrics, and engineered features.

### Paired Statistical Comparison

I compared Greedy and Random using an exact two-sided McNemar test, with Holm correction for multiple comparisons.

| Level | Greedy − Random | 95% Bootstrap CI | Adjusted p-value |
|---|---:|---:|---:|
| Wooden Trouble v2 | −7.5 pp | [−16.5, +1.5] | 0.2568 |
| Double Trouble v2 | −0.5 pp | [−9.5, +8.5] | 1.0000 |
| Frozen Garden v2 | +9.0 pp | [−0.5, +18.5] | 0.2466 |

None of the differences were statistically significant after correction.

This does not establish that the agents perform equally. It means the observed differences were not sufficient to demonstrate a statistically significant difference with this sample.

These comparisons use development data rather than an independent final test set.

### Win Prediction

I compared a historical baseline with Logistic Regression and Random Forest.

| Model | Accuracy | ROC-AUC | PR-AUC |
|---|---:|---:|---:|
| Historical Baseline | 0.608 | 0.625 | 0.582 |
| Logistic Regression | 0.604 | **0.647** | **0.623** |
| Random Forest | 0.558 | 0.606 | 0.605 |

Logistic Regression achieved the highest ROC-AUC and PR-AUC in this comparison.

The modest scores also suggest that predicting wins from the available features is not straightforward.

### Spatial Feature Engineering

I explored whether spatial information about the board could improve win prediction.

The experiments used grouped cross-validation to compare different feature sets.

| Features | Mean ROC-AUC | Mean PR-AUC |
|---|---:|---:|
| Level + Agent | 0.577 | 0.525 |
| Level + Agent + Board | 0.584 | 0.539 |
| Level + Agent + Spatial | **0.591** | 0.556 |
| Level + Agent + Board + Spatial | 0.583 | **0.557** |

Adding spatial features to Level + Agent improved mean ROC-AUC by approximately 0.013.

However, Brier score and log loss were slightly worse. Better ranking performance did not necessarily translate into better probability estimates.

### Visualizations

**Agent win-rate comparison**

![Agent win-rate comparison](../analysis/figures/win_rate_comparison.png)

**Board difficulty distribution**

![Board difficulty distribution](../analysis/figures/stage2_board_difficulty_distribution.png)

**Spatial feature correlations**

![Spatial feature correlations](../analysis/figures/stage2_spatial_validation_correlations.png)

---

## 3. Key Takeaways

1. DDQN achieved the highest observed win rates on the three evaluated levels.
2. Greedy was not consistently better than Random.
3. Logistic Regression outperformed Random Forest on ROC-AUC and PR-AUC in the evaluated dataset.
4. Spatial features modestly improved ranking performance but did not improve probability calibration.

The project provides a playable environment for experimenting with game mechanics, AI agents, and gameplay analytics.

Future improvements could focus on generalization across levels, broader evaluation, and more reliable training.
