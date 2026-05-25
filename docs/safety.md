# Safety Features

OfflineRL-Lab implements three complementary safety layers designed to prevent unsafe deployment of offline RL policies. These operate at different stages of the pipeline.

## Support Estimator (k-NN Density)

The support estimator uses k-nearest-neighbor density estimation to determine whether a proposed state-action pair (s, a) falls within the support of the training dataset. For each query, the estimator finds the k nearest neighbors in the dataset's state-action space and computes a support score based on the mean distance to those neighbors:

```
score(s, a) = 1 / (1 + d_k(s, a) / d_mean)
```

where `d_k` is the mean distance to the k nearest dataset points and `d_mean` is the average in-sample k-NN distance. A score near 1.0 means the action is well-supported by the training data; near 0.0 means it is far from any training example.

This estimator is fit on the full dataset before deployment. At inference time, each proposed action is scored before execution. The main limitation is that k-NN does not scale beyond roughly 1 million transitions without approximate nearest-neighbor search (e.g., FAISS), and it is sensitive to the dimensionality of the state-action space.

## Policy Shield

The policy shield wraps a trained policy and intercepts any action with a support score below a configurable threshold. Three intervention strategies are available:

**Nearest-safe (default):** When an action has low support, the shield samples N candidate actions from a bounded set and selects the one with the highest support score. This finds an in-distribution alternative while preserving intent.

**Blend:** The proposed action is interpolated with the behavioral policy's action, weighted by the support score. Actions with very low support are replaced almost entirely by behavioral policy actions; moderately supported actions receive partial blending. This strategy is smoother but may not guarantee the resulting action is actually in-distribution.

**Reject:** The shield simply rejects the proposed action and returns the behavioral policy's action directly. This is the most conservative strategy and is equivalent to falling back to the training data distribution entirely.

The policy shield adds inference latency proportional to the number of candidate actions sampled in the nearest-safe strategy. In latency-critical systems, this may require pre-computing support scores or using a faster density estimator.

## Constraint Critic

The constraint critic is a separate Q-function trained to estimate the expected future constraint violation, rather than the expected return. For each constraint (e.g., latency SLO, error rate), a separate Q-function is trained on the dataset using the constraint violation indicator as the "reward" signal:

```
Q_c(s, a) = E[sum_t gamma^t * 1(constraint_t violated) | s_0=s, a_0=a]
```

At inference time, the constraint critic scores each candidate action by its expected cumulative constraint violations. Actions that exceed a constraint budget are filtered before the policy selects from the remaining options. Multiple constraints can be handled by training separate critics and applying a conjunction filter.

The constraint critic provides more nuanced constraint awareness than the binary support estimator, since it can estimate the severity and expected frequency of violations rather than just their presence.

## CVaR and Safety Metrics

Beyond the safety layers, OfflineRL-Lab tracks several safety-oriented evaluation metrics:

**CVaR-5% (Conditional Value at Risk):** The expected return in the worst 5% of episodes. High mean return with low CVaR indicates a policy that occasionally catastrophically fails.

**SLO violation rate:** The fraction of timesteps where a Service Level Objective is violated (e.g., latency > 200ms, error rate > 1%).

**OOD action rate:** The fraction of policy actions with support score below the safety threshold. Higher OOD rates indicate higher generalization risk.

**Catastrophic failure rate:** The fraction of episodes where the policy achieves negative total return or causes a terminal failure condition.

These metrics are reported alongside mean return in all evaluation outputs. A policy optimized for mean return alone may have unacceptably high SLO violation rates or CVaR — the full metrics suite is needed to assess deployment safety.
