# Offline Policy Evaluation

## Why OPE Matters

The fundamental challenge in offline RL is that we cannot run the learned policy in the real environment during development. Online evaluation is either too expensive (robot training), too dangerous (clinical decisions), or unavailable (historical production systems). Without environment access, we need estimators that can predict policy performance from the static training dataset alone.

Offline Policy Evaluation (OPE) refers to a class of methods that estimate the expected return of an evaluation policy using only offline data collected by a (potentially different) behavior policy. All OPE methods make assumptions that may not hold in practice, and their estimates should be treated as directional signals rather than ground-truth predictions.

## FQE: Fitted Q-Evaluation

Fitted Q-Evaluation (Precup et al., 2000; Le et al., 2019) iteratively fits a Q-function to the evaluation policy using supervised regression on the Bellman backup:

```
Q^(k+1)(s, a) = r + gamma * Q^(k)(s', pi_eval(s'))
```

Starting from Q^(0) = 0, each iteration regresses the Q-function toward the Bellman target using the evaluation policy's action at the next state. After convergence, V^pi = E_s[Q^pi(s, pi(s))] gives the value estimate.

FQE is computationally heavier than IS-based methods (requires neural network training) but produces lower variance estimates. It is consistent under the evaluation policy's stationary distribution, but may have high bias when the evaluation policy is very different from the behavior policy.

## Importance Sampling (IS) and Weighted IS (WIS)

Importance sampling corrects for the distribution mismatch between the behavior and evaluation policies by reweighting each trajectory by the density ratio:

```
rho_t = pi_eval(a_t | s_t) / pi_behavior(a_t | s_t)
```

The IS estimator multiplies per-step rewards by the cumulative product of these ratios. The main problem is variance: cumulative ratios can be exponentially large for long trajectories, producing estimates with enormous variance.

Weighted IS (WIS, also called self-normalized IS) normalizes the importance weights across episodes so they sum to 1. This reduces variance substantially at the cost of introducing a small bias. WIS is generally preferred over IS in practice for offline RL trajectories.

Both estimators clip importance ratios at 20 to prevent extreme values. When the evaluation policy is very different from the behavior policy, even clipped ratios can lead to unreliable estimates.

## DR: Doubly Robust Estimator

The Doubly Robust (DR) estimator combines the direct method (FQE) and importance sampling into a single estimator that is consistent if either component is correctly specified:

```
V^DR = V^DM + IS_correction
```

where V^DM is the direct model estimate from FQE and the IS correction accounts for the residual error in the model. If the Q-function is perfect (modeling error = 0), the IS correction vanishes. If the IS weights are correct (densities are known exactly), the estimator is consistent regardless of model quality.

In practice, neither component is perfect, and DR provides a useful combination that often outperforms either estimator alone. The implementation in OfflineRL-Lab uses a simplified version that averages FQE and WIS estimates.

## Bootstrap CIs: Uncertainty Quantification

Bootstrap confidence intervals provide a measure of uncertainty around any OPE estimate. Given N episode-level return estimates (or OPE per-episode values), we resample with replacement B=1000 times, compute the statistic on each resample, and report the 2.5th and 97.5th percentiles as the 95% confidence interval.

Bootstrap CIs capture sampling uncertainty (how much would the estimate change with different training episodes?) but not model uncertainty (is the Q-function correct?) or distribution shift (does the evaluation policy match the behavior policy?). Narrow CIs mean the estimate is stable across bootstrap resamples, not that it is correct.

## Model-Based OPE

The model-based estimator learns transition and reward models from the dataset, then uses them to simulate rollouts under the evaluation policy. This allows return estimation without explicit importance weights and can handle environments where the behavior policy is unknown. The main limitation is model error compounding: small prediction errors at each step accumulate over long rollouts, often producing overoptimistic estimates.
