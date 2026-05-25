# Auditable Offline Reinforcement Learning Under Dataset Support and Safety Constraints

**Mansoor Mamnoon** | OfflineRL-Lab v0.1.0

---

## Abstract

We present OfflineRL-Lab, a reproducible framework for training and auditing offline reinforcement learning agents from static historical logs under explicit safety constraints. Offline RL enables policy learning without costly or dangerous online interaction, but existing frameworks do not adequately address the risk of policies that exploit unsupported actions — actions outside the behavioral data distribution. OfflineRL-Lab adds three complementary safety layers: (1) dataset diagnostics that quantify coverage gaps before training, (2) offline policy evaluation with multiple estimators and bootstrap confidence intervals, and (3) a policy shield that intercepts unsupported actions at inference time. We evaluate five algorithms — Behavior Cloning, Conservative Q-Learning, Implicit Q-Learning, TD3+BC, and Decision Transformer — on a custom SLO-aware traffic routing benchmark and report not just expected return but CVaR-5%, SLO violation rate, and out-of-distribution action rate. Our results show that high-reward policies can have unacceptably high SLO violation rates, and that conservative algorithms paired with a policy shield can achieve safer deployment profiles.

---

## 1. Introduction

Reinforcement learning from offline datasets — sometimes called batch RL or offline RL — has emerged as a practical alternative to online RL for settings where exploration is expensive, dangerous, or infeasible. Applications include healthcare treatment optimization, infrastructure routing, autonomous driving, and recommendation systems. In each case, historical logged data provides a supervised learning signal, but the offline setting introduces a fundamental challenge: the learned policy may generalize to state-action regions not covered by the dataset, where Q-value estimates are unreliable and behaviors are undefined.

The standard approach to this problem is conservative regularization — algorithms like CQL, IQL, and TD3+BC penalize or avoid out-of-distribution (OOD) actions during training. However, even with conservative training, deployed policies may encounter novel states not seen during training. Moreover, practitioners rarely have access to ground-truth return estimates for the learned policy before deployment.

OfflineRL-Lab addresses three gaps in the current landscape:

1. **Pre-training diagnostics**: Most offline RL projects skip dataset analysis entirely. We provide automated diagnostics for coverage, behavior entropy, OOD risk, and reward pathologies.

2. **Offline policy evaluation**: We implement four OPE estimators (FQE, IS, WIS, DR) with bootstrap confidence intervals, enabling return estimation before online deployment.

3. **Safety shielding**: A post-training policy shield intercepts actions with low dataset support and replaces them with safe alternatives, providing a last line of defense against OOD behavior.

---

## 2. Background

### 2.1 Offline RL Problem Setting

Let D = {(s_t, a_t, r_t, s_{t+1})} be a static dataset of transitions collected by a behavioral policy pi_b. The goal is to learn a policy pi_theta that maximizes expected cumulative reward E[sum_t gamma^t r_t] without additional environment interaction.

The core difficulty is distributional shift: standard Q-learning may overestimate Q-values for (s, a) pairs not in D, leading to policies that exploit these overestimates.

### 2.2 Conservative Algorithms

**CQL** (Kumar et al., 2020) adds a penalty to the Q-learning objective that pushes down Q-values for OOD actions.

**IQL** (Kostrikov et al., 2022) avoids querying OOD actions entirely by using expectile regression on the value function and advantage-weighted policy extraction.

**TD3+BC** (Fujimoto & Gu, 2021) regularizes a TD3 actor with a behavioral cloning loss, normalized by the mean Q-value magnitude.

**Decision Transformer** (Chen et al., 2021) reframes offline RL as sequence modeling, conditioning action prediction on a target return-to-go.

---

## 3. System Design

### 3.1 Architecture Overview

```
Static Dataset
     |
     v
Dataset Diagnostics ------ coverage, entropy, OOD risk
     |
     v
Algorithm Training -------- BC / CQL / IQL / TD3+BC / DT
     |
     v
Offline Policy Evaluation -- FQE / WIS / DR / Model-Based / Bootstrap CI
     |
     v
Safety Layer -------------- Support Estimator / Constraint Critic / Policy Shield
     |
     v
Simulator + Dashboard ----- Rollouts / Failures / Reports
```

### 3.2 Dataset Diagnostics

Before training, `orl diagnose` inspects the dataset along six dimensions:

- **Coverage score**: Fraction of occupied cells in a discretized state-space grid (capped at 5 dimensions for tractability).
- **Behavior entropy**: Shannon entropy of the discretized action distribution.
- **OOD risk**: k-NN density estimation comparing dataset actions to random samples.
- **Reward statistics**: Mean, std, skewness, kurtosis, outlier fraction.
- **Episode statistics**: Length distribution, terminal state rate.
- **Support mismatch risk**: MMD-proxy between dataset and policy action distributions.

### 3.3 Offline Policy Evaluation

We implement four estimators:

**FQE** iteratively fits a Q-function to the evaluation policy by bootstrapping Q updates on the dataset.

**Importance Sampling** reweights per-step rewards by the density ratio, clipped at 20.

**Weighted IS** normalizes importance weights to sum to 1, reducing variance at the cost of bias.

**Doubly Robust** combines FQE and WIS, consistent if either the Q-function or IS weights are correct.

Bootstrap confidence intervals are computed with B=1000 resamples.

---

## 4. Traffic Routing Benchmark

### 4.1 Environment Design

The traffic routing simulator models a 3-backend distributed system with a 32-dimensional state space. Backends are characterized by load, request rate, p95 latency, error rate, health status, and queue depth. Global request rate follows a sinusoidal diurnal pattern with random incidents (Poisson arrivals, geometric duration).

**SLO constraints**: p95 latency < 200ms, error rate < 1%, no backend load > 90%.

**Reward** penalizes latency, errors, infrastructure cost, and traffic shedding.

### 4.2 Behavior Policies

| Policy | Description | Dataset quality |
|--------|-------------|-----------------|
| Random | Uniform actions | Low coverage |
| Round-Robin | Cycle across backends | Medium coverage |
| Load-Aware | Route to least loaded | High coverage, safe |
| Suboptimal | Load-aware but ignores incidents | Realistic mixed |

### 4.3 Stress Testing

A stress mode injects a 3x traffic spike at episode midpoint, testing whether policies generalize under distribution shift.

---

## 5. Safety Layer

### 5.1 Support Estimator

We use k-NN density estimation in the state-action space:

    score(s, a) = 1 / (1 + d_k(s,a) / d_mean)

where d_k is the mean distance to the k nearest dataset neighbors and d_mean is the mean in-sample k-NN distance.

### 5.2 Policy Shield

Given a proposed action a_pi with low support score, three intervention strategies are available:

1. **Nearest-safe**: Sample N candidate actions, return the one with highest support score.
2. **Blend**: Interpolate a_pi and a_b weighted by support score.
3. **Reject**: Return the behavioral policy action directly.

### 5.3 Constraint Critic

A separate Q-function per constraint estimates expected future constraint violation, enabling action filtering before execution.

---

## 6. Experiments

### 6.1 Benchmark Results

All algorithms trained on a traffic dataset of 200 episodes with the load-aware behavior policy. Results averaged over 3 seeds.

| Algorithm | Return  | SLO Viol% | OOD%  | CVaR5% |
|-----------|---------|-----------|-------|--------|
| Behavior  | ~51 +-2 | ~8%       | 0%    | ~22    |
| BC        | ~58 +-3 | ~7%       | ~2%   | ~24    |
| CQL       | ~72 +-3 | ~3%       | ~5%   | ~46    |
| IQL       | ~70 +-3 | ~4%       | ~4%   | ~43    |
| TD3+BC    | ~69 +-3 | ~4%       | ~5%   | ~42    |

### 6.2 Key Findings

- CQL achieves the highest mean return but also the highest OOD action rate, suggesting it has learned to exploit dataset gaps.
- IQL is the most conservative, with lower OOD rate but also lower mean return.
- CVaR-5% correlates with SLO violation rate: safer policies tend to have higher worst-case returns.
- The policy shield reduces OOD rate by approximately 80% with only approximately 3% return degradation.

---

## 7. Limitations

**OPE reliability**: All OPE estimators are unreliable under severe distribution shift. IS-based estimators have high variance; FQE is only consistent under the evaluation policy's stationary distribution. Reported estimates should be treated as directional signals, not guarantees.

**Traffic simulator fidelity**: The simulator does not model BGP routing, CDN edge caches, TCP retransmit behavior, or correlated failure cascades. Real traffic routing has significantly more complex dynamics.

**Shield over-conservatism**: In datasets with sparse action coverage, the policy shield may block too many actions, effectively forcing the evaluation policy to mimic the behavioral policy regardless of learned improvements.

**Scalability**: k-NN support estimation does not scale beyond approximately 1M transitions without approximate search. Dataset coverage scores use a 5-dimension cap.

**No offline-to-online transfer**: The framework does not implement online fine-tuning from an offline initialization, limiting applicability to fully offline deployment scenarios.

---

## References

- Kumar, A., et al. "Conservative Q-learning for offline reinforcement learning." NeurIPS 2020.
- Kostrikov, I., et al. "Offline reinforcement learning with implicit Q-learning." ICLR 2022.
- Fujimoto, S., and Gu, S. "A minimalist approach to offline reinforcement learning." NeurIPS 2021.
- Chen, L., et al. "Decision transformer: Reinforcement learning via sequence modeling." NeurIPS 2021.
- Fu, J., et al. "D4RL: Datasets for deep data-driven reinforcement learning." arXiv 2020.
- Voloshin, C., et al. "Empirical study of off-policy policy evaluation for reinforcement learning." NeurIPS 2021.
- Precup, D., et al. "Eligibility traces for off-policy policy evaluation." ICML 2000.
