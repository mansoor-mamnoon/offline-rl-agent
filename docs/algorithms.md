# Algorithms

This document describes the five offline RL algorithms implemented in OfflineRL-Lab and the design choices behind each.

## Behavior Cloning (BC)

Behavior Cloning treats offline RL as pure supervised learning. Given a dataset of state-action pairs collected by a behavior policy, BC trains a policy network to minimize the prediction error between its outputs and the recorded actions. For continuous action spaces, BC fits a Gaussian distribution over actions and maximizes log-likelihood. For discrete action spaces, it uses cross-entropy loss.

BC is the simplest baseline and serves as an important sanity check: if other algorithms do not outperform BC, the dataset likely lacks sufficient coverage to support value-based learning. The main limitation of BC is that it reproduces the behavior policy exactly, including its mistakes. It cannot improve upon the data-generating policy, and it fails silently in states not covered by the training distribution.

## Conservative Q-Learning (CQL)

CQL (Kumar et al., 2020) addresses the overestimation problem in offline Q-learning by adding a regularization term that penalizes Q-values for out-of-distribution (OOD) actions. The standard Bellman backup can assign arbitrarily high Q-values to state-action pairs not seen in the dataset, which causes the learned policy to exploit these overestimates. CQL adds a penalty proportional to the log-sum-exp over all actions minus the expected Q-value under the behavioral policy, pushing down Q-values for unsupported actions while preserving them for supported ones.

In practice, CQL is controlled by an alpha hyperparameter that trades off conservatism against task performance. Higher alpha produces safer but more conservative policies. CQL works with both discrete and continuous action spaces and typically outperforms BC on datasets with moderate coverage. The downside is higher computational cost (requires sampling from the current policy during training) and sensitivity to the alpha hyperparameter.

## Implicit Q-Learning (IQL)

IQL (Kostrikov et al., 2022) avoids ever querying Q-values at OOD actions — including during training. Instead of maximizing Q over actions directly, IQL uses expectile regression to fit a value function V(s) that approximates the maximum Q-value achievable within the dataset support. Policy extraction then uses advantage-weighted regression: the policy is updated by regressing toward actions with positive advantage (Q(s,a) - V(s)), weighted by an exponential.

This approach is extremely stable because it never generates OOD queries. The key hyperparameter is the expectile tau: tau=0.5 gives the median (equivalent to standard regression), while tau close to 1.0 gives the maximum (more aggressive). IQL works only with continuous action spaces and tends to be the most conservative algorithm in the suite, with the most reliable training curves.

## TD3+BC

TD3+BC (Fujimoto and Gu, 2021) takes a minimalist approach: it augments the TD3 actor-critic algorithm with a behavioral cloning regularization term on the actor loss. The actor loss is a sum of TD3's standard policy gradient term and a BC term that penalizes deviating from the behavioral policy, normalized by the mean absolute Q-value to balance the two objectives.

This normalization is important: without it, the BC term can dominate when Q-values are large, or be ignored when Q-values are small. TD3+BC uses a twin critic (two Q-networks) with delayed policy updates, following the standard TD3 recipe. It is generally competitive with CQL while being significantly simpler to implement and tune, making it a strong practical baseline for continuous-action offline RL.

## Decision Transformer (DT)

Decision Transformer (Chen et al., 2021) reframes offline RL as sequence modeling. Rather than learning Q-functions or value functions, DT trains a causal transformer to predict the next action given a history of states, actions, and return-to-go (RTG) values — the cumulative reward remaining in the trajectory. At inference time, a target return is specified, and the model generates actions conditioned on that return.

DT requires trajectory-level data (ordered sequences of transitions) and uses a context window of fixed length. The key design choice is the target return: setting it too high causes the model to generate unreachable behavior, while setting it too low leaves performance on the table. DT is robust to distributional shift by design (it conditions on return rather than trying to maximize it), but it cannot improve beyond the best trajectories in the training data. It works for both discrete and continuous action spaces.
