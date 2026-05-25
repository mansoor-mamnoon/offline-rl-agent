# Limitations

This document honestly describes the known limitations of OfflineRL-Lab. Understanding these limitations is essential for deciding whether the framework is appropriate for a given use case.

## OPE Instability Under Distribution Shift

All OPE estimators in this framework have known failure modes under distribution shift. Importance sampling estimators (IS, WIS) can produce arbitrarily biased estimates when the evaluation policy takes actions with very different probability than the behavior policy, even with importance ratio clipping. FQE estimates can have high bias when the evaluation policy visits states not well-represented in the training data. DR provides some robustness, but only when at least one of its components (model or IS weights) is reasonably accurate.

In practice, OPE estimates should be treated as directional signals: a policy with consistently higher OPE estimates than another is likely better, but the absolute value of the estimates should not be trusted. The confidence intervals produced by bootstrapping capture sampling variance, not the total estimation error including model misspecification and distribution shift.

## Traffic Simulator Fidelity

The traffic routing simulator is a simplified model intended for offline RL research. It does not model BGP routing tables, CDN edge cache hit rates, TCP retransmit behavior, connection pooling, or correlated failure cascades. Real traffic routing involves significantly more complex and heterogeneous dynamics. Policies trained on this simulator should not be expected to transfer to real infrastructure without substantial additional validation.

The diurnal traffic pattern uses a simple sinusoid; real traffic has day-of-week effects, event-driven spikes, geographic variation, and long-range correlations not captured by the model. Incident arrival processes use independent Poisson models, while real incidents (hardware failures, configuration errors, DDoS attacks) are correlated and have structured spatial and temporal patterns.

## Shield Over-Conservatism

The policy shield can over-constrain behavior when the training dataset has sparse action coverage. In this regime, most proposed actions are scored as out-of-distribution, and the shield frequently falls back to behavioral policy actions. The effective policy then closely mimics the behavior policy regardless of what the offline RL algorithm learned.

This is particularly problematic for the hospital and traffic environments where the behavior policy may be suboptimal. The shield provides safety guarantees relative to the training data distribution, but that distribution may itself be unsafe. The k-NN threshold parameter requires careful tuning: too high and the shield blocks too many actions; too low and it fails to catch genuinely dangerous OOD behavior.

## DT Needs Trajectory-Level Data

Decision Transformer requires data organized as complete trajectories with return-to-go annotations, not as independently sampled transitions. If the source dataset does not include episode boundaries and cumulative rewards, the TrajectoryReplayBuffer must infer these from done flags and accumulated rewards, which may produce incorrect RTG values if episodes are truncated.

DT also requires choosing a target return at inference time. Setting this too high (above the best trajectory in the dataset) causes the model to extrapolate into behavior it has never seen, often producing poor actions. Setting it too low leaves performance on the table. The optimal target return must be tuned empirically, which requires online evaluation — creating a chicken-and-egg problem in offline settings.

## No D4RL Integration

OfflineRL-Lab uses its own dataset format and environments rather than integrating with D4RL (Fu et al., 2020). This means results cannot be directly compared against published baselines on HalfCheetah, Hopper, or other Gym/MuJoCo benchmarks without additional conversion. The choice was made to keep the installation simple (D4RL requires MuJoCo licenses and complex environment setup), but it limits external reproducibility.

Users wanting to benchmark on standard datasets should write a converter from D4RL's HDF5 format to the expected keys (observations, actions, rewards, next_observations, dones, episode_ids).

## Scalability of Coverage Estimation

The dataset diagnostics module computes coverage scores using a discretized grid over state space, capped at 5 dimensions for tractability. For environments with 32-dimensional states like the traffic routing simulator, the coverage computation uses only the first 5 state dimensions. This may not capture coverage in the full state space.

Similarly, k-NN support estimation in the policy shield does not scale beyond roughly 1 million transitions without approximate nearest-neighbor search. For large datasets, consider using FAISS or ScaNN for the k-NN index, which would require additional dependencies not included in the default installation.
