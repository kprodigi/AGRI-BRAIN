"""Pins the off-policy-staleness claim in PolicyLearner.update().

The docstring states that the mini-batch REINFORCE update (all buffered
gradients evaluated at the *current* theta snapshot, i.e. mildly off-policy)
tracks the on-policy *online* update to within an O(lr) error that vanishes
as lr -> 0. This test verifies exactly that, isolating the staleness term:

  * mini-batch: record K samples, one averaged update of size ``lr``.
  * online:     K sequential single-sample updates of size ``lr / K`` so the
                *total* step magnitude matches the single mini-batch step;
                each gradient is recomputed at the (drifting) current theta.

The only difference between the two parameter deltas is then the off-policy
staleness (theta0 vs the drifting theta_i). If that staleness is first-order
bounded, the relative gap must shrink ~10x when lr shrinks 10x, and the two
deltas must be nearly collinear.

Data is deliberately small-scale so the gradient-clip in update() never
triggers (clip(mean) vs mean(clip) is a separate, intended nonlinearity and
would otherwise confound the staleness measurement).
"""
import numpy as np

from src.models.policy_learner import PolicyLearner


def _make_data(rng, K=200, n_actions=3, n_features=10):
    # Small scale -> per-sample gradient entries are << 1, so the [-1, 1]
    # clip in update() is never active and we measure pure staleness.
    feats = rng.normal(scale=0.3, size=(K, n_features))
    actions = rng.integers(0, n_actions, K)
    rewards = rng.normal(scale=0.3, size=K)
    rewards = rewards - rewards.mean()  # centered: baseline stays ~0
    return feats, actions, rewards


def _freeze_baseline(pl, b=0.0):
    # Pre-load a huge baseline count so record()'s running-mean nudge
    # ((r - b) / count) is negligible -> both modes share one fixed baseline,
    # leaving gradient staleness as the only moving part.
    pl._baseline = b
    pl._baseline_count = 10 ** 9


def _delta_minibatch(theta0, feats, actions, rewards, lr):
    pl = PolicyLearner(n_actions=theta0.shape[0], n_features=theta0.shape[1],
                       lr=lr, max_buffer=10 ** 6)
    _freeze_baseline(pl)
    for f, a, r in zip(feats, actions, rewards):
        pl.record(f.copy(), int(a), float(r))
    return pl.update(theta0.copy()) - theta0


def _delta_online(theta0, feats, actions, rewards, lr_total):
    k = len(feats)
    pl = PolicyLearner(n_actions=theta0.shape[0], n_features=theta0.shape[1],
                       lr=lr_total / k, max_buffer=10 ** 6)
    _freeze_baseline(pl)
    theta = theta0.copy()
    for f, a, r in zip(feats, actions, rewards):
        pl.record(f.copy(), int(a), float(r))
        theta = pl.update(theta)  # buffer of 1, then flushed
    return theta - theta0


def _cos_relgap(theta0, feats, actions, rewards, lr):
    dm = _delta_minibatch(theta0, feats, actions, rewards, lr).ravel()
    do = _delta_online(theta0, feats, actions, rewards, lr).ravel()
    cos = float(dm @ do / (np.linalg.norm(dm) * np.linalg.norm(do)))
    relgap = float(np.linalg.norm(dm - do) / max(np.linalg.norm(dm), 1e-12))
    return cos, relgap


def test_policy_learner_minibatch_tracks_online():
    rng = np.random.default_rng(0)
    n_actions, n_features = 3, 10
    theta0 = rng.normal(scale=0.1, size=(n_actions, n_features))
    feats, actions, rewards = _make_data(rng, K=200,
                                         n_actions=n_actions, n_features=n_features)

    cos_hi, rel_hi = _cos_relgap(theta0, feats, actions, rewards, lr=1e-1)
    cos_lo, rel_lo = _cos_relgap(theta0, feats, actions, rewards, lr=1e-2)

    # Same direction even at the larger learning rate.
    assert cos_hi > 0.99, f"minibatch/online not collinear at lr=0.1: cos={cos_hi}"
    # First-order (O(lr)) consistency: the relative gap shrinks as lr shrinks.
    assert rel_lo < rel_hi, f"staleness gap did not shrink with lr: {rel_hi} -> {rel_lo}"
    # The gap shrinks roughly proportionally to lr (10x lr drop -> ~>=3x gap drop).
    assert rel_hi / max(rel_lo, 1e-12) > 3.0, (
        f"gap not first-order in lr: {rel_hi} -> {rel_lo}")
    # At the operating-scale lr the mini-batch update is within ~1 % of online.
    assert rel_lo < 0.01, f"staleness gap at lr=0.01 too large: {rel_lo}"
    assert cos_lo > 0.9999, f"minibatch/online cosine at lr=0.01 too low: {cos_lo}"
