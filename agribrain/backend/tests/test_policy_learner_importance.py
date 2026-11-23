"""Pins the truncated importance-sampling correction in PolicyLearner.update
(#8a). Covers: weight==1 reproduces on-policy exactly, the V-trace truncation
caps the weight at is_clip, and an off-policy sample is reweighted upward.
"""
import numpy as np

from src.models.policy_learner import PolicyLearner


def _freeze_baseline(pl, b=0.0):
    pl._baseline = b
    pl._baseline_count = 10 ** 9  # record()'s nudge becomes negligible


def _pi(theta, f, a):
    z = theta @ f
    z -= z.max()
    e = np.exp(z)
    return float(e[a] / e.sum())


def test_is_weight_unity_matches_on_policy():
    # behavior_prob == current pi(a|s) at update time => weight 1 => identical
    # to recording with no behavior probability (on-policy).
    rng = np.random.default_rng(3)
    theta0 = rng.normal(scale=0.2, size=(3, 10))
    feats = rng.normal(scale=0.2, size=(8, 10))
    acts = rng.integers(0, 3, 8)
    rews = rng.normal(0.0, 0.3, 8)

    on = PolicyLearner(lr=0.05, max_buffer=10 ** 6)
    _freeze_baseline(on)
    for f, a, r in zip(feats, acts, rews):
        on.record(f.copy(), int(a), float(r))
    t_on = on.update(theta0.copy())

    iw = PolicyLearner(lr=0.05, max_buffer=10 ** 6)
    _freeze_baseline(iw)
    for f, a, r in zip(feats, acts, rews):
        iw.record(f.copy(), int(a), float(r), behavior_prob=_pi(theta0, f, int(a)))
    t_iw = iw.update(theta0.copy())

    assert np.allclose(t_on, t_iw, atol=1e-12), "unit IS weight must match on-policy"


def test_is_weight_is_truncated_at_is_clip():
    # A sample vanishingly rare under the behavior policy has an unbounded raw
    # weight; truncation must cap it at is_clip. We verify by comparing against
    # a behavior_prob chosen to give *exactly* weight == is_clip: identical
    # updates prove the huge raw weight was clipped, not used.
    rng = np.random.default_rng(4)
    theta0 = rng.normal(scale=0.1, size=(3, 10))
    f = rng.normal(scale=0.1, size=10)
    a, r, clip = 1, 0.5, 2.0
    pcur = _pi(theta0, f, a)

    capped = PolicyLearner(lr=0.05, is_clip=clip, max_buffer=10)
    _freeze_baseline(capped)
    capped.record(f.copy(), a, r, behavior_prob=1e-9)  # raw weight ~ 1e8
    t_capped = capped.update(theta0.copy())

    exact = PolicyLearner(lr=0.05, is_clip=clip, max_buffer=10)
    _freeze_baseline(exact)
    exact.record(f.copy(), a, r, behavior_prob=pcur / clip)  # raw weight == clip
    t_exact = exact.update(theta0.copy())

    assert np.allclose(t_capped, t_exact, atol=1e-10), "raw weight not truncated to is_clip"

    on = PolicyLearner(lr=0.05, is_clip=clip, max_buffer=10)
    _freeze_baseline(on)
    on.record(f.copy(), a, r)  # weight 1
    t_on = on.update(theta0.copy())
    assert not np.allclose(t_capped, t_on, atol=1e-10), "clipped (w=2) must differ from on-policy (w=1)"


def test_off_policy_sample_is_upweighted():
    # behavior_prob < current pi(a|s) => weight > 1 => larger step than on-policy
    # (same direction), with no grad clipping at this small scale.
    rng = np.random.default_rng(5)
    theta0 = rng.normal(scale=0.1, size=(3, 10))
    f = rng.normal(scale=0.1, size=10)
    a, r = 2, 0.5
    pcur = _pi(theta0, f, a)

    on = PolicyLearner(lr=0.05, max_buffer=10)
    _freeze_baseline(on)
    on.record(f.copy(), a, r)
    d_on = on.update(theta0.copy()) - theta0

    off = PolicyLearner(lr=0.05, max_buffer=10)
    _freeze_baseline(off)
    off.record(f.copy(), a, r, behavior_prob=pcur / 3.0)  # weight ~3
    d_off = off.update(theta0.copy()) - theta0

    cos = float(d_on.ravel() @ d_off.ravel() /
                (np.linalg.norm(d_on) * np.linalg.norm(d_off)))
    assert cos > 0.999, "upweighted step should be same direction as on-policy"
    assert np.linalg.norm(d_off) > 1.5 * np.linalg.norm(d_on), "off-policy step should be larger"
