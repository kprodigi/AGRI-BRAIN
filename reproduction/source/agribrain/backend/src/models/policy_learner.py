"""Online policy learning via REINFORCE with replay buffer.

Implements a lightweight policy gradient updater that can optionally
refine the softmax routing weights between episodes. Disabled by
default (controlled by ONLINE_LEARNING environment variable or
Policy field).

References
----------
    - Williams, R.J. (1992). Simple statistical gradient-following
      algorithms for connectionist reinforcement learning.
      Machine Learning, 8(3), 229-256.
"""
from __future__ import annotations

from typing import List, Tuple
from src.settings import SETTINGS

import numpy as np


class PolicyLearner:
    """REINFORCE-style policy gradient updater with replay buffer.

    Parameters
    ----------
    n_actions : number of routing actions.
    n_features : dimensionality of the feature vector.
    lr : learning rate for gradient updates.
    max_buffer : maximum replay buffer size.
    freeze : if True, retain state but ignore future records and updates.
    """

    def __init__(
        self,
        n_actions: int = 3,
        # n_features matches the canonical phi(s) returned by
        # action_selection.build_feature_vector, which grew to 10 dims
        # in the 2025-04 forecast-uncertainty extension. The default
        # here is informational; record() does not enforce it (it stores
        # whatever array shape is passed) and update() uses
        # np.zeros_like(theta) so the gradient buffer matches the
        # provided theta. The previous default of 6 referred to the
        # original physics-and-ops state vector.
        n_features: int = 10,
        lr: float = 0.001,
        max_buffer: int = 2000,
        is_clip: float = 10.0,
        freeze: bool = False,
    ) -> None:
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = lr
        self.max_buffer = max_buffer
        # Truncated importance-sampling cap (V-trace style). Applied only when
        # a behavior probability is recorded; bounds the variance of the
        # off-policy correction so a rare-under-behavior sample cannot destabilize
        # the gradient.
        self.is_clip = is_clip
        self.frozen = bool(freeze)

        # Buffer holds (features, action, reward, behavior_prob); behavior_prob
        # is None for on-policy callers (weight 1).
        self._buffer: List[Tuple[np.ndarray, int, float, "float | None"]] = []
        self._baseline: float = 0.0
        self._baseline_count: int = 0

    @staticmethod
    def is_enabled() -> bool:
        """Check if online learning is enabled via environment variable."""
        val = str(SETTINGS.online_learning).lower()
        return val in ("true", "1", "yes")

    def record(self, features: np.ndarray, action: int, reward: float,
               behavior_prob: "float | None" = None) -> None:
        """Add a (features, action, reward[, behavior_prob]) tuple to the buffer.

        Parameters
        ----------
        features : feature vector phi(s) of shape (n_features,).
        action : action index taken.
        reward : observed reward.
        behavior_prob : optional pi_behavior(a|s) -- the probability the
            behavior policy assigned to ``action`` when it was taken. When
            supplied, ``update()`` reweights this sample by a truncated
            importance weight to correct for off-policy staleness. When
            omitted (the default, and what every current caller passes) the
            sample is treated as on-policy (weight 1), preserving the prior
            mini-batch REINFORCE behaviour exactly.

        The 2026-05 hardening adds a shape assertion: prior to this,
        the buffer silently accepted any 1-D array, so a caller that
        passed a 6-dim phi (legacy) into a 10-dim learner produced
        misaligned theta gradients on update without raising. The
        assertion fails loudly so feature-vector regressions surface
        at record-time rather than as silent learning drift.
        """
        if self.frozen:
            return
        if features.ndim != 1 or features.shape[0] != self.n_features:
            raise ValueError(
                f"PolicyLearner.record expected features of shape "
                f"({self.n_features},); got shape={features.shape}. "
                f"This usually means the caller is using an older "
                f"phi(s) layout; see action_selection.build_feature_vector."
            )
        if not (0 <= action < self.n_actions):
            raise ValueError(
                f"PolicyLearner.record expected action in [0, {self.n_actions}); "
                f"got {action}."
            )
        self._buffer.append((features.copy(), action, reward, behavior_prob))
        if len(self._buffer) > self.max_buffer:
            self._buffer.pop(0)

        # Running mean baseline
        self._baseline_count += 1
        self._baseline += (reward - self._baseline) / self._baseline_count

    def update(self, theta: np.ndarray) -> np.ndarray:
        """Apply REINFORCE gradient update to a copy of the policy weights.

        This is a **mini-batch REINFORCE update with a running-mean
        baseline** (Williams 1992), not an approximation: the buffered
        ``(s, a, R)`` tuples are replayed, their score-function gradients
        ``(e_a - pi(.|s)) * (R - b)`` are averaged, clipped to [-1, 1] for
        stability, and applied once. The gradients are evaluated at the
        *current* ``theta`` snapshot rather than at the behavior policy that
        generated each action, so the replayed samples are mildly off-policy
        and no importance-sampling correction is applied. In the operating
        regime used here (``max_buffer <= 2000``, ``lr <= 1e-3``, buffer
        flushed every update) the policy drift within a single buffer is
        negligible, so the off-policy bias is first-order bounded: the
        mini-batch update tracks the online (per-sample) REINFORCE update to
        within an O(lr) angular error that vanishes as ``lr -> 0`` (measured
        relative gap < 1.5e-4 at lr=0.1 and < 1e-5 at lr=0.01, cosine
        similarity 1.000). That equivalence is pinned by
        ``test_policy_learner_convergence`` in the backend suite.

        **Off-policy correction.** When a sample was recorded with a behavior
        probability (``record(..., behavior_prob=p_b)``), its gradient is
        reweighted by a *truncated* importance weight
        ``w = min(pi_current(a|s) / p_b, is_clip)`` (V-trace-style truncation,
        cap ``self.is_clip``) so larger learning rates / buffers remain
        unbiased without unbounded variance. Samples recorded without a
        behavior probability use ``w = 1`` (on-policy), which is the default
        and reproduces the prior behaviour exactly.

        Parameters
        ----------
        theta : current policy weight matrix of shape (n_actions, n_features).

        Returns
        -------
        Updated theta matrix (does not modify the input).
        """
        if self.frozen or not self._buffer:
            return theta.copy()

        theta_new = theta.copy()
        grad = np.zeros_like(theta_new)

        for features, action, reward, behavior_prob in self._buffer:
            # Softmax probabilities
            logits = theta_new @ features
            logits -= logits.max()
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum()

            # Gradient of log pi(a|s): e_a - pi
            grad_log_pi = np.zeros(self.n_actions)
            grad_log_pi[action] = 1.0
            grad_log_pi -= probs

            # Advantage: R - baseline
            advantage = reward - self._baseline

            # Truncated importance weight for off-policy samples (w=1 when no
            # behavior probability was recorded -> on-policy mini-batch).
            if behavior_prob is not None and behavior_prob > 0.0:
                is_weight = min(float(probs[action]) / float(behavior_prob),
                                self.is_clip)
            else:
                is_weight = 1.0

            # Accumulate: outer product of grad_log_pi and features
            grad += np.outer(grad_log_pi, features) * advantage * is_weight

        # Average over buffer and apply update
        grad /= len(self._buffer)
        np.clip(grad, -1.0, 1.0, out=grad)
        theta_new += self.lr * grad

        # Clear buffer after update
        self._buffer.clear()

        return theta_new

    def freeze_updates(self) -> None:
        """Disable future record/update mutations without clearing state."""

        self.frozen = True

    def freeze_summary(self) -> dict:
        """Return auditable state for the retained-evaluation protocol."""

        return {
            "learner_frozen": bool(self.frozen),
            "buffer_size": len(self._buffer),
            "baseline": float(self._baseline),
            "baseline_count": int(self._baseline_count),
        }

    def reset(self) -> None:
        """Clear the replay buffer and reset baseline."""
        self._buffer.clear()
        self._baseline = 0.0
        self._baseline_count = 0
