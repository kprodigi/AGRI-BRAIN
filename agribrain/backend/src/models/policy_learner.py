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
    ) -> None:
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = lr
        self.max_buffer = max_buffer

        self._buffer: List[Tuple[np.ndarray, int, float]] = []
        self._baseline: float = 0.0
        self._baseline_count: int = 0

    @staticmethod
    def is_enabled() -> bool:
        """Check if online learning is enabled via environment variable."""
        val = str(SETTINGS.online_learning).lower()
        return val in ("true", "1", "yes")

    def record(self, features: np.ndarray, action: int, reward: float) -> None:
        """Add a (features, action, reward) tuple to the replay buffer.

        Parameters
        ----------
        features : feature vector phi(s) of shape (n_features,).
        action : action index taken.
        reward : observed reward.
        """
        self._buffer.append((features.copy(), action, reward))
        if len(self._buffer) > self.max_buffer:
            self._buffer.pop(0)

        # Running mean baseline
        self._baseline_count += 1
        self._baseline += (reward - self._baseline) / self._baseline_count

    def update(self, theta: np.ndarray) -> np.ndarray:
        """Apply REINFORCE gradient update to a copy of the policy weights.

        Uses a batch-gradient approximation: all buffered samples compute
        their gradient against the current theta snapshot rather than the
        behavior policy that generated the action. This is acceptable for
        the small buffer sizes and learning rates used here, but does not
        include importance-sampling correction for off-policy samples.

        Parameters
        ----------
        theta : current policy weight matrix of shape (n_actions, n_features).

        Returns
        -------
        Updated theta matrix (does not modify the input).
        """
        if not self._buffer:
            return theta.copy()

        theta_new = theta.copy()
        grad = np.zeros_like(theta_new)

        for features, action, reward in self._buffer:
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

            # Accumulate: outer product of grad_log_pi and features
            grad += np.outer(grad_log_pi, features) * advantage

        # Average over buffer and apply update
        grad /= len(self._buffer)
        np.clip(grad, -1.0, 1.0, out=grad)
        theta_new += self.lr * grad

        # Clear buffer after update
        self._buffer.clear()

        return theta_new

    def reset(self) -> None:
        """Clear the replay buffer and reset baseline."""
        self._buffer.clear()
        self._baseline = 0.0
        self._baseline_count = 0
