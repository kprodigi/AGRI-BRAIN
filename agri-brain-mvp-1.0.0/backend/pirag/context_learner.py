"""Online adaptation of context modifier weights.

Two learner classes:

1. ``ContextMatrixLearner`` (primary): learns the full THETA_CONTEXT (3×5)
   weight matrix via REINFORCE policy gradient with sign constraints.

2. ``ContextRuleLearner`` (legacy): per-feature scalar weights via
   exponential-weight bandit updates.  Retained for backward compatibility.

The REINFORCE update for THETA_CONTEXT is:

    THETA_CONTEXT ← THETA_CONTEXT + η · (e_a − π) · ψ^T · (R − R̄)

where e_a is the one-hot action vector, π is the softmax probability,
ψ is the context feature vector, R is observed reward, and R̄ is the
running baseline.

Sign constraints ensure that learned weights remain physically
interpretable (e.g., compliance violations always disfavor cold chain).
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np


class ContextMatrixLearner:
    """Online REINFORCE learner for THETA_CONTEXT weight matrix.

    Learns the 3×5 context weight matrix via policy gradient updates,
    with sign constraints to preserve domain-justified directions.

    Parameters
    ----------
    initial_theta : (3, 5) initial THETA_CONTEXT matrix.
    learning_rate : gradient step size (smaller than base policy lr).
    baseline_decay : exponential moving average decay for reward baseline.
    grad_clip : per-element gradient clipping bound.
    sign_constrained : if True, entries cannot flip sign from initial.
    """

    def __init__(
        self,
        initial_theta: np.ndarray,
        learning_rate: float = 0.003,
        baseline_decay: float = 0.95,
        grad_clip: float = 0.5,
        sign_constrained: bool = True,
    ) -> None:
        self.theta = initial_theta.copy()
        self.initial_theta = initial_theta.copy()
        self.lr = learning_rate
        self.baseline_decay = baseline_decay
        self.grad_clip = grad_clip
        self.sign_constrained = sign_constrained

        # Sign mask: +1 for positive, -1 for negative, 0 for zero
        self.sign_mask = np.sign(initial_theta)

        # Running reward baseline for variance reduction
        self.reward_baseline = 0.0
        self.n_updates = 0

        # SLCA amplification coefficient (also learned)
        self.slca_amp_coeff = 0.25
        self.slca_amp_initial = 0.25

        # Temporal modulation parameters (also learned)
        self.temporal_base = 1.3
        self.temporal_scale = 0.6

        self._history: List[Dict[str, Any]] = []

    def get_theta(self) -> np.ndarray:
        """Current learned THETA_CONTEXT matrix."""
        return self.theta.copy()

    def get_slca_amp(self) -> float:
        """Current SLCA amplification coefficient."""
        return self.slca_amp_coeff

    def get_temporal_params(self) -> tuple:
        """Current temporal modulation parameters (base, scale)."""
        return self.temporal_base, self.temporal_scale

    def update(
        self,
        psi: np.ndarray,
        action: int,
        probs: np.ndarray,
        reward: float,
        slca_score: float = 0.0,
    ) -> None:
        """REINFORCE gradient update on THETA_CONTEXT.

        Parameters
        ----------
        psi : (5,) context feature vector (institutional / coordination
            signals). Supply and demand forecast signals are *state*
            features and enter the policy via phi(s), not here.
        action : taken action index (0, 1, 2).
        probs : (3,) softmax probability vector at decision time.
        reward : observed reward.
        slca_score : SLCA composite (for amplification learning).
        """
        self.n_updates += 1
        self.reward_baseline = (
            self.baseline_decay * self.reward_baseline
            + (1 - self.baseline_decay) * reward
        )

        advantage = reward - self.reward_baseline

        # Policy gradient: (e_a - π) ⊗ ψ^T
        e_a = np.zeros(3)
        e_a[action] = 1.0
        grad = np.outer(e_a - probs, psi) * advantage

        # Clip gradient
        grad = np.clip(grad, -self.grad_clip, self.grad_clip)

        # Update THETA_CONTEXT
        self.theta += self.lr * grad

        # Sign constraint: clamp entries that would flip sign
        if self.sign_constrained:
            flipped = (self.theta * self.sign_mask) < 0
            self.theta[flipped] = 0.0

        # Magnitude constraint: cap each entry at its initial absolute value,
        # matching paper Section 3.9 ("caps each entry at its initial absolute
        # value, preventing runaway updates"). Entries with zero initial
        # magnitude are held at zero so sign-constrained learning cannot
        # conjure a direction from noise.
        max_mag = np.abs(self.initial_theta)
        self.theta = np.clip(self.theta, -max_mag, max_mag)

        # Update SLCA amplification coefficient
        slca_grad = advantage * abs(probs[1])
        self.slca_amp_coeff += 0.001 * slca_grad
        self.slca_amp_coeff = float(np.clip(self.slca_amp_coeff, 0.05, 0.50))

        self._history.append({
            "advantage": advantage,
            "grad_norm": float(np.linalg.norm(grad)),
            "theta_norm": float(np.linalg.norm(self.theta)),
            "slca_amp": self.slca_amp_coeff,
        })

    def summary(self) -> Dict[str, Any]:
        """Detailed statistics for paper reporting."""
        return {
            "n_updates": self.n_updates,
            "initial_theta": self.initial_theta.tolist(),
            "final_theta": self.theta.tolist(),
            "theta_change": (self.theta - self.initial_theta).tolist(),
            "theta_change_norm": float(np.linalg.norm(self.theta - self.initial_theta)),
            "max_entry_change": float(np.abs(self.theta - self.initial_theta).max()),
            "sign_preserved": bool(np.all(
                (self.sign_mask == 0) | (np.sign(self.theta) * self.sign_mask >= 0)
            )),
            "initial_slca_amp": self.slca_amp_initial,
            "final_slca_amp": self.slca_amp_coeff,
            "reward_baseline": self.reward_baseline,
            "weight_range": [float(self.theta.min()), float(self.theta.max())],
            "mean_advantage": float(np.mean([h["advantage"] for h in self._history])) if self._history else 0.0,
        }

    def reset(self) -> None:
        """Reset to initial weights."""
        self.theta = self.initial_theta.copy()
        self.slca_amp_coeff = self.slca_amp_initial
        self.temporal_base = 1.3
        self.temporal_scale = 0.6
        self.reward_baseline = 0.0
        self.n_updates = 0
        self._history.clear()


class ContextRuleLearner:
    """Legacy per-feature scalar weight learner.

    Retained for backward compatibility with tests and older code paths.

    Parameters
    ----------
    n_rules : number of context features.
    learning_rate : exponential weight update step size.
    rng : numpy random generator for deterministic behavior.
    """

    def __init__(
        self,
        n_rules: int = 5,
        learning_rate: float = 0.05,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.weights = np.ones(n_rules, dtype=np.float64)
        self.lr = learning_rate
        self.rng = rng or np.random.default_rng(42)
        self._update_log: List[Dict[str, Any]] = []

    def get_weights(self) -> np.ndarray:
        """Current rule weights (passed to compute_context_modifier)."""
        return self.weights.copy()

    def update(
        self,
        rules_fired: List[int],
        reward_with_context: float,
        reward_without_context: float,
    ) -> None:
        """Update weights based on reward comparison."""
        delta = reward_with_context - reward_without_context

        for i in rules_fired:
            if 0 <= i < len(self.weights):
                self.weights[i] *= np.exp(self.lr * delta)

        mean_w = self.weights.mean()
        if mean_w > 0:
            self.weights /= mean_w

        self.weights = np.clip(self.weights, 0.2, 3.0)

        self._update_log.append({
            "rules_fired": list(rules_fired),
            "delta_reward": float(delta),
            "weights_after": self.weights.copy(),
        })

    def summary(self) -> Dict[str, Any]:
        """Statistics for paper reporting."""
        deltas = [e["delta_reward"] for e in self._update_log]
        return {
            "final_weights": self.weights.tolist(),
            "n_updates": len(self._update_log),
            "mean_delta_reward": float(np.mean(deltas)) if deltas else 0.0,
            "weight_range": [float(self.weights.min()), float(self.weights.max())],
        }

    def reset(self) -> None:
        """Reset weights and history."""
        self.weights = np.ones_like(self.weights)
        self._update_log.clear()
