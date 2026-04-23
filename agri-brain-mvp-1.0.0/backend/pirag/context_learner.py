"""Online adaptation of policy and context weights.

Three learner classes:

1. ``ContextMatrixLearner`` (primary): learns the full THETA_CONTEXT (3×5)
   weight matrix via REINFORCE policy gradient with sign constraints.

2. ``PolicyDeltaLearner``: learns a (3, 10) additive correction ΔΘ on
   top of the hand-calibrated THETA matrix via REINFORCE with an
   empirical-Bayes Gaussian prior centred at zero, a per-entry
   magnitude cap at 25 percent of ``|THETA_initial|``, and an optional
   sign constraint. Entries the hand-calibration set to zero are held
   at zero; entries with strong priors can still move but only inside a
   25-percent band around their initial value. Replaces the earlier
   forecast-only learner by treating every THETA column as learnable
   while anchoring the whole matrix on domain priors.

3. ``ContextRuleLearner`` (legacy): per-feature scalar weights via
   exponential-weight bandit updates. Retained for backward compatibility.

The REINFORCE update for THETA_CONTEXT is:

    THETA_CONTEXT ← THETA_CONTEXT + η · (e_a − π) · ψ^T · (R − R̄)

The update for the policy delta is the same softmax-policy gradient
over the full φ plus a shrinkage term and the magnitude/sign rails:

    ΔΘ ← clip( (1 − η λ) · ΔΘ + η · (e_a − π) · φ^T · (R − R̄),
               −cap · |Θ_initial|, +cap · |Θ_initial| )

Sign constraints keep learned weights physically interpretable (e.g.
compliance violations always disfavor cold chain, freshness always
favours cold chain). The 25 percent magnitude cap on its own already
precludes sign flips for non-zero entries; the sign clamp is defence
in depth for future cap-fraction changes.
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

    def save_state(self) -> Dict[str, Any]:
        """Serialise the learnable state for checkpointing or cross-run
        persistence. Returns a JSON-friendly dict that round-trips through
        :meth:`load_state`.
        """
        return {
            "theta": self.theta.tolist(),
            "slca_amp_coeff": float(self.slca_amp_coeff),
            "temporal_base": float(self.temporal_base),
            "temporal_scale": float(self.temporal_scale),
            "reward_baseline": float(self.reward_baseline),
            "n_updates": int(self.n_updates),
        }

    def load_state(self, state: Dict[str, Any]) -> None:
        """Restore learnable state produced by :meth:`save_state`.

        History entries are not persisted (they are diagnostic, not
        load-bearing) so the post-load summary reports only activity
        since the restore point.
        """
        theta = np.asarray(state["theta"], dtype=np.float64)
        if theta.shape != self.theta.shape:
            raise ValueError(
                f"state theta shape {theta.shape} does not match "
                f"learner theta shape {self.theta.shape}"
            )
        self.theta = theta
        self.slca_amp_coeff = float(state.get("slca_amp_coeff", self.slca_amp_coeff))
        self.temporal_base = float(state.get("temporal_base", self.temporal_base))
        self.temporal_scale = float(state.get("temporal_scale", self.temporal_scale))
        self.reward_baseline = float(state.get("reward_baseline", 0.0))
        self.n_updates = int(state.get("n_updates", 0))
        self._history.clear()


class PolicyDeltaLearner:
    """Online REINFORCE learner for the full policy matrix THETA (3, 10).

    Learns a (3, 10) additive correction delta on top of the
    hand-calibrated THETA initial values. The hand-calibrated matrix
    stays fixed; only the delta moves with training. Two safety rails
    keep learning well-behaved so the ablation structure survives the
    richer learner:

    1. Per-entry magnitude cap: ``|delta[i, j]| <= cap_frac *
       |initial_theta[i, j]|``. The default ``cap_frac = 0.25`` means
       each entry can move at most 25 percent from its hand-calibrated
       value. Entries with zero initial magnitude stay at zero (the
       hand-calibration chose zero deliberately, learning respects that).
    2. Sign constraint: entries whose effective sign would flip from the
       initial sign get clamped back to zero. Preserves per-entry
       interpretability (cold chain always rewards freshness, recovery
       always punishes spoilage urgency). The 25 percent cap on its own
       already precludes sign flips; the constraint is defence in depth
       for future cap-fraction changes.

    Delta is zero-initialised so step 0 is bit-identical to the
    hand-calibrated policy. A zero-mean Gaussian prior on the delta
    (equivalent to L2 weight decay) pulls entries with no reward signal
    back toward zero with a half-life of ``log(2) / (lr * prior_precision)``
    update steps. Entries that do carry signal settle at a non-trivial
    value inside the magnitude cap.

    Parameters
    ----------
    initial_theta : (3, 10) hand-calibrated policy matrix. The learner
        stores a copy and uses it as both the prior mean and the anchor
        for the magnitude cap.
    learning_rate : gradient step size. Small because phi entries are
        clipped to known ranges so the raw gradient magnitude is already
        bounded; a modest step keeps updates stable.
    prior_precision : lambda in the zero-mean Gaussian prior. Higher
        values pull the delta back to zero more aggressively.
    baseline_decay : exponential moving average decay for the reward
        baseline used for variance reduction.
    grad_clip : per-element gradient clipping bound.
    magnitude_cap_fraction : the fraction of ``|initial_theta|`` that
        bounds each delta entry. Defaults to 0.25.
    sign_constrained : when True (default), clamp entries whose
        effective sign would flip from the initial sign.
    """

    def __init__(
        self,
        initial_theta: np.ndarray,
        learning_rate: float = 0.003,
        prior_precision: float = 0.10,
        baseline_decay: float = 0.95,
        grad_clip: float = 0.5,
        magnitude_cap_fraction: float = 0.25,
        sign_constrained: bool = True,
    ) -> None:
        if initial_theta.shape != (3, 10):
            raise ValueError(
                f"initial_theta must be shape (3, 10), got {initial_theta.shape}"
            )
        self.initial_theta: np.ndarray = initial_theta.astype(np.float64).copy()
        # Delta starts at zero so the effective matrix at step 0 is exactly
        # the hand-calibrated policy.
        self.theta_delta: np.ndarray = np.zeros_like(self.initial_theta)

        self.lr = float(learning_rate)
        self.prior_precision = float(prior_precision)
        self.baseline_decay = float(baseline_decay)
        self.grad_clip = float(grad_clip)
        self.cap_fraction = float(magnitude_cap_fraction)
        self.sign_constrained = bool(sign_constrained)

        self._sign_mask = np.sign(self.initial_theta)
        self._magnitude_bound = np.abs(self.initial_theta) * self.cap_fraction

        self.reward_baseline: float = 0.0
        self.n_updates: int = 0
        self._history: List[Dict[str, Any]] = []

    def get_theta_delta(self) -> np.ndarray:
        """Current (3, 10) correction added to the hand-calibrated THETA."""
        return self.theta_delta.copy()

    def get_effective_theta(self) -> np.ndarray:
        """The hand-calibrated THETA plus the learned correction."""
        return self.initial_theta + self.theta_delta

    def update(
        self,
        phi: np.ndarray,
        action: int,
        probs: np.ndarray,
        reward: float,
    ) -> None:
        """REINFORCE gradient step with shrinkage, magnitude cap, and
        optional sign constraint.

        Parameters
        ----------
        phi : (10,) full state feature vector from build_feature_vector.
        action : taken action index (0, 1, 2).
        probs : (3,) softmax probability vector at decision time.
        reward : observed scalar reward.
        """
        if phi.shape != (10,):
            raise ValueError(f"phi must be shape (10,), got {phi.shape}")
        if probs.shape != (3,):
            raise ValueError(f"probs must be shape (3,), got {probs.shape}")

        self.n_updates += 1
        self.reward_baseline = (
            self.baseline_decay * self.reward_baseline
            + (1.0 - self.baseline_decay) * float(reward)
        )
        advantage = float(reward) - self.reward_baseline

        e_a = np.zeros(3, dtype=np.float64)
        e_a[action] = 1.0

        # Policy gradient over the full (3, 10) matrix: (e_a - pi) ⊗ phi.
        grad = np.outer(e_a - probs, phi) * advantage
        grad = np.clip(grad, -self.grad_clip, self.grad_clip)

        # Shrinkage toward zero (Gaussian log-prior contribution).
        self.theta_delta *= (1.0 - self.lr * self.prior_precision)
        self.theta_delta += self.lr * grad

        # Per-entry magnitude cap. Entries with zero initial magnitude
        # have zero bound, so they are held at zero (hand-calibration
        # chose zero deliberately).
        self.theta_delta = np.clip(
            self.theta_delta, -self._magnitude_bound, self._magnitude_bound
        )

        # Sign constraint: zero out any entry whose effective value
        # would flip sign from the initial. The 25 percent cap precludes
        # this for non-zero entries in normal operation, so the
        # constraint is defence in depth.
        if self.sign_constrained:
            effective = self.initial_theta + self.theta_delta
            flipped = (effective * self._sign_mask) < 0.0
            if np.any(flipped):
                self.theta_delta[flipped] = 0.0

        self._history.append({
            "advantage": float(advantage),
            "grad_norm": float(np.linalg.norm(grad)),
            "delta_norm": float(np.linalg.norm(self.theta_delta)),
        })

    def summary(self) -> Dict[str, Any]:
        """Detailed statistics for paper reporting."""
        if self.n_updates == 0:
            max_entry = 0.0
            max_fractional_entry = 0.0
            mean_adv = 0.0
        else:
            max_entry = float(np.abs(self.theta_delta).max())
            # Fractional drift, per entry: |delta| / |initial|. Entries
            # with zero initial are excluded from this stat (they cannot
            # drift). A max close to cap_fraction means the learner is
            # hitting the magnitude cap somewhere.
            nonzero = np.abs(self.initial_theta) > 0
            fractional = np.zeros_like(self.initial_theta)
            fractional[nonzero] = (
                np.abs(self.theta_delta[nonzero]) / np.abs(self.initial_theta[nonzero])
            )
            max_fractional_entry = float(fractional.max())
            mean_adv = (
                float(np.mean([h["advantage"] for h in self._history]))
                if self._history else 0.0
            )
        return {
            "n_updates": self.n_updates,
            "final_theta_delta": self.theta_delta.tolist(),
            "effective_theta": self.get_effective_theta().tolist(),
            "delta_frobenius_norm": float(np.linalg.norm(self.theta_delta)),
            "max_delta_entry": max_entry,
            "max_fractional_drift": max_fractional_entry,
            "reward_baseline": float(self.reward_baseline),
            "mean_advantage": mean_adv,
            "learning_rate": self.lr,
            "prior_precision": self.prior_precision,
            "magnitude_cap_fraction": self.cap_fraction,
            "sign_constrained": self.sign_constrained,
        }

    def reset(self) -> None:
        """Reset learned delta, baseline, and history to their initial state."""
        self.theta_delta = np.zeros_like(self.initial_theta)
        self.reward_baseline = 0.0
        self.n_updates = 0
        self._history.clear()

    def save_state(self) -> Dict[str, Any]:
        """Serialise learnable state for checkpointing or cross-run
        persistence. Round-trips through :meth:`load_state`.
        """
        return {
            "theta_delta": self.theta_delta.tolist(),
            "reward_baseline": float(self.reward_baseline),
            "n_updates": int(self.n_updates),
            "learning_rate": float(self.lr),
            "prior_precision": float(self.prior_precision),
            "magnitude_cap_fraction": float(self.cap_fraction),
            "sign_constrained": bool(self.sign_constrained),
        }

    def load_state(self, state: Dict[str, Any]) -> None:
        """Restore learnable state produced by :meth:`save_state`.

        Hyperparameters in the saved state are informational: the loaded
        learner keeps whatever settings it was constructed with, so the
        trainer can resume with different configuration. History is not
        persisted.
        """
        theta_delta = np.asarray(state["theta_delta"], dtype=np.float64)
        if theta_delta.shape != (3, 10):
            raise ValueError(
                f"state theta_delta shape {theta_delta.shape} must be (3, 10)"
            )
        self.theta_delta = theta_delta
        self.reward_baseline = float(state.get("reward_baseline", 0.0))
        self.n_updates = int(state.get("n_updates", 0))
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
