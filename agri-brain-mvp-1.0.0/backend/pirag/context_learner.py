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
        With ``initial_theta`` set to zeros (cold-start ablation) this is
        ignored because there is no domain-justified sign to preserve.
    magnitude_cap_mode : one of
        - ``"abs_initial"`` (legacy): ``|theta| <= |initial_theta|``.
          Entries can only shrink; zero-init entries are locked at zero.
        - ``"relative_delta"``: ``|theta - initial_theta| <= cap_value *
          |initial_theta|`` with an absolute floor for zero-init entries
          (``magnitude_cap_abs_floor``). This is the default for the
          paper's "sign-constrained refinement" claim; cap_value=0.5 means
          ±50% around the initial magnitude.
        - ``"absolute"``: ``|theta| <= cap_value``. Used for cold-start
          where there is no meaningful relative cap from a zero initial.
    magnitude_cap_value : see ``magnitude_cap_mode``. Interpreted as
        fraction-of-initial for ``relative_delta`` and as an absolute
        bound for ``absolute``. Ignored for ``abs_initial``.
    magnitude_cap_abs_floor : absolute floor applied to the per-entry
        delta cap under ``relative_delta``, so zero-initial entries are
        still allowed to move. 0.10 lets a zero-init entry grow toward
        small but non-trivial refinements while staying bounded.
    """

    def __init__(
        self,
        initial_theta: np.ndarray,
        learning_rate: float = 0.02,
        baseline_decay: float = 0.95,
        grad_clip: float = 0.5,
        sign_constrained: bool = True,
        magnitude_cap_mode: str = "relative_delta",
        magnitude_cap_value: float = 0.5,
        magnitude_cap_abs_floor: float = 0.10,
        freeze: bool = False,
        prior_precision: float = 0.05,
        sign_mask_override: np.ndarray | None = None,
    ) -> None:
        self.theta = initial_theta.copy()
        self.initial_theta = initial_theta.copy()
        self.lr = float(learning_rate)
        self.baseline_decay = baseline_decay
        self.grad_clip = grad_clip
        self.sign_constrained = sign_constrained
        self.magnitude_cap_mode = magnitude_cap_mode
        self.magnitude_cap_value = float(magnitude_cap_value)
        self.magnitude_cap_abs_floor = float(magnitude_cap_abs_floor)
        # 2026-04: shrinkage prior precision matches PolicyDeltaLearner
        # so the documented "sign-constrained shrinkage-prior" wording
        # actually applies here. Effective shrinkage per step is
        # (1 - lr * prior_precision); with lr=0.02 and prior_precision
        # =0.05 this is a ~0.1% nudge toward initial_theta per update,
        # plus the existing magnitude/sign rails.
        self.prior_precision = float(prior_precision)
        # When freeze is True, update() is a no-op; the perturbed
        # initial_theta is held fixed across all training calls. Used
        # by the pert_*_static sensitivity ablation.
        self.freeze = bool(freeze)

        # Sign mask: +1 for positive, -1 for negative, 0 for zero.
        # 2026-04: zero-init entries (cold-start ablation, or
        # entries the calibrator left at 0) used to silently lose the
        # sign constraint because np.sign(0) == 0 and the constraint
        # check `(theta * 0) < 0` is always False. The `sign_mask_override`
        # kwarg lets cold-start callers specify the *intended* sign
        # mask (e.g. the production THETA_CONTEXT signs) so the
        # constraint applies consistently across abl ations. When
        # `sign_mask_override` is None, fall back to the np.sign of
        # initial_theta (legacy behaviour).
        if sign_mask_override is not None:
            self.sign_mask = np.asarray(sign_mask_override, dtype=float).copy()
        else:
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
        # Frozen learner: no-op update so the static sensitivity modes
        # (agribrain_pert_10/25/50_static) hold their perturbed initial
        # theta fixed across the entire run.
        if self.freeze or self.lr == 0.0:
            return
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

        # 2026-04 shrinkage prior: pull theta back toward initial_theta
        # before the gradient step. With initial_theta acting as the
        # prior mean and prior_precision controlling the strength, this
        # is a Gaussian-prior MAP step matching the wording in the
        # class docstring and consistent with PolicyDeltaLearner.
        if self.prior_precision > 0.0:
            self.theta = (
                (1.0 - self.lr * self.prior_precision) * self.theta
                + self.lr * self.prior_precision * self.initial_theta
            )

        # Update THETA_CONTEXT
        self.theta += self.lr * grad

        # Sign constraint: clamp entries that would flip sign. Uses
        # `sign_mask` which may be either np.sign(initial_theta) or a
        # caller-provided override (cold-start ablation paths). For
        # entries where sign_mask == 0 (no domain-justified sign), no
        # clamp is applied — that's the documented honest behaviour.
        if self.sign_constrained:
            flipped = (self.theta * self.sign_mask) < 0
            self.theta[flipped] = 0.0

        # Magnitude constraint. See class docstring for the three modes.
        # The default ("relative_delta", 0.5) bounds each entry to within
        # ±50% of its initial magnitude, which is what paper Section 3.9
        # now claims. "abs_initial" reproduces the legacy (shrink-only)
        # behavior for backward compatibility. "absolute" is for cold-start
        # ablation where initial is zero and a relative cap is degenerate.
        if self.magnitude_cap_mode == "abs_initial":
            max_mag = np.abs(self.initial_theta)
            self.theta = np.clip(self.theta, -max_mag, max_mag)
        elif self.magnitude_cap_mode == "relative_delta":
            delta_cap = np.maximum(
                self.magnitude_cap_value * np.abs(self.initial_theta),
                self.magnitude_cap_abs_floor,
            )
            self.theta = np.clip(
                self.theta,
                self.initial_theta - delta_cap,
                self.initial_theta + delta_cap,
            )
        elif self.magnitude_cap_mode == "absolute":
            self.theta = np.clip(
                self.theta,
                -self.magnitude_cap_value,
                self.magnitude_cap_value,
            )
        else:
            raise ValueError(
                f"unknown magnitude_cap_mode={self.magnitude_cap_mode!r}"
            )

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


class RewardShapingLearner:
    """Online REINFORCE learner for the hand-calibrated reward-shaping
    vectors ``SLCA_BONUS``, ``SLCA_RHO_BONUS``, and ``NO_SLCA_OFFSET``.

    Same sign-constrained, shrinkage-anchored delta-with-cap pattern as
    :class:`PolicyDeltaLearner`, but over three (3,) vectors instead of
    a single (3, 10) matrix. Each vector has its own per-entry 25 percent
    magnitude cap, its own sign mask derived from the initial values,
    and an independent zero-mean Gaussian shrinkage prior. Deltas are
    zero-initialised so step 0 is bit-identical to the hand-calibrated
    reward-shaping.

    The update is mode-conditional:

    - ``SLCA_BONUS`` and ``SLCA_RHO_BONUS * rho`` enter the logits only
      in ``agribrain``, ``no_pinn``, and the three context-enabled
      modes (``no_context``, ``mcp_only``, ``pirag_only``). Those modes
      accumulate gradient on the two SLCA vectors.
    - ``NO_SLCA_OFFSET`` enters the logits only in ``no_slca``. That
      mode accumulates gradient on the offset vector.
    - ``hybrid_rl`` and ``static`` do not touch any reward-shaping
      vector and skip this learner entirely (no-op update).

    Shrinkage is applied to every delta on every update (so deltas for
    inactive vectors still decay back toward zero over time), but the
    gradient term is only added to the deltas whose vector contributed
    to the logits in the current step.
    """

    _SLCA_BONUS_MODES = frozenset({
        "agribrain", "no_pinn", "no_context", "mcp_only", "pirag_only",
    })
    _NO_SLCA_MODES = frozenset({"no_slca"})

    def __init__(
        self,
        initial_slca_bonus: np.ndarray,
        initial_slca_rho_bonus: np.ndarray,
        initial_no_slca_offset: np.ndarray,
        learning_rate: float = 0.003,
        prior_precision: float = 0.10,
        baseline_decay: float = 0.95,
        grad_clip: float = 0.5,
        magnitude_cap_fraction: float = 0.25,
        sign_constrained: bool = True,
    ) -> None:
        for name, vec in (
            ("initial_slca_bonus", initial_slca_bonus),
            ("initial_slca_rho_bonus", initial_slca_rho_bonus),
            ("initial_no_slca_offset", initial_no_slca_offset),
        ):
            if np.asarray(vec).shape != (3,):
                raise ValueError(
                    f"{name} must be shape (3,), got {np.asarray(vec).shape}"
                )
        self.initial_slca_bonus: np.ndarray = np.asarray(initial_slca_bonus, dtype=np.float64).copy()
        self.initial_slca_rho: np.ndarray = np.asarray(initial_slca_rho_bonus, dtype=np.float64).copy()
        self.initial_no_slca_offset: np.ndarray = np.asarray(initial_no_slca_offset, dtype=np.float64).copy()

        self.slca_bonus_delta: np.ndarray = np.zeros(3, dtype=np.float64)
        self.slca_rho_delta: np.ndarray = np.zeros(3, dtype=np.float64)
        self.no_slca_offset_delta: np.ndarray = np.zeros(3, dtype=np.float64)

        self.lr = float(learning_rate)
        self.prior_precision = float(prior_precision)
        self.baseline_decay = float(baseline_decay)
        self.grad_clip = float(grad_clip)
        self.cap_fraction = float(magnitude_cap_fraction)
        self.sign_constrained = bool(sign_constrained)

        self._sign_bonus = np.sign(self.initial_slca_bonus)
        self._sign_rho = np.sign(self.initial_slca_rho)
        self._sign_offset = np.sign(self.initial_no_slca_offset)
        self._bound_bonus = np.abs(self.initial_slca_bonus) * self.cap_fraction
        self._bound_rho = np.abs(self.initial_slca_rho) * self.cap_fraction
        self._bound_offset = np.abs(self.initial_no_slca_offset) * self.cap_fraction

        self.reward_baseline: float = 0.0
        self.n_updates: int = 0
        self._history: List[Dict[str, Any]] = []

    def get_slca_bonus_delta(self) -> np.ndarray:
        return self.slca_bonus_delta.copy()

    def get_slca_rho_delta(self) -> np.ndarray:
        return self.slca_rho_delta.copy()

    def get_no_slca_offset_delta(self) -> np.ndarray:
        return self.no_slca_offset_delta.copy()

    def _apply_shrinkage_and_rails(
        self,
        delta: np.ndarray,
        sign_mask: np.ndarray,
        bound: np.ndarray,
        initial: np.ndarray,
        grad: np.ndarray | None,
    ) -> np.ndarray:
        """Shrinkage + optional gradient + magnitude cap + sign clamp."""
        delta = delta * (1.0 - self.lr * self.prior_precision)
        if grad is not None:
            delta = delta + self.lr * np.clip(grad, -self.grad_clip, self.grad_clip)
        delta = np.clip(delta, -bound, bound)
        if self.sign_constrained:
            effective = initial + delta
            flipped = (effective * sign_mask) < 0.0
            if np.any(flipped):
                delta[flipped] = 0.0
        return delta

    def update(
        self,
        action: int,
        probs: np.ndarray,
        reward: float,
        mode: str,
        rho: float,
    ) -> None:
        """REINFORCE step with mode-conditional gradient routing.

        Shrinkage is applied to every delta on every call so inactive
        vectors drift back toward zero; gradients are applied only to
        the vectors active in ``mode``.
        """
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
        policy_grad = (e_a - probs) * advantage

        # Route gradients based on the current mode's logit construction.
        if mode in self._SLCA_BONUS_MODES:
            grad_bonus = policy_grad
            grad_rho = policy_grad * float(rho)
            grad_offset = None
        elif mode in self._NO_SLCA_MODES:
            grad_bonus = None
            grad_rho = None
            grad_offset = policy_grad
        else:
            # hybrid_rl, static, or any unknown mode: shrinkage only.
            grad_bonus = None
            grad_rho = None
            grad_offset = None

        self.slca_bonus_delta = self._apply_shrinkage_and_rails(
            self.slca_bonus_delta, self._sign_bonus, self._bound_bonus,
            self.initial_slca_bonus, grad_bonus,
        )
        self.slca_rho_delta = self._apply_shrinkage_and_rails(
            self.slca_rho_delta, self._sign_rho, self._bound_rho,
            self.initial_slca_rho, grad_rho,
        )
        self.no_slca_offset_delta = self._apply_shrinkage_and_rails(
            self.no_slca_offset_delta, self._sign_offset, self._bound_offset,
            self.initial_no_slca_offset, grad_offset,
        )

        self._history.append({
            "mode": mode,
            "advantage": float(advantage),
            "bonus_norm": float(np.linalg.norm(self.slca_bonus_delta)),
            "rho_norm": float(np.linalg.norm(self.slca_rho_delta)),
            "offset_norm": float(np.linalg.norm(self.no_slca_offset_delta)),
        })

    def summary(self) -> Dict[str, Any]:
        return {
            "n_updates": self.n_updates,
            "slca_bonus_delta": self.slca_bonus_delta.tolist(),
            "slca_rho_delta": self.slca_rho_delta.tolist(),
            "no_slca_offset_delta": self.no_slca_offset_delta.tolist(),
            "effective_slca_bonus": (self.initial_slca_bonus + self.slca_bonus_delta).tolist(),
            "effective_slca_rho_bonus": (self.initial_slca_rho + self.slca_rho_delta).tolist(),
            "effective_no_slca_offset": (self.initial_no_slca_offset + self.no_slca_offset_delta).tolist(),
            "max_delta_entry": float(max(
                np.abs(self.slca_bonus_delta).max() if self.n_updates else 0.0,
                np.abs(self.slca_rho_delta).max() if self.n_updates else 0.0,
                np.abs(self.no_slca_offset_delta).max() if self.n_updates else 0.0,
            )),
            "reward_baseline": float(self.reward_baseline),
            "magnitude_cap_fraction": self.cap_fraction,
            "sign_constrained": self.sign_constrained,
        }

    def reset(self) -> None:
        self.slca_bonus_delta = np.zeros(3, dtype=np.float64)
        self.slca_rho_delta = np.zeros(3, dtype=np.float64)
        self.no_slca_offset_delta = np.zeros(3, dtype=np.float64)
        self.reward_baseline = 0.0
        self.n_updates = 0
        self._history.clear()

    def save_state(self) -> Dict[str, Any]:
        return {
            "slca_bonus_delta": self.slca_bonus_delta.tolist(),
            "slca_rho_delta": self.slca_rho_delta.tolist(),
            "no_slca_offset_delta": self.no_slca_offset_delta.tolist(),
            "reward_baseline": float(self.reward_baseline),
            "n_updates": int(self.n_updates),
            "magnitude_cap_fraction": float(self.cap_fraction),
            "sign_constrained": bool(self.sign_constrained),
        }

    def load_state(self, state: Dict[str, Any]) -> None:
        for key, attr in (
            ("slca_bonus_delta", "slca_bonus_delta"),
            ("slca_rho_delta", "slca_rho_delta"),
            ("no_slca_offset_delta", "no_slca_offset_delta"),
        ):
            vec = np.asarray(state[key], dtype=np.float64)
            if vec.shape != (3,):
                raise ValueError(
                    f"state {key} shape {vec.shape} must be (3,)"
                )
            setattr(self, attr, vec)
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
