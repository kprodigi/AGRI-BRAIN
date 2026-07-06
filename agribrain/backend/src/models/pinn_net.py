"""Numpy-only PINN (Physics-Informed Neural Network) for spoilage residual correction.

Implements a small feedforward network that learns residual corrections
to the Arrhenius-Baranyi ODE baseline, trained against an ODE-consistency
penalty whose gradient really propagates through the network parameters
(prior revisions of this module computed the physics term for logging
only and never differentiated through it; fixed in 2026-04 after a
hostile audit).

Architecture:
    PINN prediction = ODE baseline + NN residual correction

Input:  3 features [temp_norm, rh_norm, time_norm]
Hidden: n_hidden layers of `hidden_size` neurons, tanh activation
Output: scalar delta_C in [-0.08, 0.08] (output * 0.08 after tanh)

Loss = lambda_phys * mean(ODE_residual^2) + lambda_reg * mean(delta_C^2)

The first term drives the network to reduce residuals of
``dC/dt + k_eff(t,T,H) * C`` evaluated on the corrected trajectory
``C = C_ode + delta_C`` using finite-difference time derivatives. The
second term is an L2 anchor on the output (keeps the residual
correction small relative to the trapezoidal ODE baseline). Both
terms are explicitly back-propagated to layer weights and biases.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Normalisation constants
# ---------------------------------------------------------------------------
TEMP_MEAN = 8.0
TEMP_STD = 10.0
RH_MEAN = 0.85
RH_STD = 0.10
TIME_MEAN = 36.0
TIME_STD = 24.0


class SpoilagePINN:
    """Numpy-only feedforward PINN for spoilage residual correction.

    Parameters
    ----------
    hidden_size : neurons per hidden layer.
    n_hidden : number of hidden layers.
    lambda_phys : weight of the physics-informed loss term.
    lambda_reg : weight of the L2 anchor on the output (keeps corrections
        small relative to the ODE baseline).
    lr : learning rate for gradient descent.
    seed : random seed for reproducible Xavier initialization.
    """

    def __init__(
        self,
        hidden_size: int = 32,
        n_hidden: int = 2,
        lambda_phys: float = 1.0,
        lambda_reg: float = 1e-3,
        lr: float = 0.02,
        seed: int = 42,
    ) -> None:
        self.hidden_size = hidden_size
        self.n_hidden = n_hidden
        self.lambda_phys = lambda_phys
        self.lambda_reg = lambda_reg
        self.lr = lr

        rng = np.random.default_rng(seed)

        self.weights: List[np.ndarray] = []
        self.biases: List[np.ndarray] = []

        in_dim = 3
        for _ in range(n_hidden):
            scale = np.sqrt(2.0 / (in_dim + hidden_size))
            W = rng.normal(0.0, scale, (in_dim, hidden_size))
            b = np.zeros(hidden_size)
            self.weights.append(W)
            self.biases.append(b)
            in_dim = hidden_size

        # Output layer: small random init so the physics-loss gradient
        # has something non-zero to grip on the first epoch. A zero-init
        # output makes the very first phys_res equal to the residual on
        # C_ode alone — still nonzero in general because trapezoidal
        # integration has O(dt^2) error, so the gradient is well-defined.
        out_scale = 1e-2
        self.weights.append(rng.normal(0.0, out_scale, (in_dim, 1)))
        self.biases.append(np.array([0.0]))

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def _normalize(
        self,
        temp_C: np.ndarray,
        rh_frac: np.ndarray,
        dt_h: np.ndarray,
    ) -> np.ndarray:
        t_norm = (temp_C - TEMP_MEAN) / TEMP_STD
        h_norm = (rh_frac - RH_MEAN) / RH_STD
        time_norm = (dt_h - TIME_MEAN) / TIME_STD
        return np.column_stack([t_norm, h_norm, time_norm])

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Forward pass; returns delta_C in [-0.08, 0.08]."""
        h = X
        for i in range(self.n_hidden):
            h = np.tanh(h @ self.weights[i] + self.biases[i])
        out = np.tanh(h @ self.weights[-1] + self.biases[-1])
        return 0.08 * out.ravel()

    def _forward_with_cache(
        self, X: np.ndarray
    ) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray]:
        """Forward pass returning intermediate activations and tanh output.

        Returns
        -------
        activations : list of post-activation tensors (input + per-hidden + tanh-output).
        out : tanh of pre-output (shape (N, 1)). Used for backprop through
            the 0.08 scale and the tanh derivative.
        delta_C : flattened delta_C of shape (N,).
        """
        activations: List[np.ndarray] = [X]
        h = X
        for i in range(self.n_hidden):
            z = h @ self.weights[i] + self.biases[i]
            h = np.tanh(z)
            activations.append(h)

        z_out = h @ self.weights[-1] + self.biases[-1]
        out = np.tanh(z_out)
        delta_C = 0.08 * out.ravel()
        return activations, out, delta_C

    # ------------------------------------------------------------------
    # Physics residual + per-step coefficients (used by backprop)
    # ------------------------------------------------------------------

    def _physics_residual_with_coeffs(
        self,
        temp_C: np.ndarray,
        rh_frac: np.ndarray,
        dt_h: np.ndarray,
        C_pred: np.ndarray,
        k_ref: float,
        Ea_R: float,
        T_ref_K: float,
        beta: float,
        lag_lambda: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(phys_res, q, r)`` arrays, all of length n-1.

        For each k in [0, n-2] (corresponding to original index i = k+1):

            phys_res[k] = q[k] * C_pred[i] + r[k] * C_pred[i-1]

        with ``q[k] = 1/dt_local + k_eff(i)`` and ``r[k] = -1/dt_local``
        when the local timestep is positive, and ``q[k] = k_eff(i)``,
        ``r[k] = 0`` when it is not (matching the legacy semantics that
        a non-positive timestep contributes only the local kinetics
        term to the residual).
        """
        n = len(C_pred)
        if n < 2:
            return np.zeros(0), np.zeros(0), np.zeros(0)

        # Pre-compute per-step coefficients (length n-1, indexed by k = i-1)
        T_K = temp_C + 273.15
        k_arr = k_ref * np.exp(Ea_R * (1.0 / T_ref_K - 1.0 / T_K))
        k_arr = k_arr * (1.0 + beta * rh_frac)

        if lag_lambda > 0.0:
            denom = dt_h + lag_lambda
            alpha = np.where(denom > 0.0, dt_h / denom, 1.0)
        else:
            alpha = np.ones_like(dt_h)

        k_eff = k_arr * alpha  # length n

        q = np.zeros(n - 1)
        r = np.zeros(n - 1)
        phys_res = np.zeros(n - 1)
        for k in range(n - 1):
            i = k + 1
            delta_t = float(dt_h[i] - dt_h[i - 1])
            if delta_t > 0.0:
                inv_dt = 1.0 / delta_t
                q[k] = inv_dt + float(k_eff[i])
                r[k] = -inv_dt
            else:
                q[k] = float(k_eff[i])
                r[k] = 0.0
            phys_res[k] = q[k] * C_pred[i] + r[k] * C_pred[i - 1]

        return phys_res, q, r

    # ------------------------------------------------------------------
    # Training (with real physics-loss backprop)
    # ------------------------------------------------------------------

    def fit(
        self,
        temp_C: np.ndarray,
        rh_frac: np.ndarray,
        dt_h: np.ndarray,
        C_target: np.ndarray,
        k_ref: float,
        Ea_R: float,
        T_ref_K: float,
        beta: float,
        lag_lambda: float,
        epochs: int = 200,
    ) -> Dict[str, List[float]]:
        """Train the PINN; gradients flow through both physics and reg terms.

        Parameters
        ----------
        temp_C : temperature trajectory (Celsius).
        rh_frac : relative humidity trajectory [0, 1].
        dt_h : time trajectory (hours from start).
        C_target : ODE baseline solution; used as the L2-anchor target so
            that ``delta_C`` stays small relative to the trapezoidal
            integrator. The residual that drives the physics gradient is
            evaluated on the corrected trajectory ``C_pred = C_target +
            delta_C``.
        k_ref, Ea_R, T_ref_K, beta, lag_lambda : ODE parameters.
        epochs : number of training iterations.

        Returns
        -------
        Training history dict with keys ``"loss"``, ``"phys_loss"``,
        ``"reg_loss"``.
        """
        X = self._normalize(temp_C, rh_frac, dt_h)
        n = len(X)
        history: Dict[str, List[float]] = {"loss": [], "phys_loss": [], "reg_loss": []}

        for _epoch in range(epochs):
            # --- Forward
            activations, out, delta_C = self._forward_with_cache(X)
            C_pred = C_target + delta_C

            # --- Physics residual + per-step coefficients
            phys_res, q, r = self._physics_residual_with_coeffs(
                temp_C, rh_frac, dt_h, C_pred,
                k_ref, Ea_R, T_ref_K, beta, lag_lambda,
            )
            m = max(len(phys_res), 1)

            phys_loss = float(np.mean(phys_res ** 2)) if len(phys_res) else 0.0
            reg_loss = float(np.mean(delta_C ** 2)) if n > 0 else 0.0
            total_loss = self.lambda_phys * phys_loss + self.lambda_reg * reg_loss
            history["loss"].append(total_loss)
            history["phys_loss"].append(phys_loss)
            history["reg_loss"].append(reg_loss)

            # --- Gradient of phys_loss w.r.t. C_pred[j] (length n)
            #
            # phys_loss = (1/m) * sum_k phys_res[k]^2
            # phys_res[k] = q[k] * C_pred[k+1] + r[k] * C_pred[k]
            #
            # Contribution at j from k = j-1 (when j >= 1): q[j-1] * phys_res[j-1]
            # Contribution at j from k = j   (when j+1 < n): r[j]   * phys_res[j]
            d_C_pred_phys = np.zeros(n)
            if len(phys_res):
                # k = j-1 term
                d_C_pred_phys[1:] += q * phys_res
                # k = j term
                d_C_pred_phys[:-1] += r * phys_res
                d_C_pred_phys *= (2.0 / m)

            # --- Gradient of reg_loss w.r.t. delta_C[j]
            d_delta_reg = (2.0 / max(n, 1)) * delta_C if n > 0 else np.zeros(n)

            # --- Total gradient w.r.t. delta_C[j]
            #     d(C_pred)/d(delta_C) = 1, so phys gradient passes straight through
            d_delta = self.lambda_phys * d_C_pred_phys + self.lambda_reg * d_delta_reg

            # --- Backprop d_delta through 0.08 * tanh(z_out)
            # delta_C = 0.08 * out, out = tanh(z_out)
            # d(delta_C)/d(z_out) = 0.08 * (1 - out^2)
            d_z_out = d_delta.reshape(-1, 1) * 0.08 * (1.0 - out ** 2)

            # Output layer
            h_last = activations[-1]
            dW_out = h_last.T @ d_z_out / n
            db_out = d_z_out.mean(axis=0)
            d_h = d_z_out @ self.weights[-1].T

            # Hidden layers (reverse order)
            grad_W: List[np.ndarray] = []
            grad_b: List[np.ndarray] = []
            for i in range(self.n_hidden - 1, -1, -1):
                h_i = activations[i + 1]
                d_pre = d_h * (1.0 - h_i ** 2)

                h_prev = activations[i]
                grad_W.append(h_prev.T @ d_pre / n)
                grad_b.append(d_pre.mean(axis=0))

                if i > 0:
                    d_h = d_pre @ self.weights[i].T

            # Apply updates after backprop traversal
            grad_W.reverse()
            grad_b.reverse()
            for i in range(self.n_hidden):
                self.weights[i] -= self.lr * grad_W[i]
                self.biases[i] -= self.lr * grad_b[i]
            self.weights[-1] -= self.lr * dW_out
            self.biases[-1] -= self.lr * db_out

        return history

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(
        self,
        temp_C: np.ndarray,
        rh_frac: np.ndarray,
        dt_h: np.ndarray,
    ) -> np.ndarray:
        """Return residual corrections for the given trajectory."""
        X = self._normalize(
            np.asarray(temp_C, dtype=np.float64),
            np.asarray(rh_frac, dtype=np.float64),
            np.asarray(dt_h, dtype=np.float64),
        )
        return self.forward(X)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def state_dict(self) -> Dict[str, list]:
        return {
            "weights": [w.tolist() for w in self.weights],
            "biases": [b.tolist() for b in self.biases],
        }

    def load_state_dict(self, d: Dict[str, list]) -> None:
        self.weights = [np.array(w) for w in d["weights"]]
        self.biases = [np.array(b) for b in d["biases"]]
