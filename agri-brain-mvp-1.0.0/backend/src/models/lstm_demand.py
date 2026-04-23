"""Lightweight numpy-only LSTM demand forecaster.

Implements a single-layer LSTM with 16 hidden units trained via truncated
backpropagation through time (BPTT) with MSE loss on a rolling window
of demand observations.  No external deep learning library is required.

Architecture
------------
    Input: 1-dimensional (demand at time t)
    Hidden: 16 LSTM cells (forget, input, output gates + cell candidate)
    Output: 1-dimensional (demand at time t+1)

Training uses a simple gradient descent with clipped gradients.  The model
is retrained from scratch on each call using the recent demand history
(rolling window), ensuring adaptability without persistent state.

References
----------
    - Hochreiter, S. & Schmidhuber, J. (1997). Long Short-Term Memory.
      Neural Computation, 9(8), 1735-1780.
    - Sutskever, I., Vinyals, O. & Le, Q.V. (2014). Sequence to Sequence
      Learning with Neural Networks. NIPS.
"""
from __future__ import annotations

import os
from typing import Dict

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# LSTM primitives (numpy only)
# ---------------------------------------------------------------------------

def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(-x))


def _tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)


class _LSTMCell:
    """Single LSTM cell with combined weight matrices."""

    def __init__(self, input_size: int, hidden_size: int, rng: np.random.Generator):
        scale = 0.1
        # Combined weights: [f, i, o, g] each (input_size + hidden_size) -> hidden_size
        self.W = rng.normal(0, scale, (4 * hidden_size, input_size + hidden_size))
        self.b = np.zeros(4 * hidden_size)
        self.hidden_size = hidden_size

    def forward(self, x: np.ndarray, h_prev: np.ndarray, c_prev: np.ndarray):
        hs = self.hidden_size
        combined = np.concatenate([x, h_prev])
        gates = self.W @ combined + self.b

        f = _sigmoid(gates[:hs])
        i = _sigmoid(gates[hs:2*hs])
        o = _sigmoid(gates[2*hs:3*hs])
        g = _tanh(gates[3*hs:])

        c = f * c_prev + i * g
        h = o * _tanh(c)
        return h, c, (combined, f, i, o, g, c_prev, c, h)


class _OutputLayer:
    """Linear output projection: hidden_size -> 1."""

    def __init__(self, hidden_size: int, rng: np.random.Generator):
        self.W = rng.normal(0, 0.1, (1, hidden_size))
        self.b = np.zeros(1)

    def forward(self, h: np.ndarray) -> float:
        return float((self.W @ h + self.b)[0])


# ---------------------------------------------------------------------------
# LSTM Model
# ---------------------------------------------------------------------------

class LSTMDemandModel:
    """Numpy-only single-layer LSTM for demand forecasting.

    Parameters
    ----------
    hidden_size : number of LSTM hidden units.
    lr : learning rate.
    epochs : training epochs.
    seed : random seed for reproducibility.
    """

    def __init__(
        self,
        hidden_size: int = 16,
        lr: float = 0.005,
        epochs: int = 80,
        seed: int = 42,
    ):
        self.hidden_size = hidden_size
        self.lr = lr
        self.epochs = epochs
        self.rng = np.random.default_rng(seed)
        self.cell = _LSTMCell(1, hidden_size, self.rng)
        self.out = _OutputLayer(hidden_size, self.rng)

    def _forward_seq(self, xs: np.ndarray):
        """Forward pass through the sequence, returning predictions and caches."""
        T = len(xs) - 1
        h = np.zeros(self.hidden_size)
        c = np.zeros(self.hidden_size)
        preds = []
        caches = []
        hs = [h.copy()]

        for t in range(T):
            x_t = np.array([xs[t]])
            h, c, cache = self.cell.forward(x_t, h, c)
            pred = self.out.forward(h)
            preds.append(pred)
            caches.append(cache)
            hs.append(h.copy())

        return preds, caches, hs

    def fit(self, series: np.ndarray) -> None:
        """Train the LSTM on the given 1-D demand series."""
        if len(series) < 3:
            return

        # Normalize
        self._mean = series.mean()
        self._std = max(series.std(), 1e-8)
        xs = (series - self._mean) / self._std

        T = len(xs) - 1

        for epoch in range(self.epochs):
            preds, caches, hs = self._forward_seq(xs)
            targets = xs[1:]

            # MSE loss gradients (simplified BPTT with gradient clipping)
            total_loss = 0.0
            dW_cell = np.zeros_like(self.cell.W)
            db_cell = np.zeros_like(self.cell.b)
            dW_out = np.zeros_like(self.out.W)
            db_out = np.zeros_like(self.out.b)

            dh_next = np.zeros(self.hidden_size)
            dc_next = np.zeros(self.hidden_size)

            for t in reversed(range(T)):
                error = preds[t] - targets[t]
                total_loss += error ** 2

                # Output layer gradient
                dW_out += error * hs[t + 1].reshape(1, -1)
                db_out += error

                # Backprop through output
                dh = self.out.W.T.flatten() * error + dh_next

                combined, f, i, o, g, c_prev, c_t, h_t = caches[t]
                hs_ = self.hidden_size

                # Backprop through LSTM cell
                do_ = dh * _tanh(c_t)
                dc = dh * o * (1.0 - _tanh(c_t) ** 2) + dc_next

                df_ = dc * c_prev
                di_ = dc * g
                dg_ = dc * i

                # Gate derivatives
                df_raw = df_ * f * (1.0 - f)
                di_raw = di_ * i * (1.0 - i)
                do_raw = do_ * o * (1.0 - o)
                dg_raw = dg_ * (1.0 - g ** 2)

                dgates = np.concatenate([df_raw, di_raw, do_raw, dg_raw])

                # Accumulate weight gradients
                dW_cell += dgates.reshape(-1, 1) @ combined.reshape(1, -1)
                db_cell += dgates

                # Propagate to previous hidden state
                d_combined = self.cell.W.T @ dgates
                dh_next = d_combined[1:]  # skip input portion
                dc_next = dc * f

            # Clip gradients
            for grad in [dW_cell, db_cell, dW_out, db_out]:
                np.clip(grad, -1.0, 1.0, out=grad)

            # Update weights
            self.cell.W -= self.lr * dW_cell / T
            self.cell.b -= self.lr * db_cell / T
            self.out.W -= self.lr * dW_out / T
            self.out.b -= self.lr * db_out / T

    def predict(self, series: np.ndarray, horizon: int) -> np.ndarray:
        """Produce multi-step forecast from the trained model."""
        if not hasattr(self, '_mean'):
            return np.full(horizon, series[-1] if len(series) > 0 else 0.0)

        xs = (series - self._mean) / self._std

        # Run forward to get final hidden state
        h = np.zeros(self.hidden_size)
        c = np.zeros(self.hidden_size)
        for t in range(len(xs)):
            x_t = np.array([xs[t]])
            h, c, _ = self.cell.forward(x_t, h, c)

        # Auto-regressive forecasting
        forecasts = []
        for _ in range(horizon):
            pred_norm = self.out.forward(h)
            pred = pred_norm * self._std + self._mean
            forecasts.append(max(0.0, pred))
            # Feed prediction back
            x_t = np.array([pred_norm])
            h, c, _ = self.cell.forward(x_t, h, c)

        return np.array(forecasts)

    def in_sample_residual_std(self, series: np.ndarray, tail: int = 8) -> float:
        """Estimate one-step-ahead prediction uncertainty as the standard
        deviation of in-sample residuals over the most recent ``tail``
        timesteps of the training window.

        This is the classical residual-standard-deviation approach to
        prediction-interval construction for a point-forecast model
        (Hyndman & Athanasopoulos, 2018, *Forecasting: Principles and
        Practice*, 2nd ed., Ch. 8.7, eq. 8.16). Under iid residuals,
        ``sigma_hat = std(y - y_hat)`` is the maximum-likelihood estimate
        of the Gaussian one-step-ahead prediction-error standard
        deviation and yields symmetric prediction intervals
        ``y_hat +/- z * sigma_hat``.

        Returns the standard deviation in the original (unnormalised)
        units of the input series.
        """
        if not hasattr(self, "_mean") or len(series) < 3:
            return 0.0
        xs = (series - self._mean) / self._std
        preds, _, _ = self._forward_seq(xs)
        preds = np.asarray(preds, dtype=float)
        targets = xs[1:]
        if len(preds) == 0:
            return 0.0
        residuals_norm = targets - preds
        # Use only the recent tail so the uncertainty tracks current
        # regime rather than the full training window's errors.
        if len(residuals_norm) > tail:
            residuals_norm = residuals_norm[-tail:]
        residuals = residuals_norm * self._std  # de-normalise
        return float(np.std(residuals, ddof=0))


# ---------------------------------------------------------------------------
# Public API (matches forecast.py interface)
# ---------------------------------------------------------------------------

def lstm_demand_forecast(
    df: pd.DataFrame,
    horizon: int = 24,
    lookback: int = 48,
    ci_z: float = 1.96,
    series_col: str = "demand_units",
    hidden_size: int = 16,
    epochs: int = 80,
    seed: int = 42,
) -> Dict[str, object]:
    """Produce a horizon-step demand forecast using a numpy-only LSTM.

    Parameters
    ----------
    df : DataFrame with at least a *series_col* column.
    horizon : number of future steps to forecast.
    lookback : number of most-recent observations used for training.
    ci_z : z-score multiplier for the confidence interval (default 1.96 = 95%).
    series_col : column name to forecast.
    hidden_size : LSTM hidden units.
    epochs : training epochs.
    seed : random seed for reproducibility.

    Returns
    -------
    dict with keys:
        ``forecast``     - list[float] of length *horizon* (point forecast)
        ``ci_lower``     - list[float] lower bound of CI
        ``ci_upper``     - list[float] upper bound of CI
        ``std``          - float, in-sample residual standard deviation
                           (one-step-ahead prediction-uncertainty estimate,
                           following Hyndman & Athanasopoulos 2018, Ch. 8.7).
        ``series_std``   - float, historical rolling std of the training tail
                           (kept for backward compatibility with code that
                           used the previous series-std semantics).
    """
    d = df[series_col].astype(float).to_numpy()

    if len(d) == 0:
        zeros = [0.0] * horizon
        return {
            "forecast": zeros, "ci_lower": zeros, "ci_upper": zeros,
            "std": 0.0, "series_std": 0.0,
        }

    # Use the most recent observations
    tail = d[-min(lookback, len(d)):]

    if len(tail) < 3:
        # Not enough data for LSTM; return simple repeat
        val = float(tail[-1]) if len(tail) > 0 else 0.0
        forecast = [max(0.0, val)] * horizon
        return {
            "forecast": forecast, "ci_lower": forecast, "ci_upper": forecast,
            "std": 0.0, "series_std": 0.0,
        }

    # Train LSTM and predict
    model = LSTMDemandModel(hidden_size=hidden_size, epochs=epochs, seed=seed)
    model.fit(tail)
    forecast_arr = model.predict(tail, horizon)
    forecast = [round(float(v), 4) for v in forecast_arr]

    # Prediction uncertainty: residual standard deviation on the recent
    # training tail. This is the proper one-step-ahead prediction-error
    # sigma (Hyndman & Athanasopoulos 2018, eq. 8.16), not the raw
    # dispersion of the observations.
    residual_std = model.in_sample_residual_std(tail, tail=8)

    # Historical rolling std retained under ``series_std`` for any caller
    # that still wants the simple dispersion metric.
    series_std = float(np.std(tail)) if len(tail) >= 2 else 0.0

    # CI bounds use the residual std so they are proper Gaussian
    # prediction intervals, not pseudo-intervals based on series variance.
    ci_lower = [round(max(0.0, f - ci_z * residual_std), 4) for f in forecast]
    ci_upper = [round(f + ci_z * residual_std, 4) for f in forecast]

    return {
        "forecast": forecast,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "std": round(residual_std, 6),
        "series_std": round(series_std, 6),
    }
