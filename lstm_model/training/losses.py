"""
losses.py

Contains loss functions and helpers for quantile regression with LSTM models.

Author: Nakul Mody
Date: 2025-10-24
"""

import tensorflow as tf

# ============================================================
# Pinball (Quantile) Loss
# ============================================================

def pinball_loss(y_true, y_pred, quantile):
    """
    Basic pinball loss for one quantile.

    L_q(y, ŷ) = max(q * (y - ŷ), (q - 1) * (y - ŷ))
    This is a shorter way to represent the piecewise function (vector form)

    Args:
        y_true: Tensor of true target values, shape (batch, 1)
        y_pred: Tensor of predicted quantile values, shape (batch, 1)
        quantile: Float between (0,1), e.g. 0.1, 0.5, 0.9

    Returns:
        Tensor scalar: mean pinball loss
    """
    error = y_true - y_pred
    return tf.reduce_mean(tf.maximum(quantile * error, (quantile - 1) * error))

# ============================================================
# Helper: Convert raw logits → monotonic quantile predictions
# ============================================================

def logits_to_monotonic_quantiles(logits, num_quantiles, eps=1e-6, delta_transform=tf.nn.softplus):
    """
    Convert model raw outputs (base + delta logits) into non-decreasing quantile predictions.

    Example:
        model outputs [base, Δ1, Δ2, ...]
        quantiles = [base, base + Δ1, base + Δ1 + Δ2, ...]

    Args:
        logits: (batch, num_quantiles)
        num_quantiles: total number of quantiles (e.g., 3 for [0.1, 0.5, 0.9])
        eps: small positive number to prevent equal quantiles
        delta_transform: function to make deltas non-negative (e.g., tf.nn.softplus)

    Returns:
        Tensor (batch, num_quantiles) of ordered quantile predictions.
    """
    base = logits[:, :1]                      # first output is base
    deltas_raw = logits[:, 1:num_quantiles]   # rest are deltas
    if deltas_raw.shape[-1] == 0:             # only one quantile predicted
        return base
    deltas = delta_transform(deltas_raw) + eps
    cumulative = base + tf.cumsum(deltas, axis=1)
    quantiles = tf.concat([base, cumulative], axis=1)
    return quantiles[:, :num_quantiles]


# ============================================================
# Quantile Loss Class (Keras compatible)
# ============================================================

class QuantileLoss(tf.keras.losses.Loss):
    """
    Differentiable loss enforcing monotonic, non-overlapping quantile predictions.

    Supports optional crossing penalty.
    """

    def __init__(self, quantiles, monotonic=True, penalty_lambda=0.0, name="quantile_loss"):
        super().__init__(name=name)
        self.quantiles = tf.constant(quantiles, dtype=tf.float32)
        self.num_q = len(quantiles)
        self.monotonic = monotonic
        self.penalty_lambda = float(penalty_lambda)

    def call(self, y_true, y_pred_logits):
        """
        y_true: (batch,) or (batch,1)
        y_pred_logits: raw model outputs (batch, num_quantiles)
        """
        y_true = tf.reshape(y_true, [-1, 1])
        if self.monotonic:
            q_preds = logits_to_monotonic_quantiles(y_pred_logits, self.num_q)
        else:
            q_preds = y_pred_logits

        # Compute pinball loss across all quantiles
        losses = []
        for i, q in enumerate(self.quantiles):
            losses.append(pinball_loss(y_true, q_preds[:, i:i+1], q))
        main_loss = tf.add_n(losses) / self.num_q

        # Optional: penalize any quantile crossings (should be rare)
        if self.penalty_lambda > 0:
            violation = tf.nn.relu(q_preds[:, :-1] - q_preds[:, 1:])
            penalty = tf.reduce_mean(violation)
            main_loss += self.penalty_lambda * penalty

        return main_loss


# ============================================================
# Optional: test harness
# ============================================================

if __name__ == "__main__":
    # simple smoke test
    y_true = tf.constant([[10.], [12.], [14.]])
    logits = tf.constant([[9., 0.2, 0.3],
                          [11., 0.5, 0.7],
                          [13., 0.1, 0.2]])
    quantiles = [0.1, 0.5, 0.9]
    loss_fn = QuantileLoss(quantiles, monotonic=True)
    loss_val = loss_fn(y_true, logits)
    print("Sample loss:", loss_val.numpy())
