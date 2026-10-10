"""Decimal-return metrics; undefined quantities are JSON null, never fabricated."""

import numpy as np
from scipy import stats
from sklearn import metrics


def safe_ratio(a, b):
    return float(a / b) if np.isfinite(b) and abs(b) > 1e-15 else None


def correlation(a, b, rank=False):
    if len(a) < 3 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return None
    return float(stats.spearmanr(a, b).statistic if rank else np.corrcoef(a, b)[0, 1])


def evaluate(y, prediction, train_y) -> dict:
    y, train_y = np.asarray(y), np.asarray(train_y)
    if not len(y) or not np.isfinite(y).all() or not len(train_y):
        raise ValueError("Finite nonempty targets required")
    result = {
        "n": len(y),
        "positive_fraction": float(np.mean(y > 0)),
        "training_positive_fraction": float(np.mean(train_y > 0)),
    }
    forecast, probability = prediction.get("return"), prediction.get("probability")
    direction = prediction.get("direction")
    if forecast is not None:
        forecast = np.asarray(forecast)
        if forecast.shape != y.shape or not np.isfinite(forecast).all():
            raise ValueError("Forecast alignment/nonfinite error")
        error = forecast - y
        mse = float(np.mean(error**2))
        denominator = np.mean((y - np.mean(train_y)) ** 2)
        ratio = safe_ratio(mse, denominator)
        zero_ratio = safe_ratio(mse, np.mean(y**2))
        result["regression"] = {
            "mae": float(np.mean(abs(error))),
            "mse": mse,
            "rmse": float(np.sqrt(mse)),
            "median_absolute_error": float(np.median(abs(error))),
            "r2": float(metrics.r2_score(y, forecast)) if len(y) > 1 and np.std(y) > 1e-12 else None,
            "oos_r2_train_mean": None if ratio is None else 1 - ratio,
            "oos_r2_zero": None if zero_ratio is None else 1 - zero_ratio,
            "mase": safe_ratio(np.mean(abs(error)), np.mean(abs(np.diff(train_y)))) if len(train_y) > 1 else None,
            "bias": float(np.mean(error)),
            "residual_mean": float(np.mean(-error)),
            "residual_std": float(np.std(error, ddof=1)) if len(y) > 1 else None,
        }
        result["relationship"] = {"pearson": correlation(y, forecast), "spearman_ic": correlation(y, forecast, True)}
        if direction is None:
            direction = forecast > 0
    if probability is not None:
        p = np.asarray(probability)
        if p.shape != y.shape or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
            raise ValueError("Aligned probabilities in [0,1] required")
        bins = np.minimum((p * 10).astype(int), 9)
        calibration = [
            {
                "bin": i,
                "count": int(np.sum(bins == i)),
                "mean_probability": float(np.mean(p[bins == i])),
                "positive_fraction": float(np.mean(y[bins == i] > 0)),
            }
            for i in range(10)
            if np.any(bins == i)
        ]
        both = len(np.unique(y > 0)) == 2
        result["probability"] = {
            "brier": float(metrics.brier_score_loss(y > 0, p)),
            "log_loss": float(metrics.log_loss(y > 0, np.clip(p, 1e-7, 1 - 1e-7), labels=[False, True])),
            "roc_auc": float(metrics.roc_auc_score(y > 0, p)) if both else None,
            "pr_auc_average_precision": float(metrics.average_precision_score(y > 0, p)) if both else None,
            "ece_10_equal_width": sum(
                b["count"] * abs(b["mean_probability"] - b["positive_fraction"]) for b in calibration
            )
            / len(y),
            "calibration": calibration,
        }
        direction = p >= 0.5
    if direction is not None:
        direction = np.asarray(direction)
        if direction.shape != y.shape:
            raise ValueError("Direction alignment error")
        truth = y > 0
        tn, fp, fn, tp = metrics.confusion_matrix(truth, direction, labels=[False, True]).ravel()
        tpr, tnr = safe_ratio(tp, tp + fn), safe_ratio(tn, tn + fp)
        result["direction"] = {
            "accuracy": float(np.mean(truth == direction)),
            "balanced_accuracy": (tpr + tnr) / 2 if tpr is not None and tnr is not None else None,
            "precision": safe_ratio(tp, tp + fp),
            "recall": tpr,
            "f1": safe_ratio(2 * tp, 2 * tp + fp + fn),
            "tpr": tpr,
            "tnr": tnr,
            "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
            "train_majority_accuracy": float(np.mean(truth == (np.mean(train_y > 0) >= 0.5))),
        }
    return result


def diebold_mariano(y, first, second, horizon=1):
    """Squared-loss differential, Bartlett HAC with >= h-1 lags; asymptotic normal."""
    d = (np.asarray(y) - first) ** 2 - (np.asarray(y) - second) ** 2
    n = len(d)
    lag = min(n - 1, max(horizon - 1, int(n ** (1 / 3))))
    centered = d - np.mean(d)
    variance = np.dot(centered, centered) / n
    for k in range(1, lag + 1):
        variance += 2 * (1 - k / (lag + 1)) * np.dot(centered[k:], centered[:-k]) / n
    if n < 8 or variance <= 1e-25:
        return {"statistic": None, "p_value": None, "hac_lags": lag}
    statistic = float(np.mean(d) / np.sqrt(variance / n))
    return {
        "statistic": statistic,
        "p_value": float(2 * stats.norm.sf(abs(statistic))),
        "hac_lags": lag,
        "mean_loss_difference": float(np.mean(d)),
        "negative_favors": "first",
    }


def block_interval(values, horizon=1, repeats=500, seed=7):
    values = np.asarray(values)
    if not len(values):
        return [None, None]
    block = min(len(values), max(horizon, int(np.ceil(len(values) ** (1 / 3)))))
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(repeats):
        starts = rng.integers(0, len(values), size=int(np.ceil(len(values) / block)))
        ix = np.concatenate([(s + np.arange(block)) % len(values) for s in starts])[: len(values)]
        estimates.append(np.mean(values[ix]))
    return np.quantile(estimates, [0.025, 0.975]).tolist()
