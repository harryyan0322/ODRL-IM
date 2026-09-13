"""Statistical summaries and paired significance tests for influence spread."""

import math

import numpy as np
from scipy import stats


def mean_confidence_interval(values, confidence=0.95):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        raise ValueError("At least one value is required")
    mean = float(values.mean())
    std = float(values.std(ddof=1)) if values.size > 1 else 0.0
    if values.size > 1:
        critical = stats.t.ppf((1.0 + confidence) / 2.0, values.size - 1)
        half_width = float(critical * std / math.sqrt(values.size))
    else:
        half_width = 0.0
    return {
        "n": int(values.size),
        "mean": mean,
        "std": std,
        "confidence_level": confidence,
        "ci_low": mean - half_width,
        "ci_high": mean + half_width,
    }


def paired_significance(values_a, values_b):
    values_a = np.asarray(values_a, dtype=np.float64)
    values_b = np.asarray(values_b, dtype=np.float64)
    if values_a.shape != values_b.shape or values_a.size < 2:
        raise ValueError("Paired tests require matching arrays with at least two values")
    differences = values_a - values_b
    paired_t = stats.ttest_rel(values_a, values_b)
    try:
        wilcoxon = stats.wilcoxon(values_a, values_b, zero_method="wilcox")
        wilcoxon_statistic = float(wilcoxon.statistic)
        wilcoxon_pvalue = float(wilcoxon.pvalue)
    except ValueError:
        wilcoxon_statistic = 0.0
        wilcoxon_pvalue = 1.0
    difference_std = float(differences.std(ddof=1))
    effect_size = (
        float(differences.mean() / difference_std)
        if difference_std > 0 else (math.inf if differences.mean() > 0 else 0.0)
    )
    baseline_mean = float(values_b.mean())
    improvement = (
        float(100.0 * differences.mean() / baseline_mean)
        if baseline_mean != 0 else math.nan
    )
    return {
        "n_pairs": int(values_a.size),
        "mean_a": float(values_a.mean()),
        "mean_b": baseline_mean,
        "mean_difference": float(differences.mean()),
        "relative_improvement_percent": improvement,
        "paired_t_statistic": float(paired_t.statistic),
        "paired_t_pvalue": float(paired_t.pvalue),
        "wilcoxon_statistic": wilcoxon_statistic,
        "wilcoxon_pvalue": wilcoxon_pvalue,
        "cohen_dz": effect_size,
    }


def holm_adjust(pvalues):
    pvalues = np.asarray(pvalues, dtype=np.float64)
    order = np.argsort(pvalues)
    adjusted = np.empty_like(pvalues)
    running_max = 0.0
    count = len(pvalues)
    for rank, index in enumerate(order):
        candidate = min(1.0, (count - rank) * pvalues[index])
        running_max = max(running_max, candidate)
        adjusted[index] = running_max
    return adjusted.tolist()
