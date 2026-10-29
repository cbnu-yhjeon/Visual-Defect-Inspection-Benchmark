"""이미지 · 픽셀 단위 AUROC와 F1."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import precision_recall_curve, roc_auc_score


def image_auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    """labels: (N,) 0=정상 1=불량, scores: (N,) 클수록 불량."""
    return float(roc_auc_score(np.asarray(labels).astype(int), np.asarray(scores, dtype=np.float64)))


def pixel_auroc(masks: np.ndarray, maps: np.ndarray) -> float:
    """masks: (N,H,W) bool, maps: (N,H,W) float. 모든 영상의 픽셀을 한 번에 모아 계산."""
    y = np.asarray(masks).ravel().astype(np.uint8)
    s = np.asarray(maps, dtype=np.float32).ravel()
    return float(roc_auc_score(y, s))


def f1_at_threshold(labels: np.ndarray, scores: np.ndarray, threshold: float) -> float:
    """score >= threshold 를 불량으로 판정했을 때의 F1."""
    labels = np.asarray(labels).astype(bool)
    pred = np.asarray(scores) >= threshold
    tp = np.sum(pred & labels)
    fp = np.sum(pred & ~labels)
    fn = np.sum(~pred & labels)
    denom = 2 * tp + fp + fn
    return float(2 * tp / denom) if denom else 0.0


def f1_max(labels: np.ndarray, scores: np.ndarray) -> float:
    """시험 세트로 최적 임계값을 고른 F1 (오라클, 참고용)."""
    precision, recall, _ = precision_recall_curve(np.asarray(labels).astype(int), scores)
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) > 0,
    )
    return float(f1.max())
