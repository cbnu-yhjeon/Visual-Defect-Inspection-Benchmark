"""PRO (Per-Region Overlap) 곡선과 FPR 0.3까지의 정규화 면적 (AUPRO).

MVTec AD 공식 평가 코드(Bergmann et al., IJCV 2021)와 같은 정의다.
임계값을 하나씩 돌리는 대신, 픽셀을 점수 내림차순으로 정렬하고 각 픽셀이
FPR과 PRO에 기여하는 양을 누적합으로 더한다.
  - 정상 픽셀 1개는 FPR에 1/(정상 픽셀 수)만큼 기여
  - 결함 영역 c의 픽셀 1개는 PRO에 1/(|c| · 전체 영역 수)만큼 기여
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import label

_EIGHT_CONNECTED = np.ones((3, 3), dtype=int)


def pro_curve(masks: np.ndarray, maps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """masks: (N,H,W) bool, maps: (N,H,W) float -> (fpr, pro). 둘 다 0에서 시작해 1로 끝난다."""
    masks = np.asarray(masks).astype(bool)
    maps = np.asarray(maps, dtype=np.float32)
    fp_changes = np.zeros(masks.shape, dtype=np.float64)
    pro_changes = np.zeros(masks.shape, dtype=np.float64)
    num_ok_pixels = 0
    num_regions = 0
    for i, mask in enumerate(masks):
        labeled, n = label(mask, structure=_EIGHT_CONNECTED)
        ok = labeled == 0
        num_ok_pixels += int(ok.sum())
        num_regions += n
        fp_changes[i][ok] = 1.0
        if n:
            sizes = np.bincount(labeled.ravel(), minlength=n + 1).astype(np.float64)
            region = labeled[~ok]
            pro_changes[i][~ok] = 1.0 / sizes[region]
    if num_regions == 0:
        raise ValueError("PRO는 결함 영역이 하나 이상 있어야 정의된다")

    scores = maps.ravel()
    order = np.argsort(scores, kind="stable")[::-1]
    scores = scores[order]
    fpr = np.cumsum(fp_changes.ravel()[order]) / num_ok_pixels
    pro = np.cumsum(pro_changes.ravel()[order]) / num_regions
    # 같은 점수의 픽셀은 한 임계값에서 동시에 켜지므로 마지막 위치만 남긴다
    keep = np.append(np.diff(scores) != 0, True)
    fpr = np.clip(fpr[keep], None, 1.0)
    pro = np.clip(pro[keep], None, 1.0)
    return np.concatenate(([0.0], fpr, [1.0])), np.concatenate(([0.0], pro, [1.0]))


def area_under_curve(x: np.ndarray, y: np.ndarray, x_max: float) -> float:
    """x에 대해 정렬된 곡선의 [0, x_max] 구간 사다리꼴 면적. x_max 지점은 선형 보간."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    inside = x <= x_max
    xs, ys = x[inside], y[inside]
    if xs[-1] < x_max:
        y_at = np.interp(x_max, x, y)
        xs = np.append(xs, x_max)
        ys = np.append(ys, y_at)
    return float(np.trapezoid(ys, xs))


def aupro(masks: np.ndarray, maps: np.ndarray, fpr_limit: float = 0.3) -> float:
    """FPR 0~fpr_limit 구간 PRO 곡선 면적을 fpr_limit로 나눈 값 (0~1)."""
    fpr, pro = pro_curve(masks, maps)
    return area_under_curve(fpr, pro, fpr_limit) / fpr_limit
