import numpy as np
import pytest
from scipy.ndimage import label

from src.eval.auroc import f1_at_threshold, f1_max, image_auroc, pixel_auroc
from src.eval.pro import area_under_curve, aupro, pro_curve


def _random_masks(rng, n=4, h=24, w=24):
    masks = np.zeros((n, h, w), dtype=bool)
    masks[0, 2:6, 3:9] = True
    masks[0, 15:20, 15:18] = True  # 같은 영상에 영역 두 개
    masks[1, 10:14, 10:12] = True
    # masks[2], masks[3]은 정상 영상
    return masks


def _brute_force_aupro(masks, maps, fpr_limit):
    """임계값마다 정의대로 FPR · PRO를 직접 세는 기준 구현."""
    regions = []
    for i, m in enumerate(masks):
        lab, n = label(m, structure=np.ones((3, 3), dtype=int))
        regions += [(i, lab == k) for k in range(1, n + 1)]
    ok = ~masks
    fprs, pros = [0.0], [0.0]
    for t in np.unique(maps)[::-1]:
        pred = maps >= t
        fprs.append((pred & ok).sum() / ok.sum())
        pros.append(np.mean([(pred[i] & r).sum() / r.sum() for i, r in regions]))
    fprs.append(1.0)
    pros.append(1.0)
    return area_under_curve(np.array(fprs), np.array(pros), fpr_limit) / fpr_limit


def test_image_auroc_perfect_and_inverted():
    labels = np.array([0, 0, 1, 1])
    assert image_auroc(labels, np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    assert image_auroc(labels, np.array([0.9, 0.8, 0.2, 0.1])) == 0.0


def test_pixel_auroc_perfect_map():
    rng = np.random.default_rng(0)
    masks = _random_masks(rng)
    assert pixel_auroc(masks, masks.astype(np.float32)) == 1.0


def test_aupro_is_one_for_perfect_map():
    masks = _random_masks(np.random.default_rng(0))
    assert aupro(masks, masks.astype(np.float32)) == pytest.approx(1.0)


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("quantize", [False, True])
def test_aupro_matches_brute_force(seed, quantize):
    rng = np.random.default_rng(seed)
    masks = _random_masks(rng)
    maps = rng.random(masks.shape).astype(np.float32) + 0.5 * masks
    if quantize:  # 동점 처리 확인
        maps = np.round(maps * 8) / 8
    for limit in (0.3, 1.0):
        assert aupro(masks, maps, limit) == pytest.approx(
            _brute_force_aupro(masks, maps, limit), abs=1e-6
        )


def test_pro_weights_regions_equally_regardless_of_size():
    # 큰 영역만 맞히고 작은 영역은 놓치면 PRO는 0.5 근처 (픽셀 비율이 아니라 영역 평균)
    masks = np.zeros((1, 40, 40), dtype=bool)
    masks[0, 0:20, 0:20] = True  # 400 픽셀
    masks[0, 30:32, 30:32] = True  # 4 픽셀
    maps = np.zeros((1, 40, 40), dtype=np.float32)
    maps[0, 0:20, 0:20] = 1.0
    fpr, pro = pro_curve(masks, maps)
    assert pro[1] == pytest.approx(0.5)
    assert fpr[1] == pytest.approx(0.0)


def test_f1_helpers():
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.6, 0.7, 0.9])
    assert f1_at_threshold(labels, scores, 0.65) == 1.0
    assert f1_at_threshold(labels, scores, 0.5) == pytest.approx(0.8)
    assert f1_max(labels, scores) == 1.0
