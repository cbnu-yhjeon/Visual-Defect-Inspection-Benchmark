"""합성 데이터로 각 방법이 '정상과 다른 부분'을 잡는지 확인하는 스모크 테스트 (CPU, 수 초)."""

import numpy as np
import pytest
import torch

from src.classical.diff_otsu import DiffOtsu, DiffOtsuConfig
from src.classical.texture_lbp_gabor import TextureConfig, TextureLbpGabor
from src.supervised.gradcam import GradCAM
from src.unsupervised.conv_ae import ConvAEConfig, ConvAutoencoder, ssim_map
from src.unsupervised.patchcore import PatchCore, PatchCoreConfig, greedy_coreset

SIZE = 64


def _normals(n, seed=0):
    rng = np.random.default_rng(seed)
    _yy, xx = np.mgrid[:SIZE, :SIZE]
    stripes = (np.sin(xx / 2.0) * 40 + 128).astype(np.float32)
    imgs = stripes[None, ..., None] + rng.normal(0, 3, (n, SIZE, SIZE, 1))
    return np.clip(np.repeat(imgs, 3, axis=-1), 0, 255).astype(np.uint8)


def _with_defect(imgs):
    out = imgs.copy()
    out[:, 20:32, 36:48] = 255
    return out


def _check(det, n_train=12):
    det.fit(_normals(n_train))
    good, bad = _normals(4, seed=1), _with_defect(_normals(4, seed=2))
    s_good, _m_good = det.predict(good)
    s_bad, m_bad = det.predict(bad)
    assert m_bad.shape == (4, SIZE, SIZE) and s_bad.shape == (4,)
    assert s_bad.min() > s_good.max()
    # 맵의 최댓값이 결함 사각형 근처에 있어야 한다
    y, x = np.unravel_index(m_bad[0].argmax(), m_bad[0].shape)
    assert 12 <= y <= 40 and 28 <= x <= 56


def test_diff_otsu_detects_and_segments_blob():
    det = DiffOtsu(DiffOtsuConfig(align=True, sigma=1.0, min_area=5))
    _check(det)
    _, m_good = det.predict(_normals(4, seed=3))
    det.calibrate_segmentation(m_good)
    _, m_bad = det.predict(_with_defect(_normals(1, seed=4)))
    mask, contours = det.segment(m_bad[0])
    assert len(contours) >= 1 and mask[26, 42]


def test_texture_lbp_gabor_detects_blob():
    _check(TextureLbpGabor(TextureConfig(window=8, stride=2, bank_size=5000, sigma=1.0), device="cpu"))


def test_patchcore_detects_blob_without_pretrained_weights():
    cfg = PatchCoreConfig(backbone="resnet18", pretrained=False, coreset_ratio=0.5, sigma=1.0)
    _check(PatchCore(cfg, device="cpu"), n_train=6)


def test_greedy_coreset_covers_every_cluster():
    torch.manual_seed(0)
    centers = torch.tensor([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0], [10.0, 10.0]])
    pts = torch.cat([c + 0.1 * torch.randn(50, 2) for c in centers])
    sel = greedy_coreset(pts, 4, projection_dim=0, seed=0)
    assert len(set(sel.tolist())) == 4
    assert sorted((sel // 50).tolist()) == [0, 1, 2, 3]


def test_conv_ae_shapes_and_ssim_identity():
    x = torch.rand(2, 3, 32, 32)
    assert torch.allclose(ssim_map(x, x).mean(), torch.tensor(1.0), atol=1e-4)
    ae = ConvAutoencoder(ConvAEConfig(steps=2, batch_size=2), device="cpu")
    ae.fit(_normals(4)[:, :, :, :])
    s, m = ae.predict(_normals(2))
    assert s.shape == (2,) and m.shape == (2, SIZE, SIZE) and np.isfinite(m).all()


def test_gradcam_map_is_normalized_and_nonnegative():
    net = torch.nn.Sequential(
        torch.nn.Conv2d(3, 4, 3, padding=1), torch.nn.ReLU(),
        torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(), torch.nn.Linear(4, 1),
    )
    cam = GradCAM(net, net[1])
    p, m = cam(torch.rand(3, 3, 16, 16), size=32)
    assert p.shape == (3,) and m.shape == (3, 32, 32)
    assert (m >= 0).all() and (m.flatten(1).amax(1) <= 1 + 1e-6).all()


@pytest.mark.parametrize("texture", [True, False])
def test_augment_keeps_range_and_shape(texture):
    from src.common import augment

    x = torch.rand(4, 3, 16, 16)
    y = augment(x, texture, torch.Generator().manual_seed(0))
    assert y.shape == x.shape and y.min() >= 0 and y.max() <= 1
