"""A2. Gabor · LBP 텍스처 특징 + 지역 이상 점수.

조밀한 격자(stride 4)의 각 위치에서 주변 창(32x32)의 손설계 특징을 뽑는다.
  - LBP (uniform, P=8, R=1·2) 지역 히스토그램 20차원 (산업 CV Lec 12)
  - Gabor 필터 뱅크 (방향 4 x 파장 2) 에너지의 지역 평균 8차원 (Lec 4)
  - 밝기 지역 평균 · 표준편차 2차원, a · b 색 채널 지역 평균 2차원
정상 학습 영상의 특징으로 메모리 뱅크를 만들고, 시험 위치마다 가장 가까운 정상 특징까지의
거리(k-NN 평균)를 지역 이상 점수로 쓴다. 위치를 보지 않으므로 텍스처에 맞는 모델이다.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import torch
from skimage.feature import local_binary_pattern

from src.common import Detector, gaussian_smooth, get_device, knn_distance, upsample


@dataclass
class TextureConfig:
    window: int = 32
    stride: int = 4
    lbp_radii: tuple[int, ...] = (1, 2)
    gabor_wavelengths: tuple[float, ...] = (4.0, 8.0)
    gabor_orientations: int = 4
    clahe_clip: float = 2.0
    bank_size: int = 100000
    k: int = 3
    sigma: float = 4.0
    seed: int = 0


class TextureLbpGabor(Detector):
    def __init__(self, cfg: TextureConfig | None = None, device: str = "auto"):
        self.cfg = cfg or TextureConfig()
        self.device = get_device(device)
        self._clahe = cv2.createCLAHE(self.cfg.clahe_clip, (8, 8))
        self._gabor = []
        for lam in self.cfg.gabor_wavelengths:
            for i in range(self.cfg.gabor_orientations):
                theta = np.pi * i / self.cfg.gabor_orientations
                ks = int(2 * round(1.5 * lam) + 1)
                even = cv2.getGaborKernel((ks, ks), 0.56 * lam, theta, lam, 0.5, 0)
                odd = cv2.getGaborKernel((ks, ks), 0.56 * lam, theta, lam, 0.5, np.pi / 2)
                self._gabor.append((even - even.mean(), odd))
        self.feat_mean: np.ndarray | None = None
        self.feat_std: np.ndarray | None = None
        self.bank: torch.Tensor | None = None

    def features(self, img: np.ndarray) -> np.ndarray:
        """(H,W,3) uint8 -> (H/stride, W/stride, D) float32."""
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        gray = self._clahe.apply(lab[..., 0])
        g = gray.astype(np.float32) / 255.0
        w = (self.cfg.window, self.cfg.window)

        def local_mean(x: np.ndarray) -> np.ndarray:
            return cv2.boxFilter(x, cv2.CV_32F, w, borderType=cv2.BORDER_REFLECT)

        channels = []
        for r in self.cfg.lbp_radii:
            codes = local_binary_pattern(gray, 8, r, method="uniform").astype(np.int32)  # 0..9
            for c in range(10):
                channels.append(local_mean((codes == c).astype(np.float32)))
        for even, odd in self._gabor:
            e = cv2.filter2D(g, cv2.CV_32F, even)
            o = cv2.filter2D(g, cv2.CV_32F, odd)
            channels.append(local_mean(np.sqrt(e * e + o * o)))
        m = local_mean(g)
        channels.append(m)
        channels.append(np.sqrt(np.maximum(local_mean(g * g) - m * m, 0)))
        ab = lab[..., 1:].astype(np.float32) / 255.0
        channels.append(local_mean(ab[..., 0]))
        channels.append(local_mean(ab[..., 1]))

        f = np.stack(channels, axis=-1)
        # 창이 영상 안에 완전히 들어가는 중심만 쓴다. 경계에서는 반사 패딩된 가짜 패턴이
        # 섞여 정상 영상에서도 최댓값이 가장자리에 몰렸다.
        s, half = self.cfg.stride, self.cfg.window // 2
        h, w = gray.shape
        return f[half : h - half + 1 : s, half : w - half + 1 : s]

    def _feature_stack(self, images: np.ndarray) -> np.ndarray:
        return np.stack([self.features(x) for x in images])

    def fit(self, images: np.ndarray) -> None:
        feats = self._feature_stack(images)
        flat = feats.reshape(-1, feats.shape[-1])
        self.feat_mean = flat.mean(axis=0)
        self.feat_std = flat.std(axis=0) + 1e-6
        rng = np.random.default_rng(self.cfg.seed)
        n = min(self.cfg.bank_size, len(flat))
        pick = rng.choice(len(flat), n, replace=False)
        bank = (flat[pick] - self.feat_mean) / self.feat_std
        self.bank = torch.from_numpy(bank).float().to(self.device)

    @torch.no_grad()
    def predict(self, images: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        size = images.shape[1]
        half = self.cfg.window // 2
        maps = []
        for img in images:
            f = self.features(img)
            h, w, d = f.shape
            q = torch.from_numpy((f.reshape(-1, d) - self.feat_mean) / self.feat_std).float()
            knn = knn_distance(q.to(self.device), self.bank, self.cfg.k)[0].mean(dim=1)
            inner = upsample(knn.view(1, h, w), size - 2 * half)[0].cpu().numpy()
            maps.append(np.pad(inner, half, mode="edge"))
        maps = gaussian_smooth(np.stack(maps), self.cfg.sigma)
        return maps.reshape(len(maps), -1).max(axis=1), maps
