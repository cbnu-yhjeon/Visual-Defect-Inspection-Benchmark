"""방법 공통 인터페이스와 보조 함수."""

from __future__ import annotations

from abc import ABC, abstractmethod

import cv2
import numpy as np
import torch
import torch.nn.functional as F

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class Detector(ABC):
    """정상 영상만으로 학습하는 방법(A1 · A2 · C1 · C2)의 공통 인터페이스."""

    @abstractmethod
    def fit(self, images: np.ndarray) -> None:
        """images: (N,H,W,3) uint8 RGB, 전부 정상."""

    @abstractmethod
    def predict(self, images: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """-> (이미지 점수 (N,), 이상 맵 (N,H,W) float32). 둘 다 클수록 불량."""


def get_device(name: str = "auto") -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def gaussian_smooth(maps: np.ndarray, sigma: float) -> np.ndarray:
    """(N,H,W) 맵마다 가우시안 블러. anomalib · PatchCore와 같이 sigma=4가 기본."""
    if sigma <= 0:
        return maps.astype(np.float32)
    return np.stack([cv2.GaussianBlur(m.astype(np.float32), (0, 0), sigma) for m in maps])


def to_tensor(images: np.ndarray, device: torch.device) -> torch.Tensor:
    """(N,H,W,3) uint8 -> (N,3,H,W) float [0,1]."""
    return torch.from_numpy(images).to(device).permute(0, 3, 1, 2).float().div_(255.0)


def imagenet_normalize(x: torch.Tensor) -> torch.Tensor:
    mean = x.new_tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = x.new_tensor(IMAGENET_STD).view(1, 3, 1, 1)
    return (x - mean) / std


def knn_distance(
    query: torch.Tensor, bank: torch.Tensor, k: int = 1, chunk: int = 1024
) -> tuple[torch.Tensor, torch.Tensor]:
    """query (Q,D)마다 bank (B,D)에서 가장 가까운 k개의 거리 · 인덱스 (Q,k). 메모리 절약용 청크."""
    dists, idxs = [], []
    for start in range(0, query.shape[0], chunk):
        d = torch.cdist(query[start : start + chunk], bank)
        v, i = d.topk(k, dim=1, largest=False)
        dists.append(v)
        idxs.append(i)
    return torch.cat(dists), torch.cat(idxs)


def upsample(maps: torch.Tensor, size: int) -> torch.Tensor:
    """(N,h,w) -> (N,size,size) 쌍선형 보간."""
    return F.interpolate(maps[:, None], size=(size, size), mode="bilinear", align_corners=False)[:, 0]


def augment(x: torch.Tensor, texture: bool, generator: torch.Generator | None = None) -> torch.Tensor:
    """(N,3,H,W) [0,1] 배치 증강.

    텍스처: 뒤집기 · 90도 회전 (방향이 의미 없음).
    물체: 밝기 · 대비만. 뒤집기를 쓰지 않는 이유 — metal_nut의 'flip' 결함은 좌우 반전된
    너트이므로, 정상품을 뒤집으면 그 자체가 결함 영상이 된다.
    """
    n = x.shape[0]
    dev = x.device

    def rand(*shape):
        return torch.rand(*shape, generator=generator, device="cpu").to(dev)

    if texture:
        flip_h = rand(n) < 0.5
        flip_v = rand(n) < 0.5
        x = torch.where(flip_h.view(-1, 1, 1, 1), x.flip(3), x)
        x = torch.where(flip_v.view(-1, 1, 1, 1), x.flip(2), x)
        k = int(torch.randint(0, 4, (1,), generator=generator))
        x = torch.rot90(x, k, dims=(2, 3))
    brightness = 1.0 + (rand(n, 1, 1, 1) - 0.5) * 0.2
    contrast = 1.0 + (rand(n, 1, 1, 1) - 0.5) * 0.2
    mean = x.mean(dim=(1, 2, 3), keepdim=True)
    return ((x - mean) * contrast + mean * brightness).clamp_(0, 1)
