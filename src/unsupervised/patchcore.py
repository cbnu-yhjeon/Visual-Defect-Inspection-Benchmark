"""C2. PatchCore (Roth et al., CVPR 2022).

1) ImageNet 사전학습 WideResNet-50의 layer2 · layer3 특징을 3x3 이웃 평균으로 지역 집계 후
   layer2 해상도(256 입력 기준 32x32)로 맞춰 이어붙임 → 패치마다 1536차원
2) 정상 패치 전체에서 greedy k-center coreset으로 메모리 뱅크를 줄임 (기본 1%)
3) 시험 패치 점수 = 뱅크 최근접 거리. 이미지 점수는 최대 패치 점수에 논문 식 (7)의
   재가중을 적용. 맵은 패치 점수를 업샘플 + 가우시안(sigma=4).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torch import nn

from src.common import (
    Detector,
    gaussian_smooth,
    get_device,
    imagenet_normalize,
    knn_distance,
    to_tensor,
    upsample,
)


@dataclass
class PatchCoreConfig:
    backbone: str = "wide_resnet50_2"
    pretrained: bool = True
    coreset_ratio: float = 0.01  # 논문 표 기준 1%에서도 성능 손실이 거의 없음
    projection_dim: int = 128  # coreset 거리 계산용 무작위 투영 차원
    num_neighbors: int = 9  # 이미지 점수 재가중에 쓰는 이웃 수 (논문 b)
    sigma: float = 4.0
    batch_size: int = 32
    seed: int = 0


class FeatureExtractor(nn.Module):
    """torchvision ResNet 계열에서 layer2 · layer3 출력만 꺼낸다."""

    def __init__(self, backbone: str, pretrained: bool):
        super().__init__()
        weights = "DEFAULT" if pretrained else None
        if backbone == "wide_resnet50_2" and pretrained:
            weights = torchvision.models.Wide_ResNet50_2_Weights.IMAGENET1K_V1  # 논문과 같은 가중치
        net = getattr(torchvision.models, backbone)(weights=weights)
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool, net.layer1)
        self.layer2, self.layer3 = net.layer2, net.layer3

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        f2 = self.layer2(self.stem(x))
        return f2, self.layer3(f2)


def greedy_coreset(features: torch.Tensor, n_select: int, projection_dim: int, seed: int) -> torch.Tensor:
    """k-center greedy: 이미 고른 점들과 가장 먼 점을 하나씩 추가. 거리는 무작위 투영 공간에서 계산.

    반복마다 .item()으로 GPU를 동기화하지 않도록 선택 인덱스를 텐서로 유지한다.
    """
    n = features.shape[0]
    if n_select >= n:
        return torch.arange(n, device=features.device)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    if projection_dim and features.shape[1] > projection_dim:
        proj = torch.randn(features.shape[1], projection_dim, generator=gen) / projection_dim**0.5
        z = features @ proj.to(features.device, features.dtype)
    else:
        z = features
    selected = torch.empty(n_select, dtype=torch.long, device=features.device)
    idx = torch.randint(0, n, (1,), generator=gen).to(features.device)
    min_dist = torch.full((n,), float("inf"), device=features.device)
    for i in range(n_select):
        selected[i] = idx[0]
        d = torch.linalg.vector_norm(z - z[idx], dim=1)
        min_dist = torch.minimum(min_dist, d)
        idx = min_dist.argmax().view(1)
    return selected


class PatchCore(Detector):
    def __init__(self, cfg: PatchCoreConfig | None = None, device: str = "auto"):
        self.cfg = cfg or PatchCoreConfig()
        self.device = get_device(device)
        self.extractor = FeatureExtractor(self.cfg.backbone, self.cfg.pretrained).to(self.device).eval()
        self.memory_bank: torch.Tensor | None = None
        self.feature_hw: tuple[int, int] = (0, 0)

    @torch.no_grad()
    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """(N,3,H,W) [0,1] -> (N, h*w, D) 패치 특징."""
        f2, f3 = self.extractor(imagenet_normalize(x))
        f2 = F.avg_pool2d(f2, 3, 1, 1)
        f3 = F.avg_pool2d(f3, 3, 1, 1)
        f3 = F.interpolate(f3, size=f2.shape[-2:], mode="bilinear", align_corners=False)
        f = torch.cat([f2, f3], dim=1)
        self.feature_hw = tuple(f.shape[-2:])
        return f.flatten(2).transpose(1, 2).contiguous()

    def _embed_images(self, images: np.ndarray) -> torch.Tensor:
        out = []
        for i in range(0, len(images), self.cfg.batch_size):
            out.append(self.embed(to_tensor(images[i : i + self.cfg.batch_size], self.device)))
        return torch.cat(out)

    @torch.no_grad()
    def fit(self, images: np.ndarray) -> None:
        feats = self._embed_images(images)
        feats = feats.reshape(-1, feats.shape[-1])
        n_select = max(1, int(len(feats) * self.cfg.coreset_ratio))
        keep = greedy_coreset(feats, n_select, self.cfg.projection_dim, self.cfg.seed)
        self.memory_bank = feats[keep].contiguous()
        del feats
        torch.cuda.empty_cache() if self.device.type == "cuda" else None

    @torch.no_grad()
    def _score_batch(self, feats: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """feats (B,P,D) -> (이미지 점수 (B,), 패치 점수 (B,P))."""
        b, p, d = feats.shape
        dist, nn_idx = knn_distance(feats.reshape(-1, d), self.memory_bank, k=1)
        patch = dist[:, 0].view(b, p)
        nn_idx = nn_idx[:, 0].view(b, p)
        # 논문 식 (7): 최대 패치의 최근접 뱅크 원소 m*가 뱅크 안에서 외딴 점이면 점수를 키운다
        top = patch.argmax(dim=1)
        rows = torch.arange(b, device=feats.device)
        s_star = patch[rows, top]
        m_test = feats[rows, top]  # (B,D)
        m_star = self.memory_bank[nn_idx[rows, top]]  # (B,D)
        k = min(self.cfg.num_neighbors, len(self.memory_bank))
        _, support = knn_distance(m_star, self.memory_bank, k=k)  # (B,k), 첫 원소는 m* 자신
        d_support = torch.linalg.vector_norm(m_test[:, None] - self.memory_bank[support], dim=-1)
        weight = 1 - torch.softmax(d_support, dim=1)[:, 0]
        return weight * s_star, patch

    @torch.no_grad()
    def predict(self, images: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        size = images.shape[1]
        scores, maps = [], []
        for i in range(0, len(images), self.cfg.batch_size):
            feats = self.embed(to_tensor(images[i : i + self.cfg.batch_size], self.device))
            s, patch = self._score_batch(feats)
            h, w = self.feature_hw
            scores.append(s.cpu().numpy())
            maps.append(upsample(patch.view(-1, h, w), size).cpu().numpy())
        maps = gaussian_smooth(np.concatenate(maps), self.cfg.sigma)
        return np.concatenate(scores), maps
