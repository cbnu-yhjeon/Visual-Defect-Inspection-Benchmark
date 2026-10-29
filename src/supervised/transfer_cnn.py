"""B1 · B2. ImageNet 전이학습 CNN 정상/불량 분류 + Grad-CAM 위치.

MVTec AD 학습 세트에는 불량이 없으므로 시험 세트 불량을 k-fold로 나눠 학습에 쓴다
(표준 프로토콜이 아님 — 결과를 따로 표시). 모든 시험 영상은 자신을 학습에 쓰지 않은
fold 모델로 한 번씩 평가된다(out-of-fold).

- 배치는 정상 · 불량을 절반씩 뽑는다 (불량 1장이어도 증강해 반복 사용).
- 이상 맵 = p(불량) x Grad-CAM. CAM만 쓰면 정상 영상에서도 어딘가 1이 되므로 확률로 눌러준다.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torch import nn

from src.common import augment, get_device, imagenet_normalize, to_tensor
from src.supervised.gradcam import GradCAM


@dataclass
class SupervisedConfig:
    backbone: str = "resnet18"
    steps: int = 300
    batch_size: int = 32
    lr: float = 3e-4
    weight_decay: float = 1e-4
    texture: bool = False
    seed: int = 0


def build_classifier(backbone: str) -> tuple[nn.Module, nn.Module]:
    """-> (모델, Grad-CAM 대상 층)."""
    if backbone == "resnet18":
        net = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        net.fc = nn.Linear(net.fc.in_features, 1)
        return net, net.layer4
    if backbone == "mobilenet_v2":
        net = torchvision.models.mobilenet_v2(weights=torchvision.models.MobileNet_V2_Weights.IMAGENET1K_V2)
        net.classifier[1] = nn.Linear(net.classifier[1].in_features, 1)
        return net, net.features[-1]
    raise ValueError(backbone)


class TransferClassifier:
    def __init__(self, cfg: SupervisedConfig | None = None, device: str = "auto"):
        self.cfg = cfg or SupervisedConfig()
        self.device = get_device(device)
        torch.manual_seed(self.cfg.seed)
        self.net, layer = build_classifier(self.cfg.backbone)
        self.net.to(self.device)
        self.cam = GradCAM(self.net, layer)
        self.train_seconds = 0.0

    def fit(self, normal: np.ndarray, defect: np.ndarray) -> None:
        cfg = self.cfg
        gen = torch.Generator().manual_seed(cfg.seed)
        xn = torch.from_numpy(normal).to(self.device)
        xd = torch.from_numpy(defect).to(self.device)
        half = cfg.batch_size // 2
        target = torch.cat([torch.zeros(half), torch.ones(half)]).to(self.device)
        opt = torch.optim.AdamW(self.net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg.steps)
        self.net.train()
        t0 = time.perf_counter()
        for _ in range(cfg.steps):
            i_n = torch.randint(0, len(xn), (half,), generator=gen).to(self.device)
            i_d = torch.randint(0, len(xd), (half,), generator=gen).to(self.device)
            x = torch.cat([xn[i_n], xd[i_d]]).permute(0, 3, 1, 2).float().div_(255.0)
            x = imagenet_normalize(augment(x, cfg.texture, gen))
            with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
                logit = self.net(x).view(-1)
            loss = F.binary_cross_entropy_with_logits(logit.float(), target)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
        self.train_seconds = time.perf_counter() - t0

    def predict(self, images: np.ndarray, batch: int = 32) -> tuple[np.ndarray, np.ndarray]:
        """-> (불량 확률 (N,), 이상 맵 (N,H,W) = 확률 x 정규화 CAM)."""
        size = images.shape[1]
        probs, maps = [], []
        for i in range(0, len(images), batch):
            x = imagenet_normalize(to_tensor(images[i : i + batch], self.device))
            p, cam = self.cam(x, size)
            probs.append(p.cpu().numpy())
            maps.append((p.view(-1, 1, 1) * cam).cpu().numpy())
        return np.concatenate(probs), np.concatenate(maps).astype(np.float32)
