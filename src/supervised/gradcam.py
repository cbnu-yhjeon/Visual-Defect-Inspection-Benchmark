"""Grad-CAM (Selvaraju et al., ICCV 2017) — 단일 로짓 이진 분류기용."""

from __future__ import annotations

import torch
from torch import nn

from src.common import upsample


class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self._act: torch.Tensor | None = None
        target_layer.register_forward_hook(self._save)

    def _save(self, _module, _inp, out):
        self._act = out
        out.register_hook(lambda g: setattr(self, "_grad", g))

    def __call__(self, x: torch.Tensor, size: int) -> tuple[torch.Tensor, torch.Tensor]:
        """x: 정규화된 입력 (N,3,H,W) -> (불량 확률 (N,), CAM (N,size,size), 영상별 최댓값 1로 정규화).

        eval 모드라 배치 안 영상끼리 기울기가 섞이지 않으므로 로짓 합 하나로 역전파한다.
        """
        self.model.eval()
        x = x.requires_grad_(True)
        logit = self.model(x).view(-1)
        self.model.zero_grad(set_to_none=True)
        logit.sum().backward()
        weights = self._grad.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * self._act).sum(dim=1)).detach()
        cam = upsample(cam, size)
        cam = cam / cam.flatten(1).amax(dim=1).clamp_min(1e-8).view(-1, 1, 1)
        return torch.sigmoid(logit).detach(), cam
