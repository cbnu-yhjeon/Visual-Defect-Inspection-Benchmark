"""C1. 합성곱 오토인코더 재구성 오차.

정상 영상만 복원하도록 학습하면, 결함 부위는 복원되지 않아 오차가 커진다는 가정.
256 -> 8x8xlatent 병목 -> 256. 손실은 MSE + (1 - SSIM) (Bergmann et al. 2019의 SSIM-AE 착안),
이상 맵은 채널 평균 제곱 오차를 가우시안 평활한 것.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from src.common import Detector, augment, gaussian_smooth, get_device, to_tensor


@dataclass
class ConvAEConfig:
    latent_channels: int = 16  # 8x8x16 = 1024차원 병목
    steps: int = 2000
    batch_size: int = 32
    lr: float = 1e-3
    ssim_weight: float = 1.0
    sigma: float = 4.0
    texture: bool = False  # 텍스처면 뒤집기 · 회전 증강
    seed: int = 0


def _down(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(cin, cout, 4, 2, 1), nn.BatchNorm2d(cout), nn.LeakyReLU(0.2, True))


def _up(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.ConvTranspose2d(cin, cout, 4, 2, 1), nn.BatchNorm2d(cout), nn.LeakyReLU(0.2, True)
    )


class ConvAENet(nn.Module):
    def __init__(self, latent_channels: int = 16):
        super().__init__()
        self.encoder = nn.Sequential(
            _down(3, 32), _down(32, 64), _down(64, 128), _down(128, 128), _down(128, 256),
            nn.Conv2d(256, latent_channels, 1),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(latent_channels, 256, 1), nn.LeakyReLU(0.2, True),
            _up(256, 128), _up(128, 128), _up(128, 64), _up(64, 32),
            nn.ConvTranspose2d(32, 3, 4, 2, 1), nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


def ssim_map(x: torch.Tensor, y: torch.Tensor, window: int = 11, sigma: float = 1.5) -> torch.Tensor:
    """(N,C,H,W) [0,1] -> 픽셀별 SSIM (N,C,H,W)."""
    coords = torch.arange(window, device=x.device, dtype=x.dtype) - window // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = (g / g.sum()).view(1, 1, 1, -1)
    c = x.shape[1]

    def blur(t):
        t = F.conv2d(t, g.expand(c, 1, 1, window), padding=(0, window // 2), groups=c)
        return F.conv2d(t, g.view(1, 1, -1, 1).expand(c, 1, window, 1), padding=(window // 2, 0), groups=c)

    mx, my = blur(x), blur(y)
    vx = blur(x * x) - mx * mx
    vy = blur(y * y) - my * my
    cov = blur(x * y) - mx * my
    c1, c2 = 0.01**2, 0.03**2
    return ((2 * mx * my + c1) * (2 * cov + c2)) / ((mx * mx + my * my + c1) * (vx + vy + c2))


class ConvAutoencoder(Detector):
    def __init__(self, cfg: ConvAEConfig | None = None, device: str = "auto"):
        self.cfg = cfg or ConvAEConfig()
        self.device = get_device(device)
        torch.manual_seed(self.cfg.seed)
        self.net = ConvAENet(self.cfg.latent_channels).to(self.device)
        self.train_seconds = 0.0

    def fit(self, images: np.ndarray) -> None:
        cfg = self.cfg
        x_all = torch.from_numpy(images).to(self.device)  # uint8로 GPU에 상주
        gen = torch.Generator().manual_seed(cfg.seed)
        opt = torch.optim.AdamW(self.net.parameters(), lr=cfg.lr, weight_decay=1e-5)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg.steps)
        self.net.train()
        t0 = time.perf_counter()
        for _ in range(cfg.steps):
            idx = torch.randint(0, len(x_all), (cfg.batch_size,), generator=gen)
            x = x_all[idx.to(self.device)].permute(0, 3, 1, 2).float().div_(255.0)
            x = augment(x, cfg.texture, gen)
            with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
                recon = self.net(x)
            recon = recon.float()
            loss = F.mse_loss(recon, x) + cfg.ssim_weight * (1 - ssim_map(recon, x).mean())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
        self.train_seconds = time.perf_counter() - t0

    @torch.no_grad()
    def reconstruct(self, images: np.ndarray, batch: int = 64) -> np.ndarray:
        self.net.eval()
        out = []
        for i in range(0, len(images), batch):
            x = to_tensor(images[i : i + batch], self.device)
            out.append(self.net(x).clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy())
        return np.concatenate(out)

    @torch.no_grad()
    def predict(self, images: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        recon = self.reconstruct(images)
        err = ((images.astype(np.float32) / 255.0 - recon) ** 2).mean(axis=-1)
        maps = gaussian_smooth(err, self.cfg.sigma)
        return maps.reshape(len(maps), -1).max(axis=1), maps
