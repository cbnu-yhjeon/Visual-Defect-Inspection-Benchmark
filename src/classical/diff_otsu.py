"""A1. 정상 평균 영상과의 차분 + Otsu + 모폴로지.

1) Lab 변환 후 L 채널에 CLAHE (조명 편차 완화, 산업 CV Lec 3)
2) ECC로 기준 영상에 유클리드 정렬 (회전 + 평행이동, 물체 카테고리만)
3) 정상 영상의 픽셀별 평균 · 표준편차로 z-점수 차분 맵 → 가우시안 평활
4) 결함 영역: max(Otsu, 정상 기준 하한)로 이진화 → 열림 · 닫힘 → 작은 컨투어 제거 (Lec 4 · 5)

이미지 점수는 차분 맵의 최댓값이다. 픽셀 지표는 연속 맵으로, 이진 영역은 갤러리 ·
설명용으로 쓴다.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from src.common import Detector, gaussian_smooth


@dataclass
class DiffOtsuConfig:
    clahe_clip: float = 2.0
    clahe_tiles: int = 8
    align: bool = True
    align_rounds: int = 2  # 기준 영상 = 정렬된 영상의 평균, 이를 다시 기준으로 재정렬
    ecc_iterations: int = 60
    ecc_downscale: int = 2  # ECC는 1/2 해상도에서 추정해 속도 확보
    std_floor: float = 0.02  # [0,1] 스케일. 배경처럼 분산이 0에 가까운 곳의 z 폭주 방지
    sigma: float = 4.0
    morph_kernel: int = 5
    min_area: int = 30


class DiffOtsu(Detector):
    def __init__(self, cfg: DiffOtsuConfig | None = None):
        self.cfg = cfg or DiffOtsuConfig()
        self._clahe = cv2.createCLAHE(self.cfg.clahe_clip, (self.cfg.clahe_tiles,) * 2)
        self.reference: np.ndarray | None = None  # L 채널 기준 영상
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None
        self.pixel_floor = 0.0

    # --- 전처리 · 정렬 ---------------------------------------------------------------
    def _prep(self, img: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        lab[..., 0] = self._clahe.apply(lab[..., 0])
        return lab.astype(np.float32) / 255.0

    def _align(self, lab: np.ndarray) -> np.ndarray:
        if not self.cfg.align or self.reference is None:
            return lab
        s = self.cfg.ecc_downscale
        h, w = lab.shape[:2]
        ref = cv2.resize(self.reference, (w // s, h // s), interpolation=cv2.INTER_AREA)
        mov = cv2.resize(lab[..., 0], (w // s, h // s), interpolation=cv2.INTER_AREA)
        warp = np.eye(2, 3, dtype=np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, self.cfg.ecc_iterations, 1e-4)
        try:
            _, warp = cv2.findTransformECC(ref, mov, warp, cv2.MOTION_EUCLIDEAN, criteria, None, 5)
        except cv2.error:
            return lab  # 수렴 실패 — 정렬하지 않은 채로 비교 (큰 회전에서 흔함)
        warp[:, 2] *= s
        return cv2.warpAffine(
            lab, warp, (w, h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_REPLICATE,
        )

    # --- 학습 · 추론 ------------------------------------------------------------------
    def fit(self, images: np.ndarray) -> None:
        labs = np.stack([self._prep(x) for x in images])
        self.reference = np.median(labs[..., 0], axis=0).astype(np.float32)
        if self.cfg.align:
            aligned = labs
            for _ in range(self.cfg.align_rounds):
                aligned = np.stack([self._align(x) for x in labs])
                self.reference = aligned[..., 0].mean(axis=0).astype(np.float32)
            labs = aligned
        self.mean = labs.mean(axis=0)
        self.std = np.maximum(labs.std(axis=0), self.cfg.std_floor)

    def _raw_map(self, img: np.ndarray) -> np.ndarray:
        z = (self._align(self._prep(img)) - self.mean) / self.std
        return np.sqrt((z**2).mean(axis=-1))

    def predict(self, images: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        maps = gaussian_smooth(np.stack([self._raw_map(x) for x in images]), self.cfg.sigma)
        return maps.reshape(len(maps), -1).max(axis=1), maps

    # --- 결함 영역 (설명용) -------------------------------------------------------------
    def calibrate_segmentation(self, normal_maps: np.ndarray, quantile: float = 0.999) -> None:
        """정상 검증 영상 맵의 상위 분위수를 이진화 하한으로 쓴다 (Otsu가 정상 영상에서도
        항상 무언가를 분할해 버리는 문제 방지)."""
        self.pixel_floor = float(np.quantile(normal_maps, quantile))

    def segment(self, amap: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
        """연속 맵 -> (이진 결함 마스크, 컨투어 목록)."""
        scaled = np.clip(amap / (amap.max() + 1e-8) * 255, 0, 255).astype(np.uint8)
        otsu, _ = cv2.threshold(scaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thr = max(otsu / 255 * amap.max(), self.pixel_floor)
        binary = (amap > thr).astype(np.uint8)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.cfg.morph_kernel,) * 2)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = [c for c in contours if cv2.contourArea(c) >= self.cfg.min_area]
        mask = np.zeros_like(binary)
        cv2.drawContours(mask, contours, -1, 1, thickness=cv2.FILLED)
        return mask.astype(bool), contours
