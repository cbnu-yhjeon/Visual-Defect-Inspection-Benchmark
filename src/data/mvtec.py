"""MVTec AD 로더.

모든 방법이 같은 해상도(기본 256x256, 정사각 리사이즈 · 크롭 없음)의 영상과 마스크를 쓴다.
크롭하지 않는 이유: 가장자리 결함이 잘려 나가면 픽셀 지표가 방법마다 달라진다.
흑백 카테고리(grid · screw · zipper)는 3채널로 복제해 모든 방법에 같은 입력 형식을 준다.
디코딩이 느려서(1024² PNG) 리사이즈 결과를 cache/에 npz로 저장한다.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

TEXTURES = ("carpet", "grid", "leather", "tile", "wood")
OBJECTS = (
    "bottle", "cable", "capsule", "hazelnut", "metal_nut",
    "pill", "screw", "toothbrush", "transistor", "zipper",
)
CATEGORIES = TEXTURES + OBJECTS


def category_type(category: str) -> str:
    return "texture" if category in TEXTURES else "object"


@dataclass
class CategoryData:
    category: str
    train_images: np.ndarray  # (N,H,W,3) uint8 RGB, 전부 정상
    test_images: np.ndarray  # (M,H,W,3) uint8 RGB
    test_labels: np.ndarray  # (M,) int, 0=정상 1=불량
    test_masks: np.ndarray  # (M,H,W) bool
    test_defect_types: np.ndarray  # (M,) str, 정상은 "good"
    test_names: np.ndarray  # (M,) str, "defect_type/000.png"

    @property
    def type(self) -> str:
        return category_type(self.category)


def _read_rgb(path: Path, size: int) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)  # 흑백도 3채널로 읽힘
    if img is None:
        raise FileNotFoundError(path)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _read_mask(path: Path, size: int) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(path)
    return cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST) > 127


def _scan(root: Path, category: str, size: int) -> CategoryData:
    base = root / category
    train_paths = sorted((base / "train" / "good").glob("*.png"))
    if not train_paths:
        raise FileNotFoundError(f"학습 영상이 없음: {base / 'train' / 'good'}")
    train = np.stack([_read_rgb(p, size) for p in train_paths])

    images, labels, masks, types, names = [], [], [], [], []
    for defect_dir in sorted(p for p in (base / "test").iterdir() if p.is_dir()):
        defect = defect_dir.name
        for p in sorted(defect_dir.glob("*.png")):
            images.append(_read_rgb(p, size))
            if defect == "good":
                labels.append(0)
                masks.append(np.zeros((size, size), dtype=bool))
            else:
                labels.append(1)
                masks.append(_read_mask(base / "ground_truth" / defect / f"{p.stem}_mask.png", size))
            types.append(defect)
            names.append(f"{defect}/{p.name}")
    return CategoryData(
        category=category,
        train_images=train,
        test_images=np.stack(images),
        test_labels=np.array(labels, dtype=np.int64),
        test_masks=np.stack(masks),
        test_defect_types=np.array(types),
        test_names=np.array(names),
    )


def load_category(
    root: str | Path, category: str, size: int = 256, cache_dir: str | Path | None = "cache"
) -> CategoryData:
    if category not in CATEGORIES:
        raise ValueError(f"알 수 없는 카테고리: {category}")
    cache_file = Path(cache_dir) / f"mvtec_{size}" / f"{category}.npz" if cache_dir else None
    if cache_file is not None and cache_file.exists():
        z = np.load(cache_file)
        return CategoryData(category=category, **{k: z[k] for k in z.files})
    data = _scan(Path(root), category, size)
    if cache_file is not None:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            cache_file,
            train_images=data.train_images,
            test_images=data.test_images,
            test_labels=data.test_labels,
            test_masks=data.test_masks,
            test_defect_types=data.test_defect_types,
            test_names=data.test_names,
        )
    return data
