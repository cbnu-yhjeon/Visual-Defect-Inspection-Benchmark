"""실험 실행기.

    python -m src.run --method C2 --categories all
    python -m src.run --method B  --categories bottle carpet

결과
  results/metrics/<method>.csv      카테고리별 지표 (재실행하면 해당 카테고리 행만 교체)
  cache/maps/<method>/<cat>.npz     이미지 점수 · 이상 맵(float16) · 임계값 (갤러리용)
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from src.classical.diff_otsu import DiffOtsu, DiffOtsuConfig
from src.classical.texture_lbp_gabor import TextureConfig, TextureLbpGabor
from src.common import Detector
from src.data.mvtec import CATEGORIES, OBJECTS, TEXTURES, CategoryData, load_category
from src.data.splits import defect_kfold, split_train_val, subsample_defects
from src.eval.auroc import f1_at_threshold, f1_max, image_auroc, pixel_auroc
from src.eval.pro import aupro
from src.supervised.transfer_cnn import SupervisedConfig, TransferClassifier
from src.unsupervised.conv_ae import ConvAEConfig, ConvAutoencoder
from src.unsupervised.patchcore import PatchCore, PatchCoreConfig

UNSUPERVISED = ("A1", "A2", "C1", "C2")
METRICS_DIR = Path("results/metrics")


def make_detector(method: str, category: str, params: dict, seed: int) -> Detector:
    texture = category in TEXTURES
    if method == "A1":
        # 텍스처는 정해진 자세가 없어 정렬이 의미 없다
        return DiffOtsu(DiffOtsuConfig(align=category in OBJECTS, **params))
    if method == "A2":
        return TextureLbpGabor(TextureConfig(seed=seed, **params))
    if method == "C1":
        return ConvAutoencoder(ConvAEConfig(texture=texture, seed=seed, **params))
    if method == "C2":
        return PatchCore(PatchCoreConfig(seed=seed, **params))
    raise ValueError(method)


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def evaluate(data: CategoryData, scores: np.ndarray, maps: np.ndarray, threshold: float) -> dict:
    return {
        "image_auroc": image_auroc(data.test_labels, scores),
        "pixel_auroc": pixel_auroc(data.test_masks, maps),
        "pro": aupro(data.test_masks, maps),
        "f1": f1_at_threshold(data.test_labels, scores, threshold),
        "f1_max": f1_max(data.test_labels, scores),
    }


def save_maps(method: str, category: str, scores, maps, **extra) -> None:
    out = Path("cache/maps") / method / f"{category}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, scores=scores, maps=maps.astype(np.float16), **extra)


def run_unsupervised(method: str, data: CategoryData, cfg: dict) -> dict:
    seed = cfg["seed"]
    fit_idx, val_idx = split_train_val(len(data.train_images), cfg["val_fraction"], seed)
    det = make_detector(method, data.category, cfg["methods"].get(method, {}), seed)

    t0 = time.perf_counter()
    det.fit(data.train_images[fit_idx])
    _sync()
    fit_s = time.perf_counter() - t0

    val_scores, val_maps = det.predict(data.train_images[val_idx])
    threshold = float(np.quantile(val_scores, cfg["image_threshold_quantile"]))
    pixel_threshold = float(np.quantile(val_maps, cfg["pixel_threshold_quantile"]))

    det.predict(data.test_images[:2])  # 워밍업 (cuDNN 알고리즘 선택 등)
    _sync()
    t0 = time.perf_counter()
    scores, maps = det.predict(data.test_images)
    _sync()
    ms = (time.perf_counter() - t0) / len(scores) * 1000

    save_maps(method, data.category, scores, maps, threshold=threshold, pixel_threshold=pixel_threshold)
    return {
        **evaluate(data, scores, maps, threshold),
        "ms_per_image": ms,
        "fit_seconds": fit_s,
        "n_train_normal": len(fit_idx),
        "n_train_defect": 0,
    }


def run_supervised(data: CategoryData, cfg: dict) -> list[dict]:
    p = dict(cfg["methods"]["B"])
    folds, settings = p.pop("folds"), p.pop("n_defects")
    seed = cfg["seed"]
    splits = defect_kfold(data.test_defect_types, folds, seed)
    rows = []
    for setting in settings:
        n = None if setting == "all" else int(setting)
        scores = np.zeros(len(data.test_images), dtype=np.float32)
        maps = np.zeros(data.test_masks.shape, dtype=np.float32)
        times, fit_s, n_defects = [], 0.0, []
        for k, (tr, ev) in enumerate(splits):
            defect_pool = tr[data.test_labels[tr] == 1]
            defect_idx = subsample_defects(defect_pool, data.test_defect_types, n, seed + k)
            normal = np.concatenate([data.train_images, data.test_images[tr[data.test_labels[tr] == 0]]])
            clf = TransferClassifier(
                SupervisedConfig(texture=data.category in TEXTURES, seed=seed + k, **p)
            )
            clf.fit(normal, data.test_images[defect_idx])
            fit_s += clf.train_seconds
            clf.predict(data.test_images[ev[:2]])
            _sync()
            t0 = time.perf_counter()
            scores[ev], maps[ev] = clf.predict(data.test_images[ev])
            _sync()
            times.append((time.perf_counter() - t0) / len(ev) * 1000)
            n_defects.append(len(defect_idx))
            del clf
        method = "B1" if setting == "all" else f"B2-n{setting}"
        if setting == "all":
            save_maps(method, data.category, scores, maps, threshold=0.5, pixel_threshold=0.5)
        rows.append({
            "method": method,
            **evaluate(data, scores, maps, 0.5),
            "ms_per_image": float(np.mean(times)),
            "fit_seconds": fit_s / folds,
            "n_train_normal": len(data.train_images),
            "n_train_defect": float(np.mean(n_defects)),
        })
    return rows


def upsert(method: str, rows: list[dict]) -> None:
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    path = METRICS_DIR / f"{method}.csv"
    new = pd.DataFrame(rows)
    if path.exists():
        old = pd.read_csv(path)
        old = old[~old["category"].isin(new["category"])]
        new = pd.concat([old, new], ignore_index=True)
    order = {c: i for i, c in enumerate(CATEGORIES)}
    new = new.sort_values("category", key=lambda s: s.map(order))
    new.to_csv(path, index=False, float_format="%.4f")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=[*UNSUPERVISED, "B"])
    ap.add_argument("--categories", nargs="+", default=["all"])
    ap.add_argument("--config", default="configs/default.yaml")
    a = ap.parse_args()
    cfg = yaml.safe_load(Path(a.config).read_text())
    cats = CATEGORIES if a.categories == ["all"] else a.categories

    for cat in cats:
        data = load_category(cfg["data_root"], cat, cfg["image_size"], cfg["cache_dir"])
        base = {"category": cat, "type": data.type}
        if a.method == "B":
            for row in run_supervised(data, cfg):
                method = row.pop("method")
                upsert(method, [{"method": method, **base, **row}])
                print(f"{method:7s} {cat:11s} img={row['image_auroc']:.3f} "
                      f"pix={row['pixel_auroc']:.3f} pro={row['pro']:.3f}", flush=True)
        else:
            row = run_unsupervised(a.method, data, cfg)
            upsert(a.method, [{"method": a.method, **base, **row}])
            print(f"{a.method:7s} {cat:11s} img={row['image_auroc']:.3f} pix={row['pixel_auroc']:.3f} "
                  f"pro={row['pro']:.3f} f1={row['f1']:.3f} {row['ms_per_image']:.1f}ms", flush=True)
        torch.cuda.empty_cache() if torch.cuda.is_available() else None


if __name__ == "__main__":
    main()
