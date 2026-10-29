"""15개 카테고리를 256x256으로 디코딩 · 리사이즈해 cache/에 저장 (카테고리별 병렬)."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor

from src.data.mvtec import CATEGORIES, load_category


def _one(args: tuple[str, str, int]) -> str:
    root, category, size = args
    d = load_category(root, category, size=size)
    return f"{category:12s} train={len(d.train_images):4d} test={len(d.test_images):4d}"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="data")
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--workers", type=int, default=6)
    a = p.parse_args()
    jobs = [(a.root, c, a.size) for c in CATEGORIES]
    with ProcessPoolExecutor(a.workers) as ex:
        for line in ex.map(_one, jobs):
            print(line, flush=True)


if __name__ == "__main__":
    main()
