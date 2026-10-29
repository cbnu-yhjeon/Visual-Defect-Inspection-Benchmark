"""분할 규칙.

- 정상 학습 영상 → fit / val: val은 임계값 결정에만 쓴다 (시험 세트로 임계값을 정하지 않기 위해).
- 시험 세트 → k-fold (지도학습 B 전용): 결함 종류로 층화해, 모든 시험 영상이 정확히 한 번
  학습에 쓰이지 않은 모델로 평가된다 (out-of-fold).
"""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import StratifiedKFold


def split_train_val(n: int, val_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_val = max(1, round(n * val_fraction))
    return np.sort(perm[n_val:]), np.sort(perm[:n_val])


def defect_kfold(
    defect_types: np.ndarray, n_splits: int, seed: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    """시험 세트 인덱스를 (train_idx, eval_idx) k개로 나눈다. 결함 종류("good" 포함)로 층화."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    dummy = np.zeros(len(defect_types))
    return [(tr, ev) for tr, ev in skf.split(dummy, defect_types)]


def subsample_defects(
    candidates: np.ndarray, defect_types: np.ndarray, n: int | None, seed: int
) -> np.ndarray:
    """불량 후보 인덱스에서 n개를 결함 종류별로 번갈아(round-robin) 뽑는다. n=None이면 전부."""
    candidates = np.asarray(candidates)
    if n is None or n >= len(candidates):
        return candidates
    rng = np.random.default_rng(seed)
    pools = {}
    for t in np.unique(defect_types[candidates]):
        members = candidates[defect_types[candidates] == t]
        pools[t] = list(rng.permutation(members))
    order = list(rng.permutation(sorted(pools)))
    picked = []
    while len(picked) < n:
        for t in order:
            if pools[t] and len(picked) < n:
                picked.append(pools[t].pop())
    return np.sort(np.array(picked))
