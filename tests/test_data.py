import cv2
import numpy as np

from src.data.mvtec import CATEGORIES, OBJECTS, TEXTURES, load_category
from src.data.splits import defect_kfold, split_train_val, subsample_defects


def _write_fake_category(root, category="bottle", size=64):
    base = root / category
    (base / "train" / "good").mkdir(parents=True)
    (base / "test" / "good").mkdir(parents=True)
    (base / "test" / "crack").mkdir(parents=True)
    (base / "ground_truth" / "crack").mkdir(parents=True)
    for i in range(3):
        cv2.imwrite(str(base / "train" / "good" / f"{i:03d}.png"), np.full((size, size, 3), 100, np.uint8))
    cv2.imwrite(str(base / "test" / "good" / "000.png"), np.full((size, size), 100, np.uint8))  # 흑백
    for i in range(2):
        cv2.imwrite(str(base / "test" / "crack" / f"{i:03d}.png"), np.full((size, size, 3), 50, np.uint8))
        mask = np.zeros((size, size), np.uint8)
        mask[: size // 2, : size // 2] = 255
        cv2.imwrite(str(base / "ground_truth" / "crack" / f"{i:03d}_mask.png"), mask)


def test_category_lists():
    assert len(CATEGORIES) == 15 and len(TEXTURES) == 5 and len(OBJECTS) == 10


def test_load_category_shapes_labels_and_cache(tmp_path):
    _write_fake_category(tmp_path / "mvtec")
    cache = tmp_path / "cache"
    d = load_category(tmp_path / "mvtec", "bottle", size=32, cache_dir=cache)
    assert d.train_images.shape == (3, 32, 32, 3) and d.train_images.dtype == np.uint8
    assert d.test_images.shape == (3, 32, 32, 3)  # 흑백 영상도 3채널
    assert list(d.test_defect_types) == ["crack", "crack", "good"]
    assert list(d.test_labels) == [1, 1, 0]
    assert d.test_masks[0, :16, :16].all() and not d.test_masks[0, 16:, 16:].any()
    assert not d.test_masks[2].any()
    assert d.type == "object"

    cached = load_category(tmp_path / "does-not-exist", "bottle", size=32, cache_dir=cache)
    np.testing.assert_array_equal(cached.test_masks, d.test_masks)
    assert list(cached.test_names) == list(d.test_names)


def test_split_train_val_is_disjoint_and_complete():
    fit, val = split_train_val(100, 0.15, seed=0)
    assert len(val) == 15 and len(set(fit) & set(val)) == 0
    assert sorted(np.concatenate([fit, val])) == list(range(100))


def test_defect_kfold_out_of_fold_coverage_and_stratification():
    types = np.array(["good"] * 20 + ["crack"] * 10 + ["hole"] * 15)
    folds = defect_kfold(types, n_splits=5, seed=0)
    evals = np.concatenate([ev for _, ev in folds])
    assert sorted(evals) == list(range(len(types)))  # 모든 영상이 정확히 한 번 평가
    for tr, ev in folds:
        assert len(set(tr) & set(ev)) == 0
        assert (types[ev] == "crack").sum() == 2 and (types[ev] == "hole").sum() == 3


def test_subsample_defects_round_robin():
    types = np.array(["a"] * 10 + ["b"] * 10 + ["c"] * 10)
    cand = np.arange(30)
    picked = subsample_defects(cand, types, 3, seed=0)
    assert len(picked) == 3 and sorted(types[picked]) == ["a", "b", "c"]
    assert len(subsample_defects(cand, types, None, seed=0)) == 30
    one = subsample_defects(cand, types, 1, seed=1)
    assert len(one) == 1 and one[0] in cand
