"""results/metrics/*.csv → results/summary.md · results/figures/*.png · results/gallery/*.png."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data.mvtec import CATEGORIES, load_category

RESULTS = Path("results")
FIG = RESULTS / "figures"
GALLERY = RESULTS / "gallery"

MAIN = ["A1", "A2", "C1", "C2", "B1"]
LABELS = {
    "A1": "A1 차분+Otsu",
    "A2": "A2 LBP·Gabor",
    "C1": "C1 Conv AE",
    "C2": "C2 PatchCore",
    "B1": "B1 지도 CNN*",
}
# 범주형 색: 방법마다 고정 (reference palette 슬롯 순서)
COLORS = {"A1": "#2a78d6", "A2": "#eb6834", "C1": "#1baf7a", "C2": "#eda100", "B1": "#e87ba4"}
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
SEQ = "Blues"
METRICS = [("image_auroc", "이미지 AUROC"), ("pixel_auroc", "픽셀 AUROC"), ("pro", "PRO (AUPRO@0.3)")]


def _style() -> None:
    # 한글 폰트가 시스템에 없으면 cache/fonts/의 NanumGothic(OFL)을 쓴다 (scripts/run_all.sh가 받음)
    for ttf in Path("cache/fonts").glob("*.ttf"):
        matplotlib.font_manager.fontManager.addfont(str(ttf))
    for name in ("NanumGothic", "Noto Sans CJK KR", "Noto Sans KR", "UnDotum"):
        if any(name in f.name for f in matplotlib.font_manager.fontManager.ttflist):
            plt.rcParams["font.family"] = name
            break
    plt.rcParams.update({
        "axes.unicode_minus": False, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
        "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True, "font.size": 10,
    })


def load_metrics() -> pd.DataFrame:
    frames = [pd.read_csv(p) for p in sorted((RESULTS / "metrics").glob("*.csv"))]
    return pd.concat(frames, ignore_index=True)


def _fmt(v: float) -> str:
    return "–" if pd.isna(v) else f"{v:.3f}"


def table(df: pd.DataFrame, metric: str, methods: list[str]) -> str:
    p = df[df.method.isin(methods)].pivot(index="category", columns="method", values=metric)
    p = p.reindex(index=[c for c in CATEGORIES if c in p.index], columns=methods)
    types = df.drop_duplicates("category").set_index("category")["type"]
    lines = ["| 카테고리 | 유형 | " + " | ".join(LABELS.get(m, m) for m in methods) + " |",
             "|---|---|" + "---:|" * len(methods)]
    for cat, row in p.iterrows():
        best = row.max()
        cells = [f"**{_fmt(v)}**" if v == best else _fmt(v) for v in row]
        lines.append(f"| {cat} | {types[cat]} | " + " | ".join(cells) + " |")
    for name, mask in (("텍스처 평균", p.index.map(types) == "texture"),
                       ("물체 평균", p.index.map(types) == "object"), ("전체 평균", slice(None))):
        mean = p[mask].mean()
        lines.append(f"| **{name}** | | " + " | ".join(f"**{_fmt(v)}**" for v in mean) + " |")
    return "\n".join(lines)


def practical_table(df: pd.DataFrame) -> str:
    g = df[df.method.isin(MAIN)].groupby("method")
    agg = g.agg(f1=("f1", "mean"), f1_max=("f1_max", "mean"), ms=("ms_per_image", "median"),
                fit=("fit_seconds", "median"), normal=("n_train_normal", "mean"),
                defect=("n_train_defect", "mean")).reindex(MAIN)
    lines = ["| 방법 | F1 (검증 임계값) | F1-max (오라클) | 추론 ms/장 (중앙값) | 학습 초 (중앙값) | 학습 정상 수 | 학습 불량 수 |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for m, r in agg.iterrows():
        lines.append(f"| {LABELS[m]} | {r.f1:.3f} | {r.f1_max:.3f} | {r.ms:.1f} | {r.fit:.1f} | "
                     f"{r.normal:.0f} | {r.defect:.0f} |")
    return "\n".join(lines)


def b2_table(df: pd.DataFrame) -> str:
    settings = ["B2-n1", "B2-n5", "B2-n10", "B1"]
    p = df[df.method.isin(settings + ["C2"])].pivot(index="category", columns="method", values="image_auroc")
    p = p.reindex(index=[c for c in CATEGORIES if c in p.index], columns=settings + ["C2"])
    lines = ["| 카테고리 | 불량 1 | 불량 5 | 불량 10 | 불량 전체 | C2 PatchCore | 역전되는 불량 수 |",
             "|---|---:|---:|---:|---:|---:|---|"]
    for cat, row in p.iterrows():
        beats = [s for s in settings if row[s] >= row["C2"]]
        cross = {"B2-n1": "1장", "B2-n5": "5장", "B2-n10": "10장", "B1": "전체"}[beats[0]] if beats else "역전 없음"
        lines.append(f"| {cat} | " + " | ".join(_fmt(row[s]) for s in settings + ["C2"]) + f" | {cross} |")
    mean = p.mean()
    lines.append("| **평균** | " + " | ".join(f"**{_fmt(mean[s])}**" for s in settings + ["C2"]) + " | |")
    return "\n".join(lines)


# --- 그림 -----------------------------------------------------------------------------------
def fig_heatmap(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 7), sharey=True)
    for ax, (metric, title) in zip(axes, METRICS):
        p = df[df.method.isin(MAIN)].pivot(index="category", columns="method", values=metric)
        p = p.reindex(index=[c for c in CATEGORIES if c in p.index], columns=MAIN)
        ax.imshow(p.values, cmap=SEQ, vmin=0.5, vmax=1.0, aspect="auto")
        ax.grid(False)
        for (i, j), v in np.ndenumerate(p.values):
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8.5,
                        color="white" if v > 0.85 else INK)
        ax.set_xticks(range(len(MAIN)), [LABELS[m].split(" ", 1)[0] for m in MAIN])
        ax.set_yticks(range(len(p.index)), p.index)
        ax.axhline(4.5, color=INK, linewidth=1.5)  # 텍스처 | 물체 경계
        ax.set_title(title, color=INK, fontsize=12, loc="left")
    fig.suptitle("카테고리 × 방법 (위 5줄 텍스처, 아래 10줄 물체 · 0.5 미만은 같은 색) — B1은 비표준 프로토콜",
                 color=INK2, fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(FIG / "heatmap_by_category.png", dpi=130)
    plt.close(fig)


def fig_texture_vs_object(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
    x = np.arange(2)
    w = 0.16
    for ax, (metric, title) in zip(axes, METRICS):
        means = df[df.method.isin(MAIN)].groupby(["method", "type"])[metric].mean()
        for k, m in enumerate(MAIN):
            vals = [means.get((m, t), np.nan) for t in ("texture", "object")]
            bars = ax.bar(x + (k - 2) * w, vals, w * 0.88, color=COLORS[m], label=LABELS[m])
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center",
                        fontsize=7.5, color=INK2)
        ax.set_xticks(x, ["텍스처 (5)", "물체 (10)"])
        ax.set_ylim(0.4, 1.05)
        ax.grid(axis="x", visible=False)
        ax.set_title(title, color=INK, fontsize=12, loc="left")
    axes[0].legend(loc="lower left", frameon=False, fontsize=8.5, ncol=1)
    fig.tight_layout()
    fig.savefig(FIG / "texture_vs_object.png", dpi=130)
    plt.close(fig)


def fig_b2_curve(df: pd.DataFrame) -> None:
    settings = ["B2-n1", "B2-n5", "B2-n10", "B1"]
    xs = np.arange(len(settings))
    xt = ["1", "5", "10", "전체"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), gridspec_kw={"width_ratios": [1, 1.6]})

    ax = axes[0]
    for t, ls in (("texture", "--"), ("object", "-")):
        sub = df[df.type == t]
        ys = [sub[sub.method == s].image_auroc.mean() for s in settings]
        ax.plot(xs, ys, ls, color=COLORS["B1"], linewidth=2, marker="o", markersize=7,
                label=f"지도 CNN ({'텍스처' if t == 'texture' else '물체'})")
        c2 = sub[sub.method == "C2"].image_auroc.mean()
        ax.axhline(c2, color=COLORS["C2"], linestyle=ls, linewidth=2,
                   label=f"PatchCore, 불량 0장 ({'텍스처' if t == 'texture' else '물체'})")
    ax.set_xticks(xs, xt)
    ax.set_xlabel("학습에 쓴 불량 영상 수 (fold당)")
    ax.set_ylabel("이미지 AUROC (평균)")
    ax.set_title("불량 수에 따른 지도학습 vs 비지도 기준선", color=INK, fontsize=12, loc="left")
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")

    ax = axes[1]
    gap = df[df.method.isin(settings)].pivot(index="category", columns="method", values="image_auroc")
    c2 = df[df.method == "C2"].set_index("category").image_auroc
    gap = gap.reindex(index=[c for c in CATEGORIES if c in gap.index], columns=settings).sub(c2, axis=0)
    lim = max(0.05, np.nanmax(np.abs(gap.values)))
    ax.imshow(gap.values.T, cmap="RdBu", vmin=-lim, vmax=lim, aspect="auto")
    ax.grid(False)
    for (i, j), v in np.ndenumerate(gap.values.T):
        ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=7.5,
                color="white" if abs(v) > lim * 0.6 else INK)
    ax.set_yticks(range(len(settings)), [f"불량 {t}" for t in xt])
    ax.set_xticks(range(len(gap.index)), gap.index, rotation=45, ha="right")
    ax.set_title("지도 CNN − PatchCore (이미지 AUROC) · 파랑 = 지도학습 우세, 빨강 = PatchCore 우세",
                 color=INK, fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig(FIG / "b2_defect_count.png", dpi=130)
    plt.close(fig)


def fig_speed(df: pd.DataFrame) -> None:
    agg = df[df.method.isin(MAIN)].groupby("method").agg(
        ms=("ms_per_image", "median"), auroc=("image_auroc", "mean")).reindex(MAIN)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for m, r in agg.iterrows():
        ax.scatter(r.ms, r.auroc, s=90, color=COLORS[m], edgecolor=SURFACE, linewidth=2, zorder=3)
        ax.annotate(LABELS[m], (r.ms, r.auroc), xytext=(7, -3), textcoords="offset points",
                    fontsize=9, color=INK)
    ax.set_xscale("log")
    ax.set_xlabel("추론 시간 ms/장 (중앙값, 로그 축 · GPU 공유 상태에서 측정)")
    ax.set_ylabel("이미지 AUROC (15개 평균)")
    ax.set_title("정확도 vs 속도", color=INK, fontsize=12, loc="left")
    fig.tight_layout()
    fig.savefig(FIG / "accuracy_vs_speed.png", dpi=130)
    plt.close(fig)


def gallery(categories: list[str]) -> None:
    from src.classical.diff_otsu import DiffOtsu  # 결함 영역 이진화 규칙 재사용

    seg = DiffOtsu()
    for cat in categories:
        maps = {}
        for m in MAIN:
            f = Path("cache/maps") / m / f"{cat}.npz"
            if f.exists():
                maps[m] = np.load(f)
        if not maps:
            continue
        d = load_category("data", cat)
        rng = np.random.default_rng(0)
        bad = [rng.choice(np.where(d.test_defect_types == t)[0])
               for t in sorted(set(d.test_defect_types) - {"good"})][:3]
        rows = bad + [int(np.where(d.test_labels == 0)[0][0])]
        cols = 2 + len(maps)
        fig, axes = plt.subplots(len(rows), cols, figsize=(1.9 * cols, 1.95 * len(rows)))
        for r, i in enumerate(rows):
            axes[r, 0].imshow(d.test_images[i])
            axes[r, 0].set_ylabel(d.test_defect_types[i], fontsize=8, color=INK)
            axes[r, 1].imshow(d.test_images[i])
            axes[r, 1].contour(d.test_masks[i], levels=[0.5], colors="#e34948", linewidths=1)
            for c, (m, z) in enumerate(maps.items(), start=2):
                amap = z["maps"][i].astype(np.float32)
                vmax = float(np.quantile(z["maps"].astype(np.float32), 0.999))
                axes[r, c].imshow(d.test_images[i])
                axes[r, c].imshow(amap, cmap="inferno", alpha=0.55, vmin=float(z["maps"].min()), vmax=vmax)
                seg.pixel_floor = float(z["pixel_threshold"])
                region, _ = seg.segment(amap)
                if region.any():
                    axes[r, c].contour(region, levels=[0.5], colors="#1baf7a", linewidths=1)
                verdict = "불량" if z["scores"][i] >= z["threshold"] else "정상"
                axes[r, c].set_xlabel(f"{verdict} ({z['scores'][i]:.2f})", fontsize=7, color=INK2)
                if r == 0:
                    axes[r, c].set_title(LABELS[m].split(" ", 1)[0], fontsize=9, color=INK)
        axes[0, 0].set_title("입력", fontsize=9, color=INK)
        axes[0, 1].set_title("정답(빨강)", fontsize=9, color=INK)
        for ax in axes.ravel():
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
        fig.suptitle(f"{cat} — 맵 색 = 이상 점수, 초록 윤곽 = 정상 검증 기준으로 이진화한 결함 영역, "
                     "아래 글자 = 이미지 판정(점수)", fontsize=8, color=INK2, x=0.01, ha="left")
        fig.tight_layout()
        fig.savefig(GALLERY / f"{cat}.png", dpi=100)
        plt.close(fig)


def main() -> None:
    _style()
    FIG.mkdir(parents=True, exist_ok=True)
    GALLERY.mkdir(parents=True, exist_ok=True)
    df = load_metrics()
    df.to_csv(RESULTS / "all_metrics.csv", index=False, float_format="%.4f")

    parts = ["# 결과 요약 (자동 생성: `python -m scripts.make_report`)\n",
             "\\* B1은 시험 세트 불량을 5-fold로 학습에 쓴 **비표준 프로토콜** 결과다 (out-of-fold 평가).\n"]
    for metric, title in METRICS:
        parts += [f"## {title}\n", table(df, metric, MAIN), ""]
    parts += ["## 현장성 (15개 카테고리 평균 · 중앙값)\n", practical_table(df), "",
              "## G3 — 불량 수별 지도학습 vs PatchCore (이미지 AUROC)\n", b2_table(df), ""]
    (RESULTS / "summary.md").write_text("\n".join(parts), encoding="utf-8")

    fig_heatmap(df)
    fig_texture_vs_object(df)
    if "B2-n1" in set(df.method):
        fig_b2_curve(df)
    fig_speed(df)
    gallery(CATEGORIES)
    print((RESULTS / "summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
