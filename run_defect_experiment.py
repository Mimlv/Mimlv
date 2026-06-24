from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


DATASETS = {
    "KC1": {
        "path": Path("data/kc1.arff"),
        "openml_id": 1067,
        "download": "https://www.openml.org/data/download/53950/kc1.arff",
    },
    "JM1": {
        "path": Path("data/jm1.arff"),
        "openml_id": 1053,
        "download": "https://www.openml.org/data/download/53936/jm1.arff",
    },
    "PC1": {
        "path": Path("data/pc1.arff"),
        "openml_id": 1068,
        "download": "https://www.openml.org/data/download/53951/pc1.arff",
    },
}

BUDGETS = [0.10, 0.20, 0.30]

FEATURE_GROUPS = {
    "规模度量": ["loc", "lOCode", "lOComment", "lOBlank", "locCodeAndComment"],
    "控制复杂度": ["v(g)", "ev(g)", "iv(g)", "branchCount"],
    "Halstead度量": ["n", "v", "l", "d", "i", "e", "b", "t", "uniq_Op", "uniq_Opnd", "total_Op", "total_Opnd"],
}


def read_arff(path: Path) -> pd.DataFrame:
    attrs: list[str] = []
    rows: list[list[str]] = []
    in_data = False
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("%"):
                continue
            lower = line.lower()
            if lower.startswith("@attribute"):
                parts = line.split()
                attrs.append(parts[1])
            elif lower.startswith("@data"):
                in_data = True
            elif in_data:
                rows.append([x.strip() for x in line.split(",")])
    df = pd.DataFrame(rows, columns=attrs)
    for col in df.columns[:-1]:
        df[col] = pd.to_numeric(df[col].replace("?", np.nan), errors="coerce")
        df[col] = df[col].fillna(df[col].median())
    df["defects"] = df["defects"].astype(str).str.lower().map({"true": 1, "false": 0}).astype(int)
    return df


def stratified_split(y: np.ndarray, test_ratio: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train_idx: list[int] = []
    test_idx: list[int] = []
    for cls in [0, 1]:
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        n_test = max(1, int(round(len(idx) * test_ratio)))
        test_idx.extend(idx[:n_test].tolist())
        train_idx.extend(idx[n_test:].tolist())
    rng.shuffle(train_idx)
    rng.shuffle(test_idx)
    return np.array(train_idx), np.array(test_idx)


def standardize(train_x: np.ndarray, test_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = train_x.mean(axis=0)
    sigma = train_x.std(axis=0)
    sigma[sigma == 0] = 1.0
    return (train_x - mu) / sigma, (test_x - mu) / sigma


def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -35, 35)
    return 1.0 / (1.0 + np.exp(-z))


def train_logistic_balanced(x: np.ndarray, y: np.ndarray, seed: int) -> tuple[np.ndarray, float]:
    rng = np.random.default_rng(seed)
    w = rng.normal(0, 0.01, x.shape[1])
    b = 0.0
    pos = max(1, int(y.sum()))
    neg = max(1, int(len(y) - y.sum()))
    weights = np.where(y == 1, len(y) / (2 * pos), len(y) / (2 * neg))
    lr = 0.08
    l2 = 0.002
    for _ in range(650):
        p = sigmoid(x @ w + b)
        err = (p - y) * weights
        grad_w = x.T @ err / len(y) + l2 * w
        grad_b = float(err.mean())
        w -= lr * grad_w
        b -= lr * grad_b
    return w, b


def gaussian_nb_score(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray) -> np.ndarray:
    eps = 1e-6
    priors = []
    means = []
    vars_ = []
    for cls in [0, 1]:
        x_cls = train_x[train_y == cls]
        priors.append(np.log((len(x_cls) + 1) / (len(train_y) + 2)))
        means.append(x_cls.mean(axis=0))
        vars_.append(x_cls.var(axis=0) + eps)
    logps = []
    for cls in [0, 1]:
        log_likelihood = -0.5 * np.sum(np.log(2 * np.pi * vars_[cls]) + ((test_x - means[cls]) ** 2) / vars_[cls], axis=1)
        logps.append(priors[cls] + log_likelihood)
    logps = np.vstack(logps).T
    m = logps.max(axis=1, keepdims=True)
    probs = np.exp(logps - m)
    probs = probs / probs.sum(axis=1, keepdims=True)
    return probs[:, 1]


def complexity_score(feature_cols: list[str], x_test_std: np.ndarray) -> np.ndarray:
    wanted = ["loc", "v(g)", "ev(g)", "iv(g)", "branchCount", "e", "b"]
    idx = [feature_cols.index(col) for col in wanted if col in feature_cols]
    if not idx:
        return x_test_std.mean(axis=1)
    return x_test_std[:, idx].mean(axis=1)


def auc_score(y: np.ndarray, score: np.ndarray) -> float:
    pos = score[y == 1]
    neg = score[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    combined = np.concatenate([pos, neg])
    order = np.argsort(combined)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(combined) + 1)
    sorted_scores = combined[order]
    start = 0
    while start < len(combined):
        end = start + 1
        while end < len(combined) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2
        start = end
    pos_ranks = ranks[: len(pos)]
    return float((pos_ranks.sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def effort_metrics(y: np.ndarray, score: np.ndarray, loc: np.ndarray, budget_ratio: float) -> dict[str, float]:
    order = np.argsort(-score)
    total_loc = float(loc.sum())
    budget = total_loc * budget_ratio
    selected: list[int] = []
    used = 0.0
    for idx in order:
        if used + loc[idx] > budget and selected:
            continue
        if used + loc[idx] <= budget or not selected:
            selected.append(int(idx))
            used += float(loc[idx])
        if used >= budget:
            break
    selected_arr = np.array(selected, dtype=int)
    found = float(y[selected_arr].sum()) if len(selected_arr) else 0.0
    total_defects = max(1.0, float(y.sum()))
    return {
        f"Recall@{int(budget_ratio * 100)}%LOC": found / total_defects,
        f"Precision@{int(budget_ratio * 100)}%LOC": found / max(1.0, float(len(selected_arr))),
        f"InspectedModules@{int(budget_ratio * 100)}%LOC": float(len(selected_arr)),
    }


def metrics_all_budgets(y: np.ndarray, score: np.ndarray, loc: np.ndarray) -> dict[str, float]:
    out = {"AUC": auc_score(y, score)}
    for budget in BUDGETS:
        out.update(effort_metrics(y, score, loc, budget))
    return out


def run_model_comparison(df: pd.DataFrame, dataset_name: str) -> list[dict[str, float | str]]:
    feature_cols = [c for c in df.columns if c != "defects"]
    x_all = df[feature_cols].to_numpy(dtype=float)
    y_all = df["defects"].to_numpy(dtype=int)
    loc_all = np.maximum(1.0, df["loc"].to_numpy(dtype=float))
    rows: list[dict[str, float | str]] = []
    for seed in range(10):
        train_idx, test_idx = stratified_split(y_all, 0.3, 2026 + seed)
        x_train, x_test = standardize(x_all[train_idx], x_all[test_idx])
        y_train, y_test = y_all[train_idx], y_all[test_idx]
        loc_test = loc_all[test_idx]

        w, b = train_logistic_balanced(x_train, y_train, 2026 + seed)
        scores = {
            "LOC排序基线": loc_test,
            "复杂度加权": complexity_score(feature_cols, x_test),
            "GaussianNB": gaussian_nb_score(x_train, y_train, x_test),
            "平衡Logistic": sigmoid(x_test @ w + b),
        }
        for model, score in scores.items():
            rows.append({"Dataset": dataset_name, "Seed": seed, "Model": model, **metrics_all_budgets(y_test, score, loc_test)})
    return rows


def run_ablation(df: pd.DataFrame, dataset_name: str) -> list[dict[str, float | str]]:
    all_cols = [c for c in df.columns if c != "defects"]
    group_map = {**FEATURE_GROUPS, "全部度量": all_cols}
    y_all = df["defects"].to_numpy(dtype=int)
    loc_all = np.maximum(1.0, df["loc"].to_numpy(dtype=float))
    rows: list[dict[str, float | str]] = []
    for group, cols in group_map.items():
        cols = [c for c in cols if c in df.columns]
        x_all = df[cols].to_numpy(dtype=float)
        for seed in range(10):
            train_idx, test_idx = stratified_split(y_all, 0.3, 3026 + seed)
            x_train, x_test = standardize(x_all[train_idx], x_all[test_idx])
            y_train, y_test = y_all[train_idx], y_all[test_idx]
            loc_test = loc_all[test_idx]
            w, b = train_logistic_balanced(x_train, y_train, 3026 + seed)
            score = sigmoid(x_test @ w + b)
            rows.append({"Dataset": dataset_name, "Seed": seed, "FeatureGroup": group, **metrics_all_budgets(y_test, score, loc_test)})
    return rows


def run_cross_project(datasets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for source, source_df in datasets.items():
        for target, target_df in datasets.items():
            if source == target:
                continue
            feature_cols = [c for c in source_df.columns if c != "defects" and c in target_df.columns]
            train_x = source_df[feature_cols].to_numpy(dtype=float)
            train_y = source_df["defects"].to_numpy(dtype=int)
            test_x = target_df[feature_cols].to_numpy(dtype=float)
            test_y = target_df["defects"].to_numpy(dtype=int)
            test_loc = np.maximum(1.0, target_df["loc"].to_numpy(dtype=float))
            train_x, test_x = standardize(train_x, test_x)
            w, b = train_logistic_balanced(train_x, train_y, 4026)
            score = sigmoid(test_x @ w + b)
            rows.append({"Source": source, "Target": target, **metrics_all_budgets(test_y, score, test_loc)})
    return pd.DataFrame(rows)


def run_feature_correlation(datasets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for name, df in datasets.items():
        y = df["defects"]
        for feature in [c for c in df.columns if c != "defects"]:
            corr = df[feature].corr(y)
            rows.append({"Dataset": name, "Feature": feature, "Correlation": float(corr), "AbsCorrelation": abs(float(corr))})
    out = pd.DataFrame(rows)
    return out.sort_values(["Dataset", "AbsCorrelation"], ascending=[True, False]).groupby("Dataset").head(6).reset_index(drop=True)


def summarize_model(rows: list[dict[str, float | str]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    metric_cols = ["AUC"] + [f"Recall@{int(b * 100)}%LOC" for b in BUDGETS] + [f"Precision@{int(b * 100)}%LOC" for b in BUDGETS]
    summary = df.groupby(["Dataset", "Model"])[metric_cols].agg(["mean", "std"]).reset_index()
    summary.columns = ["_".join(col).strip("_") for col in summary.columns]
    return summary


def summarize_ablation(rows: list[dict[str, float | str]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    metric_cols = ["AUC", "Recall@20%LOC", "Precision@20%LOC"]
    summary = df.groupby(["Dataset", "FeatureGroup"])[metric_cols].agg(["mean", "std"]).reset_index()
    summary.columns = ["_".join(col).strip("_") for col in summary.columns]
    return summary


def load_fonts():
    from PIL import ImageFont

    try:
        font = ImageFont.truetype("C:/Windows/Fonts/simsun.ttc", 24)
        font_b = ImageFont.truetype("C:/Windows/Fonts/simhei.ttf", 30)
        font_s = ImageFont.truetype("C:/Windows/Fonts/simsun.ttc", 20)
    except OSError:
        font = font_b = font_s = ImageFont.load_default()
    return font, font_b, font_s


def make_grouped_bar(summary: pd.DataFrame, out_path: Path) -> None:
    from PIL import Image, ImageDraw

    width, height = 1400, 760
    margin_l, margin_r, margin_t, margin_b = 110, 40, 80, 125
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    font, font_b, font_s = load_fonts()
    draw.text((width // 2, 24), "20%代码量测试预算下的缺陷召回率", fill="#111111", font=font_b, anchor="ma")
    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b
    x0, y0 = margin_l, margin_t
    y_axis = y0 + plot_h
    draw.line((x0, y0, x0, y_axis), fill="#222222", width=2)
    draw.line((x0, y_axis, x0 + plot_w, y_axis), fill="#222222", width=2)
    for tick in range(0, 7):
        val = tick / 6
        y = y_axis - val * plot_h
        draw.line((x0 - 6, y, x0 + plot_w, y), fill="#dddddd", width=1)
        draw.text((x0 - 12, y), f"{val:.2f}", fill="#333333", font=font_s, anchor="rm")
    datasets = ["KC1", "JM1", "PC1"]
    models = ["LOC排序基线", "复杂度加权", "GaussianNB", "平衡Logistic"]
    colors = {"LOC排序基线": "#78909C", "复杂度加权": "#F9A825", "GaussianNB": "#2E7D32", "平衡Logistic": "#1565C0"}
    group_w = plot_w / len(datasets)
    bar_w = group_w / 6
    for i, ds in enumerate(datasets):
        center = x0 + group_w * (i + 0.5)
        for j, model in enumerate(models):
            row = summary[(summary["Dataset"] == ds) & (summary["Model"] == model)].iloc[0]
            val = float(row["Recall@20%LOC_mean"])
            h = val * plot_h
            bx0 = center - 2 * bar_w + j * bar_w
            bx1 = bx0 + bar_w * 0.8
            draw.rectangle((bx0, y_axis - h, bx1, y_axis), fill=colors[model])
            draw.text(((bx0 + bx1) / 2, y_axis - h - 8), f"{val:.2f}", fill="#111111", font=font_s, anchor="mb")
        draw.text((center, y_axis + 30), ds, fill="#111111", font=font, anchor="ma")
    legend_x, legend_y = margin_l, height - 65
    for model in models:
        draw.rectangle((legend_x, legend_y, legend_x + 24, legend_y + 16), fill=colors[model])
        draw.text((legend_x + 32, legend_y - 4), model, fill="#111111", font=font_s)
        legend_x += 240
    img.save(out_path)


def make_budget_line(summary: pd.DataFrame, out_path: Path) -> None:
    from PIL import Image, ImageDraw

    width, height = 1200, 720
    margin_l, margin_r, margin_t, margin_b = 100, 60, 80, 105
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    font, font_b, font_s = load_fonts()
    draw.text((width // 2, 24), "平衡Logistic在不同测试预算下的缺陷召回率", fill="#111111", font=font_b, anchor="ma")
    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b
    x0, y0 = margin_l, margin_t
    y_axis = y0 + plot_h
    draw.line((x0, y0, x0, y_axis), fill="#222222", width=2)
    draw.line((x0, y_axis, x0 + plot_w, y_axis), fill="#222222", width=2)
    for tick in range(0, 6):
        val = tick / 5
        y = y_axis - val * plot_h
        draw.line((x0 - 6, y, x0 + plot_w, y), fill="#dddddd", width=1)
        draw.text((x0 - 12, y), f"{val:.1f}", fill="#333333", font=font_s, anchor="rm")
    xs = [x0 + plot_w * 0.15, x0 + plot_w * 0.50, x0 + plot_w * 0.85]
    budgets = [10, 20, 30]
    colors = {"KC1": "#1565C0", "JM1": "#2E7D32", "PC1": "#C62828"}
    for b, x in zip(budgets, xs):
        draw.text((x, y_axis + 28), f"{b}%LOC", fill="#111111", font=font, anchor="ma")
    for ds, color in colors.items():
        row = summary[(summary["Dataset"] == ds) & (summary["Model"] == "平衡Logistic")].iloc[0]
        vals = [float(row[f"Recall@{b}%LOC_mean"]) for b in budgets]
        pts = [(xs[i], y_axis - vals[i] * plot_h) for i in range(3)]
        draw.line(pts, fill=color, width=4)
        for x, y, val in zip(xs, [p[1] for p in pts], vals):
            draw.ellipse((x - 6, y - 6, x + 6, y + 6), fill=color)
            draw.text((x, y - 15), f"{val:.2f}", fill=color, font=font_s, anchor="mb")
    lx, ly = margin_l, height - 55
    for ds, color in colors.items():
        draw.line((lx, ly + 8, lx + 28, ly + 8), fill=color, width=4)
        draw.text((lx + 38, ly - 4), ds, fill="#111111", font=font_s)
        lx += 150
    img.save(out_path)


def make_ablation_chart(summary: pd.DataFrame, out_path: Path) -> None:
    from PIL import Image, ImageDraw

    width, height = 1300, 720
    margin_l, margin_r, margin_t, margin_b = 100, 50, 80, 115
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    font, font_b, font_s = load_fonts()
    draw.text((width // 2, 24), "不同静态度量组的AUC对比", fill="#111111", font=font_b, anchor="ma")
    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b
    x0, y0 = margin_l, margin_t
    y_axis = y0 + plot_h
    draw.line((x0, y0, x0, y_axis), fill="#222222", width=2)
    draw.line((x0, y_axis, x0 + plot_w, y_axis), fill="#222222", width=2)
    groups = ["规模度量", "控制复杂度", "Halstead度量", "全部度量"]
    colors = {"规模度量": "#78909C", "控制复杂度": "#F9A825", "Halstead度量": "#2E7D32", "全部度量": "#1565C0"}
    datasets = ["KC1", "JM1", "PC1"]
    for tick in range(5, 10):
        val = tick / 10
        y = y_axis - (val - 0.5) / 0.5 * plot_h
        draw.line((x0 - 6, y, x0 + plot_w, y), fill="#dddddd", width=1)
        draw.text((x0 - 12, y), f"{val:.1f}", fill="#333333", font=font_s, anchor="rm")
    group_w = plot_w / len(datasets)
    bar_w = group_w / 6
    for i, ds in enumerate(datasets):
        center = x0 + group_w * (i + 0.5)
        for j, group in enumerate(groups):
            row = summary[(summary["Dataset"] == ds) & (summary["FeatureGroup"] == group)].iloc[0]
            val = float(row["AUC_mean"])
            h = max(0, (val - 0.5) / 0.5 * plot_h)
            bx0 = center - 2 * bar_w + j * bar_w
            bx1 = bx0 + bar_w * 0.8
            draw.rectangle((bx0, y_axis - h, bx1, y_axis), fill=colors[group])
            draw.text(((bx0 + bx1) / 2, y_axis - h - 8), f"{val:.2f}", fill="#111111", font=font_s, anchor="mb")
        draw.text((center, y_axis + 30), ds, fill="#111111", font=font, anchor="ma")
    lx, ly = margin_l, height - 60
    for group in groups:
        draw.rectangle((lx, ly, lx + 24, ly + 16), fill=colors[group])
        draw.text((lx + 32, ly - 4), group, fill="#111111", font=font_s)
        lx += 220
    img.save(out_path)


def make_workflow_chart(out_path: Path) -> None:
    from PIL import Image, ImageDraw

    width, height = 1300, 420
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    font, font_b, _ = load_fonts()
    draw.text((width // 2, 26), "实验流程：从静态度量到测试优先级", fill="#111111", font=font_b, anchor="ma")
    boxes = [
        ("NASA/PROMISE\nARFF数据", 70),
        ("缺失值处理\n特征标准化", 320),
        ("模型训练\n排序打分", 570),
        ("测试预算\n10/20/30%LOC", 820),
        ("AUC/召回率\n精确率分析", 1070),
    ]
    y, w, h = 145, 170, 100
    for text, x in boxes:
        draw.rounded_rectangle((x, y, x + w, y + h), radius=12, outline="#1F4E79", width=3, fill="#EAF2F8")
        for k, line in enumerate(text.split("\n")):
            draw.text((x + w / 2, y + 30 + k * 34), line, fill="#111111", font=font, anchor="ma")
    for _, x in boxes[:-1]:
        draw.line((x + w + 18, y + h / 2, x + 232, y + h / 2), fill="#1F4E79", width=4)
        draw.polygon([(x + 232, y + h / 2), (x + 215, y + h / 2 - 10), (x + 215, y + h / 2 + 10)], fill="#1F4E79")
    img.save(out_path)


def main() -> None:
    out_dir = Path("output")
    out_dir.mkdir(exist_ok=True)
    dfs = {name: read_arff(info["path"]) for name, info in DATASETS.items()}
    dataset_meta = []
    model_rows: list[dict[str, float | str]] = []
    ablation_rows: list[dict[str, float | str]] = []
    for name, df in dfs.items():
        dataset_meta.append(
            {
                "Dataset": name,
                "OpenML_ID": DATASETS[name]["openml_id"],
                "Instances": int(len(df)),
                "Features": int(df.shape[1] - 1),
                "Defective": int(df["defects"].sum()),
                "DefectRate": float(df["defects"].mean()),
                "URL": DATASETS[name]["download"],
            }
        )
        model_rows.extend(run_model_comparison(df, name))
        ablation_rows.extend(run_ablation(df, name))

    raw = pd.DataFrame(model_rows)
    raw.to_csv(out_dir / "experiment_raw.csv", index=False, encoding="utf-8-sig")
    model_summary = summarize_model(model_rows)
    model_summary.to_csv(out_dir / "experiment_summary.csv", index=False, encoding="utf-8-sig")
    budget_summary = model_summary[model_summary["Model"] == "平衡Logistic"].copy()
    budget_summary.to_csv(out_dir / "budget_summary.csv", index=False, encoding="utf-8-sig")
    ablation_raw = pd.DataFrame(ablation_rows)
    ablation_raw.to_csv(out_dir / "ablation_raw.csv", index=False, encoding="utf-8-sig")
    ablation_summary = summarize_ablation(ablation_rows)
    ablation_summary.to_csv(out_dir / "ablation_summary.csv", index=False, encoding="utf-8-sig")
    cross = run_cross_project(dfs)
    cross.to_csv(out_dir / "cross_project.csv", index=False, encoding="utf-8-sig")
    corr = run_feature_correlation(dfs)
    corr.to_csv(out_dir / "feature_correlation.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(dataset_meta).to_csv(out_dir / "dataset_meta.csv", index=False, encoding="utf-8-sig")

    make_workflow_chart(out_dir / "workflow.png")
    make_grouped_bar(model_summary, out_dir / "recall20_bar.png")
    make_budget_line(model_summary, out_dir / "budget_line.png")
    make_ablation_chart(ablation_summary, out_dir / "ablation_auc.png")
    (out_dir / "experiment_meta.json").write_text(json.dumps(dataset_meta, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
