"""
Expose pipeline with exploratory pattern discovery.

Outputs:
- gap matrix
- variable catalog
- event-level analysis dataset
- validation summary
- discovered pattern diagnostics and centroids
- descriptive pattern tables
- models for shot probability and shot xG
- expose text blocks
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "cleaned"
OUT_DIR = INPUT_DIR / "expose_outputs"

INPUT_CROSSES_WITH_FORMATION = INPUT_DIR / "n_flanken_mit_formation_0201.csv"
INPUT_LINK_GLOB = "flanke_shot_links_*_openplay.csv"
INPUT_SEGMENTS = INPUT_DIR / "f_zeitsegmente_match_level.csv"
INPUT_FORMATIONS_MATCH_TEAM = INPUT_DIR / "n_formationen_0201_match_team.csv"
INPUT_QA_FORMATION = INPUT_DIR / "o_qa_flanken_ohne_formation_summary.csv"

K_CANDIDATES = list(range(3, 11))
EVAL_SEEDS = [11, 17, 23, 31]
MIN_CLUSTER_SHARE = 0.03
MIN_K_FOR_EXPOSE = 5
SAMPLE_FOR_K_SELECTION = 6000
SAMPLE_FOR_SILHOUETTE = 2000
KM_N_INIT_MAIN = 6
KM_N_INIT_STABILITY = 2
KM_N_INIT_GAP = 2
KM_N_INIT_FULL = 12
GAP_B = 4


def require_columns(df: pd.DataFrame, cols: Iterable[str], context: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{context}: missing columns: {missing}")


def parse_dfl_gametime_to_seconds(x) -> float:
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    parts = s.split(":")
    try:
        if len(parts) == 3:
            mm, ss, cs = parts
            return int(mm) * 60 + int(ss) + int(cs) / 100.0
        if len(parts) == 2:
            mm, ss = parts
            return int(mm) * 60 + float(ss)
        return float(s)
    except Exception:
        return np.nan


def normal_cdf(x: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.vectorize(math.erf)(x / math.sqrt(2.0)))


def p_values_from_stats(stat: np.ndarray) -> np.ndarray:
    return 2.0 * (1.0 - normal_cdf(np.abs(stat)))


def classify_formation_group(lineup: str | float | int | None) -> str:
    if lineup is None or (isinstance(lineup, float) and pd.isna(lineup)):
        return "unknown"
    s = str(lineup).strip()
    m = re.match(r"^([0-9]+)\s*-", s)
    if not m:
        return "unknown"
    first = m.group(1)
    if first == "3":
        return "3er"
    if first == "4":
        return "4er"
    if first == "5":
        return "5er"
    return "other"


def classify_formation_macro(formation_group: str | None) -> str:
    if formation_group in ["3er", "5er"]:
        return "wingback"
    if formation_group == "4er":
        return "back4"
    return "unknown"


def read_links_union() -> pd.DataFrame:
    files = sorted(INPUT_DIR.glob(INPUT_LINK_GLOB))
    if not files:
        raise FileNotFoundError(f"no link files found: {INPUT_DIR}/{INPUT_LINK_GLOB}")

    parts = []
    for fp in files:
        df = pd.read_csv(fp)
        df["SourceFile"] = fp.name
        parts.append(df)

    links = pd.concat(parts, ignore_index=True)
    require_columns(links, ["EventId", "ShotxG", "ShotTimeSec", "GameTime"], "links")

    links["EventId"] = links["EventId"].astype(str)
    links["ShotxG"] = pd.to_numeric(links["ShotxG"], errors="coerce")
    links["ShotTimeSec"] = pd.to_numeric(links["ShotTimeSec"], errors="coerce")
    links["CrossTimeSec"] = links["GameTime"].map(parse_dfl_gametime_to_seconds)
    links["DeltaSec"] = links["ShotTimeSec"] - links["CrossTimeSec"]

    links = links.sort_values(["EventId", "ShotTimeSec"], na_position="last")
    links = links.drop_duplicates(subset=["EventId"], keep="first")

    links["shot_within_6s"] = ((links["DeltaSec"] >= 0) & (links["DeltaSec"] <= 6)).astype(int)
    links["shot_within_8s"] = ((links["DeltaSec"] >= 0) & (links["DeltaSec"] <= 8)).astype(int)
    links["shot_within_10s"] = ((links["DeltaSec"] >= 0) & (links["DeltaSec"] <= 10)).astype(int)

    return links[
        [
            "EventId",
            "ShotxG",
            "ShotTimeSec",
            "DeltaSec",
            "shot_within_6s",
            "shot_within_8s",
            "shot_within_10s",
        ]
    ].copy()


def build_cross_event_base() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not INPUT_CROSSES_WITH_FORMATION.exists():
        raise FileNotFoundError(f"not found: {INPUT_CROSSES_WITH_FORMATION}")

    crosses = pd.read_csv(INPUT_CROSSES_WITH_FORMATION)
    require_columns(
        crosses,
        [
            "EventId",
            "MatchId",
            "TeamId",
            "GameTime",
            "X",
            "Y",
            "X_rec",
            "Y_rec",
            "MaxHeight",
            "NumAttInBox",
            "NumDefInBox",
            "MatchStateForTeam",
            "LineUp",
            "Season",
            "Competition",
        ],
        "crosses",
    )

    crosses["EventId"] = crosses["EventId"].astype(str)
    crosses = crosses.sort_values("EventId").drop_duplicates(subset=["EventId"], keep="first")

    for col in ["X", "Y", "X_rec", "Y_rec", "MaxHeight", "NumAttInBox", "NumDefInBox"]:
        crosses[col] = pd.to_numeric(crosses[col], errors="coerce")

    links = read_links_union()
    base = crosses.merge(links, on="EventId", how="left")

    base["shot_within_6s"] = base["shot_within_6s"].fillna(0).astype(int)
    base["shot_within_8s"] = base["shot_within_8s"].fillna(0).astype(int)
    base["shot_within_10s"] = base["shot_within_10s"].fillna(0).astype(int)
    base["ShotxG"] = pd.to_numeric(base["ShotxG"], errors="coerce")

    base["formation_group"] = base["LineUp"].map(classify_formation_group)
    base["formation_macro"] = base["formation_group"].map(classify_formation_macro)
    base["league"] = base["Competition"].astype(str)

    base["abs_x"] = base["X"].abs()
    base["abs_y"] = base["Y"].abs()
    base["abs_x_rec"] = base["X_rec"].abs()
    base["abs_y_rec"] = base["Y_rec"].abs()
    base["delta_abs_x"] = base["abs_x_rec"] - base["abs_x"]
    base["delta_abs_y"] = base["abs_y_rec"] - base["abs_y"]
    base["switch_side"] = ((base["Y"] * base["Y_rec"]) < 0).astype(int)
    base["box_balance"] = base["NumAttInBox"] - base["NumDefInBox"]

    crowded_threshold = float(base["NumDefInBox"].quantile(0.75))
    base["box_crowded"] = (base["NumDefInBox"] >= crowded_threshold).astype(int)

    return base, links


def standardize_features(df: pd.DataFrame, cols: list[str]) -> tuple[np.ndarray, pd.DataFrame]:
    work = df[cols].copy()
    for c in cols:
        work[c] = pd.to_numeric(work[c], errors="coerce")

    medians = work.median(numeric_only=True)
    work = work.fillna(medians)

    means = work.mean(axis=0)
    stds = work.std(axis=0, ddof=0).replace(0, 1.0)

    z = (work - means) / stds

    stats = pd.DataFrame(
        {
            "feature": cols,
            "mean": [means[c] for c in cols],
            "std": [stds[c] for c in cols],
            "median_impute": [medians[c] for c in cols],
        }
    )
    return z.to_numpy(dtype=float), stats


def squared_euclidean_matrix(X: np.ndarray, C: np.ndarray) -> np.ndarray:
    # ||x-c||^2 = ||x||^2 + ||c||^2 - 2x.c
    x2 = np.sum(X * X, axis=1, keepdims=True)
    c2 = np.sum(C * C, axis=1, keepdims=True).T
    cross = X @ C.T
    d2 = x2 + c2 - 2.0 * cross
    return np.clip(d2, 0.0, None)


def kmeans_plus_plus_init(X: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    n = X.shape[0]
    centers = []

    i0 = int(rng.integers(0, n))
    centers.append(X[i0])

    for _ in range(1, k):
        C = np.vstack(centers)
        d2 = squared_euclidean_matrix(X, C)
        min_d2 = d2.min(axis=1)
        total = float(min_d2.sum())
        if total <= 0:
            idx = int(rng.integers(0, n))
        else:
            probs = min_d2 / total
            idx = int(rng.choice(n, p=probs))
        centers.append(X[idx])

    return np.vstack(centers)


def kmeans_single(
    X: np.ndarray,
    k: int,
    seed: int,
    max_iter: int = 200,
) -> tuple[np.ndarray, np.ndarray, float]:
    rng = np.random.default_rng(seed)
    centers = kmeans_plus_plus_init(X, k, rng)

    labels = np.full(X.shape[0], -1, dtype=int)

    for _ in range(max_iter):
        d2 = squared_euclidean_matrix(X, centers)
        new_labels = np.argmin(d2, axis=1)

        if np.array_equal(new_labels, labels):
            labels = new_labels
            break
        labels = new_labels

        new_centers = np.zeros_like(centers)
        for j in range(k):
            mask = labels == j
            if not mask.any():
                # Re-seed empty cluster with a random data point.
                new_centers[j] = X[int(rng.integers(0, X.shape[0]))]
            else:
                new_centers[j] = X[mask].mean(axis=0)

        shift = np.max(np.abs(new_centers - centers))
        centers = new_centers
        if shift < 1e-7:
            break

    d2_final = squared_euclidean_matrix(X, centers)
    inertia = float(np.take_along_axis(d2_final, labels[:, None], axis=1).sum())
    return labels, centers, inertia


def kmeans(
    X: np.ndarray,
    k: int,
    n_init: int = 10,
    random_state: int = 123,
) -> tuple[np.ndarray, np.ndarray, float]:
    best_labels = None
    best_centers = None
    best_inertia = np.inf

    for i in range(n_init):
        seed = random_state + 1009 * (i + 1)
        labels, centers, inertia = kmeans_single(X=X, k=k, seed=seed)
        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels
            best_centers = centers

    assert best_labels is not None
    assert best_centers is not None
    return best_labels, best_centers, float(best_inertia)


def comb2(x: np.ndarray) -> np.ndarray:
    return x * (x - 1.0) / 2.0


def adjusted_rand_index(labels_a: np.ndarray, labels_b: np.ndarray) -> float:
    if labels_a.shape[0] != labels_b.shape[0]:
        raise ValueError("ARI label lengths differ")

    tab = pd.crosstab(pd.Series(labels_a), pd.Series(labels_b)).to_numpy(dtype=float)
    n = tab.sum()
    if n <= 1:
        return 0.0

    sum_ij = comb2(tab).sum()
    sum_i = comb2(tab.sum(axis=1)).sum()
    sum_j = comb2(tab.sum(axis=0)).sum()
    comb_n = comb2(np.array([n]))[0]

    expected = (sum_i * sum_j) / comb_n if comb_n > 0 else 0.0
    max_index = 0.5 * (sum_i + sum_j)
    denom = max_index - expected
    if abs(denom) < 1e-12:
        return 0.0
    return float((sum_ij - expected) / denom)


def centroid_separation_score(X: np.ndarray, labels: np.ndarray, centers: np.ndarray) -> float:
    d2 = squared_euclidean_matrix(X, centers)
    # distance to own centroid
    a = np.sqrt(np.take_along_axis(d2, labels[:, None], axis=1).flatten())

    # distance to nearest non-own centroid
    if centers.shape[0] <= 1:
        return 0.0

    d2_sorted = np.sort(d2, axis=1)
    b = np.sqrt(d2_sorted[:, 1])
    denom = np.maximum(a, b)
    denom = np.where(denom <= 1e-12, 1.0, denom)
    s = (b - a) / denom
    return float(np.mean(s))


def silhouette_score_from_labels(X: np.ndarray, labels: np.ndarray) -> float:
    n = X.shape[0]
    if n <= 2:
        return np.nan

    unique_labels = np.unique(labels)
    if unique_labels.shape[0] <= 1:
        return np.nan

    # Full pairwise Euclidean distance matrix on sampled data.
    x2 = np.sum(X * X, axis=1, keepdims=True)
    d2 = np.clip(x2 + x2.T - 2.0 * (X @ X.T), 0.0, None)
    d = np.sqrt(d2)

    cluster_indices = {lab: np.where(labels == lab)[0] for lab in unique_labels}
    sil = np.zeros(n, dtype=float)

    for i in range(n):
        own = labels[i]
        own_idx = cluster_indices[own]

        if own_idx.shape[0] <= 1:
            sil[i] = 0.0
            continue

        a = float((d[i, own_idx].sum() - 0.0) / (own_idx.shape[0] - 1))

        b_vals = []
        for other_lab in unique_labels:
            if other_lab == own:
                continue
            other_idx = cluster_indices[other_lab]
            if other_idx.shape[0] == 0:
                continue
            b_vals.append(float(d[i, other_idx].mean()))

        if not b_vals:
            sil[i] = 0.0
            continue

        b = float(min(b_vals))
        denom = max(a, b)
        if denom <= 1e-12:
            sil[i] = 0.0
        else:
            sil[i] = (b - a) / denom

    return float(np.mean(sil))


def elbow_k_by_line_distance(k_values: np.ndarray, inertia_values: np.ndarray) -> int:
    if k_values.shape[0] < 3:
        return int(k_values[0])

    x = k_values.astype(float)
    y = inertia_values.astype(float)

    x_min, x_max = float(x.min()), float(x.max())
    y_min, y_max = float(y.min()), float(y.max())

    x_n = (x - x_min) / (x_max - x_min + 1e-12)
    y_n = (y - y_min) / (y_max - y_min + 1e-12)

    x0, y0 = float(x_n[0]), float(y_n[0])
    x1, y1 = float(x_n[-1]), float(y_n[-1])
    denom = math.sqrt((y1 - y0) ** 2 + (x1 - x0) ** 2) + 1e-12

    d = np.abs((y1 - y0) * x_n - (x1 - x0) * y_n + x1 * y0 - y1 * x0) / denom
    d[0] = -np.inf
    d[-1] = -np.inf

    idx = int(np.argmax(d))
    return int(k_values[idx])


def gap_rule_k(k_values: np.ndarray, gap_values: np.ndarray, gap_std_values: np.ndarray) -> int:
    # Tibshirani rule: smallest k with Gap(k) >= Gap(k+1) - s(k+1)
    if k_values.shape[0] == 0:
        raise ValueError("gap_rule_k: empty k_values")
    if k_values.shape[0] == 1:
        return int(k_values[0])

    for i in range(k_values.shape[0] - 1):
        lhs = float(gap_values[i])
        rhs = float(gap_values[i + 1] - gap_std_values[i + 1])
        if lhs >= rhs:
            return int(k_values[i])

    return int(k_values[int(np.argmax(gap_values))])


def pick_constrained_k_by_metric(
    metrics: pd.DataFrame,
    metric_col: str,
    candidate_mask: pd.Series,
    prefer_high: bool = True,
) -> int:
    cand = metrics.loc[candidate_mask].copy()
    if cand.empty:
        cand = metrics.copy()

    order = False if prefer_high else True
    cand = cand.sort_values([metric_col, "k"], ascending=[order, True])
    return int(cand.iloc[0]["k"])


@dataclass
class PatternDiscoveryResult:
    k_selected: int
    labels_full: np.ndarray
    centers_full: np.ndarray
    metrics: pd.DataFrame
    decision: pd.DataFrame


def discover_patterns(X_full: np.ndarray) -> PatternDiscoveryResult:
    rng = np.random.default_rng(20260208)
    n = X_full.shape[0]
    sample_n = min(SAMPLE_FOR_K_SELECTION, n)
    sample_idx = rng.choice(n, size=sample_n, replace=False)
    X_eval = X_full[sample_idx]
    sil_n = min(SAMPLE_FOR_SILHOUETTE, sample_n)
    sil_idx = rng.choice(sample_n, size=sil_n, replace=False)

    mins = X_eval.min(axis=0)
    maxs = X_eval.max(axis=0)

    metric_rows = []
    models = {}

    for k in K_CANDIDATES:
        eval_labels = []
        for seed in EVAL_SEEDS:
            labels_e, _, _ = kmeans(X_eval, k=k, n_init=KM_N_INIT_STABILITY, random_state=seed + 1000 * k)
            eval_labels.append(labels_e)

        ari_vals = []
        for i in range(len(eval_labels)):
            for j in range(i + 1, len(eval_labels)):
                ari_vals.append(adjusted_rand_index(eval_labels[i], eval_labels[j]))
        stability = float(np.mean(ari_vals)) if ari_vals else np.nan

        labels_eval, centers_eval, inertia_eval = kmeans(X_eval, k=k, n_init=KM_N_INIT_MAIN, random_state=2000 + k)
        labels_sil = np.argmin(squared_euclidean_matrix(X_eval[sil_idx], centers_eval), axis=1)
        sil = silhouette_score_from_labels(X_eval[sil_idx], labels_sil)

        ref_logs = []
        for b in range(GAP_B):
            X_ref = rng.uniform(low=mins, high=maxs, size=X_eval.shape)
            _, _, inertia_ref = kmeans(X_ref, k=k, n_init=KM_N_INIT_GAP, random_state=5000 + 100 * k + b)
            ref_logs.append(float(np.log(max(inertia_ref, 1e-12))))
        ref_logs_arr = np.array(ref_logs, dtype=float)
        gap = float(ref_logs_arr.mean() - np.log(max(inertia_eval, 1e-12)))
        if GAP_B > 1:
            gap_std = float(np.sqrt(1.0 + 1.0 / GAP_B) * ref_logs_arr.std(ddof=1))
        else:
            gap_std = 0.0

        labels_full, centers_full, inertia_full = kmeans(X_full, k=k, n_init=KM_N_INIT_FULL, random_state=900 + k)

        sep = centroid_separation_score(X_full, labels_full, centers_full)
        cnt = pd.Series(labels_full).value_counts().sort_index()
        min_share = float((cnt.min() / len(labels_full)) if len(cnt) > 0 else np.nan)
        entropy = float(-(cnt / cnt.sum() * np.log((cnt / cnt.sum()) + 1e-12)).sum()) if cnt.sum() > 0 else np.nan

        metric_rows.append(
            {
                "k": k,
                "inertia_eval": inertia_eval,
                "inertia_full": inertia_full,
                "silhouette_mean": sil,
                "gap": gap,
                "gap_std": gap_std,
                "stability_ari_mean": stability,
                "centroid_separation": sep,
                "min_cluster_share": min_share,
                "cluster_entropy": entropy,
            }
        )
        models[k] = (labels_full, centers_full)

    metrics = pd.DataFrame(metric_rows).sort_values("k").reset_index(drop=True)
    metrics["passes_min_cluster_share"] = metrics["min_cluster_share"] >= MIN_CLUSTER_SHARE
    metrics["passes_min_k"] = metrics["k"] >= MIN_K_FOR_EXPOSE
    candidate_mask = metrics["passes_min_cluster_share"] & metrics["passes_min_k"]
    if not bool(candidate_mask.any()):
        candidate_mask = metrics["passes_min_k"]
    if not bool(candidate_mask.any()):
        candidate_mask = pd.Series(np.ones(len(metrics), dtype=bool), index=metrics.index)
    metrics["eligible_k_for_selection"] = candidate_mask

    all_k = metrics["k"].to_numpy(dtype=int)
    elbow_all = elbow_k_by_line_distance(all_k, metrics["inertia_eval"].to_numpy(dtype=float))
    sil_all = pick_constrained_k_by_metric(metrics, "silhouette_mean", pd.Series(np.ones(len(metrics), dtype=bool), index=metrics.index), True)
    gap_all = gap_rule_k(
        all_k,
        metrics["gap"].to_numpy(dtype=float),
        metrics["gap_std"].to_numpy(dtype=float),
    )

    cand = metrics.loc[candidate_mask].sort_values("k").reset_index(drop=True)
    cand_k = cand["k"].to_numpy(dtype=int)
    elbow_k = elbow_k_by_line_distance(cand_k, cand["inertia_eval"].to_numpy(dtype=float))
    sil_k = pick_constrained_k_by_metric(
        metrics=metrics,
        metric_col="silhouette_mean",
        candidate_mask=candidate_mask,
        prefer_high=True,
    )
    gap_k = gap_rule_k(
        cand_k,
        cand["gap"].to_numpy(dtype=float),
        cand["gap_std"].to_numpy(dtype=float),
    )

    vote_counts = {int(k): 0 for k in cand_k}
    for pick in [elbow_k, sil_k, gap_k]:
        vote_counts[int(pick)] = vote_counts.get(int(pick), 0) + 1

    max_votes = max(vote_counts.values())
    vote_winners = sorted([k for k, v in vote_counts.items() if v == max_votes])
    if len(vote_winners) == 1:
        k_selected = int(vote_winners[0])
    else:
        tie_df = metrics.loc[metrics["k"].isin(vote_winners)].sort_values(["silhouette_mean", "k"], ascending=[False, True])
        k_selected = int(tie_df.iloc[0]["k"])

    metrics["method_pick_elbow"] = metrics["k"] == elbow_k
    metrics["method_pick_silhouette"] = metrics["k"] == sil_k
    metrics["method_pick_gap"] = metrics["k"] == gap_k
    metrics["selected_k"] = metrics["k"] == k_selected

    votes_str = ",".join([f"{k}:{vote_counts[k]}" for k in sorted(vote_counts.keys())])
    decision = pd.DataFrame(
        [
            {
                "k_candidates_min": int(min(K_CANDIDATES)),
                "k_candidates_max": int(max(K_CANDIDATES)),
                "min_k_for_expose": int(MIN_K_FOR_EXPOSE),
                "min_cluster_share_threshold": float(MIN_CLUSTER_SHARE),
                "candidate_k_values": ",".join([str(int(x)) for x in cand_k]),
                "elbow_k_unconstrained": int(elbow_all),
                "silhouette_k_unconstrained": int(sil_all),
                "gap_k_unconstrained": int(gap_all),
                "elbow_k_constrained": int(elbow_k),
                "silhouette_k_constrained": int(sil_k),
                "gap_k_constrained": int(gap_k),
                "votes_by_k": votes_str,
                "k_selected": int(k_selected),
            }
        ]
    )

    labels_full, centers_full = models[k_selected]
    return PatternDiscoveryResult(
        k_selected=k_selected,
        labels_full=labels_full,
        centers_full=centers_full,
        metrics=metrics,
        decision=decision,
    )


def relabel_clusters_by_frequency(labels: np.ndarray) -> tuple[np.ndarray, dict[int, int]]:
    counts = pd.Series(labels).value_counts().sort_values(ascending=False)
    old_order = counts.index.to_list()
    mapping = {old: new for new, old in enumerate(old_order)}
    relabeled = np.array([mapping[x] for x in labels], dtype=int)
    return relabeled, mapping


def pattern_profile_table(base: pd.DataFrame, label_col: str, feature_cols: list[str]) -> pd.DataFrame:
    agg = (
        base.groupby(label_col, dropna=False)
        .agg(
            Crosses=("EventId", "nunique"),
            ShotRate=("shot_within_8s", "mean"),
            MeanShotxG=("ShotxG", "mean"),
            SharePct=("EventId", lambda s: s.nunique() / base["EventId"].nunique() * 100.0),
        )
        .reset_index()
    )

    means = base.groupby(label_col, dropna=False)[feature_cols].mean().reset_index()
    out = agg.merge(means, on=label_col, how="left")
    out = out.sort_values("Crosses", ascending=False)
    return out


def quantile_bins(series: pd.Series, q1: float, q2: float, labels: tuple[str, str, str]) -> pd.Series:
    a = float(series.quantile(q1))
    b = float(series.quantile(q2))

    def label_fn(v: float) -> str:
        if pd.isna(v):
            return "unknown"
        if v <= a:
            return labels[0]
        if v <= b:
            return labels[1]
        return labels[2]

    return series.map(label_fn)


def build_pattern_name_map(base: pd.DataFrame, cluster_col: str) -> pd.DataFrame:
    work = base.copy()

    # Global quantiles for human-readable descriptors.
    q_abs_x = work["abs_x"]
    q_abs_y = work["abs_y"]
    q_abs_x_rec = work["abs_x_rec"]
    q_height = work["MaxHeight"]

    profile = (
        work.groupby(cluster_col, dropna=False)
        .agg(
            n=("EventId", "nunique"),
            abs_x=("abs_x", "mean"),
            abs_y=("abs_y", "mean"),
            abs_x_rec=("abs_x_rec", "mean"),
            delta_abs_x=("delta_abs_x", "mean"),
            switch_side=("switch_side", "mean"),
            MaxHeight=("MaxHeight", "mean"),
        )
        .reset_index()
    )

    # Per-cluster descriptor categories.
    start_depth_labels = quantile_bins(profile["abs_x"], 0.33, 0.66, ("deep_start", "mid_start", "high_start"))
    start_width_labels = quantile_bins(profile["abs_y"], 0.33, 0.66, ("central", "halfspace", "wing"))
    target_depth_labels = quantile_bins(profile["abs_x_rec"], 0.33, 0.66, ("to_deep", "to_mid", "to_high"))
    height_labels = quantile_bins(profile["MaxHeight"], 0.33, 0.66, ("low", "medium", "high"))

    trajectory = []
    for v in profile["delta_abs_x"].to_list():
        if v <= -4:
            trajectory.append("backward")
        elif v >= 4:
            trajectory.append("forward")
        else:
            trajectory.append("level")

    side_move = ["switch" if v >= 0.55 else "same_side" for v in profile["switch_side"].to_list()]

    profile["pattern_name"] = [
        f"{a}|{b}|{c}|{d}|{e}|{f}"
        for a, b, c, d, e, f in zip(
            start_depth_labels,
            start_width_labels,
            target_depth_labels,
            trajectory,
            height_labels,
            side_move,
        )
    ]

    return profile[[cluster_col, "pattern_name"]].copy()


def build_pattern_name_suggestions(centers_orig: pd.DataFrame) -> pd.DataFrame:
    c = centers_orig.copy()

    q_start_low = float(c["abs_x"].quantile(0.33))
    q_start_high = float(c["abs_x"].quantile(0.66))
    q_width_low = float(c["abs_y"].quantile(0.33))
    q_width_high = float(c["abs_y"].quantile(0.66))
    q_traj_low = float(c["delta_abs_x"].quantile(0.33))
    q_traj_high = float(c["delta_abs_x"].quantile(0.66))
    q_height_low = float(c["MaxHeight"].quantile(0.33))
    q_height_high = float(c["MaxHeight"].quantile(0.66))

    rows = []
    for _, r in c.iterrows():
        start = "frueh" if r["abs_x"] <= q_start_low else ("grundlinie" if r["abs_x"] >= q_start_high else "zwischenraum")
        width = "breit" if r["abs_y"] >= q_width_high else ("zentral" if r["abs_y"] <= q_width_low else "halbraum")
        traj = "vorwaerts" if r["delta_abs_x"] >= q_traj_high else ("rueckraum" if r["delta_abs_x"] <= q_traj_low else "neutral")
        height = "hoch" if r["MaxHeight"] >= q_height_high else ("flach" if r["MaxHeight"] <= q_height_low else "mittel")
        side = "seitenwechsel" if r["switch_side"] >= 0.45 else "gleichseite"

        if start == "frueh" and width == "breit" and traj == "vorwaerts":
            label = "Fruehe Fluegelhereingabe"
        elif start == "grundlinie" and traj == "rueckraum":
            if height == "hoch":
                label = "Hoher Grundlinien-Rueckraumball"
            elif height == "flach":
                label = "Flacher Grundlinien-Rueckraumball"
            else:
                label = "Grundlinien-Rueckraumball"
        elif height == "hoch" and side == "seitenwechsel":
            label = "Hohe Seitenwechsel-Flanke"
        elif start == "grundlinie" and width == "breit" and traj == "neutral":
            label = "Breite Grundlinienflanke"
        elif start == "zwischenraum" and width == "zentral" and traj == "vorwaerts":
            label = "Zentrale Zwischenraum-Flanke"
        elif start == "zwischenraum" and width == "breit" and traj == "neutral":
            label = "Breite Zwischenraum-Flanke"
        elif width == "halbraum" and traj == "vorwaerts":
            label = "Halbraum-Hereingabe"
        elif traj == "rueckraum":
            label = "Rueckraumorientierte Flanke"
        else:
            label = f"{start.title()}-{width.title()}-{traj.title()}-Typ"

        rows.append(
            {
                "pattern_cluster": r["pattern_cluster"],
                "suggested_name": label,
                "start_zone": start,
                "width_zone": width,
                "trajectory": traj,
                "height_level": height,
                "side_mode": side,
            }
        )

    return pd.DataFrame(rows)


def build_pattern_difference_summary(
    base: pd.DataFrame,
    profiles: pd.DataFrame,
    naming: pd.DataFrame,
    feature_cols: list[str],
) -> pd.DataFrame:
    global_means = base[feature_cols].mean(numeric_only=True)
    global_stds = base[feature_cols].std(ddof=0, numeric_only=True).replace(0, 1.0)

    global_shot = float(base["shot_within_8s"].mean())
    global_xg = float(base["ShotxG"].mean())

    rows = []
    for _, row in profiles.iterrows():
        pc = row["pattern_cluster"]
        subset = base.loc[base["pattern_cluster"] == pc]
        if subset.empty:
            continue

        z_diffs = {}
        for f in feature_cols:
            m = float(subset[f].mean())
            z = (m - float(global_means[f])) / float(global_stds[f])
            z_diffs[f] = z

        top = sorted(z_diffs.items(), key=lambda kv: abs(kv[1]), reverse=True)[:3]
        top_feats = [f"{k}:{v:+.2f}" for k, v in top]
        while len(top_feats) < 3:
            top_feats.append("")

        rows.append(
            {
                "pattern_cluster": pc,
                "Crosses": int(subset["EventId"].nunique()),
                "ShotRate": float(subset["shot_within_8s"].mean()),
                "ShotRateDiffVsGlobal": float(subset["shot_within_8s"].mean() - global_shot),
                "MeanShotxG": float(subset["ShotxG"].mean()),
                "MeanShotxGDiffVsGlobal": float(subset["ShotxG"].mean() - global_xg),
                "TopFeatureDiff1": top_feats[0],
                "TopFeatureDiff2": top_feats[1],
                "TopFeatureDiff3": top_feats[2],
            }
        )

    out = pd.DataFrame(rows)
    out = out.merge(naming, on="pattern_cluster", how="left")
    out = out.sort_values("Crosses", ascending=False)
    return out


def write_gap_matrix() -> pd.DataFrame:
    rows = [
        {
            "GapID": "G1",
            "LiteraturLuecke": "Flankenmuster werden oft vordefiniert statt datengetrieben ermittelt.",
            "TestbarkeitMitDatensatz": "voll",
            "VerfuegbareVariablen": "X, Y, X_rec, Y_rec, MaxHeight, NumAttInBox, NumDefInBox",
            "Hinweis": "Explorative Clusteranalyse auf Event-Level-Flankenmerkmalen.",
        },
        {
            "GapID": "G2",
            "LiteraturLuecke": "Muster-Effektivitaet wird selten simultan auf Abschlusswahrscheinlichkeit und xG geprueft.",
            "TestbarkeitMitDatensatz": "voll",
            "VerfuegbareVariablen": "shot_within_8s, ShotxG",
            "Hinweis": "Zwei Endpunkte (binar + kontinuierlich).",
        },
        {
            "GapID": "G3",
            "LiteraturLuecke": "Bedingte Effektivitaet (Spielzustand, Strafraumdichte) ist unzureichend modelliert.",
            "TestbarkeitMitDatensatz": "voll",
            "VerfuegbareVariablen": "MatchStateForTeam, NumDefInBox, box_crowded",
            "Hinweis": "Interaktionen Muster x Bedingungen.",
        },
        {
            "GapID": "G4",
            "LiteraturLuecke": "Formationseinfluss auf Muster-Effekt wird selten als Moderator modelliert.",
            "TestbarkeitMitDatensatz": "voll",
            "VerfuegbareVariablen": "LineUp -> formation_macro (wingback vs back4)",
            "Hinweis": "Interaktionen Muster x Formation.",
        },
        {
            "GapID": "G5",
            "LiteraturLuecke": "Stichproben oft klein/turnierzentriert statt Ligabetrieb.",
            "TestbarkeitMitDatensatz": "voll",
            "VerfuegbareVariablen": "34.604 Flanken, 1.836 Matches, 2 Ligen, 3 Saisons",
            "Hinweis": "Breiter Ligadaten-Scope.",
        },
        {
            "GapID": "G6",
            "LiteraturLuecke": "Dynamische Ingame-Formationswechsel.",
            "TestbarkeitMitDatensatz": "teilweise",
            "VerfuegbareVariablen": "statische Match-Formation",
            "Hinweis": "Keine zeitaufgeloesten Formation-Switch-Events.",
        },
    ]
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "01_gap_matrix.csv", index=False)
    return df


def write_variable_catalog() -> pd.DataFrame:
    rows = [
        {"Variable": "EventId", "Rolle": "ID", "Quelle": "n_flanken_mit_formation_0201.csv", "Transformation": "string", "Regel": "Event-Key"},
        {"Variable": "MatchId", "Rolle": "Cluster", "Quelle": "n_flanken_mit_formation_0201.csv", "Transformation": "string", "Regel": "Pflicht"},
        {"Variable": "TeamId", "Rolle": "Cluster", "Quelle": "n_flanken_mit_formation_0201.csv", "Transformation": "string", "Regel": "Pflicht"},
        {"Variable": "Season", "Rolle": "Kontrolle", "Quelle": "n_flanken_mit_formation_0201.csv", "Transformation": "kategorial", "Regel": "Pflicht"},
        {"Variable": "Competition", "Rolle": "Kontrolle", "Quelle": "n_flanken_mit_formation_0201.csv", "Transformation": "kategorial", "Regel": "Pflicht"},
        {"Variable": "X,Y,X_rec,Y_rec,MaxHeight", "Rolle": "Musterentdeckung", "Quelle": "n_flanken_mit_formation_0201.csv", "Transformation": "numeric", "Regel": "in Clusterfeatures"},
        {"Variable": "NumAttInBox,NumDefInBox", "Rolle": "Muster/Bedingung", "Quelle": "n_flanken_mit_formation_0201.csv", "Transformation": "numeric", "Regel": "in Clusterfeatures + Modelle"},
        {"Variable": "box_balance", "Rolle": "Bedingung", "Quelle": "abgeleitet", "Transformation": "NumAttInBox-NumDefInBox", "Regel": "Modellpradiktor"},
        {"Variable": "box_crowded", "Rolle": "Bedingung", "Quelle": "abgeleitet", "Transformation": "NumDefInBox >= P75", "Regel": "Interaktion"},
        {"Variable": "MatchStateForTeam", "Rolle": "Bedingung", "Quelle": "n_flanken_mit_formation_0201.csv", "Transformation": "winning/drawing/losing", "Regel": "Interaktion"},
        {"Variable": "LineUp", "Rolle": "Moderator-Basis", "Quelle": "n_flanken_mit_formation_0201.csv", "Transformation": "string", "Regel": "Pflicht"},
        {"Variable": "formation_group", "Rolle": "Aux", "Quelle": "abgeleitet", "Transformation": "3er/4er/5er", "Regel": "klassische Ableitung"},
        {"Variable": "formation_macro", "Rolle": "Moderator", "Quelle": "abgeleitet", "Transformation": "wingback/back4", "Regel": "Interaktion"},
        {"Variable": "pattern_cluster", "Rolle": "UV-Kern", "Quelle": "datengetrieben", "Transformation": "k-means Cluster", "Regel": "zentrale Musterkategorie"},
        {"Variable": "shot_within_8s", "Rolle": "AV1", "Quelle": "Join auf flanke_shot_links", "Transformation": "0/1", "Regel": "primaerer Endpunkt"},
        {"Variable": "ShotxG", "Rolle": "AV2", "Quelle": "Join auf flanke_shot_links", "Transformation": "numeric", "Regel": "nur bei shot_within_8s=1"},
    ]
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "02_variablenkatalog.csv", index=False)
    return df


def write_validation(
    base: pd.DataFrame,
    links: pd.DataFrame,
    k_selected: int,
    discovery_metrics: pd.DataFrame,
    discovery_decision: pd.DataFrame,
) -> pd.DataFrame:
    total = int(base["EventId"].nunique())
    matched = int(base.loc[base["shot_within_8s"] == 1, "EventId"].nunique())

    valid_form = int(base["formation_group"].isin(["3er", "4er", "5er"]).sum())
    share_form = valid_form / total if total > 0 else np.nan
    valid_macro = int(base["formation_macro"].isin(["wingback", "back4"]).sum())
    share_macro = valid_macro / total if total > 0 else np.nan

    y6 = float(base["shot_within_6s"].mean())
    y8 = float(base["shot_within_8s"].mean())
    y10 = float(base["shot_within_10s"].mean())

    qa_df = pd.read_csv(INPUT_QA_FORMATION)
    qa_overall = qa_df.loc[qa_df["Scope"] == "overall"].head(1)
    qa_missing_share = float(qa_overall["ShareWithoutLineUpPct"].iloc[0]) if not qa_overall.empty else np.nan

    selected_row = discovery_metrics.loc[discovery_metrics["k"] == k_selected].head(1)
    selected_stability = float(selected_row["stability_ari_mean"].iloc[0]) if not selected_row.empty else np.nan
    selected_sep = float(selected_row["silhouette_mean"].iloc[0]) if not selected_row.empty else np.nan
    selected_gap = float(selected_row["gap"].iloc[0]) if not selected_row.empty else np.nan
    selected_gap_std = float(selected_row["gap_std"].iloc[0]) if not selected_row.empty else np.nan
    selected_min_share = float(selected_row["min_cluster_share"].iloc[0]) if not selected_row.empty else np.nan

    drow = discovery_decision.head(1)
    elbow_k = int(drow["elbow_k_constrained"].iloc[0]) if not drow.empty else np.nan
    silhouette_k = int(drow["silhouette_k_constrained"].iloc[0]) if not drow.empty else np.nan
    gap_k = int(drow["gap_k_constrained"].iloc[0]) if not drow.empty else np.nan
    votes = str(drow["votes_by_k"].iloc[0]) if not drow.empty else ""

    rows = [
        {"Check": "join_integritaet_eventid", "Wert": round(matched / total, 6), "Details": f"{matched}/{total} events with shot<=8s"},
        {"Check": "formationsmapping_3_4_5", "Wert": round(share_form, 6), "Details": f"valid_3_4_5={valid_form}"},
        {"Check": "formationsmapping_wingback_back4", "Wert": round(share_macro, 6), "Details": f"valid_macro={valid_macro}"},
        {"Check": "qa_share_without_lineup_pct", "Wert": round(qa_missing_share, 6), "Details": "from o_qa_flanken_ohne_formation_summary.csv"},
        {"Check": "outcome_konsistenz_shot_within_8s", "Wert": round(y8, 6), "Details": "target ~0.20"},
        {"Check": "sensitivitaet_6s", "Wert": round(y6, 6), "Details": "same event-link basis"},
        {"Check": "sensitivitaet_8s", "Wert": round(y8, 6), "Details": "primary endpoint"},
        {"Check": "sensitivitaet_10s", "Wert": round(y10, 6), "Details": "same event-link basis"},
        {"Check": "pattern_discovery_k_selected", "Wert": float(k_selected), "Details": "chosen by majority vote of elbow/silhouette/gap with constraints"},
        {"Check": "pattern_discovery_k_elbow", "Wert": float(elbow_k), "Details": "line-distance elbow on inertia_eval"},
        {"Check": "pattern_discovery_k_silhouette", "Wert": float(silhouette_k), "Details": "max silhouette on constrained candidates"},
        {"Check": "pattern_discovery_k_gap", "Wert": float(gap_k), "Details": "Tibshirani gap rule on constrained candidates"},
        {"Check": "pattern_discovery_votes", "Wert": np.nan, "Details": votes},
        {"Check": "pattern_discovery_stability_ari", "Wert": round(selected_stability, 6), "Details": "mean pairwise ARI across seeds"},
        {"Check": "pattern_discovery_silhouette", "Wert": round(selected_sep, 6), "Details": "sample silhouette mean"},
        {"Check": "pattern_discovery_gap", "Wert": round(selected_gap, 6), "Details": "gap statistic"},
        {"Check": "pattern_discovery_gap_std", "Wert": round(selected_gap_std, 6), "Details": "gap standard error term"},
        {"Check": "pattern_discovery_min_cluster_share", "Wert": round(selected_min_share, 6), "Details": "small-cluster guard"},
    ]

    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "04_validierung_summary.csv", index=False)

    links_stats = pd.DataFrame(
        [
            {
                "LinksRowsUniqueEventId": int(links["EventId"].nunique()),
                "MeanDeltaSec": float(pd.to_numeric(links["DeltaSec"], errors="coerce").mean()),
            }
        ]
    )
    links_stats.to_csv(OUT_DIR / "04_validierung_links_stats.csv", index=False)
    return out


def write_descriptive(base: pd.DataFrame) -> None:
    total = base["EventId"].nunique()

    overall = (
        base.groupby("pattern_cluster", dropna=False)
        .agg(
            Crosses=("EventId", "nunique"),
            ShotRate=("shot_within_8s", "mean"),
            MeanShotxG=("ShotxG", "mean"),
        )
        .reset_index()
        .sort_values("Crosses", ascending=False)
    )
    overall["SharePct"] = overall["Crosses"] / total * 100.0 if total > 0 else np.nan
    overall.to_csv(OUT_DIR / "05_deskriptiv_muster_gesamt.csv", index=False)

    by_league_season = (
        base.groupby(["Season", "league", "pattern_cluster"], dropna=False)
        .agg(
            Crosses=("EventId", "nunique"),
            ShotRate=("shot_within_8s", "mean"),
            MeanShotxG=("ShotxG", "mean"),
        )
        .reset_index()
        .sort_values(["Season", "league", "Crosses"], ascending=[True, True, False])
    )
    by_league_season.to_csv(OUT_DIR / "05_deskriptiv_muster_liga_saison.csv", index=False)

    by_formation = (
        base.loc[base["formation_macro"].isin(["wingback", "back4"])]
        .groupby(["formation_macro", "pattern_cluster"], dropna=False)
        .agg(
            Crosses=("EventId", "nunique"),
            ShotRate=("shot_within_8s", "mean"),
            MeanShotxG=("ShotxG", "mean"),
        )
        .reset_index()
        .sort_values(["formation_macro", "Crosses"], ascending=[True, False])
    )
    by_formation.to_csv(OUT_DIR / "05_deskriptiv_muster_formation.csv", index=False)

    pattern_by_condition = (
        base.groupby(["pattern_cluster", "MatchStateForTeam", "box_crowded"], dropna=False)
        .agg(
            Crosses=("EventId", "nunique"),
            ShotRate=("shot_within_8s", "mean"),
            MeanShotxG=("ShotxG", "mean"),
            MeanBoxBalance=("box_balance", "mean"),
            MeanNumDefInBox=("NumDefInBox", "mean"),
            MeanNumAttInBox=("NumAttInBox", "mean"),
        )
        .reset_index()
        .sort_values(["pattern_cluster", "MatchStateForTeam", "box_crowded"])
    )
    pattern_by_condition.to_csv(OUT_DIR / "05_pattern_effektivitaet_bedingungen.csv", index=False)

    pattern_usage_state = (
        base.groupby(["MatchStateForTeam", "pattern_cluster"], dropna=False)["EventId"]
        .nunique()
        .reset_index(name="Crosses")
    )
    state_totals = pattern_usage_state.groupby("MatchStateForTeam", dropna=False)["Crosses"].sum().rename("StateTotal")
    pattern_usage_state = pattern_usage_state.merge(state_totals, on="MatchStateForTeam", how="left")
    pattern_usage_state["ShareWithinStatePct"] = pattern_usage_state["Crosses"] / pattern_usage_state["StateTotal"] * 100.0
    pattern_usage_state = pattern_usage_state.sort_values(["MatchStateForTeam", "Crosses"], ascending=[True, False])
    pattern_usage_state.to_csv(OUT_DIR / "05_pattern_nutzung_nach_state.csv", index=False)

    by_state = (
        base.groupby("MatchStateForTeam", dropna=False)
        .agg(
            Crosses=("EventId", "nunique"),
            ShotRate=("shot_within_8s", "mean"),
            MeanShotxG=("ShotxG", "mean"),
        )
        .reset_index()
        .sort_values("Crosses", ascending=False)
    )
    by_state.to_csv(OUT_DIR / "05_deskriptiv_state.csv", index=False)

    # Existing H1 rate table on segment basis.
    seg = pd.read_csv(INPUT_SEGMENTS)
    require_columns(seg, ["match_id", "team_id", "minutes", "crosses_openplay", "match_state_for_team"], "segments")

    fm = pd.read_csv(INPUT_FORMATIONS_MATCH_TEAM)
    require_columns(fm, ["MatchId", "TeamId", "LineUp"], "formations match-team")
    fm["match_id"] = fm["MatchId"].astype(str)
    fm["team_id"] = fm["TeamId"].astype(str)
    fm["formation_group"] = fm["LineUp"].map(classify_formation_group)
    fm["formation_macro"] = fm["formation_group"].map(classify_formation_macro)
    fm = fm[["match_id", "team_id", "formation_group", "formation_macro"]].drop_duplicates()

    seg["match_id"] = seg["match_id"].astype(str)
    seg["team_id"] = seg["team_id"].astype(str)
    segx = seg.merge(fm, on=["match_id", "team_id"], how="left")

    h1 = (
        segx.groupby("match_state_for_team", dropna=False)
        .agg(Minutes=("minutes", "sum"), Crosses=("crosses_openplay", "sum"))
        .reset_index()
    )
    h1["CrossesPer90"] = h1["Crosses"] / h1["Minutes"] * 90.0
    h1.to_csv(OUT_DIR / "05_h1_state_crosses_per90.csv", index=False)

    h1_form = (
        segx.loc[segx["formation_macro"].isin(["wingback", "back4"])]
        .groupby(["match_state_for_team", "formation_macro"], dropna=False)
        .agg(Minutes=("minutes", "sum"), Crosses=("crosses_openplay", "sum"))
        .reset_index()
    )
    h1_form["CrossesPer90"] = h1_form["Crosses"] / h1_form["Minutes"] * 90.0
    h1_form.to_csv(OUT_DIR / "05_h1_state_crosses_per90_by_formation.csv", index=False)


def build_design_matrix(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str], str]:
    m = df.copy()

    x_cols: list[np.ndarray] = []
    names: list[str] = []

    x_cols.append(np.ones(len(m)))
    names.append("Intercept")

    # Numeric controls.
    for c in ["box_balance", "X", "Y", "box_crowded"]:
        m[c] = pd.to_numeric(m[c], errors="coerce")
        x_cols.append(m[c].to_numpy())
        names.append(c)

    # Match state dummies (baseline: drawing)
    state_winning = (m["MatchStateForTeam"] == "winning").astype(int).to_numpy()
    state_losing = (m["MatchStateForTeam"] == "losing").astype(int).to_numpy()
    x_cols.append(state_winning)
    names.append("state_winning")
    x_cols.append(state_losing)
    names.append("state_losing")

    # Formation macro dummy (baseline: back4)
    f_wb = (m["formation_macro"] == "wingback").astype(int).to_numpy()
    x_cols.append(f_wb)
    names.append("form_wingback")

    # Pattern dummies with most frequent cluster as baseline.
    pattern_counts = m["pattern_cluster"].value_counts().sort_values(ascending=False)
    baseline_pattern = str(pattern_counts.index[0])
    pattern_levels = [str(x) for x in pattern_counts.index.to_list() if str(x) != baseline_pattern]

    pattern_arrays = {}
    for p in pattern_levels:
        arr = (m["pattern_cluster"].astype(str) == p).astype(int).to_numpy()
        pattern_arrays[p] = arr
        x_cols.append(arr)
        names.append(f"pattern_{p}")

    # Conditions and moderator interactions.
    crowded = m["box_crowded"].to_numpy()
    for p in pattern_levels:
        arr = pattern_arrays[p]
        x_cols.append(arr * state_losing)
        names.append(f"pattern_{p}_x_state_losing")
        x_cols.append(arr * crowded)
        names.append(f"pattern_{p}_x_box_crowded")
        x_cols.append(arr * f_wb)
        names.append(f"pattern_{p}_x_form_wingback")

    # League dummy (baseline: Bundesliga)
    x_cols.append((m["league"] == "2. Bundesliga").astype(int).to_numpy())
    names.append("league_2bl")

    # Season dummies (baseline: earliest season string)
    season_values = sorted(m["Season"].dropna().astype(str).unique().tolist())
    if season_values:
        for season in season_values[1:]:
            arr = (m["Season"].astype(str) == season).astype(int).to_numpy()
            x_cols.append(arr)
            names.append(f"season_{season.replace('/', '_')}")

    X = np.column_stack(x_cols).astype(float)
    return X, m.index.to_numpy(), names, baseline_pattern


def fit_logit_irls(
    X: np.ndarray,
    y: np.ndarray,
    max_iter: int = 200,
    tol: float = 1e-7,
    ridge: float = 1e-7,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    _, p = X.shape
    beta = np.zeros(p, dtype=float)

    for _ in range(max_iter):
        z = np.clip(X @ beta, -30, 30)
        pr = 1.0 / (1.0 + np.exp(-z))
        w = np.clip(pr * (1.0 - pr), 1e-8, None)

        grad = X.T @ (y - pr)
        h = X.T @ (X * w[:, None]) + ridge * np.eye(p)

        step = np.linalg.solve(h, grad)
        beta_new = beta + step

        if np.max(np.abs(step)) < tol:
            beta = beta_new
            break
        beta = beta_new

    z = np.clip(X @ beta, -30, 30)
    pr = 1.0 / (1.0 + np.exp(-z))
    w = np.clip(pr * (1.0 - pr), 1e-8, None)
    h_final = X.T @ (X * w[:, None]) + ridge * np.eye(p)

    cov = np.linalg.pinv(h_final)
    se = np.sqrt(np.clip(np.diag(cov), 1e-12, None))
    stat = beta / se

    ll = float(np.sum(y * np.log(np.clip(pr, 1e-12, 1.0)) + (1.0 - y) * np.log(np.clip(1.0 - pr, 1e-12, 1.0))))
    p0 = float(np.clip(np.mean(y), 1e-12, 1.0 - 1e-12))
    ll0 = float(np.sum(y * np.log(p0) + (1.0 - y) * np.log(1.0 - p0)))
    mcfadden = 1.0 - ll / ll0 if ll0 != 0 else np.nan

    return beta, se, stat, ll, mcfadden


def fit_ols(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    n, p = X.shape
    xtx_inv = np.linalg.pinv(X.T @ X)
    beta = xtx_inv @ (X.T @ y)
    resid = y - X @ beta

    dof = max(n - p, 1)
    sigma2 = float((resid.T @ resid) / dof)
    cov = sigma2 * xtx_inv
    se = np.sqrt(np.clip(np.diag(cov), 1e-12, None))
    stat = beta / se

    ss_tot = float(np.sum((y - y.mean()) ** 2))
    ss_res = float(np.sum(resid ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return beta, se, stat, r2


def run_models(base: pd.DataFrame) -> None:
    model_df = base.loc[
        base["formation_macro"].isin(["wingback", "back4"])
        & base["MatchStateForTeam"].isin(["winning", "drawing", "losing"])
        & base["pattern_cluster"].notna()
    ].copy()

    model_df = model_df.dropna(
        subset=[
            "shot_within_8s",
            "box_balance",
            "box_crowded",
            "X",
            "Y",
            "formation_macro",
            "pattern_cluster",
            "MatchStateForTeam",
        ]
    )

    X, idx, terms, pattern_baseline = build_design_matrix(model_df)
    y_logit = model_df.loc[idx, "shot_within_8s"].astype(float).to_numpy()

    beta, se, stat, ll, mcfadden = fit_logit_irls(X=X, y=y_logit)
    pval = p_values_from_stats(stat)

    coef_a = pd.DataFrame(
        {
            "term": terms,
            "beta": beta,
            "se": se,
            "z": stat,
            "p_value": pval,
            "odds_ratio": np.exp(beta),
        }
    )
    coef_a.to_csv(OUT_DIR / "06_model_a_logit_coefficients.csv", index=False)

    pred = 1.0 / (1.0 + np.exp(-np.clip(X @ beta, -30, 30)))
    acc = float(((pred >= 0.5).astype(int) == y_logit).mean())

    metrics_a = pd.DataFrame(
        [
            {
                "model": "A_logit_shot_within_8s",
                "n_obs": int(len(y_logit)),
                "share_positive": float(y_logit.mean()),
                "log_likelihood": ll,
                "mcfadden_r2": mcfadden,
                "accuracy_threshold_0_5": acc,
                "pattern_baseline": pattern_baseline,
            }
        ]
    )
    metrics_a.to_csv(OUT_DIR / "06_model_a_logit_metrics.csv", index=False)

    b_df = model_df.loc[model_df["shot_within_8s"] == 1].copy()
    b_df = b_df.dropna(subset=["ShotxG"])
    if len(b_df) > 0:
        Xb, idxb, terms_b, pattern_baseline_b = build_design_matrix(b_df)
        yb = b_df.loc[idxb, "ShotxG"].astype(float).to_numpy()

        beta_b, se_b, stat_b, r2_b = fit_ols(X=Xb, y=yb)
        pval_b = p_values_from_stats(stat_b)

        coef_b = pd.DataFrame(
            {
                "term": terms_b,
                "beta": beta_b,
                "se": se_b,
                "t": stat_b,
                "p_value": pval_b,
            }
        )
        coef_b.to_csv(OUT_DIR / "06_model_b_ols_coefficients.csv", index=False)

        metrics_b = pd.DataFrame(
            [
                {
                    "model": "B_ols_shotxg_given_shot",
                    "n_obs": int(len(yb)),
                    "mean_shotxg": float(np.mean(yb)),
                    "r2": r2_b,
                    "pattern_baseline": pattern_baseline_b,
                }
            ]
        )
        metrics_b.to_csv(OUT_DIR / "06_model_b_ols_metrics.csv", index=False)
    else:
        pd.DataFrame(
            [
                {
                    "model": "B_ols_shotxg_given_shot",
                    "n_obs": 0,
                    "mean_shotxg": np.nan,
                    "r2": np.nan,
                    "pattern_baseline": "",
                }
            ]
        ).to_csv(OUT_DIR / "06_model_b_ols_metrics.csv", index=False)
        pd.DataFrame(columns=["term", "beta", "se", "t", "p_value"]).to_csv(
            OUT_DIR / "06_model_b_ols_coefficients.csv", index=False
        )


def write_expose_text(base: pd.DataFrame, k_selected: int) -> None:
    total_crosses = int(base["EventId"].nunique())
    match_count = int(base["MatchId"].nunique())
    share_shot = float(base["shot_within_8s"].mean())
    seasons = sorted(base["Season"].dropna().astype(str).unique().tolist())
    leagues = sorted(base["league"].dropna().astype(str).unique().tolist())

    pattern_counts = (
        base.groupby("pattern_cluster", dropna=False)["EventId"]
        .nunique()
        .sort_values(ascending=False)
        .to_dict()
    )
    pattern_line = ", ".join([f"{k}={v}" for k, v in pattern_counts.items()])
    labels = (
        base[["pattern_cluster", "suggested_name"]]
        .dropna()
        .drop_duplicates()
        .sort_values("pattern_cluster")
    )
    label_line = ", ".join([f"{r.pattern_cluster}:{r.suggested_name}" for r in labels.itertuples(index=False)])

    txt = f"""# Expose text blocks (auto-generated)

## Forschungsfrage
Welche datengetrieben ermittelten Open-Play-Flankenmuster treten in Bundesliga und 2. Bundesliga in den Saisons {", ".join(seasons)} am haeufigsten auf, welche Muster sind effektiver in Bezug auf `shot_within_8s` und `ShotxG`, und unter welchen Bedingungen (Spielzustand, Box-Crowding) variieren diese Effekte?

## Sekundaere Moderatorfrage
Unterscheiden sich diese Muster-Effekte systematisch zwischen Wingback-Systemen (`3er/5er`) und Viererketten-Systemen (`4er`)?

## Hypothesen
H1: Die Nutzung der datengetriebenen Flankenmuster unterscheidet sich zwischen Spielzustaenden (`winning`, `drawing`, `losing`).
H2: Die Effektivitaet der Muster ist bedingungsabhaengig von `MatchStateForTeam` und `box_crowded`.
H3: Die Muster-Effekte auf `shot_within_8s` und `ShotxG` werden durch `formation_macro` (`wingback` vs `back4`) moderiert.

## Datengrundlage
- Event-Level Flanken: {total_crosses}
- Matches: {match_count}
- Wettbewerbe: {", ".join(leagues)}
- Anteil Flanken mit Schuss <=8s: {share_shot:.4f}
- Entdeckte Muster (k-means, K={k_selected}): {pattern_line}
- Namensvorschlaege: {label_line}

## Methodik
1. Explorative Musterermittlung mit k-means auf Flankenmerkmalen (`X`, `Y`, `X_rec`, `Y_rec`, `MaxHeight`, `NumAttInBox`, `NumDefInBox`, abgeleitete Raum-/Trajektorienmerkmale).
2. K-Auswahl methodisch ueber drei Standardkriterien: Elbow (Inertia), Silhouette und Gap Statistic; finale K-Entscheidung per Mehrheitsregel unter Nebenbedingungen (`K>=5`, Mindest-Clusteranteil).
3. Deskriptive Auswertung der Musterhaeufigkeit und Muster-Effektivitaet nach Liga, Saison, Spielzustand, Box-Crowding.
4. Modell A: logistische Regression fuer `P(shot_within_8s=1)` mit Interaktionen Muster x Bedingungen und Muster x Formation-Makrogruppe.
5. Modell B: lineares Modell fuer `ShotxG | shot_within_8s=1` mit derselben Interaktionslogik.

## Operationalisierung
- UV-Kern: `pattern_cluster` (datengetrieben, keine vorab fixe Musterliste)
- Bedingungen: `MatchStateForTeam`, `box_crowded`, `box_balance`
- Moderator: `formation_macro` (`wingback` vs `back4`)
- AV1: `shot_within_8s`
- AV2: `ShotxG` (konditional auf Schuss)

## Limitationen
- Cluster sind daten- und feature-abhaengig; externe Replikation in anderen Ligen/Saisons erforderlich.
- Formationen liegen statisch pro Match-Team vor (keine in-game Wechselzeitachse).
- Zeitfenster-Sensitivitaet (6/8/10s) basiert auf Event-Linking-Logik.
"""

    (OUT_DIR / "07_expose_textbausteine.md").write_text(txt, encoding="utf-8")


def write_readme() -> None:
    txt = """# Expose Output Files

01_gap_matrix.csv
02_variablenkatalog.csv
03_cross_event_base.csv
03b_pattern_selection_metrics.csv
03e_pattern_cluster_profiles.csv
03f_pattern_namensvorschlaege.csv
03g_pattern_unterschiede_summary.csv
03h_pattern_k_entscheidung.csv
04_validierung_summary.csv
04_validierung_links_stats.csv
05_deskriptiv_muster_gesamt.csv
05_deskriptiv_muster_liga_saison.csv
05_deskriptiv_muster_formation.csv
05_pattern_effektivitaet_bedingungen.csv
05_pattern_nutzung_nach_state.csv
05_deskriptiv_state.csv
05_h1_state_crosses_per90.csv
05_h1_state_crosses_per90_by_formation.csv
06_model_a_logit_coefficients.csv
06_model_a_logit_metrics.csv
06_model_b_ols_coefficients.csv
06_model_b_ols_metrics.csv
07_expose_textbausteine.md
"""
    (OUT_DIR / "README.md").write_text(txt, encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    write_gap_matrix()
    write_variable_catalog()

    base, links = build_cross_event_base()

    feature_cols = [
        "abs_x",
        "abs_y",
        "abs_x_rec",
        "abs_y_rec",
        "delta_abs_x",
        "delta_abs_y",
        "switch_side",
        "MaxHeight",
        "NumAttInBox",
        "NumDefInBox",
        "box_balance",
    ]

    X_pat, feat_stats = standardize_features(base, feature_cols)

    discovery = discover_patterns(X_pat)

    labels_rel, _ = relabel_clusters_by_frequency(discovery.labels_full)
    base["pattern_id"] = labels_rel
    base["pattern_cluster"] = base["pattern_id"].map(lambda x: f"P{int(x)+1}")

    # Persist base dataset with discovered patterns.
    base.to_csv(OUT_DIR / "03_cross_event_base.csv", index=False)

    # Discovery diagnostics.
    metrics = discovery.metrics.copy()
    metrics.to_csv(OUT_DIR / "03b_pattern_selection_metrics.csv", index=False)
    discovery.decision.to_csv(OUT_DIR / "03h_pattern_k_entscheidung.csv", index=False)

    # Reorder centers to frequency-based labels used in base.
    centers_reordered = (
        base[["pattern_id"]]
        .assign(_orig=discovery.labels_full)
        .drop_duplicates()
        .sort_values("pattern_id")
    )

    # Convert centers to original scale.
    feat_stats_idx = feat_stats.set_index("feature")
    centers_orig_rows = []
    for pid, orig in centers_reordered[["pattern_id", "_orig"]].to_numpy():
        row = {"pattern_id": int(pid), "pattern_cluster": f"P{int(pid)+1}"}
        for f in feature_cols:
            mu = float(feat_stats_idx.loc[f, "mean"])
            sd = float(feat_stats_idx.loc[f, "std"])
            row[f] = float(discovery.centers_full[int(orig), feature_cols.index(f)] * sd + mu)
        centers_orig_rows.append(row)

    centers_orig = pd.DataFrame(centers_orig_rows).sort_values("pattern_id")

    # Human-readable descriptors and suggested football labels.
    name_map = build_pattern_name_map(base=base, cluster_col="pattern_cluster")
    name_suggest = build_pattern_name_suggestions(centers_orig=centers_orig)
    name_all = name_map.merge(name_suggest, on="pattern_cluster", how="left")

    base = base.merge(name_all, on="pattern_cluster", how="left")
    base.to_csv(OUT_DIR / "03_cross_event_base.csv", index=False)

    profile_cols = [
        "abs_x",
        "abs_y",
        "abs_x_rec",
        "abs_y_rec",
        "delta_abs_x",
        "switch_side",
        "MaxHeight",
        "NumAttInBox",
        "NumDefInBox",
        "box_balance",
    ]
    profiles = pattern_profile_table(base=base, label_col="pattern_cluster", feature_cols=profile_cols)
    profiles = profiles.merge(name_all, on="pattern_cluster", how="left")
    profiles.to_csv(OUT_DIR / "03e_pattern_cluster_profiles.csv", index=False)

    names_out = (
        profiles[
            [
                "pattern_cluster",
                "pattern_name",
                "suggested_name",
                "start_zone",
                "width_zone",
                "trajectory",
                "height_level",
                "side_mode",
            ]
        ]
        .drop_duplicates()
        .sort_values("pattern_cluster")
    )
    names_out.to_csv(OUT_DIR / "03f_pattern_namensvorschlaege.csv", index=False)

    diff_summary = build_pattern_difference_summary(
        base=base,
        profiles=profiles,
        naming=names_out,
        feature_cols=profile_cols,
    )
    diff_summary.to_csv(OUT_DIR / "03g_pattern_unterschiede_summary.csv", index=False)

    write_validation(
        base=base,
        links=links,
        k_selected=discovery.k_selected,
        discovery_metrics=metrics,
        discovery_decision=discovery.decision,
    )
    write_descriptive(base=base)
    run_models(base=base)
    write_expose_text(base=base, k_selected=discovery.k_selected)
    write_readme()

    print("Expose pipeline finished.")
    print(f"Output dir: {OUT_DIR}")
    print(f"Events: {len(base)}")
    print(f"Shot<=8s share: {base['shot_within_8s'].mean():.4f}")
    print(f"Selected pattern K: {discovery.k_selected}")


if __name__ == "__main__":
    main()
