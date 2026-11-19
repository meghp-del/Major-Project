"""Build the automated GRF gait classification system (Evaluation 3).

Stage A  Unsupervised pattern discovery
         K-Means vs Ward vs Gaussian Mixture, k = 2..8, on waveform PCA space;
         silhouette / Davies-Bouldin / Calinski-Harabasz + bootstrap stability (ARI).
Stage B  Automated classification
         The chosen clustering gives pseudo-labels (gait-pattern groups). Supervised
         classifiers (KNN, SVM, Random Forest, Logistic Regression, Gradient Boosting, MLP)
         learn to assign NEW contacts to those groups. Evaluated with subject-grouped
         5-fold CV (no subject in both train and test), plus a strict protocol in which
         the clustering itself is refitted inside every fold.
Stage C  Saves the deployable model (models/gait_classifier.joblib), tables and figures.

    python build_system.py
    python build_system.py --data "path/to/threshold data" --k 5
"""
import argparse
import hashlib
import json
import re
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.optimize import linear_sum_assignment
from sklearn.base import clone
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, adjusted_rand_score, balanced_accuracy_score,
                             calinski_harabasz_score, cohen_kappa_score, confusion_matrix,
                             davies_bouldin_score, f1_score, silhouette_samples, silhouette_score)
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from gait_pipeline import SCALAR_FEATURES, WAVE_COLS, process_trial

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
DEFAULT_DATA = HERE.parent.parent / "Evaluation 2" / "Major Project" / "threshold data"
DEFAULT_RAW = HERE.parent.parent / "Evaluation 2" / "Major Project" / "drive-download-20260902T061838Z-1-001"

SEED, VAR_RETAINED, N_BOOT, N_FOLDS = 42, 0.80, 50, 5
CONDITIONS = ("LB", "FB")
NAME_RE = re.compile(r"^(?P<subject>.*?)[_ -]?Gait[_ ](?P<cond>[A-Z]+)[_ ](?P<num>\d+)"
                     r"(?:(?:__| \()(?P<rep>\d+)[_)])?$")

# ---- figure style (reference categorical palette, fixed order) -------------
PAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11, "axes.edgecolor": INK2,
    "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "axes.titleweight": "bold", "axes.titlesize": 12, "legend.frameon": False,
    "savefig.dpi": 200, "savefig.bbox": "tight", "figure.facecolor": "white",
})


# =========================================================================
# helpers
# =========================================================================
def parse_name(stem):
    m = NAME_RE.match(stem)
    if not m:
        return "unparsed", "?", stem
    subj = m["subject"].strip("_- ") or "anon"
    return subj, m["cond"], f"{m['cond']}_{m['num']}" + (f"r{m['rep']}" if m["rep"] else "")


def force_hash(path):
    d = pd.read_csv(path, usecols=lambda c: c.strip() in ("1_ground_force_vy", "2_ground_force_vy"))
    return hashlib.md5(d.round(3).to_numpy().tobytes()).hexdigest()


def align_labels(ref, new, k):
    """Permute `new` cluster ids to best match `ref` (Hungarian on the contingency table)."""
    C = np.zeros((k, k))
    for r, n in zip(ref, new):
        C[n, r] += 1
    rows, cols = linear_sum_assignment(-C)
    mapping = dict(zip(rows, cols))
    return np.array([mapping.get(n, n) for n in new])


def cluster_fn(algo, k, seed=SEED):
    if algo == "K-Means":
        return lambda Z: KMeans(k, random_state=seed, n_init=20).fit_predict(Z)
    if algo == "Ward":
        return lambda Z: AgglomerativeClustering(k, linkage="ward").fit_predict(Z)
    if algo == "GMM":
        return lambda Z: GaussianMixture(k, covariance_type="full", random_state=seed,
                                         n_init=5).fit(Z).predict(Z)
    raise ValueError(algo)


def shape_description(mu, loading_rate, lr_median):
    """Descriptive label for a cluster's mean waveform + Rodda & Graham cross-reference.
    Heuristic only - GRF alone cannot confirm a sagittal kinematic group."""
    f1 = 15 + int(np.argmax(mu[15:45])); f2 = 55 + int(np.argmax(mu[55:85]))
    dip = 100 * (min(mu[f1], mu[f2]) - mu[f1:f2 + 1].min())
    F1, F2 = 100 * mu[f1], 100 * mu[f2]
    if F1 - F2 >= 25:
        return ("Impact-dominant, weak push-off",
                "high early loading, little propulsion - hypothesis: equinus / forefoot loading")
    if dip >= 20:
        return ("Preserved double bump",
                "near-typical loading and push-off - mildest pattern")
    if F1 - F2 >= 10:
        return ("Early-dominant, reduced push-off",
                "reduced propulsion with longer stance - hypothesis: apparent equinus / mild crouch")
    if dip >= 5:
        return ("Shallow double bump",
                "little mid-stance unloading - hypothesis: stiff-knee / jump-type pattern")
    return ("Flat plateau, slow loading" if loading_rate < lr_median else "Flat plateau",
            "no mid-stance unloading - hypothesis: crouch-type loading (Rodda & Graham Group IV)")


def plot_clustering_comparison(sweep, K, path):
    """Silhouette (top) and bootstrap stability (bottom) by k for each algorithm."""
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 6.6), sharex=True)
    for i, algo in enumerate(("K-Means", "Ward", "GMM")):
        s = sweep[sweep.algorithm == algo]
        axes[0].plot(s.k, s.silhouette, marker=MARKERS[i], color=PAL[i], lw=2, ms=7, label=algo)
        axes[1].plot(s.k, s.stability_ARI_mean, marker=MARKERS[i], color=PAL[i], lw=2, ms=7, label=algo)
    for ax, t, yl in ((axes[0], "Cluster separation", "Silhouette score"),
                      (axes[1], "Cluster stability (50 bootstraps)", "Mean ARI")):
        ax.axvline(K, color=INK2, ls="--", lw=1); ax.set_title(t, loc="left"); ax.set_ylabel(yl)
    axes[1].set_xlabel("Number of clusters k")
    axes[0].legend(ncol=3, loc="upper right")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


# =========================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(DEFAULT_DATA))
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--out", default=str(HERE / "outputs"))
    ap.add_argument("--model", default=str(HERE / "models" / "gait_classifier.joblib"))
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    Path(args.model).parent.mkdir(parents=True, exist_ok=True)
    K = args.k
    log = {}

    # ---------------------------------------------------------------- data
    files = sorted(Path(args.data).glob("*Gait*.csv"))
    if not files:
        raise SystemExit(f"no *Gait*.csv files found in {args.data}")
    used, seen, info, dup, other_cond = [], set(), {}, 0, 0
    for f in files:
        subj, cond, tid = parse_name(f.stem)
        h = force_hash(f)
        if h in seen:
            dup += 1; continue
        seen.add(h)
        if cond not in CONDITIONS:
            other_cond += 1; continue
        used.append(f); info[f.stem] = (subj, cond, tid)

    # anonymise: subject names never leave this machine except in the PRIVATE key file
    subjects = sorted({v[0] for v in info.values()})
    sid = {s: f"S{i + 1:02d}" for i, s in enumerate(subjects)}
    pd.DataFrame({"subject_id": list(sid.values()), "original_name": list(sid.keys())}) \
        .to_csv(out / "PRIVATE_subject_key.csv", index=False)

    feats, waves, rejects = [], [], []
    for f in used:
        fe, wa, rj = process_trial(f)
        subj, cond, tid = info[f.stem]
        anon = f"{sid[subj]}_{tid}"
        for d in (fe, wa):
            if len(d):
                d["trial"] = anon
        if len(fe):
            fe.insert(1, "subject", sid[subj]); fe.insert(2, "condition", cond)
        feats.append(fe); waves.append(wa)
        rejects += [(f"{anon}", why) for _, why in rj]
    KEY = ["trial", "plate", "contact"]
    F = pd.concat(feats).sort_values(KEY).reset_index(drop=True)
    Wdf = pd.concat(waves).sort_values(KEY).reset_index(drop=True)
    assert (F[KEY].values == Wdf[KEY].values).all()
    W = Wdf[WAVE_COLS].to_numpy(float)
    groups = F.subject.values
    log["data"] = {"files_found": len(files), "duplicates_removed": dup,
                   "other_conditions_skipped": other_cond, "trials_used": len(used),
                   "trials_with_contacts": int(F.trial.nunique()), "subjects": int(F.subject.nunique()),
                   "contacts": len(F), "rejected_plates_or_trials": len(rejects),
                   "contacts_per_subject_median": float(F.groupby("subject").size().median())}
    print(json.dumps(log["data"], indent=1))
    pd.DataFrame(rejects, columns=["trial", "reason"]).to_csv(out / "rejected_contacts.csv", index=False)

    # ---------------------------------------------------- Stage A: clustering
    pca = PCA(n_components=VAR_RETAINED, svd_solver="full", random_state=SEED).fit(W)
    Z = pca.transform(W)
    log["pca"] = {"components": int(Z.shape[1]),
                  "variance_pct": round(float(100 * pca.explained_variance_ratio_.sum()), 1),
                  "per_pc_pct": [round(float(100 * v), 1) for v in pca.explained_variance_ratio_]}
    print("PCA:", log["pca"])

    rng = np.random.default_rng(SEED)
    rows = []
    for algo in ("K-Means", "Ward", "GMM"):
        for k in range(2, 9):
            lab = cluster_fn(algo, k)(Z)
            aris = []
            for b in range(N_BOOT):
                idx = rng.choice(len(Z), int(0.8 * len(Z)), replace=False)
                lb = cluster_fn(algo, k, seed=SEED + b + 1)(Z[idx])
                aris.append(adjusted_rand_score(lab[idx], lb))
            sizes = np.bincount(lab, minlength=k)
            rows.append({"algorithm": algo, "k": k,
                         "silhouette": silhouette_score(Z, lab),
                         "davies_bouldin": davies_bouldin_score(Z, lab),
                         "calinski_harabasz": calinski_harabasz_score(Z, lab),
                         "stability_ARI_mean": float(np.mean(aris)),
                         "stability_ARI_sd": float(np.std(aris)),
                         "smallest_cluster": int(sizes.min())})
        print(f"  {algo} done")
    sweep = pd.DataFrame(rows)
    sweep.round(4).to_csv(out / "clustering_comparison.csv", index=False)
    print(sweep.round(3).to_string(index=False))

    # final clustering: K-Means at K, ids re-ordered by mean dip depth (1 = deepest dip)
    km = KMeans(K, random_state=SEED, n_init=20).fit(Z)
    raw = km.labels_
    dip_by = pd.Series(F.dip_depth_pct.values).groupby(raw).mean().sort_values(ascending=False)
    order = {old: new for new, old in enumerate(dip_by.index)}
    y = np.array([order[l] for l in raw])
    centers = km.cluster_centers_[dip_by.index.values]
    ward_lab = AgglomerativeClustering(K, linkage="ward").fit_predict(Z)
    gmm_lab = cluster_fn("GMM", K)(Z)
    log["final_clustering"] = {
        "algorithm": "K-Means", "k": K,
        "silhouette": round(float(silhouette_score(Z, y)), 4),
        "agreement_ARI_with_Ward": round(float(adjusted_rand_score(y, ward_lab)), 3),
        "agreement_ARI_with_GMM": round(float(adjusted_rand_score(y, gmm_lab)), 3)}
    print(log["final_clustering"])

    lr_median = float(np.nanmedian(F.loading_rate))
    dist = np.linalg.norm(Z - centers[y], axis=1)
    clusters, prof = {}, []
    for c in range(K):
        m = y == c
        mu = W[m].mean(0)
        shape, hint = shape_description(mu, F.loading_rate[m].mean(), lr_median)
        clusters[c] = {"name": f"Pattern {c + 1}", "n": int(m.sum()),
                       "n_subjects": int(F.subject[m].nunique()),
                       "shape": shape, "rodda_graham_hint": hint,
                       "mean_waveform": mu, "sd_waveform": W[m].std(0),
                       "dist_p95": float(np.percentile(dist[m], 95))}
        prof.append({"cluster": f"Pattern {c + 1}", "contacts": int(m.sum()),
                     "subjects": int(F.subject[m].nunique()), "shape": shape,
                     "rodda_graham_hint": hint,
                     **F.loc[m, SCALAR_FEATURES].mean().round(3).to_dict()})
    profile = pd.DataFrame(prof)
    profile.to_csv(out / "cluster_profile.csv", index=False)
    print(profile[["cluster", "contacts", "subjects", "shape", "dip_depth_pct", "stance_s"]].to_string(index=False))

    assign = F.assign(pattern=y + 1, **{f"PC{i + 1}": Z[:, i] for i in range(Z.shape[1])})
    assign.to_csv(out / "cluster_assignments.csv", index=False)
    Wdf.to_csv(out / "vgrf_waveforms_101.csv", index=False)

    # how consistent is one subject's pattern across their contacts?
    purity = assign.groupby("subject").pattern.agg(lambda s: s.value_counts().iloc[0] / len(s))
    log["subject_consistency"] = {"mean_share_of_dominant_pattern": round(float(purity.mean()), 3),
                                  "subjects_all_contacts_same_pattern": int((purity == 1).sum())}

    # ------------------------------------------- Stage B: classification
    X = pd.concat([F[SCALAR_FEATURES].reset_index(drop=True), Wdf[WAVE_COLS]], axis=1)
    pre_scalar = ColumnTransformer([("s", make_pipeline(SimpleImputer(strategy="median"),
                                                        StandardScaler()), SCALAR_FEATURES)])
    pre_wave = ColumnTransformer([("w", make_pipeline(PCA(VAR_RETAINED, svd_solver="full"),
                                                      StandardScaler()), WAVE_COLS)])
    pre_both = ColumnTransformer([
        ("s", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), SCALAR_FEATURES),
        ("w", make_pipeline(PCA(VAR_RETAINED, svd_solver="full"), StandardScaler()), WAVE_COLS)])
    feature_sets = {"Biomechanical features (14)": pre_scalar,
                    "Waveform PCA": pre_wave,
                    "Combined": pre_both}
    models = {
        "KNN": KNeighborsClassifier(n_neighbors=7, weights="distance"),
        "SVM (RBF)": SVC(C=10, gamma="scale", probability=True, random_state=SEED),
        "Logistic Regression": LogisticRegression(C=1.0, max_iter=5000),
        "Random Forest": RandomForestClassifier(400, random_state=SEED, class_weight="balanced"),
        "Gradient Boosting": GradientBoostingClassifier(random_state=SEED),
        "MLP": MLPClassifier((64, 32), max_iter=3000, alpha=1e-3, random_state=SEED),
    }
    cv = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    splits = list(cv.split(X, y, groups))
    res, oof_store = [], {}
    for fs_name, pre in feature_sets.items():
        for m_name, clf in models.items():
            oof = np.empty_like(y)
            for tr, te in splits:
                p = Pipeline([("pre", clone(pre)), ("clf", clone(clf))]).fit(X.iloc[tr], y[tr])
                oof[te] = p.predict(X.iloc[te])
            fold_acc = [accuracy_score(y[te], oof[te]) for _, te in splits]
            res.append({"features": fs_name, "model": m_name,
                        "accuracy": accuracy_score(y, oof),
                        "accuracy_fold_sd": float(np.std(fold_acc)),
                        "balanced_accuracy": balanced_accuracy_score(y, oof),
                        "macro_F1": f1_score(y, oof, average="macro"),
                        "cohen_kappa": cohen_kappa_score(y, oof)})
            oof_store[(fs_name, m_name)] = oof
            print(f"  {fs_name:28s} {m_name:20s} acc {res[-1]['accuracy']:.3f}  F1 {res[-1]['macro_F1']:.3f}")
    results = pd.DataFrame(res).sort_values("macro_F1", ascending=False)
    results.round(4).to_csv(out / "classifier_comparison.csv", index=False)

    # deployable model: best classifier on the interpretable feature set + best overall
    best = results.iloc[0]
    best_key = (best.features, best.model)
    log["best_classifier"] = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in best.items()}
    print("BEST:", log["best_classifier"])
    final = Pipeline([("pre", clone(feature_sets[best.features])), ("clf", clone(models[best.model]))])
    final.fit(X, y)

    # strict protocol: re-cluster inside every training fold (no information from test subjects)
    strict_pred = np.empty_like(y)
    for tr, te in splits:
        p_tr = PCA(VAR_RETAINED, svd_solver="full").fit(W[tr])
        lab_tr = KMeans(K, random_state=SEED, n_init=20).fit_predict(p_tr.transform(W[tr]))
        lab_tr = align_labels(y[tr], lab_tr, K)
        p = Pipeline([("pre", clone(feature_sets[best.features])), ("clf", clone(models[best.model]))])
        strict_pred[te] = p.fit(X.iloc[tr], lab_tr).predict(X.iloc[te])
    log["strict_protocol"] = {"accuracy": round(accuracy_score(y, strict_pred), 4),
                              "macro_F1": round(f1_score(y, strict_pred, average="macro"), 4),
                              "cohen_kappa": round(cohen_kappa_score(y, strict_pred), 4)}
    print("STRICT:", log["strict_protocol"])

    # trial-level (majority vote over a trial's contacts)
    oof_best = oof_store[best_key]
    tv = pd.DataFrame({"trial": F.trial, "y": y, "p": oof_best}).groupby("trial").agg(
        y=("y", lambda s: s.mode().iloc[0]), p=("p", lambda s: s.mode().iloc[0]))
    log["trial_level_accuracy"] = round(float((tv.y == tv.p).mean()), 4)

    # permutation importance on the scalar-feature RF (interpretability)
    rf_s = Pipeline([("pre", clone(pre_scalar)), ("clf", clone(models["Random Forest"]))]).fit(X, y)
    pi = permutation_importance(rf_s, X, y, n_repeats=20, random_state=SEED, scoring="f1_macro")
    imp = pd.DataFrame({"feature": SCALAR_FEATURES,
                        "importance": pi.importances_mean[:len(SCALAR_FEATURES)]
                        if len(pi.importances_mean) == len(SCALAR_FEATURES) else
                        pi.importances_mean[[X.columns.get_loc(c) for c in SCALAR_FEATURES]],
                        }).sort_values("importance", ascending=False)
    imp.to_csv(out / "feature_importance.csv", index=False)

    # ------------------------------------------------------- save model
    bundle = {"classifier": final, "classifier_name": f"{best.model} on {best.features}",
              "pca": pca, "centers": centers, "k": K, "clusters": clusters,
              "scalar_features": SCALAR_FEATURES, "wave_cols": WAVE_COLS,
              "cv_metrics": log["best_classifier"], "strict_protocol": log["strict_protocol"],
              "n_train_contacts": int(len(y)), "n_train_subjects": int(F.subject.nunique()),
              "silhouette": log["final_clustering"]["silhouette"], "version": "eval3-1.0"}
    joblib.dump(bundle, args.model)
    with open(out / "summary.json", "w") as fh:
        json.dump(log, fh, indent=2, default=float)
    print(f"\nmodel saved -> {args.model}")

    # =================================================== figures
    figs = out / "figures"; figs.mkdir(exist_ok=True)
    x101 = np.arange(101)

    # 1. clustering comparison: silhouette + stability by k
    plot_clustering_comparison(sweep, K, figs / "01_clustering_comparison.png")

    # 2. archetypes
    fig, axes = plt.subplots(1, K, figsize=(3.1 * K, 3.4), sharey=True)
    for c in range(K):
        ax, cl = axes[c], clusters[c]
        ax.fill_between(x101, cl["mean_waveform"] - cl["sd_waveform"],
                        cl["mean_waveform"] + cl["sd_waveform"], color=PAL[c], alpha=0.18, lw=0)
        ax.plot(x101, cl["mean_waveform"], color=PAL[c], lw=2.2)
        ax.set_title(f"Pattern {c + 1}  (n={cl['n']})", loc="left")
        ax.text(0.03, 0.04, cl["shape"], transform=ax.transAxes, fontsize=8.5, color=INK2)
        ax.set_xlabel("Stance phase (%)"); ax.set_ylim(0, 1.15)
    axes[0].set_ylabel("vGRF / contact peak")
    fig.tight_layout(); fig.savefig(figs / "02_cluster_archetypes.png"); plt.close(fig)

    # 3. PCA scatter (colour + marker shape = double encoding)
    fig, ax = plt.subplots(figsize=(7.5, 5.4))
    for c in range(K):
        m = y == c
        ax.scatter(Z[m, 0], Z[m, 1], s=34, color=PAL[c], marker=MARKERS[c], edgecolor="white",
                   linewidth=0.7, alpha=0.9, label=f"Pattern {c + 1} (n={m.sum()})")
    ax.set_xlabel(f"PC1 ({100 * pca.explained_variance_ratio_[0]:.0f} % of variance)")
    ax.set_ylabel(f"PC2 ({100 * pca.explained_variance_ratio_[1]:.0f} % of variance)")
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout(); fig.savefig(figs / "03_pca_scatter.png"); plt.close(fig)

    # 4. dendrogram
    L = linkage(Z, "ward")
    cut = (L[-K, 2] + L[-K + 1, 2]) / 2
    fig, ax = plt.subplots(figsize=(11, 4.2))
    # one neutral colour: Ward groups are not the same as the K-Means patterns
    dendrogram(L, ax=ax, no_labels=True, color_threshold=0, above_threshold_color=PAL[0])
    ax.axhline(cut, color=INK2, ls="--", lw=1)
    ax.text(ax.get_xlim()[1], cut, f"  cut for k = {K}", va="bottom", ha="right", color=INK2, fontsize=9)
    ax.set_ylabel("Ward linkage distance"); ax.set_xlabel(f"{len(Z)} foot contacts"); ax.grid(False)
    fig.tight_layout(); fig.savefig(figs / "04_dendrogram.png"); plt.close(fig)

    # 5. silhouette plot
    sv = silhouette_samples(Z, y)
    fig, ax = plt.subplots(figsize=(6.5, 4.8)); lo = 0
    for c in range(K):
        v = np.sort(sv[y == c]); ax.fill_betweenx(np.arange(lo, lo + len(v)), 0, v, color=PAL[c], lw=0)
        ax.text(-0.06, lo + len(v) / 2, f"P{c + 1}", va="center", ha="right", fontsize=9, color=INK2)
        lo += len(v) + 6
    ax.axvline(sv.mean(), color=INK, ls="--", lw=1)
    ax.text(sv.mean(), lo, f" mean {sv.mean():.2f}", fontsize=9, color=INK, va="bottom")
    ax.set_yticks([]); ax.set_xlabel("Silhouette coefficient"); ax.grid(axis="y", visible=False)
    fig.tight_layout(); fig.savefig(figs / "05_silhouette.png"); plt.close(fig)

    # 6. classifier comparison (macro-F1, grouped by feature set)
    fig, ax = plt.subplots(figsize=(11, 4.4))
    mnames = list(models); fsn = list(feature_sets); w = 0.26
    for i, f_ in enumerate(fsn):
        vals = [results[(results.features == f_) & (results.model == m)].macro_F1.iloc[0] for m in mnames]
        xs = np.arange(len(mnames)) + (i - 1) * w
        ax.bar(xs, vals, w - 0.03, color=PAL[i], label=f_)
        for xx, v in zip(xs, vals):
            ax.text(xx, v + 0.01, f"{v:.2f}", ha="center", fontsize=7.5, color=INK2)
    ax.set_xticks(np.arange(len(mnames)), mnames); ax.set_ylim(0, 1.08)
    ax.set_ylabel("Macro F1 (subject-grouped 5-fold CV)"); ax.legend(ncol=3, loc="upper left")
    ax.grid(axis="x", visible=False)
    fig.tight_layout(); fig.savefig(figs / "06_classifier_comparison.png"); plt.close(fig)

    # 7. confusion matrix of the best model
    cm = confusion_matrix(y, oof_best, labels=range(K))
    cmn = cm / cm.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1); ax.grid(False)
    for i in range(K):
        for j in range(K):
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center", fontsize=10,
                    color="white" if cmn[i, j] > 0.55 else INK)
    ticks = [f"P{c + 1}" for c in range(K)]
    ax.set_xticks(range(K), ticks); ax.set_yticks(range(K), ticks)
    ax.set_xlabel("Predicted pattern"); ax.set_ylabel("Cluster (pseudo-label)")
    ax.set_title(f"{best.model} - {best.features}", loc="left", fontsize=10)
    fig.tight_layout(); fig.savefig(figs / "07_confusion_matrix.png"); plt.close(fig)

    # 8. feature importance
    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    im = imp.sort_values("importance")
    ax.barh(im.feature, im.importance, color=PAL[0], height=0.65)
    ax.set_xlabel("Permutation importance (drop in macro F1)"); ax.grid(axis="y", visible=False)
    fig.tight_layout(); fig.savefig(figs / "08_feature_importance.png"); plt.close(fig)

    # 9. dataset overview: contacts per subject
    cps = F.groupby("subject").size().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(11, 3.4))
    ax.bar(range(len(cps)), cps.values, color=PAL[0], width=0.7)
    ax.set_xticks(range(len(cps)), cps.index, rotation=90, fontsize=7)
    ax.set_ylabel("Foot contacts"); ax.grid(axis="x", visible=False)
    fig.tight_layout(); fig.savefig(figs / "09_contacts_per_subject.png"); plt.close(fig)

    # 10. preprocessing example on an anonymous trial (raw .sto -> threshold -> contact)
    raw = DEFAULT_RAW / "Gait FB 1.sto"
    if raw.exists():
        from gait_pipeline import apply_threshold, find_contacts, lowpass, read_sto, resample_101
        r = read_sto(raw); th = apply_threshold(r)
        t = r.time.values; vy_r = r["1_ground_force_vy"].values; vy_t = th["1_ground_force_vy"].values
        thr = 0.05 * max(r["1_ground_force_vy"].max(), r["2_ground_force_vy"].max())
        a, b = find_contacts(vy_t, t)[0]
        seg = lowpass(vy_t[a:b + 1], 1 / np.median(np.diff(t)))
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
        axes[0].plot(t, vy_r, color=PAL[0], lw=1.6); axes[0].axhline(thr, color=PAL[1], ls="--", lw=1.2)
        axes[0].text(t[-1], thr, "5 % of peak", color=INK2, fontsize=9, ha="right", va="bottom")
        axes[0].set_title("1  Raw .sto vertical GRF", loc="left"); axes[0].set_ylabel("Force (N)")
        axes[1].plot(t, vy_t, color=PAL[0], lw=1.6)
        axes[1].axvspan(t[a], t[b], color=PAL[2], alpha=0.15, lw=0)
        axes[1].text((t[a] + t[b]) / 2, vy_t.max() * 0.5, "detected\nstance", ha="center", fontsize=9, color=INK2)
        axes[1].set_title("2  Thresholded + contact detected", loc="left")
        for ax in axes[:2]:
            ax.set_xlabel("Time (s)")
        axes[2].plot(np.arange(101), resample_101(seg) / seg.max(), color=PAL[0], lw=2)
        axes[2].set_title("3  Filtered, 101 pts, peak = 1", loc="left")
        axes[2].set_xlabel("Stance phase (%)"); axes[2].set_ylabel("vGRF / peak")
        fig.tight_layout(); fig.savefig(figs / "10_preprocessing_example.png"); plt.close(fig)

    print(f"figures -> {figs}")


if __name__ == "__main__":
    main()
