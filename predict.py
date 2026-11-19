"""Automated gait-pattern classification for new trials (command line).

    python predict.py "Gait FB 1.sto"
    python predict.py new_trials_folder --out predictions.csv
    python predict.py a.sto b.csv

Accepts raw OpenSim .sto files or thresholded .csv files. Each valid foot contact is
classified; the trial-level pattern is the majority vote, weighted by confidence.
"""
import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from gait_pipeline import process_trial

HERE = Path(__file__).resolve().parent
MODEL_PATH = HERE / "models" / "gait_classifier.joblib"
LOW_CONFIDENCE = 0.60


def load_model(path=MODEL_PATH):
    if not Path(path).exists():
        raise FileNotFoundError(f"model not found: {path}  (run build_system.py first)")
    return joblib.load(path)


def classify(src, model, name=None):
    """Classify one trial. Returns (contacts_df, waves_df, summary_dict, rejects).

    contacts_df is None when no valid foot contact was found."""
    feats, waves, rejects = process_trial(src, name)
    if feats.empty:
        return None, None, None, rejects
    X = pd.concat([feats[model["scalar_features"]].reset_index(drop=True),
                   waves[model["wave_cols"]].reset_index(drop=True)], axis=1)
    clf = model["classifier"]
    proba = clf.predict_proba(X)
    pred = proba.argmax(1)
    conf = proba.max(1)

    # out-of-distribution check: distance to the cluster centre in the training PCA space
    Z = model["pca"].transform(waves[model["wave_cols"]].to_numpy(float))
    dist = np.linalg.norm(Z - model["centers"][pred], axis=1)
    atypical = np.array([d > model["clusters"][c]["dist_p95"] for c, d in zip(pred, dist)])

    out = feats[["trial", "plate", "contact", "stance_s", "peak_vGRF_N", "F1_pct", "F2_pct",
                 "dip_depth_pct", "loading_rate"]].copy().reset_index(drop=True)
    out["pattern"] = pred + 1
    out["confidence"] = conf
    out["atypical"] = atypical
    out["pattern_shape"] = [model["clusters"][c]["shape"] for c in pred]
    for c in range(model["k"]):
        out[f"p_pattern_{c + 1}"] = proba[:, c]

    votes = pd.Series(conf).groupby(pred).sum()
    trial_c = int(votes.idxmax())
    info = model["clusters"][trial_c]
    agree = float((pred == trial_c).mean())
    flags = []
    if conf.mean() < LOW_CONFIDENCE:
        flags.append("low classifier confidence")
    if atypical.mean() > 0.5:
        flags.append("waveform unlike the training data")
    if agree < 1:
        flags.append(f"contacts disagree ({agree:.0%} agree)")
    summary = {"pattern": trial_c + 1, "name": info["name"], "shape": info["shape"],
               "interpretation": info["rodda_graham_hint"], "n_contacts": len(out),
               "mean_confidence": float(conf[pred == trial_c].mean()),
               "contact_agreement": agree, "flags": flags}
    return out, waves.reset_index(drop=True), summary, rejects


def main():
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", help=".sto/.csv file(s) or folder(s)")
    ap.add_argument("--model", default=str(MODEL_PATH))
    ap.add_argument("--out", help="save per-contact predictions to this CSV")
    args = ap.parse_args()
    model = load_model(args.model)

    paths = []
    for p in map(Path, args.inputs):
        if p.is_dir():
            paths += sorted(q for q in p.iterdir() if q.suffix.lower() in (".sto", ".csv")
                            and q.name != "threshold_summary.csv")
        else:
            paths.append(p)

    print(f"Model: {model['classifier_name']}  |  {model['k']} gait patterns  |  "
          f"trained on {model['n_train_contacts']} contacts / {model['n_train_subjects']} subjects\n")
    allres = []
    for p in paths:
        if not p.exists():
            print(f"{p}: file not found\n"); continue
        res, _, s, rejects = classify(p, model)
        if res is None:
            print(f"{p.name}: no valid foot contact found")
            for n, why in rejects:
                print(f"    {n}: {why}")
            print(); continue
        print(f"{p.name}\n  ==> PATTERN {s['pattern']}: {s['shape']}  "
              f"(confidence {s['mean_confidence']:.0%}, {s['n_contacts']} contact(s))")
        print(f"      {s['interpretation']}")
        for f in s["flags"]:
            print(f"      ! review: {f}")
        print()
        allres.append(res.assign(file=p.name, trial_pattern=s["pattern"]))

    if args.out and allres:
        pd.concat(allres).to_csv(args.out, index=False)
        print(f"saved -> {args.out}")
    if not allres:
        sys.exit(1)


if __name__ == "__main__":
    main()
