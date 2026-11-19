"""GRF Gait Pattern Classifier - clinician-facing web app.

    streamlit run app.py          (or double-click "Run App.bat")
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from predict import MODEL_PATH, classify, load_model

HERE = Path(__file__).resolve().parent
SAMPLES = HERE / "sample_inputs"
PAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK2, GRID = "#52514e", "#e4e3df"

st.set_page_config(page_title="Gait Pattern Classifier", page_icon="🦶", layout="wide")


@st.cache_resource
def get_model():
    return load_model(MODEL_PATH)


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(color=GRID, lw=0.8); ax.set_axisbelow(True)
    ax.tick_params(colors=INK2)


def waveform_plot(waves, res, model, pattern):
    x = np.arange(101)
    cl = model["clusters"][pattern - 1]
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.fill_between(x, cl["mean_waveform"] - cl["sd_waveform"], cl["mean_waveform"] + cl["sd_waveform"],
                    color=PAL[pattern - 1], alpha=0.18, lw=0, label=f"Pattern {pattern} (mean ± SD)")
    ax.plot(x, cl["mean_waveform"], color=PAL[pattern - 1], lw=2, ls="--")
    for i, (_, r) in enumerate(res.iterrows()):
        ax.plot(x, waves.loc[i, model["wave_cols"]].to_numpy(float), color="#0b0b0b", lw=1.6,
                alpha=0.85, label="This patient's contacts" if i == 0 else None)
    ax.set_xlabel("Stance phase (%)"); ax.set_ylabel("vGRF / contact peak"); ax.set_ylim(0, 1.2)
    ax.legend(frameon=False, fontsize=8, loc="lower center"); style(ax)
    fig.tight_layout()
    return fig


def proba_plot(res, model):
    k = model["k"]
    p = res[[f"p_pattern_{c + 1}" for c in range(k)]].mean().to_numpy()
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    ax.barh([f"Pattern {c + 1}" for c in range(k)][::-1], p[::-1], color=PAL[:k][::-1], height=0.6)
    for i, v in enumerate(p[::-1]):
        ax.text(v + 0.01, i, f"{v:.0%}", va="center", fontsize=9, color=INK2)
    ax.set_xlim(0, 1.12); ax.set_xlabel("Probability"); style(ax); ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return fig


def show_result(name, src, model):
    res, waves, s, rejects = classify(src, model, name)
    if res is None:
        st.error(f"**{name}** - no valid foot contact found.")
        for n, why in rejects:
            st.caption(f"{n}: {why}")
        return None
    c1, c2, c3 = st.columns([2, 1, 1])
    c1.markdown(f"### Pattern {s['pattern']} — {s['shape']}")
    c1.caption(s["interpretation"])
    c2.metric("Confidence", f"{s['mean_confidence']:.0%}")
    c3.metric("Foot contacts", f"{s['n_contacts']}", f"{s['contact_agreement']:.0%} agree", delta_color="off")
    if s["flags"]:
        st.warning("Review recommended: " + "; ".join(s["flags"]))
    else:
        st.success("Consistent, confident classification.")
    g1, g2 = st.columns([3, 2])
    g1.pyplot(waveform_plot(waves, res, model, s["pattern"]), clear_figure=True)
    g2.pyplot(proba_plot(res, model), clear_figure=True)
    show = res[["plate", "contact", "pattern", "confidence", "atypical", "stance_s",
                "F1_pct", "F2_pct", "dip_depth_pct", "loading_rate"]].copy()
    st.dataframe(show.style.format({"confidence": "{:.0%}", "stance_s": "{:.2f}", "F1_pct": "{:.1f}",
                                    "F2_pct": "{:.1f}", "dip_depth_pct": "{:.1f}",
                                    "loading_rate": "{:.2f}"}), hide_index=True, use_container_width=True)
    return {"file": name, **{k: v for k, v in s.items() if k != "flags"},
            "flags": "; ".join(s["flags"]) or "-"}


model = get_model()

with st.sidebar:
    st.header("Model")
    st.write(f"**Classifier:** {model['classifier_name']}")
    st.write(f"**Gait patterns:** {model['k']} (K-Means, silhouette {model['silhouette']:.2f})")
    st.write(f"**Training data:** {model['n_train_contacts']} contacts, {model['n_train_subjects']} subjects")
    m = model["cv_metrics"]
    st.write(f"**Subject-grouped CV:** accuracy {m['accuracy']:.1%}, macro-F1 {m['macro_F1']:.2f}")
    st.write(f"**Strict CV (re-clustered per fold):** accuracy {model['strict_protocol']['accuracy']:.1%}")
    st.divider()
    st.caption("Research prototype. Patterns are data-driven groups of vertical GRF shape; the "
               "clinical interpretation is a hypothesis and is not a diagnosis.")

st.title("GRF Gait Pattern Classifier")
st.write("Upload raw OpenSim **.sto** files or thresholded **.csv** trials. Each foot contact is "
         "segmented, normalised to 0–100 % stance and assigned to one of the learned gait patterns.")

tab_cls, tab_pat = st.tabs(["Classify trials", "The gait patterns"])

with tab_cls:
    files = st.file_uploader("Trial files", type=["sto", "csv"], accept_multiple_files=True)
    samples = sorted(SAMPLES.glob("*.sto")) if SAMPLES.exists() else []
    use_sample = None
    if samples and not files:
        names = ["(none)"] + [p.name for p in samples]
        pre = st.query_params.get("sample")          # demo link: ?sample=<file name>
        use_sample = st.selectbox("…or try a sample trial", names,
                                  index=names.index(pre) if pre in names else 0)
    inputs = [(f.name, f) for f in files or []]
    if use_sample and use_sample != "(none)":
        inputs = [(use_sample, SAMPLES / use_sample)]

    rows = []
    for name, src in inputs:
        with st.container(border=True):
            st.markdown(f"**{name}**")
            r = show_result(name, src, model)
            if r:
                rows.append(r)
    if len(rows) > 1:
        st.subheader("Batch summary")
        summ = pd.DataFrame(rows)[["file", "pattern", "shape", "mean_confidence", "n_contacts", "flags"]]
        st.dataframe(summ.style.format({"mean_confidence": "{:.0%}"}), hide_index=True, use_container_width=True)
    if rows:
        st.download_button("Download results (CSV)", pd.DataFrame(rows).to_csv(index=False),
                           "gait_classification_results.csv", "text/csv")

with tab_pat:
    k = model["k"]
    cols = st.columns(k)
    x = np.arange(101)
    for c in range(k):
        cl = model["clusters"][c]
        fig, ax = plt.subplots(figsize=(3, 2.4))
        ax.fill_between(x, cl["mean_waveform"] - cl["sd_waveform"], cl["mean_waveform"] + cl["sd_waveform"],
                        color=PAL[c], alpha=0.2, lw=0)
        ax.plot(x, cl["mean_waveform"], color=PAL[c], lw=2); ax.set_ylim(0, 1.2); style(ax)
        ax.set_xlabel("Stance %", fontsize=8); ax.tick_params(labelsize=7); fig.tight_layout()
        with cols[c]:
            st.markdown(f"**Pattern {c + 1}**")
            st.pyplot(fig, clear_figure=True)
            st.caption(f"**{cl['shape']}** · {cl['n']} contacts / {cl['n_subjects']} subjects")
            st.caption(cl["rodda_graham_hint"])
