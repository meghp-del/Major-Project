# GRF Gait Pattern Classification System

End-to-end, label-free system that turns OpenSim force-plate recordings into one of 5 data-driven
gait patterns.

| Use | How |
|---|---|
| Clinician app | double-click `Run App.bat` (or `python -m streamlit run app.py`) |
| Command line | `python predict.py file.sto` · `python predict.py folder --out results.csv` |
| Retrain + regenerate figures | double-click `Train System.bat` (or `python build_system.py`) |

Inputs: raw OpenSim `.sto` (5 % threshold applied automatically) or thresholded `.csv` with
`time`, `1_ground_force_vx/vy/vz` and optionally `2_ground_force_*`.

Pipeline: threshold → contact detection → 20 Hz filter → 101-point normalisation →
waveform PCA → K-Means pseudo-labels (training) → classifier (deployment).

Subject IDs (`S01`, `S02`, …) are anonymous. The key mapping them to patient names is held
offline and is deliberately not part of this repository.
