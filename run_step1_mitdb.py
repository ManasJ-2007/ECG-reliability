"""
STEP 1: How good is the beat detector on normal hospital ECGs?

Runs Pan-Tompkins on all 48 MIT-BIH recordings and compares the result
with the doctors' labels. Saves:
  results/step1_mitdb.csv          - Se and PPV for every recording
  figures/fig1_pipeline.png        - what each Pan-Tompkins step does (record 100)
  figures/fig2_detections.png      - detected beats vs doctors' labels (record 100)
  figures/fig3_worst_record.png    - the recording where the detector does worst
"""
import os

import matplotlib
matplotlib.use("Agg")                 # save figures to files, no window
import matplotlib.pyplot as plt
import numpy as np

from src.data import MITDB_RECORDS, download_data, load_record
from src.evaluate import count_matches, ppv, sensitivity
from src.pan_tompkins import all_stages, detect_beats

os.makedirs("results", exist_ok=True)
os.makedirs("figures", exist_ok=True)
download_data()


# ---------------------------------------------------------------------------
# 1. Run the detector on every recording
# ---------------------------------------------------------------------------
rows = []
total_tp = total_fn = total_fp = 0

for name in MITDB_RECORDS:
    ecg, true_beats, fs = load_record(name, "mitdb")
    found_beats = detect_beats(ecg, fs)
    tp, fn, fp = count_matches(true_beats, found_beats, fs)

    total_tp += tp
    total_fn += fn
    total_fp += fp
    rows.append((name, len(true_beats), tp, fn, fp, sensitivity(tp, fn), ppv(tp, fp)))
    print(f"record {name}: Se = {sensitivity(tp, fn):.4f}   PPV = {ppv(tp, fp):.4f}")

overall_se = sensitivity(total_tp, total_fn)
overall_ppv = ppv(total_tp, total_fp)
print(f"\nOVERALL ({total_tp + total_fn} beats): Se = {overall_se:.4f}   PPV = {overall_ppv:.4f}")

with open("results/step1_mitdb.csv", "w") as f:
    f.write("record,true_beats,TP,FN,FP,Se,PPV\n")
    for name, n, tp, fn, fp, se, p in rows:
        f.write(f"{name},{n},{tp},{fn},{fp},{se:.4f},{p:.4f}\n")
    f.write(f"OVERALL,{total_tp + total_fn},{total_tp},{total_fn},{total_fp},"
            f"{overall_se:.4f},{overall_ppv:.4f}\n")


# ---------------------------------------------------------------------------
# 2. Figure: what each step of Pan-Tompkins does (record 100, 5 seconds)
# ---------------------------------------------------------------------------
ecg, true_beats, fs = load_record("100", "mitdb")
stages = all_stages(ecg, fs)
start, end = 0, 5 * fs
time = np.arange(start, end) / fs

titles = [("raw", "1. Raw ECG"),
          ("filtered", "2. Band-pass filtered (5-15 Hz)"),
          ("slope", "3. Derivative (slope)"),
          ("squared", "4. Squared"),
          ("energy", "5. Moving average (150 ms) - one bump per beat")]

fig, axes = plt.subplots(len(titles), 1, figsize=(10, 9), sharex=True)
for ax, (key, title) in zip(axes, titles):
    ax.plot(time, stages[key][start:end], color="black", linewidth=0.8)
    ax.set_title(title, fontsize=10, loc="left")
beats_here = stages["beats"][(stages["beats"] >= start) & (stages["beats"] < end)]
axes[0].plot(beats_here / fs, ecg[beats_here], "ro", label="detected R-peak")
axes[0].legend(loc="upper right")
axes[-1].set_xlabel("time (seconds)")
plt.tight_layout()
plt.savefig("figures/fig1_pipeline.png", dpi=130)
plt.close()


# ---------------------------------------------------------------------------
# 3. Helper: plot detections vs doctors' labels for any recording
# ---------------------------------------------------------------------------
def plot_detections(name, start_second, seconds, filename, title):
    ecg, true_beats, fs = load_record(name, "mitdb")
    found = detect_beats(ecg, fs)
    start, end = int(start_second * fs), int((start_second + seconds) * fs)
    time = np.arange(start, end) / fs
    true_here = true_beats[(true_beats >= start) & (true_beats < end)]
    found_here = found[(found >= start) & (found < end)]

    plt.figure(figsize=(10, 3.2))
    plt.plot(time, ecg[start:end], color="black", linewidth=0.8, label="ECG")
    plt.plot(true_here / fs, ecg[true_here] + 0.3, "gv", label="doctors' label")
    plt.plot(found_here / fs, ecg[found_here], "ro", fillstyle="none", markersize=9,
             label="my detector")
    plt.xlabel("time (seconds)")
    plt.ylabel("voltage (mV)")
    plt.title(title)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(filename, dpi=130)
    plt.close()


plot_detections("100", 0, 10, "figures/fig2_detections.png",
                "Record 100: detected beats match the doctors' labels")

# Worst recording = lowest of (Se, PPV)
worst = min(rows, key=lambda r: min(r[5], r[6]))
worst_name = worst[0]

# Find the 10 seconds in that recording with the most mistakes, to show them
ecg, true_beats, fs = load_record(worst_name, "mitdb")
found = detect_beats(ecg, fs)
most_errors, worst_start = -1, 0
for window_start in range(0, len(ecg) - 10 * fs, 10 * fs):
    t = true_beats[(true_beats >= window_start) & (true_beats < window_start + 10 * fs)]
    d = found[(found >= window_start) & (found < window_start + 10 * fs)]
    tp, fn, fp = count_matches(t, d, fs)
    if fn + fp > most_errors:
        most_errors, worst_start = fn + fp, window_start

plot_detections(worst_name, worst_start / fs, 10, "figures/fig3_worst_record.png",
                f"Record {worst_name} (worst): Se = {worst[5]:.3f}, PPV = {worst[6]:.3f}")
print(f"\nWorst record: {worst_name}. Figures saved in figures/.")
