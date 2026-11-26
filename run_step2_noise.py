"""
STEP 2: How much electrode-motion noise can the detector handle?

Uses the Noise Stress Test Database: records 118 and 119 with real
electrode-motion noise added at 24, 18, 12, 6, 0 and -6 dB.

How the noise is added (from the database description):
  - the first 5 minutes are clean
  - after that: 2 minutes noisy, 2 minutes clean, 2 minutes noisy, ...
We score ONLY the noisy minutes, so the result really shows the effect of noise.
As the "no noise" reference we use the same minutes of the original clean
records 118 and 119 from MIT-BIH.

Saves:
  results/step2_noise.csv         - Se and PPV per noise level, both detectors
  figures/fig4_noise_examples.png - the same ECG at different noise levels
  figures/fig5_se_ppv_vs_snr.png  - Se and PPV against SNR
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.data import NOISE_LEVELS, download_data, load_record, noisy_record_name
from src.evaluate import count_matches, ppv, sensitivity
from src.pan_tompkins import detect_beats
from src.sqi import second_detector

os.makedirs("results", exist_ok=True)
os.makedirs("figures", exist_ok=True)
download_data()


def is_noisy(sample, fs):
    """True if this sample lies in a minute where noise was added."""
    minute = sample / fs / 60
    return minute >= 5 and int((minute - 5) // 2) % 2 == 0


def only_noisy(beats, fs):
    return np.array([b for b in beats if is_noisy(b, fs)], dtype=int)


# ---------------------------------------------------------------------------
# 1. Score both detectors at every noise level
# ---------------------------------------------------------------------------
# "clean" = the original record, then each noise level from light to heavy
levels = ["clean"] + NOISE_LEVELS
results = {}   # results[level] = {"pan_tompkins": [tp, fn, fp], "xqrs": [tp, fn, fp]}

for level in levels:
    totals = {"pan_tompkins": [0, 0, 0], "xqrs": [0, 0, 0]}
    for base in ["118", "119"]:
        if level == "clean":
            name, database = base, "mitdb"
        else:
            name, database = noisy_record_name(base, level), "nstdb"

        ecg, true_beats, fs = load_record(name, database)
        true_noisy = only_noisy(true_beats, fs)

        detectors = {
            "pan_tompkins": detect_beats(ecg, fs),
            "xqrs": second_detector(ecg, fs, record_name=name),
        }
        for detector_name, found in detectors.items():
            tp, fn, fp = count_matches(true_noisy, only_noisy(found, fs), fs)
            totals[detector_name][0] += tp
            totals[detector_name][1] += fn
            totals[detector_name][2] += fp

    results[level] = totals
    pt = totals["pan_tompkins"]
    print(f"SNR {str(level):>5}: Pan-Tompkins Se = {sensitivity(pt[0], pt[1]):.3f}, "
          f"PPV = {ppv(pt[0], pt[2]):.3f}")

with open("results/step2_noise.csv", "w") as f:
    f.write("snr_db,detector,TP,FN,FP,Se,PPV\n")
    for level in levels:
        for detector_name, (tp, fn, fp) in results[level].items():
            f.write(f"{level},{detector_name},{tp},{fn},{fp},"
                    f"{sensitivity(tp, fn):.4f},{ppv(tp, fp):.4f}\n")


# ---------------------------------------------------------------------------
# 2. Figure: what the noise looks like (record 118, the same 6 seconds)
# ---------------------------------------------------------------------------
show_levels = ["clean", 12, 6, 0, -6]
start_second = 5 * 60 + 30          # 5 min 30 s: inside the first noisy part
fig, axes = plt.subplots(len(show_levels), 1, figsize=(10, 8), sharex=True)
for ax, level in zip(axes, show_levels):
    if level == "clean":
        ecg, true_beats, fs = load_record("118", "mitdb")
        label = "no noise"
    else:
        ecg, true_beats, fs = load_record(noisy_record_name("118", level), "nstdb")
        label = f"SNR {level} dB"
    start, end = int(start_second * fs), int((start_second + 6) * fs)
    found = detect_beats(ecg, fs)
    found_here = found[(found >= start) & (found < end)]
    true_here = true_beats[(true_beats >= start) & (true_beats < end)]
    ax.plot(np.arange(start, end) / fs, ecg[start:end], color="black", linewidth=0.7)
    ax.plot(true_here / fs, np.full(len(true_here), ecg[start:end].max()), "gv",
            label="doctors' label")
    ax.plot(found_here / fs, ecg[found_here], "ro", fillstyle="none", label="my detector")
    ax.set_title(label, fontsize=10, loc="left")
axes[0].legend(loc="upper right", fontsize=8)
axes[-1].set_xlabel("time (seconds)")
plt.tight_layout()
plt.savefig("figures/fig4_noise_examples.png", dpi=130)
plt.close()


# ---------------------------------------------------------------------------
# 3. Figure: Se and PPV against SNR
# ---------------------------------------------------------------------------
snr_axis = [30] + NOISE_LEVELS           # draw "clean" at x = 30 for the plot
tick_labels = ["clean"] + [str(s) for s in NOISE_LEVELS]

fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
for ax, metric in zip(axes, ["Se", "PPV"]):
    for detector_name, style in [("pan_tompkins", "o-"), ("xqrs", "s--")]:
        values = []
        for level in levels:
            tp, fn, fp = results[level][detector_name]
            values.append(sensitivity(tp, fn) if metric == "Se" else ppv(tp, fp))
        label = "Pan-Tompkins (mine)" if detector_name == "pan_tompkins" else "xqrs (wfdb)"
        ax.plot(snr_axis, values, style, label=label)
    ax.set_xticks(snr_axis)
    ax.set_xticklabels(tick_labels)
    ax.invert_xaxis()                   # more noise to the right
    ax.set_xlabel("SNR (dB)   ->  more noise")
    ax.set_title("Sensitivity" if metric == "Se" else "Positive predictive value")
    ax.grid(alpha=0.3)
axes[0].set_ylabel("score")
axes[0].legend()
plt.suptitle("Beat detection under electrode-motion noise (records 118 + 119, noisy minutes only)")
plt.tight_layout()
plt.savefig("figures/fig5_se_ppv_vs_snr.png", dpi=130)
plt.close()
print("Figures saved in figures/.")
