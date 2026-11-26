"""
STEP 3: Can the system tell by itself which parts of the ECG to trust?

For every 10-second piece of the noisy recordings:
  1. Run my detector (Pan-Tompkins) and a second detector (xqrs)
  2. bSQI = how much the two agree (0 to 1)
  3. Heart rate from my detector vs true heart rate from the doctors' labels

Then for different bSQI thresholds (keep a piece only if bSQI >= threshold):
  - heart-rate error on the pieces we KEEP
  - % of time we keep

We use the whole recordings (clean and noisy minutes mixed), like a real
drive where the hands sometimes move and sometimes stay still.

Saves:
  results/step3_quality.csv        - error and % kept for every threshold and SNR
  figures/fig6_tradeoff.png        - MAIN FIGURE: heart-rate error vs % of time kept
  figures/fig7_bsqi_example.png    - bSQI over time for one noisy recording
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.data import NOISE_LEVELS, download_data, load_record, noisy_record_name
from src.evaluate import beats_between, heart_rate
from src.pan_tompkins import detect_beats
from src.sqi import bsqi, second_detector, windows

os.makedirs("results", exist_ok=True)
os.makedirs("figures", exist_ok=True)
download_data()

THRESHOLDS = [None, 0.80, 0.85, 0.90, 0.95]     # None = no quality check, keep all


# ---------------------------------------------------------------------------
# 1. For every 10-second piece: bSQI and heart-rate error
# ---------------------------------------------------------------------------
pieces = {}   # pieces[snr] = list of (bsqi, heart-rate error in bpm)

for snr in NOISE_LEVELS:
    pieces[snr] = []
    for base in ["118", "119"]:
        name = noisy_record_name(base, snr)
        ecg, true_beats, fs = load_record(name, "nstdb")
        mine = detect_beats(ecg, fs)
        other = second_detector(ecg, fs, record_name=name)

        for start, end in windows(len(ecg), fs, window_seconds=10):
            true_here = beats_between(true_beats, start, end)
            if len(true_here) < 2:
                continue                          # no reference heart rate here
            mine_here = beats_between(mine, start, end)
            other_here = beats_between(other, start, end)

            quality = bsqi(mine_here, other_here, fs)
            my_hr = heart_rate(mine_here, fs)
            true_hr = heart_rate(true_here, fs)
            # If my detector found fewer than 2 beats it gives no heart rate;
            # count that as a big error (the true rate itself).
            error = abs(my_hr - true_hr) if not np.isnan(my_hr) else true_hr
            pieces[snr].append((quality, error))


# ---------------------------------------------------------------------------
# 2. For every threshold: error on kept pieces and % of time kept
# ---------------------------------------------------------------------------
def summarise(piece_list, threshold):
    qualities = np.array([q for q, e in piece_list])
    errors = np.array([e for q, e in piece_list])
    keep = np.ones(len(errors), dtype=bool) if threshold is None else qualities >= threshold
    percent_kept = 100 * keep.mean()
    mean_error = errors[keep].mean() if keep.any() else float("nan")
    return mean_error, percent_kept


all_pieces = [p for snr in NOISE_LEVELS for p in pieces[snr]]

with open("results/step3_quality.csv", "w") as f:
    f.write("snr_db,threshold,mean_abs_hr_error_bpm,percent_time_kept\n")
    for snr in NOISE_LEVELS + ["all"]:
        piece_list = all_pieces if snr == "all" else pieces[snr]
        for threshold in THRESHOLDS:
            error, kept = summarise(piece_list, threshold)
            label = "none" if threshold is None else threshold
            f.write(f"{snr},{label},{error:.2f},{kept:.1f}\n")
            print(f"SNR {str(snr):>4}, threshold {str(label):>4}: "
                  f"HR error = {error:6.2f} bpm, time kept = {kept:5.1f}%")


# ---------------------------------------------------------------------------
# 3. MAIN FIGURE: heart-rate error vs % of time kept
# ---------------------------------------------------------------------------
plt.figure(figsize=(8, 5.5))
for snr in NOISE_LEVELS + ["all"]:
    piece_list = all_pieces if snr == "all" else pieces[snr]
    points = [summarise(piece_list, t) for t in THRESHOLDS]
    errors = [p[0] for p in points]
    kept = [p[1] for p in points]
    if snr == "all":
        plt.plot(kept, errors, "k-", linewidth=3, marker="o", label="all noise levels")
    else:
        plt.plot(kept, errors, marker="o", alpha=0.8, label=f"SNR {snr} dB")
    # Mark the "no quality check" point
    plt.plot(kept[0], errors[0], marker="x", color="black", markersize=10)

plt.gca().invert_xaxis()          # left = keep everything, right = keep less
plt.yscale("log")
plt.xlabel("% of time kept (stricter quality check  ->)")
plt.ylabel("mean heart-rate error on kept windows (bpm, log scale)")
plt.title("Trade-off: accuracy vs usable monitoring time\n"
          "(x = no quality check; then bSQI >= 0.80, 0.85, 0.90, 0.95)")
plt.grid(alpha=0.3, which="both")
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig("figures/fig6_tradeoff.png", dpi=130)
plt.close()


# ---------------------------------------------------------------------------
# 4. Figure: bSQI over time for one recording (118 at 6 dB)
# ---------------------------------------------------------------------------
name = noisy_record_name("118", 6)
ecg, true_beats, fs = load_record(name, "nstdb")
mine = detect_beats(ecg, fs)
other = second_detector(ecg, fs, record_name=name)
times, qualities = [], []
for start, end in windows(len(ecg), fs, window_seconds=10):
    times.append(start / fs / 60)
    qualities.append(bsqi(beats_between(mine, start, end), beats_between(other, start, end), fs))

plt.figure(figsize=(10, 3.5))
plt.step(times, qualities, where="post", color="black")
for minute in range(5, 31, 4):                     # shade the noisy minutes
    plt.axvspan(minute, min(minute + 2, 30.1), color="red", alpha=0.12)
plt.axhline(0.9, color="blue", linestyle="--", label="threshold 0.90")
plt.xlabel("time (minutes)   [red = noise added]")
plt.ylabel("bSQI")
plt.title("Record 118 at SNR 6 dB: the two detectors disagree when noise is added")
plt.legend(loc="lower left")
plt.tight_layout()
plt.savefig("figures/fig7_bsqi_example.png", dpi=130)
plt.close()
print("Figures saved in figures/.")
