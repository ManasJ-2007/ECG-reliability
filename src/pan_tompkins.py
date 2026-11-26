"""
pan_tompkins.py - find heartbeats (R-peaks) in an ECG, written from scratch.

Based on: Pan & Tompkins, "A real-time QRS detection algorithm", 1985.

The idea in five steps:
  1. Band-pass filter 5-15 Hz  -> keep the sharp QRS, remove slow drift and fast fuzz
  2. Derivative                -> highlight the steep slopes of the QRS
  3. Square                    -> make everything positive, make big slopes much bigger
  4. Moving average (150 ms)   -> turn each QRS into one smooth "energy bump"
  5. Adaptive threshold        -> decide which bumps are real beats
Then we go back to the filtered ECG to find the exact top of each beat (the R-peak).
"""
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks


# ---------------------------------------------------------------------------
# Steps 1 to 4: turn the ECG into an "energy" signal with one bump per beat
# ---------------------------------------------------------------------------

def bandpass_filter(ecg, fs, low_hz=5, high_hz=15):
    """Step 1: keep only frequencies between 5 and 15 Hz (where the QRS lives)."""
    nyquist = fs / 2
    b, a = butter(2, [low_hz / nyquist, high_hz / nyquist], btype="band")
    # filtfilt runs the filter forwards and then backwards, so the beats
    # do not get shifted in time (zero phase). This needs the whole
    # recording, so it only works offline, not in real time.
    return filtfilt(b, a, ecg)


def derivative(signal, fs):
    """Step 2: how fast the signal is changing (the slope)."""
    # Five-point derivative from the original paper:
    # slope[n] = (2*x[n+1] + x[n+2] - x[n-2] - 2*x[n-1]) * fs / 8
    weights = np.array([1, 2, 0, -2, -1]) * (fs / 8)
    return np.convolve(signal, weights, mode="same")


def square(signal):
    """Step 3: square every value. Negative slopes become positive too."""
    return signal ** 2


def moving_average(signal, fs, window_seconds=0.150):
    """Step 4: average over 150 ms, so each QRS becomes one smooth bump."""
    width = int(window_seconds * fs)
    window = np.ones(width) / width
    return np.convolve(signal, window, mode="same")


# ---------------------------------------------------------------------------
# Step 5: decide which bumps are real heartbeats
# ---------------------------------------------------------------------------

def pick_beats(energy, fs):
    """
    Go through the bumps one by one and decide: beat or noise?

    We keep two running levels:
      signal_level - how tall a typical real beat bump is
      noise_level  - how tall a typical noise bump is
    A bump counts as a beat if it is taller than a threshold that sits
    between these two levels. Both levels keep updating as we go,
    so the threshold adapts when the signal gets bigger or smaller.
    """
    refractory = int(0.200 * fs)   # the heart cannot beat twice within 200 ms

    # All bumps in the energy signal, at least 200 ms apart
    bumps, _ = find_peaks(energy, distance=refractory)

    # Learning phase: guess the starting levels from the first 2 seconds
    start = energy[: 2 * fs]
    signal_level = 0.25 * np.max(start)
    noise_level = 0.5 * np.mean(start)

    beats = []          # positions of accepted beats
    rr_intervals = []   # time between recent beats (in samples)

    for bump in bumps:
        height = energy[bump]
        threshold = noise_level + 0.25 * (signal_level - noise_level)

        # --- Search-back: did we miss a beat? ---------------------------------
        # If the gap since the last beat is much longer than usual (1.66 times
        # the average RR), we probably missed a smaller beat. Look back for the
        # tallest bump in the gap and accept it with a lower threshold.
        if len(beats) > 0 and len(rr_intervals) > 0:
            average_rr = np.mean(rr_intervals[-8:])
            gap = bump - beats[-1]
            if gap > 1.66 * average_rr:
                in_gap = bumps[(bumps > beats[-1] + refractory) & (bumps < bump - refractory)]
                if len(in_gap) > 0:
                    tallest = in_gap[np.argmax(energy[in_gap])]
                    if energy[tallest] > threshold / 2:
                        rr_intervals.append(tallest - beats[-1])
                        beats.append(tallest)
                        signal_level = 0.25 * energy[tallest] + 0.75 * signal_level

        # --- Too close to the previous beat? Then it cannot be a new beat ------
        if len(beats) > 0 and bump - beats[-1] < refractory:
            continue

        # --- Normal decision ---------------------------------------------------
        if height > threshold:
            if len(beats) > 0:
                rr_intervals.append(bump - beats[-1])
            beats.append(bump)
            signal_level = 0.125 * height + 0.875 * signal_level   # update slowly
        else:
            noise_level = 0.125 * height + 0.875 * noise_level

    return np.array(beats, dtype=int)


# ---------------------------------------------------------------------------
# Step 6: find the exact R-peak near each detected bump
# ---------------------------------------------------------------------------

def find_r_peaks(filtered_ecg, beats, fs, search_seconds=0.100):
    """
    The energy bump is smooth, so its top is only roughly at the beat.
    Look at the filtered ECG within 100 ms on each side and take the point
    with the biggest size (positive or negative, because some leads show
    the QRS pointing downwards).
    """
    half_window = int(search_seconds * fs)
    r_peaks = []
    for beat in beats:
        left = max(beat - half_window, 0)
        right = min(beat + half_window, len(filtered_ecg))
        piece = np.abs(filtered_ecg[left:right])
        r_peaks.append(left + np.argmax(piece))
    return np.unique(r_peaks)


# ---------------------------------------------------------------------------
# Everything together
# ---------------------------------------------------------------------------

def detect_beats(ecg, fs):
    """Run all the steps. Returns the sample numbers of the detected R-peaks."""
    filtered = bandpass_filter(ecg, fs)
    slope = derivative(filtered, fs)
    squared = square(slope)
    energy = moving_average(squared, fs)

    beats = pick_beats(energy, fs)
    return find_r_peaks(filtered, beats, fs)


def all_stages(ecg, fs):
    """Same as detect_beats, but also returns every middle step (for plots)."""
    filtered = bandpass_filter(ecg, fs)
    slope = derivative(filtered, fs)
    squared = square(slope)
    energy = moving_average(squared, fs)
    beats = find_r_peaks(filtered, pick_beats(energy, fs), fs)
    return {"raw": ecg, "filtered": filtered, "slope": slope,
            "squared": squared, "energy": energy, "beats": beats}
