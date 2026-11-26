"""
sqi.py - signal quality check (bSQI) and the second detector.

Idea: run TWO different beat detectors on the same piece of ECG.
  - On clean ECG they find the same beats   -> they agree    -> trust it
  - On noisy ECG each gets fooled differently -> they disagree -> reject it

bSQI = beats both detectors found / beats found by at least one of them
     = matched / (beats_A + beats_B - matched)

bSQI = 1.0 means perfect agreement, 0.0 means no agreement at all.
(Li, Mark & Clifford, 2008)
"""
import os

import numpy as np
from wfdb import processing

from .data import DATA_FOLDER
from .evaluate import count_matches


def second_detector(ecg, fs, record_name=None):
    """
    The second detector: xqrs from the wfdb library (a different method from
    Pan-Tompkins). It is slow, so we save its answer the first time and
    reuse it after that.
    """
    cache_file = None
    if record_name is not None:
        cache_folder = os.path.join(DATA_FOLDER, "xqrs_cache")
        os.makedirs(cache_folder, exist_ok=True)
        cache_file = os.path.join(cache_folder, record_name + ".npy")
        if os.path.exists(cache_file):
            return np.load(cache_file)

    beats = processing.xqrs_detect(sig=ecg, fs=fs, verbose=False)
    beats = np.array(beats, dtype=int)

    if cache_file is not None:
        np.save(cache_file, beats)
    return beats


def bsqi(beats_a, beats_b, fs):
    """Agreement between two detectors in one piece of ECG (0 to 1)."""
    if len(beats_a) == 0 and len(beats_b) == 0:
        return 0.0      # nobody found a beat: we cannot trust this piece
    matched, _, _ = count_matches(beats_a, beats_b, fs)
    return matched / (len(beats_a) + len(beats_b) - matched)


def windows(signal_length, fs, window_seconds=10):
    """Cut the recording into 10-second pieces. Returns (start, end) pairs."""
    width = int(window_seconds * fs)
    return [(start, start + width) for start in range(0, signal_length - width + 1, width)]
