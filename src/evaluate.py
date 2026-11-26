"""
evaluate.py - check the detected beats against the doctors' labels.

A detected beat is CORRECT if it is within 150 ms of a labelled beat
(the usual rule from the ANSI/AAMI EC57 standard). Each labelled beat
can be matched only once.

    TP (true positive)  - real beat, and we found it
    FN (false negative) - real beat, but we missed it
    FP (false positive) - we found a "beat" that is not real

    Sensitivity (Se) = TP / (TP + FN)  -> of all real beats, how many did we find?
    PPV              = TP / (TP + FP)  -> of all our detections, how many were real?
"""
import numpy as np


def count_matches(true_beats, found_beats, fs, tolerance_seconds=0.150):
    """Return (TP, FN, FP) after matching found beats to true beats."""
    tolerance = int(tolerance_seconds * fs)
    found_beats = np.sort(found_beats)
    already_used = np.zeros(len(found_beats), dtype=bool)

    true_positives = 0
    for true_beat in true_beats:
        # Detections that lie within the tolerance window around this beat
        first = np.searchsorted(found_beats, true_beat - tolerance)
        last = np.searchsorted(found_beats, true_beat + tolerance, side="right")

        # Pick the closest one that has not been used yet
        best = None
        for i in range(first, last):
            if not already_used[i]:
                if best is None or abs(found_beats[i] - true_beat) < abs(found_beats[best] - true_beat):
                    best = i
        if best is not None:
            already_used[best] = True
            true_positives += 1

    false_negatives = len(true_beats) - true_positives
    false_positives = len(found_beats) - true_positives
    return true_positives, false_negatives, false_positives


def sensitivity(tp, fn):
    return tp / (tp + fn) if tp + fn > 0 else float("nan")


def ppv(tp, fp):
    return tp / (tp + fp) if tp + fp > 0 else float("nan")


def heart_rate(beats, fs):
    """
    Average heart rate (beats per minute) from a group of beats.
    Heart rate = 60 / (average time between beats in seconds).
    Needs at least 2 beats.
    """
    if len(beats) < 2:
        return float("nan")
    beats = np.sort(beats)
    average_rr_seconds = (beats[-1] - beats[0]) / (len(beats) - 1) / fs
    return 60 / average_rr_seconds


def beats_between(beats, start, end):
    """Only the beats with start <= position < end."""
    beats = np.asarray(beats)
    return beats[(beats >= start) & (beats < end)]
