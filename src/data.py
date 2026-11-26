"""
data.py - get the ECG recordings and the doctors' beat labels.

We use two free databases from PhysioNet:
  1. MIT-BIH Arrhythmia Database (mitdb): 48 recordings, 30 minutes each,
     360 samples per second, every beat marked by cardiologists.
  2. MIT-BIH Noise Stress Test Database (nstdb): records 118 and 119 with
     real electrode-motion noise added at different strengths (SNR).
"""
import os

import numpy as np
import wfdb

# Folder where the downloaded recordings are kept
PROJECT_FOLDER = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FOLDER = os.path.join(PROJECT_FOLDER, "data")

# The 48 recordings of the MIT-BIH Arrhythmia Database
MITDB_RECORDS = [
    "100", "101", "102", "103", "104", "105", "106", "107", "108", "109",
    "111", "112", "113", "114", "115", "116", "117", "118", "119", "121",
    "122", "123", "124", "200", "201", "202", "203", "205", "207", "208",
    "209", "210", "212", "213", "214", "215", "217", "219", "220", "221",
    "222", "223", "228", "230", "231", "232", "233", "234",
]

# Noise levels in the stress-test database (in dB). "_6" means -6 dB.
NOISE_LEVELS = [24, 18, 12, 6, 0, -6]


def noisy_record_name(base_record, snr):
    """Name of a stress-test record, for example ('118', 6) -> '118e06'."""
    if snr < 0:
        return f"{base_record}e_{abs(snr)}"
    return f"{base_record}e{snr:02d}"


NSTDB_RECORDS = [noisy_record_name(base, snr) for base in ["118", "119"] for snr in NOISE_LEVELS]

# Labels that mean "this is a heartbeat". Other labels in the files mark
# things like rhythm changes or noise, and are not beats.
BEAT_LABELS = set("NLRBAaJSVrFejnE/fQ?")


def download_data():
    """Download both databases once. Later calls do nothing."""
    mitdb_folder = os.path.join(DATA_FOLDER, "mitdb")
    nstdb_folder = os.path.join(DATA_FOLDER, "nstdb")

    if not os.path.exists(os.path.join(mitdb_folder, "234.dat")):
        print("Downloading MIT-BIH Arrhythmia Database (about 100 MB)...")
        wfdb.dl_database("mitdb", dl_dir=mitdb_folder, records=MITDB_RECORDS)

    if not os.path.exists(os.path.join(nstdb_folder, "119e_6.dat")):
        print("Downloading the noisy stress-test records...")
        wfdb.dl_database("nstdb", dl_dir=nstdb_folder, records=NSTDB_RECORDS)


def load_record(record_name, database="mitdb"):
    """
    Load one recording.

    Returns:
        ecg         - the signal: a long list of voltages (in mV), lead MLII
        true_beats  - sample numbers where the doctors marked a heartbeat
        fs          - sampling rate (samples per second)
    """
    path = os.path.join(DATA_FOLDER, database, record_name)

    record = wfdb.rdrecord(path)
    # Use lead MLII. It is usually the first channel, but not always
    # (in record 114 the two channels are stored the other way round).
    channel = record.sig_name.index("MLII") if "MLII" in record.sig_name else 0
    ecg = record.p_signal[:, channel]
    fs = int(record.fs)

    labels = wfdb.rdann(path, "atr")
    true_beats = [sample for sample, label in zip(labels.sample, labels.symbol)
                  if label in BEAT_LABELS]

    return ecg, np.array(true_beats), fs
