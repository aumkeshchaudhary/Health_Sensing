import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from sklearn.metrics import confusion_matrix

# Parsing PSG continuous signals
def read_psg_signal(path):
    """
    Reads files like:
    Data:
    dd.mm.yyyy HH:MM:SS,ms; value
    """
    with open(path, 'r') as f:
        lines = f.readlines()

    # Find the start of Data:
    data_idx = None
    for i, line in enumerate(lines):
        if line.strip().lower().startswith("data"):
            data_idx = i + 1
            break

    timestamps = []
    values = []

    for line in lines[data_idx:]:
        line = line.strip()
        if not line:
            continue
        if ";" not in line:
            continue

        ts_str, value_str = line.split(";")
        ts_str = ts_str.strip().replace(",", ".") 
        value_str = value_str.strip()

        try:
            ts = pd.to_datetime(ts_str, format="%d.%m.%Y %H:%M:%S.%f")
            val = float(value_str)
            timestamps.append(ts)
            values.append(val)
        except:
            continue

    return pd.Series(values, index=pd.to_datetime(timestamps))

# Parsing flow events
def read_flow_events(path):
    """
    Format:
    30.05.2024 23:48:45,119-23:49:01,408; 16;Hypopnea; N1
    """
    events = []
    with open(path, 'r') as f:
        for line in f:
            if "-" not in line or ";" not in line:
                continue

            try:
                time_part, rest = line.split(";", 1)
                start_str, end_str = time_part.split("-")

                # normalize timestamps
                start = pd.to_datetime(start_str.strip().replace(",", "."), format="%d.%m.%Y %H:%M:%S.%f")
                end = pd.to_datetime(
                    start_str[:10].replace(".", "-") + " " + end_str.strip().replace(",", "."),
                    format="%d-%m-%Y %H:%M:%S.%f"
                )

                parts = rest.split(";")
                event_name = parts[1].strip()  # example: Hypopnea

                events.append([start, end, event_name])
            except:
                continue

    df = pd.DataFrame(events, columns=["start", "end", "event"])
    return df

# Bandpass filter
def bandpass_filter(signal, fs, lowcut=0.1, highcut=0.8, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut/nyq, highcut/nyq], btype='band')
    return pd.Series(filtfilt(b, a, signal.values), index=signal.index)

# Window helper
def window_times(start_ts, window_sec=30, step_sec=15, total_seconds=None):
    end_ts = start_ts + pd.Timedelta(seconds=int(total_seconds))
    windows = []
    cur = start_ts
    while cur + pd.Timedelta(seconds=window_sec) <= end_ts:
        windows.append((cur, cur + pd.Timedelta(seconds=window_sec)))
        cur += pd.Timedelta(seconds=step_sec)
    return windows

# Metrics
def per_class_metrics(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    TP = np.diag(cm).astype(float)
    FP = cm.sum(axis=0) - TP
    FN = cm.sum(axis=1) - TP
    TN = cm.sum() - (TP + FP + FN)

    precision = TP / np.where((TP+FP)==0, 1, TP+FP)
    recall = TP / np.where((TP+FN)==0, 1, TP+FN)
    accuracy = (TP+TN) / np.where(cm.sum()==0, 1, cm.sum())
    specificity = TN / np.where((TN+FP)==0, 1, TN+FP)

    return {
        "confusion_matrix": cm,
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy,
        "sensitivity": recall,
        "specificity": specificity
    }
