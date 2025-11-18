import argparse
import os
import numpy as np
import pandas as pd

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import read_psg_signal, read_flow_events, bandpass_filter, window_times

LABELS = ["Hypopnea", "Obstructive Apnea", "Normal"]


def label_window(start, end, events):
    """Label window based on >50% overlap with any event."""
    duration = (end - start).total_seconds()
    
    for _, row in events.iterrows():
        overlap_start = max(start, row['start'])
        overlap_end = min(end, row['end'])
        overlap = (overlap_end - overlap_start).total_seconds()
        
        if overlap > duration * 0.5:
            if row['event'] in LABELS:
                return row['event']
    
    return "Normal"


def process_participant(folder):
    """Load, filter, window, and label one participant."""
    print("Processing:", folder)

    # PSG signals
    na = read_psg_signal(os.path.join(folder, "nasal_airflow.txt"))
    th = read_psg_signal(os.path.join(folder, "thoracic_movement.txt"))
    spo2 = read_psg_signal(os.path.join(folder, "spo2.txt"))

    # Flow events
    events = read_flow_events(os.path.join(folder, "flow_events.csv"))

    # filter (breathing band)
    na_f = bandpass_filter(na, fs=32)
    th_f = bandpass_filter(th, fs=32)

    # window generation
    start_ts = min(na_f.index.min(), th_f.index.min())
    total_seconds = (na_f.index.max() - na_f.index.min()).total_seconds()

    windows = window_times(start_ts, window_sec=30, step_sec=15, total_seconds=total_seconds)

    X = []
    Y = []
    META = []

    for (ws, we) in windows:

        na_seg = na_f.loc[ws:we].values
        th_seg = th_f.loc[ws:we].values
        spo2_seg = spo2.loc[ws:we].values

        if len(na_seg) == 0 or len(th_seg) == 0 or len(spo2_seg) == 0:
            continue

        # Resample SpO2 to match breathing sample count
        target_len = len(na_seg)
        spo2_interp = np.interp(
            np.linspace(0, 1, target_len),
            np.linspace(0, 1, len(spo2_seg)),
            spo2_seg
        )

        arr = np.stack([na_seg, th_seg, spo2_interp], axis=0).astype(np.float32)

        lbl = label_window(ws, we, events)

        X.append(arr)
        Y.append(lbl)
        META.append([os.path.basename(folder), ws, we])

    return X, Y, META


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-in_dir", required=True)
    parser.add_argument("-out_dir", required=True)
    args = parser.parse_args()

    folders = [
        os.path.join(args.in_dir, f)
        for f in os.listdir(args.in_dir)
        if os.path.isdir(os.path.join(args.in_dir, f))
    ]

    ALL_X, ALL_Y, ALL_META = [], [], []

    for folder in folders:
        X, Y, META = process_participant(folder)
        ALL_X += X
        ALL_Y += Y
        ALL_META += META

    label_to_id = {label: i for i, label in enumerate(LABELS)}
    Y_ids = np.array([label_to_id[l] for l in ALL_Y])

    X_arr = np.array(ALL_X)

    os.makedirs(args.out_dir, exist_ok=True)

    np.savez_compressed(
        os.path.join(args.out_dir, "breathing_windows.npz"),
        X=X_arr, y=Y_ids
    )

    df = pd.DataFrame(ALL_META, columns=["participant", "start", "end"])
    df["label"] = ALL_Y
    df.to_csv(os.path.join(args.out_dir, "breathing_labels.csv"), index=False)

    print("Dataset saved successfully!")


if __name__ == "__main__":
    main()
