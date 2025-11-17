#!/usr/bin/env python3
"""
Create sleep-stage dataset:
- 30s windows with 50% overlap
- Label windows using >50% stage overlap
- Works for all AP01–AP05 sleep profile formats
"""

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
import pandas as pd
from utils import read_psg_signal, window_times


SLEEP_LABELS = ["Wake", "REM", "N1", "N2", "N3"]


# ---------------------------------------------------------
# UNIVERSAL SLEEP PROFILE PARSER (WORKS FOR AP02!)
# ---------------------------------------------------------
def read_sleep_profile(path):
    """
    Handles:
    - CSV files with start,end,stage
    - Interval lines:    "start-end; stage"
    - Sampled lines:     "timestamp; stage" (AP02 format)
    """

    # ---------------------------------------------
    # TRY CSV FORMAT FIRST
    # ---------------------------------------------
    try:
        df = pd.read_csv(path)
        cols = [c.lower() for c in df.columns]

        if "start" in cols and "end" in cols:
            start_col = df.columns[cols.index("start")]
            end_col   = df.columns[cols.index("end")]
            stage_col = df.columns[cols.index("stage")] if "stage" in cols else df.columns[-1]

            df["start"] = pd.to_datetime(df[start_col], errors="coerce")
            df["end"]   = pd.to_datetime(df[end_col],   errors="coerce")
            df["stage"] = df[stage_col].astype(str)

            df = df[["start", "end", "stage"]].dropna()
            if len(df) > 0:
                return df

    except Exception:
        pass  # fallback to manual parser

    # ---------------------------------------------
    # MANUAL LINE PARSER FOR AP02 / AP03 / AP04
    # ---------------------------------------------
    events = []
    with open(path, "r") as f:
        lines = f.readlines()

    data_started = False

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # AP02 header contains:
        # Rate: 30 s
        if "Rate:" in line:
            data_started = True
            continue

        if "Data:" in line:
            data_started = True
            continue

        # ignore header junk until real data
        if not data_started:
            # But AP02 does NOT show "Data:", so detect stage lines:
            if ";" in line:
                data_started = True
            else:
                continue

        # ---------------------------------------------
        # TYPE A — interval:  "start-end; stage"
        # ---------------------------------------------
        if ";" in line and "-" in line:
            try:
                timepart, stage = line.split(";", 1)
                start_s, end_s = timepart.split("-", 1)

                start_s = start_s.replace(",", ".")
                end_s   = end_s.replace(",", ".")

                start = pd.to_datetime(start_s, errors="coerce",
                                       format="%d.%m.%Y %H:%M:%S.%f")
                end   = pd.to_datetime(end_s, errors="coerce",
                                       format="%d.%m.%Y %H:%M:%S.%f")

                if pd.notna(start) and pd.notna(end):
                    stage = stage.strip().split()[0]
                    events.append([start, end, stage])
                    continue

            except:
                pass

        # ---------------------------------------------
        # TYPE B — AP02 format:
        #           "timestamp; Stage"
        # Duration = 30 seconds from Rate: 30s
        # ---------------------------------------------
        if ";" in line and "-" not in line:
            try:
                time_s, stage_s = line.split(";", 1)
                time_s = time_s.replace(",", ".")

                start = pd.to_datetime(time_s,
                                       errors="coerce",
                                       format="%d.%m.%Y %H:%M:%S.%f")
                if pd.isna(start):
                    continue

                end = start + pd.Timedelta(seconds=30)
                stage = stage_s.strip().split()[0]

                events.append([start, end, stage])
                continue

            except:
                pass

    if len(events) == 0:
        raise RuntimeError(f"Could not parse sleep profile: {path}")

    df = pd.DataFrame(events, columns=["start", "end", "stage"])
    return df


# ---------------------------------------------------------
# LABEL WINDOWS BY SLEEP STAGE
# ---------------------------------------------------------
def label_window_by_sleep(ws, we, sleep_df):
    duration = (we - ws).total_seconds()

    for _, row in sleep_df.iterrows():
        s = row["start"]
        e = row["end"]

        latest_start = max(ws, s)
        earliest_end = min(we, e)

        overlap_td = earliest_end - latest_start
        overlap = overlap_td.total_seconds()

        if overlap > 0 and (overlap / duration) > 0.5:
            return row["stage"]

    return None


# ---------------------------------------------------------
# PROCESS ONE PARTICIPANT
# ---------------------------------------------------------
def process_participant(folder):
    na = read_psg_signal(os.path.join(folder, "nasal_airflow.txt"))
    th = read_psg_signal(os.path.join(folder, "thoracic_movement.txt"))
    spo2 = read_psg_signal(os.path.join(folder, "spo2.txt"))
    sleep = read_sleep_profile(os.path.join(folder, "sleep_profile.csv"))

    start = min(na.index.min(), th.index.min())
    total_seconds = (na.index.max() - na.index.min()).total_seconds()

    windows = window_times(start, window_sec=30, step_sec=15, total_seconds=total_seconds)

    X = []
    Y = []
    META = []

    for ws, we in windows:
        na_seg = na.loc[ws:we].values
        th_seg = th.loc[ws:we].values
        spo2_seg = spo2.loc[ws:we].values

        if len(na_seg)==0 or len(th_seg)==0 or len(spo2_seg)==0:
            continue

        # resample spo2 to airflow length
        target_len = len(na_seg)
        spo2_interp = np.interp(
            np.linspace(0,1,target_len),
            np.linspace(0,1,len(spo2_seg)),
            spo2_seg
        )

        arr = np.stack([na_seg, th_seg, spo2_interp], axis=0).astype(np.float32)

        label = label_window_by_sleep(ws, we, sleep)
        if label is None:
            continue

        X.append(arr)
        Y.append(label)
        META.append([os.path.basename(folder), ws, we])

    return X, Y, META


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-in_dir", required=True)
    parser.add_argument("-out_dir", required=True)
    args = parser.parse_args()

    folders = sorted([os.path.join(args.in_dir,x)
                      for x in os.listdir(args.in_dir)
                      if os.path.isdir(os.path.join(args.in_dir,x))])

    ALL_X, ALL_Y, ALL_META = [], [], []

    for f in folders:
        print("Processing", f)
        X, Y, META = process_participant(f)
        ALL_X += X
        ALL_Y += Y
        ALL_META += META

    # map stage strings → integers
    unique_stages = sorted(list(set(ALL_Y)))
    mapping = {s:i for i,s in enumerate(unique_stages)}

    y_ids = np.array([mapping[s] for s in ALL_Y], dtype=np.int64)
    X_arr = np.array(ALL_X)

    os.makedirs(args.out_dir, exist_ok=True)

    np.savez_compressed(os.path.join(args.out_dir,"sleep_windows.npz"),
                        X=X_arr, y=y_ids)

    df = pd.DataFrame(ALL_META, columns=["participant","start","end"])
    df["label"] = ALL_Y
    df.to_csv(os.path.join(args.out_dir,"sleep_labels.csv"), index=False)

    with open(os.path.join(args.out_dir,"sleep_label_mapping.txt"), "w") as f:
        for k,v in mapping.items():
            f.write(f"{v}\t{k}\n")

    print("\nSaved sleep dataset:", X_arr.shape, "labels:", len(y_ids))
    print("Stages:", mapping)


if __name__ == "__main__":
    main()