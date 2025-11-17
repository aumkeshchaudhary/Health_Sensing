import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import read_psg_signal, bandpass_filter, read_flow_events

def load_participant(folder):
    na = read_psg_signal(os.path.join(folder, "nasal_airflow.txt"))
    th = read_psg_signal(os.path.join(folder, "thoracic_movement.txt"))
    spo2 = read_psg_signal(os.path.join(folder, "spo2.txt"))
    events = read_flow_events(os.path.join(folder, "flow_events.csv"))
    return na, th, spo2, events

def plot_signals(na, th, spo2, events, out_path, name):
    na_f = bandpass_filter(na, fs=32)
    th_f = bandpass_filter(th, fs=32)

    with PdfPages(out_path) as pdf:
        fig, axs = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

        axs[0].plot(na_f.index, na_f.values)
        axs[0].set_title("Nasal Airflow (Filtered)")

        axs[1].plot(th_f.index, th_f.values)
        axs[1].set_title("Thoracic Movement (Filtered)")

        axs[2].plot(spo2.index, spo2.values)
        axs[2].set_title("SpO2 (%)")

        for _, row in events.iterrows():
            for ax in axs:
                ax.axvspan(row['start'], row['end'], alpha=0.3, color='red')

        fig.suptitle(f"Participant: {name}")
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-name", required=True)
    args = parser.parse_args()

    folder = args.name
    name = os.path.basename(folder)

    na, th, spo2, events = load_participant(folder)

    os.makedirs("Visualizations", exist_ok=True)
    out_file = f"Visualizations/{name}_visualization.pdf"
    plot_signals(na, th, spo2, events, out_file, name)

    print(f"Saved: {out_file}")

if __name__ == "__main__":
    main()
