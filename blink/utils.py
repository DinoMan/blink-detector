import matplotlib.pyplot as plt
import matplotlib as mpl
from cycler import cycler
import csv
import numpy as np


def get_mean_blink_duration(blink_limits):
    tot_blink_duration = 0

    for index in range(len(blink_limits["start"])):
        tot_blink_duration += blink_limits["end"][index] - blink_limits["start"][index]

    if len(blink_limits) > 0:
        return tot_blink_duration / len(blink_limits)


def get_derivative(signal):
    return [(signal[i + 1] - signal[i]) for i in range(len(signal) - 1)]


def save_blink_distribution(csv_file, title, save_path=None, display_results=False, label=None, transparency=1.0,
                            legend=None, font_size=22):
    mpl.rcParams['axes.prop_cycle'] = cycler(color='bgrcmyk')

    blink_amounts = []
    blink_durations = []
    counter = 0
    previous_file = "N/A"

    with open(csv_file, newline='') as blink_data:
        blink_data_reader = csv.reader(blink_data, delimiter=',', quotechar='|')

        for row_no, row in enumerate(blink_data_reader):
            current_file = row[-1]
            duration = int(row[1]) - int(row[0])

            if row_no != 0 and current_file != previous_file:
                blink_amounts.append(counter)
                counter = 0

            if duration > 0:
                blink_durations.append(duration)
                counter += 1
            previous_file = current_file

    blink_amounts.append(counter)
    max_blinks = max(blink_amounts) + 2

    plt.figure(1)
    plt.tick_params(labelsize=font_size-2)
    plt.hist(blink_amounts, bins=range(max_blinks), label=label,
             alpha=transparency)  # arguments are passed to np.histogram
    plt.title(title)
    plt.xlabel("Blinks/Video", fontsize=font_size)
    plt.ylabel("# Videos", fontsize=font_size)
    plt.tight_layout()
    if legend is not None:
        plt.legend(loc=legend, fontsize=font_size-3)

    if save_path is not None:
        plt.savefig(save_path + "/blink_amounts.png")
    if display_results:
        plt.show()

    plt.figure(2)
    plt.tick_params(labelsize=font_size-2)
    plt.hist(blink_durations, bins=list(range(41)), label=label,
             alpha=transparency)  # arguments are passed to np.histogram
    plt.title(title)
    plt.xlabel("Blink Duration", fontsize=font_size)
    plt.ylabel("# Blinks", fontsize=font_size)
    plt.tight_layout()
    if legend is not None:
        plt.legend(loc=legend, fontsize=font_size-3)

    if save_path is not None:
        if transparency < 1.0:
            extension = ".png"
        else:
            extension = ".eps"

        plt.savefig(save_path + "/blink_durations" + extension)
    if display_results:
        plt.show()

    return np.mean(blink_amounts), np.mean(blink_durations)
