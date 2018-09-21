import matplotlib.pyplot as plt
import csv


def get_mean_blink_duration(blink_limits):
    tot_blink_duration = 0

    for index in range(len(blink_limits["start"])):
        tot_blink_duration += blink_limits["end"][index] - blink_limits["start"][index]

    if len(blink_limits) > 0:
        return tot_blink_duration / len(blink_limits)


def get_derivative(signal):
    return [(signal[i + 1] - signal[i]) for i in range(len(signal) - 1)]


def save_blink_distribution(csv_file, title, save_path=None, display_results=True):
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

    plt.hist(blink_amounts, bins=range(max_blinks))  # arguments are passed to np.histogram
    plt.title(title)
    plt.xlabel("Blink Amount per Video")
    fig = plt.ylabel("Count")
    if save_path is not None:
        fig.savefig(save_path + "/blink_amounts.eps")
    if display_results:
        plt.show()

    plt.hist(blink_durations, bins=list(range(41)))  # arguments are passed to np.histogram
    plt.title(title)
    plt.xlabel("Avg Blink Duration per Video")
    fig = plt.ylabel("Count")
    if save_path is not None:
        fig.savefig(save_path + "/blink_durations.eps")
    if display_results:
        plt.show()
