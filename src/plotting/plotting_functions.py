import numpy as np
import matplotlib.pyplot as plt
from global_ import *
import os


def plotter(title: str, mean_data: list, x_data: np.ndarray, legend: list, data_set_name: str = None,
            std_data: list = None, x_scale: str = "linear", y_scale: str = "linear",
            x_label: str = r"Number of samples", y_label: str = r"Time in seconds", alpha: float = 0.05):
    """Plots the results of the varying_samples_experiment and saves the result in the directory plots.

    Args:
        title: str
        mean_data: list
            A list of lists, every inner list contains the mean of a performance measure of an algorithm's performance.
        x_data: list
            A list containing the data corresponding to the x axis.
        legend: list
            A list contatining the names of the algorithms.
        data_set_name: str, Optional
        std_data: list, Optional
            The standard deviations corresponding to mean_data.
        x_scale: str, Optional
        y_scale: str, Optional
        x_label: str, Optional
        y_label: str, Optional
        alpha: float, Optional
    """
    plt.rcParams['font.family'] = 'DeJavu Serif'
    plt.rcParams['font.serif'] = ['Computer Modern Roman']
    plt.rcParams['text.usetex'] = True
    plt.rcParams['axes.linewidth'] = linewidth_

    fig = plt.figure()
    fig.set_figheight(8)
    fig.set_figwidth(8)

    x_max = 0
    for i in range(0, len(mean_data)):
        x_max = max(len(mean_data[i]), x_max)

    for i in range(0, len(mean_data)):
        mean = np.array(mean_data[i])
        if std_data is not None:
            std = np.array(std_data[i])
        samples = x_data[:len(mean)]

        plt.plot(samples, mean, color=colors_[i], marker=markers_[i], linewidth=linewidth_, markersize=markersize_)
        if std_data is not None:
            plt.fill_between(samples, mean - std, mean + std, alpha=alpha, color=colors_[i])

    if legend is not None:
        plt.legend(legend, fontsize=font_size_legend_, loc="upper left")
    plt.xlabel(x_label, fontsize=font_size_)
    plt.ylabel(y_label, fontsize=font_size_)
    plt.tick_params(axis='both', which='major', labelsize=font_size_, width=(linewidth_ / 1.25),
                    length=(2 * linewidth_),
                    direction='in', pad=(2.5 * linewidth_))
    plt.xscale(x_scale)
    plt.yscale(y_scale)
    plt.gca().xaxis.get_offset_text().set_fontsize(font_size_)
    plt.gca().yaxis.get_offset_text().set_fontsize(font_size_)

    plt.xlim([x_data[0], x_data[x_max - 1]])
    plt.tight_layout()

    if data_set_name is not None:
        file_name = "plots/" + title + "_" + data_set_name + ".png"
    else:
        file_name = "plots/" + title + ".png"

    if not os.path.exists('plots'):
        os.makedirs('plots')

    plt.savefig(file_name, dpi=300, bbox_inches='tight')
    plt.show()
