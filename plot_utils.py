import matplotlib.pyplot as plt
import numpy as np

plt.style.use("seaborn-darkgrid")
plt.rcParams.update({"font.size": 15})

def plot_timesteps(
        values: np.ndarray,
        eval_freq: int,
        title: str,
        xlabel: str,
        ylabel: str,
        legend_name: str,
    ):
    """
    Plot values with respect to timesteps

    :param values (np.ndarray): numpy array of values to plot as y-values
    :param eval_freq (int): number of training iterations after which an evaluation is done
    :param title (str): name of algorithm
    :param xlabel (str): label of x-axis
    :param ylabel (str): label of y-axis
    :param legend_name (str): name of algorithm
    """
    plt.figure()
    plt.title(title)
    
    x_values = eval_freq + np.arange(len(values)) * eval_freq

    # plot means with respective standard deviation as shading
    plt.plot(x_values, values, label=f"{legend_name}")

    # set legend and axis-labels
    plt.legend(loc="best")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout(pad=0.3)


def plot_timesteps_shaded(
        means: np.ndarray,
        stds: np.ndarray,
        eval_freq: int,
        title: str,
        xlabel: str,
        ylabel: str,
        legend_name: str,
    ):
    """
    Plot mean and std-shading for values with respect to timesteps

    :param means (np.ndarray): numpy array of mean values to plot as y-values
    :param stds (np.ndarray): numpy array of standard deviations to plot as y-value shading
    :param eval_freq (int): number of training iterations after which an evaluation is done
    :param title (str): name of algorithm
    :param xlabel (str): label of x-axis
    :param ylabel (str): label of y-axis
    :param legend_name (str): name of algorithm
    """
    plt.figure()
    plt.title(title)

    x_values = eval_freq + np.arange(len(means)) * eval_freq

    # plot means with respective standard deviation as shading
    plt.plot(x_values, means, label=f"{legend_name}")
    plt.fill_between(
        x_values,
        np.clip(means - stds, 0, 1),
        np.clip(means + stds, 0, 1),
        alpha=0.3,
        antialiased=True,
    )

    # set legend and axis-labels
    plt.legend(loc="best")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout(pad=0.3)
