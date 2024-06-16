import numpy as np
import matplotlib.pyplot as plt


def plot_learning_curve(x, scores, figure_file):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-20):(i+1)])
    plt.figure(1)
    plt.plot(x, running_avg)
    plt.title('Running average of previous 100 scores')
    plt.savefig(figure_file)


def plot_tardiness_curve(x, scores, figure_file):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = -np.mean(scores[max(0, i-20):(i+1)])
    plt.figure(2)
    plt.plot(x, running_avg)
    plt.title('Running average of previous 100 scores')
    plt.savefig(figure_file)


def plot_each_step(x, scores, figure_file):
    plt.figure(3)
    plt.plot(x, scores)
    plt.title('tardiness of each step')
    plt.savefig(figure_file)


