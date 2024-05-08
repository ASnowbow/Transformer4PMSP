import datetime
import numpy as np
import matplotlib.pyplot as plt

"""Env Config"""

num_batch = 5
first_arrival_time = 20
interval_time = 10
last_arrival_time = (num_batch - 1) * interval_time + first_arrival_time
num_job_in_batch = 10

arrive_time_lst = list(range(first_arrival_time, last_arrival_time + 1, interval_time))

# new jobs of each release time
num_init_job = 80

N = num_init_job + num_batch * num_job_in_batch

# number of types of job and machine
max_job_family = 8
num_job_family = 8

# changing jobs need setup time
setup_time = 10

# coefficient of shorter processing time
high_spd_fkt = 0.80


# due date config
r = 0.1
R = 0.5

num_fast_machine = 6
num_slow_machine = 6
num_all_machine = num_fast_machine + num_slow_machine
num_machine_family = 2


MP = (N * 10 + (N + num_job_family) * 10 / 2) \
     / (num_fast_machine + num_slow_machine)
d_min = (1 - r - R / 2) * MP + setup_time
d_max = (1 - r + R / 2) * MP + setup_time

"""RL Config"""
input_size = 3 + num_machine_family + 2 * max_job_family
just_in_time_reward = 0
just_in_time_threshold = 1

setup_reward = 1
final_reward = 100

learning_rate_decay = 10
num_decay = 1

num_heads = 2
batch_size = 64
n_epochs = 4
alpha = 0.0001
num_episode = 12000


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

def experi_dir():
    """
    Naming Rules:
        date in format: Year-Month-Day
    """
    current_date = datetime.datetime.now().strftime('%Y%m%d')
    directory_name = f'TrainingResult\\{current_date}-{num_heads}Head'
    return directory_name
