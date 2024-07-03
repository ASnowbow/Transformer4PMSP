import salabim as sim
import numpy as np

import config as cfg
from brain_Transformer import Agent
from components import jobs, new_jobs, machs
from env_PMSP import EnvPMSP

from config import plot_learning_curve
from config import plot_tardiness_curve
from config import experi_dir
import random

import os
import inspect

np.random.seed(1337)
random.seed(1337)
sim.random.seed(1337)

env = EnvPMSP(init_jobs=jobs, new_jobs=new_jobs, machs=machs)

experiment_dir = experi_dir()
figs_dir = os.path.join(experiment_dir, 'figs')
datas_dir = os.path.join(experiment_dir, 'datas')
param_dir = os.path.join(experiment_dir, 'param')

if not os.path.exists(figs_dir):
    os.makedirs(figs_dir)
if not os.path.exists(datas_dir):
    os.makedirs(datas_dir)
if not os.path.exists(param_dir):
    os.makedirs(param_dir)

agent = Agent(batch_size=cfg.batch_size,
              alpha=cfg.alpha, n_epochs=cfg.n_epochs,
              input_size=cfg.input_size
              )

reward_file = os.path.join(figs_dir, 'reward.png')
tardiness_file = os.path.join(figs_dir, 'tardiness.png')
info_file = os.path.join(experiment_dir, 'info.txt')

with open(info_file, "w") as file:
    lines = [
        "Network Information",
        inspect.getfile(Agent),
        "learning rate decay: " + str(cfg.learning_rate_decay),
        "number of decay: " + str(cfg.num_decay),
        "\n",
        "Environment Information",
        "Setup Reward: +=" + str(cfg.setup_reward),
        "Final Reward: +=" + str(cfg.final_reward),
        "Just in Time Threshold: +=" + str(cfg.just_in_time_threshold),
        "Just in Time Reward: +=" + str(cfg.just_in_time_reward),
        "\n",
        "HyperParameter",
        "Batch Size: " + str(cfg.batch_size),
        "Learning Rate: " + str(cfg.alpha),
        "Num_Epoch: " + str(cfg.n_epochs)
    ]
    file.writelines(line + "\n" for line in lines)

best_score = env.reward_range[0]
best_tardiness = -10000000
score_history = []
tardiness_history = []
learn_iters = 0
avg_score = 0
avg_tardiness = 0
n_steps = 0
learning_rate_flg = 0

late_list = np.zeros(cfg.num_episode)
tardiness_list = np.zeros(cfg.num_episode)

# agent.load_models()
for i in range(cfg.num_episode):
    decay_flg = False
    observation = env.reset()
    done = False
    score = 0
    while not done:
        action, prob, val = agent.choose_action(observation)
        observation_, reward, done, info = env.step(action)
        n_steps += 1
        score += reward
        agent.remember(observation, action, prob, val, reward, done)
        if n_steps % cfg.batch_size == 0:
            agent.learn()
            learn_iters += 1
        observation = observation_
    score_history.append(score)
    tardiness_history.append(env.total_tardiness)
    avg_score = np.mean(score_history[-100:])
    avg_tardiness = np.mean(tardiness_history[-100:])

    if avg_tardiness > best_tardiness:
        best_tardiness = avg_tardiness
        agent.save_models()

    """if avg_score > best_score:
        best_score = avg_score
        agent.save_models()"""

    print('episode', i, 'score %.2f' % score, 'avg tardiness %.2f' % avg_tardiness,
          'time_steps', n_steps, 'just_in_time', env.just_in_time_flg, 'total_tardiness', env.total_tardiness)
    # print('num_setup', env.num_setups, 'num_positive', env.num_posi, 'num_negative', env.neg)
    late_list[i] = env.late_count
    tardiness_list[i] = env.total_tardiness

    """
    LR decay through the process

    #
    if env.optimal_found:
        decay_flg = True
        print(" >>> optimal solution is found, learning rate decay start <<<")
    if decay_flg:
        agent.learning_rate_change()
        if i % 100 == 0:
            print("new learning rate: ", agent.lr)"""
    if learning_rate_flg < cfg.num_decay and env.optimal_found:
        learning_rate_flg += 1
        print(" >>> optimal solution is found, change learning rate <<<")
        print("old learning rate: ", agent.lr)
        agent.learning_rate_change()
        print("new learning rate: ", agent.lr)

    """
    learning rate decay every step after optimal found

    if env.optimal_found:
        learning_rate_flg += 1
    if learning_rate_flg > 0:
        agent.learning_rate_change()
        if i % 10 == 0:
            print("actor: ", agent.actor.alpha, "  ", "critic: ", agent.critic.alpha)
            print(learning_rate_flg)
    """

x = [i + 1 for i in range(len(score_history))]
plot_learning_curve(x, score_history, reward_file)
plot_tardiness_curve(x, tardiness_list, tardiness_file)

np.savetxt(os.path.join(datas_dir, 'tardiness.txt'), tardiness_list)
np.savetxt(os.path.join(datas_dir, 'reward.txt'), score_history)

print(late_list[:-10])
print("average number of tardy jobs", late_list[-10].mean())
print(tardiness_list[:-10])
print("average total tardiness", tardiness_list[:-10].mean())