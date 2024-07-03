import salabim as sim
import gym
from gym import spaces
import numpy as np

import config as config
from Brain_Validation import Agent
import timeit
from config import experi_dir
import random
import pandas as pd

import os
import inspect

np.random.seed(1337)
random.seed(1337)
sim.random.seed(1337)


# one hot function
def one_hot(a, num_type):
    b = np.zeros(num_type)
    b[a - 1] = 1
    return b


def jobs_in_matrix(job_batch, num_jobs):
    job_time = np.zeros(num_jobs * 2).reshape(num_jobs, 2)
    job_type = np.zeros(num_jobs * MAX_Job_Type).reshape(num_jobs, MAX_Job_Type)

    for i in range(num_jobs):
        job_time[i] = [job_batch[i].prc_time / d_max, job_batch[i].due_date / d_max]
        job_type[i] = one_hot(job_batch[i].label, MAX_Job_Type)

    job_state = np.hstack((job_time, job_type))
    return job_state


mean_lst = []
std_lst = []
glb_time_lst = []

N_update = 64
batch_size = 64
n_epochs = 1
alpha = 0.0001
n_games = 1
state_dim = 6

""" due date configuration """
r = config.r
R = config.R

agent = Agent(batch_size=batch_size,
              alpha=alpha, n_epochs=n_epochs,
              input_size=config.Input_Size
              )


""" new job arrival configuration """
NUM_Batch = config.NUM_Batch
First_Arr = 20
Interval = 10
Last_Arr = (NUM_Batch - 1) * Interval + First_Arr
NUM_Job_InBatch = config.num_new_jobs_per_batch
ArriveTime_lst = list(range(First_Arr, Last_Arr + 1, Interval))

""" job configuration(number & type) """
NUM_0_job = config.NUM_Init_Job
N = NUM_0_job + NUM_Batch * NUM_Job_InBatch
MAX_Job_Type = config.MAX_Job_Type
NUM_Job_Type = config.NUM_Families
# changing jobs need setup time
SETUP_Time = 10

""" machine configuration(speed & type) """
NUM_Mach_Type = config.NUM_Mach_Type
# number of MachA and MachB
NUM_A_Mach = config.NUM_A_Mach
NUM_B_Mach = config.NUM_B_Mach
NUM_Machs = NUM_A_Mach + NUM_B_Mach
# coefficient of shorter processing time
HE_Fkt = 0.80


class Env_4_RNN(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 2
    }

    def __init__(self):
        self.sim_env = sim.Environment(trace=False)
        self.action_space = spaces.Discrete(999)
        self.observation_space = spaces.Box(low=-1.0, high=500.0, shape=(config.Input_Size,), dtype=np.float32)

        self.job_choose = None

        self.job_state = None
        self.state = None
        self.job_1_state = None
        self.job_2_state = None

        self.mach_state = None
        self.glb_state = None

        self.counts = None
        self.m = None  # the idle machine
        self.mach_time = None
        self.rls = None  # flag of the release time

        self.late_count = None
        self.total_tardiness = None

        self.num_setups = None
        self.num_posi = None
        self.neg = None
        self.optimal_found = None

    def _new_jobs_arrival(self, job_batch, num_jobs):
        self.rls += 1
        for job in job_batch:
            self.job_choose.append(job)
        job_state = jobs_in_matrix(job_batch, num_jobs)
        self.job_state = np.vstack((self.job_state, job_state))

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        reward = 0
        n = action
        self.job_choose[n].enter(machs[self.m].wait_line)

        # if this job is the first job of this machine, an additional Setup_time is needed
        if len(machs[self.m].wait_line) == 1:
            if machs[self.m].mch_typ == 2:  # normal machine
                machs[self.m].acm_time += self.job_choose[n].prc_time  # + SETUP_Time
            if machs[self.m].mch_typ == 1:  # high-efficiency machine
                machs[self.m].acm_time += self.job_choose[n].prc_time * HE_Fkt  # + SETUP_Time

        elif self.job_choose[n].label == machs[self.m].lst_job:
            if machs[self.m].mch_typ == 2:  # normal machine
                machs[self.m].acm_time += self.job_choose[n].prc_time
            if machs[self.m].mch_typ == 1:
                machs[self.m].acm_time += self.job_choose[n].prc_time * HE_Fkt

            # reward += config.setup_reward
            self.num_posi += 1

        elif self.job_choose[n].label != machs[self.m].lst_job:
            if machs[self.m].mch_typ == 2:  # normal machine
                machs[self.m].acm_time += (self.job_choose[n].prc_time + SETUP_Time)
            if machs[self.m].mch_typ == 1:  # high-efficiency machine
                machs[self.m].acm_time += (self.job_choose[n].prc_time * HE_Fkt + SETUP_Time)

            self.num_setups += 1

            for job in self.job_choose:
                if job.label == machs[self.m].lst_job:
                    # reward -= config.setup_reward
                    self.neg += 1
                    break
                else:
                    pass

        machs[self.m].lst_job = self.job_choose[n].label
        machs[self.m].num_jobs += 1

        # the family of the job being processed by the machine m
        self.glb_state[self.m][-(MAX_Job_Type + config.NUM_Mach_Type):-config.NUM_Mach_Type] \
            = one_hot(machs[self.m].lst_job, MAX_Job_Type)
        self.glb_state[self.m][2+MAX_Job_Type] = machs[self.m].acm_time / d_max

        """
        reward
        """
        # the goal is to minimize total tardiness
        # so if a job isn't finished on time, the agent will get a negative reward
        if (self.job_choose[n].due_date - machs[self.m].acm_time) < 0:
            self.total_tardiness += (self.job_choose[n].due_date - machs[self.m].acm_time)  # negative
            self.late_count += 1
            # reward += (self.job_choose[n].due_date - machs[self.m].acm_time)
        # elif (self.job_choose[n].due_date - machs[self.m].acm_time) < config.just_in_time_threshold:
            # reward += config.just_in_time_reward

        """
        delete the chosen job from the state 
        """
        del self.job_choose[n]
        self.state = np.delete(self.state, n, 0)
        self.job_state = self.state[:-config.NUM_Machs, :(MAX_Job_Type + 2)]

        """
        find the next idle machine according to the accumulated time of each machine
        """
        for i in range(len(machs)):
            self.mach_time[i] = machs[i].acm_time
        self.m = int(np.argmin(self.mach_time))
        if config.global_info:
            self.glb_state[:, :2 + MAX_Job_Type] = 0
            self.glb_state[self.m][:2 + MAX_Job_Type] = 1

        """arrive of new jobs"""
        if self.rls < NUM_Batch and machs[self.m].acm_time >= ArriveTime_lst[self.rls]:
            self._new_jobs_arrival(new_jobs[self.rls], len(new_jobs[self.rls]))

        """change of state"""
        if machs[self.m].acm_time != 0:
            acm_time = np.array(machs[self.m].acm_time / d_max).reshape(1, 1)
            lst_job = np.array(one_hot(machs[self.m].lst_job, MAX_Job_Type)).reshape(1, MAX_Job_Type)
            mach_type = np.array(one_hot(machs[self.m].mch_typ, NUM_Mach_Type)).reshape(1, NUM_Mach_Type)
        else:
            acm_time = np.zeros(1).reshape(1, 1)
            lst_job = np.zeros(MAX_Job_Type).reshape(1, MAX_Job_Type)
            mach_type = np.array(one_hot(machs[self.m].mch_typ, NUM_Mach_Type)).reshape(1, NUM_Mach_Type)

        self.mach_state = np.hstack((acm_time, lst_job, mach_type))
        # the number of rows of the machine state should be same as the job state
        self.mach_state = np.tile(self.mach_state, (len(self.job_state), 1))

        self.state = np.hstack((self.job_state, self.mach_state))
        self.state = np.vstack((self.state, self.glb_state))

        """
        counts
        """
        self.counts += 1

        """
        done
        """
        done = len(self.job_choose) == 0

        if done:

            if self.total_tardiness > -0.1:
                self.optimal_found = True
                reward += config.final_reward

            for mach in machs:
                mach.lst_job = -1
            self.sim_env.run()

        return self.state, reward, done, {}

    def reset(self):
        self.rls = 0
        self.counts = 0
        # job list to choose
        self.job_choose = jobs[:]
        self.late_count = 0
        self.total_tardiness = 0
        self.num_setups = 0
        self.num_posi = 0
        self.neg = 0
        self.optimal_found = False

        # reset the attribute
        for mach in machs:
            mach.acm_time = 0
            mach.num_jobs = 0
            mach.wait_line.clear()
            mach.lst_job = -1

        """First convert the jobs' info into a matrix"""
        # the number of rows of state is the number of jobs
        # first put processing time and due_date in the state
        # then turn the job_type into one-hot encoding
        self.job_state = jobs_in_matrix(jobs_0, NUM_0_job)

        """Then convert the machines' info into a matrix 
        All machines are idle at beginning, the 0th machine is selected to be the first machine to be scheduled"""
        self.m = 0

        # first generate the vector of the accumulated time of the machine
        # at first this vector is full of 0
        acm_time = np.zeros(NUM_0_job).reshape(NUM_0_job, 1)

        # the next vector is for the last job of this machine
        ltz_job = np.zeros(NUM_0_job * MAX_Job_Type).reshape(NUM_0_job, MAX_Job_Type)

        # then the vector of the type of the machine
        mach_type = np.zeros(NUM_0_job * NUM_Mach_Type).reshape(NUM_0_job, NUM_Mach_Type)
        for i in range(NUM_0_job):
            mach_type[i] = one_hot(machs[self.m].mch_typ, NUM_Mach_Type)

        # at last stack the vectors of the idle machine together
        self.mach_state = np.hstack((acm_time, ltz_job, mach_type))

        """ Merge to get the global matrix"""
        self.state = np.hstack((self.job_state, self.mach_state))

        self.mach_time = np.zeros(len(machs))
        for i in range(len(machs)):
            self.mach_time[i] = machs[i].acm_time
        self.m = 0

        """ALL Machine State"""
        self.glb_state = np.zeros(config.NUM_Machs * config.Input_Size).reshape(config.NUM_Machs, config.Input_Size)
        if config.global_info:
            for j in range(len(self.glb_state)):
                self.glb_state[j][-NUM_Mach_Type:] = one_hot(machs[j].mch_typ, NUM_Mach_Type)
        self.glb_state[self.m][:2+MAX_Job_Type] = 1

        self.state = np.vstack((self.state, self.glb_state))

        return self.state


class Job(sim.Component):

    def setup(self, prc_time, due_date, label):
        # prc_time is the processing time of the job
        # due_date is due date
        # label show the job will be processed by which machine(1 or 2)
        # if label == -1, means this job have not been allocated
        self.prc_time = prc_time
        self.due_date = due_date
        self.label = label

    def process(self):
        for mach in machs:
            if mach.ispassive():
                mach.activate()
                break  # activate at most one machine
        yield self.passivate()


class MachHigh(sim.Component):
    def setup(self, wait_line, acm_time=0, num_jobs=0, lst_job=-1, mch_typ=1):
        self.wait_line = wait_line
        self.acm_time = acm_time
        self.num_jobs = num_jobs
        self.lst_job = lst_job
        self.mch_typ = mch_typ

    def process(self):
        while True:
            while len(self.wait_line) == 0:
                yield self.passivate()
            self.job = self.wait_line.pop()
            if self.lst_job == self.job.label:
                yield self.hold(self.job.prc_time * HE_Fkt)
            else:
                yield self.hold(self.job.prc_time * HE_Fkt + SETUP_Time)
            self.lst_job = self.job.label
            self.job.activate()


class MachLow(sim.Component):
    def setup(self, wait_line, acm_time=0, num_jobs=0, lst_job=-1, mch_typ=2):
        self.wait_line = wait_line
        self.acm_time = acm_time
        self.num_jobs = num_jobs
        self.lst_job = lst_job
        self.mch_typ = mch_typ

    def process(self):
        while True:
            while len(self.wait_line) == 0:
                yield self.passivate()
            self.job = self.wait_line.pop()
            if self.lst_job == self.job.label:
                yield self.hold(self.job.prc_time)
            else:
                yield self.hold(self.job.prc_time + SETUP_Time)
            self.lst_job = self.job.label
            self.job.activate()


env = Env_4_RNN()

MP = (N * 10 + (N + NUM_Job_Type) * 10 / 2)\
       / (NUM_A_Mach + NUM_B_Mach)
d_min = (1 - r - R / 2) * MP + SETUP_Time
d_max = (1 - r + R / 2) * MP + SETUP_Time

jobs_0 = [Job(prc_time=config.init_job_data[i][0], due_date=config.init_job_data[i][1], label=int(config.init_job_data[i][2]))
          for i in range(len(config.init_job_data))]
jobs = list(jobs_0)


new_jobs = []
for job_file in config.jobs_file_list:
    job_batch = pd.read_csv(job_file).values
    new_jobs.append([Job(prc_time=job_batch[i][0], due_date=job_batch[i][1], label=int(job_batch[i][2]))
                     for i in range(len(job_batch))])


mach_A = [MachHigh(wait_line=sim.Queue("wait_line_A{}".format(i))) for i in range(config.num_fast_machine)]
mach_B = [MachLow(wait_line=sim.Queue("wait_line_B{}".format(i))) for i in range(config.num_slow_machine)]
machs = mach_A + mach_B


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

reward_file = os.path.join(figs_dir, 'reward.png')
tardiness_file = os.path.join(figs_dir, 'tardiness.png')
info_file = os.path.join(experiment_dir, 'info.txt')

best_score = env.reward_range[0]
best_tardiness = -10000000
score_history = []
tardiness_history = []
learn_iters = 0
avg_score = 0
avg_tardiness = 0
n_steps = 0
learning_rate_flg = 0

late_list = np.zeros(n_games)
tardiness_list = np.zeros(n_games)
time_list = np.zeros(n_games)

agent.load_models()
print("r=%1f, R=%1f" % (config.r, config.R))
print(config.experi_dir(), "\n")
print("%dm %dbatch %dperBatch %dinit_j %df " %
      (NUM_Machs, NUM_Batch, NUM_Job_InBatch, NUM_0_job, NUM_Job_Type))
for i in range(n_games):
    start = timeit.default_timer()
    observation = env.reset()
    done = False
    score = 0
    while not done:
        action, prob, val = agent.choose_action(observation)
        observation_, reward, done, info = env.step(action)
        # in the validation env, the agent do not need to be trained
        observation = observation_
    time = timeit.default_timer() - start
    time_list[i] = time
    late_list[i] = env.late_count
    tardiness_list[i] = env.total_tardiness

    # score_history.append(score)
    tardiness_history.append(env.total_tardiness)
    # avg_score = np.mean(score_history[-100:])
    avg_tardiness = np.mean(tardiness_history[-100:])
    print('total_tardiness', env.total_tardiness, 'num_tardy_jobs', env.late_count)

std_lst.append(tardiness_list.std())
print("std:", tardiness_list.std())
mean_lst.append(tardiness_list.mean())
print("mean:", tardiness_list.mean())
glb_time_lst.append(min(time_list))
print("time:", min(time_list))
print("job number:", env.counts)
print("")


