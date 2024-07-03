import salabim as sim
import gym
from gym import spaces
import numpy as np
import os

import config as cfg
from components import jobs, new_jobs, machs

import random


np.random.seed(1337)
random.seed(1337)
sim.random.seed(1337)

"""
2 kind of Machines [Mach1, Mach2]
Mach1 is faster, Mach2 is normal

8 type of Jobs[1, 2, 3, 4, 5, 6, 7, 8]
changing job_types needs setup time
"""


# one hot function
def one_hot(a, num_type):
    b = np.zeros(num_type)
    b[a - 1] = 1
    return b


def jobs_in_matrix(job_batch, num_jobs):
    job_time = np.zeros(num_jobs * 2).reshape(num_jobs, 2)
    job_type = np.zeros(num_jobs * cfg.max_job_family).reshape(num_jobs, cfg.max_job_family)

    for i in range(num_jobs):
        job_time[i] = [job_batch[i].prc_time / cfg.d_max, job_batch[i].due_date / cfg.d_max]
        job_type[i] = one_hot(job_batch[i].label, cfg.max_job_family)

    job_state = np.hstack((job_time, job_type))
    return job_state


class EnvPMSP(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 2
    }

    def __init__(self, init_jobs, new_jobs, machs):
        self.sim_env = sim.Environment(trace=False)
        self.action_space = spaces.Discrete(300)
        self.observation_space = spaces.Box(low=-1.0, high=500.0, shape=(cfg.input_size,), dtype=np.float32)

        self.init_jobs = init_jobs
        self.new_jobs = new_jobs
        self.machs = machs

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
        self.job_choose[n].enter(self.machs[self.m].wait_line)

        # if this job is the first job of this machine, an additional Setup_time is needed
        if len(self.machs[self.m].wait_line) == 1:
            if self.machs[self.m].mch_typ == 2:  # normal machine
                self.machs[self.m].acm_time += self.job_choose[n].prc_time  # + setup_time
            if self.machs[self.m].mch_typ == 1:  # high-efficiency machine
                self.machs[self.m].acm_time += self.job_choose[n].prc_time * cfg.high_spd_fkt  # + setup_time

        elif self.job_choose[n].label == self.machs[self.m].lst_job:
            if self.machs[self.m].mch_typ == 2:  # normal machine
                self.machs[self.m].acm_time += self.job_choose[n].prc_time
            if self.machs[self.m].mch_typ == 1:
                self.machs[self.m].acm_time += self.job_choose[n].prc_time * cfg.high_spd_fkt

            reward += cfg.setup_reward
            self.num_posi += 1

        elif self.job_choose[n].label != self.machs[self.m].lst_job:
            if self.machs[self.m].mch_typ == 2:  # normal machine
                self.machs[self.m].acm_time += (self.job_choose[n].prc_time + cfg.setup_time)
            if self.machs[self.m].mch_typ == 1:  # high-efficiency machine
                self.machs[self.m].acm_time += (self.job_choose[n].prc_time * cfg.high_spd_fkt + cfg.setup_time)

            self.num_setups += 1

            for job in self.job_choose:
                if job.label == self.machs[self.m].lst_job:
                    reward -= cfg.setup_reward
                    self.neg += 1
                    break
                else:
                    pass

        # attribute lst_job in machine will change
        self.machs[self.m].lst_job = self.job_choose[n].label
        self.machs[self.m].num_jobs += 1

        self.glb_state[self.m][-(cfg.max_job_family + cfg.num_machine_family):-cfg.num_machine_family] \
            = one_hot(self.machs[self.m].lst_job, cfg.max_job_family)
        self.glb_state[self.m][2 + cfg.max_job_family] = self.machs[self.m].acm_time / cfg.d_max

        """
        reward
        """
        # the goal is to minimize total tardiness
        # so if a job isn't finished on time, the agent will get a negative reward
        if (self.job_choose[n].due_date - self.machs[self.m].acm_time) < 0:
            self.total_tardiness += (self.job_choose[n].due_date - self.machs[self.m].acm_time)  # negative
            self.late_count += 1
            reward += (self.job_choose[n].due_date - self.machs[self.m].acm_time)
        elif (self.job_choose[n].due_date - self.machs[self.m].acm_time) < cfg.just_in_time_threshold:
            reward += cfg.just_in_time_reward
            self.just_in_time_flg += 1

        """
        delete the chosen job from the state 
        """
        del self.job_choose[n]
        self.state = np.delete(self.state, n, 0)
        self.job_state = self.state[:-cfg.num_all_machine, :(cfg.max_job_family + 2)]  # first 5 column
        # self.job_state = np.delete(self.job_state, n, 0)

        """
        find the next idle machine according to the accumulated time of each machine
        """
        for i in range(len(self.machs)):
            self.mach_time[i] = self.machs[i].acm_time
        self.m = int(np.argmin(self.mach_time))

        self.glb_state[:, :2 + cfg.max_job_family] = 0
        self.glb_state[self.m][:2 + cfg.max_job_family] = 1
        # indicate the working machine
        # self.glb_state[self.m][1] = 1

        """arrive of new jobs"""
        if self.rls < cfg.num_batch and self.machs[self.m].acm_time >= cfg.arrive_time_lst[self.rls]:
            self._new_jobs_arrival(self.new_jobs[self.rls], len(self.new_jobs[self.rls]))

        """change of state"""
        if self.machs[self.m].acm_time != 0:
            acm_time = np.array(self.machs[self.m].acm_time / cfg.d_max).reshape(1, 1)
            lst_job = np.array(one_hot(self.machs[self.m].lst_job, cfg.max_job_family)).reshape(1, cfg.max_job_family)
            mach_type = np.array(one_hot(self.machs[self.m].mch_typ, cfg.num_machine_family)).reshape(1, cfg.num_machine_family)
        else:
            acm_time = np.zeros(1).reshape(1, 1)
            lst_job = np.zeros(cfg.max_job_family).reshape(1, cfg.max_job_family)
            mach_type = np.array(one_hot(self.machs[self.m].mch_typ, cfg.num_machine_family)).reshape(1, cfg.num_machine_family)

        self.mach_state = np.hstack((acm_time, lst_job, mach_type))
        # self.mach_state = self.glb_state[self.m][-(1 + max_job_family + config.num_machine_family):]
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
                reward += cfg.final_reward

            for mach in self.machs:
                mach.lst_job = -1
            self.sim_env.run()

        return self.state, reward, done, {}

    def reset(self):
        self.rls = 0
        self.counts = 0
        # job list to choose
        self.job_choose = self.init_jobs[:]
        self.late_count = 0
        self.total_tardiness = 0
        self.num_setups = 0
        self.num_posi = 0
        self.neg = 0
        self.optimal_found = False
        self.just_in_time_flg = 0

        # reset the attribute
        for mach in self.machs:
            mach.acm_time = 0
            mach.num_jobs = 0
            mach.wait_line.clear()
            mach.lst_job = -1

        """First convert the jobs' info into a matrix"""
        # the number of rows of state is the number of jobs
        # first put processing time and due_date in the state
        # then turn the job_type into one-hot encoding
        self.job_state = jobs_in_matrix(self.init_jobs, cfg.num_init_job)

        """Then convert the machines' info into a matrix 
        All machines are idle at beginning, the 0th machine is selected to be the first machine to be scheduled"""
        self.m = 0

        # first generate the vector of the accumulated time of the machine
        # at first this vector is full of 0
        acm_time = np.zeros(cfg.num_init_job).reshape(cfg.num_init_job, 1)

        # the next vector is for the last job of this machine
        ltz_job = np.zeros(cfg.num_init_job * cfg.max_job_family).reshape(cfg.num_init_job, cfg.max_job_family)

        # then the vector of the type of the machine
        mach_type = np.zeros(cfg.num_init_job * cfg.num_machine_family).reshape(cfg.num_init_job, cfg.num_machine_family)
        for i in range(cfg.num_init_job):
            mach_type[i] = one_hot(self.machs[self.m].mch_typ, cfg.num_machine_family)

        # at last stack the vectors of the idle machine together
        self.mach_state = np.hstack((acm_time, ltz_job, mach_type))

        """ Merge to get the global matrix"""
        self.state = np.hstack((self.job_state, self.mach_state))

        """
        put all machine time into array, to realize the real-time function
        """
        self.mach_time = np.zeros(len(self.machs))
        for i in range(len(self.machs)):
            self.mach_time[i] = self.machs[i].acm_time
        self.m = 0

        """
        ALL Machine State
        """
        self.glb_state = np.zeros(cfg.num_all_machine * cfg.input_size).reshape(cfg.num_all_machine, cfg.input_size)
        for i in range(len(self.glb_state)):
            # last 2 column is the type of machine
            self.glb_state[i][-cfg.num_machine_family:] = one_hot(self.machs[i].mch_typ, cfg.num_machine_family)
        self.glb_state[self.m][:2+cfg.max_job_family] = 1
        # self.glb_state[self.m][1] = 1

        self.state = np.vstack((self.state, self.glb_state))

        return self.state


if __name__ == "__main__":
    env = EnvPMSP(init_jobs=jobs, new_jobs=new_jobs, machs=machs)
    done = False
    score = 0
    matrix_file = os.path.join(cfg.experi_dir(), 'state_matrix.txt')
    observation = env.reset()
    # np.savetxt(matrix_file, observation)
    while not done:
        action = min(range(len(env.job_choose)), key=lambda i: env.job_choose[i].due_date)
        observation_, reward, done, info = env.step(action)
        if env.counts == cfg.num_init_job / 2:
            np.savetxt(matrix_file, observation_)
        score += reward
        observation = observation_
    print(score)
    print(env.counts)
    print("total_tardiness: ", env.total_tardiness)
