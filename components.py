import salabim as sim
import config as cfg


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
                yield self.hold(self.job.prc_time * cfg.high_spd_fkt)
            else:
                yield self.hold(self.job.prc_time * cfg.high_spd_fkt + cfg.setup_time)
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
                yield self.hold(self.job.prc_time + cfg.setup_time)
            self.lst_job = self.job.label
            self.job.activate()


env = sim.Environment(trace=False)

# instantiate class Jobs
jobs_0 = [Job(prc_time=sim.Uniform(5, 15).sample(),
              due_date=sim.Uniform(cfg.d_min, cfg.d_max).sample(),
              label=sim.IntUniform(1, cfg.num_job_family).sample()) for _ in range(cfg.num_init_job)]

new_jobs = []
for i in range(cfg.num_batch):
    new_jobs.append([Job(prc_time=sim.Uniform(5, 15).sample(),
                         due_date=sim.Uniform(min(cfg.d_min, cfg.arrive_time_lst[i] +
                                                  cfg.setup_time + 15), cfg.d_max).sample(),
                         label=sim.IntUniform(1, cfg.num_job_family).sample())
                     for _ in range(cfg.num_job_in_batch)])

jobs = list(jobs_0)

# instantiate class Machine
mach_A = [MachHigh(wait_line=sim.Queue("wait_line_A{}".format(i))) for i in range(cfg.num_fast_machine)]
mach_B = [MachLow(wait_line=sim.Queue("wait_line_B{}".format(i))) for i in range(cfg.num_slow_machine)]
machs = mach_A + mach_B
