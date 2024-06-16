import os
import re
import pandas as pd
# file directory
def experi_dir():
    """
    Naming Rules:
        BrainFeature_RewardFeature_EnvFeature_NumEpisodes
    """
    return 'comparison\\1HeadAttention-1'


# Instance Setting
instance_folder = f'pmsp_instances\\r=0.4_R=0.6\\20m_8b_15new_400j_8f'

# read the configuration
pattern = re.compile(r'(\d+\.?\d*)')
numbers = pattern.findall(instance_folder)
brain_numbers = pattern.findall(experi_dir())

# read the config of the brain

NUM_Head = int(brain_numbers[0])
global_info = bool(brain_numbers[1])

# read the name and set the environment
r = float(numbers[0])
R = float(numbers[1])
m = int(numbers[2])
num_batches = int(numbers[3])
num_new_jobs_per_batch = int(numbers[4])
num_init_jobs = int(numbers[5])
num_families = int(numbers[6])



init_job_path = os.path.join(instance_folder, 'init_jobs.csv')
init_job_data = pd.read_csv(init_job_path).values


new_job_files = os.path.join(instance_folder, 'new_jobs')
jobs_file_list = [os.path.join(new_job_files, f) for f in os.listdir(new_job_files) if f.endswith('csv')]
jobs_file_list.sort()


# instantiate class Machine
machine_path = os.path.join(instance_folder, 'machines.csv')
mach_type_list = pd.read_csv(machine_path).values
num_fast_machine = 0
num_slow_machine = 0
for t in mach_type_list:
    if t == 1:
        num_fast_machine += 1
    if t == 2:
        num_slow_machine += 1


"""Env Config"""
# machine
NUM_A_Mach = num_fast_machine
NUM_B_Mach = num_slow_machine
NUM_Machs = NUM_A_Mach + NUM_B_Mach


NUM_Mach_Type = 2

# job
MAX_Job_Type = 8

"""RL Config"""
setup_reward = 1
final_reward = 100



""" Validation Loop Config """

NUM_Init_Job = num_init_jobs



NUM_Families = num_families



NUM_Batch = num_batches


Arrival_Loop = 2
NUM_ArrivalPerBatch = 10
Delta_Arrival = 5



""" State Matrix Size """
Input_Size = 3 + NUM_Mach_Type + 2 * MAX_Job_Type

