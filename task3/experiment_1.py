import params.task3_experiment_1_a_params as params_a
import params.task3_experiment_1_b_params as params_b
import params.task3_experiment_1_c_params as params_c
import params.task3_experiment_1_d_params as params_d
import task3.learning_environment as lr
import task1.britneyworld as bw

"""
We're running 4 experiments in a 10x10 britneyworld with stumble prob = 0.2:
A. vanilla sac
B. sac + entropy tuning
C. sac + emphasising recent experiments (ere)
D. sac + entropy tuning + ere
"""

params_a = params_a.PARAMS
params_b = params_b.PARAMS
params_c = params_c.PARAMS
params_d = params_d.PARAMS
params_a["experiment_name"] = "task3_experiment_1_a"
params_b["experiment_name"] = "task3_experiment_1_b"
params_c["experiment_name"] = "task3_experiment_1_c"
params_d["experiment_name"] = "task3_experiment_1_d"

environment = bw.Environment(params_a["environment_params"]) # all env params are the same in this experiment

lr.learning_environment(params=params_a, environment=environment)
lr.learning_environment(params=params_b, environment=environment)
lr.learning_environment(params=params_c, environment=environment)
lr.learning_environment(params=params_d, environment=environment)