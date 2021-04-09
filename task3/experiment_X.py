import params.task3_experiment_2_a_params as params_A
import params.task3_experiment_2_b_params as params_b
import params.task3_experiment_2_c_params as params_c
import params.task3_experiment_2_d_params as params_d
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


params_a['learning_params']['entropy_alpha'] = 0.
params_b['learning_params']['entropy_alpha'] = 0.1
params_c['learning_params']['entropy_alpha'] = 0.3
params_d['learning_params']['entropy_alpha'] = 0.7


params_a["experiment_name"] = "task3_experiment_X_a"
params_b["experiment_name"] = "task3_experiment_X_b"
params_c["experiment_name"] = "task3_experiment_X_c"
params_d["experiment_name"] = "task3_experiment_X_d"

environment_a = bw.Environment(params_a["environment_params"]) # all env params are the same in this experiment
environment_b = bw.Environment(params_b["environment_params"])
environment_c = bw.Environment(params_c["environment_params"])
environment_d = bw.Environment(params_d["environment_params"])



lr.learning_environment(params=params_a, environment=environment_a)
lr.learning_environment(params=params_b, environment=environment_b)
lr.learning_environment(params=params_c, environment=environment_c)
lr.learning_environment(params=params_d, environment=environment_d)