import params.task3_experiment_2_a_params_copy as params_A
import params.task3_experiment_2_b_params_copy as params_B
import task3.learning_env_curriculum as lr
import task1.britneyworld as bw
import task3.Discrete_SAC as sac
import task3.buffer as buf


params_a = params_A.PARAMS
params_b = params_B.PARAMS

buffer_a = buf.ReplayBuffer(params_a)
buffer_b = buf.ReplayBuffer(params_b)

DSAC_a = sac.DiscreteSAC(params_a)
DSAC_b = sac.DiscreteSAC(params_b)


params_a["experiment_name"] = "task3_experiment_2_a0"
params_b["experiment_name"] = "task3_experiment_2_b"

environment_a = bw.Environment(params_a["environment_params"]) # all env params are the same in this experiment
environment_b = bw.Environment(params_b["environment_params"])


lr.learning_environment(params_a, environment_a, DSAC_a, buffer_a)
params_a["environment_params"]["N"] = 7
params_a["experiment_name"] = "task3_experiment_2_a1"
lr.learning_environment(params_a, environment_a, DSAC_a, buffer_a)
params_a["environment_params"]["N"] = 9
params_a["experiment_name"] = "task3_experiment_2_a2"
lr.learning_environment(params_a, environment_a, DSAC_a, buffer_a)

lr.learning_environment(params_b, environment_b, DSAC_b, buffer_b)