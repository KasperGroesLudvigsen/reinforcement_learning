import params.task3_experiment_3_a_params as params_a
import params.task3_experiment_3_a2_params as params_a2
import params.task3_experiment_3_b_params as params_b
import task3.learning_environment as lr
import task1.britneyworld as bw
import task1.assassinworld as aw

params_a = params_a.PARAMS
params_a2 = params_a2.PARAMS
params_b = params_b.PARAMS
params_a["experiment_name"] = "task3_experiment_3_a"
params_a2["experiment_name"] = "task3_experiment_3_a2"
params_b["experiment_name"] = "task3_experiment_3_b"
environment_a = bw.Environment(params_a["environment_params"]) # all env params are the same in this experiment
environment_a2 = bw.Environment(params_a2["environment_params"])
environment_b = aw.AssassinWorld(environment_params=params_b["environment_params"])

lr.learning_environment(params=params_a, environment=environment_a)
lr.learning_environment(params=params_a, environment=environment_a2)
lr.learning_environment(params=params_b, environment=environment_b)

