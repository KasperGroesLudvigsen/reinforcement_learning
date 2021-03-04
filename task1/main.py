import task1.britneyworld as bw
import random 
import numpy as np
import utils.utils as utils

def random_policy():
    return random.choice(["N", "E", "S", "W", "NE", "SE", "SW", "NW" ,"push"])

env = bw.Environment(10)
env.reset()
episodes = 10
total_iterations = 0
total_reward = 0
for episode in range(episodes):
    iterations, reward = utils.run_episode(env, random_policy(), stumble_probability=0.1)
    total_iterations += iterations
    total_reward += reward
    env.reset()
print("On average over {} episodes it took {} iterations for Britney to reach "
      "the car, and the agent gained an average reward of {}".format(
          episodes, total_iterations//episodes, total_reward/episodes))


