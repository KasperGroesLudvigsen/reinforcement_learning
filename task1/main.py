import task1.britneyworld as bw
import random 
import numpy as np

def random_policy():
    return random.choice(["N", "E", "S", "W", "NE", "SE", "SW", "NW" ,"push"])

def run_episode(env, policy, stumble_probability, print_iter=False):
    total_reward = 0
    done = False
    iterations = 0
    while not done:
        iterations += 1
        observations, reward, done = env.take_action_guard(policy)
        total_reward += reward
        env.britney_stubmles(stumble_probability)
        # done = env.run(random_policy())
        if iterations % 10 == 0:
            env.display()
        if done == True and print_iter == True:
            print("It took {} iterations for Britney to reach the car".format(
                iterations))
    return iterations, total_reward
            
            
env = bw.Environment(10)
env.reset()
episodes = 10
total_iterations = 0
total_reward = 0
for episode in range(episodes):
    iterations, reward = run_episode(env, random_policy(), stumble_probability=0.1)
    total_iterations += iterations
    total_reward += reward
    env.reset()
print("On average over {} episodes it took {} iterations for Britney to reach "
      "the car, and the agent gained an average reward of {}".format(
          episodes, total_iterations//episodes, total_reward/episodes))


