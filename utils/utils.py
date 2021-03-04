# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 11:39:48 2021

@author: groes
"""

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