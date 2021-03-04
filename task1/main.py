import reinforcement_learning.task1.britneyworld as bw # H: this 1 works for me don't delete just hash
#import task1.britneyworld as bw
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
        print(policy)
        print(env.guard_location)
        if iterations % 10 == 0:
            env.display()
        if done == True and print_iter == True:
            print("It took {} iterations for Britney to reach the car".format(
                iterations))
    return iterations, total_reward
            
            
env = bw.Environment(10)
env.reset()
episodes = 1
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


for _ in range(5):
    print(random_policy())
# Random policy works as it should

print(env.is_empty([3,4]))

print(env.guard_location)
#print(env.get_next_position("SW", env.guard_location))
env.take_action_guard("S")
print(env.guard_location)

# Take guard action isn't working
False_list = 0
True_list = 0
print(env.is_empty([5,5]))
for x in range(10):
    for y in range(10):
        if env.is_empty([x,y]) == False:
            False_list +=1
        else:
            True_list +=1
    

print(False_list)
print(True_list)

# is_empty is working




