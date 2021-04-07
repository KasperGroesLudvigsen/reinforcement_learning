from matplotlib import pyplot as plt
import task3.Discrete_SAC as sac
import task1.britneyworld as bw
import task3.buffer as buf


def learning_environment(params, environment, display = False, reset = True):
    
    print("-------- Running {} -------- ".format(params["experiment_name"]))
    
    number_of_episodes = params["learning_params"]["number_of_episodes"]
    buffer = buf.ReplayBuffer(params)
    DSAC = sac.DiscreteSAC(params)
    environment.reset()
    environment.display()
    
    training_scores = []
    validation_scores = []
    
    #fill up buffer    
    for _ in range(1000):
        done = False
        if reset:
            environment.reset()
        else:
            environment.respawn()
        while not done:
            done = DSAC.environment_step(environment, buffer, buffer_fill=True)
    
    # Training and Validation
    ran_out_of_time = 0
    success = 0 
    for _ in range(number_of_episodes):
        if reset:
            environment.reset()
        else:
            environment.respawn()
        
        done = False
        rewardz = 0
        # Emphasising recent experiences sample type
        if params["learning_params"]["ere"]:
            while not done:
                done, reward = DSAC.environment_step(
                    environment, buffer, buffer_fill = False
                    )
                rewardz += reward
            if display:
                environment.display()
            
            big_k = 100
            little_k = 1
            for a in range(big_k):
                states, new_states, actions, rewards, dones = buffer.sample(
                    params['learning_params']['batch_size'], True, big_k, little_k
                    )
                DSAC.gradient_step(states, new_states, actions, rewards, dones)
                little_k += 1
        
        #normal buffer sample type
        else:
            while not done:
                done, reward = DSAC.environment_step(
                    environment, buffer, buffer_fill = False
                    )
                rewardz += reward
                if display:
                    environment.display()
                states, new_states, actions, rewards, dones = buffer.sample(
                    params['learning_params']['batch_size']
                    )
                DSAC.gradient_step(states, new_states, actions, rewards, dones)

        training_scores.append(rewardz)    
        if environment.time_elapsed == environment.time_limit:
            ran_out_of_time += 1
        else:
            success +=1
            
        if _ % 10 == 0:
            DSAC.train_mode = False
            if reset:
                environment.reset()
            else:
                environment.respawn()
        
            done = False
            val_rewardz = 0
            while not done:
                done, reward = DSAC.environment_step(
                    environment, buffer, buffer_fill = False
                    )
                if display:
                    environment.display()
                val_rewardz += reward
            validation_scores.append(val_rewardz)
            DSAC.train_mode = True
     
    filename = "plots/" + params["experiment_name"] + "_training_scores.png"             
    plt.plot(training_scores)
    plt.title('Training Scores')
    plt.xlabel('Episode')
    plt.ylabel('Episode reward')
    plt.grid(True)
    plt.show()
    try:
        plt.savefig(filename)
        print("Plot saved as {}".format(filename))
    except:
        print("Could not save plot")
    
    filename = params["experiment_name"] + "_validation_scores.png"
    plt.plot(validation_scores)
    plt.title('Validation Scores')
    plt.xlabel('Episode')
    plt.ylabel('Episode reward')
    plt.grid(True)
    plt.show()
    try:
        plt.savefig(filename)
        print("Plot saved as {}".format(filename))
    except:
        print("Could not save plot")

    
    successes_pct = round(((success/number_of_episodes)*100), 2)
    ran_out_pct = round(((ran_out_of_time/number_of_episodes)*100), 2)

    print("Number of times ran out: {}".format(ran_out_of_time))
    print("Number of successes: {}".format(success))
    print("Percentage of times ran out : {} %".format(ran_out_pct))
    print("Percentage of successes : {} %".format(successes_pct))
    

