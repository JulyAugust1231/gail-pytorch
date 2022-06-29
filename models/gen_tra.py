from safety_gridworld import PitWorld
import gym
import numpy as np
import random

Action = [[0, 0, 3, 3, 3, 0, 3, 3, 2, 0, 1, 1, 2, 0, 2, 3, 1, 3, 0, 3, 3, 0, 2, 0, 2, 1, 0, 0, 3, 3, 0, 0, 3, 1, 3, 1, 3, 0, 0, 3, 2, 3, 3, 2],
          [2, 3, 2, 2, 2, 3, 2, 1, 1, 1, 2, 1, 2, 0, 0, 3, 2, 0, 3, 3, 2, 3, 2, 0, 1, 3, 2, 2, 0, 2, 1, 0, 2, 1, 3, 2, 3, 3, 2, 3, 3, 3, 1, 3, 1, 2, 0, 2, 2, 0, 2, 1, 3, 3, 1, 0, 0, 3, 3, 0, 3, 0, 0, 0, 0, 3],
          [3, 3, 3, 2, 1, 1, 3, 3, 1, 3, 0, 3, 2, 3, 1, 0, 3, 0, 0, 0, 0, 2, 2, 3, 0, 3, 0, 3, 0, 0, 0, 2, 3, 3, 1, 2, 1, 0, 3, 3, 3],
          [1, 3, 2, 3, 2, 3, 3, 0, 2, 0, 2, 3, 0, 1, 1, 0, 3, 3, 0, 1, 3, 0, 3, 0, 3, 2, 0, 3, 0, 3, 1, 3, 3, 0, 3],
          [1, 3, 3, 1, 3, 3, 0, 1, 0, 2, 0, 3, 2, 0, 2, 0, 0, 3, 1, 0, 3, 3, 2, 0, 1, 3, 0, 3, 3, 3, 3, 0, 3, 0, 3],
          [3, 3, 3, 3, 3, 3, 3, 3, 0, 0, 3, 3, 3, 0,0,0,0,0,3],
          [1, 2, 2, 0, 2, 1, 1, 2, 1, 0, 2, 3, 3, 1, 3, 2, 3, 0, 2, 3, 0, 3, 0, 3, 1, 0, 3, 2, 0, 0, 3, 0, 3, 0, 0, 2, 0, 3, 3, 3],
          [2, 2, 1, 0, 2, 3, 3, 3, 3, 3, 0, 2, 3, 2, 0, 2, 2, 1, 0, 0, 0, 3, 0, 3, 0, 1, 1, 3, 2, 3, 0, 3, 3, 1, 0, 0, 1, 3, 3, 3, 0, 3, 0],
          [1, 2, 1, 2, 1, 3, 3, 3, 3, 1, 0, 0, 2, 3, 0, 3, 2, 3, 3, 0, 3, 1, 3, 3, 3, 0, 0, 2, 0, 3, 0, 2, 0, 3, 0, 3, 3, 3, 0],
          [2, 1, 0, 1, 1, 2, 3, 3, 3, 3, 3, 3, 2, 1, 2, 2, 0, 1, 3, 0, 3, 0, 0, 3, 0, 3, 3, 3, 0, 0, 3],
          [1, 3, 3, 3, 3, 3, 0, 1, 3, 0, 2, 0, 3, 1, 2, 0, 2, 1, 3, 3, 2, 3, 1, 1, 0, 0, 0, 3, 1, 1, 3, 2, 0, 0, 1, 3, 1, 3, 2, 0, 0, 3, 0, 0, 3, 3, 0, 0, 3, 0, 0, 0, 0, 3, 3, 0, 3, 1, 2, 1, 3, 2, 1, 0, 1, 2, 3, 3, 0, 2, 1, 3, 2, 2, 3]
        ]



if __name__ == "__main__":

    env = PitWorld(size=14, one_hot_features=True, rand_goal=False)
    #print(env.maze.shape)
    ## calculate for expert trajectories
    print(env.to_string())
    exp_rwd_iter = []
    exp_constraints_iter = []
    exp_obs = []
    exp_acts = []
    Action = []
    for i in range(30000):
        steps = 0
        ep_obs = []
        ep_ac = []
        ep_rwds = []
        ep_constraints = []
        t = 0
        done = False
        ob = env.reset()
        for a in range(200):
            act = random.randint(0,3)
            ep_obs.append(ob)
            #exp_obs.append(ob)
            #exp_acts.append(act)
            ep_ac.append(act)
            ob, rwd, done, info = env.step(act)
            ep_rwds.append(rwd)
            ep_constraints.append(info['pit'])

            t += 1
            steps += 1
            if done: break
        if done:

            if  np.sum(ep_rwds)>0 and np.sum(ep_constraints)<=4:
                exp_rwd_iter.append(np.sum(ep_rwds))
                exp_constraints_iter.append(np.sum(ep_constraints))
                exp_obs.append(ep_obs)
                exp_acts.append(ep_ac)


        #ep_obs = FloatTensor(np.array(ep_obs))
        #ep_rwds = FloatTensor(ep_rwds)



    print(exp_obs)
    print(exp_acts)
    print(exp_rwd_iter)
    print(exp_constraints_iter)