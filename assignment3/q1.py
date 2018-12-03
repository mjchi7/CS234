import math
import gym
from frozen_lake import *
import numpy as np
import time
from utils import *
import matplotlib.pyplot as plt
from tqdm import *


def rmax(env, gamma, m, R_max, epsilon, num_episodes, max_step = 6):
    """Learn state-action values using the Rmax algorithm

    Args:
    ----------
    env: gym.core.Environment
        Environment to compute Q function for. Must have nS, nA, and P as
        attributes.
    gamma: float
        Discount factor. Number in range [0, 1)
    m: int
        	Threshold of visitance
    R_max: float 
        The estimated max reward that could be obtained in the game
    epsilon: 
        accuracy paramter
    num_episodes: int 
        Number of episodes of training.
    max_step: Int
        max number of steps in each episode

    Returns
    -------
    np.array
    An array of shape [env.nS x env.nA] representing state-action values
    """

    Q = np.ones((env.nS, env.nA)) * R_max / (1 - gamma)
    R = np.zeros((env.nS, env.nA))
    nSA = np.zeros((env.nS, env.nA))
    nSASP = np.zeros((env.nS, env.nA, env.nS))
    ########################################################
    #                   YOUR CODE HERE                     #
    ########################################################
    ########################################################
    #                    END YOUR CODE                     #
    ########################################################
    return (Q, average_score)


def main():
    env = FrozenLakeEnv(is_slippery=False)
    print env.__doc__
    for m in tqdm(np.arange(1,20,2)):
        (Q, average_score) = rmax(env, gamma = 0.99, m=m, R_max = 1, epsilon = 0.1, num_episodes = 1000)
        render_single_Q(env, Q)
        plt.plot(np.arange(1000),np.array(average_score))
    plt.title('The running average score of the R-max learning agent')
    plt.xlabel('traning episodes')
    plt.ylabel('score')
    plt.legend(['m = '+str(i) for i in np.arange(1,20,2)], loc='upper right')
    #plt.show()
    plt.savefig('r-max.jpg')

if __name__ == '__main__':
    print "haha"
    main()