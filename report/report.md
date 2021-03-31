# Tennis using MADDPG

## Introduction

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. Thus, the goal of each agent is to keep the ball in play.
![](/images/arms.gif)


## Multi Agent Deep Deterministic Policy Gradient (MADDPG)

MADDPG extends a reinforcement learning algorithm called DDPG, taking inspiration from actor-critic reinforcement learning techniques. In the project it was used a MADDPG agent. The agent contains the actor and the critic, also the same actor and critic is used by the agents to select the actions, in other words, there is a single agent to select the actions in both sides.


### Parameters table

Below the table with the parameters used by the agent

| Parameter     | Value     | 
| --------------|:---------:| 
| n_episodes    | 2000      |
| buffer_size   | int(1e5)  |
| batch_size    | 128       |
| gamma         | 0.99      |
| tau           | 1e-3      |
| lr_actor      | 1e-5      |
| lr_critc      | 1e-4      |
| weight_decay  | 0         |
| update_every  | 1         |
| num_updates   | 3         |


### Neural network

It was used a neural network with 3 hidden layers with 128 nodes each one, all hidden layers used leaky ReLu as activation function.
The output layer has 2 nodes with tahn activation function (output between -1 and 1).


![](/images/nn.svg)


## Results

The results show the reward per episode, the dots represent the reward per episode, the straight red line represents the reward moving average with the windows of 100 rewards, 
the dashed red line shows when the environment was solved (when the mean of the last 100 rewards are above 0.5).

The environment was solved in the episode 831, in other words, the mean of rewards from the episode 731 to 831 ware above 0.5.

![](/images/scores.png)


## Future work

In future work, it would be an options to implement prioritized experience replay to reduce the training time and try algorithms like A2C and PPO.