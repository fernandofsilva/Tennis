import numpy as np
from collections import deque
import torch


def interact(env,
             agent,
             brain_name,
             n_episodes,
             save_model='model/checkpoint.pth'):
    """Interaction between agent and environment.

    This function define the interaction between the agent and the openai gym
    environment, and printout the partial results

    Args:
        env: openai gym's environment
        agent: class agent to interact with the environment
        brain_name: String. Name of the agent of the unity environment
        n_episodes: Integer. Maximum number of training episodes
        save_model: String. Path+file_name to save the model
    """
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    solved_env = 0

    # Loop the define episodes
    for i_episode in range(1, n_episodes+1):

        env_info = env.reset(train_mode=True)[brain_name]  # reset the environment
        states = env_info.vector_observations               # get the current state
        agent.noise.reset()                                # reset the agent noise
        score = np.zeros(len(env_info.agents))

        # Loop over the maximum number of time-steps per episode
        while True:
            actions = agent.act(states)
            next_states, rewards, dones = agent.env_step(env, actions, brain_name)
            agent.step(states, actions, rewards, next_states, dones)
            states = next_states
            score += rewards

            # Break the loop if it is final state
            if np.any(dones):
                break

        scores_window.append(np.mean(score))  # save most recent score
        scores.append(np.mean(score))         # save most recent score

        print(f'\rEpisode {i_episode}\tAverage Score: {np.mean(scores_window):.2f}', end="")

        if i_episode % 100 == 0:
            print(f'\rEpisode {i_episode}\tAverage Score: {np.mean(scores_window):.2f}')

        if np.mean(scores_window) >= 0.5 and solved_env == 0:
            print(f'\nEnvironment solved in {i_episode:d} episodes!\tAverage Score: {np.mean(scores_window):.2f}')

            # Save model
            torch.save(agent.actor_local.state_dict(), save_model)
            solved_env += 1

    return scores
