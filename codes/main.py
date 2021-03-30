from unityagents import UnityEnvironment
from monitor import interact
from agent import Agent
import argparse

# Instantiate argument parser
parser = argparse.ArgumentParser()

# Add arguments (Interaction with the environment)
parser.add_argument('--n_episodes', nargs='?', const=1, type=int, default=2000)

# Add arguments (Agent)
parser.add_argument('--buffer_size', nargs='?', const=1, type=int, default=int(1e5))
parser.add_argument('--batch_size', nargs='?', const=1, type=int, default=128)
parser.add_argument('--gamma', nargs='?', const=1, type=float, default=0.99)
parser.add_argument('--tau', nargs='?', const=1, type=float, default=1e-3)
parser.add_argument('--lr_actor', nargs='?', const=1, type=float, default=1e-5)
parser.add_argument('--lr_critic', nargs='?', const=1, type=float, default=1e-4)
parser.add_argument('--weight_decay', nargs='?', const=1, type=float, default=0)
parser.add_argument('--update_every', nargs='?', const=1, type=int, default=1)
parser.add_argument('--num_updates', nargs='?', const=1, type=int, default=3)

# Parser parameters
parser.add_argument("--mode", default='client')
parser.add_argument("--port", default=52162)

# Pass args
args = parser.parse_args()


if __name__ == '__main__':

    # Create environment
    # env = UnityEnvironment(file_name="Tennis_Linux_NoVis/Tennis.x86_64")
    # env = UnityEnvironment(file_name="unity/Tennis_Linux_NoVis/Tennis.x86_64")
    env = UnityEnvironment(file_name="/opt/project/unity/Tennis_Linux_NoVis/Tennis.x86_64")

    # Get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    # reset the environment
    env_info = env.reset(train_mode=True)[brain_name]

    # Instantiate agent
    agent = Agent(
        state_size=len(env_info.vector_observations[0]),
        action_size=brain.vector_action_space_size,
        num_agents=len(env_info.agents),
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        gamma=args.gamma,
        tau=args.tau,
        lr_actor=args.lr_actor,
        lr_critic=args.lr_critic,
        weight_decay=args.weight_decay,
        update_every=args.update_every,
        num_updates=args.num_updates,
        seed=0
    )

    # Interact with environment
    scores = interact(
        env,
        agent,
        brain_name=brain_name,
        n_episodes=args.n_episodes
    )
