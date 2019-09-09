import logging
import torch
import gym
import gym_chrome_dino
from gym_chrome_dino.utils.wrappers import make_dino
from agent import DQN, DoubleDQN, DuelDQN, DQNPrioritized


# Setup logging
formatter = logging.Formatter(r'"%(asctime)s",%(message)s')
logger = logging.getLogger("dino-rl")
logger.setLevel(logging.INFO)
fh = logging.FileHandler("./dino-log.csv")
fh.setFormatter(formatter)
logger.addHandler(fh)


# Check whether cuda is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialise the game
env = gym.make('ChromeDino-v0')
# env = gym.make('ChromeDinoNoBrowser-v0')
env = make_dino(env, timer=True, frame_stack=True)

# Get the number of actions and the dimension of input
n_actions = env.action_space.n

# ----------- Nature DQN ---------------
dqn = DQN(n_actions, device)
dqn.train(env, logger)
# dqn.load("./trained/dqn.pkl")
# dqn.test(env)

# ----------- Prioritized DQN ---------------
# dqn_p = DQNPrioritized(n_actions, device)
# dqn_p.train(env, logger)
# dqn_p.load("./trained/dqn_p.pkl")
# dqn_p.test(env)


# ----------- Double DQN ----------------
# double_dqn = DoubleDQN(n_actions, device)
# double_dqn.train(env, logger)
# double_dqn.load("./trained/double-dqn.pkl")
# double_dqn.test(env)


# ----------- Dueling DQN ----------------
# duel_dqn = DuelDQN(n_actions, device)
# duel_dqn.train(env, logger)
# duel_dqn.load("./trained/duel-dqn.pkl")
# duel_dqn.test(env)

env.render(mode="rgb_array")
env.close()
