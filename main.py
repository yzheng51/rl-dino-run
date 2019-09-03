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
# dqn.train(env, logger)
dqn.load("./trained/dqn.pkl")
dqn.test(env)

# ----------- Prioritized DQN ---------------
# dqn_p = DQNPrioritized(n_actions, device, beta=1)
# dqn_p.train(env, logger)

# ----------- Double DQN ----------------
# double_dqn = DoubleDQN(n_actions, device)
# double_dqn.train(env, logger)

# ----------- Dueling DQN ----------------
duel_dqn = DuelDQN(n_actions, device)
# duel_dqn.train(env, logger)
# duel_dqn.load("./trained/duel-dqn.pkl")
# duel_dqn.test(env)


# ----------- Test ----------------------
# dqn.load("./cases/v1/model_350.pkl")
# score = dqn.test(env)
# print(f"Score: {score}")

# dqn.save(f"model_final.pkl")
env.render(mode="rgb_array")
env.close()

# dqn ./cases/dqn/trained/model_1985.pkl
# dqn-bn ./cases/dqn-bn/trained/model_1990.pkl

# double-dqn ./cases/double-dqn/trained/model_1980.pkl
# double-dqn-bn ./cases/double-dqn-bn/trained/model_1970.pkl

# duel-dqn ./cases/duel-dqn/rerun-1/model_20.pkl
# duel-dqn-bn final model ./cases/duel-dqn-bn/trained/model_1900.pkl
