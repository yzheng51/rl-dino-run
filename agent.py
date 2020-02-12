"""
This is a script contains DQN, Double DQN, Dueling DQN and DQN wih PER

"""
import random
import itertools
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from defaults import HyperParam
from model import Net, DuelNet
from memory import Transition, ReplayMemory, PrioritizedReplayMemory
from common import StateProcessor, LinearAnneal


random.seed(777)
np.random.seed(777)
torch.manual_seed(777)


class DQN(HyperParam):
    def __init__(self, n_actions, device, batch_norm=False):
        self.device = device
        self.n_actions = n_actions
        self._memory_init()
        self._net_init(n_actions, batch_norm)
        self.epsilon = LinearAnneal(self.EPS_INIT, self.EPS_END, self.EXPLORE_STEP)
        self.optimizer = optim.RMSprop(self.policy_net.parameters(), lr=self.LR)

    def _memory_init(self):
        self.memory = ReplayMemory(self.MEMORY_SIZE)

    def _net_init(self, n_actions, batch_norm):
        """
        Initialization of two neural network

        policy net - a function return the all q values corresponding to each action
                     given the input state. This network is used to compute expected
                     q vlue and will be optimized during each iteration
        target net - a function which will be updated from policy net after N optimization
                     step (N is a hyperparameter). This network is used to compute
                     expected q value based on next state
        """
        self.policy_net = Net(n_actions, batch_norm).to(self.device)
        self.target_net = Net(n_actions, batch_norm).to(self.device)
        self._update_target()
        self.target_net.eval()

    def _choose_action(self, state):
        """
        epsilon - greedy policy to decide next action

        the value of epsilon will anneal linearly
        """
        sample = random.random()
        if sample > self.epsilon.anneal():
            with torch.no_grad():
                return self.policy_net(state).max(1)[1].view(1, 1)
        else:
            action = random.randrange(self.n_actions)
            return torch.tensor([[action]], device=self.device, dtype=torch.long)

    def _q(self, states, actions):
        return self.policy_net(states).gather(1, actions)

    def _expected_q(self, next_states, rewards):
        """
        Calculation of expected q value

        based on bellman equation: q = r + gamma * q_next
        """
        # only use those next state is not the end of the game
        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, next_states)),
            device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in next_states if s is not None])

        # put the state into the network and filter those action with the max q value
        q_next = torch.zeros(self.BATCH_SIZE, device=self.device)
        q_next[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        expected_q = rewards + self.GAMMA * q_next

        return expected_q.unsqueeze(1)

    def _optimize(self):
        if len(self.memory) < self.BATCH_SIZE:
            return
        transitions = self.memory.sample(self.BATCH_SIZE)
        batch = Transition(*zip(*transitions))
        states = torch.cat(batch.state)
        actions = torch.cat(batch.action)
        rewards = torch.cat(batch.reward)

        # calculate q value and expected q value
        q = self._q(states, actions)
        expected_q = self._expected_q(batch.next_state, rewards)
        loss = F.smooth_l1_loss(q, expected_q)

        # optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def _update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, file_name):
        torch.save(self.policy_net.state_dict(), file_name)

    def load(self, model):
        self.policy_net.load_state_dict(torch.load(model))
        self.policy_net.eval()

    def train(self, env, logger):
        """Main part for training the agent"""
        processor = StateProcessor()
        optim_cnt = 0
        for i_episode in range(self.N_EPISODE):
            total_reward = 0
            state = processor.to_tensor(env.reset()).to(self.device)
            for t in itertools.count():
                # Select and perform an action
                action = self._choose_action(state)
                next_state, reward, done, _ = env.step(action)
                # Sum up total reward for one episode, convert reward to tensor
                total_reward += reward
                reward = torch.tensor([reward], dtype=torch.float32, device=self.device)

                if done:
                    self.memory.push(state, action, None, reward)
                    self._optimize()
                    break
                else:
                    next_state = processor.to_tensor(next_state).to(self.device)
                    self.memory.push(state, action, next_state, reward)
                    self._optimize()

                state = next_state
            optim_cnt += t
            score = env.unwrapped.game.get_score()
            logger.info(f"{i_episode},{optim_cnt},{total_reward:.1f},{score},{self.epsilon.p:.6f}")

            if i_episode % self.TARGET_UPDATE == 0:
                self._update_target()
                self.save(f"model_{i_episode}.pkl")

    def test(self, env):
        while True:
            processor = StateProcessor()
            state = processor.to_tensor(env.reset()).to(self.device)
            while True:
                with torch.no_grad():
                    action = self.policy_net(state).max(1)[1].view(1, 1)
                next_state, _, done, _ = env.step(action)

                if done:
                    break
                next_state = processor.to_tensor(next_state).to(self.device)
                state = next_state


class DuelDQN(DQN):
    """
    Dueling DQN change the architecture of neural network compared with DQN
    """
    def __init__(self, n_actions, device, batch_norm=False):
        super().__init__(n_actions, device, batch_norm)

    def _net_init(self, n_actions, batch_norm):
        self.policy_net = DuelNet(n_actions, batch_norm).to(self.device)
        self.target_net = DuelNet(n_actions, batch_norm).to(self.device)
        self._update_target()
        self.target_net.eval()


class DoubleDQN(DQN):
    """
    Double DQN use target net to compute q values but choose the next action
    resulting in maximum q value using policy net
    """
    def __init__(self, n_actions, device, batch_norm=False):
        super().__init__(n_actions, device, batch_norm)

    def _expected_q(self, next_states, rewards):
        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, next_states)),
            device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in next_states if s is not None])

        q_next = torch.zeros((self.BATCH_SIZE, self.n_actions), device=self.device)
        q_next[non_final_mask] = self.target_net(non_final_next_states).detach()

        # get next action by policy net which is different to DQN
        next_a = torch.zeros(self.BATCH_SIZE, device=self.device, dtype=torch.long)
        next_a[non_final_mask] = self.policy_net(non_final_next_states).max(1)[1].detach()

        expected_q = rewards + self.GAMMA * q_next[np.arange(self.BATCH_SIZE), next_a]

        return expected_q.unsqueeze(1)


class DQNPrioritized(DQN):
    """
    DQN with PER use a different experience replay pool which shows the importance
    of each transition
    """
    def __init__(self, n_actions, device, batch_norm=False, alpha=0.6, beta=0.4, eps=1e-6):
        # alpha, beta, eps are suggested in the paper
        self.alpha = alpha
        super().__init__(n_actions, device, batch_norm)
        self.beta = LinearAnneal(beta, 1.0, self.EXPLORE_STEP)
        self.eps = eps

    def _memory_init(self):
        self.memory = PrioritizedReplayMemory(self.MEMORY_SIZE, self.alpha)

    def _optimize(self):
        """Sample batch from experience replay pool and update the policy"""
        if len(self.memory) < self.BATCH_SIZE:
            return
        transitions, weights, idxes = self.memory.sample(self.BATCH_SIZE, self.beta.anneal())
        batch = Transition(*zip(*transitions))

        states = torch.cat(batch.state)
        actions = torch.cat(batch.action)
        rewards = torch.cat(batch.reward)
        weights = torch.tensor(weights, dtype=torch.float32, device=self.device)

        q = self._q(states, actions)
        expected_q = self._expected_q(batch.next_state, rewards)

        # update the priority of each transition
        td_error = expected_q - q
        new_priorities = torch.abs(td_error) + self.eps
        self.memory.update_priorities(idxes, new_priorities.flatten())

        # Compute Huber loss
        loss = F.smooth_l1_loss(q, expected_q, reduction='none')
        loss = (weights * loss).mean()

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
