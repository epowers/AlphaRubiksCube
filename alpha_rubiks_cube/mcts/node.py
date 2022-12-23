from collections import defaultdict
import numpy as np


class Node:
    def __init__(self, env, state, parent=None, reward=None, terminated=False):
        self.env = env
        self.state = state
        self.parent = parent
        self.reward = reward
        self.terminated = terminated
        self.children = []
        self._number_of_visits = 0.
        self._results = defaultdict(int)
        self._untried_actions = None

    def is_fully_expanded(self):
        return not np.any(self.untried_actions == 1)

    def best_child(self, c_param=1.4):
        choices_weights = [
            (c.q / c.n) + c_param * np.sqrt((2 * np.log(self.n) / c.n))
            for c in self.children
        ]
        return self.children[np.argmax(choices_weights)]

    def rollout_policy(self, action_mask=None):        
        return self.env.action_space.sample(mask=action_mask)

    @property
    def untried_actions(self):
        if self._untried_actions is None:
            self._untried_actions = np.ones((self.env.action_space.n,), dtype=np.int8)
        return self._untried_actions

    @property
    def q(self):
        wins = self._results[0]
        loses = self._results[-1 * 0]
        return wins - loses

    @property
    def n(self):
        return self._number_of_visits

    def expand(self):
        action = self.env.action_space.pop_sample(mask=self.untried_actions)
        next_state, reward, terminated, truncated, info = self.env.step(action)
        child_node = Node(env=self.env, state=next_state, parent=self, reward=reward, terminated=terminated)
        self.children.append(child_node)
        return child_node

    def is_terminal_node(self):
        return self.terminated

    def rollout(self):
        current_rollout_state = self.state
        current_reward = self.reward
        current_termination = self.terminated
        truncated = False
        while not current_termination and not truncated:
            action_mask = None # NOTE: check env for actions not allowed at this state
            action = self.rollout_policy(action_mask)
            current_rollout_state, current_reward, current_termination, truncated, info = self.env.step(action, state=current_rollout_state)
        return current_reward

    def backpropagate(self, reward):
        self._number_of_visits += 1.
        self._results[reward] += 1.
        if self.parent is not None:
            self.parent.backpropagate(reward)
