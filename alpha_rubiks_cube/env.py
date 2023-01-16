import numpy as np
import torch

from .action import Action, ActionSpace
from .state import State, ObservationSpace


class Env:
    GOAL_STATE = tuple((i, 0) for i in range(8)) + tuple((i, 0) for i in range(12))
    action_space = ActionSpace()
    observation_space = ObservationSpace()
    state = None
    args = None
    _step_count = 0

    def __init__(self, state=None, args=None):
        if state is None: state = self.get_init_state()
        if args is None: args = dict()
        self.state = state
        self.args = args

    @classmethod
    def get_init_state(cls):
        return State(state=cls.GOAL_STATE)

    def get_state_size(self):
        return self.observation_space.get_size()

    def get_action_size(self):
        return self.action_space.get_size()

    def reward(self, state, action, next_state):
        return 1.0 if np.array_equal(next_state._state, self.GOAL_STATE) else 0.0

    def step(self, action, state=None):
        if state is None: state=self.state
        next_state = State(state=state)
        action = Action(action)

        action(next_state)

        observation = next_state
        reward = self.reward(state, action, next_state)
        terminated = reward == 1.0
        self._step_count += 1
        max_step_count = int(self.args.get('max_step_count', 0))
        truncated = max_step_count == 0 or self._step_count >= max_step_count
        info = None # dict

        return (observation, reward, terminated, truncated, info)

    def get_next_state(self, state, action):
        observation, reward, terminated, truncated, info = self.step(action, state=state)
        return observation

    def is_win(self, state):
        return np.array_equal(state._state, self.GOAL_STATE)

    def get_reward(self, state):
        return 1 if self.is_win(state) else None
