import gymnasium as gym
import numpy as np
import torch

from .action import ActionSpace
from .state import State, ObservationSpace


class Env:
    GOAL_STATE = tuple((i, 0) for i in range(8)) + tuple((i, 0) for i in range(12))
    action_space = ActionSpace()
    observation_space = ObservationSpace()
    state = None
    args = None
    _step_count = 0

    def __init__(self, state=None, args=None):
        if state is None: state = State(state=self.GOAL_STATE)
        if args is None: args = dict()
        self.state = state
        self.args = args

    def reward(self, state, action, next_state):
        return 1.0 if np.array_equal(next_state.state, self.GOAL_STATE) else 0.0

    def step(self, action, state=None):
        if state is None: state=self.state
        next_state = State(state=state)

        action(next_state)

        observation = next_state
        reward = self.reward(state, action, next_state)
        terminated = reward == 1.0
        self._step_count += 1
        max_step_count = int(self.args.get('max_step_count', 0))
        truncated = max_step_count == 0 or self._step_count >= max_step_count
        info = None # dict

        return (observation, reward, terminated, truncated, info)
