class TestAction:
    def test_01_action_import(self):
        import alpha_rubiks_cube.action

    def test_02_action_create(self):
        from alpha_rubiks_cube.action import Action
        action = Action(0)

    def test_03_action_call(self):
        from alpha_rubiks_cube.action import Action
        from alpha_rubiks_cube.state import State
        action = Action(0)
        state = State()
        action(state)

    def test_04_action_space(self):
        from alpha_rubiks_cube.action import ActionSpace
        import numpy as np
        action_space = ActionSpace()
        action_space.sample()

        mask = np.array([0, 0, 1, 0, 0, 0], dtype=np.int8)
        action = action_space.sample(mask=mask)
        assert int(action) == 2, "Action::sample() with a mask failed to select the only valid action."

        mask = np.array([0, 0, 1, 0, 0, 0], dtype=np.int8)
        action = action_space.pop_sample(mask=mask)
        assert int(action) == 2, "Action::pop_sample() with a mask failed to select the only valid action."

    def test_05_action_env_step(self):
        from alpha_rubiks_cube.env import Env
        env = Env()
        action = env.action_space.sample()
        env.step(action)
