class TestState:
    def test_01_state_import(self):
        import alpha_rubiks_cube.state

    def test_02_default_state(self):
        from alpha_rubiks_cube.state import State
        state = State()

    def test_03_state_to_tensor(self):
        from alpha_rubiks_cube.state import State
        state = State()
        state.to_tensor()

    def test_04_state_observation_space(self):
        from alpha_rubiks_cube.state import ObservationSpace
        observation_space = ObservationSpace()
        observation_space.sample()

class TestEnv:
    def test_01_env_import(self):
        import alpha_rubiks_cube.env

    def test_02_env_create(self):
        from alpha_rubiks_cube.env import Env
        env = Env()

    def test_03_env_reward(self):
        from alpha_rubiks_cube.env import Env
        env = Env()
        reward = env.reward(env.state, None, env.state)
        assert reward == 1.0, "Env::reward() in the default/goal state should return 1.0"
