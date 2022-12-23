class TestMCTS:
    def test_01_mcts_import(self):
        import alpha_rubiks_cube.mcts

    def test_02_mcts_create(self):
        from alpha_rubiks_cube.env import Env
        from alpha_rubiks_cube.mcts import Node, Search
        env = Env()
        node = Node(env=env, state=env.state)
        search = Search(node)

    def test_03_mcts_search(self):
        from alpha_rubiks_cube.env import Env
        from alpha_rubiks_cube.mcts import Node, Search
        env = Env(args={'max_step_count':1})
        node = Node(env=env, state=env.state)
        search = Search(node)
        search.best_action(1)
