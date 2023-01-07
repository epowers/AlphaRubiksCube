import numpy as np
import unittest
from alpha_rubiks_cube.mcts import Node, MCTS, ucb_score
from alpha_rubiks_cube.env import Env


class MCTSTests(unittest.TestCase):

    def test_mcts_from_root_with_equal_priors(self):
        class MockModel:
            def predict(self, state):
                return np.array([0.26, 0.24, 0.24, 0.24, 0.24, 0.24]), 0.0001

        env = Env()
        args = {'num_simulations': 50}

        model = MockModel()
        mcts = MCTS(env, model, args)
        canonical_state = [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0), (7, 0), (0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0), (7, 0), (8, 0), (9, 0), (10, 0), (11, 0)]
        print("starting")
        root = mcts.run(model, canonical_state)

        # the best move is to play at index 1 or 2
        best_outer_move = max(root.children[0].visit_count, root.children[0].visit_count)
        best_center_move = max(root.children[1].visit_count, root.children[2].visit_count)
        #self.assertGreater(best_center_move, best_outer_move)

    def test_mcts_finds_best_move_with_really_bad_priors(self):
        class MockModel:
            def predict(self, state):
                return np.array([0.3, 0.7, 0, 0, 0, 0]), 0.0001

        env = Env()
        args = {'num_simulations': 25}

        model = MockModel()
        mcts = MCTS(env, model, args)
        canonical_state = [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0), (7, 0), (0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0), (7, 0), (8, 0), (9, 0), (10, 0), (11, 0)]
        print("starting")
        root = mcts.run(model, canonical_state)

        # the best move is to play at index 1
        self.assertGreater(root.children[1].visit_count, root.children[0].visit_count)

    def test_mcts_finds_best_move_with_equal_priors(self):

        class MockModel:
            def predict(self, state):
                return np.array([0.51, 0.49, 0, 0, 0, 0]), 0.0001

        env = Env()
        args = { 'num_simulations': 25 }

        model = MockModel()
        mcts = MCTS(env, model, args)
        canonical_state = [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0), (7, 0), (0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0), (7, 0), (8, 0), (9, 0), (10, 0), (11, 0)]
        root = mcts.run(model, canonical_state)

        # the better move is to play at index 1
        #self.assertLess(root.children[0].visit_count, root.children[1].visit_count)

    def test_mcts_finds_best_move_with_really_really_bad_priors(self):
        class MockModel:
            def predict(self, state):
                return np.array([0, 0.3, 0.3, 0.3, 0, 0]), 0.0001

        env = Env()
        args = {'num_simulations': 100}

        model = MockModel()
        mcts = MCTS(env, model, args)
        canonical_state = [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0), (7, 0), (0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0), (7, 0), (8, 0), (9, 0), (10, 0), (11, 0)]
        root = mcts.run(model, canonical_state)

        # the best move is to play at index 1
        self.assertGreater(root.children[1].visit_count, root.children[2].visit_count)
        self.assertGreater(root.children[1].visit_count, root.children[3].visit_count)

class NodeTests(unittest.TestCase):

    def test_initialization(self):
        node = Node(0.5)

        self.assertEqual(node.visit_count, 0)
        self.assertEqual(node.prior, 0.5)
        self.assertEqual(len(node.children), 0)
        self.assertFalse(node.expanded())
        self.assertEqual(node.value(), 0)

    def test_selection(self):
        node = Node(0.5)
        c0 = Node(0.5)
        c1 = Node(0.5)
        c2 = Node(0.5)
        node.visit_count = 1
        c0.visit_count = 0
        c2.visit_count = 0
        c2.visit_count = 1

        node.children = {
            0: c0,
            1: c1,
            2: c2,
        }

        action = node.select_action(temperature=0)
        self.assertEqual(action, 2)

    def test_expansion(self):
        node = Node(0.5)

        state = [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0), (7, 0), (0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0), (7, 0), (8, 0), (9, 0), (10, 0), (11, 0)]
        action_probs = [0.25, 0.15, 0.5, 0.1, 0.1, 0.1]

        node.expand(state, action_probs)

        self.assertEqual(len(node.children), 6)
        self.assertTrue(node.expanded())
        self.assertEqual(node.children[0].prior, 0.25)
        self.assertEqual(node.children[1].prior, 0.15)
        self.assertEqual(node.children[2].prior, 0.50)
        self.assertEqual(node.children[3].prior, 0.10)

    def test_ucb_score_no_children_visited(self):
        node = Node(0.5)
        node.visit_count = 1

        state = [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0), (7, 0), (0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0), (7, 0), (8, 0), (9, 0), (10, 0), (11, 0)]
        action_probs = [0.25, 0.15, 0.5, 0.1, 0.1, 0.1]

        node.expand(state, action_probs)
        node.children[0].visit_count = 0
        node.children[1].visit_count = 0
        node.children[2].visit_count = 0
        node.children[3].visit_count = 0

        score_0 = ucb_score(node, node.children[0])
        score_1 = ucb_score(node, node.children[1])
        score_2 = ucb_score(node, node.children[2])
        score_3 = ucb_score(node, node.children[3])

        # With no visits, UCB score is just the priors
        self.assertEqual(score_0, node.children[0].prior)
        self.assertEqual(score_1, node.children[1].prior)
        self.assertEqual(score_2, node.children[2].prior)
        self.assertEqual(score_3, node.children[3].prior)

    def test_ucb_score_one_child_visited(self):
        node = Node(0.5)
        node.visit_count = 1

        state = [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0), (7, 0), (0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0), (7, 0), (8, 0), (9, 0), (10, 0), (11, 0)]
        action_probs = [0.25, 0.15, 0.5, 0.1, 0.1, 0.1]

        node.expand(state, action_probs)
        node.children[0].visit_count = 0
        node.children[1].visit_count = 0
        node.children[2].visit_count = 1
        node.children[3].visit_count = 0

        score_0 = ucb_score(node, node.children[0])
        score_1 = ucb_score(node, node.children[1])
        score_2 = ucb_score(node, node.children[2])
        score_3 = ucb_score(node, node.children[3])

        # With no visits, UCB score is just the priors
        self.assertEqual(score_0, node.children[0].prior)
        self.assertEqual(score_1, node.children[1].prior)
        # If we visit one child once, its score is halved
        self.assertEqual(score_2, node.children[2].prior / 2)
        self.assertEqual(score_3, node.children[3].prior)

        action, child = node.select_child()

        self.assertEqual(action, 0)

    def test_ucb_score_one_child_visited_twice(self):
        node = Node(0.5)
        node.visit_count = 2

        state = [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0), (7, 0), (0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0), (7, 0), (8, 0), (9, 0), (10, 0), (11, 0)]
        action_probs = [0.25, 0.15, 0.5, 0.1, 0.1, 0.1]

        node.expand(state, action_probs)
        node.children[0].visit_count = 0
        node.children[1].visit_count = 0
        node.children[2].visit_count = 2
        node.children[3].visit_count = 0

        score_0 = ucb_score(node, node.children[0])
        score_1 = ucb_score(node, node.children[1])
        score_2 = ucb_score(node, node.children[2])
        score_3 = ucb_score(node, node.children[3])

        action, child = node.select_child()

        # Now that we've visited the second action twice, we should
        # end up trying the first action
        self.assertEqual(action, 0)

    def test_ucb_score_no_children_visited(self):
        node = Node(0.5)
        node.visit_count = 1

        state = [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0), (7, 0), (0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0), (7, 0), (8, 0), (9, 0), (10, 0), (11, 0)]
        action_probs = [0.25, 0.15, 0.5, 0.1, 0.1, 0.1]

        node.expand(state, action_probs)
        node.children[0].visit_count = 0
        node.children[1].visit_count = 0
        node.children[2].visit_count = 1
        node.children[3].visit_count = 0

        score_0 = ucb_score(node, node.children[0])
        score_1 = ucb_score(node, node.children[1])
        score_2 = ucb_score(node, node.children[2])
        score_3 = ucb_score(node, node.children[3])

        # With no visits, UCB score is just the priors
        self.assertEqual(score_0, node.children[0].prior)
        self.assertEqual(score_1, node.children[1].prior)
        # If we visit one child once, its score is halved
        self.assertEqual(score_2, node.children[2].prior / 2)
        self.assertEqual(score_3, node.children[3].prior)
