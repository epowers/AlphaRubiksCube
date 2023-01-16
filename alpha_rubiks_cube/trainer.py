import os
import numpy as np
from random import shuffle

import torch
import torch.optim as optim

from .mcts import MCTS
from .state import State


class Trainer:
    def __init__(self, env, model, args):
        self.env = env
        self.model = model
        self.args = args
        self.mcts = MCTS(self.env, self.model, self.args)

    def execute_episode(self, state):
        train_examples = []

        for i in range(self.args['numItersForTrainExamplesHistory']):
            self.mcts = MCTS(self.env, self.model, self.args)
            root = self.mcts.run(self.model, state)

            action_probs = [0 for _ in range(self.env.get_action_size())]
            for k, v in root.children.items():
                action_probs[k] = v.visit_count

            action_probs = action_probs / np.sum(action_probs)
            train_examples.append((state, action_probs))

            action = root.select_action(temperature=0)
            state = self.env.get_next_state(state, action)
            reward = self.env.get_reward(state)

            if reward:
                return [(hist_state, hist_action_probs, reward)
                                  for hist_state, hist_action_probs in train_examples]

        return []

    def learn(self):
        depth = 1

        for i in range(1, self.args['numIters'] + 1):
            print("Learning iteration {}/{}".format(i, self.args['numIters']))
            train_examples = []

            for eps in range(self.args['numEps']):
                state = self.env.observation_space.sample(depth=np.random.randint(1, depth+1))
                iteration_train_examples = self.execute_episode(state)
                train_examples.extend(iteration_train_examples)

            print('Generated {} training examples with depth between 1 and {}.'.format(len(train_examples), depth))
            shuffle(train_examples)
            pi_loss, v_loss = self.train(train_examples)
            filename = self.args['checkpoint_path']
            self.save_checkpoint(folder="checkpoints", filename=filename)

            if pi_loss < 10e-1 and v_loss < 10e-3:
                depth += 1

    def train(self, examples):
        num_examples = len(examples)

        optimizer = optim.Adam(self.model.parameters(), lr=5e-4)
        pi_losses = []
        v_losses = []

        self.model.train()

        print()

        for epoch in range(self.args['epochs']):
            for batch_idx in range(max(1, num_examples // self.args['batch_size']) if num_examples else 0):
                sample_ids = np.random.randint(num_examples, size=min(num_examples, self.args['batch_size']))
                states, pis, vs = list(zip(*[examples[i] for i in sample_ids]))
                states = tuple(np.array(state._state, dtype=np.float64).flatten() for state in states)
                states = torch.FloatTensor(np.array(states, dtype=np.float64))
                target_pis = torch.FloatTensor(np.array(pis))
                target_vs = torch.FloatTensor(np.array(vs, dtype=np.float64))

                # predict
                device = self.model.device
                states = states.contiguous().to(device)
                target_pis = target_pis.contiguous().to(device)
                target_vs = target_vs.contiguous().to(device)

                # compute output
                out_pi, out_v = self.model(states)
                l_pi = self.loss_pi(target_pis, out_pi)
                l_v = self.loss_v(target_vs, out_v)
                total_loss = l_pi + l_v

                pi_losses.append(float(l_pi))
                v_losses.append(float(l_v))

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                print("Examples:")
                print(out_pi[0])
                print(target_pis[0])

        if pi_losses and v_losses:
            pi_loss = np.mean(pi_losses)
            v_loss = np.mean(v_losses)
        else:
            pi_loss = float('inf')
            v_loss = float('inf')

        print("Policy Loss", pi_loss)
        print("Value Loss", v_loss)
        return pi_loss, v_loss

    def loss_pi(self, targets, outputs):
        loss = -(targets * torch.log(outputs)).sum(dim=1)
        return loss.mean()

    def loss_v(self, targets, outputs):
        loss = torch.sum((targets-outputs.view(-1))**2)/targets.size()[0]
        return loss

    def save_checkpoint(self, folder, filename):
        if not os.path.exists(folder):
            os.mkdir(folder)

        filepath = os.path.join(folder, filename)
        checkpoint = {
            'state_dict': self.model.state_dict(),
        }
        torch.save(checkpoint, filepath)
