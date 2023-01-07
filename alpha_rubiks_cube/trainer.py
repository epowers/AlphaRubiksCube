import os
import numpy as np
from random import shuffle

import torch
import torch.optim as optim

from .mcts import MCTS


class Trainer:
    def __init__(self, env, model, args):
        self.env = env
        self.model = model
        self.args = args
        self.mcts = MCTS(self.env, self.model, self.args)

    def exceute_episode(self):
        train_examples = []
        state = self.env.get_init_state()

        while True:
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

            if reward is not None:
                ret = []
                for hist_state, hist_action_probs in train_examples:
                    # [state, actionProbabilities, reward]
                    ret.append((hist_state, hist_action_probs, reward))

                return ret

    def learn(self):
        for i in range(1, self.args['numIters'] + 1):
            # print("{}/{}".format(i, self.args['numIters']))
            train_examples = []

            for eps in range(self.args['numEps']):
                iteration_train_examples = self.exceute_episode()
                train_examples.extend(iteration_train_examples)

            shuffle(train_examples)
            self.train(train_examples)
            filename = self.args['checkpoint_path']
            self.save_checkpoint(folder="checkpoints", filename=filename)

    def train(self, examples):
        optimizer = optim.Adam(self.model.parameters(), lr=5e-4)
        pi_losses = []
        v_losses = []

        for epoch in range(self.args['epochs']):
            self.model.train()

            batch_idx = 0

            while batch_idx < int(len(examples) / self.args['batch_size']):
                sample_ids = np.random.randint(len(examples), size=self.args['batch_size'])
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

                batch_idx += 1

            print()
            print("Policy Loss", np.mean(pi_losses))
            print("Value Loss", np.mean(v_losses))
            print("Examples:")
            print(out_pi[0].detach())
            print(target_pis[0])

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
        torch.save({
            'state_dict': self.model.state_dict(),
        }, filepath)
