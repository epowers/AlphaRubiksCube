import os
import numpy as np
import torch


class Solver:
    def __init__(self, env, model, args):
        self.env = env
        self.model = model
        self.args = args

    def solve(self):
        device = self.model.device

        # load model checkpoint
        filename = self.args['checkpoint_path']
        self.load_checkpoint(folder="checkpoints", filename=filename)

        # model eval mode
        self.model.eval()

        # randomize cube
        state = self.env.observation_space.sample()

        # solve cube
        while not self.env.is_win(state):
            # prepare state
            states = np.array(state._state, dtype=np.float64).flatten().reshape((1,-1))
            states = torch.FloatTensor(states)
            states = states.contiguous().to(device)

            # predict
            out_pi, out_v = self.model(states)
            action = torch.argmax(out_pi).item()
            print(state, action, out_pi.cpu(), out_v.cpu())

            state = self.env.get_next_state(state, action)

    def load_checkpoint(self, folder, filename):
        filepath = os.path.join(folder, filename)
        map_location = None if self.args.get('cuda') else 'cpu'
        checkpoint = torch.load(filepath, map_location=map_location)
        self.model.load_state_dict(checkpoint['state_dict'])
