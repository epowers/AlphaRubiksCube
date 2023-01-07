#!/usr/bin/env python

import torch

from alpha_rubiks_cube.env import Env
from alpha_rubiks_cube.model import Model
from alpha_rubiks_cube.trainer import Trainer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

args = {
    'batch_size': 64,
    'numIters': 500,                                # Total number of training iterations
    'num_simulations': 100,                         # Total number of MCTS simulations to run when deciding on a move to play
    'numEps': 100,                                  # Number of full envs (episodes) to run during each iteration
    'numItersForTrainExamplesHistory': 20,
    'epochs': 2,                                    # Number of epochs of training per iteration
    'checkpoint_path': 'latest.pth'                 # location to save latest set of weights
}

env = Env()
state_size = env.get_state_size()
action_size = env.get_action_size()

model = Model(state_size, action_size, device)

trainer = Trainer(env, model, args)
trainer.learn()
