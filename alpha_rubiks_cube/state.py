from typing import Any, Iterable, Mapping, Sequence
import numpy as np
import torch


class State:
    '''Rubik Cube
        States:
            8 corner pieces
            8 corner positions
            3 corner orientations
            12 edge pieces
            12 edge positions
            2 edge orientations
        
        All states can be represented one-hot in a (2,12,12,3) binary sparse tensor
        Or a flat 8 * 11 + 12 * 14 one-hot binary dense tensor.
        The sparse binary feature vector always has an element-wise sum of 20,
        where position and orienation are represented in dimensions.
        The dense binary feature vector always has an element-wise sum of 40,
        one for position and one for orientation.
    '''
    COMPACT_STATE_SHAPE = (20, 2)
    OBSERVATION_SPACE_SHAPE = ((8, 3),) * 8 + ((12, 2),) * 12
    SPARSE_STATE_SHAPE = (2, 12, 12, 3)
    DENSE_STATE_SHAPE = (256,) # (8 * (8 + 3) + 12 * (12 + 2),)
    _state = None

    def __str__(self):
        return str(self._state)

    def __init__(self, state=None):
        if state is None:
            self._state = torch.zeros(size=self.COMPACT_STATE_SHAPE, dtype=torch.long)
        elif isinstance(state, State):
            self._state = torch.clone(state._state)
        elif isinstance(state, torch.Tensor) and state.size() == self.COMPACT_STATE_SHAPE:
            self._state = torch.clone(state)
        else:
            self.set_state(state)

    @property
    def is_valid(self):
        return torch.any(self._state != 0)

    def set_state(self, state):
        self._state = torch.tensor(state, dtype=torch.long)

    def set_state_from_pos_orient(self, state):
        for i, (position, orientation) in zip(range(20), state):
            if i < 8:
                assert(position >= 0 and position < 8 and orientation >= 0 and orientation < 3)
            else:
                assert(position >= 0 and position < 12 and orientation >= 0 and orientation < 2)
        self._state = torch.tensor(state, dtype=torch.long)

    def to_tensor(self):
        '''Convert an array of:

                (8 corner position elements concatenated with 12 edge position elements)
                    where each corner position element has 2 subarray values:
                        (8 corner pieces, 3 corner orientations)
                    and each edge position element has 2 subarray values:
                        (12 edge pieces, 2 edge orientations)

            ... to a sparse tensor:

                (0, corner position, corner piece, corner orientation) => 1
                    or
                (1, edge position, edge piece, edge orientation) => 1

            ... or a dense tensor with 256 elements:

                (
                    8 groups of 8 one-hot encoded corner positions
                    plus 3 one-hot encoded corner orientations

                    ... concatenated with ...

                    12 groups of 12 one-hot encoded edge positions
                    plus 2 one-hot encoded edge orientations
                )

            By example, a solved cube:
                (
                 # corners
                 (0, 0), (1, 0), (2, 0), (3, 0),
                 (4, 0), (5, 0), (6, 0), (7, 0),
                 # edges
                 (0, 0), (1, 0), (2, 0), (3, 0),
                 (4, 0), (5, 0), (6, 0), (7, 0),
                 (8, 0), (9, 0), (10, 0), (11, 0)
                )

            ... as a sparse tensor:

                indices = torch.tensor([
                    [0, 0, 0, 0],
                    [0, 1, 1, 0],
                    [0, 2, 2, 0],
                    [0, 3, 3, 0],
                    [0, 4, 4, 0],
                    [0, 5, 5, 0],
                    [0, 6, 6, 0],
                    [0, 7, 7, 0],
                    [1, 0, 0, 0],
                    [1, 1, 1, 0],
                    [1, 2, 2, 0],
                    [1, 3, 3, 0],
                    [1, 4, 4, 0],
                    [1, 5, 5, 0],
                    [1, 6, 6, 0],
                    [1, 7, 7, 0],
                    [1, 8, 8, 0],
                    [1, 9, 9, 0],
                    [1, 10, 10, 0],
                    [1, 11, 11, 0]
                ]).t()

                values = torch.tensor([
                    1, 1, 1, 1, 1, 1, 1, 1,
                    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
                ]).t()

                size = (2, 12, 12, 3)

                dtype = torch.long

            ... or as a dense tensor:

                [
                    # corner 0 position one-hot encoded
                    1, 0, 0, 0, 0, 0, 0, 0,
                    # corner 0 orientation one-hot encoded
                    1, 0, 0,
                    # corner 1 position one-hot encoded
                    0, 1, 0, 0, 0, 0, 0, 0,
                    # corner 1 orientation one-hot encoded
                    1, 0, 0,
                    ...
                    # corner 7 position one-hot encoded
                    0, 0, 0, 0, 0, 0, 0, 1,
                    # corner 7 orientation one-hot encoded
                    1, 0, 0,
                    # edge 0 position one-hot encoded
                    1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    # edge 0 orientation one-hot encoded
                    1, 0,
                    ...
                    # edge 11 position one-hot encoded
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
                    # edge 11 orientation one-hot encoded
                    1, 0
                ]

        '''
        result = torch.zeros(size=self.DENSE_STATE_SHAPE, dtype=torch.long)
        if self.is_valid:
            for i, (position, orientation) in zip(range(20), self._state):
                if i < 8:
                    i_position = i * 11 + position
                    i_orientation = i * 11 + 8 + orientation
                else:
                    i_position = 8 * 11 + (i - 8) * 14 + position
                    i_orientation = 8 * 11 + (i - 8) * 14 + 12 + orientation
                result[i_position] = 1
                result[i_orientation] = 1
        return result


class ObservationSpace:
    @classmethod
    def get_size(cls):
        return 40

    def sample(self, mask=None, depth=None):
        # scramble state with random actions
        from .action import Action, ActionSpace
        from .env import Env
        action_space = ActionSpace()
        state = Env.get_init_state()
        if depth is None:
            depth = np.random.randint(self.get_size() * ActionSpace.get_size())

        for epoch in range(depth):
            action = ActionSpace().sample()
            action.reverse = True
            state = action(state)
            state = State(state)

        return state
