import numpy as np
import torch


class Action:
    def __init__(self, value):
        assert(value >= 0 and value < 6)
        self.value = value

    def __int__(self):
        return int(self.value)

    def __repr__(self):
        return f'Action({int(self.value)})'

    def __call__(self, state):
        a = self.value
        t = state._state

        c_sel, e_sel = {
            0: ((4,5,6,7), (8,9,10,11)), # rotate posX
            1: ((), (4,5,6,7)), # rotate zeroX
            2: ((1,2,6,5), (1,6,9,5)), # rotate posY
            3: ((), (0,2,10,8)), # rotate zeroY
            4: ((2,3,7,6), (2,7,10,6)), # rotate posZ
            5: ((), (1,3,11,9)), # rotate zeroZ
        }.get(a)

        if a % 2 == 0: # rotate corners
            # corners
            c = t[0:8]
            i = torch.argwhere(torch.isin(c[:, 0], torch.tensor(c_sel)))
            # rotate corner orientations
            c[i, 1] = (c[i, 1] + 1) % 3
            # rotate corner positions
            c[i] = torch.roll(c[i], -1, dims=0)

        # rotate edges
        e = t[8:20]
        i = torch.argwhere(torch.isin(e[:, 0], torch.tensor(e_sel)))
        # rotate edge orientations
        e[i, 1] = (e[i, 1] + 1) % 2
        # rotate edge positions
        e[i] = torch.roll(e[i], -1, dims=0)


class ActionSpace:
    @classmethod
    def get_size(cls):
        return 6

    def sample(self, mask=None):
        if mask is not None:
            valid_action_mask = mask == 1
            if np.any(valid_action_mask):
                i = np.random.choice(
                    np.where(valid_action_mask)[0]
                )
                mask[i] = 0
                result = i
            else:
                result = 0
        else:
            n = self.get_size()
            result = np.random.randint(n)
        return Action(result)

    def pop_sample(self, mask):
        valid_action_mask = mask == 1
        if np.any(valid_action_mask):
            i = np.random.choice(
                np.where(valid_action_mask)[0]
            )
            mask[i] = 0
            result = i
        else:
            result = 0
        return Action(result)
