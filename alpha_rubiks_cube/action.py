import numpy as np


class Action:
    def __init__(self, value):
        assert(value >= 0 and value < 6)
        self.value = value

    def __int__(self):
        return int(self.value)

    def __repr__(self):
        return f'Action({int(self.value)})'

    def __call__(self, state):
        t = state._state

        if self.value == 0: # rotate posX
            # corners
            c = t[0:8]
            i4 = c[:, 0] == 4
            i5 = c[:, 0] == 5
            i6 = c[:, 0] == 6
            i7 = c[:, 0] == 7
            c[i4, 0] = 5
            c[i5, 0] = 6
            c[i6, 0] = 7
            c[i7, 0] = 4
            # edges
            e = t[8:20]
            i8 = e[:, 0] == 8
            i9 = e[:, 0] == 9
            i10 = e[:, 0] == 10
            i11 = e[:, 0] == 11
            e[i8, 0] = 9
            e[i9, 0] = 10
            e[i10, 0] = 11
            e[i11, 0] = 8
        elif self.value == 1: # rotate zeroX
            # edges
            e = t[8:20]
            i4 = e[:, 0] == 4
            i5 = e[:, 0] == 5
            i6 = e[:, 0] == 6
            i7 = e[:, 0] == 7
            e[i4, 0] = 5
            e[i5, 0] = 6
            e[i6, 0] = 7
            e[i7, 0] = 4
        elif self.value == 2: # rotate posY
            # corners
            c = t[0:8]
            i1 = c[:, 0] == 1
            i2 = c[:, 0] == 2
            i6 = c[:, 0] == 6
            i5 = c[:, 0] == 5
            c[i1, 0] = 2
            c[i2, 0] = 6
            c[i6, 0] = 5
            c[i5, 0] = 1
            # edges
            e = t[8:20]
            i1 = e[:, 0] == 1
            i5 = e[:, 0] == 5
            i9 = e[:, 0] == 9
            i4 = e[:, 0] == 4
            e[i1, 0] = 5
            e[i5, 0] = 9
            e[i9, 0] = 4
            e[i4, 0] = 1
        elif self.value == 3: # rotate zeroY
            # edges
            e = t[8:20]
            i4 = e[:, 0] == 4
            i5 = e[:, 0] == 5
            i6 = e[:, 0] == 6
            i7 = e[:, 0] == 7
            e[i4, 0] = 5
            e[i5, 0] = 6
            e[i6, 0] = 7
            e[i7, 0] = 4
        elif self.value == 4: # rotate posZ
            # corners
            c = t[0:8]
            i4 = c[:, 0] == 4
            i5 = c[:, 0] == 5
            i6 = c[:, 0] == 6
            i7 = c[:, 0] == 7
            c[i4, 0] = 5
            c[i5, 0] = 6
            c[i6, 0] = 7
            c[i7, 0] = 4
            # edges
            e = t[8:20]
            i8 = e[:, 0] == 8
            i9 = e[:, 0] == 9
            i10 = e[:, 0] == 10
            i11 = e[:, 0] == 11
            e[i8, 0] = 9
            e[i9, 0] = 10
            e[i10, 0] = 11
            e[i11, 0] = 8
        elif self.value == 5: # rotate zeroZ
            # edges
            e = t[8:20]
            i4 = e[:, 0] == 4
            i5 = e[:, 0] == 5
            i6 = e[:, 0] == 6
            i7 = e[:, 0] == 7
            e[i4, 0] = 5
            e[i5, 0] = 6
            e[i6, 0] = 7
            e[i7, 0] = 4


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
