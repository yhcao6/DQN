import numpy as np
import random


class Memory(object):
    def __init__(self):
        self.capacity = 100000
        self.pre_state = np.zeros((self.capacity, 4))
        self.next_state = np.zeros((self.capacity, 4))
        self.reward = np.zeros((self.capacity, 1))
        self.action = np.zeros((self.capacity, 1))
        self.terminal = np.zeros((self.capacity, 1))

        self.current = 0
        self.count = 0

    def add(self, s, a, r, s_, terminal):
        self.pre_state[self.current] = s
        self.reward[self.current] = r
        self.action[self.current] = a
        self.next_state[self.current] = s_
        self.terminal[self.current] = terminal

        self.current = (self.current + 1) % self.capacity
        self.count += 1

    def sample(self, n):
        if self.count > self.capacity:
            limit = self.capacity
        else:
            limit = self.count
        index = np.random.choice(limit, n)
        return self.pre_state[index], self.action[index], self.reward[index], self.next_state[index], self.terminal[index]



