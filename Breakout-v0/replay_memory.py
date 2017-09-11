import numpy as np
import random


class ReplayMemory:
    def __init__(self):
        self.memory_size = 1000000
        self.actions = np.empty(self.memory_size, dtype = np.uint8)
        self.rewards = np.empty(self.memory_size, dtype = np.integer)
        self.screen_height = 84
        self.screen_width = 84
        self.screens = np.empty((self.memory_size, self.screen_height, self.screen_width), dtype = np.float16)
        self.terminals = np.empty(self.memory_size, dtype = np.bool)
        self.history_length = 4
        self.dims = (self.screen_height, self.screen_width)
        self.batch_size = 32
        self.count = 0
        self.current = 0

        self.prestates = np.empty((self.batch_size, self.history_length) + self.dims, dtype = np.float16)
        self.poststates = np.empty((self.batch_size, self.history_length) + self.dims, dtype = np.float16)

    def add(self, screen, reward, action, terminal):
        assert screen.shape == self.dims
        self.actions[self.current] = action
        self.rewards[self.current] = reward
        self.screens[self.current, ...] = screen
        self.terminals[self.current] = terminal
        self.count = max(self.count, self.current + 1)
        self.current = (self.current + 1) % self.memory_size

    def getState(self, index):
        assert self.count > 0
        index = index % self.count
        if index >= self.history_length - 1:
            return self.screens[(index - (self.history_length - 1)): (index + 1), ...]
        else:
            indexes = [(index - i) % self.count for i in reversed(range(self.history_length))]
            return self.screens[indexes, ...]

    def sample(self):
        assert self.count > self.history_length

        indexes = []
        while len(indexes) < self.batch_size:
            while True:
                index = random.randint(self.history_length, self.count - 1)

                if index >= self.current and index - self.history_length < self.current:
                    continue
                if self.terminals[(index - self.history_length):index].any():
                    continue
                break

            self.prestates[len(indexes), ...] = self.getState(index - 1)
            self.poststates[len(indexes), ...] = self.getState(index)
            indexes.append(index)

        actions = self.actions[indexes]
        rewards = self.rewards[indexes]
        terminals = self.terminals[indexes]

        return self.prestates, actions, rewards, self.poststates, terminals



