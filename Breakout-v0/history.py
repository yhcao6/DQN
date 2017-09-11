import numpy as np


class History:
    def __init__(self):
        self.cnn_format = 'NCHW'
        history_length = 4
        screen_height = 84
        screen_width = 84

        self.history = np.zeros([history_length, screen_height, screen_width], dtype=np.float32)

    def add(self, screen):
        self.history[: -1] = self.history[1:]
        self.history[-1] = screen

    def reset(self):
        self.history *= 0

    def get(self):
        return self.history
