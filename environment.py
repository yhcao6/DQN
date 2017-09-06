import gym
import cv2
import random
import numpy as np


class Environment(object):
    def __init__(self):
        self.env = gym.make('Breakout-v0')
        self.env = self.env.unwrapped

        self.display = True
        self.screen_width = 84
        self.screen_height = 84
        self.dims = (self.screen_height, self.screen_width)
        self.random_start = 30
        self.action_repeat = 1

        self._screen = None
        self.reward = 0
        self.terminal = True

    def new_game(self, from_random_game=False):
        if self.lives == 0:
            self._screen = self.env.reset()
        self._step(0)
        self.render()
        return self.screen, 0, 0, self.terminal

    def new_randm_game(self):
        self.new_game(True)
        for _ in xrange(random.randint(0, self.random_start-1)):
            self._step(0)
        self.render()
        return self.screen, 0, 0, self.terminal

    def act(self, action, is_training=True):
        cumulated_reward = 0
        start_lives = self.lives

        for _ in xrange(self.action_repeat):
            self._step(action)
            cumulated_reward += self.reward

            if is_training and start_lives > self.lives:
                cumulated_reward -= 1
                self.terminal = True

            if self.terminal:
                break

        self.reward = cumulated_reward

        self.after_act(action)
        return self.state

    @property
    def screen(self):
        return cv2.resize(np.dot((self._screen/255)[..., :3], [0.299, 0.587, 0.114]), self.dims)

    @property
    def action_size(self):
        return self.env.action_space.n

    @property
    def lives(self):
        return self.env.ale.lives()

    @property
    def state(self):
        return self.screen, self.reward, self.terminal

    def _step(self, action):
        self._screen, self.reward, self.terminal, _ = self.env.step(action)

    def _random_step(self):
        action = self.env.action_space.sample()
        self._step(action)

    def render(self):
        if self.display:
            self.env.render()

    def after_act(self, action):
        self.render()


