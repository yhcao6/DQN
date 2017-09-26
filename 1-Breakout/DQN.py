import numpy as np
from keras import backend as K
from keras.layers import Conv2D, Dense
from keras.models import Sequential
from keras.optimizers import RMSprop
import gym
import random
import cv2
import tensorflow as tf


IMAGE_HEIGHT = 84
IMAGE_WIDTH = 84
HUBER_THRESHOLD = 1
BATCH_SIZE = 32
MEMORY_CAPACITY = 1000000
HISTORY_LENGTH = 4
DISCOUNT = 0.99
ACTION_REPEAT = 4
UPDATE_FREQUENCY = 4
LEARNING_RATE = 0.00025
GRADIENT_MOMENTUM = 0.95
SQUARED_GRADIENT_MOMENTUM = 0.95
MIN_SQUARED_GRADIENT = 0.01
EPSILON_INITIAL = 1
EPSILON_FINAL = 0.1
EPSILON_DECAY_STEPS = 1000000
REPLAY_START = 5e4
NO_OP_MAX = 30
LEARN_START = 50000
REPLAY_FREQUENCY = 4


def huber_loss(y_pred, y_true):
    error = y_pred - y_true
    cond = np.abs(error) < HUBER_THRESHOLD
    l1 = .5 * np.square(error)
    l2 = HUBER_THRESHOLD * (np.abs(error) - .5 * HUBER_THRESHOLD)
    loss = tf.where(cond, l1, l2)
    return K.mean(loss)


def preprocess(im):
    gray = np.dot(im, [0.299, 0.587, 0.114]) / 255.
    res = cv2.resize(gray, (IMAGE_HEIGHT, IMAGE_WIDTH))
    return res[..., np.newaxis].transpose(2, 0, 1)


class Brain:
    def __init__(self, state_size, n_action):
        self.state_size = state_size
        self.n_action = n_action
        self.model = self._build_dqn()  # prediction network
        self.model_ = self._build_dqn() # target network

    def _build_dqn(self):
        model = Sequential()

        model.add(Conv2D(32, (8, 8), strides=(4, 4), activation='relu', input_shape=self.state_size, data_format='channels_first'))
        model.add(Conv2D(64, (4, 4), strides=(2, 2), activation='relu'))
        model.add(Conv2D(64, (3, 3), strides=(1, 1), activation='relu'))

        model.add(Dense(512, activation='relu'))
        model.add(Dense(self.n_action))

        opt = RMSprop(LEARNING_RATE, epsilon=0.01)

        model.compile(optimizer=opt, loss=huber_loss)

        return model

    def train(self, X, y, epochs=1, verbose=0):
        self.model.fit(X, y, epochs=epochs, verbose=verbose)

    def predict(self, X, target=False):
        # one sample
        if len(X.shape) == 3:
            X = X[np.newaxis, ...]

        if target:
            return self.model_.predict(X)
        else:
            return self.model.predict(X)

    def update(self):
        self.model_.set_weights(self.model.get_weights())


class Agent:
    def __init__(self, state_size, n_action):
        self.state_size = state_size
        self.n_action = n_action
        self.brain = Brain(state_size, n_action)
        self.memory = Memory(MEMORY_CAPACITY)
        self.epsilon = EPSILON_INITIAL
        self.global_steps = 0

    def choose_action(self, s):
        if random.random() < self.epsilon:
            return random.randint(0, self.n_action - 1)
        else:
            return np.argmax(self.brain.predict(s))

    def observe(self, sample):
        self.memory.add(sample)

        self.global_steps += 1
        self.epsilon = EPSILON_FINAL + max(0, (EPSILON_INITIAL - EPSILON_FINAL) * (EPSILON_DECAY_STEPS - max(0, self.global_steps - LEARN_START)) / EPSILON_DECAY_STEPS)

    def replay(self):
        s, a, r, s_, terminal = self.memory.sample(BATCH_SIZE)

        p = self.brain.predict(s)
        p_ = self.brain.predict(s_)
        max_p_ = np.max(p_, axis=1)

        y = p
        for i in range(BATCH_SIZE):
            if terminal[i] == 1:
                y[i, a[i]] = r[i]
            else:
                y[i, a[i]] = r[i] + DISCOUNT * max_p_[i]

        self.brain.train(s, y)


class RandomAgent:
    def __init__(self, n_action):
        self.n_action = n_action
        self.memory = Memory(MEMORY_CAPACITY)
        self.steps = 0

    def choose_action(self, s):
        return random.randint(0, self.n_action - 1)

    def replay(self):
        pass

    def observe(self, sample):
        self.memory.add(sample)
        self.steps += 1


class Memory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.states = np.zeros((self.capacity, 1, IMAGE_HEIGHT, IMAGE_WIDTH))
        self.actions = np.zeros((self.capacity, 1), dtype=np.int)
        self.rewards = np.zeros((self.capacity, 1))
        self.terminals = np.zeros((self.capacity, 1), dtype=np.int)
        self.current = 0
        self.n = 0

    def add(self, sample):
        self.states[self.current] = sample[3]
        self.actions[self.current] = sample[1]
        self.rewards[self.current] = sample[2]
        self.terminals[self.current] = sample[4]
        self.current = (self.current + 1) % self.capacity
        self.n += 1

    def sample(self, n):
        # indexes of post_states
        if self.n <= self.capacity:
            indexes = random.sample(range(4, self.current), n)
        else:
            indexes = random.sample(range(self.current + 4, self.current + self.capacity), n)

        pre_states = np.zeros((n, HISTORY_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH))
        post_states = np.zeros((n, HISTORY_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH))
        rewards = np.zeros((n, 1))
        actions = np.zeros((n, 1), dtype=np.int)
        terminals = np.zeros((n, 1), dtype=np.int)

        for i in range(len(indexes)):
            index = (indexes[i]) % self.capacity
            pre_states[i] = self.states[index - 4: index, 0, ...]
            post_states[i] = self.states[index - 3: index + 1, 0, ...]
            actions[i] = self.actions[index, ...]
            rewards[i] = self.rewards[index, ...]
            terminals[i] = self.terminals[index, ...]

        return (pre_states, actions.flatten(), rewards.flatten(), post_states, terminals.flatten())


class Environment:
    def __init__(self, problem):
        self.problem = problem
        self.env = gym.make(problem).unwrapped
        self.episodes = 0
        self.trace_r = []

    def step(self, a):
        states = []
        cumulated_reward = 0
        start_lives = self.env.ale.lives()
        for i in range(ACTION_REPEAT):
            s_, r, done, info = self.env.step(a)
            states.append(s_)
            cumulated_reward += np.clip(r, -1., 1.)
            if info['ale.lives'] < start_lives:
                cumulated_reward -= 1
                break

        if len(states) > 1:
            s_ = np.maximum.reduce([states[-2], states[-1]])

        return preprocess(s_), cumulated_reward, done, info

    def run(self, agent):
        self.env.reset()
        R = 0

        for i in range(random.randint(1, NO_OP_MAX)):
            s = self.step(1)[0]

        s = np.tile(s[0, ...], (HISTORY_LENGTH, 1, 1))

        while True:
            a = agent.choose_action(s)

            if a == 0:
                a = 1
            elif a == 1:
                a = 2
            else:
                a = 3

            s_, r, done, _ = self.step(a)

            if done:
                done = 1
            else:
                done = 0

            agent.observe((s, a, r, s_, done))

            if self.episodes > LEARN_START:

                if self.episodes % REPLAY_FREQUENCY == 0:
                    agent.replay()

                if self.episodes % UPDATE_FREQUENCY == 0:
                    agent.brain.update()

            R += r

            if done == 1:
                self.episodes += 1
                self.trace_r.append(R)
                # print env.episodes, 'total reward is', R
                break

            new_s = s
            new_s[..., :-1] = s[..., 1:]
            new_s[..., -1] = s_[..., 0]
            s = new_s


PROBLEM = 'BreakoutNoFrameskip-v4'
env = Environment(PROBLEM)
n_action = env.env.action_space.n
random_agent = RandomAgent(n_action)
agent = Agent((HISTORY_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH), n_action)

try:
    while random_agent.steps < LEARN_START:
        env.run(random_agent)

    agent.memory = random_agent.memory
    env.episodes = 0
    env.trace_r = []
    random_agent = None

    while True:
        env.run(agent)

        if env.episodes % 1000 == 0:
            agent.brain.model.save(PROBLEM + '.h5')
            print env.episodes, 'last 1000 episodes ave rewards is', np.sum(env.trace_r[-1 - 1000: -1]) / 1000.0

finally:
    agent.brain.model.save(PROBLEM + '.h5')








