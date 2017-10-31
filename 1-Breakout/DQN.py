import gym, random
import numpy as np
import tensorflow as tf
from collections import deque
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers.convolutional import Conv2D
from keras.optimizers import RMSprop
from keras import backend as K
from skimage.color import rgb2gray
from skimage.transform import resize


# hyperparameters
# game name
GAME = 'BreakoutNoFrameskip-v4'
env = gym.make(GAME)
ACTION_COUNT = env.action_space.n  # action_count

# state size (4, 84, 84)
IMAGE_WIDTH = 84
IMAGE_HEIGHT = 84
HISTORY_LENGTH = 4

BATCH_SIZE = 32
MEMORY_SIZE = 1000000

# random start steps
NO_OP_MAX = 30

# repeat same action
ACTION_REPEAT = 4

# exploration
INITIAL_EXPLORATION = 1
FINAL_EXPLORATION = 0.1
FINAL_EXPLORATION_FRAME = 1000000

# learning_rate
LEARING_RATE = 0.00025

# DISCOUNT FACTOR
DISCOUNT = 0.99

# REPLAY_START
REPLAY_START = 50000

REPLAY_FREQUENCY = 4
UPDATE_FREQUENCY = 10000

HUBER_LOSS_DELTA = 1.0


# preprocess img
def preprocess(img):
    # 1. exact Y channel
    gray = rgb2gray(img)  # 210 x 160
    # 2. rescale to 84 x 84
    gray_rescale = resize(gray, (84, 84), mode='constant')
    return gray_rescale


class Agent(object):
    def __init__(self):
        # neural network
        self.model = self._build_model()
        self.model_ = self._build_model()

        # experience memory
        self.memory = Memory()

        self.steps = 0
        self.epsilon = INITIAL_EXPLORATION
        self.epsilon_decay_step = (INITIAL_EXPLORATION - FINAL_EXPLORATION) / FINAL_EXPLORATION_FRAME

        self.optimizer = self.optimizer()

        # os.environ['CUDA_VISIBLE_DEVICES'] = '1'
        self.sess = tf.Session()
        K.set_session(self.sess)

        self.avg_loss = 0.
        self.summary_placeholders, self.update_ops, self.summary_op = self.setup_summary()
        self.summary_writer = tf.summary.FileWriter('summary', self.sess.graph)

        self.sess.run(tf.global_variables_initializer())

    def _build_model(self):
        model = Sequential()

        model.add(Conv2D(32, (8, 8), strides=(4, 4), activation='relu', input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, HISTORY_LENGTH)))
        model.add(Conv2D(64, (4, 4), strides=(2, 2), activation='relu'))
        model.add(Conv2D(64, (3, 3), strides=(1, 1), activation='relu'))
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dense(ACTION_COUNT, activation='linear'))
        return model

    def optimizer(self):
        action = K.placeholder(shape=(None,), dtype='int32')
        target_q = K.placeholder(shape=(None,), dtype='float32')
        action_mask = K.one_hot(action, ACTION_COUNT)

        pred = self.model.output
        q = K.sum(pred * action_mask, axis=1)
        err = K.abs(target_q - q)

        # huge loss
        l2 = K.clip(err, 0.0, 1.0)
        l1 = err - l2
        loss = K.mean(0.5 * K.square(l2) + l1)

        optimizer = RMSprop(lr=LEARING_RATE, epsilon=0.01)
        updates = optimizer.get_updates(self.model.trainable_weights, [], loss)
        train = K.function([self.model.input, action, target_q], [loss], updates=updates)

        return train


    # e-greedy
    def choose_action(self, history):
        if np.random.rand() <= self.epsilon:
            return random.randint(0, ACTION_COUNT - 1)
        else:
            return np.argmax(self.model.predict(history)[0])

    def observe(self, history, action, reward, history_, dead):
        self.memory.add(history, action, reward, history_, dead)
        self.steps += 1

        if self.steps > REPLAY_START:
            # epsilon decrease as time
            self.epsilon -= self.epsilon_decay_step
            if self.steps % REPLAY_FREQUENCY == 0:
                self.replay()

            if self.steps % UPDATE_FREQUENCY == 0:
                self.update()

    def replay(self):
        if len(self.memory.memory) < REPLAY_START:
            return

        batch = self.memory.sample()

        # batch[i] (history, action, reward, history_, dead)
        historys = np.zeros((BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, HISTORY_LENGTH))
        historys_ = np.zeros((BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, HISTORY_LENGTH))
        actions = []
        rewards = []
        deads = []
        target_q = np.zeros((BATCH_SIZE,))

        for i in range(BATCH_SIZE):
            historys[i] = batch[i][0]
            historys_[i] = batch[i][3]
            actions.append(batch[i][1])
            rewards.append(batch[i][2])
            deads.append(batch[i][4])

        pred = self.model_.predict(historys_)

        # Q(s, a) = r + gamma * np.max(Q(s', a))
        for i in range(BATCH_SIZE):
            if deads[i]:
                target_q[i] = rewards[i]
            else:
                target_q[i] = rewards[i] + DISCOUNT * np.amax(pred[i])

        loss = self.optimizer([historys, actions, target_q])
        self.avg_loss += loss[0]

    def update(self):
        self.model_.set_weights(self.model.get_weights())

    def setup_summary(self):
        total_reward = tf.Variable(0.)
        avg_loss = tf.Variable(0.)

        tf.summary.scalar('total reward', total_reward)
        tf.summary.scalar('average loss', avg_loss)

        summary_vars = [total_reward, avg_loss]
        summary_placeholders = [tf.placeholder(tf.float32) for _ in range(len(summary_vars))]
        update_ops = [summary_vars[i].assign(summary_placeholders[i]) for i in range(len(summary_vars))]
        summary_op = tf.summary.merge_all()
        return summary_placeholders, update_ops, summary_op


class Memory(object):
    def __init__(self):
        self.memory = deque(maxlen=MEMORY_SIZE)

    def add(self, history, action, reward, history_, dead):
        self.memory.append((history, action, reward, history_, dead))

    def sample(self):
        return random.sample(self.memory, BATCH_SIZE)


class Environment(object):
    def __init__(self):
        self.env = gym.make(GAME).unwrapped
        self.episode = 0

    def step(self, action):
        cumulated_reward = 0
        start_lives = self.env.ale.lives()
        dead = False
        tmp = []

        for _ in range(ACTION_REPEAT):
            s, r, _, info = self.env.step(action)
            cumulated_reward += r

            if start_lives > info['ale.lives']:
                cumulated_reward -= 1
                dead = True
            else:
                tmp.append(s)

            if dead:
                break

        # take maximum in last two frames
        if len(tmp) > 1:
            s_ = np.maximum.reduce([tmp[-1], tmp[-2]])
        else:
            s_ = s

        return s_, cumulated_reward, dead



    def run(self, agent):
        s = self.env.reset()

        dead = False
        done = False

        # step 1 (1, 30) time steps
        # here each step only one frame
        for _ in range(random.randint(1, NO_OP_MAX)):
            s = self.env.step(1)[0]

        s = preprocess(s)
        history = np.stack((s, s, s, s), axis=2)  # H x W x C = 84 x 84 x 4
        history = history[np.newaxis, ...]

        total_reward = 0
        step = 0
        while not done:
            step += 1
            action = agent.choose_action(history)

            if action == 0:
                real_action = 1
            elif action == 1:
                real_action = 2
            else:
                real_action = 3

            s_, r, dead = self.step(action)
            total_reward += r  # true reward, just use for visulization
            s_ = preprocess(s_)
            s_ = s_[np.newaxis, ..., np.newaxis]
            # clip reward, reduce difficulty
            r = np.clip(r, -1., 1.)

            history_ = np.append(history[..., 1:], s_, axis=3)

            # add to memory
            agent.observe(history, action, r, history_, dead)

            if dead:
                # if lives is 0, game over
                if self.env.ale.lives() == 0:
                    self.episode += 1
                    print 'episode', self.episode, 'total reward is', total_reward
                    done = True

                    if agent.steps > REPLAY_START:
                        states = [total_reward, agent.avg_loss / float(step)]
                        for i in range(len(states)):
                            agent.sess.run(agent.update_ops[i], feed_dict={agent.summary_placeholders[i]: float(states[i])})
                        summary_str = agent.sess.run(agent.summary_op)
                        agent.summary_writer.add_summary(summary_str, self.episode)

                    agent.avg_loss = 0.
                else:
                    dead = False
            else:
                history = history_


agent = Agent()
env = Environment()

try:
    while True:
        env.run(agent)
        if env.episode % 1000 == 0:
            agent.model.save_weights('dqn.h5')
finally:
    agent.model.save_weights('latest_dqn.h5')
