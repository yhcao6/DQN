import tensorflow as tf
import numpy as np
import cv2
from utils import conv2d, linear, clipped_error
from environment import Environment
import time
from history import History
import random
from tqdm import tqdm
from replay_memory import ReplayMemory


class Agent(object):
    def __init__(self, environment, sess):
        self.env = environment
        self.sess = sess
        self.history_length = 4
        self.screen_height = 84
        self.screen_width = 84

        self.cnn_format = 'NCHW'

        self.scale = 10000
        self.learning_rate = 0.00025
        self.learning_rate_minimum = 0.00025
        self.learning_rate_decay = 0.96
        self.learning_rate_decay_step = 5 * self.scale
        self.target_q_update_step = 1 * self.scale
        self.max_step = self.scale * 5000
        self.learn_start = 5 * self.scale
        self.discount = 0.99
        self.memory_size = 100 * self.scale

        self.ep_end = 0.1
        self.ep_start = 1.
        self.ep_end_t = self.memory_size

        self.min_reward = -1.
        self.max_reward = 1.

        self.test_step = 5 * scale
        self.save_step = test_step * 10

        self.history = History()
        self.memory = ReplayMemory()

        with tf.variable_scope('step'):
            self.step_op = tf.Variable(0, trainable=False, name='step')
            self.step_input = tf.placeholder('int32', None, name='step_input')
            self.step_assign_op = self.step_op.assign(self.step_input)

        self.build_dqn()

    def train(self):
        start_step = self.step_op.eval()
        start_time = time.time()

        num_game, self.updaate_count, ep_reward = 0, 0, 0
        total_reward, self.total_loss, self.total_q = 0., 0., 0.
        max_avg_ep_reward = 0
        ep_rewards, actions = [], []

        screen, reward, action, terminal = self.env.new_randm_game()

        for _ in range(self.history_length):
            self.history.add(screen)

        for self.step in tqdm(range(start_step, self.max_step), ncols=70, initial=start_step):
            if self.step == self.learn_start:
                num_game, self.updaate_count, ep_reward = 0, 0, 0.
                total_reward, self.total_loss, self.total_q = 0., 0., 0.
                ep_rewards, actions = [], []

            action = self.predict(self.history.get())

            screen, reward, terminal = self.env.act(action, is_training=True)

            self.observe(screen, reward, action, terminal)

            if terminal:
                screen, reward, action, terminal = self.env.new_randm_game()

                num_game += 1
                ep_rewards.append(ep_reward)
                ep_reward = 0.
            else:
                ep_reward += reward

            actions.append(action)
            total_reward += reward

            if self.step >= self.learn_start:
                if self.step % self.test_step == self.test_step - 1:
                    avg_reward = total_reward / self.test_step
                    avg_loss = self.total_loss / self.updaate_count
                    avg_q = self.total_q / self.updaate_count

                    try:
                        max_ep_reward = np.max(ep_rewards)
                        min_ep_reward = np.min(ep_rewards)
                        avg_ep_reward = np.mean(ep_rewards)
                    except:
                        max_ep_reward, min_ep_reward, avg_ep_reward = 0, 0, 0

                    if max_avg_ep_reward * 0.9 <= avg_ep_reward:
                        self.step_assign_op.eval({self.step_input: self.step + 1})
                        self.save_model(self.step + 1)

                        max_avg_ep_reward = max(max_avg_ep_reward, avg_ep_reward)

                    num_game = 0
                    total_reward = 0.
                    self.total_loss = 0.
                    self.total_q = 0.
                    self.updaate_count = 0
                    ep_reward = 0.
                    ep_rewards = []
                    actions = []

    def predict(self, s_t, test_ep=None):
        ep = test_ep or (self.ep_end + max(0., (self.ep_start - self.ep_end) * (self.ep_end_t - max(0., self.step - self.learn_start)) / self.ep_end_t))

        if random.random() < ep:
            action = random.randrange(self.env.action_size)
        else:
            action = self.q_action.eval({self.s_t: [s_t]})[0]

        return action

    def observe(self, screen, reward, action, terminal):
        reward = max(self.min_reward, min(self.max_reward, reward))

        self.history.add(screen)
        self.memory.add(screen, reward, action, terminal)

        if self.step > self.learn_start:
            if self.step % self.train_frequency == 0:
                self.q_learning_mini_batch()

            if self.step % self.target_q_update_step == self.target_q_update_step - 1:
                self.update_target_q_network()

    def q_learning_mini_batch(self):
        if self.memory.count < self.history_length:
            return
        else:
            s_t, action, reward, s_t_plus_1, terminal = self.memory.sample()

        t = time.time()

        q_t_plus_1 = self.target_q.eval({self.target_s_t: s_t_plus_1})

        terminal = np.array(terminal) + 0.
        max_q_t_plus_1 = np.max(q_t_plus_1, axis=1)
        target_q_t = (1. - terminal) * self.discount * max_q_t_plus_1 + reward

        _, q_t, loss = self.sess.run([self.optim, self.q, self.loss], {self.target_q_t: target_q_t, self.action: action, self.s_t: s_t, self.learning_rate_step: self.step})

        self.total_loss += loss
        self.total_q += q_t.mean()
        self.updaate_count += 1









    def build_dqn(self):
        self.w = {}
        self.t_w = {}

        initializer = tf.truncated_normal_initializer(0, 0.02)
        activation_fn = tf.nn.relu

        with tf.variable_scope('prediction'):
            self.s_t = tf.placeholder(tf.float32, [None, self.history_length, self.screen_height, self.screen_width], name='s_t')
            self.l1, self.w['l1'], self.w['l1_b'] = conv2d(self.s_t, 32, [8, 8], [4, 4], initializer, activation_fn, self.cnn_format, name='l1')
            self.l2, self.w['l2'], self.w['l2_b'] = conv2d(self.l1, 64, [4, 4], [2, 2], initializer, activation_fn, self.cnn_format, name='l2')
            self.l3, self.w['l3'], self.w['l3_b'] = conv2d(self.l2, 64, [3, 3], [1, 1], initializer, activation_fn, self.cnn_format, name='l3')

            shape = self.l3.get_shape().as_list()
            self.l3_flat = tf.reshape(self.l3, [-1, reduce(lambda x, y: x * y, shape[1: ])])

            self.l4, self.w['l4_w'], self.w['l4_b'] = linear(self.l3_flat, 512, activation_fn=activation_fn, name='l4')
            self.q, self.w['q'], self.w['q_b'] = linear(self.l4, self.env.action_size, name='q')

            self.q_action = tf.argmax(self.q, dimension=1)


        with tf.variable_scope('target'):
            self.target_s_t = tf.placeholder(tf.float32, [None, self.history_length, self.screen_height, self.screen_width], name='target_s_t')

            self.target_l1, self.t_w['l1'], self.t_w['l1_b'] = conv2d(self.target_s_t, 32, [8, 8], [4, 4], initializer, activation_fn, self.cnn_format, name='target_l1')
            self.target_l2, self.t_w['l2'], self.t_w['l2_b'] = conv2d(self.target_s_t, 64, [4, 4], [2, 2], initializer, activation_fn, self.cnn_format, name='target_l2')
            self.target_l3, self.t_w['l3'], self.t_w['l3_b'] = conv2d(self.target_s_t, 64, [3, 3], [1, 1], initializer, activation_fn, self.cnn_format, name='target_l3')

            shape = self.target_l3.get_shape().as_list()
            self.target_l3_flat = tf.reshape(self.target_l3, [-1, reduce(lambda x, y: x * y, shape[1: ])])

            self.target_l4, self.t_w['l4_w'], self.t_w['l4_b'] = linear(self.target_l3_flat, 512, activation_fn=activation_fn, name='target_l4')
            self.target_q, self.t_w['q'], self.t_w['q_b'] = linear(self.target_l4, self.env.action_size, name='target_q')

        with tf.variable_scope('pred_to_target'):
            self.t_w_input = {}
            self.t_w_assign_op = {}
            for name in self.w.keys():
                self.t_w_input[name] = tf.placeholder('float32', self.t_w[name].get_shape().as_list(), name=name)
                self.t_w_assign_op[name] = self.t_w[name].assign(self.t_w_input[name])

        with tf.variable_scope('optimizer'):
            self.target_q_t = tf.placeholder('float32', [None], name='target_q_t')
            self.action = tf.placeholder('int64', [None], name='action')

            action_one_hot = tf.one_hot(self.action, self.env.action_size, 1.0, 0.0, name='action_one_hot')
            q_acted = tf.reduce_sum(self.q * action_one_hot, reduction_indices=1, name='q_acted')

            self.delta = self.target_q_t - q_acted

            self.global_step = tf.Variable(0, trainable=False)

            self.loss = tf.reduce_mean(clipped_error(self.delta), name='loss')

            self.learning_rate_step = tf.placeholder('int64', None, name='learning_rate_step')
            # lr = lr0 * decay_rate ^ (learning_step / decay_step)
            self.learning_rate_op = tf.maximum(self.learning_rate_minimum, tf.train.exponential_decay(self.learning_rate, self.learning_rate_step, self.learning_rate_decay_step, self.learning_rate_decay, staircase=True))

            self.optim = tf.train.RMSPropOptimizer(self.learning_rate_op, momentum=0.95, epsilon=0.01).minimize(self.loss)

        # tf.initialize_all_variables().run()
        tf.initialize_all_variables().run()

        self.update_target_q_network()

    def update_target_q_network(self):
        for name in self.w.keys():
            self.t_w_assign_op[name].eval({self.t_w_input[name]: self.t_w[name].eval()})


if __name__ == '__main__':
    env = Environment()
    agent = Agent(env)

