import tensorflow as tf
from utile import linear
import numpy as np
from memory import Memory


class Agent(object):
    def __init__(self, sess):
        self.sess = sess
        # some config
        self.state_size = 4
        self.n_action = 2
        self.epsilon = 0.1
        self.max_epsilon = 1
        self.min_epsilon = 0.01
        self.epsilon_decay_rate = 0.001
        self.discount = 0.99
        self.steps = 0
        self.batch_size = 64
        self.lr = 0.00025

        self.memory = Memory()

        # build network
        self._build_network()

        self.loss_summary = tf.summary.scalar('loss', self.loss)

        self.writer = tf.summary.FileWriter('logs/', sess.graph)

        sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()

    def _build_network(self):
        self.w = {}

        initializer = tf.truncated_normal_initializer(0, 0.02)
        activation_fn = tf.nn.relu

        with tf.variable_scope('prediction'):
            # input
            self.s = tf.placeholder(tf.float32, [None, self.state_size], name='s')

            # l1
            self.l1, self.w['l1_w'], self.w['l1_b'] = linear(self.s, 64, initializer, activation_fn, name='l1')

            # q
            self.q, self.w['q_w'], self.w['q_b'] = linear(self.l1, self.n_action, initializer, activation_fn=None, name='q')

        with tf.variable_scope('loss'):
            self.target_q = tf.placeholder(tf.float32, [None, 1], name='target_q')
            self.action = tf.placeholder('int64', [None, 1], name='action')

            action_one_hot = tf.one_hot(self.action, 2)[:, 0, :]
            q_acted = tf.reduce_sum(self.q * action_one_hot, reduction_indices=1, name='q_acted')
            q_acted = tf.reshape(q_acted, [-1, 1])

            self.loss = tf.losses.mean_squared_error(self.target_q, q_acted)

            self.optimizer = tf.train.RMSPropOptimizer(self.lr)

            self.train_op = self.optimizer.minimize(self.loss)

    def act(self, s):
        s = np.array(s)
        s = s[np.newaxis, ...]
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_action)
        else:
            return np.argmax(self.q.eval({self.s: s}), axis=1)[0]

    def observe(self, s, a, r, s_, terminal):
        self.memory.add(s, a, r, s_, terminal)

        self.steps += 1
        self.epsilon = self.min_epsilon + (self.max_epsilon - self.min_epsilon) * np.exp(-self.epsilon_decay_rate * self.steps)

    def replay(self):
        s, a, r, s_, terminal = self.memory.sample(self.batch_size)

        q_ = self.q.eval({self.s: s_})
        max_q_ = np.max(q_, axis=1).reshape([-1, 1])
        target_q = (1 - terminal) * self.discount * max_q_

        self.write_loss, _ = self.sess.run([self.loss_summary, self.train_op], {self.s: s, self.target_q: target_q, self.action: a})

    def save(self, epsodes):
        self.saver.save(self.sess, 'save/cart_pole', global_step=epsodes)












