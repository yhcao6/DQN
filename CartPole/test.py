import gym
from agent import Agent
import tensorflow as tf


class Environment(object):
    def __init__(self):
        self.env = gym.make('CartPole-v0')
        self.epsodes = 0

    def run(self, agent):
        s = self.env.reset()
        R = 0

        while True:
            # self.env.render()

            a = agent.act(s)

            s_, r, done, info = self.env.step(a)

            agent.observe(s, a, r, s_, done)

            # if self.epsodes > 40 and (self.epsodes + 1) % 40 == 0:
            #     agent.replay()
            # agent.replay()

            s = s_
            R += r

            if done:
                break

        print 'epsodes', self.epsodes, 'Total reward is', R
        self.epsodes += 1


with tf.Session() as sess:
    env = Environment()
    agent = Agent(sess)
    saver = tf.train.import_meta_graph('./save/cart_pole-449.meta')
    saver.restore(sess, tf.train.latest_checkpoint('./save/'))
    while True:
        env.run(agent)

