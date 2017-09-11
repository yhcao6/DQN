import random
import tensorflow as tf

from agent import Agent
from environment import Environment

with tf.Session() as sess:
    env = Environment()

    agent = Agent(env, sess)

    agent.train()

