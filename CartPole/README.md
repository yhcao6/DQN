@(fyp)[cart-pole]
##### README.MD

<img src="./image/cartpole-screenshot.png" style="width: 100px;"/>

The is a cart (the black box) and a pole, the cart can go left or right along the axis, the goal is to balance the pole as long as possible.

MDP define:

state define:
[cart position, velocity, pole angle, angular velocity]

action:
[0, 1] (0: left; 1: right)

reward:
If the pole is up, above certain angle, the agent receive a reward of +1. Otherwise, the episode ends.

goal:
maximize the total reward. (balance the pole as long as possible)

scenario:
There is an agent and an environment. Agent interact with environment through actions and observations. The environment take agent's actions and return next state and necessary information like rewards. The agent store the encounted experience in its memory so that the next time the he can better choose the action which will bring him higher reward.
