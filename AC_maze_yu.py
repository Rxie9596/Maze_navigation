import numpy as np
import tensorflow as tf
from maze_env_yu import Maze
import time
import scipy.io as sio



env_name = '2'
exp = '4'

np.random.seed(int(exp))
tf.set_random_seed(int(exp))  # reproducible

# Parameters
OUTPUT_GRAPH = True
MAX_EPISODE = 2000
DISPLAY_REWARD_THRESHOLD = 1000  # renders environment if total episode reward is greater then this threshold
MAX_EP_STEPS = 100   # maximum time step in one episode
RENDER = False  # rendering wastes time
RENDER_EP = 5000

GAMMA = 0.99     # reward discount in TD error
LR_A = 0.001   # learning rate for actor
LR_C = 0.01     # learning rate for critic
BETA = 0.001

env = Maze()

N_F = env.n_features
N_A = env.n_actions

AL1SIZE = 30
AL2SIZE = 30
CL1SIZE = 30
CL2SIZE = 30

DISCOUNT = True
PUNISHMENT = True
SAVE = True
save_directory = '/Users/yuxie/Lab/Maze_navigation/Data/env' + env_name + '_' + exp + '_inf_baseline' + '.mat'
step_name = 'steps' + env_name + '_' + exp + '_inf_baseline'
reward_name = 'reward' + env_name + '_' + exp + '_inf_baseline'
smooth_r_name = reward_name + "_smooth"

class Actor(object):
    def __init__(self, sess, n_features, n_actions, lr=0.001):
        self.sess = sess

        self.s = tf.placeholder(tf.float32, [1, n_features], "state")
        self.a = tf.placeholder(tf.int32, None, "act")
        self.td_error = tf.placeholder(tf.float32, None, "td_error")  # TD_error

        with tf.variable_scope('Actor'):
            l1 = tf.layers.dense(
                inputs=self.s,
                units=AL1SIZE,    # number of hidden units
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(0., .1),    # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='l1'
            )

            l2 = tf.layers.dense(
                inputs=l1,
                units=AL2SIZE,    # number of hidden units
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(0., .1),    # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='l2'
            )

            self.acts_prob = tf.layers.dense(
                inputs=l2,
                units=n_actions,    # output units
                activation=tf.nn.softmax,   # get action probabilities
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='acts_prob'
            )

        with tf.variable_scope('exp_v'):
            log_prob = tf.log(self.acts_prob[0, self.a])
            self.exp_v = tf.reduce_mean(log_prob * self.td_error) # advantage (TD_error) guided loss
            # self.exp_v = tf.reduce_mean(log_prob * self.td_error) + BETA * tf.reduce_sum(tf.math.multiply(self.acts_prob, tf.log(tf.math.truediv(1.0, self.acts_prob + 0.001))))

        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(lr).minimize(-self.exp_v)  # minimize(-exp_v) = maximize(exp_v)

    def learn(self, s, a, td):
        s = s[np.newaxis, :]
        feed_dict = {self.s: s, self.a: a, self.td_error: td}
        _, exp_v = self.sess.run([self.train_op, self.exp_v], feed_dict)
        return exp_v

    def choose_action(self, s):
        s = s[np.newaxis, :]
        probs = self.sess.run(self.acts_prob, {self.s: s})   # get probabilities for all actions
        return np.random.choice(np.arange(probs.shape[1]), p=probs.ravel())   # return a int


class Critic(object):
    def __init__(self, sess, n_features, lr=0.01):
        self.sess = sess

        self.s = tf.placeholder(tf.float32, [1, n_features], "state")
        self.v_ = tf.placeholder(tf.float32, [1, 1], "v_next")
        self.r = tf.placeholder(tf.float32, None, 'r')

        with tf.variable_scope('Critic'):
            l1 = tf.layers.dense(
                inputs=self.s,
                units=CL1SIZE,  # number of hidden units
                activation=tf.nn.relu,  # None
                # have to be linear to make sure the convergence of actor.
                # But linear approximator seems hardly learns the correct Q.
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='l1'
            )

            l2 = tf.layers.dense(
                inputs=l1,
                units=CL2SIZE,    # number of hidden units
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(0., .1),    # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='l2'
            )

            self.v = tf.layers.dense(
                inputs=l2,
                units=1,  # output units
                activation=None,
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='V'
            )

        with tf.variable_scope('squared_TD_error'):
            self.td_error = self.r + GAMMA * self.v_ - self.v
            self.loss = tf.square(self.td_error)    # TD_error = (r+gamma*V_next) - V_eval
        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(lr).minimize(self.loss)

    def learn(self, s, r, s_):
        s, s_ = s[np.newaxis, :], s_[np.newaxis, :]

        v_ = self.sess.run(self.v, {self.s: s_})
        td_error, _ = self.sess.run([self.td_error, self.train_op],
                                          {self.s: s, self.v_: v_, self.r: r})
        return td_error


sess = tf.Session()

actor = Actor(sess, n_features=N_F, n_actions=N_A, lr=LR_A)
critic = Critic(sess, n_features=N_F, lr=LR_C)     # we need a good teacher, so the teacher should learn faster than the actor

sess.run(tf.global_variables_initializer())

if OUTPUT_GRAPH:
    tf.summary.FileWriter("logs/", sess.graph)


steps = 0
track_step = []
track_r = []
track_smooth_r = []
for i_episode in range(MAX_EPISODE):

    pos = env.random_pos()
    s = env.reset(agent_cor=pos)

    if i_episode > RENDER_EP:
        RENDER = True

    if RENDER:
        env.render()
        time.sleep(0.1)

    t = 0
    track_ep_r = []
    while True:
        a = actor.choose_action(s)

        s_, r, done = env.step(a)
        if RENDER: env.render()

        if DISCOUNT:
            if r > 0:
                discount = 100.0 / float(t+1)
                r = r * discount

        if PUNISHMENT:
            if t == MAX_EP_STEPS - 1:
                r -= 10

        track_ep_r.append(r)

        td_error = critic.learn(s, r, s_)  # gradient = grad[r + gamma * V(s_) - V(s)]
        actor.learn(s, a, td_error)     # true_gradient = grad[logPi(s,a) * td_error]

        s = s_
        t += 1
        steps += 1


        if done or t >= MAX_EP_STEPS:


            ep_rs_sum = sum(track_ep_r)

            if 'running_reward' not in globals():
                running_reward = ep_rs_sum
            else:
                running_reward = running_reward * 0.95 + ep_rs_sum * 0.05

            if running_reward > DISPLAY_REWARD_THRESHOLD: RENDER = True  # rendering
            print("episode:", i_episode, "  reward:", running_reward)

            track_step.append(steps)
            track_r.append(ep_rs_sum)
            track_smooth_r.append(running_reward)

            break


if SAVE:
    env1 = {}
    env1[step_name] = track_step
    env1[reward_name] = track_r
    env1[smooth_r_name] = track_smooth_r

    sio.savemat(save_directory, env1)