import tensorflow.compat.v1 as tf
import numpy as np
import matplotlib.pyplot as plt

tf.disable_v2_behavior()  # TensorFlow configurations

# hyper parameters for RL model training

LR_A = 0.001  # learning rate for actor
LR_C = 0.001  # learning rate for critic
GAMMA = 0.9  # reward discount
TAU = 0.01  # soft replacement
MEMORY_CAPACITY = 30000
BATCH_SIZE = 128


class DDPG(object):
    def __init__(self, a_dim, s_dim, a_bound, ):
        self.memory = np.zeros((MEMORY_CAPACITY, s_dim * 2 + a_dim + 1), dtype=np.float32)
        self.pointer = 0
        self.memory_full = False
        self.sess = tf.Session()
        self.a_replace_counter, self.c_replace_counter = 0, 0

        self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound[1]
        self.S = tf.placeholder(tf.float32, [None, s_dim], 's')
        self.S_ = tf.placeholder(tf.float32, [None, s_dim], 's_')
        self.R = tf.placeholder(tf.float32, [None, 1], 'r')

        with tf.variable_scope('Actor'):
            self.a = self._build_a(self.S, scope='eval', trainable=True)
            a_ = self._build_a(self.S_, scope='target', trainable=False)
        with tf.variable_scope('Critic'):
            # assign self.a = a in memory when calculating q for td_error,
            # otherwise the self.a is from Actor when updating Actor
            q = self._build_c(self.S, self.a, scope='eval', trainable=True)
            q_ = self._build_c(self.S_, a_, scope='target', trainable=False)

        # networks parameters
        self.ae_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval')
        self.at_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target')
        self.ce_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval')
        self.ct_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target')

        # target net replacement
        self.soft_replace = [[tf.assign(ta, (1 - TAU) * ta + TAU * ea), tf.assign(tc, (1 - TAU) * tc + TAU * ec)]
                             for ta, ea, tc, ec in zip(self.at_params, self.ae_params, self.ct_params, self.ce_params)]

        q_target = self.R + GAMMA * q_
        # in the feed_dic for the td_error, the self.a should change to actions in memory
        self.td_error = tf.losses.mean_squared_error(labels=q_target, predictions=q)
        self.ctrain = tf.train.AdamOptimizer(LR_C).minimize(self.td_error, var_list=self.ce_params)

        self.a_loss = - tf.reduce_mean(q)  # maximize the q
        self.atrain = tf.train.AdamOptimizer(LR_A).minimize(self.a_loss, var_list=self.ae_params)

        self.sess.run(tf.global_variables_initializer())

        self.actor_loss_list = []
        self.critic_loss_list = []

    def choose_action(self, s):
        return self.sess.run(self.a, {self.S: s[None, :]})[0]

    def learn(self):
        # soft target replacement
        self.sess.run(self.soft_replace)

        indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)
        bt = self.memory[indices, :]
        bs = bt[:, :self.s_dim]
        ba = bt[:, self.s_dim: self.s_dim + self.a_dim]
        br = bt[:, -self.s_dim - 1: -self.s_dim]
        bs_ = bt[:, -self.s_dim:]

        # self.sess.run(self.atrain, {self.S: bs})
        # self.sess.run(self.ctrain, {self.S: bs, self.a: ba, self.R: br, self.S_: bs_})

        _, actor_loss_val = self.sess.run([self.atrain, self.a_loss], {self.S: bs})
        _, critic_loss_val = self.sess.run([self.ctrain, self.td_error],
                                           {self.S: bs, self.a: ba, self.R: br, self.S_: bs_})

        self.actor_loss_list.append(actor_loss_val)
        self.critic_loss_list.append(critic_loss_val)

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        index = self.pointer % MEMORY_CAPACITY  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.pointer += 1
        if self.pointer > MEMORY_CAPACITY:  # indicator for learning
            self.memory_full = True

    def _build_a(self, s, scope, trainable):
        with tf.variable_scope(scope):
            net1 = tf.layers.dense(s, 512, activation=tf.nn.relu, name='l1', trainable=trainable)
            net2 = tf.layers.dense(net1, 64, activation=tf.nn.relu, name='l2', trainable=trainable)
            # net3 = tf.layers.dense(net2, 32, activation=tf.nn.relu, name='l3', trainable=trainable)
            # a = tf.layers.dense(net1, self.a_dim, name='a', trainable=trainable)
            a = tf.layers.dense(net2, self.a_dim, activation=tf.nn.tanh, name='a', trainable=trainable)
            return tf.multiply(a, self.a_bound, name='scaled_a')

    def _build_c(self, s, a, scope, trainable):
        with tf.variable_scope(scope):
            n_l1 = 512
            w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1], trainable=trainable)
            w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], trainable=trainable)
            b1 = tf.get_variable('b1', [1, n_l1], trainable=trainable)
            net1 = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
            net2 = tf.layers.dense(net1, 64, activation=tf.nn.relu, name='l2_', trainable=trainable)
            # net3 = tf.layers.dense(net2, 32, activation=tf.nn.relu, name='l3_', trainable=trainable)
            return tf.layers.dense(net2, 1, trainable=trainable)  # Q(s,a)

    def save(self):
        saver = tf.train.Saver()
        saver.save(self.sess, './params', write_meta_graph=False)

    def restore(self):
        saver = tf.train.Saver()
        saver.restore(self.sess, './params')

    def plot_loss(self):
        # data visualization with filtering
        a_loss = [sum(self.actor_loss_list[i:i + 20]) / 20 for i in range(0, len(self.actor_loss_list), 20)]
        # c_loss = [sum(self.critic_loss_list[i:i + 25]) / 25 for i in range(0, len(self.critic_loss_list), 25)]
        # plt.plot(self.actor_loss_list, label='Actor Loss')
        # plt.plot(self.critic_loss_list, label='Critic Loss')
        plt.plot(a_loss, label='Actor Loss')
        # plt.plot(c_loss, label='Critic Loss')
        plt.xlabel('num_iterations')
        plt.ylabel('Loss')
        plt.title('Training Loss Over Time')
        plt.legend()
        plt.savefig("train_loss.png")
        plt.close()
