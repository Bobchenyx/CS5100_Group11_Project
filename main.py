from env import ArmEnv
from rl import DDPG
import numpy as np
import matplotlib.pyplot as plt

MAX_EPISODES = 900
MAX_EP_STEPS = 300
ON_TRAIN = False  # modify according to evaluation needs

# define env
env = ArmEnv()

# define RL method (continuous)
rl = DDPG(env.action_dim, env.state_dim, env.action_bound)

# list to store plotting data
steps = []
cumulate_r = []


# train loop
def train():
    for i in range(MAX_EPISODES):
        s = env.reset()
        ep_r = 0.
        for j in range(MAX_EP_STEPS):
            # visualization for sanity check
            # env.render()
            # env.viewer.set_vsync(True)
            a = rl.choose_action(s)
            s_, r, done = env.step(a)
            rl.store_transition(s, a, r, s_)

            ep_r += r

            # start to learn once has fulfilled the memory
            if rl.memory_full:
                rl.learn()

            s = s_

            if done or j == MAX_EP_STEPS - 1:
                print('Ep: %i | %s | ep_r: %.1f | step: %i' % (i, '---' if not done else 'done', ep_r, j))
                steps.append(j)
                cumulate_r.append(ep_r / j)
                break
    rl.plot_loss()
    rl.save()


# run at real time
def eval():
    rl.restore()
    env.render()
    env.viewer.set_vsync(True)
    s = env.reset()
    while True:
        env.render()
        a = rl.choose_action(s)
        s, r, done = env.step(a)


def plot_step1():
    # data visualization with filtering
    x = np.arange(0, MAX_EPISODES/5)
    steps1 = [sum(steps[i:i+5]) / 5 for i in range(0, len(steps), 5)]
    y = np.array(steps1)

    plt.plot(x, y)  # Plot the chart
    plt.xlabel("episodes")  # add X-axis label
    plt.ylabel("steps")  # add Y-axis label
    plt.title("Steps to Done")  # add title
    # plt.show()  # display
    plt.savefig("step1.png")
    plt.close()


def plot_reward1():
    # data visualization with filtering
    x = np.arange(0, MAX_EPISODES/5)
    cumulate_r1 = [sum(cumulate_r[i:i+5]) / 5 for i in range(0, len(cumulate_r), 5)]
    y = np.array(cumulate_r1)

    plt.plot(x, y)  # Plot the chart
    plt.xlabel("episodes")  # add X-axis label
    plt.ylabel("reward/step")  # add Y-axis label
    plt.title("Rewards")  # add title
    # plt.show()  # display
    plt.savefig("reward1.png")
    plt.close()


if ON_TRAIN:
    train()
    plot_step1()
    plot_reward1()
else:
    eval()
