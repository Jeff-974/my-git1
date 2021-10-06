import gym
import time
import random
import numpy as np
import matplotlib.pyplot as plt


class MC_RL:
    def __init__(self, env, gamma=0.8, eps=0.05):
        # 行为值函数的初始化        self.n = 0.001 * np.ones(env.observation_space, env.action_space)
        dataShape = [env.nS, env.nA]
        self.qvalue = np.random.rand(dataShape[0], dataShape[1])
        # 次数初始化
        # n[s,a]=1,2,3?? 求经验平均时，q(s,a)=G(s,a)/n(s,a)
        self.n = 0.001 * np.ones(dataShape)
        self.actions = np.array(range(env.nA))
        self.cvalue = np.zeros(dataShape)
        self.env = env
        self.gamma = gamma
        self.epsilon = eps
        self.pi = [self.greedy_policy(self.qvalue, i) for i in range(env.nS)]

    # 定义贪婪策略
    def greedy_policy(self, qfun, state):
        amax = qfun[state, :].argmax()
        return self.actions[amax]

    # 定义e-贪婪策略,蒙特卡罗方法，要评估的策略时e-greedy策略，产生数据的策略。
    def epsilon_greedy_policy(self, qfun, state, epsilon):
        amax = qfun[state, :].argmax()
        # 概率部分
        if np.random.uniform() < 1 - epsilon:
            # 最优动作
            return self.actions[amax]
        else:
            return self.actions[int(random.random() * len(self.actions))]

    def train(self, print_state=False):
        for i_episode in range(10000):
            observation = self.env.reset()
            T = 0
            gain = 0
            a_list = []
            s_list = []
            r_list = []
            weight = 1
            for t in range(200):
                if print_state:
                    self.env.render()
                action = self.epsilon_greedy_policy(self.qvalue, self.env.s,
                                                    self.epsilon)
                a_list.append(action)
                s_list.append(self.env.s)
                observation, reward, done, info = env.step(action)
                r_list.append(reward)
                if done:
                    T = t + 1
                    break
            for t in range(T - 1, 0, -1):
                a_t = a_list[t]
                r_tn = r_list[t]
                s_t = s_list[t]
                gain = self.gamma * gain + r_tn
                self.cvalue[s_t, a_t] += weight
                self.qvalue[s_t, a_t] += weight / self.cvalue[s_t, a_t] * (
                    gain - self.qvalue[s_t, a_t])
                self.pi[s_t] = self.greedy_policy(self.qvalue, s_t)
                if self.pi[s_t] != a_t:
                    break
                weight = weight / (1 - self.epsilon +
                                   self.epsilon / self.env.nA)
        self.env.close()

    def simulate(self, lives=3):
        print("Begin Simulation: ")
        for life in range(lives):
            observation = self.env.reset()
            print("lives left: %d" % (lives - life))
            rt = 0
            for t in range(100):
                self.env.render()
                print(observation)
                action = self.pi[self.env.s]
                observation, reward, done, info = env.step(action)
                rt += reward
                if done:
                    print("Episode finished after {} timesteps".format(t + 1))
                    break
            if rt > 0:
                print("Misson Complete.")
                break
            else:
                print("Try again~")
        self.env.close()


if __name__ == "__main__":
    env = gym.make("FrozenLake-v0")
    ans = env.nS
    solver = MC_RL(env)
    solver.train()
    solver.simulate()
