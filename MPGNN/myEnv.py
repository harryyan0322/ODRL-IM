import gym
import numpy as np
from gym.vector.utils import spaces


class MyEnv(gym.Env):
    def __init__(self, graph, budget, num_node, input_dim):
        super(MyEnv, self).__init__()
        # 图
        self.graph = graph
        # 代价
        self.budget = budget
        self.num_node = num_node
        self.input_dim = input_dim
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_node, self.input_dim),
                                            dtype=np.float32)
        # 动作空间，动作是节点的索引
        self.action_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_node, 1),
                                            dtype=np.float32)

    def reset(self):
        self.state = self.observation_space.sample()
        return self.state

    def step(self, action):
        # 状态转移函数

        self.state = self.observation_space.sample()

        # 计算奖励
        reward = self._calculate_reward()

        # 判断是否达到终止状态
        done = self._is_terminal_state()

        return self.state, reward, done

    def _calculate_reward(self):
        pass

    def _is_terminal_state(self):
        pass

    def render(self):
        pass

    def seed(self, seed=None):
        pass
