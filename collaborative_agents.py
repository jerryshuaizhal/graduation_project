import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import copy
from Qlearning import QLearningAgent
from initial_solution import calculate_assembly_sequence_with_components

class DQNNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DQNNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
    
    def forward(self, x):
        return self.network(x)

class DQNAgent:
    def __init__(self, processing_times, initial_sequence, assembly_times, product_components,
                 learning_rate=0.001, discount_factor=0.9, epsilon=0.3,
                 memory_size=10000, batch_size=64, hidden_size=128):
        """
        初始化DQN智能体
        
        参数:
        processing_times: 加工时间矩阵
        initial_sequence: 初始序列
        assembly_times: 装配时间数组
        product_components: 产品组成字典
        learning_rate: 学习率
        discount_factor: 折扣因子
        epsilon: 探索率
        memory_size: 经验回放缓冲区大小
        batch_size: 批量大小
        hidden_size: 隐藏层大小
        """
        self.processing_times = processing_times
        self.initial_sequence = initial_sequence
        self.assembly_times = assembly_times
        self.product_components = product_components
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.memory_size = memory_size
        self.batch_size = batch_size
        
        # 状态空间大小：当前makespan + 最近5个动作的编码
        self.state_size = 6
        # 动作空间：15个启发式规则
        self.action_size = 15
        
        # 创建主网络和目标网络
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.main_network = DQNNetwork(self.state_size, hidden_size, self.action_size).to(self.device)
        self.target_network = DQNNetwork(self.state_size, hidden_size, self.action_size).to(self.device)
        self.target_network.load_state_dict(self.main_network.state_dict())
        
        # 优化器
        self.optimizer = optim.Adam(self.main_network.parameters(), lr=learning_rate)
        
        # 经验回放缓冲区
        self.memory = deque(maxlen=memory_size)
        
        # 记录最佳序列和makespan
        self.best_sequence = copy.deepcopy(initial_sequence)
        self.best_makespan = self._get_makespan(initial_sequence)
        self.initial_makespan = self.best_makespan
        
        # 记录最近的动作历史
        self.action_history = deque(maxlen=5)
        
    def _get_makespan(self, sequence):
        """计算makespan"""
        _, assembly_completion_times = calculate_assembly_sequence_with_components(
            sequence, self.processing_times, self.assembly_times, self.product_components)
        return max(assembly_completion_times.values())
    
    def _get_state(self, sequence):
        """获取当前状态"""
        current_makespan = self._get_makespan(sequence)
        # 归一化makespan
        normalized_makespan = current_makespan / self.initial_makespan
        
        # 将最近的动作历史转换为one-hot编码
        action_history = np.zeros(5)
        for i, action in enumerate(self.action_history):
            action_history[i] = action / 15.0  # 归一化动作值
        
        # 组合状态
        state = np.concatenate([[normalized_makespan], action_history])
        return torch.FloatTensor(state).to(self.device)
    
    def _apply_action(self, sequence, action):
        """应用启发式动作"""
        new_sequence = copy.deepcopy(sequence)
        n = len(sequence)
        
        # 基础操作函数
        def swap(i, j):
            new_sequence[i], new_sequence[j] = new_sequence[j], new_sequence[i]
        
        def move_forward(i, j):
            if i != j:
                item = new_sequence.pop(i)
                new_sequence.insert(j, item)
        
        def move_backward(i, j):
            if i != j:
                item = new_sequence.pop(i)
                new_sequence.insert(j, item)
        
        # 根据动作类型应用相应的操作
        if action == 0:  # LLH1
            i, j = random.sample(range(n), 2)
            swap(i, j)
        elif action == 1:  # LLH2
            i = random.randint(0, n-1)
            j = random.randint(0, i)
            move_forward(i, j)
        elif action == 2:  # LLH3
            i = random.randint(0, n-1)
            j = random.randint(i, n-1)
            move_backward(i, j)
        # ... 其他动作的实现与Q-learning智能体类似
        
        return new_sequence
    
    def _get_reward(self, sequence):
        """计算奖励"""
        makespan = self._get_makespan(sequence)
        
        if makespan < self.best_makespan:
            improvement = self.best_makespan - makespan
            self.best_sequence = copy.deepcopy(sequence)
            self.best_makespan = makespan
            return 10 * (1 + improvement / self.initial_makespan)
        
        if makespan > self.best_makespan:
            deterioration = makespan - self.best_makespan
            return -5 * (1 + deterioration / self.initial_makespan)
        
        return 1
    
    def remember(self, state, action, reward, next_state, done):
        """存储经验到回放缓冲区"""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state, training=True):
        """选择动作"""
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_size)
        
        with torch.no_grad():
            q_values = self.main_network(state)
            return q_values.argmax().item()
    
    def replay(self):
        """从经验回放中学习"""
        if len(self.memory) < self.batch_size:
            return
        
        # 随机采样批量经验
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # 转换为张量
        states = torch.stack(states)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.stack(next_states)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # 计算当前Q值和目标Q值
        current_q_values = self.main_network(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (1 - dones) * self.discount_factor * next_q_values
        
        # 计算损失并更新网络
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def update_target_network(self):
        """更新目标网络"""
        self.target_network.load_state_dict(self.main_network.state_dict())
    
    def train(self, episodes=2000, max_steps=200, update_target_freq=10):
        """训练DQN智能体"""
        episode_best_makespans = []
        
        for episode in range(episodes):
            current_sequence = copy.deepcopy(self.initial_sequence)
            episode_best_makespan = self.best_makespan
            total_reward = 0
            
            for step in range(max_steps):
                # 获取当前状态
                state = self._get_state(current_sequence)
                
                # 选择动作
                action = self.act(state)
                
                # 应用动作
                new_sequence = self._apply_action(current_sequence, action)
                
                # 更新动作历史
                self.action_history.append(action)
                
                # 计算奖励
                reward = self._get_reward(new_sequence)
                total_reward += reward
                
                # 获取新状态
                next_state = self._get_state(new_sequence)
                
                # 存储经验
                done = (step == max_steps - 1)
                self.remember(state, action, reward, next_state, done)
                
                # 从经验中学习
                loss = self.replay()
                
                # 更新当前序列
                current_sequence = new_sequence
                
                # 更新回合最佳makespan
                current_makespan = self._get_makespan(current_sequence)
                if current_makespan < episode_best_makespan:
                    episode_best_makespan = current_makespan
            
            # 定期更新目标网络
            if (episode + 1) % update_target_freq == 0:
                self.update_target_network()
            
            # 记录每个回合的最佳makespan
            episode_best_makespans.append(episode_best_makespan)
            
            # 每100回合打印一次进度
            if (episode + 1) % 100 == 0:
                print(f"回合 {episode + 1}/{episodes}, 最佳makespan: {self.best_makespan}")
                print(f"改进比例: {((self.initial_makespan - self.best_makespan) / self.initial_makespan) * 100:.2f}%")
        
        return self.best_sequence, self.best_makespan, episode_best_makespans

class CollaborativeSystem:
    def __init__(self, processing_times, initial_sequence, assembly_times, product_components):
        """初始化协作系统"""
        self.q_learning_agent = QLearningAgent(processing_times, initial_sequence, assembly_times, product_components)
        self.dqn_agent = DQNAgent(processing_times, initial_sequence, assembly_times, product_components)
        
        self.processing_times = processing_times
        self.initial_sequence = initial_sequence
        self.assembly_times = assembly_times
        self.product_components = product_components
        
        self.best_sequence = copy.deepcopy(initial_sequence)
        self.best_makespan = self._get_makespan(initial_sequence)
    
    def _get_makespan(self, sequence):
        """计算makespan"""
        _, assembly_completion_times = calculate_assembly_sequence_with_components(
            sequence, self.processing_times, self.assembly_times, self.product_components)
        return max(assembly_completion_times.values())
    
    def train(self, episodes=2000, max_steps=200):
        """训练协作系统"""
        # 训练Q-learning智能体
        q_sequence, q_makespan, q_history = self.q_learning_agent.train(episodes, max_steps)
        
        # 如果Q-learning智能体找到了更好的解，更新最佳解
        if q_makespan < self.best_makespan:
            self.best_sequence = q_sequence
            self.best_makespan = q_makespan
        
        # 使用Q-learning智能体的最佳解作为DQN智能体的初始解
        self.dqn_agent.initial_sequence = self.best_sequence
        self.dqn_agent.best_sequence = copy.deepcopy(self.best_sequence)
        self.dqn_agent.best_makespan = self.best_makespan
        self.dqn_agent.initial_makespan = self.best_makespan
        
        # 训练DQN智能体
        dqn_sequence, dqn_makespan, dqn_history = self.dqn_agent.train(episodes, max_steps)
        
        # 如果DQN智能体找到了更好的解，更新最佳解
        if dqn_makespan < self.best_makespan:
            self.best_sequence = dqn_sequence
            self.best_makespan = dqn_makespan
        
        return self.best_sequence, self.best_makespan, q_history, dqn_history 