import numpy as np
import random
import copy
from initial_solution import calculate_makespan, calculate_assembly_sequence_with_components

class QLearningAgent:
    def __init__(self, processing_times, initial_sequence, assembly_times, product_components, learning_rate=0.1, discount_factor=0.9, epsilon=0.3):
        """
        初始化Q学习智能体
        
        参数:
        processing_times: 加工时间矩阵
        initial_sequence: 初始序列（由NEH算法生成）
        assembly_times: 装配时间数组
        product_components: 产品组成字典
        learning_rate: 学习率
        discount_factor: 折扣因子
        epsilon: 探索率
        """
        self.processing_times = processing_times
        self.initial_sequence = initial_sequence
        self.assembly_times = assembly_times
        self.product_components = product_components
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        
        # 初始化Q表
        # 状态空间：S1(开始)、S2(无改进)、S3(改进)
        # 动作空间：LLH1-LLH15
        self.q_table = {
            'S1': {f'LLH{i}': 0 for i in range(1, 16)},
            'S2': {f'LLH{i}': 0 for i in range(1, 16)},
            'S3': {f'LLH{i}': 0 for i in range(1, 16)}
        }
        
        # 记录最佳序列和其makespan（以最后一件产品装配完工时间为makespan）
        _, assembly_completion_times = calculate_assembly_sequence_with_components(
            initial_sequence, processing_times, assembly_times, product_components)
        self.best_sequence = copy.deepcopy(initial_sequence)
        self.best_makespan = max(assembly_completion_times.values())
        
        # 记录初始makespan
        self.initial_makespan = self.best_makespan
        
        # 当前状态
        self.current_state = 'S1'
    
    def _get_possible_actions(self):
        """获取所有可能的启发式动作"""
        return [f'LLH{i}' for i in range(1, 16)]
    
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
        if action == 'LLH1':
            # 随机选择两个位置交换
            i, j = random.sample(range(n), 2)
            swap(i, j)
        
        elif action == 'LLH2':
            # 随机选择一个位置，将其向前插入到任意位置
            i = random.randint(0, n-1)
            j = random.randint(0, i)
            move_forward(i, j)
        
        elif action == 'LLH3':
            # 随机选择一个位置，将其向后插入到任意位置
            i = random.randint(0, n-1)
            j = random.randint(i, n-1)
            move_backward(i, j)
        
        elif action == 'LLH4':
            # 依次执行LLH1和LLH2
            i, j = random.sample(range(n), 2)
            swap(i, j)
            i = random.randint(0, n-1)
            j = random.randint(0, i)
            move_forward(i, j)
        
        elif action == 'LLH5':
            # 依次执行LLH2和LLH1
            i = random.randint(0, n-1)
            j = random.randint(0, i)
            move_forward(i, j)
            i, j = random.sample(range(n), 2)
            swap(i, j)
        
        elif action == 'LLH6':
            # 依次执行LLH1和LLH3
            i, j = random.sample(range(n), 2)
            swap(i, j)
            i = random.randint(0, n-1)
            j = random.randint(i, n-1)
            move_backward(i, j)
        
        elif action == 'LLH7':
            # 依次执行LLH3和LLH1
            i = random.randint(0, n-1)
            j = random.randint(i, n-1)
            move_backward(i, j)
            i, j = random.sample(range(n), 2)
            swap(i, j)
        
        elif action == 'LLH8':
            # 依次执行LLH2和LLH3
            i = random.randint(0, n-1)
            j = random.randint(0, i)
            move_forward(i, j)
            i = random.randint(0, n-1)
            j = random.randint(i, n-1)
            move_backward(i, j)
        
        elif action == 'LLH9':
            # 依次执行LLH3和LLH2
            i = random.randint(0, n-1)
            j = random.randint(i, n-1)
            move_backward(i, j)
            i = random.randint(0, n-1)
            j = random.randint(0, i)
            move_forward(i, j)
        
        elif action == 'LLH10':
            # 依次执行LLH1、LLH2和LLH3
            i, j = random.sample(range(n), 2)
            swap(i, j)
            i = random.randint(0, n-1)
            j = random.randint(0, i)
            move_forward(i, j)
            i = random.randint(0, n-1)
            j = random.randint(i, n-1)
            move_backward(i, j)
        
        elif action == 'LLH11':
            # 依次执行LLH1、LLH3和LLH2
            i, j = random.sample(range(n), 2)
            swap(i, j)
            i = random.randint(0, n-1)
            j = random.randint(i, n-1)
            move_backward(i, j)
            i = random.randint(0, n-1)
            j = random.randint(0, i)
            move_forward(i, j)
        
        elif action == 'LLH12':
            # 依次执行LLH2、LLH3和LLH1
            i = random.randint(0, n-1)
            j = random.randint(0, i)
            move_forward(i, j)
            i = random.randint(0, n-1)
            j = random.randint(i, n-1)
            move_backward(i, j)
            i, j = random.sample(range(n), 2)
            swap(i, j)
        
        elif action == 'LLH13':
            # 依次执行LLH2、LLH1和LLH3
            i = random.randint(0, n-1)
            j = random.randint(0, i)
            move_forward(i, j)
            i, j = random.sample(range(n), 2)
            swap(i, j)
            i = random.randint(0, n-1)
            j = random.randint(i, n-1)
            move_backward(i, j)
        
        elif action == 'LLH14':
            # 依次执行LLH3、LLH2和LLH1
            i = random.randint(0, n-1)
            j = random.randint(i, n-1)
            move_backward(i, j)
            i = random.randint(0, n-1)
            j = random.randint(0, i)
            move_forward(i, j)
            i, j = random.sample(range(n), 2)
            swap(i, j)
        
        elif action == 'LLH15':
            # 依次执行LLH3、LLH1和LLH2
            i = random.randint(0, n-1)
            j = random.randint(i, n-1)
            move_backward(i, j)
            i, j = random.sample(range(n), 2)
            swap(i, j)
            i = random.randint(0, n-1)
            j = random.randint(0, i)
            move_forward(i, j)
        
        return new_sequence
    
    def _get_makespan(self, sequence):
        """计算makespan"""
        _, assembly_completion_times = calculate_assembly_sequence_with_components(
            sequence, self.processing_times, self.assembly_times, self.product_components)
        return max(assembly_completion_times.values())
    
    def _get_reward(self, sequence):
        """计算奖励（基于makespan的改进）"""
        makespan = self._get_makespan(sequence)
        
        # 如果makespan小于当前最佳，更新最佳序列
        if makespan < self.best_makespan:
            improvement = self.best_makespan - makespan
            self.best_sequence = copy.deepcopy(sequence)
            self.best_makespan = makespan
            # 奖励与改进成正比
            return 10 * (1 + improvement / self.initial_makespan)
        
        # 如果makespan大于当前最佳，给予惩罚
        if makespan > self.best_makespan:
            deterioration = makespan - self.best_makespan
            # 惩罚与恶化成正比
            return -5 * (1 + deterioration / self.initial_makespan)
        
        # 如果makespan等于当前最佳，给予小奖励
        return 1
    
    def _update_state(self, sequence):
        """更新状态"""
        makespan = self._get_makespan(sequence)
        
        if makespan < self.best_makespan:
            # 有改进，状态变为S3
            self.current_state = 'S3'
        else:
            # 无改进，状态变为S2
            self.current_state = 'S2'
    
    def train(self, episodes=2000, max_steps=200):
        """
        训练Q学习智能体
        
        参数:
        episodes: 训练回合数
        max_steps: 每回合最大步数
        """
        # 记录每个回合的最佳makespan
        episode_best_makespans = []
        
        for episode in range(episodes):
            current_sequence = copy.deepcopy(self.initial_sequence)
            episode_best_makespan = self.best_makespan
            
            # 重置状态为S1
            self.current_state = 'S1'
            
            for step in range(max_steps):
                # ε-贪婪策略选择动作
                if random.random() < self.epsilon:
                    action = random.choice(self._get_possible_actions())
                else:
                    # 选择Q值最大的动作
                    actions = self._get_possible_actions()
                    best_action = None
                    best_q_value = float('-inf')
                    
                    for a in actions:
                        if self.q_table[self.current_state][a] > best_q_value:
                            best_q_value = self.q_table[self.current_state][a]
                            best_action = a
                    
                    action = best_action
                
                # 应用动作
                new_sequence = self._apply_action(current_sequence, action)
                
                # 计算奖励
                reward = self._get_reward(new_sequence)
                
                # 更新状态
                old_state = self.current_state
                self._update_state(new_sequence)
                new_state = self.current_state
                
                # 获取下一状态的最大Q值
                next_max_q = max(self.q_table[new_state].values())
                
                # Q学习更新公式
                self.q_table[old_state][action] += self.learning_rate * (
                    reward + self.discount_factor * next_max_q - self.q_table[old_state][action]
                )
                
                # 更新当前序列
                current_sequence = new_sequence
                
                # 更新回合最佳makespan
                current_makespan = self._get_makespan(current_sequence)
                if current_makespan < episode_best_makespan:
                    episode_best_makespan = current_makespan
            
            # 记录每个回合的最佳makespan
            episode_best_makespans.append(episode_best_makespan)
            
            # 每100回合打印一次进度
            if (episode + 1) % 100 == 0:
                print(f"回合 {episode + 1}/{episodes}, 最佳makespan: {self.best_makespan}, 初始makespan: {self.initial_makespan}")
                print(f"改进比例: {((self.initial_makespan - self.best_makespan) / self.initial_makespan) * 100:.2f}%")
                # print(f"Q表: {self.q_table}")
        
        return self.best_sequence, self.best_makespan, episode_best_makespans

def optimize_with_q_learning(processing_times, initial_sequence, assembly_times, product_components, episodes=2000, max_steps=200):
    """
    使用Q学习优化初始序列
    
    参数:
    processing_times: 加工时间矩阵
    initial_sequence: 初始序列（由NEH算法生成）
    assembly_times: 装配时间数组
    product_components: 产品组成字典
    episodes: 训练回合数
    max_steps: 每回合最大步数
    
    返回:
    优化后的序列和其makespan
    """
    agent = QLearningAgent(processing_times, initial_sequence, assembly_times, product_components)
    optimized_sequence, optimized_makespan, episode_best_makespans = agent.train(episodes, max_steps)
    return optimized_sequence, optimized_makespan, episode_best_makespans
