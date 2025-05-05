from collections import deque
import math

class DelayReception:
    def __init__(self, L):
        """
        初始化延迟接受机制
        L: 队列长度，表示延迟步数
        """
        self.L = L
        self.queue = deque(maxlen=L)

    def add_solution(self, makespan):
        """
        添加一个新的makespan到队列
        """
        self.queue.append(makespan)

    def is_acceptable(self, new_makespan):
        """
        判断新解是否可接受：
        - 如果新解优于当前队列中的最优解，接受
        - 或者新解优于L步前的最优解（即队首），也接受
        """
        if not self.queue:
            return True  # 队列为空时直接接受
        best_now = min(self.queue)
        oldest = self.queue[0]
        return new_makespan < best_now or new_makespan < oldest

    def get_best(self):
        """
        获取当前队列中的最优解
        """
        if not self.queue:
            return None
        return min(self.queue)

    @staticmethod
    def adaptive_L(n, mode='mid'):
        """
        根据产品数n自适应选择L的大小
        mode: 'min'（0.5n），'mid'（1.25n），'max'（2n）
        """
        if mode == 'min':
            return max(1, int(round(0.5 * n)))
        elif mode == 'max':
            return max(1, int(round(2.0 * n)))
        else:  # 'mid'
            return max(1, int(round(1.25 * n))) 