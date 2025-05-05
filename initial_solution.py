import numpy as np
import concurrent.futures
import multiprocessing

def calculate_total_time(processing_times):
    """
    计算每个工件的总处理时间
    
    参数:
    processing_times: 加工时间矩阵
    
    返回:
    total_times: 每个工件的总处理时间
    """
    return np.sum(processing_times, axis=1)

def calculate_makespan(sequence, processing_times):
    """
    计算给定序列的makespan
    
    参数:
    sequence: 工件序列
    processing_times: 加工时间矩阵
    
    返回:
    makespan: 完成时间
    """
    machines = processing_times.shape[1]
    completion_times = np.zeros((len(sequence), machines))
    
    # 第一个工件
    for j in range(machines):
        if j == 0:
            completion_times[0][j] = processing_times[sequence[0]][j]
        else:
            completion_times[0][j] = completion_times[0][j-1] + processing_times[sequence[0]][j]
    
    # 其余工件
    for i in range(1, len(sequence)):
        for j in range(machines):
            if j == 0:
                completion_times[i][j] = completion_times[i-1][j] + processing_times[sequence[i]][j]
            else:
                completion_times[i][j] = max(completion_times[i][j-1], completion_times[i-1][j]) + processing_times[sequence[i]][j]
    
    return completion_times[-1][-1]

def evaluate_insertion_position(args):
    """
    评估单个插入位置的makespan
    
    参数:
    args: 包含(sequence, current_job, position, processing_times)的元组
    
    返回:
    position: 插入位置
    makespan: 该位置的makespan
    """
    sequence, current_job, position, processing_times = args
    temp_sequence = sequence.copy()
    temp_sequence.insert(position, current_job)
    current_makespan = calculate_makespan(temp_sequence, processing_times)
    return position, current_makespan

def neh_algorithm(processing_times):
    """
    NEH算法求解流水线调度问题，使用多线程并行评估插入位置
    
    参数:
    processing_times: 加工时间矩阵
    
    返回:
    sequence: 工件序列
    """
    jobs = processing_times.shape[0]
    total_times = calculate_total_time(processing_times)
    
    # 按总处理时间降序排序
    sorted_jobs = np.argsort(-total_times)
    sequence = [sorted_jobs[0]]
    
    # 获取CPU核心数和线程数
    cpu_count = multiprocessing.cpu_count()
    # AMD R9 7945HX有16个核心32个线程，但为了系统稳定性，我们使用核心数的1.5倍
    max_workers = min(jobs, int(cpu_count * 1.5))
    print(f"检测到CPU核心数: {cpu_count}，使用 {max_workers} 个线程进行并行计算")
    
    # 依次插入剩余工件
    for i in range(1, jobs):
        current_job = sorted_jobs[i]
        best_makespan = float('inf')
        best_position = 0
        
        # 准备并行评估的参数
        evaluation_args = [(sequence, current_job, j, processing_times) 
                          for j in range(len(sequence) + 1)]
        
        # 使用线程池并行评估所有可能的插入位置
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(evaluate_insertion_position, evaluation_args))
        
        # 找出最佳插入位置
        for position, makespan in results:
            if makespan < best_makespan:
                best_makespan = makespan
                best_position = position
        
        sequence.insert(best_position, current_job)
    
    return sequence

def calculate_assembly_sequence_with_components(processing_sequence, processing_times, assembly_times, product_components):
    """
    计算装配序列和完成时间
    
    参数:
    processing_sequence: 加工序列
    processing_times: 加工时间矩阵
    assembly_times: 装配时间数组
    product_components: 产品组成字典
    
    返回:
    assembly_sequence: 装配序列
    assembly_completion_times: 装配完成时间
    """
    # 计算每个工件的完成时间
    machines = processing_times.shape[1]
    completion_times = np.zeros((len(processing_sequence), machines))
    
    # 计算每个工件的完成时间
    for i, job_id in enumerate(processing_sequence):
        for j in range(machines):
            if i == 0 and j == 0:
                completion_times[i][j] = processing_times[job_id][j]
            elif i == 0:
                completion_times[i][j] = completion_times[i][j-1] + processing_times[job_id][j]
            elif j == 0:
                completion_times[i][j] = completion_times[i-1][j] + processing_times[job_id][j]
            else:
                completion_times[i][j] = max(completion_times[i][j-1], completion_times[i-1][j]) + processing_times[job_id][j]
    
    # 计算每个产品的就绪时间
    product_ready_times = {}
    for product_id, components in product_components.items():
        component_completion_times = []
        for comp_id in components:
            if comp_id in processing_sequence:
                idx = processing_sequence.index(comp_id)
                component_completion_times.append(completion_times[idx][-1])
        product_ready_times[product_id] = max(component_completion_times) if component_completion_times else 0
    
    # 按就绪时间排序产品
    assembly_sequence = sorted(product_ready_times.keys(), key=lambda x: product_ready_times[x])
    
    # 计算装配完成时间
    assembly_completion_times = {}
    current_time = 0
    for product_id in assembly_sequence:
        start_time = max(current_time, product_ready_times[product_id])
        assembly_completion_times[product_id] = start_time + assembly_times[product_id]
        current_time = assembly_completion_times[product_id]
    
    return assembly_sequence, assembly_completion_times