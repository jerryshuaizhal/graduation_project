import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体为黑体
matplotlib.rcParams['axes.unicode_minus'] = False    # 正常显示负号

def plot_q_learning_curve(episode_best_makespans, initial_makespan, optimized_makespan):
    """
    绘制Q学习优化过程的学习曲线
    
    参数:
    episode_best_makespans: 每个回合的最佳makespan列表
    initial_makespan: 初始makespan
    optimized_makespan: 优化后的makespan
    """
    plt.figure(figsize=(12, 7))
    # 增加线宽和marker使线条更明显
    plt.plot(episode_best_makespans, linewidth=2, marker='.', markersize=3, 
             color='blue', label='每回合最佳makespan')
    plt.axhline(y=initial_makespan, color='r', linestyle='--', linewidth=2, label='初始makespan')
    plt.axhline(y=optimized_makespan, color='g', linestyle='--', linewidth=2, label='最佳makespan')
    
    # 自动调整y轴范围更合理
    min_y = min(min(episode_best_makespans), optimized_makespan) * 0.98
    max_y = max(max(episode_best_makespans), initial_makespan) * 1.02
    plt.ylim(min_y, max_y)
    
    plt.xlabel('回合')
    plt.ylabel('Makespan')
    plt.title('Q学习优化过程')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('q_learning_curve.png', dpi=300)  # 增加dpi提高图像质量
    plt.close()

def plot_gantt_chart(processing_sequence, processing_times, assembly_sequence, assembly_times, product_components):
    """
    绘制严格满足流水车间调度约束的甘特图
    """
    num_jobs = len(processing_sequence)
    num_machines = processing_times.shape[1]
    total_machines = num_machines + 1  # 加装配线

    # 计算每个工件在每台机器上的开始和结束时间
    start_times = np.zeros((num_jobs, num_machines))
    end_times = np.zeros((num_jobs, num_machines))
    for i, job in enumerate(processing_sequence):
        for m in range(num_machines):
            if i == 0 and m == 0:
                start = 0
            elif i == 0:
                start = end_times[i][m-1]
            elif m == 0:
                start = end_times[i-1][m]
            else:
                start = max(end_times[i][m-1], end_times[i-1][m])
            start_times[i][m] = start
            end_times[i][m] = start + processing_times[job][m]

    # 计算装配线的开始和结束时间（以产品为单位）
    product_ready_times = {}
    for product_id in assembly_sequence:
        components = product_components[product_id]
        component_finish_times = [end_times[processing_sequence.index(comp)][-1] for comp in components]
        product_ready_times[product_id] = max(component_finish_times)
    assembly_start_times = {}
    assembly_end_times = {}
    last_assembly_end = 0
    for product_id in assembly_sequence:
        start = max(product_ready_times[product_id], last_assembly_end)
        end = start + assembly_times[product_id]
        assembly_start_times[product_id] = start
        assembly_end_times[product_id] = end
        last_assembly_end = end

    # 绘图
    fig, ax = plt.subplots(figsize=(16, 9))
    # 工件颜色
    job_colors = plt.cm.tab20(np.linspace(0, 1, num_jobs))
    # 产品颜色（与工件颜色区分开，使用Set2或tab10）
    product_colors = plt.cm.Set2(np.linspace(0, 1, len(assembly_sequence)))
    machine_positions = np.arange(total_machines)

    # 绘制加工甘特图（以机器为主）
    for m in range(num_machines):
        for i, job in enumerate(processing_sequence):
            start = start_times[i][m]
            duration = end_times[i][m] - start
            ax.barh(machine_positions[m], duration, left=start, color=job_colors[i], edgecolor='black', alpha=0.8)
            ax.text(start + duration/2, machine_positions[m], f'工{job}', ha='center', va='center', color='black', fontsize=10)

    # 绘制装配线（用不同配色）
    for idx, product_id in enumerate(assembly_sequence):
        start = assembly_start_times[product_id]
        duration = assembly_end_times[product_id] - start
        color = product_colors[idx % len(product_colors)]
        ax.barh(machine_positions[-1], duration, left=start, color=color, edgecolor='black', alpha=0.8)
        ax.text(start + duration/2, machine_positions[-1], f'产{product_id}', ha='center', va='center', color='black', fontsize=10)

    # 设置Y轴
    machine_labels = [f'机器 {i+1}' for i in range(num_machines)] + ['装配机']
    ax.set_yticks(machine_positions)
    ax.set_yticklabels(machine_labels)
    ax.set_xlabel('时间')
    ax.set_ylabel('机器')
    ax.set_title('装配置换流水车间调度甘特图')
    ax.grid(True, axis='x', linestyle='--', alpha=0.7)

    # Makespan
    makespan = max(list(assembly_end_times.values()))
    ax.axvline(x=makespan, color='red', linestyle='--')
    ax.text(makespan+5, machine_positions[-1]+0.5, f'Makespan: {makespan}', color='red', fontweight='bold')

    plt.tight_layout()
    plt.savefig('gantt_chart.png')
    plt.close()

def calculate_completion_times(sequence, processing_times):
    """
    计算每个工件的完成时间矩阵
    
    参数:
    sequence: 工件序列
    processing_times: 加工时间矩阵
    
    返回:
    completion_times: 完成时间矩阵，行表示工件，列表示机器
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
    
    return completion_times