import numpy as np
import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
from initial_solution import neh_algorithm, calculate_makespan, calculate_assembly_sequence_with_components
from Qlearning import QLearningAgent
from collaborative_agents import CollaborativeSystem
from draw import plot_q_learning_curve, plot_gantt_chart

def load_data_from_txt(file_path):
    """
    从TXT文件加载数据，适配新的数据结构
    格式：
    - 第1行：工件数
    - 第2行：机器数
    - 第3行：忽略（工厂数）
    - 接下来是每个工件在每个机器上的加工时间
    - 然后是产品数量、装配时间和产品-工件对应关系
    """
    with open(file_path, 'r') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]  # 去除空行
    
    # 解析工件数和机器数
    line_idx = 0
    num_jobs = int(lines[line_idx])
    line_idx += 1
    num_machines = int(lines[line_idx])
    line_idx += 1
    
    # 跳过第三行（工厂数）
    line_idx += 1
    
    # 解析加工时间矩阵
    processing_times = np.zeros((num_jobs, num_machines))
    for job_id in range(num_jobs):
        for machine_id in range(num_machines):
            # 跳过标号行（比如 "0"）
            line_idx += 1
            # 读取加工时间
            processing_times[job_id][machine_id] = int(lines[line_idx])
            line_idx += 1
    
    # 找到关键词"NumAssemblySet"，解析产品数量
    while line_idx < len(lines) and "NumAssemblySet" not in lines[line_idx]:
        line_idx += 1
    
    line_idx += 1  # 跳到产品数量行
    num_products = int(lines[line_idx])
    line_idx += 1
    
    # 找到关键词"ProTimeAssemblySet"，解析装配时间
    while line_idx < len(lines) and "ProTimeAssemblySet" not in lines[line_idx]:
        line_idx += 1
    
    line_idx += 1  # 进入装配时间区域
    assembly_times = np.zeros(num_products)
    for p in range(num_products):
        line_idx += 1  # 跳过序号行
        assembly_times[p] = int(lines[line_idx])
        line_idx += 1
    
    # 找到关键词"NumJob_NumSet"，解析产品-工件关系
    while line_idx < len(lines) and "NumJob_NumSet" not in lines[line_idx]:
        line_idx += 1
    
    line_idx += 1  # 进入产品-工件关系区域
    # 创建产品组件字典：key是产品ID，value是该产品包含的工件列表
    product_components = {p: [] for p in range(num_products)}
    
    for job_id in range(1, num_jobs + 1):  # 工件ID从1开始
        line_idx += 1  # 跳过工件ID行
        product_id = int(lines[line_idx]) - 1  # 产品ID调整为从0开始
        product_components[product_id].append(job_id - 1)  # 工件ID也调整为从0开始
        line_idx += 1
    
    return processing_times, product_components, assembly_times

def process_single_file(file_path, episodes=2000, generate_plots=True):
    """处理单个文件并返回结果"""
    try:
        # 从文件加载数据
        processing_times, product_components, assembly_times = load_data_from_txt(file_path)
        
        # 获取问题规模信息
        num_jobs = processing_times.shape[0]
        num_machines = processing_times.shape[1]
        num_products = len(product_components)
        
        # 使用NEH算法获取初始解
        initial_sequence = neh_algorithm(processing_times)
        
        # 使用协作系统进行优化
        collaborative_system = CollaborativeSystem(processing_times, initial_sequence, assembly_times, product_components)
        optimized_sequence, best_makespan, q_history, dqn_history = collaborative_system.train(episodes=episodes)
        
        # 使用协作系统输出的值
        initial_makespan = collaborative_system.q_learning_agent.initial_makespan
        best_makespan = collaborative_system.best_makespan
        improvement_rate = ((initial_makespan - best_makespan) / initial_makespan) * 100
        
        # 计算装配序列
        assembly_sequence, assembly_completion_times = calculate_assembly_sequence_with_components(
            optimized_sequence, processing_times, assembly_times, product_components)
        
        # 生成图表
        if generate_plots:
            # 绘制Q-learning和DQN的学习曲线
            plt.figure(figsize=(12, 7))
            plt.plot(q_history, label='Q-learning', alpha=0.7)
            plt.plot(dqn_history, label='DQN', alpha=0.7)
            plt.axhline(y=initial_makespan, color='r', linestyle='--', label='初始makespan')
            plt.axhline(y=best_makespan, color='g', linestyle='--', label='最佳makespan')
            plt.xlabel('回合')
            plt.ylabel('Makespan')
            plt.title('协作系统优化过程')
            plt.legend()
            plt.grid(True)
            plt.savefig('collaborative_learning_curve.png', dpi=300)
            plt.close()
            
            # 绘制甘特图
            plot_gantt_chart(optimized_sequence, processing_times, assembly_sequence, assembly_times, product_components)
        
        return {
            'file_name': os.path.basename(file_path),
            'num_jobs': num_jobs,
            'num_machines': num_machines,
            'num_products': num_products,
            'initial_makespan': initial_makespan,
            'best_makespan': best_makespan,
            'improvement_rate': improvement_rate
        }
    except Exception as e:
        print(f"处理文件 {file_path} 时出错: {e}")
        return None

def process_directory(directory_path, episodes=2000, generate_plots=False):
    """处理目录下的所有txt文件并汇总结果"""
    results = []
    
    # 遍历目录下的所有txt文件
    for root, _, files in os.walk(directory_path):
        for file in files:
            if file.endswith('.txt'):
                file_path = os.path.join(root, file)
                print(f"\n处理文件: {file_path}")
                result = process_single_file(file_path, episodes, generate_plots)
                if result:
                    results.append(result)
                    print(f"完成处理: {file} (改进率: {result['improvement_rate']:.2f}%)")
    
    # 将结果转换为DataFrame
    if results:
        df = pd.DataFrame(results)
        
        # 按问题规模分组计算统计信息
        grouped = df.groupby(['num_jobs', 'num_machines', 'num_products']).agg({
            'improvement_rate': ['mean', 'std', 'min', 'max', 'count'],
            'initial_makespan': ['mean', 'min', 'max'],
            'best_makespan': ['mean', 'min', 'max']
        }).round(2)
        
        # 保存结果到Excel
        output_file = os.path.join(directory_path, 'results_summary.xlsx')
        with pd.ExcelWriter(output_file) as writer:
            df.to_excel(writer, sheet_name='详细结果', index=False)
            grouped.to_excel(writer, sheet_name='统计汇总')
            
            # 添加overall汇总表
            overall = pd.DataFrame({
                '统计项': ['文件总数', '平均初始makespan', '平均优化后makespan', '平均改进率(%)', '最大改进率(%)', '最小改进率(%)'],
                '值': [
                    len(df),
                    df['initial_makespan'].mean().round(2),
                    df['best_makespan'].mean().round(2),
                    df['improvement_rate'].mean().round(2),
                    df['improvement_rate'].max().round(2),
                    df['improvement_rate'].min().round(2)
                ]
            })
            overall.to_excel(writer, sheet_name='整体汇总', index=False)
        
        print(f"\n汇总结果:")
        print(f"处理文件总数: {len(df)}")
        print(f"平均改进率: {df['improvement_rate'].mean().round(2)}%")
        print(f"最大改进率: {df['improvement_rate'].max().round(2)}%")
        print(f"最小改进率: {df['improvement_rate'].min().round(2)}%")
        print(f"结果已保存到: {output_file}")
    else:
        print("没有找到可处理的txt文件")

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='流水车间调度优化程序')
    parser.add_argument('--input', type=str, required=True, help='输入数据的文件路径或目录路径')
    parser.add_argument('--episodes', type=int, default=2000, help='Q学习训练回合数')
    parser.add_argument('--no-plots', action='store_true', help='不生成图表(提高批处理速度)')
    args = parser.parse_args()
    
    # 是否生成图表
    generate_plots = not args.no_plots
    
    if os.path.isfile(args.input):
        # 处理单个文件
        result = process_single_file(args.input, args.episodes, generate_plots)
        if result:
            print(f"\n处理结果:")
            print(f"工件数: {result['num_jobs']}")
            print(f"机器数: {result['num_machines']}")
            print(f"产品数: {result['num_products']}")
            print(f"初始makespan: {result['initial_makespan']}")
            print(f"优化后makespan: {result['best_makespan']}")
            print(f"改进率: {result['improvement_rate']:.2f}%")
    elif os.path.isdir(args.input):
        # 处理目录
        process_directory(args.input, args.episodes, generate_plots)
    else:
        print(f"错误: 输入路径 {args.input} 不存在")

if __name__ == "__main__":
    main()
