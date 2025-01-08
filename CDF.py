import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# 读取 CSV 文件
data = pd.read_csv('evaluation_results_M3FD.csv')

# 指定要分析的算法
algorithms = ['Ours', 'DenseFuse', 'RFN-Nest', 'FusionGAN', 'IFCNN', 'SDNet', 'U2Fusion', 'FLFuse',
              'SeAFusion', 'PIAFusion']

# 指定要计算 CDF 的指标
metrics = ['EN', 'MI', 'SF', 'AG', 'SD', 'VIF', 'SCD']

# 自定义的标记样式和颜色
markers = ['D', '*', '|', 'v', '^', '+', 's', 'x', 'o', 'p']  # 根据您的描述设置
marker_colors = ['#FF4500', '#00FF00', '#0000FF', '#FFA500', '#800080', '#00FFFF', '#FF00FF',
                 '#FFFF00', '#FF69B4', '#008080']  # 调整为更加对比鲜明的颜色

marker_dict = {
    'Ours': 'D',  # 菱形
    'DenseFuse': '*',  # 空心五角星
    'RFN-Nest': '|',  # 小竖线
    'FusionGAN': 'v',  # 空心倒三角
    'IFCNN': '^',  # 空心正三角
    'PMGI': '+',  # 三叉线
    'SDNet': 's',  # 空心正方形
    'U2Fusion': 'x',  # 小×
    'FLFuse': 'o',  # 圆形
    'SeAFusion': 'p',  # 空心五边形
    'PIAFusion': 'h',  # 六边形
}

# 设置字体为 Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'

for metric in metrics:
    plt.figure(figsize=(6.8, 4.5))  # 创建新的图形
    for idx, algorithm in enumerate(algorithms):
        # 筛选出当前算法的数据
        algorithm_data = data[data['Algorithm'] == algorithm]

        if metric in algorithm_data.columns:
            # 检查数据是否为空
            if algorithm_data[metric].empty:
                print(f"No data for {algorithm} with metric {metric}. Skipping...")
                continue

            # 排序数据
            sorted_data = np.sort(algorithm_data[metric].values)
            # 计算 CDF
            cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)

            # 插值以生成更平滑的曲线
            interp_func = interp1d(cdf, sorted_data, kind='linear', bounds_error=False, fill_value='extrapolate')
            cdf_new = np.linspace(0, 1, num=20)  # 生成更多的 CDF 点
            sorted_data_new = interp_func(cdf_new)

            # 获取标记样式和颜色
            marker_style = marker_dict.get(algorithm, 'x')  # 默认标记样式为 'x'
            color = marker_colors[idx % len(marker_colors)]

            # 绘制 CDF
            plt.plot(cdf_new, sorted_data_new, label=algorithm,
                     color=color,  # 折线颜色与边框颜色一致
                     marker=marker_style, markersize=5,
                     markerfacecolor='none',  # 空心标记
                     markeredgecolor=color)  # 不同的边框颜色

    # 添加图例和标签
    plt.title(metric, fontsize=14, fontweight='bold')  # 加粗标题
    plt.xlabel('Cumulative Distribution', fontsize=12, fontweight='bold')  # 加粗横坐标标签
    plt.ylabel('Values of The Metric', fontsize=12, fontweight='bold')  # 加粗纵坐标标签

    # 计算平均值并添加到图例中
    legend_labels = []
    for algorithm in algorithms:
        algorithm_data = data[data['Algorithm'] == algorithm]
        if metric in algorithm_data.columns:
            mean_value = algorithm_data[metric].mean()
            legend_labels.append(f"{algorithm}: {mean_value:.4f}")  # 保留四位小数

    # 取消图例框框并调整高度
    legend = plt.legend(legend_labels, loc='center left', bbox_to_anchor=(1, 0.5), labelspacing=1.5, fontsize=10, title_fontsize='12', frameon=False)
    for text in legend.get_texts():
        text.set_fontsize(10)  # 设置图例中文本的字体大小
        text.set_fontweight('bold')  # 加粗图例中文本

    # 加粗刻度数字
    plt.tick_params(axis='both', labelsize=10)  # 设置刻度标签的字体大小
    for label in plt.gca().get_xticklabels() + plt.gca().get_yticklabels():
        label.set_fontsize(10)
        label.set_fontweight('bold')

    plt.grid(False)  # 取消网格线
    plt.tight_layout()  # 自动调整布局以适应图例

    # 保存为矢量图（PDF格式）
    plt.savefig(f'{metric}_CDF_plot.pdf', format='pdf', bbox_inches='tight')  # 保存为PDF文件
    plt.show()
