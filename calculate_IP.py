import json
import math
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import matplotlib.colors as mcolors
import numpy as np
from scipy.stats import pearsonr
import seaborn as sns


def calculate_average_implicit_perception(frame):
    total_terrain_value = 0
    cnt = 0

    for agent in frame:
        total_terrain_value += agent['terrain_value']
        cnt += 1

    average_distance = total_terrain_value / cnt if cnt > 0 else 0
    return average_distance


def load_data_and_calculate_implicit_perception(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)

    average_implicit_perception = []
    for frame_index, frame in enumerate(data['frames']):
        frame_data = frame['frame']
        if len(frame_data) < 20:
            continue
        average_distance = calculate_average_implicit_perception(frame_data)
        if average_distance > 0:
            average_implicit_perception.append(average_distance)

    return average_implicit_perception


if __name__ == '__main__':
    # feature_combine
    file_paths_1 = [
        'record_agent_status\\eval_model_ppo-3090_1223_FeatureCombine_Turning_entity_1_implicit_0_20_35_44.json',
        'record_agent_status\\eval_model_ppo-3090_1223_FeatureCombine_Turning_entity_1_implicit_0.1_20_34_56.json',
        'record_agent_status\\eval_model_ppo-3090_1223_FeatureCombine_Turning_entity_1_implicit_0.2_20_33_50.json',
        'record_agent_status\\eval_model_ppo-3090_1223_FeatureCombine_Turning_entity_1_implicit_0.3_20_33_6.json',
        'record_agent_status\\eval_model_ppo-3090_1223_FeatureCombine_Turning_entity_1_implicit_0.4_20_32_21.json',
        'record_agent_status\\eval_model_ppo-3090_1223_FeatureCombine_Turning_entity_1_implicit_0.5_20_31_32.json',
        'record_agent_status\\eval_model_ppo-3090_1223_FeatureCombine_Turning_entity_1_implicit_0.6_20_30_38.json',
        'record_agent_status\\eval_model_ppo-3090_1223_FeatureCombine_Turning_entity_1_implicit_0.7_20_28_50.json',
        'record_agent_status\\eval_model_ppo-3090_1223_FeatureCombine_Turning_entity_1_implicit_0.8_20_27_48.json',
        'record_agent_status\\eval_model_ppo-3090_1223_FeatureCombine_Turning_entity_1_implicit_0.9_20_26_50.json',
        'record_agent_status\\eval_model_ppo-3090_1223_FeatureCombine_Turning_entity_1_implicit_1_20_25_38.json']

    # holistic
    file_paths_2 = [
        'record_agent_status\\eval_model_ppo-3090_250114_Holistic_Turning_entity_1_implicit_0_12_58_11.json',
        'record_agent_status\\eval_model_ppo-3090_250114_Holistic_Turning_entity_1_implicit_0.1_12_59_3.json',
        'record_agent_status\\eval_model_ppo-3090_250114_Holistic_Turning_entity_1_implicit_0.2_12_59_51.json',
        'record_agent_status\\eval_model_ppo-3090_250114_Holistic_Turning_entity_1_implicit_0.3_13_0_45.json',
        'record_agent_status\\eval_model_ppo-3090_250114_Holistic_Turning_entity_1_implicit_0.4_13_1_42.json',
        'record_agent_status\\eval_model_ppo-3090_250114_Holistic_Turning_entity_1_implicit_0.5_13_2_30.json',
        'record_agent_status\\eval_model_ppo-3090_250114_Holistic_Turning_entity_1_implicit_0.6_13_7_45.json',
        'record_agent_status\\eval_model_ppo-3090_250114_Holistic_Turning_entity_1_implicit_0.7_13_8_35.json',
        'record_agent_status\\eval_model_ppo-3090_250114_Holistic_Turning_entity_1_implicit_0.8_13_9_33.json',
        'record_agent_status\\eval_model_ppo-3090_250114_Holistic_Turning_entity_1_implicit_0.9_13_32_40.json',
        'record_agent_status\\eval_model_ppo-3090_250114_Holistic_Turning_entity_1_implicit_1_13_11_4.json',
    ]

    # holistic_net_no_weight_mlp
    file_paths_3 = [
        'record_agent_status\\eval_model_ppo-3080Ti_250113_Implicit_WoEnvTemp_Cross_entity_1_implicit_1_15_30_53.json',
        'record_agent_status\\eval_model_ppo-3080Ti_250113_Implicit_WoEnvTemp_Cross_entity_1_implicit_1_18_0_22.json',
        'record_agent_status\\eval_model_ppo-3080Ti_250113_Implicit_WoEnvTemp_Cross_entity_1_implicit_1_18_2_21.json',
        'record_agent_status\\eval_model_ppo-3080Ti_250113_Implicit_WoEnvTemp_Cross_entity_1_implicit_1_18_6_28.json',
        'record_agent_status\\eval_model_ppo-3080Ti_250113_Implicit_WoEnvTemp_Cross_entity_1_implicit_1_18_8_22.json',
        'record_agent_status\\eval_model_ppo-3080Ti_250113_Implicit_WoEnvTemp_Cross_entity_1_implicit_1_18_10_21.json',
        'record_agent_status\\eval_model_ppo-3080Ti_250113_Implicit_WoEnvTemp_Cross_entity_1_implicit_1_18_18_11.json',
        'record_agent_status\\eval_model_ppo-3080Ti_250113_Implicit_WoEnvTemp_Cross_entity_1_implicit_1_18_20_13.json',
        'record_agent_status\\eval_model_ppo-3080Ti_250113_Implicit_WoEnvTemp_Cross_entity_1_implicit_1_18_22_50.json',
        'record_agent_status\\eval_model_ppo-3080Ti_250113_Implicit_WoEnvTemp_Cross_entity_1_implicit_1_18_27_53.json',
        'record_agent_status\\eval_model_ppo-3080Ti_250116_Implicit_WithEnvTemp_Cross_entity_1_implicit_1_15_33_51.json',
        'record_agent_status\\eval_model_ppo-3080Ti_250116_Implicit_WithEnvTemp_Cross_entity_1_implicit_1_18_1_23.json',
        'record_agent_status\\eval_model_ppo-3080Ti_250116_Implicit_WithEnvTemp_Cross_entity_1_implicit_1_18_3_18.json',
        'record_agent_status\\eval_model_ppo-3080Ti_250116_Implicit_WithEnvTemp_Cross_entity_1_implicit_1_18_7_30.json',
        'record_agent_status\\eval_model_ppo-3080Ti_250116_Implicit_WithEnvTemp_Cross_entity_1_implicit_1_18_9_10.json',
        'record_agent_status\\eval_model_ppo-3080Ti_250116_Implicit_WithEnvTemp_Cross_entity_1_implicit_1_18_11_6.json',
        'record_agent_status\\eval_model_ppo-3080Ti_250116_Implicit_WithEnvTemp_Cross_entity_1_implicit_1_18_15_11.json',
        'record_agent_status\\eval_model_ppo-3080Ti_250116_Implicit_WithEnvTemp_Cross_entity_1_implicit_1_18_21_6.json',
        'record_agent_status\\eval_model_ppo-3080Ti_250116_Implicit_WithEnvTemp_Cross_entity_1_implicit_1_18_22_2.json',
        'record_agent_status\\eval_model_ppo-3080Ti_250116_Implicit_WithEnvTemp_Cross_entity_1_implicit_1_18_27_4.json',
    ]

    file_paths_4 = [
        'record_agent_status\\eval_model_ppo-3080Ti_250113_Implicit_WoEnvTemp_Cross_entity_1_implicit_1_15_30_53.json',
        'record_agent_status\\eval_model_ppo-3080Ti_250113_Implicit_WoEnvTemp_Cross_entity_1_implicit_1_18_0_22.json',
        'record_agent_status\\eval_model_ppo-3080Ti_250113_Implicit_WoEnvTemp_Cross_entity_1_implicit_1_18_2_21.json',
        'record_agent_status\\eval_model_ppo-3080Ti_250116_Implicit_WithEnvTemp_Cross_entity_1_implicit_1_15_33_51.json',
        'record_agent_status\\eval_model_ppo-3080Ti_250116_Implicit_WithEnvTemp_Cross_entity_1_implicit_1_18_1_23.json',
        'record_agent_status\\eval_model_ppo-3080Ti_250116_Implicit_WithEnvTemp_Cross_entity_1_implicit_1_18_3_18.json',
    ]

    file_paths = file_paths_1
    data_range = (0, 300)

    # Calculate the average implicit perception for each file
    average_implicit_perceptions = [load_data_and_calculate_implicit_perception(file_path) for file_path in file_paths]
    length = min(len(average_implicit_perception) for average_implicit_perception in average_implicit_perceptions)

    draw_image_1 = False
    draw_image_2 = False
    draw_image_3 = True

    if draw_image_1:
        # Figure 11(b): Performance comparison of with/without curriculum-based Env-templates.
        # 分为前一半和后一半文件
        mid_point = len(average_implicit_perceptions) // 2
        first_half = average_implicit_perceptions[:mid_point]
        second_half = average_implicit_perceptions[mid_point:]

        # 计算每帧的平均值和标准差
        first_half_avg = np.mean([implicit_perception[data_range[0]:data_range[1]] for implicit_perception in first_half],
                                 axis=0)
        first_half_std = np.std([implicit_perception[data_range[0]:data_range[1]] for implicit_perception in first_half],
                                axis=0)

        second_half_avg = np.mean([implicit_perception[data_range[0]:data_range[1]] for implicit_perception in second_half],
                                  axis=0)
        second_half_std = np.std([implicit_perception[data_range[0]:data_range[1]] for implicit_perception in second_half],
                                 axis=0)

        # 绘制曲线
        plt.figure(figsize=(10, 6))
        time_steps = range(data_range[0], data_range[1])

        # 绘制第一半部分曲线和标准差范围
        plt.plot(time_steps, first_half_avg, label="Without Env-Templates", color="blue", linewidth=6)
        plt.fill_between(time_steps, first_half_avg - first_half_std, first_half_avg + first_half_std, color="blue",
                         alpha=0.2)

        # 绘制第二半部分曲线和标准差范围
        plt.plot(time_steps, second_half_avg, label="With Env-Templates", color="red", linewidth=6)
        plt.fill_between(time_steps, second_half_avg - second_half_std, second_half_avg + second_half_std, color="red",
                         alpha=0.2)

        # 添加标签、图例和标题
        plt.xlabel("Time Step", fontsize=24)
        plt.ylabel("Implicit Pixel Value", fontsize=24)
        plt.legend(fontsize=24)
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    if draw_image_2:
        # Figure 9(a): Correlation between implicit weights and the average IP of crowds. (Time step)

        tick_fontsize = 16
        label_fontsize = 24
        curve_width = 3
        cmap = get_cmap("Blues")
        colors = [cmap(0.3 + 0.7 * (i / (len(average_implicit_perceptions) - 1))) for i in range(len(average_implicit_perceptions))]

        fig, ax = plt.subplots(figsize=(10, 6))
        for i, (implicit_perceptions, color) in enumerate(zip(average_implicit_perceptions, colors)):
            plt.plot(range(data_range[0], data_range[1]), implicit_perceptions[data_range[0]:data_range[1]],
                     label=f"Implicit weight =  {i / 10}", color=color, linewidth=curve_width)

        # Create a color bar as a gradient legend
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=mcolors.Normalize(vmin=0, vmax=1))
        cbar = fig.colorbar(sm, ax=ax, ticks=np.linspace(0, 1, 6))
        cbar.set_label("Implicit Weight", fontsize=24)
        cbar.ax.tick_params(labelsize=tick_fontsize)

        # Add labels, legend, and title
        ax.set_xlabel("Time Step", fontsize=label_fontsize)
        ax.set_ylabel("Implicit Pixel Value", fontsize=label_fontsize)
        plt.xticks(fontsize=tick_fontsize)  # 设置 X 轴刻度字体大小
        plt.yticks(fontsize=tick_fontsize)  # 设置 Y 轴刻度字体大小
        # plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    if draw_image_3:
        # Figure 9(b): Correlation between implicit weights and the average IP of crowds. (Correlation coefficient)
        tick_fontsize = 16
        label_fontsize = 22
        text_fontsize = 18
        curve_width = 5
        point_size = 100

        # 计算最小值
        min_implicit_value_array = [np.min(implicit_perceptions[data_range[0]:data_range[1]])
                                    for implicit_perceptions in average_implicit_perceptions]

        # 将 weights 数组设置为对应的 1 到 0 步长为 0.1 的数组
        weights = np.linspace(0, 1, 11)

        # 计算相关性系数
        average_corr, _ = pearsonr(min_implicit_value_array, weights)

        # 绘制散点图
        plt.figure(figsize=(8, 6))
        plt.scatter(weights, min_implicit_value_array, color='blue', alpha=0.7, s=point_size)
        sns.regplot(x=weights, y=min_implicit_value_array, color='blue', line_kws={'linewidth': curve_width})
        print(min_implicit_value_array)

        plt.text(
            0.5,  # x 坐标
            plt.ylim()[1] - 0.1 * (plt.ylim()[1] - plt.ylim()[0]),  # y 坐标
            f"Correlation Coefficient\nr = {average_corr:.2f}",
            fontsize=text_fontsize,
            color='black',
            ha='center',
            bbox=dict(facecolor='white', alpha=0.5, edgecolor='gray')
        )

        # 图表美化
        # plt.title(f"Scatter Plot of Average Min Implicit Values and Weights\nCorrelation Coefficient (r={average_corr:.2f})")
        plt.xlabel("Implicit Weight", fontsize=label_fontsize)
        plt.ylabel("Mean of Min IP Values", fontsize=label_fontsize)
        plt.xticks(fontsize=tick_fontsize)  # 设置 X 轴刻度字体大小
        plt.yticks(fontsize=tick_fontsize)  # 设置 Y 轴刻度字体大小
        plt.grid(True)

        # 显示图表
        plt.tight_layout()
        plt.show()
