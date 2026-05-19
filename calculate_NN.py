import json
import math
import matplotlib.pyplot as plt
import os
from matplotlib import colormaps
import matplotlib.colors as mcolors
import numpy as np
from scipy.stats import pearsonr
import seaborn as sns


def get_file_list_as_string_list(directory):
    # Get all files in the given directory
    file_list = os.listdir(directory)

    # Create the formatted string list with full paths
    string_list = [
        f"{os.path.join(directory, filename)}" for filename in file_list if
        os.path.isfile(os.path.join(directory, filename))
    ]

    return string_list


def calculate_average_distance(frame):
    threshold = 999
    total_distance = 0
    cnt = 0
    num_agents = len(frame)
    min_distances = []

    for agent in frame:
        min_distance = float('inf')
        agent_id = agent['agent_id']
        position = agent['position']

        for neighbor in frame:
            if neighbor['agent_id'] != agent_id:
                neighbor_position = neighbor['position']
                distance = math.sqrt(
                    (position['x'] - neighbor_position['x']) ** 2 +
                    (position['y'] - neighbor_position['y']) ** 2
                )
                if distance < min_distance:
                    min_distance = distance
        if min_distance < threshold:
            min_distances.append(min_distance)
            total_distance += min_distance
            cnt += 1

    average_distance = total_distance / cnt if cnt > 0 else 0
    return average_distance, min_distances


def calculate_average_collision_cnt(frame):
    threshold = 2.0
    cnt = 0

    for agent in frame:
        min_distance = float('inf')
        agent_id = agent['agent_id']
        position = agent['position']

        for neighbor in frame:
            if neighbor['agent_id'] != agent_id:
                neighbor_position = neighbor['position']
                distance = math.sqrt(
                    (position['x'] - neighbor_position['x']) ** 2 +
                    (position['y'] - neighbor_position['y']) ** 2
                )
                if distance < threshold:
                    cnt += 1

    return cnt


def load_data_and_calculate_distances(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)

    agent_num = len(data['frames'][0]['frame'])
    print(agent_num)
    average_distances = []
    min_distances_array = []
    for frame_index, frame in enumerate(data['frames']):
        average_distance = 0
        min_distances = []
        frame_data = frame['frame']
        if len(frame_data) == 0:
            continue
        if len(frame_data) == agent_num:
            average_distance, min_distances = calculate_average_distance(frame_data)
        if average_distance > 0:
            average_distances.append(average_distance)
            min_distances_array.append(min_distances)

    return average_distances, min_distances_array[average_distances.index(min(average_distances))]


def load_data_and_calculate_collision_cnt(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)

    collision_cnts = []
    for frame_index, frame in enumerate(data['frames']):
        frame_data = frame['frame']
        if len(frame_data) == 0:
            continue
        collision_cnt = calculate_average_collision_cnt(frame_data)
        if collision_cnt > 0:
            collision_cnts.append(collision_cnt)

    return collision_cnts


if __name__ == '__main__':
    directory = 'record_agent_status'
    files = get_file_list_as_string_list(directory)
    print(files)

    # feature_combine
    file_paths_1 = [
        'record_agent_status\\eval_model_ppo-3090_1223_FeatureCombine_Turning_entity_0_implicit_1_13_45_56.json',
        'record_agent_status\\eval_model_ppo-3090_1223_FeatureCombine_Turning_entity_0.1_implicit_1_13_47_7.json',
        'record_agent_status\\eval_model_ppo-3090_1223_FeatureCombine_Turning_entity_0.2_implicit_1_13_48_48.json',
        'record_agent_status\\eval_model_ppo-3090_1223_FeatureCombine_Turning_entity_0.3_implicit_1_15_10_47.json',
        'record_agent_status\\eval_model_ppo-3090_1223_FeatureCombine_Turning_entity_0.4_implicit_1_15_11_59.json',
        'record_agent_status\\eval_model_ppo-3090_1223_FeatureCombine_Turning_entity_0.5_implicit_1_15_14_21.json',
        'record_agent_status\\eval_model_ppo-3090_1223_FeatureCombine_Turning_entity_0.6_implicit_1_15_15_53.json',
        'record_agent_status\\eval_model_ppo-3090_1223_FeatureCombine_Turning_entity_0.7_implicit_1_15_17_9.json',
        'record_agent_status\\eval_model_ppo-3090_1223_FeatureCombine_Turning_entity_0.8_implicit_1_15_18_41.json',
        'record_agent_status\\eval_model_ppo-3090_1223_FeatureCombine_Turning_entity_0.9_implicit_1_15_20_8.json',
        'record_agent_status\\eval_model_ppo-3090_1223_FeatureCombine_Turning_entity_1_implicit_1_15_21_48.json',
    ]

    # holistic
    file_paths_2 = [
        'record_agent_status\\eval_model_ppo-3090_250114_Holistic_Turning_entity_0_implicit_1_16_26_20.json',
        'record_agent_status\\eval_model_ppo-3090_250114_Holistic_Turning_entity_0.1_implicit_1_16_28_16.json',
        'record_agent_status\\eval_model_ppo-3090_250114_Holistic_Turning_entity_0.2_implicit_1_16_29_10.json',
        'record_agent_status\\eval_model_ppo-3090_250114_Holistic_Turning_entity_0.3_implicit_1_16_30_26.json',
        'record_agent_status\\eval_model_ppo-3090_250114_Holistic_Turning_entity_0.4_implicit_1_16_31_25.json',
        'record_agent_status\\eval_model_ppo-3090_250114_Holistic_Turning_entity_0.5_implicit_1_16_32_16.json',
        'record_agent_status\\eval_model_ppo-3090_250114_Holistic_Turning_entity_0.6_implicit_1_16_33_8.json',
        'record_agent_status\\eval_model_ppo-3090_250114_Holistic_Turning_entity_0.7_implicit_1_16_34_10.json',
        'record_agent_status\\eval_model_ppo-3090_250114_Holistic_Turning_entity_0.8_implicit_1_16_35_6.json',
        'record_agent_status\\eval_model_ppo-3090_250114_Holistic_Turning_entity_0.9_implicit_1_16_36_1.json',
        'record_agent_status\\eval_model_ppo-3090_250114_Holistic_Turning_entity_1_implicit_1_16_37_4.json',
    ]

    # holistic_no_weight_mlp
    file_paths_3 = [
        'record_agent_status\\eval_model_ppo-2025_1_15_19_19_16_entity_0_implicit_1_12_31_11.json',
        'record_agent_status\\eval_model_ppo-2025_1_15_19_19_16_entity_0.1_implicit_1_12_32_5.json',
        'record_agent_status\\eval_model_ppo-2025_1_15_19_19_16_entity_0.2_implicit_1_12_37_57.json',
        'record_agent_status\\eval_model_ppo-2025_1_15_19_19_16_entity_0.3_implicit_1_12_39_59.json',
        'record_agent_status\\eval_model_ppo-2025_1_15_19_19_16_entity_0.4_implicit_1_12_40_50.json',
        'record_agent_status\\eval_model_ppo-2025_1_15_19_19_16_entity_0.5_implicit_1_12_41_46.json',
        'record_agent_status\\eval_model_ppo-2025_1_15_19_19_16_entity_0.6_implicit_1_12_42_37.json',
        'record_agent_status\\eval_model_ppo-2025_1_15_19_19_16_entity_0.7_implicit_1_12_43_28.json',
        'record_agent_status\\eval_model_ppo-2025_1_15_19_19_16_entity_0.8_implicit_1_12_44_20.json',
        'record_agent_status\\eval_model_ppo-2025_1_15_19_19_16_entity_0.9_implicit_1_12_45_8.json',
        'record_agent_status\\eval_model_ppo-2025_1_15_19_19_16_entity_1_implicit_1_12_45_57.json'
    ]

    file_paths = file_paths_1
    data_range = (100, 500)

    # Calculate the average distances and min distances for each file
    average_distances, min_distances_array = zip(
        *[load_data_and_calculate_distances(file_path) for file_path in file_paths])

    # 将 zip 的结果转换为列表（如果需要列表形式）
    average_distances = list(average_distances)
    min_distances_array = list(min_distances_array)
    for min_dis in average_distances:
        print(np.min(min_dis))

    length = min(len(average_average_distance) for average_average_distance in average_distances)

    # ----------------------Draw plots--------------------------#
    draw_image_1 = True
    draw_image_2 = True

    if draw_image_1:
        # Figure 7(a): Correlation between entity weight and the average DNN of crowds. (Time step)
        tick_fontsize = 16
        label_fontsize = 24
        curve_width = 3
        cmap = colormaps["Blues"]
        colors = [cmap(0.3 + 0.7 * (i / (len(average_distances) - 1))) for i in range(len(average_distances))]

        fig, ax = plt.subplots(figsize=(10, 6))
        for i, (distances, color) in enumerate(zip(average_distances, colors)):
            if len(data_range) == 2:
                plt.plot(range(data_range[0], data_range[1]), distances[data_range[0]:data_range[1]],
                         label=f"Entity weight = {i / 10}", color=color, linewidth=curve_width)
            else:
                plt.plot(distances, label=f"Entity weight = {i / 10}", color=color, linewidth=curve_width)

        # Create a color bar as a gradient legend
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=mcolors.Normalize(vmin=0, vmax=1))
        cbar = fig.colorbar(sm, ax=ax, ticks=np.linspace(0, 1, 6), orientation='vertical', location='right')
        cbar.set_label("Entity Weight", fontsize=label_fontsize)
        cbar.ax.tick_params(labelsize=tick_fontsize)

        # Add labels, legend, and title
        plt.xlabel("Time Step", fontsize=label_fontsize)
        plt.ylabel("Distance Value", fontsize=label_fontsize)
        plt.xticks(fontsize=tick_fontsize)  # 设置 X 轴刻度字体大小
        plt.yticks(fontsize=tick_fontsize)  # 设置 Y 轴刻度字体大小
        plt.grid(True)

        # Show the plot
        plt.tight_layout()
        plt.show()

    if draw_image_2:
        # Figure 7(b): Correlation between entity weight and the average DNN of crowds. (Correlation coefficient)
        tick_fontsize = 16
        label_fontsize = 22
        text_fontsize = 18
        curve_width = 5
        point_size = 100
        # --------------绘制最近距离平均值和weight的相关性-------------------#
        # 计算 min_distances_array 内每个元素的平均值
        if len(data_range) == 2:
            average_values = [np.min(distances[data_range[0]:data_range[1]]) for distances in average_distances]
        else:
            average_values = [np.min(distances) for distances in average_distances]

        # 创建 weights 数组
        weights = np.linspace(0, 1, 11)

        # 计算相关性系数
        average_corr, _ = pearsonr(average_values, weights)

        # 绘制散点图
        plt.figure(figsize=(8, 6))
        plt.scatter(weights, average_values, color='blue', alpha=0.7, label='Data Points', s=point_size)
        sns.regplot(x=weights, y=average_values, color='blue', scatter=False, line_kws={'linewidth': curve_width})

        print(weights)
        print(average_values)

        # 在图表内部居中显示相关性系数
        plt.text(
            0.5,  # x 坐标
            plt.ylim()[1] - 0.1 * (plt.ylim()[1] - plt.ylim()[0]),  # y 坐标
            f"Correlation Coefficient\nr = {average_corr:.2f}",
            fontsize=text_fontsize,
            color='black',
            ha='center',
            bbox=dict(facecolor='white', alpha=0.5, edgecolor='gray')
        )

        # 图表美化及显示
        plt.xlabel("Entity Weight", fontsize=label_fontsize)
        plt.ylabel("Mean of Min Distances", fontsize=label_fontsize)
        plt.xticks(fontsize=tick_fontsize)  # 设置 X 轴刻度字体大小
        plt.yticks(fontsize=tick_fontsize)  # 设置 Y 轴刻度字体大小
        plt.grid(True)
        plt.tight_layout()
        plt.show()


