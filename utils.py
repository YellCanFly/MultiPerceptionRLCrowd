import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.image as mpimg
from matplotlib.patches import Patch
from IPython import display
from PIL import Image

import time
import os
import numpy as np

import torch.nn as nn
import torch.onnx
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF


def get_files_with_prefix(directory, prefix):
    """
    获取指定目录下以特定前缀开头的文件名列表。

    :param directory: 目标目录路径
    :param prefix: 文件名前缀
    :return: 包含匹配文件名的列表
    """
    files = []
    for filename in os.listdir(directory):
        if filename.startswith(prefix):
            files.append(filename)
    return files


def get_local_time():
    # 获取当前时间的本地时间表示
    local_time = time.localtime()

    # 提取年、月、日、时、分、秒
    current_year = local_time.tm_year
    current_month = local_time.tm_mon
    current_day = local_time.tm_mday
    current_hour = local_time.tm_hour
    current_minute = local_time.tm_min
    current_second = local_time.tm_sec

    return current_year, current_month, current_day, current_hour, current_minute, current_second


def plot(scores):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.plot(scores)
    plt.text(len(scores) - 1, scores[-1], str(scores[-1]))
    plt.show(block=False)
    plt.pause(0.1)


def plot_csv_data(file_path):
    """
    读取 CSV 文件并用 matplotlib 生成曲线图。

    :param file_path: CSV 文件的路径。
    """
    # 使用 numpy 读取 CSV 文件
    data = np.genfromtxt(file_path, delimiter=',', skip_header=1)
    data = data

    # 假设 CSV 文件中第一列是 X 轴数据，第二列是 Y 轴数据
    print(data.shape)
    x = np.arange(len(data))
    y = data

    # 绘制曲线图
    plt.plot(x, y)
    plt.xlabel('X Axis')
    plt.ylabel('Y Axis')
    plt.title('CSV Data Plot')
    plt.show()


def plot_csv_avg_data(file_path, avg_num=50, point_interval=50, show_std=False):
    """
    读取 CSV 文件并用 matplotlib 生成曲线图，可选显示标准差范围。

    :param file_path: CSV 文件的路径。
    :param avg_num: 计算滑动平均值和标准差的窗口大小。
    :param point_interval: 每隔多少个点绘制一个标记点。
    :param show_std: 是否绘制标准差范围，默认为 False。
    """

    # 使用 numpy 读取 CSV 文件
    data = np.genfromtxt(file_path, delimiter=',', skip_header=1)

    # 假设 CSV 文件中第一列是 X 轴数据，第二列是 Y 轴数据
    x = np.arange(len(data))

    # 计算滑动平均值
    avg_data = np.array([np.mean(data[max(0, i - avg_num + 1):i + 1]) for i in range(len(data))])

    # 如果需要显示标准差范围，计算标准差
    if show_std:
        std_data = np.array([np.std(data[max(0, i - avg_num + 1):i + 1]) for i in range(len(data))])

    # 绘制滑动平均值曲线
    plt.plot(x, avg_data, label='Average', color='blue')

    # 绘制标准差范围（误差带），仅当 show_std=True 时
    if show_std:
        plt.fill_between(
            x,
            avg_data - std_data,
            avg_data + std_data,
            color='blue',
            alpha=0.2,
            label='Std Deviation'
        )

    # 添加红色圆点
    for i in range(0, len(avg_data), point_interval):
        plt.scatter(x[i], avg_data[i], color='orange', zorder=5)

    # 标记最大值点
    max_x = np.argmax(avg_data)
    max_avg_data = np.max(avg_data)
    plt.scatter(max_x, max_avg_data, color='red', zorder=5, label='Max Point')

    # 添加标签和标题
    plt.xlabel('X Axis')
    plt.ylabel('Y Axis')
    plt.title('CSV Data Plot with Optional Standard Deviation')
    plt.legend()
    plt.grid(True)

    # 显示图表
    plt.show()


def plot_csvs_avg_data(file_paths, avg_num=50, max_len=1200, line_width=2,
                       x_axis='X Axis', y_axis='Y Axis', title='CSV Data Plot',
                       labels=None, show_std=False, fontsize=14, tick_fontsize=12,
                       x_limits=None, y_limits=None):
    """
    读取多个 CSV 文件并在同一张图上绘制它们的曲线图，每个文件使用不同的颜色，可选显示标准差范围。

    :param file_paths: CSV 文件路径列表。
    :param avg_num: 用于计算滑动平均值的元素数量。
    :param max_len: 限制每条曲线的最大长度。
    :param line_width: 曲线的线宽。
    :param x_axis: X 轴标签。
    :param y_axis: Y 轴标签。
    :param title: 图表标题。
    :param labels: 曲线的标签列表。
    :param show_std: 是否绘制标准差范围，默认为 False。
    :param x_limits: 显式控制 X 轴绘制范围，格式为 (min, max)，默认为 None。
    :param y_limits: 显式控制 Y 轴绘制范围，格式为 (min, max)，默认为 None。
    """
    plt.figure(figsize=(10, 6))  # 设置图像大小

    x_size = 0
    for i, file_path in enumerate(file_paths):
        # 使用 numpy 读取 CSV 文件
        data = np.genfromtxt(file_path, delimiter=',', skip_header=1)
        if data.shape[0] > x_size:
            x_size = data.shape[0]

        # 计算 X 轴坐标
        length = len(data)
        length = np.clip(length, 0, max_len)
        x = np.arange(length)

        # 计算滑动平均值和标准差
        avg_data = np.array([np.mean(data[max(0, i - avg_num + 1):i + 1]) for i in range(length)])
        if show_std:
            std_data = np.array([np.std(data[max(0, i - avg_num + 1):i + 1]) for i in range(length)])

        # 设置曲线标签
        if labels is None:
            label = file_path
        else:
            label = labels[i]

        # 绘制曲线图
        plt.plot(x, avg_data, label=label, linewidth=line_width)

        # 绘制标准差范围，仅当 show_std=True 时
        if show_std:
            plt.fill_between(
                x,
                avg_data - std_data,
                avg_data + std_data,
                alpha=0.2,
                label=f"{label} (Std Dev)"
            )

    # 设置 X 和 Y 轴的范围
    if x_limits is not None:
        plt.xlim(x_limits)
    if y_limits is not None:
        plt.ylim(y_limits)

    # 图表美化
    plt.xlabel(x_axis, fontsize=fontsize)  # 设置 X 轴标签
    plt.ylabel(y_axis, fontsize=fontsize)  # 设置 Y 轴标签
    plt.xticks(fontsize=tick_fontsize)  # 设置 X 轴刻度字体大小
    plt.yticks(fontsize=tick_fontsize)  # 设置 Y 轴刻度字体大小
    if title != '':
        plt.title(title)  # 设置图表标题
    plt.legend(fontsize=fontsize)  # 显示图例
    plt.grid(True)
    plt.tight_layout()

    # 显示图表
    plt.show()


def plot_merged_csv_data(file_paths, avg_num=50, line_width=2,
                         x_axis='X Axis', y_axis='Y Axis', title='Merged CSV Data Plot',
                         labels=None, background_colors=None,
                         use_specified_length=False, common_length=0, specified_lengths=[]):
    """
    读取多个单列 CSV 文件，将数据拼接在一起绘制成一条曲线，并用不同颜色背景区分文件部分。

    :param file_paths: CSV 文件路径列表。
    :param avg_num: 用于计算滑动平均值的元素数量。
    :param line_width: 曲线的宽度。
    :param x_axis: X 轴标签。
    :param y_axis: Y 轴标签。
    :param title: 图表标题。
    :param background_colors: 背景颜色列表。
    :param use_specified_length: 是否使用指定数据长度。
    :param common_length: 指定通用数据长度。
    :param specified_lengths: 指定数据长度数组。
    """
    plt.figure(figsize=(10, 6))  # 设置图像大小

    merged_data = []
    section_lengths = []

    # 读取所有文件数据并拼接
    for idx, file_path in enumerate(file_paths):
        # 使用 numpy 读取 CSV 文件，假设每个文件中只有一列 Y 轴数据
        data = np.genfromtxt(file_path, delimiter=',', skip_header=1)  # 跳过标题行

        if use_specified_length:
            if common_length > 0:
                # 使用通用数据长度
                data = data[:common_length]
            elif idx < len(specified_lengths):
                # 使用指定数据长度数组中的长度
                data = data[:specified_lengths[idx]]

        # 拼接数据
        merged_data.extend(data)
        section_lengths.append(len(data))

    # 转换为 numpy 数组以便于处理
    merged_data = np.array(merged_data)

    # 计算滑动平均值
    avg_data = np.array([np.mean(merged_data[max(0, i - avg_num + 1):i + 1]) for i in range(len(merged_data))])

    # 绘制曲线
    x = np.arange(len(avg_data))  # 生成 X 轴坐标
    plt.plot(x, avg_data, label='Merged Data', linewidth=line_width)

    # 添加背景颜色区分不同文件部分
    if background_colors is None:
        background_colors = ['#FFDDC1', '#D2F6C5', '#C1E1C5', '#B5B8D3']  # 默认颜色

    legend_patches = []  # 用于存储图例的背景色块
    current_start = 0
    for i, section_length in enumerate(section_lengths):
        color = background_colors[i % len(background_colors)]  # 循环使用颜色
        plt.axvspan(current_start, current_start + section_length, color=color, alpha=0.3)
        if labels is not None and i < len(labels):
            # 为每个背景块添加图例条目
            legend_patches.append(Patch(facecolor=color, alpha=0.3, label=labels[i]))
        current_start += section_length

    # 添加背景色块图例
    if legend_patches:
        plt.legend(handles=legend_patches)

    plt.xlabel(x_axis)  # 设置 X 轴标签
    plt.ylabel(y_axis)  # 设置 Y 轴标签
    plt.title(title)  # 设置图表标题
    plt.show()


def init_net(net, f=None):
    # 初始化网络权重
    if f is None:
        f_net = 1. / np.sqrt(net.weight.data.size()[0])
    else:
        f_net = f
    nn.init.uniform_(net.weight.data, -f_net, f_net)
    nn.init.uniform_(net.bias.data, -f_net, f_net)


def Convert_ONNX(model, ray_vec_dims, agent_vec_dims, weight_vec_dims, state_img_dims, file_name='actor.onnx'):
    """
    Convert the ActorNetwork model to ONNX format.

    :param model: The trained PyTorch model (ActorNetwork).
    :param ray_vec_dims: Integer, the dimension of ray state vector.
    :param agent_vec_dims: Integer, the dimension of agent state vector.
    :param weight_vec_dims: Integer, the dimension of weight state vector.
    :param state_img_dims: Tuple, the dimensions of the state image (height, width).
    :param file_name: String, the name of the output ONNX file.
    """

    # Set model to evaluation mode
    model.eval()

    # Create dummy input tensors for testing
    batch_size = 1  # ONNX supports dynamic batch size, but we provide a sample input
    state_vec_dummy = torch.randn(batch_size, ray_vec_dims + weight_vec_dims + agent_vec_dims).to(model.device)
    state_img_dummy = torch.randn(batch_size, 1, state_img_dims[0], state_img_dims[1]).to(model.device)  # 1 channel image

    # Define dynamic axes for batch size flexibility
    dynamic_axes = {
        'state_vec': {0: 'batch_size'},  # Allow batch_size to be variable
        'state_img': {0: 'batch_size'},
        'output_mean': {0: 'batch_size'}
    }

    # Export the model to ONNX
    torch.onnx.export(
        model,  # The model being converted
        (state_vec_dummy, state_img_dummy),  # Input sample for ONNX tracing
        file_name,  # Output ONNX file name
        export_params=True,  # Store the trained parameter weights inside the model file
        opset_version=11,  # ONNX version (>= 11 recommended for better compatibility)
        do_constant_folding=True,  # Optimization: fold constant expressions
        input_names=['state_vec', 'state_img'],  # Input names
        output_names=['output_mean'],  # Output name (mean action from the distribution)
        dynamic_axes=dynamic_axes  # Dynamic batch size support
    )

    print(f"\nModel has been successfully converted to ONNX format and saved as {file_name}")


def image_to_tensor(image):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # transforms.ToTensor() 将 PIL 图像或 ndarray 转换为 PyTorch 张量
    transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    tensor = transform(image).to(device)
    return tensor


def crop_and_rotate_tensor(tensor, rect_size, center, angle, new_size=(32, 32)):
    """
    Crop and rotate a region from a tensor based on rect_size, center, and angle.

    :param tensor: PyTorch tensor of the grayscale image
    :param rect_size: tuple (width, height) of the rectangle
    :param center: tuple (x, y) representing the center of the rectangle, values are in [0, 1] range
    :param angle: rotation angle in degrees
    :param new_size: tuple (new_width, new_height) of the new size after cropping and rotating
    :return: PyTorch tensor of the cropped and rotated rectangle
    """
    # Convert center and size to actual pixel values
    original_size = tensor.squeeze().shape
    center = (center[0] * original_size[0], center[1] * original_size[1])
    rect_size = (int(rect_size[0] * original_size[0]), int(rect_size[1] * original_size[1]))

    # Rotate the image
    rotated_tensor = TF.rotate(tensor, angle, center=center, expand=True)

    # Calculate top-left corner of the cropping box
    left = center[0] - rect_size[0] / 2
    top = center[1] - rect_size[1] / 2

    # Crop the image
    cropped_rotated_tensor = TF.crop(rotated_tensor, int(top), int(left), rect_size[1], rect_size[0])

    # Resize the cropped and rotated tensor to new size
    resized_tensor = TF.resize(cropped_rotated_tensor, new_size)

    return resized_tensor


def batch_crop_and_rotate_tensor_parallel(tensor, rect_size, centers, angles, new_size=(32, 32)):
    """
    Apply multiple crop and rotate operations on a single grayscale image tensor in parallel,
    where rect_size represents the proportion of the original image size to be cropped.

    :param tensor: PyTorch tensor of the grayscale image [1, H, W]
    :param rect_size: tuple (width, height) of the rectangle, values are relative to image size (0 to 1)
    :param centers: numpy array of shape [N, 2], each row is a tuple (x, y) representing the center of the rectangle
    :param angles: numpy array of angles, shape [N], rotation angle in rad for each operation
    :param new_size: tuple (new_width, new_height) of the new size after cropping and rotating
    :return: PyTorch tensor of cropped and rotated rectangles, shape [N, 1, new_height, new_width]
    """
    N = len(angles)
    C, H, W = tensor.shape
    tensor = tensor.unsqueeze(0)  # Add a batch dimension [1, C, H, W]
    # print("batch crop num = ", N)

    # Convert angles to radians and centers to pixel units
    angles_rad = -(torch.tensor(angles) + 0.5 * np.pi)  # Negative for correct direction
    centers_abs = torch.tensor(centers) * torch.tensor([W, H]).unsqueeze(0)  # [N,2]

    # Calculate scale based on the rect_size relative to the new_size
    scale = torch.tensor(rect_size).unsqueeze(0)

    # Calculate affine matrices for each transformation
    affine_matrices = torch.zeros(N, 2, 3)
    affine_matrices[:, 0, 0] = torch.cos(angles_rad) * scale[:, 0]
    affine_matrices[:, 0, 1] = torch.sin(angles_rad) * scale[:, 1]
    affine_matrices[:, 1, 0] = -torch.sin(angles_rad) * scale[:, 0]
    affine_matrices[:, 1, 1] = torch.cos(angles_rad) * scale[:, 1]

    # Adjust centers to be in [-1, 1] range for affine_grid
    centers_normalized = 2.0 * centers_abs / torch.tensor([W, H]).unsqueeze(0) - 1
    affine_matrices[:, :, 2] = centers_normalized

    affine_matrices = affine_matrices.to(tensor.device)

    # Create affine grids and sample
    grids = F.affine_grid(affine_matrices, [N, C, *new_size], align_corners=True)
    result_tensor = F.grid_sample(tensor.repeat(N, 1, 1, 1), grids, align_corners=True)

    return result_tensor


def visualize_tensors(tensor_batch):
    """
    Visualize a batch of tensors.

    :param tensor_batch: PyTorch tensor of shape [N, C, H, W]
    """
    # Check if the tensor batch contains a single image
    if tensor_batch.dim() == 3:
        tensor_batch = tensor_batch.unsqueeze(0)  # Add the batch dimension

    # Set the figure size
    plt.figure(figsize=(20, 10))

    # Number of images
    N = tensor_batch.size(0)

    # Iterate through the tensor batch and plot each image
    for i in range(N):
        ax = plt.subplot(1, N, i + 1)
        img = tensor_batch[i]  # Get the i-th tensor in the batch

        # Check for grayscale images and adjust accordingly
        if img.size(0) == 1:
            img = img.squeeze(0)  # Remove the channel dimension for grayscale images

        img = img.detach().cpu().numpy()  # Convert to numpy array

        # Display the image
        ax.imshow(img, cmap='gray')
        ax.axis('off')  # Turn off axis numbers and ticks

    plt.show()


def calculate_average_nearest_neighbor_distance(positions_array):
    """
    Calculate the average distance from each agent to its nearest neighbor for each frame,
    considering only agents whose positions are not inf.

    :param positions_array: NumPy array of shape (num_frames, num_agents, 2)
    :return: List of average distances to the nearest neighbor for each frame
    """
    num_frames, num_agents, _ = positions_array.shape
    average_distances = []

    for frame_idx in range(num_frames):
        positions = positions_array[frame_idx]
        distances = []

        for i in range(num_agents):
            if np.isinf(positions[i]).any():
                distances.append(float('inf'))
                continue

            min_distance = float('inf')
            for j in range(num_agents):
                if i != j and not np.isinf(positions[j]).any():
                    dist = np.linalg.norm(positions[i] - positions[j])
                    if dist < min_distance:
                        min_distance = dist
            distances.append(min_distance)

        valid_distances = [d for d in distances if not np.isinf(d)]
        average_distance = np.mean(valid_distances) if valid_distances else float('inf')
        average_distances.append(average_distance)

    return average_distances


def visualize_trajectories_polar(positions, interval=1, offset=(0, 0)):
    """
    Visualize trajectories using positions array.

    :param positions: NumPy array of shape (num_frames, num_agents, 2)
    :param interval: Interval between scatter points
    :param offset: Tuple (x_offset, y_offset) for position offset
    """
    num_frames, num_agents, _ = positions.shape
    x_offset, y_offset = offset

    # Prepare the figure
    fig = plt.figure()
    ax = fig.add_subplot(111, polar=True)

    # Iterate over each agent
    for agent_idx in range(num_agents):
        theta = []
        r = []

        # Collect the trajectory of the current agent across all frames
        for frame_index in range(0, num_frames, interval):
            x, y = positions[frame_index, agent_idx]
            x -= x_offset
            y -= y_offset
            theta.append(np.arctan2(y, x))
            r.append(np.sqrt(x ** 2 + y ** 2))

        # Plot the trajectory with a color gradient
        points = np.array([theta, r]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        norm = plt.Normalize(0, len(segments))
        lc = cm.ScalarMappable(norm=norm, cmap='viridis').to_rgba(range(len(segments)))

        for i in range(len(segments)):
            ax.plot(segments[i][:, 0], segments[i][:, 1], color=lc[i])

        # Scatter plot the positions with a color gradient
        sc = ax.scatter(theta, r, c=range(len(r)), cmap='viridis', s=2)

    plt.colorbar(sc, ax=ax, orientation='horizontal')
    plt.show()


def visualize_trajectories_cartesian0(positions, interval=1, offset=(0, 0)):
    """
    Visualize trajectories using positions array in Cartesian coordinates.

    :param positions: NumPy array of shape (num_frames, num_agents, 2)
    :param interval: Interval between scatter points
    :param offset: Tuple (x_offset, y_offset) for position offset
    """
    num_frames, num_agents, _ = positions.shape
    x_offset, y_offset = offset

    # Prepare the figure
    fig, ax = plt.subplots()

    # Iterate over each agent
    for agent_idx in range(num_agents):
        x_vals = []
        y_vals = []

        # Collect the trajectory of the current agent across all frames
        for frame_index in range(0, num_frames, interval):
            x, y = positions[frame_index, agent_idx]
            x -= x_offset
            y -= y_offset
            x_vals.append(x)
            y_vals.append(y)

        # Plot the trajectory with a color gradient
        points = np.array([x_vals, y_vals]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        norm = plt.Normalize(0, len(segments))
        lc = cm.ScalarMappable(norm=norm, cmap='viridis').to_rgba(range(len(segments)))

        for i in range(len(segments)):
            ax.plot(segments[i][:, 0], segments[i][:, 1], color=lc[i])

        # Scatter plot the positions with a color gradient
        sc = ax.scatter(x_vals, y_vals, c=range(len(x_vals)), cmap='viridis', s=2)

    plt.colorbar(sc, ax=ax, orientation='horizontal')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Trajectories in Cartesian Coordinates')
    plt.show()


def visualize_trajectories_cartesian(positions, interval=1, offset=(0, 0), background_image_path=None, xlim=None,
                                     ylim=None):
    """
    Visualize trajectories using positions array in Cartesian coordinates.

    :param positions: NumPy array of shape (num_frames, num_agents, 2)
    :param interval: Interval between scatter points
    :param offset: Tuple (x_offset, y_offset) for position offset
    :param background_image_path: Path to the background image file
    """
    num_frames, num_agents, _ = positions.shape
    x_offset, y_offset = offset

    # Prepare the figure
    fig, ax = plt.subplots()

    # If a background image path is provided, set it as the background
    if background_image_path:
        img = mpimg.imread(background_image_path)
        ax.imshow(img, extent=[ylim[0], ylim[1], xlim[0], xlim[1]], alpha=1.0)

    # Iterate over each agent
    for agent_idx in range(num_agents):
        x_vals = []
        y_vals = []

        # Collect the trajectory of the current agent across all frames
        for frame_index in range(0, num_frames, interval):
            x, y = positions[frame_index, agent_idx]
            if np.isinf(x) or np.isinf(y):
                continue
            x -= x_offset
            y -= y_offset
            x_vals.append(x)
            y_vals.append(y)

        if len(x_vals) > 1:
            # Plot the trajectory with a color gradient
            points = np.array([x_vals, y_vals]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            norm = plt.Normalize(0, len(segments))
            lc = cm.ScalarMappable(norm=norm, cmap='viridis').to_rgba(range(len(segments)))

            for i in range(len(segments)):
                ax.plot(segments[i][:, 0], segments[i][:, 1], color=lc[i])

            # Scatter plot the positions with a color gradient
            sc = ax.scatter(x_vals, y_vals, c=range(len(x_vals)), cmap='viridis', s=2)

    # plt.colorbar(sc, ax=ax, orientation='horizontal')
    # ax.set_aspect('equal', 'box')  # Ensure the x and y axes have the same scale

    if xlim is not None and ylim is not None:
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Trajectories in Cartesian Coordinates')
    plt.show()


def get_all_subdirectories(directory):
    # Get all subdirectories in the given directory
    subdirectories = [
        folder for folder in os.listdir(directory) if os.path.isdir(os.path.join(directory, folder))
    ]

    return subdirectories


if __name__ == '__main__':
    pass
    # Figure 11(a): Performance comparison of with/without curriculum-based Env-templates.
    file_paths = [
        'checkpoint_ppo/2025_1_13_20_20_6/update_agent_score_2025_1_13_20_20_6.csv',
        'checkpoint_ppo/2025_1_16_11_21_54/update_agent_score_2025_1_16_11_21_54.csv',
    ]
    plot_csvs_avg_data(file_paths, avg_num=300, max_len=2500, x_axis='Episodes', y_axis='Reward',
                       line_width=6, title='', fontsize=24, tick_fontsize=18,
                       labels=['Reward with Env-Templates', 'Reward without Env-Templates'])

    # Figure 13: Reward function comparison (W/Wo Modular 2-Phase Training)
    file_paths = [
        'checkpoint_ppo/Reward_W_WO_Modular_2-Phase_Learning/update_agent_score_2024_12_23_23_4_40.csv',
        'checkpoint_ppo/Reward_W_WO_Modular_2-Phase_Learning/update_agent_score_2025_1_14_18_36_25.csv',
    ]

    plot_csvs_avg_data(file_paths, avg_num=500, x_axis='Episodes', y_axis='Average Reward (500 episodes)',
                       max_len=3000,
                       title='',
                       line_width=5,
                       y_limits=[-20000, 0],
                       labels=['With Modular 2-Phase Learning', 'Without Modular 2-Phase Learning'],
                       fontsize=18, tick_fontsize=14)



    # ------------------- Example Usage -------------------
    # file_paths = [
    #     'checkpoint_ppo/2024_12_16_16_32_37/update_agent_score_2024_12_16_16_32_37.csv',
    #     'checkpoint_ppo/2025_1_13_20_20_6/update_agent_score_2025_1_13_20_20_6.csv',
    #     'checkpoint_ppo/2025_1_14_10_19_55/update_agent_score_2025_1_14_10_19_55.csv',
    #     ]
    # plot_csvs_avg_data(file_paths, avg_num=1000, max_len=7000, x_axis='Episodes', y_axis='Average Score',
    #                    title='Comparison of Scores in Ablation Experiments')

    # file_paths = [
    #     'checkpoint_ppo/2024_12_16_11_0_34/update_agent_score_2024_12_16_11_0_34.csv',
    #     'checkpoint_ppo/2024_12_16_16_32_37/update_agent_score_2024_12_16_16_32_37.csv',
    #     ]
    # plot_merged_csv_data(file_paths, avg_num=100, line_width=2, x_axis='Episodes', y_axis='Average Score')

    # 测试image_to_tensor
    # terrain_filename = "route_1.png"
    # terrain_filename = "turning_1.png"
    # terrain_image_path = os.path.join('terrain_maps', terrain_filename)
    # terrain_greyscale = Image.open(terrain_image_path).convert('L')
    # terrain_tensor = image_to_tensor(terrain_greyscale)
    # print(terrain_tensor.shape)
    #
    # centers = np.array([[0.5, 0.5], [0.0, 0.0]])
    # angles = np.array([0, np.pi*0.25])
    # ans = batch_crop_and_rotate_tensor_parallel(
    #     terrain_tensor,  # 环境图像Tensor
    #     [0.375, 0.375],  # 智能体感知范围
    #     centers,  # 智能体位置
    #     angles,  # 智能体朝向
    #     new_size=(32, 32)  # 缩放统一大小
    # )
    # print(ans.shape)
    #
    # visualize_tensors(ans)
