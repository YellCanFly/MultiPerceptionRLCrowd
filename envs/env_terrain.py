import os.path
import gym
import torch
from gym import spaces
import numpy as np
import time
import random
from PIL import Image
import json
import subprocess
from server import start_env_server, receive_all, send_data
from config import parse_args
from utils import get_files_with_prefix, image_to_tensor, batch_crop_and_rotate_tensor_parallel


class MultiAgentEnv(gym.Env):
    """多Agent避障训练环境"""

    def __init__(self, num_agents,
                 agent_radius=0.5,
                 num_rays=20,
                 max_ray_distance=20.0,
                 ray_vision_lamda=0.2,
                 max_speed_orientation=1.6,
                 max_speed_perpendicular=0.5,
                 max_acceleration=1.0,
                 max_angular_velocity=0.5 * np.pi,
                 agent_terrain_obs_range=[10.0, 10.0],
                 dt=0.04,
                 action_interval_step_num=1,
                 width=20.0,
                 height=20.0,
                 min_walking_distance=5.0,
                 max_alive_time=50.0,
                 max_time_overlapped_with_obstacle=2.0,
                 num_new_agents_per_episode=50,
                 env_title='',
                 auto_launch_ue=True,
                 args=None,
                 num_user_agents_chase=0,
                 num_user_agents_random=0,
                 num_obstacles=0,
                 terrain_filename='',
                 use_terrain=True,):
        super(MultiAgentEnv, self).__init__()

        # 初始化随机种子
        seed = 666
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        # 智能体参数
        self.num_agents = num_agents
        self.agent_radius = agent_radius
        self.num_rays = num_rays
        self.max_ray_distance = max_ray_distance
        self.ray_vision_lamda = ray_vision_lamda
        self.max_speed_orientation = max_speed_orientation
        self.max_speed_perpendicular = max_speed_perpendicular
        self.max_acceleration = max_acceleration
        self.max_angular_velocity = max_angular_velocity

        # 环境基础参数
        self.dt = dt
        self.action_interval_step_num = action_interval_step_num
        self.width = width
        self.height = height
        self.env_size = [width, height]
        self.terrain_texture_index = 0

        # Terrain特征相关参数
        self.use_terrain = use_terrain
        self.terrain_tensor = torch.ones([1, 256, 256])
        self.terrain_obs_range = agent_terrain_obs_range  # 观察范围，单位为m

        # Terrain图像路径相关
        terrains_folder_path = 'terrain_maps'
        while_terrains = get_files_with_prefix(terrains_folder_path, 'white')
        single_terrains = get_files_with_prefix(terrains_folder_path, 'single')
        turning_terrains = get_files_with_prefix(terrains_folder_path, 'turning')
        cross_terrains = get_files_with_prefix(terrains_folder_path, 'cross')
        route_terrains = get_files_with_prefix(terrains_folder_path, 'route')
        hallway_terrains = get_files_with_prefix(terrains_folder_path, 'hallway')
        all_terrains = while_terrains + turning_terrains + single_terrains + cross_terrains + route_terrains + hallway_terrains
        self.terrain_type_dict = {'white': while_terrains,
                                  'turning': turning_terrains,
                                  'single': single_terrains,
                                  'cross': cross_terrains,
                                  'route': route_terrains,
                                  'mix': while_terrains + turning_terrains + single_terrains + cross_terrains + route_terrains,
                                  'hallway': hallway_terrains}
        print(self.terrain_type_dict)

        # 预处理所有的Terrain图像数据
        self.terrain_tensors = {}
        for terrain_name in all_terrains:
            terrain_image_path = os.path.join('terrain_maps', terrain_name)
            terrain_greyscale = Image.open(terrain_image_path).convert('L')
            self.terrain_tensors[terrain_name.split('.')[0]] = image_to_tensor(terrain_greyscale)
        print(self.terrain_tensors.keys())

        # 配置参数
        self.target_loc_margin = args.target_loc_margin
        self.min_walking_distance = min_walking_distance
        self.max_alive_time = max_alive_time
        self.arrive_range_scale = args.arrive_range_scale
        self.max_time_overlapped_with_obstacle = max_time_overlapped_with_obstacle
        self.agent_terrain_threshold = args.agent_terrain_threshold
        self.implicit_agent_spawn_threshold = args.implicit_agent_spawn_threshold
        self.implicit_agent_target_threshold = args.implicit_agent_target_threshold
        self.use_nav_point = args.use_nav_point
        self.update_nav_point_threshold = args.update_nav_point_threshold
        self.num_new_agents_per_episode = num_new_agents_per_episode  # 一次episode内部，补充的新Agent初始数量

        # 更新参数
        self.all_agent_done = False
        self.num_left_new_agent = self.num_new_agents_per_episode
        self.before_update_agent_indexes = [i for i in range(self.num_agents)]
        self.exist_agent_indexes = [i for i in range(self.num_agents)]
        self.before_update_agent_locs = []
        self.exist_agent_locs = []

        # 环境标题，用于UE端展示
        self.env_title = env_title
        if args is None:
            args = parse_args()
        self.args = args

        # 用于模拟真实用户行为为的Agent配置
        self.num_user_agents_chase = num_user_agents_chase
        self.num_user_agents_random = num_user_agents_random
        self.user_agent_max_speed_orientation = max_speed_orientation
        self.user_agent_max_speed_perpendicular = max_speed_perpendicular
        self.user_agent_max_acceleration = max_acceleration
        self.user_agent_max_angular_velocity = max_angular_velocity

        # 障碍物数量配置
        self.num_obstacles = num_obstacles

        # 定义观察空间和行动空间
        self.obs_dim = num_rays * 3 + 9  # 射线数量*3 + 权重（2） + 朝向(1) + 相对速度(2) + 目标相对位置(2) + 导航点相对位置(2)
        self.vector_observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(num_agents, self.obs_dim),
                                                   dtype=float)
        self.terrain_obs_dim = args.terrain_obs_dim  # 环境信息观察空间
        self.image_observation_space = spaces.Box(low=0.0, high=1.0, shape=(num_agents, *self.terrain_obs_dim),
                                                  dtype=np.float32)
        # Combine the observation spaces
        self.observation_space = spaces.Tuple((self.vector_observation_space, self.image_observation_space))

        self.action_dims = 3  # x方向加速度 + y方向加速度 + 角速度
        self.action_space = spaces.Box(
            low=np.array([-max_acceleration, -max_acceleration, -max_angular_velocity] * num_agents).
            reshape(-1, self.action_dims),
            high=np.array([max_acceleration, max_acceleration, max_angular_velocity] * num_agents).
            reshape(-1, self.action_dims),
            shape=(num_agents, self.action_dims),
            dtype=float)

        # 初始化环境状态
        self.state = None

        # 根据配置创建服务器socket
        self.server_socket = None
        self.client_socket = None
        self.client_addr = None
        if auto_launch_ue:
            self.launch_ue()
        self.server_socket = start_env_server(host=self.args.ip, port=self.args.port)
        self.client_socket, self.client_addr = self.server_socket.accept()
        print(f"Connection from {self.client_addr}")

    def step(self, actions):
        time_start = time.time()
        # 构建actions数据协议
        actions_data = []
        for action in actions:
            action = np.array(action, dtype=float)
            action = action * np.array([self.max_acceleration, self.max_acceleration, self.max_angular_velocity])
            actions_data.append({'floats': list(action)})

        # 发送actions给UE
        send_data_dict = {'type': 'action', 'actions': actions_data}
        send_data(self.client_socket, send_data_dict)

        # 接收UE返回的数据，并解析为numpy
        recv_data = receive_all(self.client_socket, buffer_size=self.args.data_buffer_size)
        recv_data_str = recv_data.decode('utf-8')
        recv_data_str_json = {}
        try:
            recv_data_str_json = json.loads(recv_data_str)
        except json.JSONDecodeError as e:
            print("JSON 解析错误:", e)
            print("原始数据长度:", len(recv_data_str))

        self.before_update_agent_locs = [[item['x'], item['y']]
                                         for item in recv_data_str_json['before_update_agent_locs']]
        self.exist_agent_locs = [[item['x'], item['y']]
                                 for item in recv_data_str_json['exist_agent_locs']]

        # 获取所有Agent的观察
        vector_obs_list = [item['floats'] for item in recv_data_str_json['obs']]
        vector_obs = np.array(vector_obs_list, dtype=float)
        if self.use_terrain:
            img_obs = batch_crop_and_rotate_tensor_parallel(
                self.terrain_tensor,  # 环境图像Tensor
                np.array(self.terrain_obs_range) / np.array(self.env_size),  # 智能体感知范围
                np.array(self.before_update_agent_locs) / np.array(self.env_size),  # 智能体位置
                vector_obs[:, self.num_rays * 3 + 2],  # 智能体朝向
                new_size=self.terrain_obs_dim  # 缩放统一大小
            ).cpu()
        else:
            img_obs = None

        # 获取所有Agent的reward
        rewards_list = [item for item in recv_data_str_json['rewards']]
        rewards = np.array(rewards_list, dtype=float)

        # 检查所有Agent是否达到结束条件
        dones_list = [item for item in recv_data_str_json['dones']]
        dones = np.array(dones_list, dtype=bool)

        # 获取下一迭代使用的state
        next_state_list = [item['floats'] for item in recv_data_str_json['next_state']]
        next_state = np.array(next_state_list, dtype=float)
        if not self.exist_agent_locs or self.use_terrain is False:
            next_img_obs = None
        else:
            next_img_obs = batch_crop_and_rotate_tensor_parallel(
                self.terrain_tensor,  # 环境图像Tensor
                np.array(self.terrain_obs_range) / np.array(self.env_size),  # 智能体感知范围
                np.array(self.exist_agent_locs) / np.array(self.env_size),  # 智能体位置
                next_state[:, self.num_rays * 3 + 2],  # 智能体朝向
                new_size=self.terrain_obs_dim  # 缩放统一大小
            ).cpu()

        # 把UE中的all_agent_done、left_new_agent传回并更新
        self.all_agent_done = recv_data_str_json['all_agent_done']
        self.num_left_new_agent = recv_data_str_json['num_left_new_agent']
        self.before_update_agent_indexes = [item for item in recv_data_str_json['before_update_agent_indexes']]
        self.exist_agent_indexes = [item for item in recv_data_str_json['exist_agent_indexes']]

        spent_time = time.time() - time_start

        # 附加信息
        infos = {'time': spent_time, 'next_state': (next_state, next_img_obs)}

        return (vector_obs, img_obs), rewards, dones, infos

    def reset(self, width=20.0, height=20.0,
              env_type='train',
              terrain_type='white',
              terrain_specific_name=None,
              terrain_prev_set=None,
              reward_weight_collision_range=None,
              reward_weight_terrain_range=None):

        self.width = width
        self.height = height
        self.all_agent_done = False
        if terrain_specific_name is None:
            if terrain_prev_set is not None and np.random.rand() < self.args.terrain_prev_prob:
                terrain_type = random.choice(terrain_prev_set)
            terrain_name = random.choice(self.terrain_type_dict[terrain_type]).split('.')[0]
        else:
            terrain_name = terrain_specific_name
        self.terrain_tensor = self.terrain_tensors[terrain_name]
        print('-------terrain info-------')
        print('terrain texture name = ' + terrain_name)

        # Set default reward weight range
        if reward_weight_collision_range is None:
            reward_weight_collision_range = [1.0, 1.0]
        if reward_weight_terrain_range is None:
            reward_weight_terrain_range = [1.0, 1.0]

        env_info = {
            'env_type': env_type,
            'env_title': self.env_title,
            'width': self.width,
            'height': self.height,
            'implicit_env_texture_name': terrain_name,
            # 'terrain_texture_name': terrain_name,

            'num_agents': self.num_agents,
            'agent_radius': self.agent_radius,
            'num_rays': self.num_rays,
            'target_loc_margin': self.target_loc_margin,
            'max_ray_distance': self.max_ray_distance,
            'ray_vision_lamda': self.ray_vision_lamda,
            'max_speed_orientation': self.max_speed_orientation,
            'max_speed_perpendicular': self.max_speed_perpendicular,
            'max_acceleration': self.max_acceleration,
            'max_angular_velocity': self.max_angular_velocity,
            'dt': self.dt,
            'action_interval_step_num': self.action_interval_step_num,
            'min_walking_distance': self.min_walking_distance,
            'max_alive_time': self.max_alive_time,
            'arrive_range_scale': self.arrive_range_scale,
            'max_time_overlapped_with_obstacle': self.max_time_overlapped_with_obstacle,
            'implicit_env_obs_range': {'x':self.terrain_obs_range[0], 'y':self.terrain_obs_range[1]},
            "implicit_env_value_threshold": self.agent_terrain_threshold,
            "implicit_agent_spawn_threshold": self.implicit_agent_spawn_threshold,
            "implicit_agent_target_threshold": self.implicit_agent_target_threshold,
            # "agent_terrain_threshold": self.agent_terrain_threshold,
            "use_nav_point": self.use_nav_point,
            "update_nav_point_threshold": self.update_nav_point_threshold,
            'num_new_agents_per_episode': self.num_new_agents_per_episode,

            'num_user_agents_chase': self.num_user_agents_chase,
            'num_user_agents_random': self.num_user_agents_random,
            'user_agent_max_speed_orientation': self.user_agent_max_speed_orientation,
            'user_agent_max_speed_perpendicular': self.user_agent_max_speed_perpendicular,
            'user_agent_max_acceleration': self.user_agent_max_acceleration,
            'user_agent_max_angular_velocity': self.user_agent_max_angular_velocity,

            'num_obstacles': self.num_obstacles
        }
        reward_info = {
            'alive_penalty': self.args.alive_penalty,
            'velocity_smooth_penalty': self.args.velocity_smooth_penalty,
            'collision_penalty': self.args.collision_penalty,
            'target_approach_distance_reward': self.args.target_approach_distance_reward,
            'target_away_distance_penalty_ratio': self.args.target_away_distance_penalty_ratio,
            'nav_point_approach_distance_reward': self.args.nav_point_approach_distance_reward,
            'target_arrive_reward': self.args.target_arrive_reward,
            'obstacle_penalty': self.args.obstacle_penalty,
            'obstacle_time_coef': self.args.obstacle_time_coef,
            'implicit_env_penalty': self.args.terrain_penalty,

            'reward_weight_entity_low': reward_weight_collision_range[0],
            'reward_weight_entity_up': reward_weight_collision_range[1],
            'reward_weight_implicit_low': reward_weight_terrain_range[0],
            'reward_weight_implicit_up': reward_weight_terrain_range[1],
            # 'reward_weight_collision_low': reward_weight_collision_range[0],
            # 'reward_weight_collision_up': reward_weight_collision_range[1],
            # 'reward_weight_terrain_low': reward_weight_terrain_range[0],
            # 'reward_weight_terrain_up': reward_weight_terrain_range[1]
        }
        send_data_dict = {'type': 'reset', 'env_info': env_info, 'reward_info': reward_info}
        send_data(self.client_socket, send_data_dict)
        # self.client_socket.sendall(json.dumps(data).encode('utf-8'))

        # 接收UE返回的数据，并解析为numpy
        recv_data = receive_all(self.client_socket, buffer_size=self.args.data_buffer_size)
        recv_data_str = recv_data.decode('utf-8')
        recv_data_json = json.loads(recv_data_str)

        # 获取所有Agent的观察
        vector_obs_list = [item['floats'] for item in recv_data_json['obs']]
        vector_obs = np.array(vector_obs_list, dtype=float)
        self.exist_agent_locs = [[item['x'], item['y']]
                                 for item in recv_data_json['exist_agent_locs']]
        if self.use_terrain:
            img_obs = batch_crop_and_rotate_tensor_parallel(
                self.terrain_tensor,  # 环境图像Tensor
                np.array(self.terrain_obs_range) / np.array(self.env_size),  # 智能体感知范围
                np.array(self.exist_agent_locs) / np.array(self.env_size),  # 智能体位置
                vector_obs[:, self.num_rays * 3 + 2],  # 智能体朝向
                new_size=self.terrain_obs_dim  # 缩放统一大小
            ).cpu()
        else:
            img_obs = None

        self.all_agent_done = False
        self.num_left_new_agent = self.num_new_agents_per_episode
        self.before_update_agent_indexes = [i for i in range(self.num_agents)]
        self.exist_agent_indexes = [i for i in range(self.num_agents)]

        return vector_obs, img_obs

    def render(self, mode='human', step=None):
        pass

    def launch_ue(self):
        # 定义命令和参数
        command = self.args.ue_command
        uproject_path = self.args.uproject_path
        map_path = self.args.map_path

        # 完整命令
        full_command = [command, uproject_path, map_path, '-game', '-log',
                        '-resx=1280', '-resy=720',
                        '-ip=%s' % self.args.ip, '-port=%d' % self.args.port]

        # 启动程序
        process = subprocess.Popen(full_command)


if __name__ == '__main__':
    args = parse_args()
    env = MultiAgentEnv(
        num_agents=20,
        agent_radius=0.5,
        num_rays=20,
        max_ray_distance=10.0,
        ray_vision_lamda=0.2,
        max_speed_orientation=args.max_speed_orientation,
        max_speed_perpendicular=args.max_speed_perpendicular,
        max_acceleration=args.max_acceleration,
        max_angular_velocity=args.max_angular_velocity,
        dt=args.dt,
        width=args.start_env_size[0],
        height=args.start_env_size[1],
        min_walking_distance=args.min_walking_distance,
        max_alive_time=args.max_alive_time,
        max_time_overlapped_with_obstacle=args.max_time_overlapped_with_obstacle,
        num_new_agents_per_episode=args.num_new_agents_per_episode,
        env_title='%s-%s-%s' % (args.model_name, args.train_strategy, 'loc_time_str'),
        auto_launch_ue=False,
        args=args,
        num_user_agents_chase=args.num_user_agents_chase,
        num_user_agents_random=args.num_user_agents_random,
        num_obstacles=args.num_obstacles,
        terrain_filename=args.terrain_filename
    )
