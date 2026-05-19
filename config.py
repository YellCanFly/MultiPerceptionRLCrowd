import argparse
import ast
import torch
import os
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description='RLInteractiveCrowd')
    # Agent config
    parser.add_argument('--num_agents', type=int, default=20)
    parser.add_argument('--agent_radius', type=float, default=0.5)
    parser.add_argument('--num_rays', type=int, default=50)
    parser.add_argument('--max_ray_distance', type=float, default=10.0)
    parser.add_argument('--ray_vision_lamda', type=float, default=0.2)
    parser.add_argument('--max_speed_orientation', type=float, default=1.6)
    parser.add_argument('--max_speed_perpendicular', type=float, default=0.5)
    parser.add_argument('--max_acceleration', type=float, default=2.0)
    parser.add_argument('--max_angular_velocity', type=float, default=0.25 * np.pi)
    parser.add_argument('--agent_terrain_obs_range', type=ast.literal_eval, default=[15.0, 15.0])

    # Env config
    parser.add_argument('--start_env_size', type=ast.literal_eval, default=[40.0, 40.0])
    parser.add_argument('--end_env_size', type=ast.literal_eval, default=[40.0, 40.0])
    parser.add_argument('--env_scale_frequency', type=int, default=20)
    parser.add_argument('--dt', type=float, default=0.04)
    parser.add_argument('--action_interval_step_num', type=int, default=1)
    parser.add_argument('--min_walking_distance', type=float, default=10.0)
    parser.add_argument('--max_alive_time', type=float, default=100.0)
    parser.add_argument('--arrive_range_scale', type=float, default=4.0)
    parser.add_argument('--target_loc_margin', type=float, default=3.0)
    parser.add_argument('--max_time_overlapped_with_obstacle', type=float, default=2.0)
    parser.add_argument('--agent_terrain_threshold', type=float, default=0.9)
    parser.add_argument('--implicit_agent_spawn_threshold', type=float, default=0.0)
    parser.add_argument('--implicit_agent_target_threshold', type=float, default=0.9)
    parser.add_argument('--update_nav_point_threshold', type=float, default=5.0)
    parser.add_argument('--auto_launch_ue', type=bool, default=True)
    parser.add_argument('--ip', type=str, default='127.0.0.1')
    parser.add_argument('--port', type=int, default=12345)
    parser.add_argument('--ue_command', type=str, default=r"C:\Program Files\Epic Games\UE_5.1"
                                                          r"\Engine\Binaries\Win64\UnrealEditor.exe")
    parser.add_argument('--uproject_path', type=str, default=os.getcwd() +
                                                             r"\RLInteractiveCrowd_UE\RLInteractiveCrowd.uproject")
    parser.add_argument('--map_path', type=str, default=r"/Game/RLCrowd/Maps/L_RLTrain.umap")
    parser.add_argument('--data_buffer_size', type=int, default=200000)
    parser.add_argument('--num_user_agents_chase', type=int, default=0)
    parser.add_argument('--num_user_agents_random', type=int, default=0)
    parser.add_argument('--num_obstacles', type=int, default=0)
    parser.add_argument('--terrain_filename', type=str, default='route_1.png')
    parser.add_argument('--terrain_obs_dim', type=ast.literal_eval, default=[32, 32])

    # Reward config
    parser.add_argument('--alive_penalty', type=float, default=7.0)
    parser.add_argument('--velocity_smooth_penalty', type=float, default=0.25)
    parser.add_argument('--collision_penalty', type=float, default=100.0)
    parser.add_argument('--target_approach_distance_reward', type=float, default=100.0)
    parser.add_argument('--target_away_distance_penalty_ratio', type=float, default=1.0)
    parser.add_argument('--use_nav_point', action='store_true', help='Enable navigation point')
    parser.add_argument('--no_use_nav_point', action='store_false', dest='use_nav_point',
                        help='Disable navigation point')
    parser.add_argument('--nav_point_approach_distance_reward', type=float, default=0.0)
    parser.add_argument('--target_arrive_reward', type=float, default=1000.0)
    parser.add_argument('--obstacle_penalty', type=float, default=1.0)
    parser.add_argument('--obstacle_time_coef', type=float, default=0.1)
    parser.add_argument('--terrain_penalty', type=float, default=50.0)

    # Training config
    # General
    parser.add_argument('--train_strategy', type=str, default='single_individual')
    parser.add_argument('--model_name', type=str, default='ppo')
    parser.add_argument('--ablation_type', type=str, default='RNT')
    parser.add_argument('--train_terrain_specific_name', type=str, default='white')
    parser.add_argument('--terrain_prev_set', type=ast.literal_eval, default=[])
    parser.add_argument('--terrain_prev_prob', type=float, default=0.2)
    parser.add_argument('--n_episodes', type=int, default=5000)
    parser.add_argument('--num_new_agents_per_episode', type=int, default=200)
    parser.add_argument('--lr_actor', type=float, default=0.000025)
    parser.add_argument('--lr_critic', type=float, default=0.00025)
    parser.add_argument('--fc1_dims', type=int, default=400)
    parser.add_argument('--fc2_dims', type=int, default=300)
    parser.add_argument('--fc2_dim_img', type=int, default=150)
    parser.add_argument('--pretrained_model_path', type=str, default='')
    parser.add_argument('--pretrained_entity_model_path', type=str, default='')
    parser.add_argument('--pretrained_implicit_model_path', type=str, default='')
    parser.add_argument('--fix_entity', action='store_true')
    parser.add_argument('--no_fix_entity', action='store_false', dest='fix_entity')
    parser.add_argument('--use_user_weight_range', action='store_true')
    parser.add_argument('--no_use_user_weight_range', action='store_false', dest='use_user_weight_range')
    parser.add_argument('--entity_weight_range', type=ast.literal_eval, default=[1.0, 1.0])
    parser.add_argument('--implicit_weight_range', type=ast.literal_eval, default=[1.0, 1.0])
    parser.add_argument('--delay_learn_step', type=int, default=256)
    parser.add_argument('--general_update_frequency', type=int, default=20)
    parser.add_argument('--update_best_avg_num', type=int, default=100)
    parser.add_argument('--checkpoint_save_freq', type=int, default=50)
    # PPO
    parser.add_argument('--action_std_decay_freq', type=int, default=50000)
    parser.add_argument('--ppo_max_mem', type=int, default=256)
    parser.add_argument('--action_std_init', type=float, default=0.6)
    parser.add_argument('--action_std_decay_rate', type=float, default=0.0005)
    parser.add_argument('--min_action_std', type=float, default=0.6)
    parser.add_argument('--ppo_batch_size', type=int, default=32)
    parser.add_argument('--ppo_n_epochs', type=int, default=8)
    # DDPG
    parser.add_argument('--ddpg_max_mem', type=int, default=100000)
    parser.add_argument('--ddpg_batch_size', type=int, default=1024)
    # TD3
    parser.add_argument('--td3_max_mem', type=int, default=100000)
    parser.add_argument('--td3_batch_size', type=int, default=64)
    parser.add_argument('--policy_noise', type=float, default=0.1)
    parser.add_argument('--noise_clip', type=float, default=0.25)
    parser.add_argument('--explore_noise', type=float, default=0.1)

    # Evaluate config
    parser.add_argument('--eval_n_episodes', type=int, default=100)
    parser.add_argument('--load_time_str', type=str, default='')
    parser.add_argument('--eval_env_size', type=ast.literal_eval, default=[40.0, 40.0])
    parser.add_argument('--eval_save_frame', type=bool, default=True)
    parser.add_argument('--eval_terrain_specific_name', type=str, default='white')
    parser.add_argument('--eval_train_type', type=str, default='feature_combine_weight')
    parser.add_argument('--eval_reward_weight_collision', type=float, default=1.0)
    parser.add_argument('--eval_reward_weight_terrain', type=float, default=1.0)

    # Log and painting
    parser.add_argument('--log_update_freq', type=int, default=100)

    args = parser.parse_args()
    args.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    return args


def save_args_to_file(args, path, file_name):
    """
    将 args 中的参数信息保存到文本文件。

    :param args: 包含参数的对象，通常是 argparse.Namespace 的实例。
    :param file_path: 要保存文件的路径。
    """
    if not os.path.exists(path):
        os.mkdir(path)

    file_path = os.path.join(path, file_name)
    with open(file_path, 'w') as file:
        for arg, value in vars(args).items():
            file.write(f'{arg}: {value}\n')


if __name__ == '__main__':
    print(parse_args())
