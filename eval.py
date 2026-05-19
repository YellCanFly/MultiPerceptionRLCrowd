import os.path
import random
import numpy as np

from envs.env_terrain import MultiAgentEnv

# PPO Entity
from model.ppo_raycast_nav_pure_entity_net import  PPO as PPO_RNT_pure_entity
from model.ppo_raycast_nav_pure_entity_weight_mlp_net import PPO as PPO_RNT_pure_entity_weight
# PPO Implicit
from model.ppo_raycast_nav_pure_implicit_net import PPO as PPO_RNT_pure_implicit
from model.ppo_raycast_nav_pure_implicit_weight_mlp_net import PPO as PPO_RNT_pure_implicit_weight
# PPO Feature combine
from model.ppo_raycast_nav_feature_combine_net import PPO as PPO_RNT_feature_combine
from model.ppo_raycast_nav_feature_combine_weight_mlp_net import PPO as PPO_RNT_feature_combine_weight
# PPO Ablation
from model.ppo_raycast_nav_holistic_net import PPO as PPO_RNT_holisitc
from model.ppo_raycast_nav_holistic_no_weight_mlp import PPO as PPO_RNT_holisitc_no_weight_mlp

from config import parse_args
from utils import get_local_time


def main():
    global eval_model
    args = parse_args()
    loc_time_str = '%d_%d_%d_%d_%d_%d' % get_local_time()

    # 创建环境
    env = MultiAgentEnv(
        num_agents=args.num_agents,
        agent_radius=args.agent_radius,
        num_rays=args.num_rays,
        max_ray_distance=args.max_ray_distance,
        ray_vision_lamda=args.ray_vision_lamda,
        max_speed_orientation=args.max_speed_orientation,
        max_speed_perpendicular=args.max_speed_perpendicular,
        max_acceleration=args.max_acceleration,
        max_angular_velocity=args.max_angular_velocity,
        agent_terrain_obs_range=args.agent_terrain_obs_range,
        dt=args.dt,
        width=args.eval_env_size[0],
        height=args.eval_env_size[1],
        min_walking_distance=args.min_walking_distance,
        max_alive_time=args.max_alive_time,
        max_time_overlapped_with_obstacle=args.max_time_overlapped_with_obstacle,
        num_new_agents_per_episode=args.num_new_agents_per_episode,
        env_title='eval_model_%s-%s' % (args.model_name, args.load_time_str),
        auto_launch_ue=args.auto_launch_ue,
        args=args,
        num_user_agents_chase=args.num_user_agents_chase,
        num_user_agents_random=args.num_user_agents_random,
        num_obstacles=args.num_obstacles,
        terrain_filename=args.terrain_filename,
        use_terrain=True,
    )

    if args.model_name == 'ppo':
        train_types = {
            "pure_entity": PPO_RNT_pure_entity,
            "pure_entity_weight": PPO_RNT_pure_entity_weight,
            "pure_implicit": PPO_RNT_pure_implicit,
            "pure_implicit_weight": PPO_RNT_pure_implicit_weight,
            "feature_combine": PPO_RNT_feature_combine,
            "feature_combine_weight": PPO_RNT_feature_combine_weight,

            "holistic": PPO_RNT_holisitc,
            "holistic_no_weight_mlp": PPO_RNT_holisitc_no_weight_mlp
        }
        eval_model = train_types[args.eval_train_type](
            lr_actor=args.lr_actor,
            lr_critic=args.lr_critic,

            ray_vec_dims=args.num_rays * 3,
            agent_vec_dims=7,
            weight_vec_dims=2,
            state_img_dims=env.terrain_obs_dim,
            action_dims=env.action_dims,

            fc1_dims=args.fc1_dims,
            fc2_dims=args.fc2_dims,
            fc2_dim_img=args.fc2_dim_img,

            action_std_init=args.action_std_init,
            max_mem=args.ppo_max_mem,
            batch_size=args.ppo_batch_size,
            n_epochs=args.ppo_n_epochs,
            cur_time_str=loc_time_str
        )

    eval_model.load_models(load_path='checkpoint_%s/%s' % (args.model_name, args.load_time_str))
    print('init finished')

    total_step = 0
    episode_history = []

    for episode_id in range(args.eval_n_episodes):
        # 初始化记录
        episode_step = 0  # 一次episode内部的step计数
        score_history = []  # 一次episode内部，每个step的rewords之和记录

        # 初始化环境
        terrain_specific_name = args.eval_terrain_specific_name
        vec_states, img_states = env.reset(args.eval_env_size[0], args.eval_env_size[1],
                                           env_type='eval', terrain_specific_name=terrain_specific_name,
                                           reward_weight_terrain_range=[args.eval_reward_weight_terrain,
                                                                        args.eval_reward_weight_terrain],
                                           reward_weight_collision_range=[args.eval_reward_weight_collision,
                                                                          args.eval_reward_weight_collision])
        all_agent_done = False

        while not all_agent_done:
            # 获取行为，更新环境
            actions = eval_model.choose_action_eval((vec_states, img_states)).reshape(-1, env.action_dims)
            states_, rewards, dones, infos = env.step(actions)
            # time.sleep(0.5)

            # 获得下次迭代需要的状态
            vec_states, img_states = infos['next_state']

            step_score = np.sum(rewards)
            score_history.append(step_score)
            total_step += 1
            episode_step += 1
            all_agent_done = (len(env.exist_agent_indexes) == 1 and dones[0]) or len(env.exist_agent_indexes) == 0

            if total_step % args.log_update_freq == 0:
                avg_past = np.mean(score_history[-args.log_update_freq:])
                print('episode_id=', episode_id,
                      'total_step=', total_step,
                      'episode_step=', episode_step,
                      'step_score=%.2f' % step_score,
                      'trailing_%d_step_avg=%.3f' % (args.log_update_freq, float(avg_past)),
                      'current_agents=%d' % len(env.exist_agent_indexes),
                      'left_new_agents=%d' % env.num_left_new_agent)

        # 分数记录
        episode_score = np.sum(np.array(score_history))
        episode_history.append(episode_score)

        # 分数文件保存
        score_np = np.array(episode_history).reshape(-1, 1)
        score_path = 'eval_score/'
        if not os.path.exists(score_path):
            os.mkdir(score_path)
        filename = '%s_reward_score_%s.csv' % (loc_time_str, args.eval_train_type)
        filename = os.path.join(score_path, filename)
        np.savetxt(filename, score_np, delimiter=",", header="Value", comments="")


if __name__ == '__main__':
    main()
