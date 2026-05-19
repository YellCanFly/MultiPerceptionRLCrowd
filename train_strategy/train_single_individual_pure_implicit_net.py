import time
import numpy as np
from envs.env_terrain import MultiAgentEnv
from model.ppo_raycast_nav_pure_implicit_net import PPO
from utils import plot, get_local_time
from config import parse_args, save_args_to_file


def add_new_model(args, env, loc_time_str):
    rl_model = None
    if args.model_name == 'ppo':
        rl_model = PPO(
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

            fix_entity=args.fix_entity,
            cur_time_str=loc_time_str
        )
    return rl_model


def main():
    args = parse_args()
    loc_time_str = '%d_%d_%d_%d_%d_%d' % get_local_time()
    save_args_to_file(args, 'checkpoint_%s/' % args.model_name + loc_time_str + '/', 'config_info.txt')
    action_std_decay_freq = args.action_std_decay_freq

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
        action_interval_step_num=args.action_interval_step_num,
        width=args.start_env_size[0],
        height=args.start_env_size[1],
        min_walking_distance=args.min_walking_distance,
        max_alive_time=args.max_alive_time,
        max_time_overlapped_with_obstacle=args.max_time_overlapped_with_obstacle,
        num_new_agents_per_episode=args.num_new_agents_per_episode,
        env_title='%s-%s-%s' % (args.model_name, args.train_strategy, loc_time_str),
        auto_launch_ue=args.auto_launch_ue,
        args=args,
        num_user_agents_chase=args.num_user_agents_chase,
        num_user_agents_random=args.num_user_agents_random,
        num_obstacles=args.num_obstacles,
        terrain_filename=args.terrain_filename,
        use_terrain=True,
    )

    # 创建RL模型
    rl_general_model = add_new_model(args, env, loc_time_str)
    rl_update_model = add_new_model(args, env, loc_time_str)
    rl_general_model.copy_models(rl_update_model)
    rl_models = {'general': rl_general_model, 'update': rl_update_model}

    # 根据配置初始化模型
    if args.pretrained_model_path != '':
        for rl_model in rl_models.values():
            rl_model.load_models(args.pretrained_model_path)

    if args.pretrained_entity_model_path != '':
        for rl_model in rl_models.values():
            rl_model.load_models_entity()

    print('init finished')

    # 初始化数值和数组
    total_step = 0
    best_score = -np.inf
    action_std_history = []
    episode_update_agent_history = []
    episode_update_agent_avg_history = []

    # 开始迭代episode
    for episode_id in range(args.n_episodes):
        episode_step = 0  # 一次episode内部的step计数
        update_agent_score = 0  # 本次update的agent的得分

        # 设置terrain_type
        terrain_type = args.train_terrain_specific_name
        terrain_prev_set = args.terrain_prev_set
        if len(terrain_prev_set) == 0:
            terrain_prev_set = None

        # 每env_scale_frequency次episode，环境大小按线性比例在start_env_size和end_env_size间缩放。
        alpha = ((episode_id + 1) // args.env_scale_frequency) / (args.n_episodes // args.env_scale_frequency)
        env_size = (1 - alpha) * np.array(args.start_env_size) + alpha * np.array(args.end_env_size)
        if args.use_user_weight_range:
            vec_states, img_states = env.reset(width=env_size[0], height=env_size[1],
                                               terrain_type=terrain_type,
                                               terrain_prev_set=terrain_prev_set,
                                               reward_weight_collision_range=args.entity_weight_range,
                                               reward_weight_terrain_range=args.implicit_weight_range
                                               )
        else:
            vec_states, img_states = env.reset(width=env_size[0], height=env_size[1],
                                               terrain_type=terrain_type,
                                               terrain_prev_set=terrain_prev_set,
                                               reward_weight_collision_range=[1.0, 1.0],
                                               reward_weight_terrain_range=[1.0, 1.0]
                                               )

        # 如果是ppo，在开始新的episode时清空memory
        if args.model_name == 'ppo':
            for rl_model in rl_models.values():
                rl_model.memories.clear_all_memories()

        # 在一个episode开始前，把上一个episode中的update模型参数复制给general模型
        # rl_general_model.copy_models(rl_update_model)

        # 决定这轮episode要更新的agent的index
        update_agent_index = 0
        print('update agent id=', update_agent_index)

        # 初始化程序运行记录时间
        time_1 = 0  # time_update_and_sense
        time_2 = 0  # time_remember
        time_3 = 0  # time_learn

        while not env.all_agent_done:
            # 获取行为，更新环境
            time_1_start = time.time()
            if args.model_name == 'ddpg' or args.model_name == 'td3':
                actions = rl_general_model.choose_action_eval((vec_states, img_states)).reshape(-1, env.action_dims)
                actions[update_agent_index] = rl_update_model.choose_action((vec_states[update_agent_index],
                                                                             img_states[None, update_agent_index]))
            elif args.model_name == 'ppo':
                actions = rl_update_model.choose_action((vec_states, img_states)).reshape(-1, env.action_dims)

            # 更新环境
            (states_, img_states_), rewards, dones, infos = env.step(actions)
            time_1 += (time.time() - time_1_start)

            # 模型记忆，只记忆选定的update agent即可。
            time_2_start = time.time()
            if args.model_name == 'ddpg' or args.model_name == 'td3':
                for agent_id in range(len(vec_states)):
                    rl_update_model.remember(vec_states[agent_id],
                                             img_states[agent_id],
                                             actions[agent_id],
                                             rewards[agent_id],
                                             states_[agent_id],
                                             img_states_[agent_id],
                                             dones[agent_id])
            elif args.model_name == 'ppo':
                if len(vec_states) > 1:
                    for id in range(len(vec_states)):
                        rl_update_model.remember(env.before_update_agent_indexes[id], id,
                                                 vec_states[id],
                                                 img_states[id],
                                                 actions[id],
                                                 rewards[id],
                                                 dones[id])
                elif len(vec_states) == 1:
                    rl_update_model.remember(env.before_update_agent_indexes[0], -1,
                                             vec_states[0],
                                             img_states[0],
                                             actions[0],
                                             rewards[0],
                                             dones[0])
            time_2 += (time.time() - time_2_start)

            # 模型学习，只学习update_model即可。
            time_3_start = time.time()
            if args.model_name == 'ddpg' or args.model_name == 'td3':
                if episode_step % args.delay_learn_step == 0 or env.all_agent_done:
                    rl_update_model.learn()
            elif args.model_name == 'ppo':
                if (episode_step != 0 and episode_step % args.delay_learn_step == 0) or env.all_agent_done:
                    # print('---start updating---')
                    print("Learn " + str(episode_step))
                    rl_update_model.learn()
                if total_step % action_std_decay_freq == 0:
                    rl_update_model.decay_action_std(
                        action_std_decay_rate=args.action_std_decay_rate,
                        min_action_std=args.min_action_std
                    )
            time_3 += (time.time() - time_3_start)

            # 获得下次迭代需要的状态
            vec_states, img_states = infos['next_state']

            # 判断选定update_agent是否还活着
            if dones[update_agent_index]:
                env.all_agent_done = True

            # 更新记录
            update_agent_score += rewards[update_agent_index]
            total_step += 1
            episode_step += 1

            # 每log_update_freq次step输出和绘图一次
            if episode_step % args.log_update_freq == 0:
                print('episode_id=', episode_id,
                      'total_step=', total_step,
                      'episode_step=', episode_step,
                      'step_score=%.2f' % rewards[update_agent_index],
                      'current_agents_num=%d' % len(env.exist_agent_indexes),
                      'left_new_agents_num=%d' % env.num_left_new_agent)

                print('time_update_and_sense=%.2f' % time_1,
                      'time_remember=%.2f' % time_2,
                      'time_learn=%.2f' % time_3)
                time_1 = 0
                time_2 = 0
                time_3 = 0

        # 存储和更新update agent最佳记录，更新general模型
        episode_update_agent_history.append(update_agent_score)
        update_agent_score_avg = np.mean(episode_update_agent_history[-args.update_best_avg_num:])
        episode_update_agent_avg_history.append(update_agent_score_avg)
        recent_avg_scores = episode_update_agent_avg_history[-args.update_best_avg_num:]

        if args.model_name == 'ppo':
            action_std_history.append(rl_update_model.action_std)
            print('action std = %f', rl_update_model.action_std)

        # Save the best model
        if update_agent_score_avg > best_score:
            best_score = update_agent_score_avg
            rl_update_model.save_models()
            rl_general_model.copy_models(rl_update_model)

        # Save the ascent record
        if update_agent_score_avg >= max(recent_avg_scores):
            rl_update_model.save_models(save_path='checkpoint_%s/%s/ascent_record/%04d' %
                                                  (args.model_name, loc_time_str, episode_id))

        # Save the checkpoint in fixed frequency
        if episode_id % args.checkpoint_save_freq == 0:
            rl_update_model.save_models(save_path='checkpoint_%s/%s/checkpoints/%04d' %
                                                  (args.model_name, loc_time_str, episode_id))

        # Copy the update model to general model in fixed frequency
        if episode_id % args.general_update_frequency == 0:
            rl_general_model.copy_models(rl_update_model)

        # Save the newest model
        rl_update_model.save_models(save_path='checkpoint_%s/%s/up_to_date' % (args.model_name, loc_time_str))

        # 存储分数记录（update_agent的分数）
        update_agent_score_np = np.array(episode_update_agent_history).reshape(-1, 1)
        filename = 'checkpoint_%s/%s/update_agent_score_%s.csv' % (args.model_name, loc_time_str, loc_time_str)
        np.savetxt(filename, update_agent_score_np, delimiter=",", header="Value", comments="")

        if args.model_name == 'ppo':
            action_std_history_np = np.array(action_std_history).reshape(-1, 1)
            filename = 'checkpoint_%s/%s/action_std_history_%s.csv' % (args.model_name, loc_time_str, loc_time_str)
            np.savetxt(filename, action_std_history_np, delimiter=",", header="Value", comments="")


if __name__ == '__main__':
    main()
