import os
import torch
import torch.nn as nn
import torch.optim as optim
from PIL.ImageFont import load_path
from torch.distributions import MultivariateNormal
import numpy as np

import utils
from model.rl_model import RLModel
import time
from utils import init_net
from config import parse_args


class PPOMemory:
    def __init__(self, max_size):
        self.state_vecs = []
        self.state_imgs = []
        self.actions = []
        self.probs = []
        self.vals = []
        self.rewards = []
        self.dones = []
        self.max_size = max_size
        self.cur_size = 0

    def sample(self):
        try:
            return np.array(self.state_vecs), \
                   np.array(self.state_imgs), \
                   np.array(self.actions), \
                   np.array(self.probs), \
                   np.array(self.vals), \
                   np.array(self.rewards), \
                   np.array(self.dones)
        except ValueError as e:
            # 输出错误信息和每个列表的大小
            print("Error occurred during sampling:", str(e))
            print("Shapes of each element in memory:")
            print("Actions Shapes:", [np.array(a) for a in self.actions])
            print("Probs Shapes:", [np.array(p).shape for p in self.probs])
            print("Vals Shapes:", [np.array(v).shape for v in self.vals])
            print("Rewards Shapes:", [np.array(r).shape for r in self.rewards])
            print("Dones Shapes:", [np.array(d).shape for d in self.dones])
            raise  # 可以选择重新抛出异常或处理错误

    def store_memory(self, state_vec, state_img, action, probs, val, reward, done):
        if self.cur_size >= self.max_size:
            # 删除列表最前端的元素
            self.state_vecs.pop(0)
            self.state_imgs.pop(0)
            self.actions.pop(0)
            self.probs.pop(0)
            self.vals.pop(0)
            self.rewards.pop(0)
            self.dones.pop(0)
        else:
            self.cur_size += 1

        # 追加新的记忆
        self.state_vecs.append(state_vec)
        self.state_imgs.append(state_img)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(val)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear_memory(self):
        self.state_vecs = []
        self.state_imgs = []
        self.actions = []
        self.probs = []
        self.vals = []
        self.rewards = []
        self.dones = []
        self.cur_size = 0


class PPOMemories:
    def __init__(self, max_size=20):
        self.agent_memories = {}
        self.max_size = max_size

    def add_memory(self, agent_id):
        self.agent_memories[agent_id] = PPOMemory(self.max_size)

    def remove_memory(self, agent_id):
        if agent_id in self.agent_memories:
            del self.agent_memories[agent_id]

    def store_memory(self, agent_id, *args):
        if agent_id in self.agent_memories:
            self.agent_memories[agent_id].store_memory(*args)
        else:
            self.add_memory(agent_id)
            self.agent_memories[agent_id].store_memory(*args)

    def clear_full_memories(self):
        for memory in self.agent_memories.values():
            if memory.cur_size == memory.max_size:
                memory.clear_memory()

    def clear_all_memories(self):
        for memory in self.agent_memories.values():
            memory.clear_memory()

    def get_all_memories(self):
        # 初始化存储所有智能体样本的列表
        state_vecs, state_imgs, actions, probs, vals, rewards, dones = [], [], [], [], [], [], []

        for agent_id, memory in self.agent_memories.items():
            if memory.cur_size == memory.max_size:
                (agent_state_vecs, agent_state_imgs,
                 agent_actions, agent_probs, agent_vals, agent_rewards, agent_dones) = memory.sample()

                # 将每个智能体的数据添加到对应的列表中
                state_vecs.append(agent_state_vecs)
                state_imgs.append(agent_state_imgs)
                actions.append(agent_actions)
                probs.append(agent_probs)
                vals.append(agent_vals)
                rewards.append(agent_rewards)
                dones.append(agent_dones)
        # 将列表转换为 NumPy 数组
        return np.array(state_vecs), \
               np.array(state_imgs), \
               np.array(actions), \
               np.array(probs), \
               np.array(vals), \
               np.array(rewards), \
               np.array(dones)

    def sample(self, batch_size=5):
        batch_start = np.arange(0, self.max_size, batch_size)
        indices = np.arange(self.max_size)
        np.random.shuffle(indices)
        batches = [indices[i:i + batch_size] for i in batch_start]

        # 将列表转换为 NumPy 数组
        return batches


class ActorNetwork(nn.Module):
    def __init__(self, lr, ray_vec_dims, agent_vec_dims, weight_vec_dims, state_img_dims, action_dims,
                 fc1_dims, fc2_dims, fc2_dim_img,
                 action_std_init=0.6,
                 fix_entity=False,
                 chkpt_dir='checkpoint/', name='actor'):
        super(ActorNetwork, self).__init__()

        self.action_var = torch.full((action_dims,), action_std_init * action_std_init)

        self.lr = lr
        self.ray_vec_dims = ray_vec_dims
        self.agent_vec_dims = agent_vec_dims
        self.weight_vec_dims = weight_vec_dims
        self.state_img_dims = state_img_dims
        self.action_dims = action_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.fc2_dim_img = fc2_dim_img
        self.weight_fc_dims = 128

        self.name = name
        self.chkpt_dir = chkpt_dir
        self.chkpt_file = os.path.join(chkpt_dir, self.name)

        self.fix_entity=fix_entity
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.model_entity_weight = nn.Sequential(
            nn.Linear(1, self.fc1_dims),  # 显式处理 weight_state[..., 0]
            nn.LayerNorm(self.fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, self.weight_fc_dims),
            nn.LayerNorm(self.weight_fc_dims),
            nn.ReLU(),
        )
        init_net(self.model_entity_weight[0])
        init_net(self.model_entity_weight[3])

        # ray_vec_dims + agent_vec_dims + 1(weight) -> fc2_dims
        self.model_entity = nn.Sequential(
            nn.Linear(self.ray_vec_dims + self.agent_vec_dims + self.weight_fc_dims, self.fc1_dims),
            nn.LayerNorm(self.fc1_dims),
            nn.ReLU(),
            nn.Linear(self.fc1_dims, self.fc2_dims),
            nn.LayerNorm(self.fc2_dims),
            nn.ReLU()
        )
        init_net(self.model_entity[0])
        init_net(self.model_entity[3])

        # state_img_dims -> fc2_dim_img
        self.model_img = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Added MaxPool layer
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Added MaxPool layer
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Added MaxPool layer
            nn.Flatten(),
            nn.Linear(128 * (self.state_img_dims[0] // 8) * (self.state_img_dims[1] // 8), 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),  # Added Dropout for regularization
            nn.Linear(512, self.fc2_dim_img),
            nn.ReLU()
        )
        init_net(self.model_img[0])
        init_net(self.model_img[4])
        init_net(self.model_img[8])
        init_net(self.model_img[13])
        init_net(self.model_img[17])

        self.model_implicit_weight = nn.Sequential(
            nn.Linear(1, self.fc1_dims),  # 显式处理 weight_state[..., 1]
            nn.LayerNorm(self.fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, self.weight_fc_dims),
            nn.LayerNorm(self.weight_fc_dims),
            nn.ReLU(),
        )
        init_net(self.model_implicit_weight[0])
        init_net(self.model_implicit_weight[3])

        # img_feature_dims + agent_vec_dims + 1(weight) -> fc2_dims
        self.model_implicit = nn.Sequential(
            nn.Linear(self.fc2_dim_img + self.agent_vec_dims + self.weight_fc_dims, self.fc1_dims),
            nn.LayerNorm(self.fc1_dims),
            nn.ReLU(),
            nn.Linear(self.fc1_dims, self.fc2_dims),
            nn.LayerNorm(self.fc2_dims),
            nn.ReLU()
        )
        init_net(self.model_implicit[0])
        init_net(self.model_implicit[3])

        self.model_mu = nn.Sequential(
            nn.Linear(self.fc2_dims * 2 + self.weight_fc_dims * 2, self.action_dims),
            nn.Tanh()
        )
        init_net(self.model_mu[0], 0.003)

        self.optimizer = optim.Adam(self.model_mu.parameters(), lr=self.lr)

    def forward(self, state_vec, state_img):
        ray_state, weight_state, agent_state = \
            torch.split(state_vec, [self.ray_vec_dims, self.weight_vec_dims, self.agent_vec_dims], dim=-1)
        if len(agent_state.shape) < 2:
            agent_state = agent_state[None, :]
            weight_state = weight_state[None, :]
            ray_state = ray_state[None, :]

        self.model_entity_weight.eval()
        self.model_implicit_weight.eval()
        self.model_entity.eval()
        self.model_img.eval()
        self.model_implicit.eval()
        with torch.no_grad():
            entity_weight_feature = self.model_entity_weight(weight_state[..., 0].unsqueeze(-1))
            entity_feature = self.model_entity(
                torch.cat((ray_state, agent_state, entity_weight_feature), dim=-1))

            state_img = state_img.to(self.device)
            state_img = state_img.reshape(-1, 1, self.state_img_dims[0], self.state_img_dims[1])
            img_feature = self.model_img(state_img)
            img_feature = img_feature.reshape(entity_feature.size()[:-1] + img_feature.size()[-1:])
            implicit_weight_feature = self.model_implicit_weight(weight_state[..., 1].unsqueeze(-1))
            implicit_feature = self.model_implicit(
                torch.cat((img_feature, agent_state, implicit_weight_feature), dim=-1))


        combined_feature = torch.cat((entity_feature, entity_weight_feature, implicit_feature, implicit_weight_feature), dim=-1)
        action_mean = self.model_mu(combined_feature)

        return action_mean

        # cov_mat = torch.diag(self.action_var).unsqueeze(dim=0).to(self.device)
        # dist = MultivariateNormal(action_mean, cov_mat)
        #
        # return dist

    def set_action_std(self, new_action_std):
        self.action_var = torch.full((self.action_dims,), new_action_std * new_action_std)

    def save_model(self, save_path=None):
        if save_path is None:
            chkpt_dir = self.chkpt_dir
        else:
            chkpt_dir = save_path
        chkpt_file = os.path.join(chkpt_dir, self.name)
        if not os.path.exists(chkpt_dir):
            os.makedirs(chkpt_dir)
        torch.save(self.state_dict(), chkpt_file)

    def load_model(self, load_path=None):
        if load_path is None:
            load_file_name = self.chkpt_file
        else:
            load_file_name = os.path.join(load_path, self.name)
        self.load_state_dict(torch.load(load_file_name))

    def load_entity(self, load_path=None):
        """
        Load the parameters from the checkpoint file into model_entity.
        """
        if load_path is None:
            load_file_name = self.chkpt_file
            load_file_name_entity_weight = self.chkpt_file
        else:
            load_file_name = os.path.join(load_path, self.name + '_entity')
            load_file_name_entity_weight = os.path.join(load_path, self.name + '_entity_weight')
        if os.path.exists(load_file_name):
            print(f"Loading model from {load_file_name}")
            self.model_entity.load_state_dict(torch.load(load_file_name, map_location=self.device))
            self.model_entity_weight.load_state_dict(torch.load(load_file_name_entity_weight, map_location=self.device))
            for param in self.model_entity.parameters():
                param.requires_grad = False
            for param in self.model_entity_weight.parameters():
                param.requires_grad = False
        else:
            print(f"No checkpoint found at {load_file_name}, skipping loading.")

    def load_implicit(self, load_path=None):
        """
        Load the parameters from the checkpoint file into model_entity.
        """
        if load_path is None:
            load_file_name_implicit = self.chkpt_file
            load_file_name_img = self.chkpt_file
            load_file_name_implicit_weight = self.chkpt_file
        else:
            load_file_name_implicit = os.path.join(load_path, self.name + '_implicit')
            load_file_name_img = os.path.join(load_path, self.name + '_img')
            load_file_name_implicit_weight = os.path.join(load_path, self.name + '_implicit_weight')

        if os.path.exists(load_file_name_implicit) and os.path.exists(load_file_name_img):
            print(f"Loading model from {load_path}")
            self.model_implicit.load_state_dict(torch.load(load_file_name_implicit, map_location=self.device))
            self.model_img.load_state_dict(torch.load(load_file_name_img, map_location=self.device))
            self.model_implicit_weight.load_state_dict(torch.load(load_file_name_implicit_weight, map_location=self.device))
            for param in self.model_implicit.parameters():
                param.requires_grad = False
            for param in self.model_img.parameters():
                param.requires_grad = False
            for param in self.model_implicit_weight.parameters():
                param.requires_grad = False
        else:
            print(f"No checkpoint found at {load_path}, skipping loading.")


class CriticNetwork(nn.Module):
    def __init__(self, lr, ray_vec_dims, agent_vec_dims, weight_vec_dims, state_img_dims, action_dims,
                 fc1_dims, fc2_dims, fc2_dim_img,
                 fix_entity=False,
                 chkpt_dir='checkpoint/', name='critic'):
        super(CriticNetwork, self).__init__()

        self.lr = lr
        self.ray_vec_dims = ray_vec_dims
        self.agent_vec_dims = agent_vec_dims
        self.weight_vec_dims = weight_vec_dims
        self.state_img_dims = state_img_dims
        self.action_dims = action_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.fc2_dim_img = fc2_dim_img
        self.weight_fc_dims = 128
        self.chkpt_dir = chkpt_dir
        self.fix_entity = fix_entity
        self.name = name
        self.chkpt_file = os.path.join(self.chkpt_dir, self.name)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.model_entity_weight = nn.Sequential(
            nn.Linear(1, self.fc1_dims),  # 显式处理 weight_state[..., 0]
            nn.LayerNorm(self.fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, self.weight_fc_dims),
            nn.LayerNorm(self.weight_fc_dims),
            nn.ReLU(),
        )
        init_net(self.model_entity_weight[0])
        init_net(self.model_entity_weight[3])

        # ray_vec_dims + agent_vec_dims + 1(weight) -> fc2_dims
        self.model_entity = nn.Sequential(
            nn.Linear(self.ray_vec_dims + self.agent_vec_dims + self.weight_fc_dims, self.fc1_dims),
            nn.LayerNorm(self.fc1_dims),
            nn.ReLU(),
            nn.Linear(self.fc1_dims, self.fc2_dims),
            nn.LayerNorm(self.fc2_dims),
            nn.ReLU()
        )
        init_net(self.model_entity[0])
        init_net(self.model_entity[3])

        # state_img_dims -> fc2_dim_img
        self.model_img = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Added MaxPool layer
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Added MaxPool layer
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Added MaxPool layer
            nn.Flatten(),
            nn.Linear(128 * (self.state_img_dims[0] // 8) * (self.state_img_dims[1] // 8), 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),  # Added Dropout for regularization
            nn.Linear(512, self.fc2_dim_img),
            nn.ReLU()
        )
        init_net(self.model_img[0])
        init_net(self.model_img[4])
        init_net(self.model_img[8])
        init_net(self.model_img[13])
        init_net(self.model_img[17])

        self.model_implicit_weight = nn.Sequential(
            nn.Linear(1, self.fc1_dims),  # 显式处理 weight_state[..., 1]
            nn.LayerNorm(self.fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, self.weight_fc_dims),
            nn.LayerNorm(self.weight_fc_dims),
            nn.ReLU(),
        )
        init_net(self.model_implicit_weight[0])
        init_net(self.model_implicit_weight[3])

        # img_feature_dims + agent_vec_dims + 1(weight) -> fc2_dims
        self.model_implicit = nn.Sequential(
            nn.Linear(self.fc2_dim_img + self.agent_vec_dims + self.weight_fc_dims, self.fc1_dims),
            nn.LayerNorm(self.fc1_dims),
            nn.ReLU(),
            nn.Linear(self.fc1_dims, self.fc2_dims),
            nn.LayerNorm(self.fc2_dims),
            nn.ReLU()
        )
        init_net(self.model_implicit[0])
        init_net(self.model_implicit[3])

        self.critic = nn.Sequential(
            nn.Linear(self.fc2_dims * 2 + self.weight_fc_dims * 2, 1),
        )
        init_net(self.critic[0], 0.003)

        self.optimizer = optim.Adam(self.critic.parameters(), lr=lr)

    def forward(self, state_vec, state_img):
        ray_state, weight_state, agent_state = \
            torch.split(state_vec, [self.ray_vec_dims, self.weight_vec_dims, self.agent_vec_dims], dim=-1)
        if len(agent_state.shape) < 2:
            agent_state = agent_state[None, :]
            weight_state = weight_state[None, :]
            ray_state = ray_state[None, :]

        self.model_entity_weight.eval()
        self.model_implicit_weight.eval()
        self.model_entity.eval()
        self.model_img.eval()
        self.model_implicit.eval()
        with torch.no_grad():
            entity_weight_feature = self.model_entity_weight(weight_state[..., 0].unsqueeze(-1))
            entity_feature = self.model_entity(
                torch.cat((ray_state, agent_state, entity_weight_feature), dim=-1))

            state_img = state_img.to(self.device)
            state_img = state_img.reshape(-1, 1, self.state_img_dims[0], self.state_img_dims[1])
            img_feature = self.model_img(state_img)
            img_feature = img_feature.reshape(entity_feature.size()[:-1] + img_feature.size()[-1:])
            implicit_weight_feature = self.model_implicit_weight(weight_state[..., 1].unsqueeze(-1))
            implicit_feature = self.model_implicit(
                torch.cat((img_feature, agent_state, implicit_weight_feature), dim=-1))

        combined_feature = torch.cat((entity_feature, entity_weight_feature, implicit_feature, implicit_weight_feature), dim=-1)
        return self.critic(combined_feature)

    def save_model(self, save_path=None):
        if save_path is None:
            chkpt_dir = self.chkpt_dir
        else:
            chkpt_dir = save_path
        chkpt_file = os.path.join(chkpt_dir, self.name)
        if not os.path.exists(chkpt_dir):
            os.makedirs(chkpt_dir)
        torch.save(self.state_dict(), chkpt_file)

    def load_model(self, load_path=None):
        if load_path is None:
            load_file_name = self.chkpt_file
        else:
            load_file_name = os.path.join(load_path, self.name)
        self.load_state_dict(torch.load(load_file_name))

    def load_entity(self, load_path=None):
        """
        Load the parameters from the checkpoint file into model_entity.
        """
        if load_path is None:
            load_file_name = self.chkpt_file
            load_file_name_entity_weight = self.chkpt_file
        else:
            load_file_name = os.path.join(load_path, self.name + '_entity')
            load_file_name_entity_weight = os.path.join(load_path, self.name + '_entity_weight')
        if os.path.exists(load_file_name):
            print(f"Loading model from {load_file_name}")
            self.model_entity.load_state_dict(torch.load(load_file_name, map_location=self.device))
            self.model_entity_weight.load_state_dict(torch.load(load_file_name_entity_weight, map_location=self.device))
            for param in self.model_entity.parameters():
                param.requires_grad = False
            for param in self.model_entity_weight.parameters():
                param.requires_grad = False
        else:
            print(f"No checkpoint found at {load_file_name}, skipping loading.")

    def load_implicit(self, load_path=None):
        """
        Load the parameters from the checkpoint file into model_entity.
        """
        if load_path is None:
            load_file_name_implicit = self.chkpt_file
            load_file_name_img = self.chkpt_file
            load_file_name_implicit_weight = self.chkpt_file
        else:
            load_file_name_implicit = os.path.join(load_path, self.name + '_implicit')
            load_file_name_img = os.path.join(load_path, self.name + '_img')
            load_file_name_implicit_weight = os.path.join(load_path, self.name + '_implicit_weight')

        if os.path.exists(load_file_name_implicit) and os.path.exists(load_file_name_img):
            print(f"Loading model from {load_path}")
            self.model_implicit.load_state_dict(torch.load(load_file_name_implicit, map_location=self.device))
            self.model_img.load_state_dict(torch.load(load_file_name_img, map_location=self.device))
            self.model_implicit_weight.load_state_dict(
                torch.load(load_file_name_implicit_weight, map_location=self.device))
            for param in self.model_implicit.parameters():
                param.requires_grad = False
            for param in self.model_img.parameters():
                param.requires_grad = False
            for param in self.model_implicit_weight.parameters():
                param.requires_grad = False
        else:
            print(f"No checkpoint found at {load_path}, skipping loading.")


class PPO(RLModel):
    def __init__(self, lr_actor, lr_critic,
                 ray_vec_dims, agent_vec_dims, weight_vec_dims, state_img_dims, action_dims,
                 fc1_dims, fc2_dims, fc2_dim_img,
                 action_std_init=0.6, max_mem=20, gamma=0.99, gae_lamda=0.95, policy_clip=0.2, batch_size=64, n_epochs=10,
                 fix_entity=False,
                 cur_time_str='default'):
        super(PPO, self).__init__()
        self.gamma = gamma
        self.gae_lamda = gae_lamda
        self.policy_clip = policy_clip
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.max_mem = max_mem
        self.action_std = action_std_init

        self.probs = None
        self.state_value = None

        # Replay buffer
        self.memory = PPOMemory(self.batch_size)
        self.memories = PPOMemories(self.max_mem)

        # Actor and target actor
        self.actor = ActorNetwork(lr_actor,
                                  ray_vec_dims, agent_vec_dims, weight_vec_dims,
                                  state_img_dims,
                                  action_dims, fc1_dims, fc2_dims, fc2_dim_img,
                                  action_std_init=action_std_init,
                                  fix_entity=fix_entity,
                                  chkpt_dir='checkpoint_ppo/%s/' % cur_time_str, name='actor').to(self.device)
        self.critic = CriticNetwork(lr_critic,
                                    ray_vec_dims, agent_vec_dims, weight_vec_dims,
                                    state_img_dims,
                                    action_dims, fc1_dims, fc2_dims, fc2_dim_img,
                                    fix_entity=fix_entity,
                                    chkpt_dir='checkpoint_ppo/%s/' % cur_time_str, name='critic').to(self.device)

    # 存储经验，其中probs是选择行为的概率，val是对state的评估值。
    def remember(self, agent_id, index, state_vec, state_img, action, reward, done):
        # 防止只剩一个agent时去访问单个值的索引
        if index >= 0:
            probs = self.probs[index]
            val = self.state_value[index]
        else:
            probs = np.float32(self.probs)
            val = np.float32(self.state_value)
        self.memories.store_memory(agent_id, state_vec, state_img, action, probs, val, reward, done)

    # 选择动作函数，传入state，输出action的采样，action的log_prob，以及state的评估值。
    def choose_action(self, state):
        state_vec = torch.tensor(np.array(state[0]), dtype=torch.float).to(self.device)
        state_img = state[1].to(self.device)
        dist = self.actor(state_vec, state_img)
        value = self.critic(state_vec, state_img)
        action = dist.sample()

        probs = torch.squeeze(dist.log_prob(action)).cpu().detach().numpy()
        action = torch.squeeze(action)
        value = torch.squeeze(value).cpu().detach().numpy()

        self.probs = probs
        self.state_value = value

        return action.cpu().detach().numpy()

    def choose_action_eval(self, state):
        self.actor.eval()
        with torch.no_grad():
            state_vec = torch.tensor(np.array(state[0]), dtype=torch.float).to(self.device)
            state_img = state[1].to(self.device)
            dist = self.actor(state_vec, state_img)
            action = dist.sample()
        return action.cpu().detach().numpy()

    # 存储网络模型。
    def save_models(self, save_path=None):
        print('... Saving models ...')
        self.actor.save_model(save_path)
        self.critic.save_model(save_path)

    # 加载网络模型。
    def load_models(self, load_path=None):
        print('... Loading models ...')
        self.actor.load_model(load_path)
        self.critic.load_model(load_path)

    def load_models_entity(self, load_path=None):
        self.actor.load_entity(load_path)
        self.critic.load_entity(load_path)

    def load_models_implicit(self, load_path=None):
        self.actor.load_implicit(load_path)
        self.critic.load_implicit(load_path)

    # 赋值另一个PPO中的参数到此PPO中
    def copy_models(self, other_models=None):
        if not isinstance(other_models, PPO):
            return
        self.actor.load_state_dict(other_models.actor.state_dict())
        self.critic.load_state_dict(other_models.critic.state_dict())

    # 逐渐衰减行为空间分布的标准差
    def decay_action_std(self, action_std_decay_rate=0.05, min_action_std=0.1):
        self.action_std = self.action_std - action_std_decay_rate
        self.action_std = round(self.action_std, 4)
        if self.action_std <= min_action_std:
            self.action_std = min_action_std
            print("setting actor output action_std to min_action_std : ", self.action_std)
        else:
            print("setting actor output action_std to : ", self.action_std)
        self.set_action_std(self.action_std)

    def set_action_std(self, new_action_std):
        self.action_std = new_action_std
        self.actor.set_action_std(new_action_std)

    # 训练和更新模型过程。
    def learn(self):
        # 获取所有满足长度的经验
        memory = self.memories.get_all_memories()
        state_vec_arr = memory[0]  # agent_num * max_mem * state_dims
        state_img_arr = memory[1]  # agent_num * max_mem * state_dims
        action_arr = memory[2]  # agent_num * max_mem * action_dims
        old_probs_arr = memory[3]  # agent_num * max_mem
        vals_arr = memory[4]  # agent_num * max_mem
        reward_arr = memory[5]  # agent_num * max_mem
        done_arr = np.array(memory[6], dtype=int)  # agent_num * max_mem

        for _ in range(self.n_epochs):
            # 经验回放池采样
            batches = self.memories.sample(self.batch_size)

            if len(state_vec_arr) <= 0:
                return

            # time1 = time.time()
            # Eq.11 and Eq.12 in paper
            advantage = np.zeros_like(reward_arr)  # agent_num * max_mem
            gae = np.zeros((reward_arr.shape[0]))

            for t in reversed(range(reward_arr.shape[1] - 1)):
                delta = reward_arr[:, t] + self.gamma * vals_arr[:, t + 1] * (1 - done_arr[:, t]) - vals_arr[:, t]
                gae = delta + self.gamma * self.gae_lamda * gae * (1 - done_arr[:, t])
                advantage[:, t] = gae

            advantage = torch.tensor(advantage).to(self.device)

            # print('loop time=%f', time.time()-time1)

            # agent_num * max_mem
            values = torch.tensor(vals_arr).to(self.device)

            for batch in batches:
                # agent_num * batch_size * state_dims
                state_vecs = torch.tensor(state_vec_arr[:, batch, :], dtype=torch.float).to(self.device)
                # agent_num * batch_size * state_dims
                state_imgs = torch.tensor(state_img_arr[:, batch, :], dtype=torch.float).to(self.device)
                # agent_num * batch_size
                old_probs = torch.tensor(old_probs_arr[:, batch], dtype=torch.float).to(self.device)
                # agent_num * batch_size * action_dims
                actions = torch.tensor(action_arr[:, batch, :]).to(self.device)

                # agent_num * batch_size * action_dims
                dist = self.actor(state_vecs, state_imgs)

                # agent_num * batch_size
                critic_value = self.critic(state_vecs, state_imgs)
                critic_value = torch.squeeze(critic_value)

                # agent_num * batch_size -> float
                dist_entropy = dist.entropy().mean()  # 加入total loss

                # agent_num * batch_size
                new_probs = dist.log_prob(actions)
                # agent_num * batch_size
                prob_ratio = new_probs.exp() / old_probs.exp()

                # Eq.6
                weighted_probs = advantage[:, batch] * prob_ratio
                # Eq.7
                weighted_clipped_probs = torch.clamp(prob_ratio, 1 - self.policy_clip, 1 + self.policy_clip) \
                                         * advantage[:, batch]
                actor_loss = -torch.min(weighted_probs, weighted_clipped_probs).mean()

                returns = advantage[:, batch] + values[:, batch]
                critic_loss = (returns - critic_value) ** 2
                critic_loss = critic_loss.mean()

                total_loss = actor_loss + 0.5 * critic_loss - 0.01 * dist_entropy

                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                total_loss.backward()
                self.actor.optimizer.step()
                self.critic.optimizer.step()

        self.memories.clear_full_memories()


if __name__ == '__main__':
    args = parse_args()
    rl_model = PPO(
        lr_actor=args.lr_actor,
        lr_critic=args.lr_critic,

        ray_vec_dims=args.num_rays * 3,
        agent_vec_dims=7,
        weight_vec_dims=2,
        state_img_dims=[32,32],
        action_dims=3,

        fc1_dims=args.fc1_dims,
        fc2_dims=args.fc2_dims,
        fc2_dim_img=args.fc2_dim_img,
    )
    pretrained_entity_model_path = '../checkpoint_ppo/3090_1223_FeatureCombine_Turning'
    rl_model.load_models(pretrained_entity_model_path)
    print(rl_model.actor)
    utils.Convert_ONNX(rl_model.actor, ray_vec_dims=args.num_rays * 3, agent_vec_dims=7,
                       weight_vec_dims=2, state_img_dims=[32, 32],
                       file_name='actor.onnx')