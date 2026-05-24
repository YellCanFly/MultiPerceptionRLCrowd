# RLImplicitPerceptionCrowd

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2.1%2Bcu121-orange)
![Unreal Engine](https://img.shields.io/badge/Unreal%20Engine-5.1-blueviolet)
![Platform](https://img.shields.io/badge/Platform-Windows-lightgrey)

## Table of Contents

- [Abstract](#abstract)
- [Project Structure](#project-structure)
- [Environment Setup](#environment-setup)
  - [Python Environment Setup](#1-python-environment-setup)
  - [Unreal Engine 5.1 Setup](#2-unreal-engine-51-setup)
- [Run Preparation](#run-preparation)
- [Run Trained Model](#run-trained-model)
- [Retrain Model](#retrain-model)
  - [Train the Entity Module](#1-train-the-entity-module-first-stage)
  - [Train the Implicit Module](#2-train-the-implicit-module-first-stage)
  - [Train the Feature Fuse Module](#3-train-the-feature-fuse-module-second-stage)
- [Citation](#citation)
- [License](#license)

---

## Abstract

This study presents a novel approach to crowd simulation by integrating implicit environmental perception with reinforcement learning (RL)-based control strategies. Implicit environmental rules, represented as grayscale maps, are utilized to encode movement preferences, enhancing the realism and diversity of simulated crowd behaviors. The proposed method employs a two-phase, modular RL training strategy. In the first phase, entity perception and implicit environmental perception modules are trained independently. In the second phase, a multi-perception fusion module combines these capabilities, leveraging dynamically adjusted perception weights to simulate heterogeneous agent behaviors. The approach includes curriculum-based environmental templates to improve generalization to complex scenarios. Experimental results demonstrate that the proposed method achieves realistic crowd movement patterns, effective collision avoidance, and adaptability across diverse environments, validated through both quantitative metrics and qualitative analysis. This work advances the field of crowd simulation by balancing environmental rule adherence and dynamic obstacle interaction to create more heterogeneous and adaptable virtual agents.

---

## Project Structure

```
RLImplicitPerceptionCrowd/
├── config.py                        # Global configuration and argument parser
├── train.py                         # Training entry point
├── eval.py                          # Evaluation entry point
├── server.py                        # Communication server with Unreal Engine
├── utils.py                         # Utility functions
├── calculate_IP.py                  # Implicit perception calculation
├── calculate_NN.py                  # Neural network calculation helpers
├── requirements.txt                 # Python dependencies
│
├── envs/
│   └── env_terrain.py               # RL environment with terrain support
│
├── model/                           # Network architecture definitions
│   ├── rl_model.py
│   ├── ppo_raycast_nav_feature_combine_net.py
│   ├── ppo_raycast_nav_pure_implicit_net.py
│   └── ...
│
├── train_strategy/                  # Modular training strategy scripts
│   ├── train_single_individual_entity_net.py
│   ├── train_single_individual_pure_implicit_weight_net.py
│   ├── train_single_individual_feature_combine_weight_net.py
│   └── ...
│
├── checkpoint_ppo/                  # Saved model checkpoints
├── terrain_maps/                    # Grayscale implicit environment maps
└── RLInteractiveCrowd_UE/           # Unreal Engine 5.1 simulation project
```

---

## Environment Setup

This project requires both a Python environment and Unreal Engine 5.1. Follow the steps below to set up the environment on **Windows**.

### 1. Python Environment Setup
Before proceeding, ensure **Python 3.8 or later** is installed on your system. You can download the latest version from the [official Python website](https://www.python.org/downloads/).

#### Steps to Set Up Python Environment:
1. **Create a Virtual Environment** (Recommended for dependency management):
   - Open **Command Prompt (cmd)** and navigate to your project directory:
     ```cmd
     cd path\to\your\project
     ```
   - Create a virtual environment:
     ```cmd
     python -m venv venv
     ```
   - Activate the virtual environment:
     ```cmd
     venv\Scripts\activate
     ```
     *(Your command prompt should now show `(venv)` at the beginning.)*

2. **Install Dependencies**:
   - Run the following command to install all required packages:
     ```cmd
     pip install -r requirements.txt
     ```

#### Dependencies Installed:
The following Python packages will be installed:
- `gym==0.26.2` – RL environment framework
- `ipython==8.12.3` – Interactive Python shell
- `matplotlib==3.8.3` – Data visualization library
- `numpy==2.2.2` – Numerical computing library
- `Pillow==11.1.0` – Image processing toolkit
- `scipy==1.15.1` – Scientific computing library
- `seaborn==0.13.2` – Statistical visualization
- `torch==2.2.1+cu121` & `torchvision==0.17.1+cu121` – PyTorch with **CUDA 12.1** support

3. **Verify the Installation**:
   - Check that Python is installed:
     ```cmd
     python --version
     ```
   - Verify that PyTorch can use your GPU (if available):
     ```python
     python -c "import torch; print(torch.cuda.is_available())"
     ```
   - If the output is `True`, PyTorch has successfully detected your GPU.

---

### 2. Unreal Engine 5.1 Setup
This project utilizes **Unreal Engine 5.1** for crowd simulation. Follow these steps to install and configure it on Windows.

#### Steps to Install Unreal Engine 5.1:
1. **Download & Install Epic Games Launcher**:
   - Download from [Epic Games official website](https://www.unrealengine.com/en-US/download).
   - Install and launch the **Epic Games Launcher**.

2. **Install Unreal Engine 5.1**:
   - Open **Epic Games Launcher** and navigate to the **Unreal Engine** tab.
   - Click **Library** → **Install Engine**.
   - Select **Version 5.1** and click **Install**.
   - Ensure that **Visual Studio 2019/2022** is installed (required for compilation).

3. **Verify Installation**:
   - After installation, open **Unreal Engine 5.1** from the Epic Games Launcher.
   - Create or open a sample project to confirm the setup works.


## Run Preparation

Follow these steps to ensure the project is properly set up and ready to run.

### 1. Clone This Repository
First, clone the repository to your local machine:
```cmd
git clone https://github.com/YellCanFly/MultiPerceptionRLCrowd.git
cd RLImplicitPerceptionCrowd
```

### 2. Set Up Unreal Engine Project
1. Navigate to the **RLInteractiveCrowd_UE** folder.
2. Right-click on **RLInteractiveCrowd.uproject**, and select **Generate Visual Studio project files**.
3. Open the newly generated `.sln` file with **Visual Studio 2019/2022**.
4. In **Visual Studio**, click **Build** to compile the project.
5. Once the build is complete, double-click **RLInteractiveCrowd.uproject**.
6. If the Unreal Engine project opens successfully, the UE environment setup is complete.

### 3. Configure `config.py`
1. Open the `config.py` file in a text editor.
2. Locate the parameter `--ue_command`.
3. Update its path to match your **actual** Unreal Engine installation directory. The default value is:
   ```python
   --ue_command="C:\Program Files\Epic Games\UE_5.1"
   ```
   If your Unreal Engine is installed in a different location, modify the path accordingly.


## Run Trained Model

To evaluate a pre-trained model, run the following command in the **project root directory**:

```cmd
python eval.py --port 12350 --load_time_str "3090_1223_FeatureCombine_Turning" --eval_terrain_specific_name "route_1" --eval_n_episodes 1 --num_agents 30 --num_new_agents_per_episode 200
```

### Explanation of Arguments:
- `--load_time_str`: Specifies the pre-trained model name, stored in the **checkpoint_ppo** directory.
- `--eval_terrain_specific_name`: Defines the implicit environment configuration name. Other environments can be found in the **terrain_maps** directory.
- `--eval_n_episodes`: Sets the number of evaluation runs.
- `--num_agents`: Determines the initial number of agents in the environment.
- `--num_new_agents_per_episode`: Specifies the number of new agents generated per test episode.


## Retrain Model

To train the model from scratch, follow the three training stages outlined below.

### 1. Train the **Entity** Module (First Stage)
Run the following command to train the **Entity** module:

```cmd
python train.py --n_episodes 3000 --train_terrain_specific_name "white" --num_agents 50 --train_strategy "single_individual_entity_weight_net"
```

#### Explanation of Parameters:
- `--n_episodes`: Number of training episodes (**3000** in this case).
- `--train_terrain_specific_name`: The environment used for training. Here, `"white"` is a simple environment for entity training.
- `--num_agents`: The number of agents present in the environment.
- `--train_strategy`: Specifies the training strategy. `"single_individual_entity_weight_net"` focuses on training the **Entity** module separately.

---

### 2. Train the **Implicit** Module (First Stage)
Run the following command to train the **Implicit** module:

```cmd
python train.py --n_episodes 10000 --train_terrain_specific_name "white/single/turning/cross" --num_agents 30 --train_strategy "single_individual_pure_implicit_weight_net" --pretrained_model_path "checkpoint_ppo/xxxxx"
```

#### Explanation of Parameters:
- `--n_episodes`: Number of training episodes (**10000** in this case).
- `--train_terrain_specific_name`: Specifies the training environments.  
  - `"white"` → `"single"` → `"turning"` → `"cross"` (Increasing complexity of implicit environments).
- `--num_agents`: Number of agents present in the environment.
- `--train_strategy`: Training strategy for **Implicit** module. `"single_individual_pure_implicit_weight_net"` is used for training the implicit perception module.
- `--pretrained_model_path`: Path to the previously trained model checkpoint. This should be set to the **output model from the previous training stage**.

#### Curriculum Learning Approach:
- The **Implicit module training** follows a curriculum learning approach, gradually increasing the complexity of the environment.
- Training starts from the **simplest environment (`white`)** and progressively moves towards more complex scenarios (`single`, `turning`, `cross`).
- Before moving to the next training phase, **set `--pretrained_model_path` to the model checkpoint from the previous stage**.

---

### 3. Train the **Feature Fuse** Module (Second Stage)
Run the following command to train the **Feature Fuse** module:

```cmd
python train.py --n_episodes 10000 --train_terrain_specific_name "turning" --num_agents 30 --train_strategy "single_individual_feature_combine_weight_net" --pretrained_entity_model_path "checkpoint_ppo/3090_1218_WeightMLP_Pretrained" --pretrained_implicit_model_path "checkpoint_ppo/3090_1218_WeightMLP_Pretrained"
```

#### Explanation of Parameters:
- `--n_episodes`: Number of training episodes (**10000** in this case).
- `--train_terrain_specific_name`: The environment used for training. `"turning"` is used to ensure smooth feature fusion training.
- `--num_agents`: The number of agents present in the environment.
- `--train_strategy`: Specifies the training strategy. `"single_individual_feature_combine_weight_net"` is used to train the **Feature Fusion module**.
- `--pretrained_entity_model_path`: Path to the **pre-trained Entity module** checkpoint.
- `--pretrained_implicit_model_path`: Path to the **pre-trained Implicit module** checkpoint.

#### Feature Fusion Training:
- In this stage, the model **freezes the parameters** of the previously trained **Entity** and **Implicit** modules.
- Only the **Feature Fusion** module is trained, allowing it to learn how to combine the learned entity-based and implicit perceptions effectively.

---

## Citation

If you find this work useful in your research, please consider citing:

```bibtex
@article{huang2026multi,
  title={Multi-Perception Crowd: Learning to combine entity and implicit perception for diverse crowd simulation},
  author={Huang, Kexiang and Argudo, Oscar and Ding, Gangyi and Pelechano, Nuria},
  year={2026},
  publisher={TechRxiv}
}
```

---

## License

This project is released under the [MIT License](LICENSE).

