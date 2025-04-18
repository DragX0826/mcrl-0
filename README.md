# Minecraft DRL Robot Training

This project implements a Deep Reinforcement Learning (DRL) system for training robot locomotion in procedurally generated Minecraft-like environments. The system combines terrain generation, physics simulation, and reinforcement learning to train robots to navigate complex 3D environments.

## Features

- Procedural terrain generation using fractal noise
- Biome-based environment variation
- Physics-based robot simulation using PyBullet
- Reinforcement learning using Stable Baselines3
- Support for parallel environment training
- TensorBoard integration for training visualization

## Requirements

- Python 3.8+
- PyBullet
- Stable Baselines3
- NumPy
- PyTorch
- Gym
- TensorBoard

Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

- `terrain_generator.py`: Implements procedural terrain generation using fractal noise
- `robot_model.py`: Defines the robot model and its physics simulation
- `minecraft_env.py`: Implements the Gym environment interface
- `train.py`: Main training script using PPO algorithm
- `robot.urdf`: Robot model description file (needs to be provided)

## Usage

1. Prepare your robot URDF file and place it in the project root directory as `robot.urdf`

2. Start training:
```bash
python train.py
```

3. Monitor training progress using TensorBoard:
```bash
tensorboard --logdir logs
```

## Training Configuration

The training script uses the following key parameters:
- Number of parallel environments: 4
- Total training timesteps: 1,000,000
- Learning rate: 3e-4
- Batch size: 64
- Number of epochs: 10
- Gamma: 0.99
- GAE Lambda: 0.95

## Environment Details

The environment provides the following observations:
- Joint positions
- Joint velocities
- IMU orientation
- IMU angular velocity
- IMU linear acceleration
- Foot contact states

Reward function components:
- Distance to target
- Stability (based on IMU orientation)
- Energy efficiency (based on joint torques)
- Foot contact patterns

## License

MIT License

## Acknowledgments

This project is inspired by recent advances in:
- Procedural content generation
- Physics-based simulation
- Deep reinforcement learning
- Robot locomotion

   
這個專案實現了一個深度強化學習 (DRL) 系統,用於在程序生成的 Minecraft 類似環境中訓練機器人運動。該系統結合了地形生成、物理模擬和強化學習,以訓練機器人在複雜的 3D 環境中導航。

### 主要功能點
- 使用分形噪聲實現程序化地形生成
- 基於生物群落的環境變化
- 使用 PyBullet 進行基於物理的機器人模擬
- 使用 Stable Baselines3 進行強化學習
- 支持並行環境訓練
- 使用 TensorBoard 進行訓練可視化

### 技術棧
- Python 3.8+
- PyBullet
- Stable Baselines3
- NumPy
- PyTorch
- Gym
- TensorBoard

### 授權
MIT 授權

