## README 摘要

### 專案簡介
這個專案旨在實現一個深度強化學習系統，專注於在類似 Minecraft 的環境中訓練機器人運動。使用程序生成技術創建複雜的 3D 環境，並利用強化學習算法來優化機器人的行為。

### 主要功能
- **程序化地形生成**：使用分形噪聲來生成隨機地形。
- **物理模擬**：採用 PyBullet 進行物理模擬，確保機器人在環境中的運動真實。
- **強化學習訓練**：利用 Stable Baselines3 進行 DRL 訓練。
- **環境變化**：基於生物群落的變化來影響環境。
- **並行訓練**：支持多個環境的並行訓練以提高效率。
- **可視化工具**：使用 TensorBoard 進行訓練過程的可視化。

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

---

## 未解的問題與解法

### 問題1：環境穩定性
- **描述**：在某些情況下，環境可能不穩定，導致訓練過程中出現異常行為。
- **解法**：檢查物理參數設置，調整模擬步長，並增加環境重置的頻率。

### 問題2：訓練速度慢
- **描述**：在大規模環境中，訓練速度可能會變得非常緩慢。
- **解法**：優化模型架構，使用更高效的算法，或增加硬體資源以加快計算速度。

### 問題3：行為學習不佳
- **描述**：機器人可能無法學習到有效的行為策略。
- **解法**：調整獎勵函數，增加訓練樣本的多樣性，或使用不同的強化學習算法進行實驗。


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
