import gymnasium as gym
import os
import torch as th
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch.nn as nn
from minecraft_env import MinecraftEnv

class CurriculumCallback:
    def __init__(self, env, success_threshold=0.8, eval_freq=10000):
        self.env = env
        self.success_threshold = success_threshold
        self.eval_freq = eval_freq
        self.success_history = []
        
    def __call__(self, locals_, globals_):
        if locals_['self'].num_timesteps % self.eval_freq == 0:
            # Get success rate from last evaluation
            success_rate = np.mean(self.success_history[-10:]) if self.success_history else 0.0
            
            if success_rate > self.success_threshold:
                self.env.increase_difficulty()
                print(f"Increased difficulty to {self.env.difficulty}")
                
        return True

class CustomFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict):
        super().__init__(observation_space, features_dim=256)
        
        # Process joint states
        self.joint_net = nn.Sequential(
            nn.Linear(observation_space['joints'].shape[0], 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
        
        # Process IMU data
        self.imu_net = nn.Sequential(
            nn.Linear(observation_space['imu'].shape[0], 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
        
        # Process contacts
        self.contact_net = nn.Sequential(
            nn.Linear(observation_space['contacts'].shape[0], 16),
            nn.ReLU()
        )
        
        # Process target position
        self.target_net = nn.Sequential(
            nn.Linear(observation_space['target'].shape[0], 16),
            nn.ReLU()
        )
        
        # Final feature processing
        self.final_net = nn.Sequential(
            nn.Linear(32 + 32 + 16 + 16, 256),
            nn.ReLU()
        )
        
    def forward(self, observations: Dict[str, th.Tensor]) -> th.Tensor:
        # Process each observation component
        joint_features = self.joint_net(observations['joints'])
        imu_features = self.imu_net(observations['imu'])
        contact_features = self.contact_net(observations['contacts'])
        target_features = self.target_net(observations['target'])
        
        # Concatenate all features
        combined_features = th.cat([
            joint_features,
            imu_features,
            contact_features,
            target_features
        ], dim=1)
        
        # Process final features
        return self.final_net(combined_features)

def make_env(urdf_path: str, render: bool = False):
    def _init():
        env = MinecraftEnv(urdf_path, render)
        return env
    return _init

def main():
    # Create output directory
    os.makedirs('output', exist_ok=True)
    
    # Create environment
    env = DummyVecEnv([make_env('robot.urdf')])
    env = VecNormalize(env, norm_obs=True, norm_reward=True)
    
    # Create callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path='output/',
        name_prefix='rl_model'
    )
    
    eval_env = DummyVecEnv([make_env('robot.urdf')])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=True)
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path='output/',
        log_path='output/',
        eval_freq=10000,
        deterministic=True,
        render=False
    )
    
    curriculum_callback = CurriculumCallback(env.envs[0])
    
    # Create model
    policy_kwargs = dict(
        features_extractor_class=CustomFeatureExtractor,
        features_extractor_kwargs=dict()
    )
    
    model = PPO(
        "MultiInputPolicy",
        env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log='output/'
    )
    
    # Train model
    model.learn(
        total_timesteps=1000000,
        callback=[checkpoint_callback, eval_callback, curriculum_callback]
    )
    
    # Save final model
    model.save('output/final_model')
    env.save('output/vec_normalize.pkl')

if __name__ == '__main__':
    main() 