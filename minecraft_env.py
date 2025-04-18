import gymnasium as gym
import numpy as np
from typing import Dict, Tuple, Any
import pybullet as p
from terrain_generator import TerrainGenerator
from robot_model import RobotModel

class MinecraftEnv(gym.Env):
    def __init__(self, urdf_path: str, render: bool = False):
        """Initialize the Minecraft environment."""
        super().__init__()
        
        # Initialize PyBullet
        if render:
            self.physics_client_id = p.connect(p.GUI)
        else:
            self.physics_client_id = p.connect(p.DIRECT)
            
        p.setGravity(0, 0, -9.81, physicsClientId=self.physics_client_id)
        p.setTimeStep(1/240, physicsClientId=self.physics_client_id)
        
        # Initialize components
        self.terrain_generator = TerrainGenerator()
        self.robot = RobotModel(urdf_path, self.physics_client_id)
        
        # Define action and observation spaces
        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.robot.num_joints,),
            dtype=np.float32
        )
        
        # Enhanced observation space using Dict
        self.observation_space = gym.spaces.Dict({
            'joints': gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.robot.num_joints * 2,),
                dtype=np.float32
            ),
            'imu': gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(9,),  # 3 orientation + 3 angular vel + 3 linear acc
                dtype=np.float32
            ),
            'contacts': gym.spaces.MultiBinary(2),
            'target': gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(3,),
                dtype=np.float32
            )
        })
        
        # Initialize environment state
        self.terrain = None
        self.target_position = None
        self.current_position = None
        self.steps = 0
        self.max_steps = 1000
        self.difficulty = 0.0  # Initial difficulty level
        
    def _generate_terrain(self):
        """Generate a new terrain and set up the environment."""
        # Generate terrain data with current difficulty
        terrain_data = self.terrain_generator.generate_terrain(difficulty=self.difficulty)
        
        # Create ground plane
        ground_shape = p.createCollisionShape(
            shapeType=p.GEOM_HEIGHTFIELD,
            heightfieldData=terrain_data['height_map'].flatten(),
            numHeightfieldRows=terrain_data['height_map'].shape[0],
            numHeightfieldColumns=terrain_data['height_map'].shape[1],
            physicsClientId=self.physics_client_id
        )
        
        ground_visual = p.createVisualShape(
            shapeType=p.GEOM_HEIGHTFIELD,
            heightfieldData=terrain_data['height_map'].flatten(),
            numHeightfieldRows=terrain_data['height_map'].shape[0],
            numHeightfieldColumns=terrain_data['height_map'].shape[1],
            physicsClientId=self.physics_client_id
        )
        
        self.ground_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=ground_shape,
            baseVisualShapeIndex=ground_visual,
            basePosition=[0, 0, 0],
            physicsClientId=self.physics_client_id
        )
        
        # Set biome-specific properties
        biome_type = self.terrain_generator.get_biome_type(
            terrain_data['temperature'].mean(),
            terrain_data['humidity'].mean()
        )
        biome_props = self.terrain_generator.get_biome_properties(biome_type)
        p.changeDynamics(
            self.ground_id,
            -1,
            lateralFriction=biome_props['friction'],
            restitution=biome_props['bounce']
        )
        
        # Set random target position
        size = terrain_data['height_map'].shape[0]
        self.target_position = np.array([
            np.random.uniform(-size/2, size/2),
            np.random.uniform(-size/2, size/2),
            terrain_data['height_map'][size//2, size//2]
        ])
        
    def increase_difficulty(self):
        """Increase the environment difficulty."""
        self.difficulty = min(1.0, self.difficulty + 0.1)
        
    def reset(self, seed=None, options=None) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        
        # Reset physics simulation
        p.resetSimulation(physicsClientId=self.physics_client_id)
        
        # Generate new terrain
        self._generate_terrain()
        
        # Reset robot
        self.robot = RobotModel(self.robot.urdf_path, self.physics_client_id)
        
        # Reset environment state
        self.steps = 0
        self.current_position = np.array([0, 0, 1])
        
        # Get initial observation
        obs = self._get_observation()
        info = {'difficulty': self.difficulty}
        
        return obs, info
    
    def _get_observation(self) -> Dict[str, np.ndarray]:
        """Get the current observation."""
        state = self.robot.get_state()
        
        # Concatenate joint states
        joint_states = np.concatenate([
            state['joint_positions'],
            state['joint_velocities']
        ])
        
        # Concatenate IMU states
        imu_states = np.concatenate([
            state['imu_orientation'],
            state['imu_angular_velocity'],
            state['imu_linear_acceleration']
        ])
        
        # Convert foot contacts to array
        contacts = np.array([
            float(state['foot_contacts']['left_foot']),
            float(state['foot_contacts']['right_foot'])
        ])
        
        return {
            'joints': joint_states,
            'imu': imu_states,
            'contacts': contacts,
            'target': self.target_position
        }
    
    def _calculate_reward(self) -> float:
        """Calculate the reward for the current state."""
        # Get current position
        position, _ = p.getBasePositionAndOrientation(
            self.robot.robot_id,
            physicsClientId=self.physics_client_id
        )
        self.current_position = np.array(position)
        
        # Distance to target
        distance_to_target = np.linalg.norm(
            self.current_position[:2] - self.target_position[:2]
        )
        
        # Stability reward (based on IMU orientation)
        imu_data = self.robot.get_imu_measurements()
        stability_reward = -np.sum(np.abs(imu_data['orientation']))
        
        # Energy efficiency (based on joint torques)
        state = self.robot.get_state()
        energy_reward = -np.sum(np.square(state['joint_torques']))
        
        # Foot contact reward
        foot_contacts = self.robot.get_foot_contact()
        contact_reward = sum(foot_contacts.values())
        
        # Combine rewards with difficulty scaling
        reward = (
            -0.1 * distance_to_target +  # Distance penalty
            0.1 * stability_reward * (1 + self.difficulty) +  # Stability reward
            0.01 * energy_reward +  # Energy efficiency
            0.1 * contact_reward * (1 + self.difficulty)  # Foot contact reward
        )
        
        return reward
    
    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """Execute one time step in the environment."""
        # Apply action to robot
        self.robot.set_joint_commands(action)
        
        # Step simulation
        p.stepSimulation(physicsClientId=self.physics_client_id)
        
        # Update environment state
        self.steps += 1
        
        # Get observation
        obs = self._get_observation()
        
        # Calculate reward
        reward = self._calculate_reward()
        
        # Check termination conditions
        terminated = False
        truncated = False
        info = {}
        
        # Check if target reached
        if np.linalg.norm(self.current_position[:2] - self.target_position[:2]) < 1.0:
            reward += 1000  # Large reward for reaching target
            terminated = True
            info['success'] = True
            
        # Check if robot fell
        if self.current_position[2] < 0.5:  # Fell below half height
            reward -= 500  # Large penalty for falling
            terminated = True
            info['success'] = False
            
        # Check if max steps reached
        if self.steps >= self.max_steps:
            truncated = True
            info['success'] = False
            
        # Add difficulty information
        info['difficulty'] = self.difficulty
            
        return obs, reward, terminated, truncated, info
    
    def close(self):
        """Clean up the environment."""
        p.disconnect(physicsClientId=self.physics_client_id) 