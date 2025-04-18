import numpy as np
from typing import Dict, List, Tuple
import pybullet as p
from scipy.signal import butter, filtfilt

class RobotModel:
    def __init__(self, urdf_path: str, physics_client_id: int = 0):
        """Initialize the robot model with its URDF file and physics client."""
        self.physics_client_id = physics_client_id
        
        # Configure debug visualization
        p.configureDebugVisualizer(
            p.COV_ENABLE_RENDERING, 0,
            physicsClientId=physics_client_id
        )
        
        # Load robot model
        self.robot_id = p.loadURDF(
            urdf_path,
            basePosition=[0, 0, 1],
            baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
            physicsClientId=physics_client_id
        )
        
        # Enable rendering after loading
        p.configureDebugVisualizer(
            p.COV_ENABLE_RENDERING, 1,
            physicsClientId=physics_client_id
        )
        
        # Get joint information
        self.num_joints = p.getNumJoints(self.robot_id, physicsClientId=physics_client_id)
        self.joint_info = self._get_joint_info()
        
        # Initialize joint states
        self.joint_states = {
            'positions': np.zeros(self.num_joints),
            'velocities': np.zeros(self.num_joints),
            'torques': np.zeros(self.num_joints)
        }
        
        # IMU simulation parameters
        self.imu_position = [0, 0, 0.5]  # Approximate center of mass
        self.imu_orientation = [0, 0, 0]
        
        # Initialize safety parameters
        self.max_velocity = 5.0  # rad/s
        self.max_acceleration = 10.0  # rad/s²
        self.last_velocities = np.zeros(self.num_joints)
        
        # Initialize low-pass filter for joint commands
        self.filter_order = 2
        self.cutoff_freq = 10.0  # Hz
        self.sampling_freq = 240.0  # Hz
        self.b, self.a = butter(
            self.filter_order,
            self.cutoff_freq / (self.sampling_freq / 2),
            btype='low'
        )
        self.filter_state = np.zeros((self.filter_order, self.num_joints))
        
    def _get_joint_info(self) -> List[Dict]:
        """Get information about all joints in the robot."""
        joint_info = []
        for i in range(self.num_joints):
            info = p.getJointInfo(self.robot_id, i, physicsClientId=self.physics_client_id)
            joint_info.append({
                'joint_index': i,
                'joint_name': info[1].decode('utf-8'),
                'joint_type': info[2],
                'q_index': info[3],
                'u_index': info[4],
                'flags': info[5],
                'joint_damping': info[6],
                'joint_friction': info[7],
                'joint_lower_limit': info[8],
                'joint_upper_limit': info[9],
                'joint_max_force': info[10],
                'joint_max_velocity': info[11],
                'link_name': info[12].decode('utf-8'),
                'joint_axis': info[13],
                'parent_frame_pos': info[14],
                'parent_frame_orn': info[15],
                'parent_index': info[16]
            })
        return joint_info
    
    def _enforce_joint_limits(self, target_positions: np.ndarray) -> np.ndarray:
        """Enforce joint position limits."""
        for i in range(self.num_joints):
            low = self.joint_info[i]['joint_lower_limit']
            high = self.joint_info[i]['joint_upper_limit']
            target_positions[i] = np.clip(target_positions[i], low, high)
        return target_positions
    
    def _enforce_velocity_limits(self, target_velocities: np.ndarray) -> np.ndarray:
        """Enforce joint velocity limits."""
        return np.clip(target_velocities, -self.max_velocity, self.max_velocity)
    
    def _enforce_acceleration_limits(self, target_velocities: np.ndarray) -> np.ndarray:
        """Enforce joint acceleration limits."""
        dt = 1.0 / self.sampling_freq
        current_accelerations = (target_velocities - self.last_velocities) / dt
        limited_accelerations = np.clip(
            current_accelerations,
            -self.max_acceleration,
            self.max_acceleration
        )
        limited_velocities = self.last_velocities + limited_accelerations * dt
        self.last_velocities = limited_velocities
        return limited_velocities
    
    def _filter_joint_commands(self, commands: np.ndarray) -> np.ndarray:
        """Apply low-pass filter to joint commands."""
        filtered_commands = np.zeros_like(commands)
        for i in range(self.num_joints):
            filtered_commands[i] = filtfilt(
                self.b,
                self.a,
                commands[i],
                padlen=len(commands[i])-1
            )[-1]
        return filtered_commands
    
    def get_imu_measurements(self) -> Dict[str, np.ndarray]:
        """Get simulated IMU measurements (orientation, angular velocity, linear acceleration)."""
        position, orientation = p.getBasePositionAndOrientation(
            self.robot_id,
            physicsClientId=self.physics_client_id
        )
        linear_velocity, angular_velocity = p.getBaseVelocity(
            self.robot_id,
            physicsClientId=self.physics_client_id
        )
        
        # Convert quaternion to Euler angles
        euler_angles = p.getEulerFromQuaternion(orientation)
        
        return {
            'orientation': np.array(euler_angles),
            'angular_velocity': np.array(angular_velocity),
            'linear_acceleration': np.array(linear_velocity)  # Simplified for simulation
        }
    
    def get_foot_contact(self) -> Dict[str, bool]:
        """Get contact state for each foot."""
        contact_points = p.getContactPoints(
            self.robot_id,
            physicsClientId=self.physics_client_id
        )
        
        # Assuming feet are the last two links
        foot_contacts = {
            'left_foot': False,
            'right_foot': False
        }
        
        for contact in contact_points:
            link_index = contact[3]  # Link index of the robot
            if link_index == self.num_joints - 2:  # Left foot
                foot_contacts['left_foot'] = True
            elif link_index == self.num_joints - 1:  # Right foot
                foot_contacts['right_foot'] = True
                
        return foot_contacts
    
    def set_joint_commands(self, target_positions: np.ndarray, target_velocities: np.ndarray = None):
        """Set target positions and velocities for all joints with safety checks."""
        if target_velocities is None:
            target_velocities = np.zeros_like(target_positions)
        
        # Apply safety limits
        target_positions = self._enforce_joint_limits(target_positions)
        target_velocities = self._enforce_velocity_limits(target_velocities)
        target_velocities = self._enforce_acceleration_limits(target_velocities)
        
        # Apply low-pass filter
        target_positions = self._filter_joint_commands(target_positions)
        target_velocities = self._filter_joint_commands(target_velocities)
            
        for i in range(self.num_joints):
            p.setJointMotorControl2(
                self.robot_id,
                i,
                controlMode=p.POSITION_CONTROL,
                targetPosition=target_positions[i],
                targetVelocity=target_velocities[i],
                force=self.joint_info[i]['joint_max_force'],
                physicsClientId=self.physics_client_id
            )
    
    def update_joint_states(self):
        """Update the current joint states."""
        for i in range(self.num_joints):
            joint_state = p.getJointState(
                self.robot_id,
                i,
                physicsClientId=self.physics_client_id
            )
            self.joint_states['positions'][i] = joint_state[0]
            self.joint_states['velocities'][i] = joint_state[1]
            self.joint_states['torques'][i] = joint_state[3]
    
    def get_state(self) -> Dict[str, np.ndarray]:
        """Get the complete state of the robot."""
        self.update_joint_states()
        imu_data = self.get_imu_measurements()
        foot_contacts = self.get_foot_contact()
        
        return {
            'joint_positions': self.joint_states['positions'],
            'joint_velocities': self.joint_states['velocities'],
            'joint_torques': self.joint_states['torques'],
            'imu_orientation': imu_data['orientation'],
            'imu_angular_velocity': imu_data['angular_velocity'],
            'imu_linear_acceleration': imu_data['linear_acceleration'],
            'foot_contacts': foot_contacts
        } 