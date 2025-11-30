
import numpy as np
import os
import torch

from isaacgym import gymutil, gymtorch, gymapi
from .base.vec_task import VecTask

class CartpoleSpinGPT(VecTask):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        self.cfg = cfg

        self.reset_dist = self.cfg["env"]["resetDist"]

        self.max_push_effort = self.cfg["env"]["maxEffort"]
        self.max_episode_length = 1000

        self.cfg["env"]["numObservations"] = 4
        self.cfg["env"]["numActions"] = 1

        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)

        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]

        self.cumulative_rotation = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.prev_pole_angle = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

        self.consecutive_successes = torch.zeros(1, dtype=torch.float, device=self.device)

    def create_sim(self):
        self.up_axis = self.cfg["sim"]["up_axis"]

        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0) if self.up_axis == 'z' else gymapi.Vec3(0.0, 1.0, 0.0)
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(0.5 * -spacing, -spacing, 0.0) if self.up_axis == 'z' else gymapi.Vec3(0.5 * -spacing, 0.0, -spacing)
        upper = gymapi.Vec3(0.5 * spacing, spacing, spacing)

        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../assets")
        asset_file = "urdf/cartpole.urdf"

        if "asset" in self.cfg["env"]:
            asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.cfg["env"]["asset"].get("assetRoot", asset_root))
            asset_file = self.cfg["env"]["asset"].get("assetFileName", asset_file)

        asset_path = os.path.join(asset_root, asset_file)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        cartpole_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(cartpole_asset)

        pose = gymapi.Transform()
        if self.up_axis == 'z':
            pose.p.z = 2.0
            pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
        else:
            pose.p.y = 2.0
            pose.r = gymapi.Quat(-np.sqrt(2)/2, 0.0, 0.0, np.sqrt(2)/2)

        self.cartpole_handles = []
        self.envs = []
        for i in range(self.num_envs):
            env_ptr = self.gym.create_env(
                self.sim, lower, upper, num_per_row
            )
            cartpole_handle = self.gym.create_actor(env_ptr, cartpole_asset, pose, "cartpole", i, 1, 0)

            dof_props = self.gym.get_actor_dof_properties(env_ptr, cartpole_handle)
            dof_props['driveMode'][0] = gymapi.DOF_MODE_EFFORT
            dof_props['driveMode'][1] = gymapi.DOF_MODE_NONE
            dof_props['stiffness'][:] = 0.0
            dof_props['damping'][:] = 0.0
            self.gym.set_actor_dof_properties(env_ptr, cartpole_handle, dof_props)

            self.envs.append(env_ptr)
            self.cartpole_handles.append(cartpole_handle)

    def compute_reward(self):
        self.rew_buf[:], self.rew_dict = compute_reward(self.cumulative_rotation, self.dof_vel)
        self.extras['gpt_reward'] = self.rew_buf.mean()
        for rew_state in self.rew_dict: self.extras[rew_state] = self.rew_dict[rew_state].mean()
        pole_angle = self.obs_buf[:, 2]
        pole_vel = self.obs_buf[:, 3]
        cart_vel = self.obs_buf[:, 1]
        cart_pos = self.obs_buf[:, 0]

        self.gt_rew_buf, self.reset_buf[:], self.consecutive_successes[:] = compute_success(
            pole_angle, pole_vel, cart_vel, cart_pos, self.cumulative_rotation,
            self.reset_dist, self.reset_buf, self.consecutive_successes, self.progress_buf, self.max_episode_length
        )
        self.extras['gt_reward'] = self.gt_rew_buf.mean()
        self.extras['consecutive_successes'] = self.consecutive_successes.mean() 
        self.extras['cumulative_rotation'] = self.cumulative_rotation.mean() 

    def compute_observations(self, env_ids=None):
        if env_ids is None:
            env_ids = np.arange(self.num_envs)

        self.gym.refresh_dof_state_tensor(self.sim)

        current_pole_angle = self.dof_pos[env_ids, 1].squeeze()
        
        angle_diff = current_pole_angle - self.prev_pole_angle[env_ids]
        angle_diff = torch.where(angle_diff > np.pi, angle_diff - 2*np.pi, angle_diff)
        angle_diff = torch.where(angle_diff < -np.pi, angle_diff + 2*np.pi, angle_diff)
        
        self.cumulative_rotation[env_ids] += angle_diff
        self.prev_pole_angle[env_ids] = current_pole_angle

        self.obs_buf[env_ids, 0] = self.dof_pos[env_ids, 0].squeeze()
        self.obs_buf[env_ids, 1] = self.dof_vel[env_ids, 0].squeeze()
        self.obs_buf[env_ids, 2] = self.dof_pos[env_ids, 1].squeeze()
        self.obs_buf[env_ids, 3] = self.dof_vel[env_ids, 1].squeeze()

        return self.obs_buf

    def reset_idx(self, env_ids):
        positions = 0.2 * (torch.rand((len(env_ids), self.num_dof), device=self.device) - 0.5)
        velocities = 0.5 * (torch.rand((len(env_ids), self.num_dof), device=self.device) - 0.5)

        self.dof_pos[env_ids, :] = positions[:]
        self.dof_vel[env_ids, :] = velocities[:]

        self.cumulative_rotation[env_ids] = 0.0
        self.prev_pole_angle[env_ids] = self.dof_pos[env_ids, 1].squeeze()

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0

    def pre_physics_step(self, actions):
        actions_tensor = torch.zeros(self.num_envs * self.num_dof, device=self.device, dtype=torch.float)
        actions_tensor[::self.num_dof] = actions.to(self.device).squeeze() * self.max_push_effort
        forces = gymtorch.unwrap_tensor(actions_tensor)
        self.gym.set_dof_actuation_force_tensor(self.sim, forces)

    def post_physics_step(self):
        self.progress_buf += 1

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        self.compute_observations()
        self.compute_reward()



@torch.jit.script
def compute_success(pole_angle, pole_vel, cart_vel, cart_pos, cumulative_rotation,
                                reset_dist, reset_buf, consecutive_successes, progress_buf, max_episode_length):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, float, Tensor, Tensor, Tensor, float) -> Tuple[Tensor, Tensor, Tensor]

    spin_reward = torch.abs(pole_vel) * 0.1
    
    rotation_bonus = cumulative_rotation * 0.05
    
    
    reward = rotation_bonus + spin_reward

    reset = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), reset_buf)

    if reset.sum() > 0:
        consecutive_successes = (torch.abs(cumulative_rotation) * reset).sum() / reset.sum()
    else:
        consecutive_successes = torch.zeros_like(consecutive_successes).mean()
    
    return reward, reset, consecutive_successes

from typing import Tuple, Dict
import math
import torch
from torch import Tensor
@torch.jit.script
def compute_reward(cumulative_rotation: torch.Tensor, dof_vel: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Reward function for CartpoleSpin task:
    - Encourage the pole to spin full rotations consecutively (reward proportional to absolute cumulative rotation)
    - Encourage slow spinning (penalize high angular velocity of the pole's second DOF)

    Args:
        cumulative_rotation: Tensor of shape (num_envs,) -- accumulated rotation angle of the pole (radians)
        dof_vel: Tensor of shape (num_envs, 2) -- velocities of the two degrees of freedom (only second DOF velocity used)

    Returns:
        reward: Tensor of shape (num_envs,) total reward for each environment
        info_dict: dict with individual reward components
    """
    device = cumulative_rotation.device
    # Temperature parameters for scaling
    rotation_temp: float = 1.0  # adjust to control sharpness of rotation reward
    velocity_temp: float = 1.0  # adjust to control sharpness of velocity penalty

    # Reward component 1: encourage absolute cumulative rotation (full rotations)
    # Normalize by (2*pi) to count number of full rotations approximately
    rotations = torch.abs(cumulative_rotation) / (2 * 3.141592653589793)
    # Apply an exponential transformation to keep reward in (0,1)
    rotation_reward = torch.exp(-rotation_temp * (1.0 - torch.clamp(rotations, max=1.0)))

    # Reward component 2: penalize high angular velocity of pole (dof index 1 velocity)
    pole_vel = dof_vel[:, 1]
    # Desired slow spinning means small absolute velocity; penalize large velocities
    # Scale and exponentiate negative squared velocity to produce smooth penalty
    velocity_reward = torch.exp(-velocity_temp * pole_vel * pole_vel)

    # Combine rewards multiplicatively so that both are required
    reward = rotation_reward * velocity_reward

    info = {
        "rotation_reward": rotation_reward,
        "velocity_reward": velocity_reward,
        "total_reward": reward,
    }

    return reward, info
