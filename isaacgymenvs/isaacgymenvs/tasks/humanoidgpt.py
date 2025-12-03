from typing import Tuple, Dict
import math
import torch
from torch import Tensor

import numpy as np
import os
import torch

from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym.torch_utils import *

from isaacgymenvs.utils.torch_jit_utils import *
from isaacgymenvs.tasks.base.vec_task import VecTask

class HumanoidGPT(VecTask):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        self.cfg = cfg
        
        self.randomization_params = self.cfg["task"]["randomization_params"]
        self.randomize = self.cfg["task"]["randomize"]
        self.dof_vel_scale = self.cfg["env"]["dofVelocityScale"]
        self.angular_velocity_scale = self.cfg["env"].get("angularVelocityScale", 0.1)
        self.contact_force_scale = self.cfg["env"]["contactForceScale"]
        self.power_scale = self.cfg["env"]["powerScale"]
        self.heading_weight = self.cfg["env"]["headingWeight"]
        self.up_weight = self.cfg["env"]["upWeight"]
        self.actions_cost_scale = self.cfg["env"]["actionsCost"]
        self.energy_cost_scale = self.cfg["env"]["energyCost"]
        self.joints_at_limit_cost_scale = self.cfg["env"]["jointsAtLimitCost"]
        self.death_cost = self.cfg["env"]["deathCost"]
        self.termination_height = self.cfg["env"]["terminationHeight"]

        self.debug_viz = self.cfg["env"]["enableDebugVis"]
        self.plane_static_friction = self.cfg["env"]["plane"]["staticFriction"]
        self.plane_dynamic_friction = self.cfg["env"]["plane"]["dynamicFriction"]
        self.plane_restitution = self.cfg["env"]["plane"]["restitution"]

        self.max_episode_length = self.cfg["env"]["episodeLength"]

        self.cfg["env"]["numObservations"] = 108
        self.cfg["env"]["numActions"] = 21

        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)

        if self.viewer != None:
            cam_pos = gymapi.Vec3(50.0, 25.0, 2.4)
            cam_target = gymapi.Vec3(45.0, 25.0, 0.0)
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim)

        sensors_per_env = 2
        self.vec_sensor_tensor = gymtorch.wrap_tensor(sensor_tensor).view(self.num_envs, sensors_per_env * 6)

        dof_force_tensor = self.gym.acquire_dof_force_tensor(self.sim)
        self.dof_force_tensor = gymtorch.wrap_tensor(dof_force_tensor).view(self.num_envs, self.num_dof)

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)

        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.initial_root_states = self.root_states.clone()
        self.initial_root_states[:, 7:13] = 0

        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.initial_dof_pos = torch.zeros_like(self.dof_pos, device=self.device, dtype=torch.float)
        zero_tensor = torch.tensor([0.0], device=self.device)
        self.initial_dof_pos = torch.where(self.dof_limits_lower > zero_tensor, self.dof_limits_lower,
                                           torch.where(self.dof_limits_upper < zero_tensor, self.dof_limits_upper, self.initial_dof_pos))
        self.initial_dof_vel = torch.zeros_like(self.dof_vel, device=self.device, dtype=torch.float)

        self.up_vec = to_torch(get_axis_params(1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.heading_vec = to_torch([1, 0, 0], device=self.device).repeat((self.num_envs, 1))
        self.inv_start_rot = quat_conjugate(self.start_rotation).repeat((self.num_envs, 1))

        self.basis_vec0 = self.heading_vec.clone()
        self.basis_vec1 = self.up_vec.clone()

        self.targets = to_torch([1000, 0, 0], device=self.device).repeat((self.num_envs, 1))
        self.target_dirs = to_torch([1, 0, 0], device=self.device).repeat((self.num_envs, 1))
        self.dt = self.cfg["sim"]["dt"]
        self.potentials = to_torch([-1000./self.dt], device=self.device).repeat(self.num_envs)
        self.prev_potentials = self.potentials.clone()

        self.consecutive_successes = torch.zeros(1, dtype=torch.float, device=self.device)

    def create_sim(self):
        self.up_axis_idx = 2 # index of up axis: Y=1, Z=2
        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)

        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

        if self.randomize:
            self.apply_randomizations(self.randomization_params)

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.plane_static_friction
        plane_params.dynamic_friction = self.plane_dynamic_friction
        plane_params.restitution = self.plane_restitution
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../assets')
        asset_file = "mjcf/nv_humanoid.xml"

        if "asset" in self.cfg["env"]:
            asset_file = self.cfg["env"]["asset"].get("assetFileName", asset_file)

        asset_path = os.path.join(asset_root, asset_file)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.angular_damping = 0.01
        asset_options.max_angular_velocity = 100.0
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        humanoid_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)

        actuator_props = self.gym.get_asset_actuator_properties(humanoid_asset)
        motor_efforts = [prop.motor_effort for prop in actuator_props]

        right_foot_idx = self.gym.find_asset_rigid_body_index(humanoid_asset, "right_foot")
        left_foot_idx = self.gym.find_asset_rigid_body_index(humanoid_asset, "left_foot")
        sensor_pose = gymapi.Transform()
        self.gym.create_asset_force_sensor(humanoid_asset, right_foot_idx, sensor_pose)
        self.gym.create_asset_force_sensor(humanoid_asset, left_foot_idx, sensor_pose)

        self.max_motor_effort = max(motor_efforts)
        self.motor_efforts = to_torch(motor_efforts, device=self.device)

        self.torso_index = 0
        self.num_bodies = self.gym.get_asset_rigid_body_count(humanoid_asset)
        self.num_dof = self.gym.get_asset_dof_count(humanoid_asset)
        self.num_joints = self.gym.get_asset_joint_count(humanoid_asset)

        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*get_axis_params(1.34, self.up_axis_idx))
        start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        self.start_rotation = torch.tensor([start_pose.r.x, start_pose.r.y, start_pose.r.z, start_pose.r.w], device=self.device)

        self.humanoid_handles = []
        self.envs = []
        self.dof_limits_lower = []
        self.dof_limits_upper = []

        for i in range(self.num_envs):
            env_ptr = self.gym.create_env(
                self.sim, lower, upper, num_per_row
            )
            handle = self.gym.create_actor(env_ptr, humanoid_asset, start_pose, "humanoid", i, 0, 0)

            self.gym.enable_actor_dof_force_sensors(env_ptr, handle)

            for j in range(self.num_bodies):
                self.gym.set_rigid_body_color(
                    env_ptr, handle, j, gymapi.MESH_VISUAL, gymapi.Vec3(0.97, 0.38, 0.06))

            self.envs.append(env_ptr)
            self.humanoid_handles.append(handle)

        dof_prop = self.gym.get_actor_dof_properties(env_ptr, handle)
        for j in range(self.num_dof):
            if dof_prop['lower'][j] > dof_prop['upper'][j]:
                self.dof_limits_lower.append(dof_prop['upper'][j])
                self.dof_limits_upper.append(dof_prop['lower'][j])
            else:
                self.dof_limits_lower.append(dof_prop['lower'][j])
                self.dof_limits_upper.append(dof_prop['upper'][j])

        self.dof_limits_lower = to_torch(self.dof_limits_lower, device=self.device)
        self.dof_limits_upper = to_torch(self.dof_limits_upper, device=self.device)

        self.extremities = to_torch([5, 8], device=self.device, dtype=torch.long)

    def compute_reward(self, actions):
        self.rew_buf[:], self.rew_dict = compute_reward(self.root_states, self.heading_vec, self.up_vec, self.potentials, self.prev_potentials, self.vec_sensor_tensor, self.actions)
        self.extras['gpt_reward'] = self.rew_buf.mean()
        for rew_state in self.rew_dict: self.extras[rew_state] = self.rew_dict[rew_state].mean()
        self.gt_rew_buf, _ = compute_success(self.root_states, self.heading_vec, self.up_vec, self.potentials, self.prev_potentials, self.vec_sensor_tensor, self.actions)
        self.consecutive_successes[:] = self.gt_rew_buf.mean()
        self.extras['gt_reward'] = self.gt_rew_buf.mean()
        self.extras['consecutive_successes'] = self.consecutive_successes.mean()

    def compute_observations(self):
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)

        self.gym.refresh_force_sensor_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)
        self.obs_buf[:], self.potentials[:], self.prev_potentials[:], self.up_vec[:], self.heading_vec[:] = compute_humanoid_observations(
            self.obs_buf, self.root_states, self.targets, self.potentials,
            self.inv_start_rot, self.dof_pos, self.dof_vel, self.dof_force_tensor,
            self.dof_limits_lower, self.dof_limits_upper, self.dof_vel_scale,
            self.vec_sensor_tensor, self.actions, self.dt, self.contact_force_scale, self.angular_velocity_scale,
            self.basis_vec0, self.basis_vec1)

    def reset_idx(self, env_ids):
        if self.randomize:
            self.apply_randomizations(self.randomization_params)

        positions = torch_rand_float(-0.2, 0.2, (len(env_ids), self.num_dof), device=self.device)
        velocities = torch_rand_float(-0.1, 0.1, (len(env_ids), self.num_dof), device=self.device)

        self.dof_pos[env_ids] = tensor_clamp(self.initial_dof_pos[env_ids] + positions, self.dof_limits_lower, self.dof_limits_upper)
        self.dof_vel[env_ids] = velocities

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.initial_root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        to_target = self.targets[env_ids] - self.initial_root_states[env_ids, 0:3]
        to_target[:, self.up_axis_idx] = 0
        self.prev_potentials[env_ids] = -torch.norm(to_target, p=2, dim=-1) / self.dt
        self.potentials[env_ids] = self.prev_potentials[env_ids].clone()

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0

    def pre_physics_step(self, actions):
        self.actions = actions.to(self.device).clone()
        forces = self.actions * self.motor_efforts.unsqueeze(0) * self.power_scale
        force_tensor = gymtorch.unwrap_tensor(forces)
        self.gym.set_dof_actuation_force_tensor(self.sim, force_tensor)

    def post_physics_step(self):
        self.progress_buf += 1
        self.randomize_buf += 1

        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        self.compute_observations()
        self.compute_reward(self.actions)

        if self.viewer and self.debug_viz:
            self.gym.clear_lines(self.viewer)

            points = []
            colors = []
            for i in range(self.num_envs):
                origin = self.gym.get_env_origin(self.envs[i])
                pose = self.root_states[:, 0:3][i].cpu().numpy()
                glob_pos = gymapi.Vec3(origin.x + pose[0], origin.y + pose[1], origin.z + pose[2])
                points.append([glob_pos.x, glob_pos.y, glob_pos.z, glob_pos.x + 4 * self.heading_vec[i, 0].cpu().numpy(),
                               glob_pos.y + 4 * self.heading_vec[i, 1].cpu().numpy(),
                               glob_pos.z + 4 * self.heading_vec[i, 2].cpu().numpy()])
                colors.append([0.97, 0.1, 0.06])
                points.append([glob_pos.x, glob_pos.y, glob_pos.z, glob_pos.x + 4 * self.up_vec[i, 0].cpu().numpy(), glob_pos.y + 4 * self.up_vec[i, 1].cpu().numpy(),
                               glob_pos.z + 4 * self.up_vec[i, 2].cpu().numpy()])
                colors.append([0.05, 0.99, 0.04])

            self.gym.add_lines(self.viewer, None, self.num_envs * 2, points, colors)

@torch.jit.script
def compute_humanoid_observations(obs_buf, root_states, targets, potentials, inv_start_rot, dof_pos, dof_vel,
                                  dof_force, dof_limits_lower, dof_limits_upper, dof_vel_scale,
                                  sensor_force_torques, actions, dt, contact_force_scale, angular_velocity_scale,
                                  basis_vec0, basis_vec1):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, float, Tensor, Tensor, float, float, float, Tensor, Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]

    torso_position = root_states[:, 0:3]
    torso_rotation = root_states[:, 3:7]
    velocity = root_states[:, 7:10]
    ang_velocity = root_states[:, 10:13]

    to_target = targets - torso_position
    to_target[:, 2] = 0

    prev_potentials_new = potentials.clone()
    potentials = -torch.norm(to_target, p=2, dim=-1) / dt

    torso_quat, up_proj, heading_proj, up_vec, heading_vec = compute_heading_and_up(
        torso_rotation, inv_start_rot, to_target, basis_vec0, basis_vec1, 2)

    vel_loc, angvel_loc, roll, pitch, yaw, angle_to_target = compute_rot(
        torso_quat, velocity, ang_velocity, targets, torso_position)

    roll = normalize_angle(roll).unsqueeze(-1)
    yaw = normalize_angle(yaw).unsqueeze(-1)
    angle_to_target = normalize_angle(angle_to_target).unsqueeze(-1)
    dof_pos_scaled = unscale(dof_pos, dof_limits_lower, dof_limits_upper)

    obs = torch.cat((torso_position[:, 2].view(-1, 1), vel_loc, angvel_loc * angular_velocity_scale,
                     yaw, roll, angle_to_target, up_proj.unsqueeze(-1), heading_proj.unsqueeze(-1),
                     dof_pos_scaled, dof_vel * dof_vel_scale, dof_force * contact_force_scale,
                     sensor_force_torques.view(-1, 12) * contact_force_scale, actions), dim=-1)

    return obs, potentials, prev_potentials_new, up_vec, heading_vec

@torch.jit.script
def compute_success(
    root_states: torch.Tensor,
    heading_vec: torch.Tensor,
    up_vec: torch.Tensor,
    potentials: torch.Tensor,
    prev_potentials: torch.Tensor,
    vec_sensor_tensor: torch.Tensor,
    actions: torch.Tensor
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Ensure unit heading vector for safe projections
    heading_norm = torch.norm(heading_vec, p=2, dim=-1) + 1e-6
    heading_unit = heading_vec / heading_norm.unsqueeze(-1)

    # Extract COM/world velocity
    velocity = root_states[:, 7:10]

    # Forward speed along the heading direction
    forward_speed = torch.sum(velocity * heading_unit, dim=-1)

    # Lateral speed (perpendicular to heading)
    v_perp = velocity - forward_speed.unsqueeze(-1) * heading_unit
    lateral_speed = torch.norm(v_perp, p=2, dim=-1)

    # Uprightness from up vector (projection onto world up z-axis)
    up_proj = up_vec[:, 2]

    # Contact force magnitude aggregation over all sensors/axes
    abs_sft = torch.abs(vec_sensor_tensor)
    if abs_sft.dim() == 2:
        # Shape: (num_envs, K)
        contact_force_mag = torch.sum(abs_sft, dim=1)
    elif abs_sft.dim() == 3:
        # Shape: (num_envs, num_sensors, 6)
        contact_force_mag = torch.sum(abs_sft, dim=2)
        contact_force_mag = torch.sum(contact_force_mag, dim=1)
    else:
        # Fallback: flatten non-batch dims
        contact_force_mag = abs_sft.view(abs_sft.size(0), -1).sum(dim=1)

    # Temperature parameters for transformed components (per requirement)
    temp_progress = torch.tensor(1.0, device=potentials.device)
    temp_speed = torch.tensor(1.5, device=root_states.device)
    temp_lateral = torch.tensor(1.0, device=root_states.device)
    temp_upright = torch.tensor(0.25, device=up_vec.device)
    temp_contact = torch.tensor(300.0, device=vec_sensor_tensor.device)
    temp_action = torch.tensor(0.5, device=actions.device)

    # Component computations with transformations
    # Progress: change in potential (positive when moving toward target)
    delta_progress = potentials - prev_potentials
    progress_raw = torch.exp(delta_progress / (temp_progress + 1e-6)) - 1.0
    progress_raw = torch.clamp(progress_raw, min=-1.0, max=1.0)

    # Forward speed: encourage strong forward motion; only positive contribution
    speed_raw = torch.tanh(forward_speed / (temp_speed + 1e-6))
    speed_raw = torch.clamp(speed_raw, min=0.0, max=1.0)

    # Lateral penalty: discourage sideways motion
    lateral_penalty_raw = torch.tanh(lateral_speed / (temp_lateral + 1e-6))

    # Uprightness: keep torso upright
    upright_raw = torch.exp((up_proj - 1.0) / (temp_upright + 1e-6))
    upright_raw = torch.clamp(upright_raw, min=0.0, max=1.0)

    # Contact force reward: encourage strong ground reaction forces while moving forward
    contact_raw = torch.tanh(contact_force_mag / (temp_contact + 1e-6)) * speed_raw

    # Action penalty: discourage excessively large actions while still allowing power
    action_mag = torch.norm(actions, p=2, dim=-1)
    action_penalty_raw = torch.tanh(action_mag / (temp_action + 1e-6))

    # Weights for combining components
    w_progress = torch.tensor(1.0, device=potentials.device)
    w_speed = torch.tensor(2.5, device=potentials.device)
    w_upright = torch.tensor(0.5, device=potentials.device)
    w_lateral = torch.tensor(0.5, device=potentials.device)  # penalty (negative contribution)
    w_contact = torch.tensor(0.3, device=potentials.device)
    w_action = torch.tensor(0.05, device=potentials.device)  # penalty (negative contribution)

    # Weighted components
    progress = w_progress * progress_raw
    speed = w_speed * speed_raw
    upright = w_upright * upright_raw
    lateral_penalty = -w_lateral * lateral_penalty_raw
    contact = w_contact * contact_raw
    action_penalty = -w_action * action_penalty_raw

    # Total reward
    total_reward = progress + speed + upright + lateral_penalty + contact + action_penalty

    components: Dict[str, torch.Tensor] = {
        "progress": progress,
        "speed": speed,
        "upright": upright,
        "lateral_penalty": lateral_penalty,
        "contact": contact,
        "action_penalty": action_penalty,
    }
    return total_reward, components

from typing import Tuple, Dict
import math
import torch
from torch import Tensor
@torch.jit.script
def compute_reward(
    root_states: torch.Tensor,
    heading_vec: torch.Tensor,
    up_vec: torch.Tensor,
    potentials: torch.Tensor,
    prev_potentials: torch.Tensor,
    vec_sensor_tensor: torch.Tensor,
    actions: torch.Tensor
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Ensure unit heading vector for safe projections
    heading_norm = torch.norm(heading_vec, p=2, dim=-1) + 1e-6
    heading_unit = heading_vec / heading_norm.unsqueeze(-1)

    # Extract COM/world velocity
    velocity = root_states[:, 7:10]

    # Forward speed along the heading direction
    forward_speed = torch.sum(velocity * heading_unit, dim=-1)

    # Lateral speed (perpendicular to heading)
    v_perp = velocity - forward_speed.unsqueeze(-1) * heading_unit
    lateral_speed = torch.norm(v_perp, p=2, dim=-1)

    # Uprightness from up vector (projection onto world up z-axis)
    up_proj = up_vec[:, 2]

    # Contact force magnitude aggregation over all sensors/axes
    abs_sft = torch.abs(vec_sensor_tensor)
    if abs_sft.dim() == 2:
        # Shape: (num_envs, K)
        contact_force_mag = torch.sum(abs_sft, dim=1)
    elif abs_sft.dim() == 3:
        # Shape: (num_envs, num_sensors, 6)
        contact_force_mag = torch.sum(abs_sft, dim=2)
        contact_force_mag = torch.sum(contact_force_mag, dim=1)
    else:
        # Fallback: flatten non-batch dims
        contact_force_mag = abs_sft.view(abs_sft.size(0), -1).sum(dim=1)

    # Temperature parameters for transformed components (per requirement)
    temp_progress = torch.tensor(1.0, device=potentials.device)
    temp_speed = torch.tensor(1.5, device=root_states.device)
    temp_lateral = torch.tensor(1.0, device=root_states.device)
    temp_upright = torch.tensor(0.25, device=up_vec.device)
    temp_contact = torch.tensor(300.0, device=vec_sensor_tensor.device)
    temp_action = torch.tensor(0.5, device=actions.device)

    # Component computations with transformations
    # Progress: change in potential (positive when moving toward target)
    delta_progress = potentials - prev_potentials
    progress_raw = torch.exp(delta_progress / (temp_progress + 1e-6)) - 1.0
    progress_raw = torch.clamp(progress_raw, min=-1.0, max=1.0)

    # Forward speed: encourage strong forward motion; only positive contribution
    speed_raw = torch.tanh(forward_speed / (temp_speed + 1e-6))
    speed_raw = torch.clamp(speed_raw, min=0.0, max=1.0)

    # Lateral penalty: discourage sideways motion
    lateral_penalty_raw = torch.tanh(lateral_speed / (temp_lateral + 1e-6))

    # Uprightness: keep torso upright
    upright_raw = torch.exp((up_proj - 1.0) / (temp_upright + 1e-6))
    upright_raw = torch.clamp(upright_raw, min=0.0, max=1.0)

    # Contact force reward: encourage strong ground reaction forces while moving forward
    contact_raw = torch.tanh(contact_force_mag / (temp_contact + 1e-6)) * speed_raw

    # Action penalty: discourage excessively large actions while still allowing power
    action_mag = torch.norm(actions, p=2, dim=-1)
    action_penalty_raw = torch.tanh(action_mag / (temp_action + 1e-6))

    # Weights for combining components
    w_progress = torch.tensor(1.0, device=potentials.device)
    w_speed = torch.tensor(2.5, device=potentials.device)
    w_upright = torch.tensor(0.5, device=potentials.device)
    w_lateral = torch.tensor(0.5, device=potentials.device)  # penalty (negative contribution)
    w_contact = torch.tensor(0.3, device=potentials.device)
    w_action = torch.tensor(0.05, device=potentials.device)  # penalty (negative contribution)

    # Weighted components
    progress = w_progress * progress_raw
    speed = w_speed * speed_raw
    upright = w_upright * upright_raw
    lateral_penalty = -w_lateral * lateral_penalty_raw
    contact = w_contact * contact_raw
    action_penalty = -w_action * action_penalty_raw

    # Total reward
    total_reward = progress + speed + upright + lateral_penalty + contact + action_penalty

    components: Dict[str, torch.Tensor] = {
        "progress": progress,
        "speed": speed,
        "upright": upright,
        "lateral_penalty": lateral_penalty,
        "contact": contact,
        "action_penalty": action_penalty,
    }
    return total_reward, components
