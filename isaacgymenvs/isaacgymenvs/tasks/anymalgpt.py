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

from isaacgymenvs.tasks.base.vec_task import VecTask

from typing import Tuple, Dict

class AnymalGPT(VecTask):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):

        self.cfg = cfg
        
        self.lin_vel_scale = self.cfg["env"]["learn"]["linearVelocityScale"]
        self.ang_vel_scale = self.cfg["env"]["learn"]["angularVelocityScale"]
        self.dof_pos_scale = self.cfg["env"]["learn"]["dofPositionScale"]
        self.dof_vel_scale = self.cfg["env"]["learn"]["dofVelocityScale"]
        self.action_scale = self.cfg["env"]["control"]["actionScale"]

        self.rew_scales = {}
        self.rew_scales["lin_vel_xy"] = self.cfg["env"]["learn"]["linearVelocityXYRewardScale"]
        self.rew_scales["ang_vel_z"] = self.cfg["env"]["learn"]["angularVelocityZRewardScale"]
        self.rew_scales["torque"] = self.cfg["env"]["learn"]["torqueRewardScale"]

        self.randomization_params = self.cfg["task"]["randomization_params"]
        self.randomize = self.cfg["task"]["randomize"]

        self.command_x_range = self.cfg["env"]["randomCommandVelocityRanges"]["linear_x"]
        self.command_y_range = self.cfg["env"]["randomCommandVelocityRanges"]["linear_y"]
        self.command_yaw_range = self.cfg["env"]["randomCommandVelocityRanges"]["yaw"]

        self.plane_static_friction = self.cfg["env"]["plane"]["staticFriction"]
        self.plane_dynamic_friction = self.cfg["env"]["plane"]["dynamicFriction"]
        self.plane_restitution = self.cfg["env"]["plane"]["restitution"]

        pos = self.cfg["env"]["baseInitState"]["pos"]
        rot = self.cfg["env"]["baseInitState"]["rot"]
        v_lin = self.cfg["env"]["baseInitState"]["vLinear"]
        v_ang = self.cfg["env"]["baseInitState"]["vAngular"]
        state = pos + rot + v_lin + v_ang

        self.base_init_state = state

        self.named_default_joint_angles = self.cfg["env"]["defaultJointAngles"]

        self.cfg["env"]["numObservations"] = 48
        self.cfg["env"]["numActions"] = 12

        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)

        self.dt = self.sim_params.dt
        self.max_episode_length_s = self.cfg["env"]["learn"]["episodeLength_s"]
        self.max_episode_length = int(self.max_episode_length_s / self.dt + 0.5)
        self.Kp = self.cfg["env"]["control"]["stiffness"]
        self.Kd = self.cfg["env"]["control"]["damping"]

        for key in self.rew_scales.keys():
            self.rew_scales[key] *= self.dt

        if self.viewer != None:
            p = self.cfg["env"]["viewer"]["pos"]
            lookat = self.cfg["env"]["viewer"]["lookat"]
            cam_pos = gymapi.Vec3(p[0], p[1], p[2])
            cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        torques = self.gym.acquire_dof_force_tensor(self.sim)

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)

        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3)  # shape: num_envs, num_bodies, xyz axis
        self.torques = gymtorch.wrap_tensor(torques).view(self.num_envs, self.num_dof)

        self.commands = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.commands_y = self.commands.view(self.num_envs, 3)[..., 1]
        self.commands_x = self.commands.view(self.num_envs, 3)[..., 0]
        self.commands_yaw = self.commands.view(self.num_envs, 3)[..., 2]
        self.default_dof_pos = torch.zeros_like(self.dof_pos, dtype=torch.float, device=self.device, requires_grad=False)

        for i in range(self.cfg["env"]["numActions"]):
            name = self.dof_names[i]
            angle = self.named_default_joint_angles[name]
            self.default_dof_pos[:, i] = angle

        self.extras = {}
        self.initial_root_states = self.root_states.clone()
        self.initial_root_states[:] = to_torch(self.base_init_state, device=self.device, requires_grad=False)
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        
        self.consecutive_successes = torch.zeros(1, dtype=torch.float, device=self.device)
        self.reset_idx(torch.arange(self.num_envs, device=self.device))

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
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):
        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../assets')
        asset_file = "urdf/anymal_c/urdf/anymal.urdf"

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        asset_options.collapse_fixed_joints = True
        asset_options.replace_cylinder_with_capsule = True
        asset_options.flip_visual_attachments = True
        asset_options.fix_base_link = self.cfg["env"]["urdfAsset"]["fixBaseLink"]
        asset_options.density = 0.001
        asset_options.angular_damping = 0.0
        asset_options.linear_damping = 0.0
        asset_options.armature = 0.0
        asset_options.thickness = 0.01
        asset_options.disable_gravity = False

        anymal_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(anymal_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(anymal_asset)

        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        body_names = self.gym.get_asset_rigid_body_names(anymal_asset)
        self.dof_names = self.gym.get_asset_dof_names(anymal_asset)
        extremity_name = "SHANK" if asset_options.collapse_fixed_joints else "FOOT"
        feet_names = [s for s in body_names if extremity_name in s]
        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        knee_names = [s for s in body_names if "THIGH" in s]
        self.knee_indices = torch.zeros(len(knee_names), dtype=torch.long, device=self.device, requires_grad=False)
        self.base_index = 0

        dof_props = self.gym.get_asset_dof_properties(anymal_asset)
        for i in range(self.num_dof):
            dof_props['driveMode'][i] = gymapi.DOF_MODE_POS
            dof_props['stiffness'][i] = self.cfg["env"]["control"]["stiffness"] #self.Kp
            dof_props['damping'][i] = self.cfg["env"]["control"]["damping"] #self.Kd

        env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)
        self.anymal_handles = []
        self.envs = []

        for i in range(self.num_envs):
            env_ptr = self.gym.create_env(self.sim, env_lower, env_upper, num_per_row)
            anymal_handle = self.gym.create_actor(env_ptr, anymal_asset, start_pose, "anymal", i, 1, 0)
            self.gym.set_actor_dof_properties(env_ptr, anymal_handle, dof_props)
            self.gym.enable_actor_dof_force_sensors(env_ptr, anymal_handle)
            self.envs.append(env_ptr)
            self.anymal_handles.append(anymal_handle)

        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.anymal_handles[0], feet_names[i])
        for i in range(len(knee_names)):
            self.knee_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.anymal_handles[0], knee_names[i])

        self.base_index = self.gym.find_actor_rigid_body_handle(self.envs[0], self.anymal_handles[0], "base")

    def pre_physics_step(self, actions):
        self.actions = actions.clone().to(self.device)
        targets = self.action_scale * self.actions + self.default_dof_pos
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(targets))

    def post_physics_step(self):
        self.progress_buf += 1

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        self.compute_observations()
        self.compute_reward(self.actions)

    def compute_reward(self, actions):
        self.rew_buf[:], self.rew_dict = compute_reward(self.root_states, self.commands, self.default_dof_pos, self.dof_pos, self.dof_vel, self.gravity_vec, self.actions, self.lin_vel_scale, self.ang_vel_scale, self.dof_pos_scale, self.dof_vel_scale)
        self.extras['gpt_reward'] = self.rew_buf.mean()
        for rew_state in self.rew_dict: self.extras[rew_state] = self.rew_dict[rew_state].mean()
        self.gt_rew_buf, _ = compute_success(self.root_states, self.commands, self.dof_pos, self.default_dof_pos, self.dof_vel, self.gravity_vec, self.actions)
        self.consecutive_successes[:] = self.gt_rew_buf.mean()
        self.extras['gt_reward'] = self.gt_rew_buf.mean()
        self.extras['consecutive_successes'] = self.consecutive_successes.mean() 

    def compute_observations(self):
        self.gym.refresh_dof_state_tensor(self.sim)  # done in step
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)

        self.obs_buf[:] = compute_anymal_observations(  # tensors
                                                        self.root_states,
                                                        self.commands,
                                                        self.dof_pos,
                                                        self.default_dof_pos,
                                                        self.dof_vel,
                                                        self.gravity_vec,
                                                        self.actions,
                                                        self.lin_vel_scale,
                                                        self.ang_vel_scale,
                                                        self.dof_pos_scale,
                                                        self.dof_vel_scale
        )

    def reset_idx(self, env_ids):
        if self.randomize:
            self.apply_randomizations(self.randomization_params)

        positions_offset = torch_rand_float(0.5, 1.5, (len(env_ids), self.num_dof), device=self.device)
        velocities = torch_rand_float(-0.1, 0.1, (len(env_ids), self.num_dof), device=self.device)

        self.dof_pos[env_ids] = self.default_dof_pos[env_ids] * positions_offset
        self.dof_vel[env_ids] = velocities

        env_ids_int32 = env_ids.to(dtype=torch.int32)

        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.initial_root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        self.commands_x[env_ids] = torch_rand_float(self.command_x_range[0], self.command_x_range[1], (len(env_ids), 1), device=self.device).squeeze()
        self.commands_y[env_ids] = torch_rand_float(self.command_y_range[0], self.command_y_range[1], (len(env_ids), 1), device=self.device).squeeze()
        self.commands_yaw[env_ids] = torch_rand_float(self.command_yaw_range[0], self.command_yaw_range[1], (len(env_ids), 1), device=self.device).squeeze()

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1

@torch.jit.script
def compute_anymal_observations(root_states,
                                commands,
                                dof_pos,
                                default_dof_pos,
                                dof_vel,
                                gravity_vec,
                                actions,
                                lin_vel_scale,
                                ang_vel_scale,
                                dof_pos_scale,
                                dof_vel_scale
                                ):

    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, float, float, float, float) -> Tensor
    base_quat = root_states[:, 3:7]
    base_lin_vel = quat_rotate_inverse(base_quat, root_states[:, 7:10]) * lin_vel_scale
    base_ang_vel = quat_rotate_inverse(base_quat, root_states[:, 10:13]) * ang_vel_scale
    projected_gravity = quat_rotate(base_quat, gravity_vec)
    dof_pos_scaled = (dof_pos - default_dof_pos) * dof_pos_scale

    commands_scaled = commands*torch.tensor([lin_vel_scale, lin_vel_scale, ang_vel_scale], requires_grad=False, device=commands.device)

    obs = torch.cat((base_lin_vel,
                     base_ang_vel,
                     projected_gravity,
                     commands_scaled,
                     dof_pos_scaled,
                     dof_vel*dof_vel_scale,
                     actions
                     ), dim=-1)

    return obs

@torch.jit.script
def compute_success(
    root_states: torch.Tensor,
    commands: torch.Tensor,
    dof_pos: torch.Tensor,
    default_dof_pos: torch.Tensor,
    dof_vel: torch.Tensor,
    gravity_vec: torch.Tensor,
    actions: torch.Tensor
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Temperatures for transformed components (each component has its own temperature)
    temp_vx = 2.0
    temp_vy = 2.0
    temp_yaw = 1.5
    temp_upright = 8.0
    temp_posture = 0.4
    temp_joint_vel = 0.08
    temp_action = 0.04
    temp_stable_rate = 0.5

    # Weights for each component
    w_vx = 3.0
    w_vy = 1.0
    w_yaw = 1.0
    w_upright = 1.5
    w_posture = 0.35
    w_joint_vel = 0.25
    w_action = 0.05
    w_stable_rate = 0.8
    w_forward = 0.5

    base_quat = root_states[:, 3:7]
    base_lin_vel_world = root_states[:, 7:10]
    base_ang_vel_world = root_states[:, 10:13]

    # Express base velocities in the robot's body frame for meaningful tracking against commands
    base_lin_vel_body = quat_rotate_inverse(base_quat, base_lin_vel_world)
    base_ang_vel_body = quat_rotate_inverse(base_quat, base_ang_vel_world)

    # Gravity projected into body frame; upright when x,y components ~ 0 and z ~ -|g|
    projected_gravity = quat_rotate(base_quat, gravity_vec)

    # Velocity tracking rewards (encourage cautious forward walking and minimal lateral slip)
    vx_err = base_lin_vel_body[:, 0] - commands[:, 0]
    vy_err = base_lin_vel_body[:, 1] - commands[:, 1]
    yaw_err = base_ang_vel_body[:, 2] - commands[:, 2]

    r_vx = torch.exp(-temp_vx * (vx_err * vx_err))
    r_vy = torch.exp(-temp_vy * (vy_err * vy_err))
    r_yaw = torch.exp(-temp_yaw * (yaw_err * yaw_err))

    # Uprightness: penalize tilt magnitude via gravity x/y components in body frame
    tilt_mag = projected_gravity[:, 0] * projected_gravity[:, 0] + projected_gravity[:, 1] * projected_gravity[:, 1]
    r_upright = torch.exp(-temp_upright * tilt_mag)

    # Posture regularization: keep joints near nominal positions
    pos_err = dof_pos - default_dof_pos
    pos_err_mag = torch.mean(torch.abs(pos_err), dim=1)
    r_posture = torch.exp(-temp_posture * pos_err_mag)

    # Joint velocity regularization: discourage fast/sudden joint motions (cautious gait)
    joint_vel_mag = torch.mean(torch.abs(dof_vel), dim=1)
    r_joint_vel = torch.exp(-temp_joint_vel * joint_vel_mag)

    # Action magnitude regularization: small efforts promote cautious behavior
    action_mag = torch.mean(torch.abs(actions), dim=1)
    r_action_smooth = torch.exp(-temp_action * action_mag)

    # Stability via low roll/pitch angular rates
    rp_rate_mag = base_ang_vel_body[:, 0] * base_ang_vel_body[:, 0] + base_ang_vel_body[:, 1] * base_ang_vel_body[:, 1]
    r_stable_rate = torch.exp(-temp_stable_rate * rp_rate_mag)

    # Forward progress bonus (only reward non-negative forward body-frame speed)
    r_forward = torch.clamp(base_lin_vel_body[:, 0], min=0.0)

    # Total reward
    reward = (
        w_vx * r_vx +
        w_vy * r_vy +
        w_yaw * r_yaw +
        w_upright * r_upright +
        w_posture * r_posture +
        w_joint_vel * r_joint_vel +
        w_action * r_action_smooth +
        w_stable_rate * r_stable_rate +
        w_forward * r_forward
    )

    components: Dict[str, torch.Tensor] = {
        "r_vx": r_vx,
        "r_vy": r_vy,
        "r_yaw": r_yaw,
        "r_upright": r_upright,
        "r_posture": r_posture,
        "r_joint_vel": r_joint_vel,
        "r_action_smooth": r_action_smooth,
        "r_stable_rate": r_stable_rate,
        "r_forward": r_forward,
    }

    return reward, components

from typing import Tuple, Dict
import math
import torch
from torch import Tensor
@torch.jit.script
def compute_reward(
    root_states: torch.Tensor,
    commands: torch.Tensor,
    default_dof_pos: torch.Tensor,
    dof_pos: torch.Tensor,
    dof_vel: torch.Tensor,
    gravity_vec: torch.Tensor,
    actions: torch.Tensor,
    lin_vel_scale: float,
    ang_vel_scale: float,
    dof_pos_scale: float,
    dof_vel_scale: float
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract base orientation and velocities (world) and transform to base frame, consistent with observations
    base_quat = root_states[:, 3:7]
    base_lin_vel = quat_rotate_inverse(base_quat, root_states[:, 7:10]) * lin_vel_scale
    base_ang_vel = quat_rotate_inverse(base_quat, root_states[:, 10:13]) * ang_vel_scale
    projected_gravity = quat_rotate(base_quat, gravity_vec)

    # Desired (commanded) velocities, scaled like in observations
    desired_lin_vel_x = commands[:, 0] * lin_vel_scale
    desired_lin_vel_y = commands[:, 1] * lin_vel_scale
    desired_yaw_rate = commands[:, 2] * ang_vel_scale

    # Scaled joint states
    dof_pos_scaled = (dof_pos - default_dof_pos) * dof_pos_scale
    dof_vel_scaled = dof_vel * dof_vel_scale

    # Temperatures for exponential shaping (each component has its own temperature)
    temp_vx_track = 0.6
    temp_vy_stability = 0.4
    temp_yaw_track = 0.6
    temp_upright = 0.2
    temp_dof_vel = 0.8
    temp_posture = 1.2
    temp_actions = 0.8

    # Forward velocity tracking reward (cautious tracking of commanded forward speed)
    vx_error = torch.abs(base_lin_vel[:, 0] - desired_lin_vel_x)
    r_vx_track = torch.exp(-vx_error / temp_vx_track)

    # Encourage moving forward (non-negative body x velocity), softly
    vx_positive = torch.clamp(base_lin_vel[:, 0], min=0.0)

    # Lateral stability: discourage sideways motion (body y velocity)
    vy_mag = torch.abs(base_lin_vel[:, 1])
    r_vy_stability = torch.exp(-vy_mag / temp_vy_stability)

    # Yaw rate tracking (keep heading changes small unless commanded)
    yaw_error = torch.abs(base_ang_vel[:, 2] - desired_yaw_rate)
    r_yaw_track = torch.exp(-yaw_error / temp_yaw_track)

    # Uprightness: penalize tilt (horizontal components of gravity in base frame)
    tilt_mag = torch.sqrt(projected_gravity[:, 0] * projected_gravity[:, 0] +
                          projected_gravity[:, 1] * projected_gravity[:, 1] + 1e-8)
    r_upright = torch.exp(-tilt_mag / temp_upright)

    # Cautious stepping: keep joint velocities moderate
    dof_vel_mean_abs = torch.mean(torch.abs(dof_vel_scaled), dim=1)
    r_smooth_joints = torch.exp(-dof_vel_mean_abs / temp_dof_vel)

    # Conservative posture: limit deviation from neutral stance (but not too strongly)
    dof_pos_mean_abs = torch.mean(torch.abs(dof_pos_scaled), dim=1)
    r_posture = torch.exp(-dof_pos_mean_abs / temp_posture)

    # Action smoothness: discourage large action magnitudes
    actions_mean_abs = torch.mean(torch.abs(actions), dim=1)
    r_action_smooth = torch.exp(-actions_mean_abs / temp_actions)

    # Weights for combining components
    w_vx_track = 2.0
    w_vx_positive = 0.5
    w_vy_stability = 1.0
    w_yaw_track = 1.0
    w_upright = 1.5
    w_smooth_joints = 0.4
    w_posture = 0.3
    w_action_smooth = 0.3

    # Combine into total reward
    total_reward = (
        w_vx_track * r_vx_track +
        w_vx_positive * vx_positive +
        w_vy_stability * r_vy_stability +
        w_yaw_track * r_yaw_track +
        w_upright * r_upright +
        w_smooth_joints * r_smooth_joints +
        w_posture * r_posture +
        w_action_smooth * r_action_smooth
    )

    # Assemble components dictionary
    components = torch.jit.annotate(Dict[str, torch.Tensor], {})
    components["r_vx_track"] = r_vx_track
    components["vx_positive"] = vx_positive
    components["r_vy_stability"] = r_vy_stability
    components["r_yaw_track"] = r_yaw_track
    components["r_upright"] = r_upright
    components["r_smooth_joints"] = r_smooth_joints
    components["r_posture"] = r_posture
    components["r_action_smooth"] = r_action_smooth
    components["total_reward"] = total_reward

    return total_reward, components