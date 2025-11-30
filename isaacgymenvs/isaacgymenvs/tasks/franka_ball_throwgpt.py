
import numpy as np
import os
import torch

from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym.torch_utils import *

from isaacgymenvs.utils.torch_jit_utils import *
from isaacgymenvs.tasks.base.vec_task import VecTask


@torch.jit.script
def axisangle2quat(vec, eps=1e-6):
    # type: (Tensor, float) -> Tensor
    input_shape = vec.shape[:-1]
    vec = vec.reshape(-1, 3)

    angle = torch.norm(vec, dim=-1, keepdim=True)

    quat = torch.zeros(torch.prod(torch.tensor(input_shape)), 4, device=vec.device)
    quat[:, 3] = 1.0

    idx = angle.reshape(-1) > eps
    quat[idx, :] = torch.cat([
        vec[idx, :] * torch.sin(angle[idx, :] / 2.0) / angle[idx, :],
        torch.cos(angle[idx, :] / 2.0)
    ], dim=-1)

    quat = quat.reshape(list(input_shape) + [4, ])
    return quat


class FrankaBallThrowGPT(VecTask):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        self.cfg = cfg

        self.max_episode_length = self.cfg["env"]["episodeLength"]

        self.action_scale = self.cfg["env"]["actionScale"]
        self.start_position_noise = self.cfg["env"]["startPositionNoise"]
        self.franka_position_noise = self.cfg["env"]["frankaPositionNoise"]
        self.franka_rotation_noise = self.cfg["env"]["frankaRotationNoise"]
        self.franka_dof_noise = self.cfg["env"]["frankaDofNoise"]
        self.aggregate_mode = self.cfg["env"]["aggregateMode"]

        self.reward_settings = {
            "r_dist_scale": self.cfg["env"]["distRewardScale"],
            "r_lift_scale": self.cfg["env"]["liftRewardScale"],
            "r_throw_scale": self.cfg["env"]["throwRewardScale"],
            "r_target_scale": self.cfg["env"]["targetRewardScale"],
        }

        self.control_type = self.cfg["env"]["controlType"]
        assert self.control_type in {"osc", "joint_tor"},\
            "Invalid control type specified. Must be one of: {osc, joint_tor}"

        self.cfg["env"]["numObservations"] = 18 if self.control_type == "osc" else 25
        self.cfg["env"]["numActions"] = 7 if self.control_type == "osc" else 8

        self.states = {}                        # will be dict filled with relevant states to use for reward calculation
        self.handles = {}                       # will be dict mapping names to relevant sim handles
        self.num_dofs = None                    # Total number of DOFs per env
        self.actions = None                     # Current actions to be deployed
        self._init_ball_state = None            # Initial state of ball for the current env
        self._ball_state = None                 # Current state of ball for the current env
        self._ball_id = None                    # Actor ID corresponding to ball for a given env
        self._target_pos = None                 # Target position for ball throwing

        self._root_state = None             # State of root body        (n_envs, 13)
        self._dof_state = None  # State of all joints       (n_envs, n_dof)
        self._q = None  # Joint positions           (n_envs, n_dof)
        self._qd = None                     # Joint velocities          (n_envs, n_dof)
        self._rigid_body_state = None  # State of all rigid bodies             (n_envs, n_bodies, 13)
        self._contact_forces = None     # Contact forces in sim
        self._eef_state = None  # end effector state (at grasping point)
        self._eef_lf_state = None  # end effector state (at left fingertip)
        self._eef_rf_state = None  # end effector state (at left fingertip)
        self._j_eef = None  # Jacobian for end effector
        self._mm = None  # Mass matrix
        self._arm_control = None  # Tensor buffer for controlling arm
        self._gripper_control = None  # Tensor buffer for controlling gripper
        self._pos_control = None            # Position actions
        self._effort_control = None         # Torque actions
        self._franka_effort_limits = None        # Actuator effort limits for franka
        self._global_indices = None         # Unique indices corresponding to all envs in flattened array

        self.debug_viz = self.cfg["env"]["enableDebugVis"]

        self.up_axis = "z"
        self.up_axis_idx = 2

        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)

        self.franka_default_dof_pos = to_torch(
            [0, 0.1963, 0, -2.6180, 0, 2.9416, 0.7854, 0.035, 0.035], device=self.device
        )

        self.kp = to_torch([150.] * 6, device=self.device)
        self.kd = 2 * torch.sqrt(self.kp)
        self.kp_null = to_torch([10.] * 7, device=self.device)
        self.kd_null = 2 * torch.sqrt(self.kp_null)

        self.cmd_limit = to_torch([0.1, 0.1, 0.1, 0.5, 0.5, 0.5], device=self.device).unsqueeze(0) if \
        self.control_type == "osc" else self._franka_effort_limits[:7].unsqueeze(0)

        self.successes = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.consecutive_successes = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

        self.reset_idx(torch.arange(self.num_envs, device=self.device))

        self._refresh()

    def create_sim(self):
        self.sim_params.up_axis = gymapi.UP_AXIS_Z
        self.sim_params.gravity.x = 0
        self.sim_params.gravity.y = 0
        self.sim_params.gravity.z = -9.81
        self.sim = super().create_sim(
            self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../assets")
        franka_asset_file = "urdf/franka_description/robots/franka_panda_gripper.urdf"

        if "asset" in self.cfg["env"]:
            asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.cfg["env"]["asset"].get("assetRoot", asset_root))
            franka_asset_file = self.cfg["env"]["asset"].get("assetFileNameFranka", franka_asset_file)

        asset_options = gymapi.AssetOptions()
        asset_options.flip_visual_attachments = True
        asset_options.fix_base_link = True
        asset_options.collapse_fixed_joints = False
        asset_options.disable_gravity = True
        asset_options.thickness = 0.001
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_EFFORT
        asset_options.use_mesh_materials = True
        franka_asset = self.gym.load_asset(self.sim, asset_root, franka_asset_file, asset_options)

        franka_dof_stiffness = to_torch([0, 0, 0, 0, 0, 0, 0, 5000., 5000.], dtype=torch.float, device=self.device)
        franka_dof_damping = to_torch([0, 0, 0, 0, 0, 0, 0, 1.0e2, 1.0e2], dtype=torch.float, device=self.device)

        table_pos = [0.0, 0.0, 1.0]
        table_thickness = 0.05
        table_opts = gymapi.AssetOptions()
        table_opts.fix_base_link = True
        table_asset = self.gym.create_box(self.sim, *[1.2, 1.2, table_thickness], table_opts)

        table_stand_height = 0.1
        table_stand_pos = [-0.5, 0.0, 1.0 + table_thickness / 2 + table_stand_height / 2]
        table_stand_opts = gymapi.AssetOptions()
        table_stand_opts.fix_base_link = True
        table_stand_asset = self.gym.create_box(self.sim, *[0.2, 0.2, table_stand_height], table_opts)

        self.ball_radius = 0.0335
        ball_opts = gymapi.AssetOptions()
        ball_opts.density = 370.0  # Tennis ball density in kg/m^3
        ball_asset = self.gym.create_sphere(self.sim, self.ball_radius, ball_opts)
        ball_color = gymapi.Vec3(0.9, 0.9, 0.1)  # Yellow tennis ball

        self.target_radius = 0.05
        target_opts = gymapi.AssetOptions()
        target_opts.fix_base_link = True
        target_opts.disable_gravity = True
        target_asset = self.gym.create_sphere(self.sim, self.target_radius, target_opts)
        target_color = gymapi.Vec3(1.0, 0.0, 0.0)  # Red target

        self.num_franka_bodies = self.gym.get_asset_rigid_body_count(franka_asset)
        self.num_franka_dofs = self.gym.get_asset_dof_count(franka_asset)

        print("num franka bodies: ", self.num_franka_bodies)
        print("num franka dofs: ", self.num_franka_dofs)

        franka_dof_props = self.gym.get_asset_dof_properties(franka_asset)
        self.franka_dof_lower_limits = []
        self.franka_dof_upper_limits = []
        self._franka_effort_limits = []
        for i in range(self.num_franka_dofs):
            franka_dof_props['driveMode'][i] = gymapi.DOF_MODE_POS if i > 6 else gymapi.DOF_MODE_EFFORT
            if self.physics_engine == gymapi.SIM_PHYSX:
                franka_dof_props['stiffness'][i] = franka_dof_stiffness[i]
                franka_dof_props['damping'][i] = franka_dof_damping[i]
            else:
                franka_dof_props['stiffness'][i] = 7000.0
                franka_dof_props['damping'][i] = 50.0

            self.franka_dof_lower_limits.append(franka_dof_props['lower'][i])
            self.franka_dof_upper_limits.append(franka_dof_props['upper'][i])
            self._franka_effort_limits.append(franka_dof_props['effort'][i])

        self.franka_dof_lower_limits = to_torch(self.franka_dof_lower_limits, device=self.device)
        self.franka_dof_upper_limits = to_torch(self.franka_dof_upper_limits, device=self.device)
        self._franka_effort_limits = to_torch(self._franka_effort_limits, device=self.device)
        self.franka_dof_speed_scales = torch.ones_like(self.franka_dof_lower_limits)
        self.franka_dof_speed_scales[[7, 8]] = 0.1
        franka_dof_props['effort'][7] = 200
        franka_dof_props['effort'][8] = 200

        franka_start_pose = gymapi.Transform()
        franka_start_pose.p = gymapi.Vec3(-0.45, 0.0, 1.0 + table_thickness / 2 + table_stand_height)
        franka_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        table_start_pose = gymapi.Transform()
        table_start_pose.p = gymapi.Vec3(*table_pos)
        table_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
        self._table_surface_pos = np.array(table_pos) + np.array([0, 0, table_thickness / 2])
        self.reward_settings["table_height"] = self._table_surface_pos[2]

        table_stand_start_pose = gymapi.Transform()
        table_stand_start_pose.p = gymapi.Vec3(*table_stand_pos)
        table_stand_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        ball_start_pose = gymapi.Transform()
        ball_start_pose.p = gymapi.Vec3(0.0, 0.0, self._table_surface_pos[2] + self.ball_radius)
        ball_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        target_start_pose = gymapi.Transform()
        target_start_pose.p = gymapi.Vec3(0.5, 0.0, 1.5)
        target_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        num_franka_bodies = self.gym.get_asset_rigid_body_count(franka_asset)
        num_franka_shapes = self.gym.get_asset_rigid_shape_count(franka_asset)
        max_agg_bodies = num_franka_bodies + 4     # 1 for table, table stand, ball, target
        max_agg_shapes = num_franka_shapes + 4     # 1 for table, table stand, ball, target

        self.frankas = []
        self.envs = []

        for i in range(self.num_envs):
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)

            if self.aggregate_mode >= 3:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            if self.franka_position_noise > 0:
                rand_xy = self.franka_position_noise * (-1. + np.random.rand(2) * 2.0)
                franka_start_pose.p = gymapi.Vec3(-0.45 + rand_xy[0], 0.0 + rand_xy[1],
                                                 1.0 + table_thickness / 2 + table_stand_height)
            if self.franka_rotation_noise > 0:
                rand_rot = torch.zeros(1, 3)
                rand_rot[:, -1] = self.franka_rotation_noise * (-1. + np.random.rand() * 2.0)
                new_quat = axisangle2quat(rand_rot).squeeze().numpy().tolist()
                franka_start_pose.r = gymapi.Quat(*new_quat)
            franka_actor = self.gym.create_actor(env_ptr, franka_asset, franka_start_pose, "franka", i, 0, 0)
            self.gym.set_actor_dof_properties(env_ptr, franka_actor, franka_dof_props)

            if self.aggregate_mode == 2:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            table_actor = self.gym.create_actor(env_ptr, table_asset, table_start_pose, "table", i, 1, 0)
            table_stand_actor = self.gym.create_actor(env_ptr, table_stand_asset, table_stand_start_pose, "table_stand",
                                                      i, 1, 0)

            if self.aggregate_mode == 1:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            self._ball_id = self.gym.create_actor(env_ptr, ball_asset, ball_start_pose, "ball", i, 2, 0)
            self._target_id = self.gym.create_actor(env_ptr, target_asset, target_start_pose, "target", i, 1, 0)
            
            self.gym.set_rigid_body_color(env_ptr, self._ball_id, 0, gymapi.MESH_VISUAL, ball_color)
            self.gym.set_rigid_body_color(env_ptr, self._target_id, 0, gymapi.MESH_VISUAL, target_color)

            if self.aggregate_mode > 0:
                self.gym.end_aggregate(env_ptr)

            self.envs.append(env_ptr)
            self.frankas.append(franka_actor)

        self._init_ball_state = torch.zeros(self.num_envs, 13, device=self.device)
        self._init_target_state = torch.zeros(self.num_envs, 13, device=self.device)
        self._target_pos = torch.zeros(self.num_envs, 3, device=self.device)

        self.init_data()

    def init_data(self):
        env_ptr = self.envs[0]
        franka_handle = 0
        self.handles = {
            "hand": self.gym.find_actor_rigid_body_handle(env_ptr, franka_handle, "panda_hand"),
            "leftfinger_tip": self.gym.find_actor_rigid_body_handle(env_ptr, franka_handle, "panda_leftfinger_tip"),
            "rightfinger_tip": self.gym.find_actor_rigid_body_handle(env_ptr, franka_handle, "panda_rightfinger_tip"),
            "grip_site": self.gym.find_actor_rigid_body_handle(env_ptr, franka_handle, "panda_grip_site"),
            "ball_body_handle": self.gym.find_actor_rigid_body_handle(self.envs[0], self._ball_id, "sphere"),
        }

        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs

        _actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        _dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        _rigid_body_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self._root_state = gymtorch.wrap_tensor(_actor_root_state_tensor).view(self.num_envs, -1, 13)
        self._dof_state = gymtorch.wrap_tensor(_dof_state_tensor).view(self.num_envs, -1, 2)
        self._rigid_body_state = gymtorch.wrap_tensor(_rigid_body_state_tensor).view(self.num_envs, -1, 13)
        self._q = self._dof_state[..., 0]
        self._qd = self._dof_state[..., 1]
        self._eef_state = self._rigid_body_state[:, self.handles["grip_site"], :]
        self._eef_lf_state = self._rigid_body_state[:, self.handles["leftfinger_tip"], :]
        self._eef_rf_state = self._rigid_body_state[:, self.handles["rightfinger_tip"], :]
        _jacobian = self.gym.acquire_jacobian_tensor(self.sim, "franka")
        jacobian = gymtorch.wrap_tensor(_jacobian)
        hand_joint_index = self.gym.get_actor_joint_dict(env_ptr, franka_handle)['panda_hand_joint']
        self._j_eef = jacobian[:, hand_joint_index, :, :7]
        _massmatrix = self.gym.acquire_mass_matrix_tensor(self.sim, "franka")
        mm = gymtorch.wrap_tensor(_massmatrix)
        self._mm = mm[:, :7, :7]
        self._ball_state = self._root_state[:, self._ball_id, :]
        self._target_state = self._root_state[:, self._target_id, :]

        self.states.update({
            "ball_radius": torch.ones_like(self._eef_state[:, 0]) * self.ball_radius,
        })

        self._pos_control = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        self._effort_control = torch.zeros_like(self._pos_control)

        self._arm_control = self._effort_control[:, :7]
        self._gripper_control = self._pos_control[:, 7:9]

        self._global_indices = torch.arange(self.num_envs * 5, dtype=torch.int32,
                                           device=self.device).view(self.num_envs, -1)

    def _update_states(self):
        self.states.update({
            "q": self._q[:, :],
            "q_gripper": self._q[:, -2:],
            "eef_pos": self._eef_state[:, :3],
            "eef_quat": self._eef_state[:, 3:7],
            "eef_vel": self._eef_state[:, 7:],
            "eef_lf_pos": self._eef_lf_state[:, :3],
            "eef_rf_pos": self._eef_rf_state[:, :3],
            "ball_pos": self._ball_state[:, :3],
            "ball_quat": self._ball_state[:, 3:7],
            "ball_vel": self._ball_state[:, 7:10],
            "ball_ang_vel": self._ball_state[:, 10:13],
            "ball_pos_relative": self._ball_state[:, :3] - self._eef_state[:, :3],
            "target_pos": self._target_pos[:,:3],
            "ball_to_target": self._target_pos - self._ball_state[:, :3]
        })
        # Franka
        self.q = self._q[:, :]
        self.q_gripper = self._q[:, -2:]
        self.eef_pos = self._eef_state[:, :3]
        self.eef_quat = self._eef_state[:, 3:7]
        self.eef_vel = self._eef_state[:, 7:]
        self.eef_lf_pos = self._eef_lf_state[:, :3]
        self.eef_rf_pos = self._eef_rf_state[:, :3]
        self.table_surface_pos = self._table_surface_pos
        # Ball
        self.ball_pos = self._ball_state[:, :3]
        self.ball_quat = self._ball_state[:, 3:7]
        self.ball_vel = self._ball_state[:, 7:10]
        self.ball_ang_vel = self._ball_state[:, 10:13]
        self.ball_pos_relative = self._ball_state[:, :3] - self._eef_state[:, :3]
        # Target
        self.target_pos = self._target_pos[:,:3]
        self.ball_to_target = self._target_pos - self._ball_state[:, :3]



    def _refresh(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.refresh_mass_matrix_tensors(self.sim)

        self._update_states()

    def compute_reward(self, actions):
        self.rew_buf[:], self.rew_dict = compute_reward(self.ball_pos, self.ball_vel, self.ball_to_target)
        self.extras['gpt_reward'] = self.rew_buf.mean()
        for rew_state in self.rew_dict: self.extras[rew_state] = self.rew_dict[rew_state].mean()
        self.gt_rew_buf, self.reset_buf[:], self.successes[:], self.consecutive_successes[:] = compute_success(
            self.reset_buf, self.progress_buf, self.successes, self.consecutive_successes, self.actions, self.states, self.reward_settings, self.max_episode_length
        )
        self.extras['gt_reward'] = self.gt_rew_buf.mean()
        self.extras['successes'] = self.successes
        self.extras['consecutive_successes'] = self.consecutive_successes.mean() 
        
    def compute_observations(self):
        self._refresh()
        obs = ["ball_pos", "ball_vel", "target_pos", "eef_pos", "eef_quat"]
        obs += ["q_gripper"] if self.control_type == "osc" else ["q"]
        self.obs_buf = torch.cat([self.states[ob] for ob in obs], dim=-1)

        return self.obs_buf

    def reset_idx(self, env_ids):
        env_ids_int32 = env_ids.to(dtype=torch.int32)

        self._reset_init_ball_state(env_ids=env_ids)
        
        self._reset_target_state(env_ids=env_ids)

        self._ball_state[env_ids] = self._init_ball_state[env_ids]
        self._target_state[env_ids] = self._init_target_state[env_ids]

        reset_noise = torch.rand((len(env_ids), 9), device=self.device)
        pos = tensor_clamp(
            self.franka_default_dof_pos.unsqueeze(0) +
            self.franka_dof_noise * 2.0 * (reset_noise - 0.5),
            self.franka_dof_lower_limits.unsqueeze(0), self.franka_dof_upper_limits)

        pos[:, -2:] = self.franka_default_dof_pos[-2:]

        self._q[env_ids, :] = pos
        self._qd[env_ids, :] = torch.zeros_like(self._qd[env_ids])

        self._pos_control[env_ids, :] = pos
        self._effort_control[env_ids, :] = torch.zeros_like(pos)

        multi_env_ids_int32 = self._global_indices[env_ids, 0].flatten()
        self.gym.set_dof_position_target_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(self._pos_control),
                                                        gymtorch.unwrap_tensor(multi_env_ids_int32),
                                                        len(multi_env_ids_int32))
        self.gym.set_dof_actuation_force_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(self._effort_control),
                                                        gymtorch.unwrap_tensor(multi_env_ids_int32),
                                                        len(multi_env_ids_int32))
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self._dof_state),
                                              gymtorch.unwrap_tensor(multi_env_ids_int32),
                                              len(multi_env_ids_int32))

        multi_env_ids_objects_int32 = self._global_indices[env_ids, 3:5].flatten()
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self._root_state),
            gymtorch.unwrap_tensor(multi_env_ids_objects_int32), len(multi_env_ids_objects_int32))

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self.successes[env_ids] = 0

    def _reset_init_ball_state(self, env_ids):
        if env_ids is None:
            env_ids = torch.arange(start=0, end=self.num_envs, device=self.device, dtype=torch.long)

        num_resets = len(env_ids)
        sampled_ball_state = torch.zeros(num_resets, 13, device=self.device)

        centered_ball_xy_state = torch.tensor(self._table_surface_pos[:2], device=self.device, dtype=torch.float32)

        sampled_ball_state[:, :2] = centered_ball_xy_state.unsqueeze(0) + \
                                     self.start_position_noise * (torch.rand(num_resets, 2, device=self.device) - 0.5)

        sampled_ball_state[:, 2] = self._table_surface_pos[2] + self.ball_radius

        sampled_ball_state[:, 6] = 1.0

        sampled_ball_state[:, 7:] = 0.0

        self._init_ball_state[env_ids, :] = sampled_ball_state

    def _reset_target_state(self, env_ids):
        if env_ids is None:
            env_ids = torch.arange(start=0, end=self.num_envs, device=self.device, dtype=torch.long)

        num_resets = len(env_ids)
        
        sampled_target_state = torch.zeros(num_resets, 13, device=self.device)
        
        sampled_target_state[:, 0] = 0.3 + 0.7 * torch.rand(num_resets, device=self.device)
        sampled_target_state[:, 1] = -0.5 + 1.0 * torch.rand(num_resets, device=self.device)
        sampled_target_state[:, 2] = 1.2 + 0.8 * torch.rand(num_resets, device=self.device)
        
        sampled_target_state[:, 6] = 1.0
        
        sampled_target_state[:, 7:] = 0.0
        
        self._init_target_state[env_ids, :] = sampled_target_state
        self._target_pos[env_ids] = sampled_target_state[:, :3]

    def _compute_osc_torques(self, dpose):
        q, qd = self._q[:, :7], self._qd[:, :7]
        mm_inv = torch.inverse(self._mm)
        m_eef_inv = self._j_eef @ mm_inv @ torch.transpose(self._j_eef, 1, 2)
        m_eef = torch.inverse(m_eef_inv)

        u = torch.transpose(self._j_eef, 1, 2) @ m_eef @ (
                self.kp * dpose - self.kd * self.states["eef_vel"]).unsqueeze(-1)

        j_eef_inv = m_eef @ self._j_eef @ mm_inv
        u_null = self.kd_null * -qd + self.kp_null * (
                (self.franka_default_dof_pos[:7] - q + np.pi) % (2 * np.pi) - np.pi)
        u_null[:, 7:] *= 0
        u_null = self._mm @ u_null.unsqueeze(-1)
        u += (torch.eye(7, device=self.device).unsqueeze(0) - torch.transpose(self._j_eef, 1, 2) @ j_eef_inv) @ u_null

        u = tensor_clamp(u.squeeze(-1),
                         -self._franka_effort_limits[:7].unsqueeze(0), self._franka_effort_limits[:7].unsqueeze(0))

        return u

    def pre_physics_step(self, actions):
        self.actions = actions.clone().to(self.device)

        u_arm, u_gripper = self.actions[:, :-1], self.actions[:, -1]

        u_arm = u_arm * self.cmd_limit / self.action_scale
        if self.control_type == "osc":
            u_arm = self._compute_osc_torques(dpose=u_arm)
        self._arm_control[:, :] = u_arm

        u_fingers = torch.zeros_like(self._gripper_control)
        u_fingers[:, 0] = torch.where(u_gripper >= 0.0, self.franka_dof_upper_limits[-2].item(),
                                      self.franka_dof_lower_limits[-2].item())
        u_fingers[:, 1] = torch.where(u_gripper >= 0.0, self.franka_dof_upper_limits[-1].item(),
                                      self.franka_dof_lower_limits[-1].item())
        self._gripper_control[:, :] = u_fingers

        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self._pos_control))
        self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self._effort_control))

    def post_physics_step(self):
        self.progress_buf += 1

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        self.compute_observations()
        self.compute_reward(self.actions)

        if self.viewer and self.debug_viz:
            self.gym.clear_lines(self.viewer)
            self.gym.refresh_rigid_body_state_tensor(self.sim)

            eef_pos = self.states["eef_pos"]
            eef_rot = self.states["eef_quat"]
            ball_pos = self.states["ball_pos"]
            ball_rot = self.states["ball_quat"]
            target_pos = self.states["target_pos"]

            for i in range(self.num_envs):
                for pos, rot in zip((eef_pos, ball_pos), (eef_rot, ball_rot)):
                    px = (pos[i] + quat_apply(rot[i], to_torch([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
                    py = (pos[i] + quat_apply(rot[i], to_torch([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
                    pz = (pos[i] + quat_apply(rot[i], to_torch([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()

                    p0 = pos[i].cpu().numpy()
                    self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], px[0], px[1], px[2]], [0.85, 0.1, 0.1])
                    self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], py[0], py[1], py[2]], [0.1, 0.85, 0.1])
                    self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], pz[0], pz[1], pz[2]], [0.1, 0.1, 0.85])
                
                ball_p = ball_pos[i].cpu().numpy()
                target_p = target_pos[i].cpu().numpy()
                self.gym.add_lines(self.viewer, self.envs[i], 1, 
                                 [ball_p[0], ball_p[1], ball_p[2], target_p[0], target_p[1], target_p[2]], 
                                 [1.0, 1.0, 0.0])


@torch.jit.script
def compute_success(
    reset_buf, progress_buf, successes, consecutive_successes, actions, states, reward_settings, max_episode_length
):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Dict[str, Tensor], Dict[str, float], float) -> Tuple[Tensor, Tensor, Tensor, Tensor]
    
    d_eef_ball = torch.norm(states["ball_pos_relative"], dim=-1)
    
    d_ball_target = torch.norm(states["ball_to_target"], dim=-1)
    
    ball_height = states["ball_pos"][:, 2] - reward_settings["table_height"]
    
    ball_speed = torch.norm(states["ball_vel"], dim=-1)
    
    dist_reward = 1.0 - torch.tanh(5.0 * d_eef_ball)
    
    lift_reward = torch.where(d_eef_ball < 0.1, 
                             torch.tanh(ball_height * 10.0),
                             torch.zeros_like(ball_height))
    
    throw_reward = torch.where(ball_height > 0.2,
                               torch.tanh(ball_speed),
                               torch.zeros_like(ball_speed))
    
    target_reward = torch.exp(-d_ball_target * 3.0)
    
    success_condition = d_ball_target < 0.15
    successes = torch.where(success_condition,
                           torch.ones_like(successes),
                           successes)
    
    rewards = (reward_settings["r_dist_scale"] * dist_reward +
              reward_settings["r_lift_scale"] * lift_reward +
              reward_settings["r_throw_scale"] * throw_reward +
              reward_settings["r_target_scale"] * target_reward)
    
    reset_buf = torch.where(progress_buf >= max_episode_length - 1,
                           torch.ones_like(reset_buf),
                           reset_buf)
    
    reset_buf = torch.where(states["ball_pos"][:, 2] < reward_settings["table_height"] - 0.5,
                           torch.ones_like(reset_buf),
                           reset_buf)
    
    consecutive_successes = torch.where(reset_buf > 0,
                                       successes * reset_buf,
                                       consecutive_successes).mean()
    
    return rewards, reset_buf, successes, consecutive_successes
from typing import Tuple, Dict
import math
import torch
from torch import Tensor
@torch.jit.script
def compute_reward(
    ball_pos: torch.Tensor,
    ball_vel: torch.Tensor,
    ball_to_target: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Improved reward function for throwing ball to hit target.

    Inputs:
    - ball_pos: (N,3) tensor of ball positions
    - ball_vel: (N,3) tensor of ball velocities
    - ball_to_target: (N,3) vector from ball pos to target pos

    Returns:
    - reward: (N,) tensor of scalar rewards
    - reward_info: dict with individual reward components
    """
    device = ball_pos.device
    batch_size = ball_pos.shape[0]

    eps = 1e-8

    # Distance to target
    dist_to_target = torch.norm(ball_to_target, p=2, dim=1)  # (N,)
    
    # ---- Reward component: distance to target ----
    # Previous version reward_dist was too small and nearly constant (~0.01-0.02),
    # making it not helpful for learning.
    # Adjust temperature scale to be larger to provide stronger gradient and range.
    dist_temp = 0.05
    reward_dist = torch.exp(-dist_to_target / dist_temp)  # [0,1], sharper decay near target
    
    # ---- Reward component: velocity towards target ----
    # reward_vel showed good variation and increase over training,
    # but might overshadow distance reward.
    # Rescale and clip velocity reward to range [0,1].
    ball_to_target_dir = ball_to_target / (dist_to_target.unsqueeze(1) + eps)  # unit vector toward target
    vel_proj = torch.sum(ball_vel * ball_to_target_dir, dim=1).clamp(min=0.0)  # only positive progress
    vel_temp = 1.0
    reward_vel = torch.tanh(vel_proj / vel_temp)  # smooth saturating reward [0,1]

    # ---- New Reward component: hammer hitting the target (ball within threshold dist) ----
    # We add a sparse success reward to enable learning the hitting of the target
    hit_threshold = 0.05  # 5 cm
    reward_hit = (dist_to_target < hit_threshold).to(ball_pos.dtype)

    # ---- Combine the rewards ----
    # Weight reward_hit more to encourage actually hitting the target
    # Also balance distance and velocity rewards
    reward = reward_dist * 0.6 + reward_vel * 0.3 + reward_hit * 1.0

    reward_info = {
        "reward_dist": reward_dist,
        "reward_vel": reward_vel,
        "reward_hit": reward_hit,
    }

    return reward, reward_info
