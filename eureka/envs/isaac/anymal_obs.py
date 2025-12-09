class Anymal(VecTask):
    """Rest of the environment definition omitted.

    Available tensors for reward function (accessible as self.X):
    - root_states: (num_envs, 13) - positions [0:3], rotations [3:7], linear velocity [7:10], angular velocity [10:13]
    - dof_pos: (num_envs, 12) - joint positions
    - dof_vel: (num_envs, 12) - joint velocities
    - default_dof_pos: (num_envs, 12) - default joint positions
    - contact_forces: (num_envs, num_bodies, 3) - contact forces per body
    - torques: (num_envs, 12) - joint torques
    - commands: (num_envs, 3) - velocity commands [x_vel, y_vel, yaw_rate]
    - actions: (num_envs, 12) - current actions
    - gravity_vec: (num_envs, 3) - gravity vector
    - dt: float - simulation timestep

    Note: To get linear velocity, use root_states[:, 7:10]. To get angular velocity, use root_states[:, 10:13].
    There is NO self.velocities attribute - always use root_states slicing.
    """
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



