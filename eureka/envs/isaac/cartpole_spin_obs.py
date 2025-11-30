class CartpoleSpin(VecTask):
    """Rest of the environment definition omitted."""
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
