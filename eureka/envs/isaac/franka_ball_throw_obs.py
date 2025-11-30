class FrankaBallThrow(VecTask):
    """Rest of the environment definition omitted."""
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
            "ball_to_target": self._target_pos - self._ball_state[:, :3],
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

    def compute_observations(self):
        self._refresh()
        obs = ["ball_pos", "ball_vel", "target_pos", "eef_pos", "eef_quat"]
        obs += ["q_gripper"] if self.control_type == "osc" else ["q"]
        self.obs_buf = torch.cat([self.states[ob] for ob in obs], dim=-1)

        return self.obs_buf
