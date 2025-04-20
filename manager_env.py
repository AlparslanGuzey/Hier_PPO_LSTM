# File: manager_env.py

import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces

import ray
from ray.rllib.algorithms.ppo import PPO
from Environment import MultiUAVEnv

class ManagerEnv(gym.Env):
    """
    A hierarchical 'Manager' environment that picks sub-goals for all UAVs.
    Then it runs a short sub-episode of the Worker environment (MultiUAVEnv)
    using a pre-trained PPO+LSTM 'Worker Policy' to move towards that sub-goal.

    On each Manager step:
      - Action = index of a sub-goal from a discrete set
      - We set that sub-goal for the Worker env, run up to N sub-steps or until done
      - Manager reward is computed based on progress/collisions, etc.
    """

    def __init__(self, config=None):
        super().__init__()
        self.config = config or {}

        # 1) Setup Worker environment (the low-level env)
        self.worker_env = MultiUAVEnv(self.config.get("worker_env_config", {}))

        # 2) Load pre-trained Worker policy from checkpoint
        worker_ckpt = self.config.get("worker_ckpt_path", None)
        if not worker_ckpt or not os.path.exists(worker_ckpt):
            raise ValueError(
                f"[ManagerEnv] Invalid worker_ckpt_path: '{worker_ckpt}'. "
                "Please provide a valid checkpoint for the Worker policy."
            )
        # Build a PPO trainer for the worker, then restore from checkpoint
        self.worker_trainer = PPO(config=self.config.get("worker_policy_rllib_config", {}))
        self.worker_trainer.restore(worker_ckpt)
        # We'll reference the worker policy by a known ID (default: "default_policy" or "shared_policy")
        self.worker_policy_id = self.config.get("worker_policy_id", "default_policy")

        # 3) Manager action space: picking from a discrete set of sub-goals (waypoints)
        self.candidate_goals = [(0, 0), (2, 2), (5, 5), (8, 8)]
        self.action_space = spaces.Discrete(len(self.candidate_goals))

        # 4) Manager observation space: We'll observe the average (x, y) among UAVs
        low = np.array([0, 0], dtype=np.float32)
        high = np.array([self.worker_env.grid_size - 1, self.worker_env.grid_size - 1], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, shape=(2,), dtype=np.float32)

        # 5) Hierarchical hyperparameters
        self.max_manager_steps = self.config.get("max_manager_steps", 10)
        self.max_sub_steps = self.config.get("max_sub_steps", 20)  # how many worker steps per manager action

        self.manager_step_count = 0

        # We'll track LSTM states for each agent in the worker
        self.rnn_states = {}

    def reset(self, *, seed=None, options=None):
        """
        Gymnasium-style reset:
          Returns:
            obs (np.ndarray): manager-level observation
            info (dict): extra info
        """
        super().reset(seed=seed)
        self.manager_step_count = 0

        # Reset the worker env (which is multi-agent)
        obs_dict, _info = self.worker_env.reset(seed=seed, options=options)

        # Initialize worker's LSTM states
        worker_policy = self.worker_trainer.get_policy(self.worker_policy_id)
        self.rnn_states = {}
        for agent_id in obs_dict.keys():
            self.rnn_states[agent_id] = worker_policy.get_initial_state()

        # Return manager observation + info dict
        manager_obs = self._compute_manager_obs(obs_dict)
        return manager_obs, {}

    def step(self, action):
        """
        Gymnasium-style step:
          Args:
            action (int): sub-goal index
          Returns:
            obs (np.ndarray): manager-level observation
            reward (float): manager reward
            terminated (bool): True if we consider the manager episode "done"
            truncated (bool): True if we truncated (not used much here)
            info (dict): extra info
        """
        self.manager_step_count += 1

        # 1) Manager picks a sub-goal
        chosen_goal = self.candidate_goals[action]
        # Overwrite the worker_env's goals for all UAVs
        self.worker_env.goal_coords = [chosen_goal] * self.worker_env.num_uavs

        # 2) Run sub-episode with worker
        sub_steps = 0
        terminated_worker = False
        ep_collided = False
        ep_reached_goal = False

        obs_dict = self.worker_env._get_obs_dict()
        worker_policy = self.worker_trainer.get_policy(self.worker_policy_id)

        while not terminated_worker and sub_steps < self.max_sub_steps:
            action_dict = {}

            for agent_id, single_obs in obs_dict.items():
                state_in = self.rnn_states[agent_id]
                action_out, new_rnn_state, _ = worker_policy.compute_single_action(
                    single_obs, state_in, explore=False
                )
                action_dict[agent_id] = action_out
                self.rnn_states[agent_id] = new_rnn_state

            next_obs_dict, rew_dict, done_dict, trunc_dict, info_dict = self.worker_env.step(action_dict)
            obs_dict = next_obs_dict
            sub_steps += 1

            # Worker is done if __all__ in either terminated or truncated
            worker_terminated = done_dict["__all__"]
            worker_truncated = trunc_dict["__all__"]
            terminated_worker = (worker_terminated or worker_truncated)

        # Evaluate collisions or goal reached
        for i in range(self.worker_env.num_uavs):
            x, y, battery = self.worker_env.state[i]
            if (x, y) in self.worker_env.obstacle_coords:
                ep_collided = True
            if (x, y) == chosen_goal:
                ep_reached_goal = True

        # 3) Manager reward
        manager_reward = 0.0
        if ep_collided:
            manager_reward -= 5.0
        if ep_reached_goal:
            manager_reward += 5.0

        # manager is "terminated" if we used up all steps
        manager_terminated = (self.manager_step_count >= self.max_manager_steps)
        manager_truncated = False  # we won't do any horizon-based truncation here

        # next obs
        manager_obs = self._compute_manager_obs(obs_dict)

        return manager_obs, manager_reward, manager_terminated, manager_truncated, {}

    def _compute_manager_obs(self, obs_dict):
        """
        Compute average (x, y) among all UAVs in worker_env state.
        """
        x_sum, y_sum = 0.0, 0.0
        for i in range(self.worker_env.num_uavs):
            x, y, _batt = self.worker_env.state[i]
            x_sum += x
            y_sum += y
        avg_x = x_sum / self.worker_env.num_uavs
        avg_y = y_sum / self.worker_env.num_uavs
        return np.array([avg_x, avg_y], dtype=np.float32)