import numpy as np
import gymnasium as gym
from gymnasium import spaces
from ray.rllib.env.multi_agent_env import MultiAgentEnv

class MultiUAVEnv(MultiAgentEnv):
    """
    A multi-agent environment for multiple UAVs route planning.
    """

    def __init__(self, config=None):
        super().__init__()
        if config is None:
            config = {}

        # 1) Check for scenario_type
        scenario_type = config.get("scenario_type", None)

        # 2) Provide default scenario-based settings
        default_num_uavs = 4
        default_obstacles = [(3, 3), (4, 5), (7, 2)]
        default_goals = [(8, 8), (2, 7), (1, 1), (9, 0)]

        if scenario_type == "small":
            default_num_uavs = 2
            default_obstacles = [(3, 3), (4, 5)]
            default_goals = [(8, 8), (2, 7)]
        elif scenario_type == "medium":
            default_num_uavs = 4
            default_obstacles = [(3, 3), (4, 5), (7, 2)]
            default_goals = [(8, 8), (2, 7), (1, 1), (9, 0)]
        elif scenario_type == "large":
            default_num_uavs = 6
            default_obstacles = [(3, 3), (4, 5), (7, 2), (5, 8)]
            default_goals = [
                (8, 8), (2, 7), (1, 1), (9, 0), (0, 9), (7, 7)
            ]

        # 3) Assign environment parameters
        self.num_uavs = config.get("num_uavs", default_num_uavs)
        self.grid_size = config.get("grid_size", 10)
        self.max_episode_steps = config.get("max_episode_steps", 60)
        self.initial_battery = config.get("initial_battery", 50)

        # Convert obstacles to a set for fast membership checks
        default_obstacles = set(config.get("obstacle_coords", default_obstacles))
        self.obstacle_coords = default_obstacles

        # Goals
        self.goal_coords = config.get("goal_coords", default_goals)

        # --- REQUIRED FIX: Declare the agent IDs this env supports ---
        # RLlib's new environment checks expect self._agent_ids to be a set of agent IDs
        self._agent_ids = {f"uav_{i}" for i in range(self.num_uavs)}

        # The rest of your existing code
        self.current_step = 0
        self.state = None  # each UAV: [x, y, battery]
        self.prev_dists = []

        # Action space: 0=stay,1=up,2=down,3=left,4=right
        self.action_space = spaces.Discrete(5)

        # Observation space: [x, y, battery, gx, gy]
        low_obs = np.array([0, 0, 0, 0, 0], dtype=np.float32)
        high_obs = np.array([
            self.grid_size - 1, self.grid_size - 1,
            self.initial_battery,
            self.grid_size - 1, self.grid_size - 1
        ], dtype=np.float32)
        self.observation_space = spaces.Box(low=low_obs, high=high_obs, dtype=np.float32)


    def reset(self, *, seed=None, options=None):
        self.current_step = 0
        self.state = []
        self.prev_dists = []

        for i in range(self.num_uavs):
            x = np.random.randint(0, self.grid_size)
            y = np.random.randint(0, self.grid_size)
            while (x, y) in self.obstacle_coords:
                x = np.random.randint(0, self.grid_size)
                y = np.random.randint(0, self.grid_size)

            battery = self.initial_battery
            self.state.append([x, y, battery])

        # Initialize prev_dists for each UAV
        for i in range(self.num_uavs):
            x, y, _ = self.state[i]
            gx, gy = self._get_goal(i)
            dist = abs(x - gx) + abs(y - gy)
            self.prev_dists.append(dist)

        return self._get_obs_dict(), {}


    def step(self, action_dict):
        self.current_step += 1

        # 1) Apply each UAV's action
        for agent_id, act in action_dict.items():
            idx = self._agent_id_to_index(agent_id)
            self._apply_action(idx, act)

        # 2) Compute rewards
        reward_dict = self._compute_rewards()

        # 3) Check terminations
        done_dict = {}
        any_terminated = False
        for i in range(self.num_uavs):
            agent_id = self._index_to_agent_id(i)
            done_i = self._check_done_for_agent(i)
            done_dict[agent_id] = done_i
            if done_i:
                any_terminated = True
        done_dict["__all__"] = any_terminated

        # 4) Check truncated
        truncated_dict = {}
        any_truncated = False
        if self.current_step >= self.max_episode_steps:
            any_truncated = True
        for i in range(self.num_uavs):
            agent_id = self._index_to_agent_id(i)
            truncated_dict[agent_id] = False
        truncated_dict["__all__"] = any_truncated

        # 5) Next obs + info
        obs_dict = self._get_obs_dict()
        info_dict = {
            agent_id: {} for agent_id in done_dict.keys() if agent_id != "__all__"
        }

        return obs_dict, reward_dict, done_dict, truncated_dict, info_dict


    def _apply_action(self, i, action):
        x, y, battery = self.state[i]
        if battery <= 0:
            return

        # 0=stay,1=up,2=down,3=left,4=right
        if action == 1 and y < self.grid_size - 1:
            y += 1
            battery -= 1.0
        elif action == 2 and y > 0:
            y -= 1
            battery -= 1.0
        elif action == 3 and x > 0:
            x -= 1
            battery -= 1.0
        elif action == 4 and x < self.grid_size - 1:
            x += 1
            battery -= 1.0
        else:
            battery -= 0.5  # stay or invalid

        battery = max(0, battery)
        self.state[i] = [x, y, battery]


    def _compute_rewards(self):
        reward_dict = {}
        total_dist = 0.0
        dists = []
        # 1) Collect distances
        for i in range(self.num_uavs):
            x, y, battery = self.state[i]
            gx, gy = self._get_goal(i)
            dist = abs(x - gx) + abs(y - gy)
            dists.append(dist)
            total_dist += dist
        avg_dist = total_dist / self.num_uavs

        # 2) Compute each UAV’s reward
        for i in range(self.num_uavs):
            agent_id = self._index_to_agent_id(i)
            x, y, battery = self.state[i]
            dist = dists[i]

            rew = -0.1  # Step penalty

            # Collision penalty
            if (x, y) in self.obstacle_coords:
                rew -= 0.7

            # Battery penalty
            if battery <= 0 and dist > 0:
                rew -= 0.5

            # Distance penalty
            rew -= 0.02 * dist

            # Bonus if at goal
            if dist == 0:
                rew += 20.0

            # Synergy penalty
            rew -= 0.0005 * avg_dist

            # Progress bonus
            prev_dist = self.prev_dists[i]
            progress = prev_dist - dist
            rew += 0.02 * progress

            # Update prev_dist
            self.prev_dists[i] = dist
            reward_dict[agent_id] = rew

        return reward_dict


    def _check_done_for_agent(self, i):
        x, y, battery = self.state[i]
        # collision
        if (x, y) in self.obstacle_coords:
            return True
        # battery out
        if battery <= 0 and not self._is_at_goal(i):
            return True
        # goal
        if self._is_at_goal(i):
            return True
        return False


    def _get_goal(self, i):
        if i < len(self.goal_coords):
            return self.goal_coords[i]
        return self.goal_coords[-1]


    def _is_at_goal(self, i):
        x, y, _ = self.state[i]
        gx, gy = self._get_goal(i)
        return (x == gx and y == gy)


    def _get_obs_dict(self):
        obs_dict = {}
        for i in range(self.num_uavs):
            agent_id = self._index_to_agent_id(i)
            x, y, battery = self.state[i]
            gx, gy = self._get_goal(i)
            obs = np.array([x, y, battery, gx, gy], dtype=np.float32)
            obs_dict[agent_id] = obs
        return obs_dict

    def _agent_id_to_index(self, agent_id):
        return int(agent_id.split("_")[1])

    def _index_to_agent_id(self, idx):
        return f"uav_{idx}"