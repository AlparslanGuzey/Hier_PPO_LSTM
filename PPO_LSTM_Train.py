# File: train.py

import os
import csv
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env

from Environment import MultiUAVEnv

def env_creator(config):
    return MultiUAVEnv(config)

def policy_mapping_fn(agent_id, *args, **kwargs):
    return "shared_policy"


def evaluate_final_policy(trainer, scenario, n_episodes=100, ppo_folder="/path/to/PPO"):
    env = MultiUAVEnv({"scenario_type": scenario})
    collisions = 0
    successes = 0
    total_steps = 0
    total_battery_used = 0.0

    # We'll reference the trained policy once
    policy = trainer.get_policy("shared_policy")

    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = {"__all__": False}
        ep_collided = False
        ep_success = True
        ep_steps = 0
        ep_battery_used = 0.0

        # 1) Initialize each agent's LSTM state
        rnn_states = {}
        for agent_id in obs.keys():
            rnn_states[agent_id] = policy.get_initial_state()

        while not done["__all__"]:
            action_dict = {}

            # 2) For each UAV, pass in (observation, rnn_states[agent_id]) -> get (action, new_state)
            for agent_id, single_obs in obs.items():
                state_in = rnn_states[agent_id]
                # compute_single_action requires (obs, state_in, seq_lens=1)
                action, new_state, _ = policy.compute_single_action(
                    single_obs,
                    state_in,
                    explore=False  # or True, if you want exploration
                )
                action_dict[agent_id] = action
                rnn_states[agent_id] = new_state  # update the hidden state

            # Step the environment
            obs, rew, done, truncated, info = env.step(action_dict)
            ep_steps += 1

            # Check collisions/out-of-battery
            for i in range(env.num_uavs):
                x, y, battery = env.state[i]
                if battery <= 0 and not env._is_at_goal(i):
                    ep_collided = True
                    ep_success = False
                if (x, y) in env.obstacle_coords:
                    ep_collided = True
                    ep_success = False

        # Count collisions, successes, battery usage, etc.
        if ep_collided:
            collisions += 1
        if ep_success:
            successes += 1

        # Battery usage: sum(initial - final) for all UAVs
        for i in range(env.num_uavs):
            _, _, battery = env.state[i]
            ep_battery_used += (env.initial_battery - battery)

        total_steps += ep_steps
        total_battery_used += ep_battery_used

    # Final metrics
    success_rate = successes / n_episodes
    collision_rate = collisions / n_episodes
    avg_steps = total_steps / n_episodes
    avg_battery_used = total_battery_used / (n_episodes * env.num_uavs)

    print(f"\n=== Final Policy Evaluation for scenario: {scenario.upper()} ===")
    print(f"Success Rate:        {success_rate*100:.1f}%")
    print(f"Collision Rate:      {collision_rate*100:.1f}%")
    print(f"Avg Steps/Episode:   {avg_steps:.2f}")
    print(f"Avg Battery Used/UAV:{avg_battery_used:.2f}\n")

    # Save to a second CSV file, e.g. "evaluation_medium.csv"
    eval_csv = os.path.join(ppo_folder, f"evaluation_{scenario}.csv")
    with open(eval_csv, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["scenario", "episodes", "success_rate", "collision_rate",
                         "avg_steps_per_episode", "avg_battery_used_per_uav"])
        writer.writerow([scenario, n_episodes, success_rate, collision_rate, avg_steps, avg_battery_used])


if __name__ == "__main__":
    # -----------------------------
    # 1) Initialize Ray
    ray.init()

    # Decide the scenario: "small", "medium", or "large"
    scenario = "large"

    # 2) Register environment
    register_env("multi_uav_env", env_creator)

    # 3) Create a PPOConfig
    config = PPOConfig()

    # Turn off new API stack (old pipeline):
    config.api_stack(
        enable_rl_module_and_learner=False,
        enable_env_runner_and_connector_v2=False
    )

    # Environment settings
    config.env = "multi_uav_env"
    config.env_config = {
        "scenario_type": scenario,
    }
    config.disable_env_checking = True

    # Multi-agent
    config.policies = {
        "shared_policy": (None, None, None, {}),
    }
    config.policy_mapping_fn = policy_mapping_fn

    # Basic PPO config
    config.framework_str = "torch"
    config.train_batch_size = 2000
    config.sgd_minibatch_size = 128
    config.num_sgd_iter = 5
    config.lr = 1e-5
    config.clip_param = 0.05
    config.grad_clip = 0.5
    config.use_gae = True
    config.lambda_ = 0.95
    config.entropy_coeff = 0.01
    config.vf_clip_param = 1.0
    config.vf_share_layers = False

    # LSTM model config
    config.model = {
        "use_lstm": True,
        "lstm_cell_size": 256,
        "max_seq_len": 20,
        "fcnet_hiddens": [],
        "lstm_use_prev_action": False,
        "lstm_use_prev_reward": False,
    }

    # 4) Multi-worker
    config.num_workers = 2
    config.num_gpus = 0

    # 5) Build the trainer
    trainer = config.build()

    # ---- A) Save an initial checkpoint (optional) ----
    initial_ckpt = trainer.save()
    print(f"[{scenario.upper()}] Initial checkpoint saved at: {initial_ckpt}")

    # ---- B) Prepare the CSV file for results in PPO folder ----
    ppo_folder = "/Users/alparslanguzey/Desktop/CDRP/PPO"
    csv_file = os.path.join(ppo_folder, f"training_results_{scenario}.csv")

    file_exists = os.path.isfile(csv_file)
    with open(csv_file, mode="a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            # Write header for training logs
            writer.writerow(["iteration", "mean_reward"])

        # 6) Train for some number of iterations (e.g., 40 for demonstration)
        total_iterations = 200
        for i in range(total_iterations):
            result = trainer.train()
            mean_reward = float("nan")
            if "env_runners" in result and "episode_reward_mean" in result["env_runners"]:
                mean_reward = result["env_runners"]["episode_reward_mean"]

            print(f"[{scenario.upper()}] Iter {i} - Mean Reward: {mean_reward:.2f}")
            writer.writerow([i, mean_reward])

    # ---- C) Save the final checkpoint after training completes ----
    final_ckpt = trainer.save()
    print(f"[{scenario.upper()}] Final checkpoint saved at: {final_ckpt}")

    # ---- D) Evaluate final policy (custom loop) ----
    #     e.g., 100 evaluation episodes -> success/collision rates, etc.
    evaluate_final_policy(trainer, scenario=scenario, n_episodes=100, ppo_folder=ppo_folder)

    # Done. Shutdown Ray.
    ray.shutdown()