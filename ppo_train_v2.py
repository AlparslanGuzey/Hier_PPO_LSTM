# File: ppo_train_large.py

import os
import csv
import json  # For trajectory logging output
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env

# Replace with your actual import path or environment file:
from Environment import MultiUAVEnv


def env_creator(config):
    return MultiUAVEnv(config)


def policy_mapping_fn(agent_id, *args, **kwargs):
    return "shared_policy"


def evaluate_final_policy(trainer, scenario, n_episodes=100, ppo_folder="/path/to/PPO"):
    """
    Evaluates the trained policy on `n_episodes` episodes, measuring
    collision rate, success rate, average steps, and battery usage.
    Also logs (x, y) positions and battery at each timestep for each UAV
    in a JSON file for later trajectory/battery analysis.
    """
    env = MultiUAVEnv({"scenario_type": scenario})
    collisions = 0
    successes = 0
    total_steps = 0
    total_battery_used = 0.0

    # For trajectory logging across episodes
    all_trajectories = []

    # We'll reference the trained policy
    policy = trainer.get_policy("shared_policy")

    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = {"__all__": False}
        ep_collided = False
        ep_success = True
        ep_steps = 0
        ep_battery_used = 0.0

        # Per-episode logging structure
        episode_trajectories = {agent_id: [] for agent_id in obs.keys()}

        while not done["__all__"]:
            action_dict = {}
            for agent_id, single_obs in obs.items():
                action, _, _ = policy.compute_single_action(
                    single_obs,
                    explore=False  # or True if you'd like exploration
                )
                action_dict[agent_id] = action

            obs, rew, done, truncated, info = env.step(action_dict)
            ep_steps += 1

            # Log each UAV's (x, y, battery)
            for i, agent_id in enumerate(obs.keys()):
                x, y, battery = env.state[i]
                episode_trajectories[agent_id].append((x, y, battery))

            # Check collisions / battery usage
            for i in range(env.num_uavs):
                x, y, battery = env.state[i]
                if battery <= 0 and not env._is_at_goal(i):
                    ep_collided = True
                    ep_success = False
                if (x, y) in env.obstacle_coords:
                    ep_collided = True
                    ep_success = False

        # Store trajectories from this episode
        all_trajectories.append(episode_trajectories)

        # Collisions & successes
        if ep_collided:
            collisions += 1
        if ep_success:
            successes += 1

        # Battery usage
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

    # Save evaluation metrics to CSV
    eval_csv = os.path.join(ppo_folder, "ppo_evaluation_large.csv")
    with open(eval_csv, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["scenario", "episodes", "success_rate", "collision_rate",
                         "avg_steps_per_episode", "avg_battery_used_per_uav"])
        writer.writerow([scenario, n_episodes, success_rate, collision_rate,
                         avg_steps, avg_battery_used])

    # Save trajectories & battery logs to JSON
    trajectories_file = os.path.join(ppo_folder, "ppo_trajectories_large.json")
    with open(trajectories_file, "w") as tf:
        json.dump(all_trajectories, tf, indent=4)
    print(f"Trajectory data (positions + battery) saved to {trajectories_file}")


if __name__ == "__main__":
    # 1) Initialize Ray
    ray.init()

    # 2) Scenario
    scenario = "large"

    # 3) Register your environment
    register_env("multi_uav_env", env_creator)

    # 4) Build the PPOConfig using new builder-style methods
    config = (
        PPOConfig()
        # Environment setup
        .environment(
            env="multi_uav_env",
            env_config={"scenario_type": "large"},
            disable_env_checking=True
        )
        # Multi-agent config
        .multi_agent(
            policies={
                "shared_policy": (None, None, None, {}),
            },
            policy_mapping_fn=policy_mapping_fn
        )
        # Torch framework
        .framework("torch")
        # Rollout workers (instead of config.num_workers)
        .rollouts(num_rollout_workers=2)
        # Resource usage
        .resources(num_gpus=0)
        # Training hyperparameters
        .training(
            train_batch_size=2000,
            sgd_minibatch_size=128,
            num_sgd_iter=5,
            lr=1e-5,
            clip_param=0.05,
            grad_clip=0.5,
            use_gae=True,
            lambda_=0.95,
            entropy_coeff=0.01,
            vf_clip_param=1.0,
            # Pass model settings (including vf_share_layers) here
            model={
                "use_lstm": False,
                "max_seq_len": 1,
                "fcnet_hiddens": [256, 256],
                "vf_share_layers": False
            }
        )
    )

    # 5) Build the Trainer
    trainer = config.build()

    # Directory for logs
    ppo_folder = "/Users/alparslanguzey/Desktop/CDRP/PPO"
    os.makedirs(ppo_folder, exist_ok=True)

    # CSV for training results
    train_csv_file = os.path.join(ppo_folder, "ppo_training_results_large.csv")
    file_exists = os.path.isfile(train_csv_file)

    # Optional: save initial checkpoint
    initial_ckpt = trainer.save()
    print(f"[{scenario.upper()}] Initial checkpoint saved at: {initial_ckpt}")

    # Train and record average rewards
    total_iterations = 10
    with open(train_csv_file, mode="a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["iteration", "mean_reward"])

        for i in range(total_iterations):
            result = trainer.train()
            mean_reward = float("nan")

            # If the result dict contains "episode_reward_mean"
            # in the old pipeline location:
            if "env_runners" in result and "episode_reward_mean" in result["env_runners"]:
                mean_reward = result["env_runners"]["episode_reward_mean"]

            print(f"[{scenario.upper()}] Iter {i} - Mean Reward: {mean_reward:.2f}")
            writer.writerow([i, mean_reward])

    # Final checkpoint
    final_ckpt = trainer.save()
    print(f"[{scenario.upper()}] Final checkpoint saved at: {final_ckpt}")

    # 6) Evaluate the final policy
    evaluate_final_policy(
        trainer,
        scenario=scenario,
        n_episodes=100,
        ppo_folder=ppo_folder
    )

    # Done
    ray.shutdown()