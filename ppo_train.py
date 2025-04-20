# File: ppo_train_large.py

import os
import csv
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
    """
    env = MultiUAVEnv({"scenario_type": scenario})
    collisions = 0
    successes = 0
    total_steps = 0
    total_battery_used = 0.0

    # We'll reference the trained policy
    policy = trainer.get_policy("shared_policy")

    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = {"__all__": False}
        ep_collided = False
        ep_success = True
        ep_steps = 0
        ep_battery_used = 0.0

        while not done["__all__"]:
            action_dict = {}
            for agent_id, single_obs in obs.items():
                action, _, _ = policy.compute_single_action(
                    single_obs,
                    explore=False  # or True if you want exploration
                )
                action_dict[agent_id] = action

            obs, rew, done, truncated, info = env.step(action_dict)
            ep_steps += 1

            # Check collisions / battery usage
            for i in range(env.num_uavs):
                x, y, battery = env.state[i]
                # collision?
                if battery <= 0 and not env._is_at_goal(i):
                    ep_collided = True
                    ep_success = False
                if (x, y) in env.obstacle_coords:
                    ep_collided = True
                    ep_success = False

        if ep_collided:
            collisions += 1
        if ep_success:
            successes += 1

        # battery usage
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

    # Save to CSV
    eval_csv = os.path.join(ppo_folder, "ppo_evaluation_large.csv")
    with open(eval_csv, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["scenario", "episodes", "success_rate", "collision_rate",
                         "avg_steps_per_episode", "avg_battery_used_per_uav"])
        writer.writerow([scenario, n_episodes, success_rate, collision_rate, avg_steps, avg_battery_used])


if __name__ == "__main__":
    # 1) Initialize Ray
    ray.init()

    # 2) Scenario = "large"
    scenario = "large"

    # 3) Register environment
    register_env("multi_uav_env", env_creator)

    # 4) Create PPOConfig
    config = PPOConfig()

    # Turn off new connector pipeline
    config.api_stack(
        enable_rl_module_and_learner=False,
        enable_env_runner_and_connector_v2=False
    )

    config.env = "multi_uav_env"
    config.env_config = {
        "scenario_type": scenario,
    }
    config.disable_env_checking = True

    # Single (shared) policy
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

    # No LSTM => MLP
    config.model = {
        "use_lstm": False,
        "max_seq_len": 1,
        "fcnet_hiddens": [256, 256],
    }

    # Workers
    config.num_workers = 2
    config.num_gpus = 0

    # Build the trainer
    trainer = config.build()

    # Where to store logs
    ppo_folder = "/Users/alparslanguzey/Desktop/CDRP/PPO"
    os.makedirs(ppo_folder, exist_ok=True)

    # CSV for training => "ppo_training_results_large.csv"
    train_csv_file = os.path.join(ppo_folder, "ppo_training_results_large.csv")
    file_exists = os.path.isfile(train_csv_file)

    # Optional: save initial checkpoint
    initial_ckpt = trainer.save()
    print(f"[{scenario.upper()}] Initial checkpoint saved at: {initial_ckpt}")

    with open(train_csv_file, mode="a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["iteration", "mean_reward"])

        total_iterations = 200
        for i in range(total_iterations):
            result = trainer.train()
            mean_reward = float("nan")
            # Old pipeline location for mean reward
            if "env_runners" in result and "episode_reward_mean" in result["env_runners"]:
                mean_reward = result["env_runners"]["episode_reward_mean"]

            print(f"[{scenario.upper()}] Iter {i} - Mean Reward: {mean_reward:.2f}")
            writer.writerow([i, mean_reward])

    # Final checkpoint
    final_ckpt = trainer.save()
    print(f"[{scenario.upper()}] Final checkpoint saved at: {final_ckpt}")

    # Evaluate final policy => "ppo_evaluation_large.csv"
    evaluate_final_policy(
        trainer,
        scenario=scenario,
        n_episodes=100,
        ppo_folder=ppo_folder
    )

    # Done
    ray.shutdown()