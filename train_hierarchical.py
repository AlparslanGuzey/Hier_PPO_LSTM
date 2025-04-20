# File: train_hierarchical_all.py

import os
import csv
import ray
from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPO, PPOConfig

from hiearchical_environment import MultiUAVEnv
from manager_env import ManagerEnv

# The directory where you want CSV logs:
PPO_FOLDER = "/Users/alparslanguzey/Desktop/CDRP/PPO"


def worker_env_creator(config):
    return MultiUAVEnv(config)


def manager_env_creator(config):
    return ManagerEnv(config)


def evaluate_final_policy(manager_trainer, scenario, n_episodes=50):
    """
    Evaluate the final hierarchical policy (Manager) in ManagerEnv and compute:
      - Success rate
      - Collision rate
      - Avg steps/episode
      - Avg battery used per UAV

    The environment is built from manager_trainer.config["env_config"], which includes
    'worker_ckpt_path'. We'll run n_episodes, gather stats, then print and save them
    in a CSV file:
        /Users/alparslanguzey/Desktop/CDRP/PPO/hierarchical_evaluation_results_{scenario}.csv
    """
    # Reuse the manager_trainer's environment config (so we have the correct worker_ckpt_path)
    env_config = manager_trainer.config["env_config"]
    env = ManagerEnv(env_config)

    collisions = 0
    successes = 0
    total_steps = 0
    total_battery_used = 0.0

    # We'll reference the manager policy
    manager_policy = manager_trainer.get_policy("default_policy")

    for ep in range(n_episodes):
        obs, info = env.reset()  # ManagerEnv reset
        manager_terminated = False
        manager_truncated = False
        ep_steps = 0
        ep_collided = False
        ep_success = True
        ep_battery_used = 0.0

        # Run Manager steps until done
        while not manager_terminated and not manager_truncated:
            action, _states, _info = manager_policy.compute_single_action(obs, explore=False)
            obs, rew, manager_terminated, manager_truncated, info = env.step(action)
            ep_steps += 1

        # Once Manager episode ends, check collisions & battery usage from Worker environment
        for i in range(env.worker_env.num_uavs):
            x, y, battery = env.worker_env.state[i]
            # Collision if out of battery but not at goal
            if battery <= 0 and not env.worker_env._is_at_goal(i):
                ep_collided = True
                ep_success = False
            # Or if location is an obstacle
            if (x, y) in env.worker_env.obstacle_coords:
                ep_collided = True
                ep_success = False

        if ep_collided:
            collisions += 1
        if ep_success:
            successes += 1

        # Battery usage for this episode
        for i in range(env.worker_env.num_uavs):
            _, _, batt = env.worker_env.state[i]
            ep_battery_used += (env.worker_env.initial_battery - batt)

        total_steps += ep_steps
        total_battery_used += ep_battery_used

    # Final metrics
    success_rate = successes / n_episodes if n_episodes else 0.0
    collision_rate = collisions / n_episodes if n_episodes else 0.0
    avg_steps = total_steps / n_episodes if n_episodes else 0.0

    avg_battery_used = 0.0
    if n_episodes and env.worker_env.num_uavs:
        avg_battery_used = total_battery_used / (n_episodes * env.worker_env.num_uavs)

    # Print
    print(f"\n=== Final Policy Evaluation for scenario: {scenario.upper()} ===")
    print(f"Success Rate:        {success_rate * 100:.1f}%")
    print(f"Collision Rate:      {collision_rate * 100:.1f}%")
    print(f"Avg Steps/Episode:   {avg_steps:.2f}")
    print(f"Avg Battery Used/UAV:{avg_battery_used:.2f}\n")

    # Save to CSV
    eval_csv = os.path.join(PPO_FOLDER, f"hierarchical_evaluation_results_{scenario}.csv")
    with open(eval_csv, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["scenario", "episodes", "success_rate", "collision_rate",
                         "avg_steps_per_episode", "avg_battery_used_per_uav"])
        writer.writerow([
            scenario, n_episodes, success_rate, collision_rate, avg_steps, avg_battery_used
        ])


def train_and_evaluate_scenario(scenario, n_iterations=5, n_eval_episodes=50):
    """
    Train hierarchical PPO on the given scenario, logging iteration results to CSV
    with lines like: [SCENARIO] Iter 0 - Mean Reward: X
    Then evaluate final manager policy and log results to another CSV.
    """
    # 1) Worker Env
    register_env("worker_uav_env", worker_env_creator)

    worker_config = (
        PPOConfig()
        .rl_module(_enable_rl_module_api=False)
        .training(_enable_learner_api=False)
        .environment(env="worker_uav_env", env_config={"scenario_type": scenario})
        .framework("torch")
        .training(
            model={
                "use_lstm": True,
                "lstm_cell_size": 256,
                "max_seq_len": 20,
            },
            train_batch_size=2000,
            sgd_minibatch_size=128,
            num_sgd_iter=5,
            lr=1e-4,
        )
        .resources(num_gpus=0, num_cpus_per_worker=1)
        .rollouts(num_rollout_workers=1)
    )

    worker_trainer = worker_config.build()

    # CSV for worker training logs
    train_csv_path = os.path.join(PPO_FOLDER, f"hierarchical_train_results_{scenario}.csv")
    with open(train_csv_path, mode="w", newline="") as train_f:
        train_writer = csv.writer(train_f)
        train_writer.writerow(["iteration", "mean_reward"])

        for i in range(n_iterations):
            result = worker_trainer.train()
            mean_rew = result["episode_reward_mean"]
            print(f"[{scenario.upper()}] Iter {i} - Mean Reward: {mean_rew:.2f}")
            train_writer.writerow([i, mean_rew])

    worker_ckpt = worker_trainer.save()
    print(f"[{scenario.upper()}] Final worker checkpoint: {worker_ckpt}")

    # Extract path from Ray's new checkpoint object if needed
    if hasattr(worker_ckpt, "checkpoint") and hasattr(worker_ckpt.checkpoint, "path"):
        worker_ckpt_path = worker_ckpt.checkpoint.path
    else:
        worker_ckpt_path = worker_ckpt

    # 2) Manager Env
    register_env("manager_env", manager_env_creator)

    manager_env_cfg = {
        "scenario_type": scenario,
        "worker_ckpt_path": worker_ckpt_path,
        "worker_env_config": {"scenario_type": scenario},
        "worker_policy_rllib_config": worker_config.to_dict(),
        "worker_policy_id": "default_policy",
        "max_manager_steps": 10,
        "max_sub_steps": 20,
    }

    manager_config = (
        PPOConfig()
        .rl_module(_enable_rl_module_api=False)
        .training(_enable_learner_api=False)
        .environment(env="manager_env", env_config=manager_env_cfg)
        .framework("torch")
        .training(
            model={
                "use_lstm": False,
                "fcnet_hiddens": [256, 256],
            },
            train_batch_size=800,
            sgd_minibatch_size=128,
            num_sgd_iter=5,
            lr=1e-4,
        )
        .resources(num_gpus=0, num_cpus_per_worker=1)
        .rollouts(num_rollout_workers=1)
    )

    manager_trainer = manager_config.build()

    manager_iterations = 50
    manager_train_csv_path = os.path.join(PPO_FOLDER, f"hierarchical_train_results_manager_{scenario}.csv")
    with open(manager_train_csv_path, mode="w", newline="") as mtrain_f:
        mtrain_writer = csv.writer(mtrain_f)
        mtrain_writer.writerow(["iteration", "mean_reward"])

        for i in range(manager_iterations):
            result = manager_trainer.train()
            mean_rew = result["episode_reward_mean"]
            print(f"[{scenario.upper()}][MANAGER] Iter {i} - Mean Reward: {mean_rew:.2f}")
            mtrain_writer.writerow([i, mean_rew])

    manager_ckpt = manager_trainer.save()
    print(f"[{scenario.upper()}] Final manager checkpoint: {manager_ckpt}")

    # *** IMPORTANT: Actually call the final evaluation now ***
    evaluate_final_policy(manager_trainer, scenario, n_eval_episodes)


def main():
    ray.init()

    # Run each scenario in a loop
    for scenario in ["small", "medium", "large"]:
        train_and_evaluate_scenario(scenario, n_iterations=200, n_eval_episodes=50)

    ray.shutdown()


if __name__ == "__main__":
    main()