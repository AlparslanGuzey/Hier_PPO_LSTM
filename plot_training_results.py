import matplotlib.pyplot as plt

def parse_training_log(logfile_path):
    iters = []
    rewards = []
    with open(logfile_path, 'r') as f:
        for line in f:
            line = line.strip()
            # Each line looks like: "Iter 1 - Mean Reward: -6.10"
            if line.startswith("Iter"):
                # e.g. ["Iter", "1", "-", "Mean", "Reward:", "-6.10"]
                parts = line.split()
                try:
                    iteration_str = parts[1]  # e.g. "1"
                    reward_str = parts[-1]    # e.g. "-6.10"
                    iteration = int(iteration_str)
                    reward = float(reward_str)
                    iters.append(iteration)
                    rewards.append(reward)
                except:
                    # If there's an unexpected line format, just ignore it
                    pass
    return iters, rewards


def plot_training_results(iters, rewards):
    plt.figure()
    # Plot with lines and markers
    plt.plot(iters, rewards, marker='o', linestyle='-')
    plt.title("Mean Reward vs. Training Iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Mean Reward")
    plt.grid(True)
    plt.show()

    # Optionally, save to a file
    # plt.savefig("training_curve.png", dpi=200)


if __name__ == "__main__":
    # Path to your log file
    log_file = "/Users/alparslanguzey/Desktop/CDRP/PPO/training_log.txt"

    iters, rewards = parse_training_log(log_file)

    # Quick check
    print("Parsed iterations:", iters[:10], "...")
    print("Parsed rewards:", rewards[:10], "...")

    # Plot
    plot_training_results(iters, rewards)