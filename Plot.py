import pandas as pd
import matplotlib.pyplot as plt

# 1. Read the CSV data for each method.
ppo_data = pd.read_csv("ppo_training_results_large.csv")
lstm_data = pd.read_csv("training_results_large.csv")
hier_data = pd.read_csv("hierarchical_train_results_large.csv")

# 2. Create the figure and plot each method's line.
plt.figure(figsize=(8, 6))  # optional: control figure size
plt.plot(ppo_data["iteration"], ppo_data["mean_reward"], label="PPO")
plt.plot(lstm_data["iteration"], lstm_data["mean_reward"], label="PPO + LSTM")
plt.plot(hier_data["iteration"], hier_data["mean_reward"], label="Hierarchical PPO + LSTM")

# 3. Label axes, add title, legend.
plt.xlabel("Iteration")
plt.ylabel("Mean Reward")
plt.title("Convergence in the Large Scenario")
plt.legend()

# 4. Display the plot.
plt.show()