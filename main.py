import random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
k_factor = 20


def win_probability(rating, opponent_rating):
    return 1 / (1 + 10 ** ((opponent_rating - rating) / 400))


def update_elo(rating, opponent_rating, result, k):
    return rating + k * (result - win_probability(rating, opponent_rating))


def win_real(rating_a, rating_b):
    return 1.0 if random.random() < win_probability(rating_a, rating_b) else 0.0


def play_game(account1_elo, account2_elo, account1_eloband=(-50, 50), account2_eloband=(-50, 50), k=k_factor):
    opponent_a_elo = account1_elo + random.randint(account1_eloband[0], account1_eloband[1])
    opponent_b_elo = account2_elo + random.randint(account2_eloband[0], account2_eloband[1])

    result_for_a = win_real(opponent_a_elo, opponent_b_elo)

    account1_elo = update_elo(account1_elo, opponent_a_elo, 1 - result_for_a, k)
    account2_elo = update_elo(account2_elo, opponent_b_elo, result_for_a, k)

    return account1_elo, account2_elo


def run_simulation(num_games, account1_eloband, account2_eloband, k=k_factor, seed=None, start_elo=1000):
    if seed is not None:
        random.seed(seed)

    account1_elo = float(start_elo)
    account2_elo = float(start_elo)
    history1, history2 = [account1_elo], [account2_elo]

    for _ in range(num_games):
        account1_elo, account2_elo = play_game(account1_elo, account2_elo, account1_eloband, account2_eloband, k=k)
        history1.append(account1_elo)
        history2.append(account2_elo)


    return history1, history2


def run_more_simulations(num_trials, num_games, account1_eloband, account2_eloband, start_elo=1000, k=k_factor, seed=None):
    if seed is not None:
        random.seed(seed)

    all_h1 = np.empty((num_trials, num_games + 1))
    all_h2 = np.empty((num_trials, num_games + 1))
    for t in range(num_trials):
        h1, h2 = run_simulation(num_games, account1_eloband, account2_eloband, k=k, start_elo=start_elo)
        all_h1[t] = h1
        all_h2[t] = h2
    return all_h1, all_h2

if __name__ == "__main__":
    NUM_GAMES = 3000
    N_TRIALS = 150

    scenarios = {
        "Even bands, 50 to 0": ((0, 50), (0, 50)),
        "Even bands, -50 to 0": ((-50, 0), (-50, 0)),
    }

    fig, axes = plt.subplots(1, len(scenarios), figsize=(12, 5), sharey=True)
    x = range(NUM_GAMES + 1)

    for ax, (title, (band1, band2)) in zip(axes, scenarios.items()):
        all_h1, all_h2 = run_more_simulations(N_TRIALS, NUM_GAMES, band1, band2)
        mean1, mean2 = all_h1.mean(axis=0), all_h2.mean(axis=0)
        std1, std2 = all_h1.std(axis=0), all_h2.std(axis=0)

        ax.plot(x, all_h1[0], color="C0", alpha=0.25, linewidth=0.8)
        ax.plot(x, all_h2[0], color="C1", alpha=0.25, linewidth=0.8)

        ax.plot(x, mean1, color="C0", label="Account 1 (avg)", linewidth=2)
        ax.plot(x, mean2, color="C1", label="Account 2 (avg)", linewidth=2)
        ax.fill_between(x, mean1 - std1, mean1 + std1, color="C0", alpha=0.12)
        ax.fill_between(x, mean2 - std2, mean2 + std2, color="C1", alpha=0.12)

        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Games")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

        print(f"{title}")
        print(f"  Account 1: {mean1[0]:.0f} -> {mean1[-1]:.0f} avg over {N_TRIALS} runs ({mean1[-1]-mean1[0]:+.0f})")
        print(f"  Account 2: {mean2[0]:.0f} -> {mean2[-1]:.0f} avg over {N_TRIALS} runs ({mean2[-1]-mean2[0]:+.0f})")
        print()

    axes[0].set_ylabel("Elo")
    plt.tight_layout()
    plt.savefig("result", dpi=150)
    print("done")
