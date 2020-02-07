import gym
import matplotlib.pyplot as plt
import numpy as np
import random
from time import sleep

from sarsa import SARSA
from utils import visualise_q_table, visualise_policy
from plot_utils import plot_timesteps, plot_timesteps_shaded

CONFIG_SLIPPERY = {
    "env": "FrozenLake-v0",
    "total_eps": 100000,
    "max_episode_steps": 100,
    "eval_episodes": 100,
    "eval_freq": 1000,
    "gamma": 0.99,
    "alpha": 1e-1,
    "epsilon": 0.9,
}

CONFIG_NOTSLIPPERY = {
    "env": "FrozenLakeNotSlippery-v0",
    "total_eps": 500,
    "max_episode_steps": 100,
    "eval_episodes": 100,
    "eval_freq": 5,
    "gamma": 0.99,
    "alpha": 0.1,
    "epsilon": 0.9,
}

# CONFIG = CONFIG_SLIPPERY
CONFIG = CONFIG_NOTSLIPPERY

RENDER = False
SEEDS = [i for i in range(10)]


def evaluate(env, config, q_table, episode, render=False, output=True):
    """
    Evaluate configuration of SARSA on given environment initialised with given Q-table

    :param env (gym.Env): environment to execute evaluation on
    :param config (Dict[str, float]): configuration dictionary containing hyperparameters
    :param q_table (Dict[(Obs, Act), float]): Q-table mapping observation-action to Q-values
    :param episode (int): episodes of training completed
    :param render (bool): flag whether evaluation runs should be rendered
    :param output (bool): flag whether mean evaluation performance should be printed
    :return (float, float): mean and standard deviation of reward received over episodes
    """
    eval_agent = SARSA(
            num_acts=env.action_space.n,
            gamma=config["gamma"],
            epsilon=0.0, 
            alpha=config["alpha"],
    )
    eval_agent.q_table = q_table
    episodic_rewards = []
    for eps_num in range(config["eval_episodes"]):
        obs = env.reset()
        if render:
            env.render()
            sleep(1)
        episodic_reward = 0
        done = False

        steps = 0
        while not done and steps <= config["max_episode_steps"]:
            steps += 1
            act = eval_agent.act(obs)
            n_obs, reward, done, info = env.step(act)
            if render:
                env.render()
                sleep(1)

            episodic_reward += reward

            obs = n_obs

        episodic_rewards.append(episodic_reward)

    mean_reward = np.mean(episodic_rewards)
    std_reward = np.std(episodic_rewards)

    if output:
        print(f"EVALUATION ({episode}/{CONFIG['total_eps']}): MEAN REWARD OF {mean_reward}")
        if mean_reward >= 0.9:
            print(f"EVALUATION: SOLVED")
        else:
            print(f"EVALUATION: NOT SOLVED!")
    return mean_reward, std_reward


def train(env, config, output=True):
    """
    Train and evaluate SARSA on given environment with provided hyperparameters

    :param env (gym.Env): environment to execute evaluation on
    :param config (Dict[str, float]): configuration dictionary containing hyperparameters
    :param output (bool): flag if mean evaluation results should be printed
    :return (float, List[float], List[float], Dict[(Obs, Act), float]):
        total reward over all episodes, list of means and standard deviations of evaluation
        rewards, final Q-table
    """
    agent = SARSA(
            num_acts=env.action_space.n,
            gamma=config["gamma"],
            epsilon=config["epsilon"],
            alpha=config["alpha"],
    )

    step_counter = 0
    # 100 as estimate of max steps to take in an episode
    max_steps = config["total_eps"] * config["max_episode_steps"]
    
    total_reward = 0
    evaluation_reward_means = []
    evaluation_reward_stds = []
    evaluation_epsilons = []

    for eps_num in range(config["total_eps"]):
        obs = env.reset()
        episodic_reward = 0
        steps = 0
        done = False

        # take first action
        act = agent.act(obs)

        while not done and steps < config["max_episode_steps"]:
            n_obs, reward, done, info = env.step(act)
            step_counter += 1
            episodic_reward += reward

            agent.schedule_hyperparameters(step_counter, max_steps)
            n_act = agent.act(n_obs)
            agent.learn(obs, act, reward, n_obs, n_act, done)

            obs = n_obs
            act = n_act

        total_reward += episodic_reward

        if eps_num > 0 and eps_num % config["eval_freq"] == 0:
            mean_reward, std_reward = evaluate(
                    env,
                    config,
                    agent.q_table,
                    eps_num,
                    render=RENDER,
                    output=output
            )
            evaluation_reward_means.append(mean_reward)
            evaluation_reward_stds.append(std_reward)
            evaluation_epsilons.append(agent.epsilon)

    return total_reward, evaluation_reward_means, evaluation_reward_stds, evaluation_epsilons, agent.q_table

if __name__ == "__main__":
    env = gym.make(CONFIG["env"])
    eval_reward_means = []
    eval_reward_stds = []
    eval_epsilons = []
    for seed in SEEDS:
        print(f"SARSA Training for seed={seed}")
        random.seed(seed)
        np.random.seed(seed)
        
        reward_means = []
        reward_stds = []
        total_reward, reward_means, reward_stds, epsilons, q_table = train(env, CONFIG, output=False)
        # print("Q-table:")
        # visualise_q_table(q_table)
        visualise_policy(q_table)
        eval_reward_means.append(reward_means)
        eval_reward_stds.append(reward_stds)
        eval_epsilons.append(epsilons)

    eval_reward_means = np.array(eval_reward_means).mean(axis=0)
    eval_reward_stds = np.array(eval_reward_stds).mean(axis=0)
    eval_epsilons = np.array(eval_epsilons).mean(axis=0)
    plot_timesteps_shaded(
        eval_reward_means,
        eval_reward_stds,
        CONFIG["eval_freq"],
        "SARSA Evaluation Returns",
        "Timesteps",
        "Mean Evaluation Returns",
        "SARSA",
    )

    plot_timesteps(
        eval_epsilons,
        CONFIG["eval_freq"],
        "SARSA Epsilon Decay",
        "Timesteps",
        "Epsilon",
        "SARSA",
    )

    plt.show()
