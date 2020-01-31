import gym
import numpy as np
from time import sleep

from sarsa import SARSA
from utils import visualise_q_table, visualise_policy


CONFIG = {
    "env": "FrozenLakeNotSlippery-v0",
    "target_reward": 1,
    "eval_solved_goal": 10,
    "total_eps": 1000,
    "eval_episodes": 1,
    "eval_freq": 10,
    "gamma": 0.99,
    "alpha": 0.1,
    "epsilon": 0.9,
}

RENDER = False

def evaluate(env, config, q_table, eval_episodes=10, render=False, output=True):
    """
    Evaluate configuration of SARSA on given environment initialised with given Q-table

    :param env (gym.Env): environment to execute evaluation on
    :param config (Dict[str, float]): configuration dictionary containing hyperparameters
    :param q_table (Dict[(Obs, Act), float]): Q-table mapping observation-action to Q-values
    :param eval_episodes (int): number of evaluation episodes
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
    for eps_num in range(eval_episodes):
        obs = env.reset()
        if render:
            env.render()
            sleep(1)
        episodic_reward = 0
        done = False

        while not done:
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
        print(f"EVALUATION: MEAN REWARD OF {mean_reward}")
        if mean_reward == 1.0:
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
    max_steps = config["total_eps"]
    
    total_reward = 0
    evaluation_reward_means = []
    evaluation_reward_stds = []
    eval_solved = 0

    for eps_num in range(config["total_eps"]):
        obs = env.reset()
        episodic_reward = 0
        done = False

        # take first action
        act = agent.act(obs)

        while not done:
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
                    eval_episodes=config["eval_episodes"],
                    render=RENDER,
                    output=output
            )
            evaluation_reward_means.append(mean_reward)
            evaluation_reward_stds.append(std_reward)

            if mean_reward >= config["target_reward"]:
                eval_solved += 1
                if output:
                    print(f"Reached reward {mean_reward} >= {config['target_reward']} (target reward)")
                if eval_solved == config["eval_solved_goal"]:
                    if output:
                        print(f"Solved evaluation {eval_solved} times in a row --> terminate training")
                    break
            else:
                eval_solved = 0

    return total_reward, evaluation_reward_means, evaluation_reward_stds, agent.q_table


if __name__ == "__main__":
    env = gym.make(CONFIG["env"])
    
    total_reward, _, _, q_table = train(env, CONFIG)
    # print("Q-table:")
    # visualise_q_table(q_table)
    print()
    visualise_policy(q_table)
