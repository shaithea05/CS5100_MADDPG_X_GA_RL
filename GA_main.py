import argparse
import datetime
import os
from time import time

import numpy as np
import torch
from matplotlib import pyplot as plt
from mpe.multiagent import scenarios
from mpe.multiagent.environment import MultiAgentEnv

from GA_agent import Agent
from GA import GeneticAlgorithm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('env', type=str, help='name of the environment',
                        choices=['simple_adversary', 'simple_crypto', 'simple_push', 'simple_reference',
                                 'simple_speaker_listener', 'simple_spread', 'simple_tag',
                                 'simple_world_comm'])
    parser.add_argument('--episode-length', type=int, default=25, help='steps per episode')
    parser.add_argument('--gen-num', type=int, default=1200, help='total number of generations')
    parser.add_argument('--pop-size', type=int, default=50, help='population size')
    parser.add_argument('--mutation-rate', type=float, default=0.1)
    parser.add_argument('--sigma', type=float, default=0.1, help='mutation noise strength')
    parser.add_argument('--save-interval', type=int, default=50)
    
    args = parser.parse_args()
    start = time()

    # Create folders
    env_dir = os.path.join('GA_results', args.env)
    if not os.path.exists(env_dir):
        os.makedirs(env_dir)
    total_files = len(os.listdir(env_dir))
    res_dir = os.path.join(env_dir, f'{total_files + 1}')
    os.makedirs(res_dir)
    model_dir = os.path.join(res_dir, 'model')
    os.makedirs(model_dir)

    # Create env
    scenario = scenarios.load(f'{args.env}.py').Scenario()
    world = scenario.make_world()
    env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)

    # Get dimensions
    obs_dim_list = [obs.shape[0] for obs in env.observation_space]
    act_dim_list = [act.n for act in env.action_space]

    # Initialize GA
    ga = GeneticAlgorithm(Agent, args.pop_size, obs_dim_list, act_dim_list, 
                          mutation_rate=args.mutation_rate, sigma=args.sigma)

    # Track max fitness per generation for plotting
    history_max_fitness = []

    for gen in range(args.gen_num):
        fitness_scores = []

        # Fitness
        for agent_group in ga.population:
            obs = env.reset()
            
            total_interceptions = 0
            total_proximity_reward = 0
            total_survival_reward = 0

            for _ in range(args.episode_length):
                actions = []
                for i, agent in enumerate(agent_group):
                    o = torch.from_numpy(obs[i]).unsqueeze(0).float()
                    a = agent.action(o).squeeze(0).numpy()
                    actions.append(a)
                
                next_obs, rewards, dones, infos = env.step(actions)

                # simple_tag logic: 
                # agents 0,1,2 = predators, agent 3 = prey
                for i in range(len(agent_group)):
                    # Predator Logic
                    if i < 3: 
                        # Interception reward from env
                        total_interceptions += rewards[i]
                        
                        # Plus points for being close
                        dist_to_prey = np.linalg.norm(next_obs[i][4:6])
                        total_proximity_reward += (1.0 / (dist_to_prey + 0.1))

                    # Prey Logic
                    else:
                        # Reward for being far away from predators
                        dist_to_pred1 = np.linalg.norm(next_obs[i][4:6])
                        dist_to_pred2 = np.linalg.norm(next_obs[i][6:8])
                        dist_to_pred3 = np.linalg.norm(next_obs[i][8:10])
                        min_dist = min(dist_to_pred1, dist_to_pred2, dist_to_pred3)
                        
                        total_survival_reward += min_dist

                obs = next_obs

            # Weighted to prioritize interceptions over distance
            adversary_fitness = total_interceptions + (0.1 * total_proximity_reward)
            cooperative_fitness = total_survival_reward
            
            # Total group fitness
            fitness_scores.append(adversary_fitness + cooperative_fitness)

        # Record and printing generations & stats
        max_fit = max(fitness_scores)
        avg_fit = np.mean(fitness_scores)
        history_max_fitness.append(max_fit)
        print(f"Gen {gen} | Max Fitness: {max_fit:.2f} | Avg: {avg_fit:.2f}")

        # Evolution
        ga.evolve(fitness_scores)

        # Saving models
        if gen % args.save_interval == 0:
            best_idx = np.argmax(fitness_scores)
            best_group = ga.population[best_idx]
            torch.save([a.actor.state_dict() for a in best_group], 
                       os.path.join(model_dir, f'gen_{gen}_best.pt'))

    # Final save
    np.save(os.path.join(res_dir, 'fitness_history.npy'), history_max_fitness)

    # Plotting
    plt.figure()
    plt.plot(range(args.gen_num), history_max_fitness, label='Max Fitness')
    plt.xlabel('Generation')
    plt.ylabel('Total Reward')
    plt.title(f'GA Progress on {args.env}')
    plt.legend()
    plt.savefig(os.path.join(res_dir, 'training_plot.png'))

    print(f'Training finished in: {datetime.timedelta(seconds=int(time() - start))}')