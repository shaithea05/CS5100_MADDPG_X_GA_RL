import argparse
import os
import time
import numpy as np
import torch
from matplotlib import pyplot as plt
from mpe.multiagent import scenarios
from mpe.multiagent.environment import MultiAgentEnv

from GA_agent import Agent

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('env', type=str, help='name of the environment',
                        choices=['simple_adversary', 'simple_crypto', 'simple_push', 'simple_reference',
                                 'simple_speaker_listener', 'simple_spread', 'simple_tag',
                                 'simple_world_comm'])
    parser.add_argument('--folder', type=str, default='1', help='name of the folder where model is saved')
    parser.add_argument('--episode-length', type=int, default=150, help='steps per episode')
    parser.add_argument('--episode-num', type=int, default=30, help='total number of episodes to watch')
    args = parser.parse_args()

    # Create env
    scenario = scenarios.load(f'{args.env}.py').Scenario()
    world = scenario.make_world()
    env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)

    # Get dimensions
    obs_dim_list = [obs.shape[0] for obs in env.observation_space]
    act_dim_list = [act.n for act in env.action_space]
    agent_group = [Agent(obs_dim_list[i], act_dim_list[i]) for i in range(env.n)]

    # Load weights
    model_path = os.path.join('GA_results', args.env, args.folder, 'model.pt')
    if not os.path.exists(model_path):
        model_path = os.path.join('GA_results', args.env, args.folder, 'model', 'gen_0_best.pt')
    
    assert os.path.exists(model_path), f"Could not find model at {model_path}"
    
    saved_weights = torch.load(model_path)
    for agent, weight in zip(agent_group, saved_weights):
        agent.actor.load_state_dict(weight)
    
    print(f'Successfully loaded GA models from {model_path}')

    # Eval Loop
    # Tracks a Primary (Adversaries) and Secondary (Cooperative)
    primary_metric_history = []
    secondary_metric_history = []

    for episode in range(args.episode_num):
        obs = env.reset()
        ep_primary = 0
        ep_secondary = 0
        
        for step in range(args.episode_length):
            actions = []
            for i, agent in enumerate(agent_group):
                o = torch.from_numpy(obs[i]).unsqueeze(0).float()
                a_probs = agent.action(o).squeeze(0).numpy()
                actions.append(a_probs)
            
            next_obs, rewards, dones, infos = env.step(actions)
            
            # Splits this by environment
            # Tracks interceptions, distance, and whether caught or not
            if args.env == 'simple_tag':
                ep_primary += sum(rewards[:3])
                dists = [np.linalg.norm(next_obs[3][i:i+2]) for i in range(4, 10, 2)]
                ep_secondary += min(dists) if dists else 0
                
            # Calculates together because team effort. Uses distance to landmarks, smaller is better
            elif args.env == 'simple_spread':
                ep_primary += sum(rewards)
                landmark_dists = [np.linalg.norm(next_obs[0][i:i+2]) for i in range(4, 10, 2)]
                ep_secondary -= np.mean(landmark_dists)

            # Primary (adversary) get reward and Secondary (cooperative) has a different tracker for its rewards
            elif args.env == 'simple_adversary':
                ep_primary += rewards[0]
                ep_secondary += sum(rewards[1:])
            
            env.render()
            time.sleep(0.02)
            obs = next_obs

        primary_metric_history.append(ep_primary)
        secondary_metric_history.append(ep_secondary)
        print(f'Episode {episode + 1} | Primary: {ep_primary:.2f} | Secondary: {ep_secondary:.2f}')

    # Plotting
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    ax1.plot(primary_metric_history, color='blue', label='Primary (Adversary/Team)')
    ax1.set_title(f'Evaluation Primary Metric: {args.env}')
    ax1.legend()

    ax2.plot(secondary_metric_history, color='orange', label='Secondary (Cooperative)')
    ax2.set_title(f'Evaluation Secondary Metric: {args.env}')
    ax2.legend()
    
    plt.tight_layout()
    plot_save_path = os.path.join('GA_results', args.env, args.folder, 'evaluation_metrics.png')
    plt.savefig(plot_save_path)
    print(f"Evaluation plot saved to {plot_save_path}")
    plt.show()













# import argparse
# import os
# import time

# import numpy as np
# import torch
# from matplotlib import pyplot as plt
# from mpe.multiagent import scenarios
# from mpe.multiagent.environment import MultiAgentEnv

# from GA_agent import Agent

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('env', type=str, help='name of the environment',
#                         choices=['simple_adversary', 'simple_crypto', 'simple_push', 'simple_reference',
#                                  'simple_speaker_listener', 'simple_spread', 'simple_tag',
#                                  'simple_world_comm'])
#     parser.add_argument('--folder', type=str, default='1', help='name of the folder where model is saved')
#     parser.add_argument('--episode-length', type=int, default=50, help='steps per episode')
#     parser.add_argument('--episode-num', type=int, default=10, help='total number of episodes to watch')
#     args = parser.parse_args()

#     # 1. Create Environment
#     scenario = scenarios.load(f'{args.env}.py').Scenario()
#     world = scenario.make_world()
#     env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)

#     # 2. Reconstruct Agent Group
#     obs_dim_list = [obs.shape[0] for obs in env.observation_space]
#     act_dim_list = [act.n for act in env.action_space]
    
#     agent_group = [Agent(obs_dim_list[i], act_dim_list[i]) for i in range(env.n)]

#     # 3. Load the Weights
#     # Assuming the folder structure results_ga/env/folder/model.pt
#     model_path = os.path.join('GA_results', args.env, args.folder, 'model.pt')
#     if not os.path.exists(model_path):
#         # Fallback to check if you saved a specific gen file
#         model_path = os.path.join('GA_results', args.env, args.folder, 'model', 'gen_0_best.pt')
    
#     assert os.path.exists(model_path), f"Could not find model at {model_path}"
    
#     saved_weights = torch.load(model_path)
#     for agent, weight in zip(agent_group, saved_weights):
#         agent.actor.load_state_dict(weight)
    
#     print(f'Successfully loaded GA models from {model_path}')

#     # 4. Eval Loop
#     total_interceptions = []
#     total_survival_dist = []

#     for episode in range(args.episode_num):
#         obs = env.reset()

#         # debugging
#         for i, o in enumerate(obs):
#             print(f"Agent {i} observation shape: {o.shape}")
#             print(f"Agent {i} sample data: {o}")

#         ep_interceptions = 0
#         ep_survival = 0
        
#         for step in range(args.episode_length):
#             actions = []
#             for i, agent in enumerate(agent_group):
#                 o = torch.from_numpy(obs[i]).unsqueeze(0).float()
#                 a_probs = agent.action(o).squeeze(0).numpy()
#                 actions.append(a_probs)
            
#             next_obs, rewards, dones, infos = env.step(actions)
            
#             # --- Track the metrics we care about ---
#             # Adversaries (0,1,2): count rewards as interceptions
#             ep_interceptions += sum(rewards[:3]) 
            
#             # Cooperative/Prey (3): Track distance from closest predator
#             # This logic should match whatever you put in main_ga.py
#             dist_to_pred1 = np.linalg.norm(next_obs[3][4:6])
#             dist_to_pred2 = np.linalg.norm(next_obs[3][6:8])
#             dist_to_pred3 = np.linalg.norm(next_obs[3][8:10])
#             ep_survival += min(dist_to_pred1, dist_to_pred2, dist_to_pred3)
            
#             env.render()
#             time.sleep(0.02)
#             obs = next_obs

#         total_interceptions.append(ep_interceptions)
#         total_survival_dist.append(ep_survival)
#         print(f'Episode {episode + 1} | Interceptions: {ep_interceptions:.2f} | Survival Score: {ep_survival:.2f}')

#     # 5. Plotting Results
#     fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
#     ax1.plot(total_interceptions, color='red', label='Adversary Interceptions')
#     ax1.set_title('Adversary Success')
#     ax1.legend()

#     ax2.plot(total_survival_dist, color='green', label='Prey Survival Distance')
#     ax2.set_title('Cooperative Agent Survival')
#     ax2.legend()
    
#     plt.tight_layout()
#     # Save the evaluation plot in the folder we loaded from
#     plt.savefig(os.path.join('GA_results', args.env, args.folder, 'evaluation_metrics.png'))
#     plt.show()

#     '''This eval loop was for the previous fitness.

#     total_reward = np.zeros((args.episode_num, env.n))
#     for episode in range(args.episode_num):
#         obs = env.reset()
#         episode_reward = np.zeros((args.episode_length, env.n))
        
#         for step in range(args.episode_length):
#             actions = []
#             for i, agent in enumerate(agent_group):
#                 o = torch.from_numpy(obs[i]).unsqueeze(0).float()
#                 # Use the new agent.action() method
#                 a_probs = agent.action(o).squeeze(0).numpy()
#                 actions.append(a_probs)
            
#             next_obs, rewards, dones, infos = env.step(actions)
#             episode_reward[step] = rewards
            
#             env.render() # Visualizing the performance
#             time.sleep(0.05) # Slow down so human eyes can follow
            
#             obs = next_obs
#             if all(dones):
#                 break

#         cumulative_reward = episode_reward.sum(axis=0)
#         total_reward[episode] = cumulative_reward
#         print(f'Episode {episode + 1}: Total Group Reward: {sum(cumulative_reward):.2f}')

#     # 5. Plotting Results
#     plt.figure()
#     x = range(1, args.episode_num + 1)
#     for i in range(env.n):
#         plt.plot(x, total_reward[:, i], label=f'Agent {i}')
#     plt.xlabel('Episode')
#     plt.ylabel('Reward')
#     plt.title(f'Evaluation of GA on {args.env}')
#     plt.legend()
#     plt.show()'''
    