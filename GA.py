import torch
import numpy as np
import copy
import random

class GeneticAlgorithm:
    def __init__(self, agent_class, pop_size, obs_dims, act_dims, mutation_rate=0.1, sigma=0.05):
        self.pop_size = pop_size
        self.mutation_rate = mutation_rate
        self.sigma = sigma

        self.population = []
        for _ in range(pop_size):
            agent_group = [agent_class(obs_dims[i], act_dims[i]) for i in range(len(obs_dims))]
            self.population.append(agent_group)

    def crossover(self, group1, group2):
        """Makes two groups of agents"""
        child_group = copy.deepcopy(group1)
        with torch.no_grad():
            for a_idx in range(len(group1)): # For each agent in the group
                parent1_params = list(group1[a_idx].actor.parameters())
                parent2_params = list(group2[a_idx].actor.parameters())
                child_params = list(child_group[a_idx].actor.parameters())
                
                for p_idx in range(len(parent1_params)):
                    # 50% chance from each parent
                    mask = torch.bernoulli(torch.full(parent1_params[p_idx].shape, 0.5))
                    child_params[p_idx].data.copy_(
                        mask * parent1_params[p_idx].data + (1 - mask) * parent2_params[p_idx].data
                    )
        return child_group

    def mutate(self, agent_group):
        """Applies Gaussian noise to the weights"""
        for agent in agent_group:
            with torch.no_grad():
                for param in agent.actor.parameters():
                    if random.random() < self.mutation_rate:
                        noise = torch.randn(param.size()) * self.sigma
                        param.add_(noise)
        return agent_group

    def evolve(self, fitness_scores):
        """Selection, Crossover, and Mutation."""
        indices = np.argsort(fitness_scores)[::-1]
        
        # Keep top 10%
        num_elites = max(1, self.pop_size // 10)
        new_population = [self.population[i] for i in indices[:num_elites]]
        
        # Mating includes top 50%
        mating_pool = [self.population[i] for i in indices[:self.pop_size // 2]]
        
        # Pick parents, create child, and mutate
        while len(new_population) < self.pop_size:
            p1, p2 = random.sample(mating_pool, 2)
            
            child = self.crossover(p1, p2)
            child = self.mutate(child)
            
            new_population.append(child)
            
        self.population = new_population