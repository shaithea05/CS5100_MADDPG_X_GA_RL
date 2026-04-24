import torch
import torch.nn as nn

class MLPNetwork(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=64):
        super(MLPNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)

class Agent:
    def __init__(self, obs_dim, act_dim):
        self.actor = MLPNetwork(obs_dim, act_dim)
        self.obs_dim = obs_dim
        self.act_dim = act_dim

    def action(self, obs):
        """Returns action probabilities (softmax) for MPE."""
        with torch.no_grad():
            logits = self.actor(obs)
            return torch.softmax(logits, dim=-1)