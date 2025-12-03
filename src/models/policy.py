import torch.nn as nn

class Policy(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.actor = nn.Linear(512, 3)   # forward / left / right
        self.critic = nn.Linear(512, 1)

    def forward(self, obs):
        z = self.encoder(obs)
        return self.actor(z), self.critic(z)
