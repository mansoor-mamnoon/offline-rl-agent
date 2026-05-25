import torch
import torch.nn as nn
import torch.nn.functional as F
from models import PolicyNetwork


class QNetwork(nn.Module):

    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)  # concatenate state and action
        return self.net(x)




class CQLAgent:
    def __init__(self, state_dim, action_dim, lr=1e-3):
        self.q_net = QNetwork(state_dim, action_dim)
        self.q_target = QNetwork(state_dim, action_dim)
        self.policy = PolicyNetwork(state_dim, action_dim)

        self.q_target.load_state_dict(self.q_net.state_dict())
        self.optimizer = torch.optim.Adam(list(self.q_net.parameters()) + list(self.policy.parameters()), lr=lr)

        self.discount = 0.99
        self.action_dim = action_dim

    def train(self, batch):
        s = torch.tensor(batch['observations'], dtype=torch.float32)
        a = torch.tensor(batch['actions'], dtype=torch.long).unsqueeze(1)
        r = torch.tensor(batch['rewards'], dtype=torch.float32).unsqueeze(1)
        s2 = torch.tensor(batch['next_observations'], dtype=torch.float32)
        d = torch.tensor(batch['dones'], dtype=torch.float32).unsqueeze(1)

        a_onehot = F.one_hot(a.squeeze(), self.action_dim).float()
        q_pred = self.q_net(s, a_onehot)

        # Target Q
        with torch.no_grad():
            logits = self.policy(s2)
            pi_s2 = F.softmax(logits, dim=1)
            q_s2_all = torch.stack([
    self.q_target(s2, F.one_hot(torch.full((len(s2),), i), self.action_dim).float())
    for i in range(self.action_dim)
], dim=1).squeeze(-1)

            v_next = (pi_s2 * q_s2_all).sum(dim=1, keepdim=True)
            q_target = r + self.discount * (1 - d) * v_next

        bellman_loss = F.mse_loss(q_pred, q_target)

        # Conservative Q Loss
        q_all = torch.stack([
    self.q_net(s, F.one_hot(torch.full((len(s),), i), self.action_dim).float())
    for i in range(self.action_dim)
], dim=1).squeeze(-1)

        logsumexp_q = torch.logsumexp(q_all, dim=1, keepdim=True)
        q_taken = self.q_net(s, a_onehot)
        conservative_loss = (logsumexp_q - q_taken).mean()

        # Behavior Cloning Loss (optional)
        logits = self.policy(s)
        bc_loss = F.cross_entropy(logits, a.squeeze())

        total_loss = bellman_loss + conservative_loss + 0.1 * bc_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return {
            'bellman_loss': bellman_loss.item(),
            'conservative_loss': conservative_loss.item(),
            'bc_loss': bc_loss.item()
        }
