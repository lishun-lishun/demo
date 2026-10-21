"""Shared recurrent actor and centralized-training value network."""
from __future__ import annotations

import torch
from torch import nn
from torch.distributions import Bernoulli, Categorical, Normal

from .env import MAX_TARGETS, NEIGHBOR_DIM, SELF_DIM, TARGET_DIM


class SharedActorCritic(nn.Module):
    def __init__(self, hidden_dim=128):
        super().__init__()
        h = hidden_dim
        self.hidden_dim = h
        self.self_encoder = nn.Sequential(nn.Linear(SELF_DIM, h), nn.Tanh())
        self.neighbor_encoder = nn.Sequential(nn.Linear(NEIGHBOR_DIM, h), nn.Tanh())
        self.neighbor_query = nn.Linear(h, h)
        self.memory = nn.GRUCell(h * 2, h)
        self.task_head = nn.Linear(h, 5)
        self.target_encoder = nn.Linear(TARGET_DIM, h)
        self.target_query = nn.Linear(h, h)
        self.motion_head = nn.Linear(h * 2 + 5, 3)
        self.motion_log_std = nn.Parameter(torch.full((3,), -0.7))
        self.grasp_head = nn.Linear(h * 2 + 5, 1)
        self.value_net = nn.Sequential(nn.Linear(h + SELF_DIM, h), nn.Tanh(),
                                       nn.Linear(h, h), nn.Tanh(), nn.Linear(h, 1))

    def forward_step(self, obs, hidden, global_state):
        own = self.self_encoder(obs["self"])
        nbr = self.neighbor_encoder(obs["neighbors"])
        scores = (nbr * self.neighbor_query(own).unsqueeze(-2)).sum(-1) / self.hidden_dim**0.5
        mask = obs["neighbor_mask"].bool()
        scores = scores.masked_fill(~mask, -1e9)
        weights = torch.softmax(scores, dim=-1) * mask.float()
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-8)
        summary = (weights.unsqueeze(-1) * nbr).sum(-2)
        hidden = self.memory(torch.cat((own, summary), dim=-1), hidden)
        task_logits = self.task_head(hidden)
        target_emb = self.target_encoder(obs["targets"])
        target_logits = (target_emb * self.target_query(hidden).unsqueeze(-2)).sum(-1)
        target_logits = target_logits.masked_fill(~obs["target_mask"].bool(), -1e9)
        value = self.value_net(torch.cat((hidden, global_state), -1)).squeeze(-1)
        return hidden, task_logits, target_logits, target_emb, value

    def distributions(self, hidden, task_logits, target_logits, target_emb, task_choice, target_choice):
        index = target_choice.long().unsqueeze(-1).unsqueeze(-1).expand(*target_choice.shape, 1, self.hidden_dim)
        chosen = target_emb.gather(-2, index).squeeze(-2)
        mixed = torch.cat((hidden, chosen, torch.nn.functional.one_hot(task_choice.long(), 5).float()), dim=-1)
        motion = Normal(self.motion_head(mixed).tanh(), self.motion_log_std.exp())
        grasp = Bernoulli(logits=self.grasp_head(mixed).squeeze(-1))
        return Categorical(logits=task_logits), Categorical(logits=target_logits), motion, grasp

    def act(self, obs, hidden, global_state, deterministic=False):
        h, task_logits, target_logits, target_emb, value = self.forward_step(obs, hidden, global_state)
        task_dist = Categorical(logits=task_logits)
        target_dist = Categorical(logits=target_logits)
        task = task_logits.argmax(-1) if deterministic else task_dist.sample()
        target = target_logits.argmax(-1) if deterministic else target_dist.sample()
        _, _, motion_dist, grasp_dist = self.distributions(h, task_logits, target_logits,
                                                           target_emb, task, target)
        motion_raw = motion_dist.mean if deterministic else motion_dist.sample()
        grasp = (grasp_dist.probs >= .5).float() if deterministic else grasp_dist.sample()
        logp = (task_dist.log_prob(task) + target_dist.log_prob(target)
                + motion_dist.log_prob(motion_raw).sum(-1) + grasp_dist.log_prob(grasp))
        return {"task": task, "target": target, "motion_raw": motion_raw,
                "motion": motion_raw.tanh(), "grasper": grasp}, logp, value, h

    def evaluate_step(self, obs, hidden, global_state, action):
        h, task_logits, target_logits, target_emb, value = self.forward_step(obs, hidden, global_state)
        task_dist, target_dist, motion_dist, grasp_dist = self.distributions(
            h, task_logits, target_logits, target_emb, action["task"], action["target"])
        logp = (task_dist.log_prob(action["task"]) + target_dist.log_prob(action["target"])
                + motion_dist.log_prob(action["motion_raw"]).sum(-1)
                + grasp_dist.log_prob(action["grasper"]))
        entropy = (task_dist.entropy() + target_dist.entropy()
                   + motion_dist.entropy().sum(-1) + grasp_dist.entropy())
        return logp, entropy, value, h
