"""Behavior clone legal trajectories from ten official example controllers."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from .env import N_AGENTS, SwarmEODTenAgentEnv
from .model import SharedActorCritic
from .teacher import TenDroneTeacher
from .train import tensor_obs


def collect_teacher(env, seed):
    obs, _ = env.reset(seed)
    rows = []
    while True:
        commands, labels, actions = {}, [], {}
        for i, drone in enumerate(env.drones):
            cmd = drone.control()
            commands[i] = cmd
            seen_bomb = bool(obs[i]["target_mask"][4:7].any())
            carry = bool(obs[i]["self"][2])
            task = 2 if carry else (1 if seen_bomb else 0)
            target = 7 if carry and obs[i]["target_mask"][7] else (
                int(4 + np.argmax(obs[i]["target_mask"][4:7])) if seen_bomb else 0)
            labels.append((task, target, float(cmd["forward"]), float(cmd["lateral"]),
                           float(cmd["rotation"]), int(cmd["grasper"])))
            state = obs[i]["self"]
            actions[i] = {"motion": [cmd["forward"], cmd["lateral"], cmd["rotation"]],
                          "grasper": cmd["grasper"],
                          "message": [state[0], state[4], state[5], state[1], state[2],
                                      task / 4, target / 7, state[3]]}
        rows.append((obs, np.asarray(labels, dtype=np.float32)))
        obs, _, terms, truncs, _ = env.step(actions)
        if all(terms[i] or truncs[i] for i in range(N_AGENTS)):
            break
    return rows, env.stats()


def clone(net, rows, epochs, device):
    optimizer = torch.optim.Adam(net.parameters(), lr=3e-4)
    losses = []
    for _ in range(epochs):
        for start in range(0, len(rows), 32):
            hidden = torch.zeros(N_AGENTS, net.hidden_dim, device=device)
            loss = torch.zeros((), device=device)
            end = min(start + 32, len(rows))
            for obs, labels in rows[start:end]:
                o = tensor_obs(obs, device)
                y = torch.as_tensor(labels, device=device)
                hidden, task_logits, target_logits, target_emb, _ = net.forward_step(
                    o, hidden, torch.zeros(N_AGENTS, 32, device=device))
                tasks, targets = y[:, 0].long(), y[:, 1].long()
                _, _, motion_dist, grasp_dist = net.distributions(
                    hidden, task_logits, target_logits, target_emb, tasks, targets)
                predicted_motion = torch.tanh(motion_dist.mean)
                valid = torch.as_tensor([obs[i]["alive"] for i in range(N_AGENTS)],
                                        device=device, dtype=torch.float32)
                per_agent = (F.cross_entropy(task_logits, tasks, reduction="none")
                             + F.cross_entropy(target_logits, targets, reduction="none")
                             + 4 * (predicted_motion - y[:, 2:5]).square().mean(-1)
                             + F.binary_cross_entropy_with_logits(
                                 grasp_dist.logits, y[:, 5], reduction="none"))
                loss = loss + (per_agent * valid).sum() / valid.sum().clamp_min(1)
            loss = loss / (end - start)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
            optimizer.step()
            losses.append(float(loss.item()))
    return float(np.mean(losses))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--output", default="runs/actor_init.pt")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    torch.manual_seed(int(cfg["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = SwarmEODTenAgentEnv(cfg["map"], cfg["bombs"], drone_type=TenDroneTeacher)
    try:
        rows, stats = collect_teacher(env, int(cfg["seed"]))
    finally:
        env.close()
    net = SharedActorCritic(int(cfg["hidden_dim"])).to(device)
    loss = clone(net, rows, args.epochs, device)
    output = Path(args.output)
    output.parent.mkdir(exist_ok=True)
    torch.save(net.cpu().state_dict(), output)
    print(json.dumps({"teacher_steps": len(rows), "teacher_disposed": stats.disposed,
                      "teacher_score": stats.official_score, "bc_loss": loss,
                      "checkpoint": str(output)}))


if __name__ == "__main__":
    main()
