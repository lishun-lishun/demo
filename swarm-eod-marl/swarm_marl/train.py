"""On-policy ten-drone recurrent MAPPO training entry point."""
from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import yaml

from .env import N_AGENTS, SwarmEODTenAgentEnv
from .model import SharedActorCritic
from .scenarios import BombScenarioGenerator
from .visualize import LiveRenderer

OBS_KEYS = ("self", "neighbors", "neighbor_mask", "targets", "target_mask")
ACTION_KEYS = ("task", "target", "motion_raw", "grasper")


def tensor_obs(obs, device):
    return {key: torch.as_tensor(np.stack([obs[i][key] for i in range(N_AGENTS)]),
                                 device=device) for key in OBS_KEYS}


def global_state(obs, device):
    own = np.stack([obs[i]["self"] for i in range(N_AGENTS)])
    return torch.as_tensor(np.repeat(own.mean(0)[None, :], N_AGENTS, axis=0),
                           device=device, dtype=torch.float32)


def collect_episode(env, net, device, seed, render=False, render_every=20):
    obs, _ = env.reset(seed=seed)
    renderer = LiveRenderer(env, every=render_every) if render else None
    hidden = torch.zeros(N_AGENTS, net.hidden_dim, device=device)
    rows = []
    try:
        while True:
            own = tensor_obs(obs, device)
            glob = global_state(obs, device)
            alive = torch.as_tensor([obs[i]["alive"] for i in range(N_AGENTS)], device=device)
            with torch.no_grad():
                action, logp, value, next_hidden = net.act(own, hidden, glob)
            env_actions = {}
            for i in range(N_AGENTS):
                a = {"motion": action["motion"][i].cpu().numpy(),
                     "grasper": float(action["grasper"][i].item())}
                if not alive[i]:
                    a["motion"] = np.zeros(3)
                    a["grasper"] = 0
                # All information sent is a compact summary of this agent's own legal state.
                self_state = obs[i]["self"]
                a["message"] = [self_state[0], self_state[4], self_state[5],
                                self_state[1], self_state[2], float(action["task"][i]) / 4,
                                float(action["target"][i]) / 7, self_state[3]]
                env_actions[i] = a
            next_obs, rewards, terms, truncs, _ = env.step(env_actions)
            rows.append({"obs": {k: own[k].cpu() for k in OBS_KEYS}, "global": glob.cpu(),
                         "hidden": hidden.cpu(), "action": {k: action[k].cpu() for k in ACTION_KEYS},
                         "logp": logp.cpu(), "value": value.cpu(),
                         "reward": torch.tensor([rewards[i] for i in range(N_AGENTS)]),
                         "done": torch.tensor([terms[i] for i in range(N_AGENTS)]),
                         "alive": alive.cpu()})
            hidden = next_hidden.detach()
            for i in range(N_AGENTS):
                if terms[i]:
                    hidden[i] = 0
            obs = next_obs
            if renderer is not None:
                renderer.render()
            if all(terms[i] or truncs[i] for i in range(N_AGENTS)):
                break
    finally:
        if renderer is not None:
            renderer.close()
    with torch.no_grad():
        _, _, _, _, final_value = net.forward_step(tensor_obs(obs, device), hidden,
                                                   global_state(obs, device))
    return rows, final_value.cpu(), env.stats()


def advantages(rows, bootstrap, gamma, lam):
    gae = torch.zeros(N_AGENTS)
    next_value = bootstrap
    adv = []
    for row in reversed(rows):
        keep = (~row["done"]).float()
        delta = row["reward"] + gamma * next_value * keep - row["value"]
        gae = delta + gamma * lam * keep * gae
        adv.append(gae.clone())
        next_value = row["value"]
    adv.reverse()
    return torch.stack(adv), torch.stack([r["value"] for r in rows]) + torch.stack(adv)


def ppo_update(net, actor_opt, critic_opt, rows, advantage, returns, cfg, device):
    valid = torch.stack([r["alive"].float() for r in rows])
    norm_adv = (advantage - (advantage * valid).sum() / valid.sum().clamp_min(1))
    norm_adv = norm_adv / ((norm_adv.square() * valid).sum() / valid.sum().clamp_min(1)).sqrt().clamp_min(1e-6)
    metrics = []
    chunk_size = 32
    for _ in range(int(cfg["ppo_epochs"])):
        starts = list(range(0, len(rows), chunk_size))
        random.shuffle(starts)
        for start in starts:
            end = min(start + chunk_size, len(rows))
            h = rows[start]["hidden"].to(device)
            logps, entropies, values = [], [], []
            for t in range(start, end):
                row = rows[t]
                obs = {k: row["obs"][k].to(device) for k in OBS_KEYS}
                action = {k: row["action"][k].to(device) for k in ACTION_KEYS}
                logp, entropy, value, h = net.evaluate_step(obs, h, row["global"].to(device), action)
                logps.append(logp)
                entropies.append(entropy)
                values.append(value)
                h = h * (~row["done"].to(device)).float().unsqueeze(-1)
            new_logp = torch.stack(logps)
            entropy = torch.stack(entropies)
            values = torch.stack(values)
            old_logp = torch.stack([r["logp"] for r in rows[start:end]]).to(device)
            active = valid[start:end].to(device)
            denom = active.sum().clamp_min(1)
            ratio = (new_logp - old_logp).clamp(-20, 20).exp()
            adv = norm_adv[start:end].to(device)
            clip = float(cfg["clip_ratio"])
            policy_loss = -(torch.min(ratio * adv, ratio.clamp(1-clip, 1+clip) * adv) * active).sum() / denom
            value_loss = ((values - returns[start:end].to(device)).square() * active).sum() / denom
            entropy_mean = (entropy * active).sum() / denom
            loss = policy_loss + float(cfg["value_coef"]) * value_loss - float(cfg["entropy_coef"]) * entropy_mean
            actor_opt.zero_grad(set_to_none=True)
            critic_opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
            actor_opt.step()
            critic_opt.step()
            with torch.no_grad():
                kl = ((old_logp - new_logp) * active).sum() / denom
            metrics.append((policy_loss.item(), value_loss.item(), entropy_mean.item(), kl.item()))
    mean = np.asarray(metrics).mean(axis=0)
    return dict(zip(("actor_loss", "critic_loss", "entropy", "approx_kl"), mean.tolist()))


def save_checkpoint(path, net, actor_opt, critic_opt, episode, cfg, best_score, best_rank):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": net.state_dict(), "actor_optimizer": actor_opt.state_dict(),
                "critic_optimizer": critic_opt.state_dict(), "episode": episode,
                "config": cfg, "best_score": best_score, "best_rank": best_rank,
                "torch_rng": torch.get_rng_state(), "numpy_rng": np.random.get_state(),
                "python_rng": random.getstate()}, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--resume")
    parser.add_argument("--init-actor", help="Behavior cloned actor initialization")
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--run-dir", default="runs")
    parser.add_argument("--render", action="store_true", help="Show official top-down simulation while training")
    parser.add_argument("--render-every", type=int, default=20, help="Draw one frame per N physics steps")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if int(cfg["num_drones"]) != 10:
        raise ValueError("Training always uses 10 drones")
    if "maps" in cfg and int(cfg["bombs_per_episode"]) != 10:
        raise ValueError("Each generated full-task episode requires 10 bombs")
    if "maps" in cfg and cfg.get("bomb_motion") != "static":
        raise ValueError("Generated bombs must be static")
    seed = int(cfg["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else
                          "cpu" if args.device == "auto" else args.device)
    net = SharedActorCritic(int(cfg["hidden_dim"])).to(device)
    if args.init_actor and not args.resume:
        net.load_state_dict(torch.load(args.init_actor, map_location=device, weights_only=True), strict=False)
    actor_params = [p for n, p in net.named_parameters() if not n.startswith("value_net.")]
    critic_params = list(net.value_net.parameters())
    actor_opt = torch.optim.Adam(actor_params, lr=float(cfg["actor_lr"]))
    critic_opt = torch.optim.Adam(critic_params, lr=float(cfg["critic_lr"]))
    first_episode, best_score = 0, float("-inf")
    best_rank = (-1, -1, float("-inf"))
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        net.load_state_dict(checkpoint["model"])
        actor_opt.load_state_dict(checkpoint["actor_optimizer"])
        critic_opt.load_state_dict(checkpoint["critic_optimizer"])
        first_episode = int(checkpoint["episode"]) + 1
        best_score = float(checkpoint["best_score"])
        best_rank = tuple(checkpoint.get("best_rank", (-1, -1, float("-inf"))))
        torch.set_rng_state(checkpoint["torch_rng"].cpu())
        np.random.set_state(checkpoint["numpy_rng"])
        random.setstate(checkpoint["python_rng"])
    out = Path(args.run_dir)
    out.mkdir(exist_ok=True)
    logfile = out / "episodes.csv"
    generator = (BombScenarioGenerator(cfg["map_data_dir"],
                                       clearance=int(cfg.get("spawn_clearance", 24)),
                                       min_spacing=float(cfg.get("bomb_min_spacing", 58)))
                 if "maps" in cfg else None)
    with logfile.open("a", newline="", encoding="utf-8") as stream:
        writer = None
        for episode in range(first_episode, first_episode + int(args.episodes or cfg["episodes"])):
            if generator is None:
                map_name, bombs = cfg["map"], cfg["bombs"]
                scenario_meta = {"categories": []}
            else:
                map_name = cfg["maps"][episode % len(cfg["maps"])]
                bombs, scenario_meta = generator.generate(map_name, seed + 104729 * episode)
                layout_path = out / "layouts" / f"episode_{episode:06d}.json"
                layout_path.parent.mkdir(parents=True, exist_ok=True)
                layout_path.write_text(json.dumps({"map_name": map_name, "zones_config": [],
                                                   "bombs": bombs, "training_layout": scenario_meta},
                                                  indent=2), encoding="utf-8")
            env = SwarmEODTenAgentEnv(map_name, bombs)
            try:
                rows, bootstrap, stats = collect_episode(
                    env, net, device, seed + episode,
                    render=args.render, render_every=args.render_every)
            finally:
                env.close()
            adv, returns = advantages(rows, bootstrap, float(cfg["gamma"]), float(cfg["gae_lambda"]))
            before = torch.cat([p.detach().flatten().cpu() for p in net.parameters()])
            metrics = ppo_update(net, actor_opt, critic_opt, rows, adv, returns, cfg, device)
            after = torch.cat([p.detach().flatten().cpu() for p in net.parameters()])
            result = {"episode": episode, **asdict(stats),
                      "spawn_categories": "/".join(scenario_meta["categories"]),
                      **metrics, "parameter_delta_l2": float((after-before).norm())}
            if writer is None:
                writer = csv.DictWriter(stream, fieldnames=result.keys())
                if stream.tell() == 0:
                    writer.writeheader()
            writer.writerow(result)
            stream.flush()
            print(json.dumps(result), flush=True)
            rank = (int(stats.all_bomb_success), int(stats.half_time_success),
                    stats.official_score)
            if rank > best_rank:
                best_rank = rank
                best_score = stats.official_score
                save_checkpoint(out / "best.pt", net, actor_opt, critic_opt, episode, cfg, best_score, best_rank)
            if (episode + 1) % int(cfg["checkpoint_every"]) == 0:
                save_checkpoint(out / "latest.pt", net, actor_opt, critic_opt, episode, cfg, best_score, best_rank)


if __name__ == "__main__":
    main()
