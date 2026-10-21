"""Export only actor tensors; no critic or optimizer in competition package."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .model import SharedActorCritic


def export(checkpoint, destination):
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    hidden_dim = int(saved["config"]["hidden_dim"])
    net = SharedActorCritic(hidden_dim)
    net.load_state_dict(saved["model"])
    actor = {key: tensor.detach().cpu().numpy() for key, tensor in net.state_dict().items()
             if not key.startswith("value_net.")}
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **actor)
    manifest = {"format": "swarm-eod-numpy-actor-v1", "hidden_dim": hidden_dim,
                "checkpoint_episode": int(saved["episode"]),
                "training_map": saved["config"]["map"], "num_drones": 10}
    destination.with_suffix(".json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return destination


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("destination")
    args = parser.parse_args()
    print(export(args.checkpoint, args.destination))


if __name__ == "__main__":
    main()
