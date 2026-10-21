from pathlib import Path

import numpy as np
import torch

from swarm_marl.model import SharedActorCritic
from swarm_marl.policy_numpy import NumpyActor


def test_numpy_actor_matches_torch_deterministic(tmp_path: Path):
    torch.manual_seed(4)
    net = SharedActorCritic(128).eval()
    state = {k: v.detach().numpy() for k, v in net.state_dict().items()
             if not k.startswith("value_net.")}
    path = tmp_path / "actor.npz"
    np.savez_compressed(path, **state)
    actor = NumpyActor(path)
    rng = np.random.default_rng(3)
    observation = {"self": rng.normal(size=32).astype(np.float32),
                   "neighbors": rng.normal(size=(9, 8)).astype(np.float32),
                   "neighbor_mask": np.array([True, True] + [False] * 7),
                   "targets": rng.normal(size=(8, 4)).astype(np.float32),
                   "target_mask": np.array([True] * 6 + [False] * 2)}
    state0 = rng.normal(size=128).astype(np.float32)
    tensor_observation = {k: torch.as_tensor(v).unsqueeze(0) for k, v in observation.items()}
    with torch.no_grad():
        expected, _, _, expected_hidden = net.act(
            tensor_observation, torch.as_tensor(state0).unsqueeze(0),
            torch.zeros(1, 32), deterministic=True)
    actual, actual_hidden = actor.act(observation, state0)
    assert actual["task"] == int(expected["task"][0])
    assert actual["target"] == int(expected["target"][0])
    assert actual["grasper"] == int(expected["grasper"][0])
    np.testing.assert_allclose(actual["motion"], expected["motion"][0].numpy(), atol=1e-6)
    np.testing.assert_allclose(actual_hidden, expected_hidden[0].numpy(), atol=1e-6)
