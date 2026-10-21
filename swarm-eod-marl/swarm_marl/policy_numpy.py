"""NumPy-only actor inference for the official solutions directory."""
from __future__ import annotations

import numpy as np


def _linear(x, state, name):
    return x @ state[name + ".weight"].T + state[name + ".bias"]


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40, 40)))


def _softmax(x):
    z = np.exp(x - np.max(x))
    return z / max(float(z.sum()), 1e-8)


class NumpyActor:
    def __init__(self, path):
        with np.load(path, allow_pickle=False) as data:
            self.state = {k: data[k].astype(np.float32) for k in data.files}
        self.hidden_dim = self.state["memory.weight_hh"].shape[1]

    def initial_hidden(self):
        return np.zeros(self.hidden_dim, dtype=np.float32)

    def act(self, obs, hidden):
        s = self.state
        own = np.tanh(_linear(np.asarray(obs["self"], dtype=np.float32), s, "self_encoder.0"))
        nbr = np.tanh(_linear(np.asarray(obs["neighbors"], dtype=np.float32), s, "neighbor_encoder.0"))
        mask = np.asarray(obs["neighbor_mask"], dtype=bool)
        if mask.any():
            scores = (nbr @ _linear(own, s, "neighbor_query")) / self.hidden_dim**0.5
            scores[~mask] = -1e9
            weights = _softmax(scores)
            summary = weights @ nbr
        else:
            summary = np.zeros(self.hidden_dim, dtype=np.float32)
        inp = np.concatenate((own, summary))
        gi = _linear(inp, s, "memory") if "memory.weight" in s else (
            inp @ s["memory.weight_ih"].T + s["memory.bias_ih"])
        gh = hidden @ s["memory.weight_hh"].T + s["memory.bias_hh"]
        ir, iz, inn = np.split(gi, 3)
        hr, hz, hn = np.split(gh, 3)
        reset = _sigmoid(ir + hr)
        update = _sigmoid(iz + hz)
        new = np.tanh(inn + reset * hn)
        hidden = new + update * (hidden - new)
        task_logits = _linear(hidden, s, "task_head")
        task = int(np.argmax(task_logits))
        targets = np.asarray(obs["targets"], dtype=np.float32)
        target_emb = _linear(targets, s, "target_encoder")
        target_scores = target_emb @ _linear(hidden, s, "target_query")
        target_scores[~np.asarray(obs["target_mask"], dtype=bool)] = -1e9
        target = int(np.argmax(target_scores))
        one_hot = np.eye(5, dtype=np.float32)[task]
        mixed = np.concatenate((hidden, target_emb[target], one_hot))
        motion = np.tanh(np.tanh(_linear(mixed, s, "motion_head")))
        grasp = int(_sigmoid(_linear(mixed, s, "grasp_head"))[0] >= 0.5)
        return {"task": task, "target": target, "motion": motion, "grasper": grasp}, hidden
