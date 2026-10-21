"""Public-API, simultaneous ten-agent adapter.

The actor inputs below come solely from each drone's public measured sensors,
own state, and messages actually received through Communicator.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass

import numpy as np

from swarm_rescue.maps.map_01 import Map01
from swarm_rescue.maps.map_02 import Map02
from swarm_rescue.maps.map_03 import Map03
from swarm_rescue.maps.map_04 import Map04
from swarm_rescue.maps.map_05 import Map05
from swarm_rescue.simulation.drone.drone_abstract import DroneAbstract
from swarm_rescue.simulation.ray_sensors.drone_semantic_sensor import DroneSemanticSensor
from swarm_rescue.simulation.reporting.score_manager import ScoreManager

MAPS = {c.__name__: c for c in (Map01, Map02, Map03, Map04, Map05)}
TASKS = ("EXPLORE", "PICKUP_BOMB", "TRANSPORT", "RETURN", "RELAY")
N_AGENTS = 10
SELF_DIM = 32
NEIGHBOR_DIM = 8
MAX_NEIGHBORS = 9
TARGET_DIM = 4
MAX_TARGETS = 8


class TrainingDrone(DroneAbstract):
    def define_message_for_all(self):
        return None

    def control(self):
        return {"forward": 0.0, "lateral": 0.0, "rotation": 0.0, "grasper": 0}


def _finite(value, default=0.0):
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def observe(drone):
    """Build fixed arrays from one drone's legal, local observation."""
    state = np.zeros(SELF_DIM, dtype=np.float32)
    state[0] = float(drone.identifier) / 9.0
    state[1] = _finite(drone.drone_health) / 100.0
    state[2] = float(bool(drone.grasped_bombs()))
    state[3] = _finite(drone.elapsed_timestep) / 2000.0
    gps = drone.measured_gps_position()
    if gps is not None:
        state[4:6] = np.asarray(gps[:2], dtype=np.float32) / 800.0
        state[6] = 1.0
    angle = drone.measured_compass_angle()
    if angle is not None:
        state[7:9] = math.sin(angle), math.cos(angle)
        state[9] = 1.0
    vel = drone.measured_velocity()
    if vel is not None:
        state[10:12] = np.clip(np.asarray(vel[:2], dtype=np.float32) / 50.0, -4, 4)
    state[12] = _finite(drone.measured_angular_velocity()) / 10.0
    lidar = drone.lidar_values()
    if lidar is not None:
        distances = np.asarray(lidar, dtype=np.float32).reshape(-1)
        if len(distances):
            for k, part in enumerate(np.array_split(distances, 16)):
                state[13 + k] = np.clip(np.nanmin(part) / 200.0, 0, 1)
    state[29] = float(drone.communicator_is_disabled())
    state[30] = float(drone.gps_is_disabled())
    state[31] = float(drone.semantic_is_disabled())

    targets = np.zeros((MAX_TARGETS, TARGET_DIM), dtype=np.float32)
    mask = np.zeros(MAX_TARGETS, dtype=bool)
    # Four relative exploration headings are available even without GPS.
    for k, bearing in enumerate((0.0, math.pi / 2, math.pi, -math.pi / 2)):
        targets[k] = (math.sin(bearing), math.cos(bearing), 1.0, 0.0)
        mask[k] = True
    semantic = drone.semantic_values()
    bombs, centers = [], []
    if semantic is not None:
        for hit in semantic:
            if not math.isfinite(_finite(hit.distance, float('nan'))):
                continue
            if hit.entity_type == DroneSemanticSensor.TypeEntity.BOMB and not hit.grasped:
                bombs.append(hit)
            elif hit.entity_type == DroneSemanticSensor.TypeEntity.DISPOSAL_CENTER:
                centers.append(hit)
    for k, hit in enumerate(sorted(bombs, key=lambda x: x.distance)[:3], 4):
        targets[k] = (math.sin(hit.angle), math.cos(hit.angle), min(hit.distance / 200, 2), 1.0)
        mask[k] = True
    if centers:
        hit = min(centers, key=lambda x: x.distance)
        targets[7] = (math.sin(hit.angle), math.cos(hit.angle), min(hit.distance / 200, 2), 2.0)
        mask[7] = True

    neighbors = np.zeros((MAX_NEIGHBORS, NEIGHBOR_DIM), dtype=np.float32)
    neighbor_mask = np.zeros(MAX_NEIGHBORS, dtype=bool)
    messages = drone.communicator.received_messages
    for k, (_, msg) in enumerate(messages[:MAX_NEIGHBORS]):
        if not isinstance(msg, (tuple, list)) or len(msg) != NEIGHBOR_DIM:
            continue
        neighbors[k] = np.asarray(msg, dtype=np.float32)
        neighbor_mask[k] = True
    return {"self": state, "neighbors": neighbors, "neighbor_mask": neighbor_mask,
            "targets": targets, "target_mask": mask, "alive": drone.drone_health > 0}


def legal_command(action):
    motion = np.asarray(action["motion"], dtype=float).reshape(3)
    motion = np.nan_to_num(motion, nan=0, posinf=0, neginf=0)
    return {"forward": float(np.clip(motion[0], -1, 1)),
            "lateral": float(np.clip(motion[1], -1, 1)),
            "rotation": float(np.clip(motion[2], -1, 1)),
            "grasper": int(bool(action["grasper"]))}


@dataclass
class RoundStats:
    map_name: str
    steps: int
    max_steps: int
    bombs: int
    disposed: int
    official_score: float
    all_bomb_success: bool
    half_time_success: bool


class SwarmEODTenAgentEnv:
    """Creates ten real Drone instances and steps the official Playground.

    `step` is public; there is no private simulator access or altered dynamics.
    Training rewards only use public per-agent reward, health, carry count and
    the team's observed disposal reward. Official score is computed at the end.
    """

    def __init__(self, map_name="Map01", bombs=None, drone_type=TrainingDrone):
        if map_name not in MAPS:
            raise ValueError(f"Unknown official map: {map_name}")
        self.map_name = map_name
        self.drone_type = drone_type
        self.bombs = list(bombs or [])
        if not self.bombs:
            raise ValueError("Rescue training requires explicit bombs; Map01 starts empty")
        self.the_map = None

    def reset(self, seed=None):
        if self.the_map is not None:
            self.close()
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        self.the_map = MAPS[self.map_name](drone_type=self.drone_type, number_drones=N_AGENTS)
        self.the_map.spawn_bombs(self.bombs)
        self.drones = self.the_map.drones
        assert len(self.drones) == N_AGENTS
        for d in self.drones:
            d.set_competition_mode(True)
        self.playground = self.the_map.playground
        self.t = 0
        self.disposed = 0
        self.prev_health = np.array([d.drone_health for d in self.drones], dtype=float)
        self.prev_carry = np.zeros(N_AGENTS, dtype=float)
        self.playground.step()
        return self._observe(), {i: {} for i in range(N_AGENTS)}

    def _observe(self):
        for d in self.drones:
            d.elapsed_timestep = self.t
        return {i: observe(d) for i, d in enumerate(self.drones)}

    def step(self, actions):
        if set(actions) != set(range(N_AGENTS)):
            raise ValueError("Exactly ten individual drone actions are required")
        commands = {d: legal_command(actions[i]) for i, d in enumerate(self.drones)}
        messages = {}
        for i, d in enumerate(self.drones):
            msg = actions[i].get("message")
            if msg is not None:
                msg = tuple(np.asarray(msg, dtype=np.float32).reshape(NEIGHBOR_DIM).tolist())
                messages[d] = {d.communicator: (None, msg)}
        _, raw = self.playground.step(all_commands=commands, all_messages=messages)
        self.t += 1
        self.the_map.explored_map.update_drones(self.drones)
        carry = np.array([len(d.grasped_bombs()) for d in self.drones], dtype=float)
        health = np.array([d.drone_health for d in self.drones], dtype=float)
        disposal = max(0, int(round(sum(raw.values()))))
        self.disposed += disposal
        complete = self.disposed >= self.the_map.number_bombs
        truncated = self.t >= self.the_map.max_timestep_limit
        # All agents share disposal gains. Individual shaping is bounded.
        rewards = {i: float(10 * disposal - 0.002 + 0.5 * max(0, carry[i] - self.prev_carry[i])
                            - 0.1 * max(0, self.prev_health[i] - health[i]))
                   for i in range(N_AGENTS)}
        if complete:
            for i in rewards:
                rewards[i] += 40.0 + 20.0 * (self.t <= self.the_map.max_timestep_limit / 2)
        if complete or truncated:
            # The official score is available only after the round has ended.
            terminal_score = self.stats().official_score
            for i in rewards:
                rewards[i] += terminal_score / 10.0
        self.prev_carry, self.prev_health = carry, health
        obs = self._observe()
        terms = {i: complete or health[i] <= 0 for i in range(N_AGENTS)}
        truncs = {i: truncated and not complete for i in range(N_AGENTS)}
        infos = {i: {"disposed": self.disposed, "carry": int(carry[i]), "health": float(health[i])}
                 for i in range(N_AGENTS)}
        return obs, rewards, terms, truncs, infos

    def stats(self):
        m = self.the_map
        manager = ScoreManager(N_AGENTS, m.max_timestep_limit, m.max_walltime_limit,
                               m.number_bombs)
        score, _, _ = manager.compute_score(
            self.disposed, m.explored_map.score() * 100,
            m.compute_score_health_returned() * 100, self.t)
        done = self.disposed == m.number_bombs
        return RoundStats(self.map_name, self.t, m.max_timestep_limit,
                          m.number_bombs, self.disposed, score, done,
                          done and self.t <= m.max_timestep_limit / 2)

    def close(self):
        if self.the_map is not None:
            self.the_map.playground.cleanup()
            self.the_map.playground.close_window()
            self.the_map = None
