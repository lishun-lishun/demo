"""Competition drone. Deployed beside marl_env.py, policy_numpy.py and actor.npz."""
from pathlib import Path

from swarm_rescue.simulation.drone.drone_abstract import DroneAbstract
from swarm_rescue.solutions.marl_env import observe, legal_command
from swarm_rescue.solutions.policy_numpy import NumpyActor


class MyDroneRescueRL(DroneAbstract):
    _shared_actor = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if type(self)._shared_actor is None:
            path = Path(__file__).with_name("actor.npz")
            type(self)._shared_actor = NumpyActor(path)
        self.hidden = self._shared_actor.initial_hidden()
        self.last_task = 0
        self.last_target = 0

    def define_message_for_all(self):
        local = observe(self)["self"]
        return (float(local[0]), float(local[4]), float(local[5]),
                float(local[1]), float(local[2]), self.last_task / 4.0,
                self.last_target / 7.0, float(local[3]))

    def control(self):
        if self.drone_health <= 0:
            return {"forward": 0.0, "lateral": 0.0, "rotation": 0.0, "grasper": 0}
        action, self.hidden = self._shared_actor.act(observe(self), self.hidden)
        self.last_task = action["task"]
        self.last_target = action["target"]
        return legal_command(action)
