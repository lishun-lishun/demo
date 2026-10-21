"""Legal full-mission demonstration source for behavior initialization."""

from swarm_rescue.solutions.my_drone_rescue_example import MyDroneRescueExample


class TenDroneTeacher(MyDroneRescueExample):
    """Run the official sample's sensor-based controller on every drone."""

    def _is_active_drone(self):
        return True
