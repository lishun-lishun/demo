"""Optional live view built from the official public TopDownView renderer."""
from __future__ import annotations

import cv2

from swarm_rescue.simulation.gui_map.top_down_view import TopDownView


class LiveRenderer:
    def __init__(self, env, every=20):
        if every < 1:
            raise ValueError("render interval must be positive")
        self.env = env
        self.every = every
        self.title = "Swarm-EOD training"
        width, height = env.playground.size
        scale = min(1.0, 1100.0 / width, 750.0 / height)
        size = (max(1, round(width * scale)), max(1, round(height * scale)))
        self.view = TopDownView(env.playground, size=size, zoom=scale)
        self.enabled = True
        cv2.namedWindow(self.title, cv2.WINDOW_NORMAL)

    def render(self):
        if not self.enabled or self.env.t % self.every:
            return
        try:
            self.view.update_and_draw_in_framebuffer(force=True)
            image = cv2.cvtColor(cv2.flip(self.view.get_np_img(), 0), cv2.COLOR_RGB2BGR)
            label = (f"{self.env.map_name}  step {self.env.t}/{self.env.the_map.max_timestep_limit}"
                     f"  disposed {self.env.disposed}/{self.env.the_map.number_bombs}")
            cv2.putText(image, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, (30, 30, 220), 2, cv2.LINE_AA)
            cv2.imshow(self.title, image)
            cv2.waitKey(1)
        except cv2.error:
            # Closing the preview window must not stop a training episode.
            self.enabled = False

    def close(self):
        if self.enabled:
            try:
                cv2.destroyWindow(self.title)
            except cv2.error:
                pass
