"""Offline ten-bomb layouts from the official map_data JSON files.

Map geometry is used only before reset to choose reachable spawn coordinates.
The deployed actor never receives this obstacle mask or bomb ground truth.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


def _rectangle(mask, item, pad):
    x0, y0 = item["top_left"]
    x1, y1 = item["bottom_right"]
    cv2.rectangle(mask, (int(x0 - pad), int(y0 - pad)),
                  (int(x1 + pad), int(y1 + pad)), 1, -1)


class BombScenarioGenerator:
    def __init__(self, map_data_dir, clearance=24, min_spacing=58):
        self.map_data_dir = Path(map_data_dir)
        self.clearance = int(clearance)
        self.min_spacing = float(min_spacing)
        self._cache = {}

    def _geometry(self, map_name):
        if map_name in self._cache:
            return self._cache[map_name]
        number = map_name.removeprefix("Map")
        path = self.map_data_dir / f"map_{number}.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        width, height = int(data["canvas"]["width"]), int(data["canvas"]["height"])
        walls = np.zeros((height, width), dtype=np.uint8)
        exclusions = np.zeros_like(walls)
        segments = []
        center = None
        for item in data["elements"]:
            kind = item["type"]
            if kind == "line_wall":
                p0, p1 = [tuple(map(int, p)) for p in item["points"]]
                thickness = int(item.get("thickness", 6))
                cv2.line(walls, p0, p1, 1, thickness + 2 * self.clearance)
                segments.append((np.asarray(p0, dtype=float), np.asarray(p1, dtype=float)))
            elif kind == "rect_wall":
                _rectangle(walls, item, self.clearance)
            elif kind in ("disposal_center", "start_area"):
                _rectangle(exclusions, item, 14)
                if kind == "disposal_center":
                    a, b = item["top_left"], item["bottom_right"]
                    center = (int((a[0] + b[0]) / 2), int((a[1] + b[1]) / 2))
        if center is None:
            raise ValueError(f"{path} has no disposal center")
        margin = self.clearance + 6
        walls[:margin, :] = walls[-margin:, :] = 1
        walls[:, :margin] = walls[:, -margin:] = 1
        free = (walls == 0).astype(np.uint8)
        _, labels = cv2.connectedComponents(free, connectivity=8)
        component = int(labels[center[1], center[0]])
        if component == 0:
            raise ValueError(f"Disposal center blocked in {map_name} at clearance={self.clearance}")
        usable = (labels == component) & (exclusions == 0)
        distance = cv2.distanceTransform(free, cv2.DIST_L2, 5)
        endpoints = self._angles(segments)
        value = (width, height, walls, usable, distance, endpoints)
        self._cache[map_name] = value
        return value

    @staticmethod
    def _angles(segments):
        points = []
        for i, (a, b) in enumerate(segments):
            for p in (a, b):
                for c, d in segments[i+1:]:
                    for q in (c, d):
                        if np.linalg.norm(p - q) > 9:
                            continue
                        u, v = b-a, d-c
                        norm = np.linalg.norm(u) * np.linalg.norm(v)
                        if norm > 0 and abs(float(u @ v)) / norm < 0.8:
                            points.append((p + q) / 2)
        return np.asarray(points, dtype=float).reshape(-1, 2)

    def generate(self, map_name, seed, count=10):
        if count != 10:
            raise ValueError("The complete task uses exactly 10 bombs")
        width, height, walls, usable, distance, angles = self._geometry(map_name)
        rng = np.random.default_rng(seed)
        yy, xx = np.where(usable)
        if len(xx) < 10:
            raise ValueError(f"No sufficient free space in {map_name}")
        edge_x = np.minimum(xx, width - 1 - xx)
        edge_y = np.minimum(yy, height - 1 - yy)
        outer_corner = (edge_x < 140) & (edge_y < 140)
        boundary = (np.minimum(edge_x, edge_y) < 85) & ~outer_corner
        wall_near = (distance[yy, xx] >= self.clearance + 5) & (distance[yy, xx] < 75)
        open_area = (distance[yy, xx] >= 90) & (np.minimum(edge_x, edge_y) >= 90)
        near_angle = np.zeros(len(xx), dtype=bool)
        for point in angles:
            near_angle |= (xx - point[0])**2 + (yy - point[1])**2 < 100**2
        near_angle &= wall_near
        pools = {"corner": np.flatnonzero(outer_corner),
                 "boundary": np.flatnonzero(boundary),
                 "wall_angle": np.flatnonzero(near_angle),
                 "wall": np.flatnonzero(wall_near),
                 "open": np.flatnonzero(open_area),
                 "any": np.arange(len(xx))}
        # Exactly ten; allow corner-like fallback where a map has no inner junction.
        pattern = ["corner", "boundary", "wall_angle", "wall", "open",
                   "corner", "boundary", "wall_angle", "wall", "open"]
        selected = []
        categories = []
        for category in pattern:
            fallback = [category, "wall" if category == "wall_angle" else "any", "any"]
            choice = None
            used_category = None
            for name in fallback:
                pool = pools[name]
                if len(pool) == 0:
                    continue
                for index in rng.permutation(pool)[:min(len(pool), 2500)]:
                    x, y = int(xx[index]), int(yy[index])
                    if all((x - px)**2 + (y - py)**2 >= self.min_spacing**2
                           for px, py in selected):
                        choice, used_category = (x, y), name
                        break
                if choice is not None:
                    break
            if choice is None:
                raise ValueError(f"Cannot place ten separated bombs in {map_name}")
            selected.append(choice)
            categories.append(used_category)
        bombs = []
        for i, ((x, y), category) in enumerate(zip(selected, categories)):
            world = lambda px, py: [round(px - width / 2, 2), round(height / 2 - py, 2)]
            entry = {"x": world(x, y)[0], "y": world(x, y)[1], "theta": 0.0}
            bombs.append(entry)
        meta = {"map_name": map_name, "seed": int(seed), "count": 10,
                "categories": categories}
        return bombs, meta


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate a ten-bomb official JSON layout")
    parser.add_argument("--map-data-dir", default="../swarm-eod-main/map_data")
    parser.add_argument("--map", required=True, choices=[f"Map{i:02d}" for i in range(1, 6)])
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    bombs, meta = BombScenarioGenerator(args.map_data_dir).generate(args.map, args.seed)
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"map_name": args.map, "zones_config": [],
                                "bombs": bombs, "training_layout": meta}, indent=2), encoding="utf-8")
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
