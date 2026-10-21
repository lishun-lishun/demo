#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"
export ARCADE_HEADLESS=True
export PYTHONPATH="${project_dir}/../swarm-eod-main/src:${project_dir}${PYTHONPATH:+:${PYTHONPATH}}"

python - <<'PY'
import arcade
import torch
from swarm_marl.scenarios import BombScenarioGenerator
from swarm_marl.env import SwarmEODTenAgentEnv

print("Arcade:", arcade.__version__)
print("PyTorch:", torch.__version__, "CUDA:", torch.cuda.is_available())
generator = BombScenarioGenerator("../swarm-eod-main/map_data")
for map_name in [f"Map{i:02d}" for i in range(1, 6)]:
    bombs, metadata = generator.generate(map_name, 1234)
    print(map_name, "bombs:", len(bombs), "static:", all("path" not in bomb for bomb in bombs))
env = SwarmEODTenAgentEnv("Map01", generator.generate("Map01", 1234)[0])
try:
    observations, _ = env.reset(seed=1234)
    print("Official simulator drones:", len(observations))
finally:
    env.close()
PY
