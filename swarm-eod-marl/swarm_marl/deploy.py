"""Copy a validated NumPy actor and its self-contained runtime to solutions."""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from .export import export


def deploy(checkpoint, official_root):
    root = Path(official_root).resolve()
    solutions = root / "src" / "swarm_rescue" / "solutions"
    if not (solutions / "my_drone_eval.py").is_file():
        raise FileNotFoundError("Official solutions directory not found")
    source = Path(__file__).resolve().parent
    archive = source.parent / "reports" / "my_drone_eval_before_marl.py"
    archive.parent.mkdir(exist_ok=True)
    if not archive.exists():
        shutil.copy2(solutions / "my_drone_eval.py", archive)
    export(checkpoint, solutions / "actor.npz")
    shutil.copy2(source / "env.py", solutions / "marl_env.py")
    shutil.copy2(source / "policy_numpy.py", solutions / "policy_numpy.py")
    shutil.copy2(source / "eval_drone.py", solutions / "eval_drone.py")
    (solutions / "my_drone_eval.py").write_text(
        '"""Competition entry point for the ten-drone MAPPO actor."""\n'
        'from swarm_rescue.solutions.eval_drone import MyDroneRescueRL\n'
        'from swarm_rescue.solutions.my_drone_place_example import MyDronePlaceExample\n'
        'from swarm_rescue.simulation.reporting.team_mode import TeamMode\n\n'
        'def drone_class_for_mode(mode: str):\n'
        '    return MyDronePlaceExample if TeamMode.from_string(mode) == TeamMode.PLACE else MyDroneRescueRL\n',
        encoding="utf-8")
    return solutions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--official-root", default="../swarm-eod-main")
    args = parser.parse_args()
    print(deploy(args.checkpoint, args.official_root))


if __name__ == "__main__":
    main()
