# Swarm-EOD ten-drone MARL

## Windows CMD 实时可视化

在已激活 `swarm_rescue` 的 CMD 中运行。`--render` 显示官方地图俯视图，`--render-every 20` 表示每 20 个仿真步刷新一次。窗口显示地图、无人机、静止炸弹、当前步数和已处置炸弹数；刷新越频繁，训练越慢。

```cmd
cd /d E:\code\compete\1\swarm-eod-marl
set PYTHONPATH=E:\code\compete\1\swarm-eod-main\src;E:\code\compete\1\swarm-eod-marl
python -m swarm_marl.train --config configs\train_10bombs.yaml --init-actor runs\actor_init.pt --episodes 1000 --run-dir runs\ten_bomb_training --device cuda --render --render-every 20
```

已开始的训练进程需要重启才能加载新增参数。等待 `runs\ten_bomb_training\latest.pt` 出现后按 `Ctrl+C`，然后执行：

```cmd
python -m swarm_marl.train --config configs\train_10bombs.yaml --resume runs\ten_bomb_training\latest.pt --episodes 1000 --run-dir runs\ten_bomb_training --device cuda --render --render-every 20
```

`--episodes` 在续训时表示**额外**训练的回合数。`latest.pt` 每 5 个完整回合写入一次；`best.pt` 在成绩刷新时写入，也可用于 `--resume`。可视化需要 Windows 图形桌面，远程无桌面的 Linux 服务器上请关闭 `--render`。

独立训练工程，复用 `../swarm-eod-main` 的原始地图、传感器、通信、物理和评分代码。每个回合固定创建 10 个真实 `DroneAbstract` 实例。训练 Actor 的输入只包含该机公开传感器读数、自身状态和实际收到的通信消息；Critic 在训练时使用 10 机合法观测的均值。

## 环境与入口

已经验证的本机环境是 `E:\envs\swarm_rescue\python.exe`（Python 3.11，PyTorch CUDA，Arcade 2.6.17）。无需修改官方仿真器；在 PowerShell 中设置：

```powershell
$env:PYTHONPATH='E:\code\compete\1\swarm-eod-main\src;E:\code\compete\1\swarm-eod-marl'
& 'E:\envs\swarm_rescue\python.exe' -m swarm_marl.pretrain --config configs/train.yaml --epochs 20
& 'E:\envs\swarm_rescue\python.exe' -m swarm_marl.train --config configs/train.yaml --init-actor runs/actor_init.pt --episodes 10 --run-dir runs/bc_mappo --device cuda
```

`SwarmEODTenAgentEnv.reset` 构造官方 Map 并通过公开 `spawn_bombs` 注入训练布局；`step` 调用公开 `Playground.step(all_commands, all_messages)`。一个 step 必须为 10 架无人机各给一个动作。`reset` 返回逐机观测，`step` 返回逐机观测、奖励、终止、截断和信息。物理和传感器规则保持官方原样。

## 五地图、每回合随机十弹训练

`configs/train_10bombs.yaml` 每回合轮换 Map01–Map05，从官方 `map_data/map_XX.json` 生成新的 10 弹布局。生成器把墙、方形障碍和边界膨胀 24 像素，排除出生区与处置中心，并保留与处置中心连通的位置；10 枚炸弹覆盖角落、边界、墙角、近墙和开阔区，至少相隔 58 像素。所有炸弹均为静止放置，不设置移动路径。地图真值只在离线场景生成阶段使用，Actor 的在线输入仍是该机传感器和真实收到的消息。每回合布局保存到 `runs/ten_bomb_training/layouts/`。

```powershell
$env:PYTHONPATH='E:\code\compete\1\swarm-eod-main\src;E:\code\compete\1\swarm-eod-marl'
& 'E:\envs\swarm_rescue\python.exe' -m swarm_marl.train --config configs/train_10bombs.yaml --init-actor runs/actor_init.pt --episodes 1000 --run-dir runs/ten_bomb_training --device cuda
```

单独导出可用于官方 `--bombs-file` 的随机布局：

```powershell
& 'E:\envs\swarm_rescue\python.exe' -m swarm_marl.scenarios --map-data-dir ../swarm-eod-main/map_data --map Map05 --seed 42 --out runs/map05_seed42.json
```

官方 Map 默认没有炸弹，因此每回合显式放置随机布局。训练奖励使用公开的抓取状态、健康、处置事件及终局官方评分，不读取隐藏真值。通信邻居只由 `communicator.received_messages` 构成。

## 迁移到 Linux GPU 服务器

把 `swarm-eod-main` 和 `swarm-eod-marl` 两个目录原样复制到服务器同一父目录，例如 `/srv/swarm/`。运行目录保持 `/srv/swarm/swarm-eod-marl`，这样 `../swarm-eod-main/map_data` 能正确解析。`runs/actor_init.pt` 可一起复制，用作行为预训练初始化；若不复制，可先在服务器重新运行 `pretrain`。

```bash
cd /srv/swarm/swarm-eod-marl
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# 根据服务器 NVIDIA 驱动，在 https://pytorch.org/get-started/locally/ 选 Linux/Pip/CUDA 安装命令；下行只是 CUDA 13.0 示例。
python -m pip install torch --index-url https://download.pytorch.org/whl/cu130
python -m pip install -e ../swarm-eod-main
python -m pip install -e . --no-deps
nvidia-smi
python -c 'import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))'
bash scripts/check_linux.sh
bash scripts/train_linux.sh --init-actor runs/actor_init.pt --episodes 1000
```

Arcade 2.6.17 在无桌面的 Linux 上需要 EGL/OpenGL。脚本预设 `ARCADE_HEADLESS=True`；`check_linux.sh` 会在正式训练前创建官方地图和 10 机实例。若服务器的 EGL 初始化失败，先安装匹配的 NVIDIA 图形驱动及 EGL 运行库，再运行 `ARCADE_HEADLESS=True python -m arcade` 检查渲染器。训练的神经网络由 CUDA 执行，仿真传感器还依赖 OpenGL。可用 `--resume runs/ten_bomb_training/latest.pt` 从 checkpoint 继续。

## 导出与官方运行

```powershell
& 'E:\envs\swarm_rescue\python.exe' -m swarm_marl.deploy runs/bc_mappo/best.pt --official-root ../swarm-eod-main
$env:PYTHONPATH='E:\code\compete\1\swarm-eod-main\src'
& 'E:\envs\swarm_rescue\python.exe' -m swarm_rescue.launcher -c configs/official_smoke.yaml --bombs-file configs/smoke_bombs.json --headless --competition
```

部署脚本仅向官方 `solutions/` 写入 `my_drone_eval.py`、`eval_drone.py`、`marl_env.py`、`policy_numpy.py`、`actor.npz` 和 manifest。当前官方入口已恢复原样；请在十弹训练和评测达到要求后再部署 checkpoint。导出 Actor 使用纯 NumPy 推理，不包含 Critic 和优化器。旧入口备份在本工程 `reports/`。`tests/test_export_parity.py` 校验 NumPy 和 PyTorch 确定性推理结果一致。

## 已审计的官方公开接口

- `Map01`–`Map05` 构造器接受 `drone_type` 和 `number_drones`，`MapAbstract.spawn_bombs` 注入训练炸弹。
- `Playground.step` 接收每机 CommandsDict 和通信消息，返回逐机物理奖励。
- `DroneAbstract` 提供 `measured_gps_position`、`measured_compass_angle`、`measured_velocity`、`semantic_values`、`lidar_values`、`grasped_bombs`、`drone_health`；`Communicator.received_messages` 是实际收到的邻居消息。
- `ScoreManager.compute_score` 提供终局官方评分；全弹处置和半时限成功分别报告。
- 正式入口 `solutions/my_drone_eval.py` 通过 `drone_class_for_mode` 返回逐机 Drone 类。正式运行启用 `--competition` 真值防护。

## 当前范围

已实现共享 Actor、邻居掩码注意力、GRU、任务与候选目标离散头、连续运动和抓手头、训练专用 Critic、PPO/GAE、行为预训练、checkpoint/恢复、CSV 日志、十弹多图场景生成及 NumPy 部署。当前 checkpoint 只经过 Map01 单弹训练；高层动作尚未跨多步保持，地图融合、Frontier、A*、任务认领协议和通信断开专项验证尚未完成。现有结果不能证明十弹半时限完成率或比赛最优策略。
