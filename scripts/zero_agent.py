# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to an environment with random action agent."""

"""Launch Isaac Sim Simulator first."""

import argparse
import time
from collections import deque

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Random agent for Isaac Lab environments.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--playback_speed",
    type=float,
    default=1.0,
    help="Initial playback speed multiplier (real-time x1.0).",
)
parser.add_argument(
    "--trajectory_history",
    type=int,
    default=300,
    help="Maximum history length for each leg trajectory in the live 2D plot.",
)
parser.add_argument(
    "--disable_live_plot",
    action="store_true",
    default=False,
    help="Disable live 2D trajectory plotting and playback speed slider UI.",
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import sys
from pathlib import Path
_PROJECT_PATH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_PATH))
import tasks
# PLACEHOLDER: Extension template (do not remove this comment)


class DiagonalLegTrajectoryPlotter:
    """Live 2D (x-z) trajectory visualizer for diagonal leg pairs."""

    def __init__(self, robot, initial_speed: float, history_length: int):
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Slider

        self._plt = plt
        self._robot = robot
        self.playback_speed = max(float(initial_speed), 0.1)
        self._leg_colors = {"FL": "tab:blue", "RR": "tab:orange", "FR": "tab:green", "RL": "tab:red"}
        self._history = {leg: deque(maxlen=max(int(history_length), 10)) for leg in self._leg_colors}
        self._foot_body_ids = self._resolve_leg_body_ids("Foot")
        self._haa_body_ids = self._resolve_leg_body_ids("Hip")

        self._plt.ion()
        self._fig, (self._ax_fl_rr, self._ax_fr_rl, self._ax_all) = self._plt.subplots(1, 3, figsize=(16, 5))
        self._fig.subplots_adjust(bottom=0.2, wspace=0.32)
        self._fig.suptitle("Diagonal-leg actual trajectories (x-z, relative to HAA origin)")

        self._axes = {
            "FL": self._ax_fl_rr,
            "RR": self._ax_fl_rr,
            "FR": self._ax_fr_rl,
            "RL": self._ax_fr_rl,
        }
        self._traj_lines = {}
        self._curr_points = {}
        self._all_traj_lines = {}
        self._all_curr_points = {}

        for axis, title in ((self._ax_fl_rr, "FL / RR"), (self._ax_fr_rl, "FR / RL"), (self._ax_all, "All legs")):
            axis.set_title(title)
            axis.set_xlabel("x (m)")
            axis.set_ylabel("z (m)")
            axis.grid(True, alpha=0.3)

        for leg, color in self._leg_colors.items():
            axis = self._axes[leg]
            (traj_line,) = axis.plot([], [], color=color, linewidth=1.6, label=f"{leg} traj")
            (curr_point,) = axis.plot([], [], marker="o", linestyle="None", color=color, label=f"{leg} now")
            self._traj_lines[leg] = traj_line
            self._curr_points[leg] = curr_point
            (all_traj_line,) = self._ax_all.plot([], [], color=color, linewidth=1.6, label=f"{leg} traj")
            (all_curr_point,) = self._ax_all.plot([], [], marker="o", linestyle="None", color=color, label=f"{leg} now")
            self._all_traj_lines[leg] = all_traj_line
            self._all_curr_points[leg] = all_curr_point

        self._ax_fl_rr.legend(loc="upper right")
        self._ax_fr_rl.legend(loc="upper right")

        slider_ax = self._fig.add_axes([0.2, 0.06, 0.6, 0.03])
        self._speed_slider = Slider(
            slider_ax,
            "Playback speed (x)",
            valmin=0.1,
            valmax=3.0,
            valinit=self.playback_speed,
            valstep=0.1,
        )
        self._speed_slider.on_changed(self._on_speed_slider_changed)
        self._fig.canvas.draw_idle()
        self._fig.canvas.flush_events()

    def _resolve_leg_body_ids(self, body_name_keyword: str) -> dict[str, int]:
        body_names = list(self._robot.data.body_names)
        body_candidates = [idx for idx, name in enumerate(body_names) if body_name_keyword in name]
        if len(body_candidates) != 4:
            raise RuntimeError(
                f"Expected 4 '{body_name_keyword}' bodies, but found {len(body_candidates)}: "
                f"{[body_names[idx] for idx in body_candidates]}"
            )

        body_xy = self._robot.data.body_pos_w[0, body_candidates, :2].detach().cpu()
        front_order = torch.argsort(body_xy[:, 0], descending=True)
        front_local = front_order[:2]
        rear_local = front_order[2:]

        def split_left_right(local_ids: torch.Tensor) -> tuple[int, int]:
            local_y = body_xy[local_ids, 1]
            left_local = local_ids[torch.argmax(local_y)].item()
            right_local = local_ids[torch.argmin(local_y)].item()
            return left_local, right_local

        fl_local, fr_local = split_left_right(front_local)
        rl_local, rr_local = split_left_right(rear_local)

        return {
            "FL": body_candidates[fl_local],
            "FR": body_candidates[fr_local],
            "RL": body_candidates[rl_local],
            "RR": body_candidates[rr_local],
        }

    def _on_speed_slider_changed(self, value):
        self.playback_speed = max(float(value), 0.1)

    def _autoscale(self, axis, legs: tuple[str, ...]):
        points = [point for leg in legs for point in self._history[leg]]
        if not points:
            return

        xs = [point[0] for point in points]
        zs = [point[1] for point in points]
        x_span = max(xs) - min(xs)
        z_span = max(zs) - min(zs)
        x_margin = max(0.02, 0.12 * x_span)
        z_margin = max(0.02, 0.12 * z_span)
        axis.set_xlim(min(xs) - x_margin, max(xs) + x_margin)
        axis.set_ylim(min(zs) - z_margin, max(zs) + z_margin)

    def update(self):
        body_pos = self._robot.data.body_pos_w[0].detach().cpu()

        for leg in self._leg_colors:
            foot_pos = body_pos[self._foot_body_ids[leg]]
            haa_pos = body_pos[self._haa_body_ids[leg]]
            rel_pos = foot_pos - haa_pos
            xz = (float(rel_pos[0]), float(rel_pos[2]))
            self._history[leg].append(xz)

            xs, zs = zip(*self._history[leg])
            self._traj_lines[leg].set_data(xs, zs)
            self._curr_points[leg].set_data([xz[0]], [xz[1]])
            self._all_traj_lines[leg].set_data(xs, zs)
            self._all_curr_points[leg].set_data([xz[0]], [xz[1]])

        self._autoscale(self._ax_fl_rr, ("FL", "RR"))
        self._autoscale(self._ax_fr_rl, ("FR", "RL"))
        self._autoscale(self._ax_all, ("FL", "FR", "RL", "RR"))
        self._fig.canvas.draw_idle()
        self._fig.canvas.flush_events()
        self._plt.pause(0.001)

    def close(self):
        self._plt.close(self._fig)


def main():
    """Random actions agent with Isaac Lab environment."""
    # create environment configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)

    # print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")
    # reset environment
    env.reset()
    step_dt = float(getattr(env.unwrapped, "step_dt", 1.0 / 60.0))

    live_plotter = None
    if not args_cli.disable_live_plot and not getattr(args_cli, "headless", False):
        try:
            live_plotter = DiagonalLegTrajectoryPlotter(
                robot=env.unwrapped.scene["robot"],
                initial_speed=args_cli.playback_speed,
                history_length=args_cli.trajectory_history,
            )
            live_plotter.update()
            print("[INFO]: Live diagonal-leg 2D trajectory plot is enabled.")
            print("[INFO]: Use the plot slider to control playback speed.")
        except (ImportError, KeyError, RuntimeError, AttributeError) as err:
            print(f"[WARN]: Live trajectory plot is unavailable: {err}")
    elif args_cli.disable_live_plot:
        print("[INFO]: Live trajectory plot disabled by --disable_live_plot.")
    else:
        print("[INFO]: Headless mode detected, live trajectory plot disabled.")

    # simulate environment
    try:
        while simulation_app.is_running():
            loop_start_time = time.time()
            # run everything in inference mode
            with torch.inference_mode():
                # Use -1 to map to minimum values (robot stops/moves minimally)
                # Action mapping: [-1,1] -> [min,max]
                # -1 -> minimum step height, length, frequency (nearly stopped)
                # 0  -> middle values (walking at medium speed)
                # +1 -> maximum values (fast walking)
                # actions = -torch.ones(env.action_space.shape, device=env.unwrapped.device)
                actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
                # apply actions
                env.step(actions)

            playback_speed = max(args_cli.playback_speed, 0.1)
            if live_plotter is not None:
                live_plotter.update()
                playback_speed = live_plotter.playback_speed

            sleep_time = (step_dt / playback_speed) - (time.time() - loop_start_time)
            if sleep_time > 0:
                time.sleep(sleep_time)
    finally:
        if live_plotter is not None:
            live_plotter.close()
        # close the simulator
        env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
