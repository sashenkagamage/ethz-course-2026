"""Mouse-drag teleop for SO-100 demos (MuJoCo viewer).

Drag the red mocap target with the mouse instead of using WASD/arrows for EE
motion. Gripper open/close and recording still use keyboard shortcuts.

Controls
--------
EE motion (viewer):
  1. Double-click the red sphere at the end-effector (mocap target) to select it.
  2. Ctrl + Right-drag to move it in 3D.
  3. Scroll / middle-drag to orbit the camera as usual.

Keyboard:
  Space     toggle recording
  Enter     end episode & reset
  R         discard episode (if recording) & reset
  J / ]     open gripper
  N / [     close gripper
  Esc       save & quit

Multicube extras (same as key teleop): X/Z/C select goal cube colour.

Output layout matches ``record_teleop_demos.py`` so ``compute_actions.py`` works
unchanged.

Usage:
    python scripts/record_teleop_mouse.py
    python scripts/record_teleop_mouse.py --multicube
"""

from __future__ import annotations

import argparse
import collections
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import mujoco
import mujoco.viewer

from record_teleop_demos import (
    MulticubeTeleopRecorder,
    SO100Cv2TeleopRecorder,
)
from so101_gym.constants import ASSETS_DIR

# GLFW keycodes used by mujoco.viewer key_callback
_GLFW_SPACE = 32
_GLFW_ESCAPE = 256
_GLFW_ENTER = 257
_GLFW_LEFT_BRACKET = 91
_GLFW_RIGHT_BRACKET = 93
_GLFW_C = 67
_GLFW_J = 74
_GLFW_N = 78
_GLFW_R = 82
_GLFW_X = 88
_GLFW_Z = 90


def _make_mocap_visible(model: mujoco.MjModel) -> None:
    """Enlarge + colour the mocap site so it is easy to double-click."""
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "mocap_target_site")
    if site_id == -1:
        print("Warning: mocap_target_site not found; mouse selection may be hard.")
        return
    model.site_type[site_id] = mujoco.mjtGeom.mjGEOM_SPHERE
    model.site_size[site_id] = [0.03, 0.0, 0.0]
    model.site_rgba[site_id] = [1.0, 0.15, 0.1, 0.95]


class _MouseViewerMixin:
    """Replace the OpenCV run loop with MuJoCo passive viewer + mouse mocap drag."""

    def _glfw_to_actions(self, keycode: int) -> list[str]:
        """Map one GLFW key to zero or more teleop action names."""
        fixed = {
            _GLFW_SPACE: ["record"],
            _GLFW_ENTER: ["end_episode"],
            _GLFW_ESCAPE: ["escape"],
            _GLFW_R: ["reset"],
            _GLFW_J: ["gripper_open"],
            _GLFW_RIGHT_BRACKET: ["gripper_open"],
            _GLFW_N: ["gripper_close"],
            _GLFW_LEFT_BRACKET: ["gripper_close"],
            _GLFW_X: ["goal_cube_red"],
            _GLFW_Z: ["goal_cube_green"],
            _GLFW_C: ["goal_cube_blue"],
        }
        return fixed.get(keycode, [])

    def _dispatch_action(self, action: str) -> None:
        """Route *action* through the subclass ``_handle_key`` when possible."""
        # Prefer keymap raw codes so MulticubeTeleopRecorder goal/record logic runs.
        for raw, act in self._key_to_action.items():
            if act == action:
                self._handle_key(raw, raw & 0xFF)
                return

        # Gripper brackets may not be in keymap — apply directly.
        if action in ("gripper_open", "gripper_close"):
            from hw3.teleop_utils import handle_teleop_key
            from record_teleop_demos import MOCAP_INDEX

            handle_teleop_key(
                action, self.data, self.model, MOCAP_INDEX, self.act_id["Jaw"]
            )
            return

        # Session keys with fixed GLFW bindings if missing from keymap.
        if action == "escape":
            if self.recording:
                self.writer.end_episode()
                self.episodes_done += 1
                print(f"Episode {self.episodes_done} saved on exit.")
                self.recording = False
            self.running = False
        elif action == "record":
            self.recording = not self.recording
            print("RECORDING ON" if self.recording else "RECORDING OFF")
        elif action == "end_episode":
            if self.recording:
                self.writer.end_episode()
                self.episodes_done += 1
                print(f"Episode {self.episodes_done} saved.")
                self.recording = False
            self._reset_episode()
        elif action == "reset":
            if self.recording:
                self.writer.discard_episode()
                self.recording = False
                print("Episode DISCARDED. Press Space to start a new recording.")
            self._reset_episode()

    def run(self) -> None:
        _make_mocap_visible(self.model)
        mujoco.mj_forward(self.model, self.data)

        print(
            "\n=== Mouse teleop ===\n"
            "EE: double-click the RED sphere, then Ctrl+Right-drag to move.\n"
            "Keys: Space=record | Enter=end ep | R=reset | J/]=open | N/[=close | Esc=quit\n"
        )

        pending: collections.deque[int] = collections.deque()

        def key_callback(keycode: int) -> None:
            pending.append(int(keycode))

        last = time.perf_counter()
        with mujoco.viewer.launch_passive(
            self.model,
            self.data,
            key_callback=key_callback,
        ) as viewer:
            try:
                while self.running and viewer.is_running():
                    while pending:
                        keycode = pending.popleft()
                        for action in self._glfw_to_actions(keycode):
                            self._dispatch_action(action)

                    now = time.perf_counter()
                    dt = now - last
                    if dt < self.dt_ctrl:
                        time.sleep(self.dt_ctrl - dt)
                    last = time.perf_counter()

                    if self.recording:
                        self._record_step()

                    for _ in range(self.substeps):
                        mujoco.mj_step(self.model, self.data)

                    viewer.sync()
            finally:
                self._finalize_on_exit()
                self.writer.flush()
                print(
                    f"Flushed buffers. {self.episodes_done} episode(s) saved. Done."
                )


class MouseSO100Recorder(_MouseViewerMixin, SO100Cv2TeleopRecorder):
    pass


class MouseMulticubeRecorder(_MouseViewerMixin, MulticubeTeleopRecorder):
    pass


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Record teleop demos by dragging the EE in the MuJoCo viewer."
    )
    parser.add_argument(
        "--multicube",
        action="store_true",
        help="Record multicube goal-conditioned demonstrations.",
    )
    parser.add_argument(
        "--xml",
        type=Path,
        default=None,
        help="Path to the MuJoCo XML scene file.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible multicube shuffling.",
    )
    args = parser.parse_args()

    ts = datetime.now(ZoneInfo("Europe/Berlin")).strftime("%Y-%m-%d_%H-%M-%S")

    if args.multicube:
        xml_path = args.xml or (ASSETS_DIR / "so100_multicube_ee.xml")
        run_dir = Path("./datasets/raw/multi_cube/teleop") / ts
        out = run_dir / "so100_multicube_teleop.zarr"
        MouseMulticubeRecorder(
            xml_path=xml_path,
            out_zarr=out,
            control_hz=10.0,
            seed=args.seed,
        ).run()
        return

    xml_path = args.xml or (ASSETS_DIR / "so100_transfer_cube_obstacle_ee.xml")
    run_dir = Path("./datasets/raw/single_cube/teleop") / ts
    out = run_dir / "so100_transfer_cube_teleop.zarr"

    MouseSO100Recorder(
        xml_path=xml_path,
        out_zarr=out,
        control_hz=10.0,
        render_w=640,
        render_h=480,
    ).run()


if __name__ == "__main__":
    main()
