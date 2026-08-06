"""Mouse-drag teleop for SO-100 demos (OpenCV, all camera views).

Left-drag moves the end-effector. Mouse wheel (or right-drag vertically)
moves it up/down. Keyboard only for gripper + recording.

Controls
--------
  Left-drag     move EE in the camera plane (follow the cursor)
  Mouse wheel   raise / lower EE (world Z)
  Right-drag    also raise / lower EE (vertical mouse motion)

  Space         toggle recording
  Enter         end episode & reset
  R             discard episode (if recording) & reset
  J / ]         open gripper
  N / [         close gripper
  Esc           save & quit

  Multicube: X / Z / C select goal cube colour (before recording)

Output matches ``record_teleop_demos.py`` (``compute_actions.py`` works as-is).

Usage:
    python scripts/record_teleop_mouse.py
    python scripts/record_teleop_mouse.py --multicube
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import cv2
import mujoco
import numpy as np

from hw3.teleop_utils import CAMERA_NAMES, compose_camera_views, handle_teleop_key
from record_teleop_demos import (
    MOCAP_INDEX,
    MulticubeTeleopRecorder,
    SO100Cv2TeleopRecorder,
)
from so101_gym.constants import ASSETS_DIR

# metres of EE motion per pixel of mouse motion
_DRAG_SCALE = 0.0012
# metres of Z per mouse-wheel notch / per pixel of right-drag
_Z_WHEEL = 0.012
_Z_DRAG_SCALE = 0.0010


class _OpenCvMouseMixin:
    """OpenCV window with left-drag EE control + all camera views."""

    # Axes for drag mapping (top-down is most intuitive for XY)
    _drag_camera: str = "top"

    def _setup_mouse_state(self) -> None:
        self._dragging_xy = False
        self._dragging_z = False
        self._last_mouse = (0, 0)
        self._cam_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_CAMERA, self._drag_camera
        )
        if self._cam_id == -1:
            self._drag_camera = "angle"
            self._cam_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_CAMERA, self._drag_camera
            )

    def _camera_axes(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (right, up) unit vectors of the drag camera in world frame."""
        mujoco.mj_forward(self.model, self.data)
        if self._cam_id == -1:
            return np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])
        R = self.data.cam_xmat[self._cam_id].reshape(3, 3)
        right = R[:, 0].copy()
        up = R[:, 1].copy()
        right[2] = 0.0
        up[2] = 0.0
        rn = np.linalg.norm(right)
        un = np.linalg.norm(up)
        if rn < 1e-6 or un < 1e-6:
            return np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])
        return right / rn, up / un

    def _apply_xy_delta(self, dx_pix: float, dy_pix: float) -> None:
        right, up = self._camera_axes()
        delta = right * (dx_pix * _DRAG_SCALE) + up * (-dy_pix * _DRAG_SCALE)
        self.data.mocap_pos[MOCAP_INDEX] = self.data.mocap_pos[MOCAP_INDEX] + delta

    def _apply_z_delta(self, dz: float) -> None:
        self.data.mocap_pos[MOCAP_INDEX, 2] += dz

    def _on_mouse(self, event: int, x: int, y: int, flags: int, _param) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            self._dragging_xy = True
            self._last_mouse = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self._dragging_xy = False
        elif event == cv2.EVENT_RBUTTONDOWN:
            self._dragging_z = True
            self._last_mouse = (x, y)
        elif event == cv2.EVENT_RBUTTONUP:
            self._dragging_z = False
        elif event == cv2.EVENT_MOUSEMOVE:
            dx = x - self._last_mouse[0]
            dy = y - self._last_mouse[1]
            self._last_mouse = (x, y)
            if self._dragging_xy:
                self._apply_xy_delta(dx, dy)
            elif self._dragging_z:
                self._apply_z_delta(-dy * _Z_DRAG_SCALE)
        elif event == cv2.EVENT_MOUSEWHEEL:
            direction = 1.0 if flags > 0 else -1.0
            self._apply_z_delta(direction * _Z_WHEEL)

    def _render_all_views(self) -> np.ndarray:
        images = {cam: self._render_bgr(cam) for cam in CAMERA_NAMES}
        img = compose_camera_views(images, CAMERA_NAMES)
        ee = self.data.site_xpos[self.ee_site_id]
        status = (
            f"{'REC' if self.recording else 'IDLE'} | ep {self.episodes_done} | "
            f"EE xyz=({ee[0]:.3f}, {ee[1]:.3f}, {ee[2]:.3f})"
        )
        cv2.putText(
            img, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
        )
        hint = (
            "L-drag move | wheel/R-drag height | "
            "Space rec | Enter end | R reset | J/N grip | Esc quit"
        )
        cv2.putText(
            img, hint, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 255, 200), 1
        )
        return img

    def _key_to_named_action(self, k_raw: int, k_ascii: int) -> str | None:
        action = self._key_to_action.get(k_raw)
        if action is not None:
            return action
        fixed_ascii = {
            ord(" "): "record",
            13: "end_episode",
            10: "end_episode",
            27: "escape",
            ord("r"): "reset",
            ord("R"): "reset",
            ord("j"): "gripper_open",
            ord("J"): "gripper_open",
            ord("]"): "gripper_open",
            ord("n"): "gripper_close",
            ord("N"): "gripper_close",
            ord("["): "gripper_close",
            ord("x"): "goal_cube_red",
            ord("z"): "goal_cube_green",
            ord("c"): "goal_cube_blue",
        }
        return fixed_ascii.get(k_ascii)

    def _dispatch_named(self, action: str) -> None:
        for raw, act in self._key_to_action.items():
            if act == action:
                self._handle_key(raw, raw & 0xFF)
                return

        if action in ("gripper_open", "gripper_close"):
            handle_teleop_key(
                action, self.data, self.model, MOCAP_INDEX, self.act_id["Jaw"]
            )
            return

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
                print("Episode DISCARDED.")
            self._reset_episode()

    def run(self) -> None:
        self._setup_mouse_state()
        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(self.window_name, self._on_mouse)

        print(
            "\n=== Mouse teleop (all cameras) ===\n"
            f"Views: {CAMERA_NAMES}\n"
            "Left-drag to move EE | wheel or right-drag for height\n"
            "Space=record | Enter=end | R=reset | J/N=gripper | Esc=quit\n"
        )

        last = time.perf_counter()
        try:
            while self.running:
                k_raw = cv2.waitKeyEx(1)
                if k_raw != -1:
                    k_ascii = k_raw & 0xFF
                    action = self._key_to_named_action(k_raw, k_ascii)
                    if action is not None and not action.startswith(("move_", "rot_")):
                        self._dispatch_named(action)

                now = time.perf_counter()
                dt = now - last
                if dt < self.dt_ctrl:
                    time.sleep(self.dt_ctrl - dt)
                last = time.perf_counter()

                if self.recording:
                    self._record_step()

                for _ in range(self.substeps):
                    mujoco.mj_step(self.model, self.data)

                cv2.imshow(self.window_name, self._render_all_views())
        finally:
            self._finalize_on_exit()
            self.writer.flush()
            cv2.destroyAllWindows()
            print(f"Flushed buffers. {self.episodes_done} episode(s) saved. Done.")


class MouseSO100Recorder(_OpenCvMouseMixin, SO100Cv2TeleopRecorder):
    pass


class MouseMulticubeRecorder(_OpenCvMouseMixin, MulticubeTeleopRecorder):
    pass


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Record teleop demos by dragging the EE with the mouse."
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
