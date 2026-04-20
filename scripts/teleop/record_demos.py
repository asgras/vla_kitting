"""Wrapper around Isaac Lab's record_demos.py that registers our VLA pick-cube task.

The upstream `record_demos.py` launches `AppLauncher` early, then imports Isaac Lab
modules. Our `envs` package imports `isaaclab.assets` at import time, which pulls in
`omni.physics` — a module that only exists AFTER SimulationApp starts. So we cannot
`import envs` at the top of the wrapper. Instead we load the upstream script as text
and inject `import envs` on the line immediately after `AppLauncher` is instantiated.

Usage:
    ./isaaclab.sh -p scripts/teleop/record_demos.py \\
        --task Isaac-PickCube-HC10DT-Robotiq-IK-Rel-v0 \\
        --teleop_device keyboard \\
        --dataset_file datasets/teleop/cube_raw.hdf5 \\
        --num_demos 15
"""
from __future__ import annotations

import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

ISAACLAB_RECORD = pathlib.Path.home() / "IsaacLab/scripts/tools/record_demos.py"
if not ISAACLAB_RECORD.exists():
    print(f"ERROR: Isaac Lab record_demos.py not found at {ISAACLAB_RECORD}", file=sys.stderr)
    sys.exit(2)

MARKER = "simulation_app = app_launcher.app"
INJECTION = '''
# --- injected by vla_kitting wrapper: register our task once sim app is live ---
import envs  # noqa: F401

# --- Replace Se3Keyboard's buggy accumulator with a key-state-driven handler. ---
# Original behavior: PRESS does `delta += mapping[key]`, RELEASE does `delta -= mapping[key]`.
# This drifts when PRESS is dropped (CHARACTER-event crash, autorepeat, etc.), leaving
# a negative delta stuck in the accumulator. Our replacement tracks which keys are
# physically held and recomputes delta from that set each event — no drift possible.
def _install_keystroke_debug():
    try:
        import numpy as _np
        from isaaclab.devices.keyboard.se3_keyboard import Se3Keyboard
        import carb.input as _cin
        _key_held = {}
        _trans_keys = {"W", "S", "A", "D", "Q", "E"}
        _rot_keys = {"Z", "X", "T", "G", "C", "V"}
        def _patched_onkey(self, event, *a, **kw):
            try:
                is_str = isinstance(event.input, str)
                name = event.input if is_str else getattr(event.input, "name", None)
                t = event.type
                if t == _cin.KeyboardEventType.KEY_PRESS:
                    _key_held[name] = True
                    print(f"[teleop-debug] PRESS {name!r}", flush=True)
                elif t == _cin.KeyboardEventType.KEY_RELEASE:
                    _key_held[name] = False
                    print(f"[teleop-debug] RELEASE {name!r}", flush=True)
                else:
                    return True  # ignore REPEAT / CHARACTER
                # Rebuild delta from currently-held translation/rotation keys.
                self._delta_pos = _np.zeros(3)
                self._delta_rot = _np.zeros(3)
                for k, held in _key_held.items():
                    if not held or k not in self._INPUT_KEY_MAPPING:
                        continue
                    if k in _trans_keys:
                        self._delta_pos = self._delta_pos + self._INPUT_KEY_MAPPING[k]
                    elif k in _rot_keys:
                        self._delta_rot = self._delta_rot + self._INPUT_KEY_MAPPING[k]
                # Edge-triggered actions (only on PRESS): gripper toggle, reset, callbacks.
                if t == _cin.KeyboardEventType.KEY_PRESS:
                    if name == "K":
                        self._close_gripper = not self._close_gripper
                        print(f"[teleop-debug] gripper -> {'CLOSE' if self._close_gripper else 'OPEN'}", flush=True)
                    if name == "L":
                        self.reset()
                        print(f"[teleop-debug] Se3Keyboard.reset() via L", flush=True)
                    if name in self._additional_callbacks:
                        self._additional_callbacks[name]()
                return True
            except Exception as _e:
                print(f"[teleop-debug] handler failed: {_e}", flush=True)
                return True
        Se3Keyboard._on_keyboard_event = _patched_onkey
        # Log advance() output on change.
        _orig_advance = Se3Keyboard.advance
        _adv_state = {"last": None, "step": 0}
        def _patched_advance(self, *a, **kw):
            out = _orig_advance(self, *a, **kw)
            _adv_state["step"] += 1
            try:
                vals = out.cpu().tolist() if hasattr(out, "cpu") else list(out)
                rounded = [round(float(v), 4) for v in vals]
                if rounded != _adv_state["last"]:
                    print(f"[teleop-debug] advance -> {rounded}", flush=True)
                    _adv_state["last"] = rounded
            except Exception as _e:
                print(f"[teleop-debug] advance log failed: {_e}", flush=True)
            return out
        Se3Keyboard.advance = _patched_advance
        print("[teleop-debug] Se3Keyboard replaced with key-state-driven handler", flush=True)
    except Exception as _e:
        print(f"[teleop-debug] patch failed: {_e}", flush=True)
_install_keystroke_debug()

def _install_env_debug():
    try:
        from isaaclab.envs.manager_based_rl_env import ManagerBasedRLEnv
        _orig_step = ManagerBasedRLEnv.step
        _state = {"call": 0, "last_jp": None}
        def _patched_step(self, action):
            ret = _orig_step(self, action)
            _state["call"] += 1
            # Log the first few steps densely (to catch initial pose) then every 0.5s.
            if _state["call"] in (1, 3, 10) or _state["call"] % 15 == 0:
                try:
                    jp = self.scene["robot"].data.joint_pos[0, :6].cpu().tolist()
                    jp_r = [round(float(v), 4) for v in jp]
                    ac0 = action[0] if hasattr(action, "__getitem__") else action
                    ac_r = [round(float(v), 4) for v in (ac0.cpu().tolist() if hasattr(ac0, "cpu") else list(ac0))]
                    try:
                        ee = self.scene["robot"].data.body_link_pos_w
                        tool0_idx = self.scene["robot"].body_names.index("tool0")
                        ee_pos = [round(float(v), 4) for v in ee[0, tool0_idx].cpu().tolist()]
                    except Exception:
                        ee_pos = None
                    delta = None
                    if _state["last_jp"] is not None:
                        delta = [round(a-b, 5) for a, b in zip(jp_r, _state["last_jp"])]
                    _state["last_jp"] = jp_r
                    print(f"[teleop-debug] step #{_state['call']} act={ac_r} joint_pos={jp_r} joint_delta={delta} ee_tool0={ee_pos}", flush=True)
                except Exception as _e:
                    print(f"[teleop-debug] step log failed: {_e}", flush=True)
            return ret
        ManagerBasedRLEnv.step = _patched_step
        print("[teleop-debug] ManagerBasedRLEnv.step instrumented", flush=True)
    except Exception as _e:
        print(f"[teleop-debug] env patch failed: {_e}", flush=True)
_install_env_debug()
'''

src = ISAACLAB_RECORD.read_text()
if MARKER not in src:
    print(
        f"ERROR: expected marker {MARKER!r} not found in {ISAACLAB_RECORD}. "
        "Isaac Lab's record_demos.py structure has changed; update this wrapper.",
        file=sys.stderr,
    )
    sys.exit(3)

patched = src.replace(MARKER, MARKER + INJECTION, 1)
code = compile(patched, str(ISAACLAB_RECORD), "exec")
exec(code, {"__name__": "__main__", "__file__": str(ISAACLAB_RECORD)})
