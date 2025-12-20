import numpy as np
import time
import os
import glob
import zipfile
import fnmatch
import cv2
import math
import sys
import csv
import builtins
from copy import deepcopy

try:
    builtins.quit = lambda *args, **kwargs: None
    builtins.exit = lambda *args, **kwargs: None
except Exception:
    pass

try:
    import mujoco
    _MUJOCO_NATIVE_AVAILABLE = True
except Exception:
    _MUJOCO_NATIVE_AVAILABLE = False

_PROJECT_ROOT = os.path.dirname(__file__)
_LEAP_LOCAL_SRC = os.path.join(
    _PROJECT_ROOT,
    "leapc-python-bindings",
    "leapc-python-api",
    "src",
)
if _LEAP_LOCAL_SRC not in sys.path:
    sys.path.insert(0, _LEAP_LOCAL_SRC)

_LOCAL_CFFI_LOCATION = os.path.join(
    _PROJECT_ROOT,
    "leapc-python-bindings",
    "leapc-cffi",
    "src",
)
_LOCAL_CFFI_PKG_DIR = os.path.join(_LOCAL_CFFI_LOCATION, "leapc_cffi")

try:
    _have_pyd = False
    if os.path.isdir(_LOCAL_CFFI_PKG_DIR):
        for _f in os.listdir(_LOCAL_CFFI_PKG_DIR):
            if fnmatch.fnmatch(_f, "_leapc_cffi*.pyd"):
                _have_pyd = True
                break

    if not _have_pyd:
        _wheel_glob = os.path.join(
            _PROJECT_ROOT,
            "leapc-python-bindings",
            "leapc-cffi",
            "dist",
            "leapc_cffi-*.whl",
        )
        _wheels = sorted(glob.glob(_wheel_glob))
        if _wheels:
            os.makedirs(_LOCAL_CFFI_PKG_DIR, exist_ok=True)
            with zipfile.ZipFile(_wheels[-1], "r") as _zf:
                for _name in _zf.namelist():
                    if fnmatch.fnmatch(_name, "leapc_cffi/_leapc_cffi*.pyd"):
                        _target_path = os.path.join(_LOCAL_CFFI_PKG_DIR, os.path.basename(_name))
                        with _zf.open(_name) as _src, open(_target_path, "wb") as _dst:
                            _dst.write(_src.read())
except Exception:
    pass

os.environ.setdefault("LEAPSDK_INSTALL_LOCATION", _LOCAL_CFFI_LOCATION)

import leap

VISUALIZE_LEAP = True
MYOSUITE_RENDER_FPS = 60
ENABLE_MYOSUITE_RENDER = True
DESKTOP_MODE_TRANSFORM = True
DESKTOP_YAW_OFFSET = -2.040

SHOW_TENSION_IN_CTRL = True
TENSION_SOURCE = 'probe_act'
TENSION_EPS = 1e-6
DISPLAY_ZERO_EPS = 1e-4
DISPLAY_SMOOTH_ALPHA = 0.25
TENSION_NORM_MODE = 'global'

PROBE_FORCE_DISPLAY = True
PROBE_BASE_ACT = 0.0
PROBE_GAIN = 1.0
PROBE_MODE = 'heuristic'
PROBE_ACT_MODE = 'mujoco'
PROBE_SUBSTEPS = 5

try:
    from myosuite.utils import gym
    MYOSUITE_AVAILABLE = True
except Exception:
    MYOSUITE_AVAILABLE = False
ENABLE_MYOSUITE = True

QPOS_SMOOTH_ALPHA = 0.35
QPOS_MAX_STEP_DEFAULT = 0.25
QPOS_MAX_STEP_YAW = 0.35
QPOS_MAX_STEP_WRIST_DEV = 0.12
QPOS_MAX_STEP_WRIST_FLEX = 0.30
QPOS_MAX_STEP_THUMB_CMC = 0.30
QPOS_MAX_STEP_MCP_ABD = 0.10
JOINT_ANGLE_OVERRIDE_BLEND = 0.70

FINGER_FLEX_DEADZONE_RAD = 0.035

AUTO_RESET_ON_THUMB_STUCK = True
THUMB_STUCK_EPS = 0.002
THUMB_STUCK_OTHER_EPS = 0.02
THUMB_STUCK_FRAMES = 45
THUMB_STUCK_COOLDOWN_S = 2.0

JOINT_TUNING = {
    "wrist_flex_gain": 1.2,
    "wrist_dev_gain": 0.35,
    "cmc_abd_scale": 0.85,
    "cmc_flex_scale": 1.35,
    "mcp_abd_gain": 1.2,
    "mcp_abd_flex_suppress": 0.4,
    "index_flex_gain": 1.0,
    "middle_flex_gain": 1.0,
    "ring_flex_gain": 1.0,
    "little_flex_gain": 1.0,
    "index_abd_gain": 1.0,
    "middle_abd_gain": 1.0,
    "ring_abd_gain": 1.0,
    "little_abd_gain": 1.0,
}
MCP_ABD_BASE_DEFAULT = {
    1: 0.05095816096908771,
    2: -0.0006556564329452622,
    3: 0.0828368594413166,
    4: 0.08490408878999552,
}
THUMB_CMC_ABD0_RAD_DEFAULT = -0.3568619307393395
THUMB_CMC_FLEX0_RAW_DEFAULT = 1.5707963267948966

MCP_ABD_BASE = deepcopy(MCP_ABD_BASE_DEFAULT)
THUMB_CMC_ABD0_RAD = float(THUMB_CMC_ABD0_RAD_DEFAULT)
THUMB_CMC_FLEX0_RAW = float(THUMB_CMC_FLEX0_RAW_DEFAULT)

def vector_to_np(v):
    """Convert Leap vector-like object to numpy array, with optional Desktop transform."""
    if DESKTOP_MODE_TRANSFORM:
        return np.array([v.x, v.z, -v.y])
    return np.array([v.x, v.y, v.z])

def quaternion_to_yaw(q):
    """Yaw (pronation/supination) from Leap quaternion (w,x,y,z)."""
    qw, qx, qy, qz = q.w, q.x, q.y, q.z
    siny_cosp = 2 * (qw * qz + qx * qy)
    cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)

def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-8 else v

def _dir(a, b):
    v = vector_to_np(b) - vector_to_np(a)
    n = np.linalg.norm(v)
    return v / n if n > 1e-8 else np.zeros(3)

def vector_angle_rad(v1, v2):
    v1 = _unit(np.array(v1))
    v2 = _unit(np.array(v2))
    dot = float(np.clip(np.dot(v1, v2), -1.0, 1.0))
    return math.acos(dot)

def extract_palm_basis_from_hand(hand):
    """Return 3x3 basis matrix [x y z] columns using palm direction/normal."""
    dir_v = vector_to_np(hand.palm.direction)
    nrm_v = vector_to_np(hand.palm.normal)
    z = _unit(dir_v)
    y = _unit(nrm_v - np.dot(nrm_v, z) * z)
    x = _unit(np.cross(y, z))
    return np.stack([x, y, z], axis=1)

def vector_angle_deg(v1, v2):
    v1 = _unit(np.array(v1))
    v2 = _unit(np.array(v2))
    dot = float(np.clip(np.dot(v1, v2), -1.0, 1.0))
    return math.degrees(math.acos(dot))

def leap_hand_to_joint_angles(hand):
    """Compute simple finger joint angles in degrees (MCP,PIP,DIP for 4 fingers; thumb MCP,IP,CMC=0)."""
    angles = []
    for idx in [1, 2, 3, 4]:
        digit = hand.digits[idx]
        mcp = vector_angle_deg(
            vector_to_np(digit.metacarpal.next_joint) - vector_to_np(digit.metacarpal.prev_joint),
            vector_to_np(digit.proximal.next_joint) - vector_to_np(digit.proximal.prev_joint),
        )
        pip = vector_angle_deg(
            vector_to_np(digit.proximal.next_joint) - vector_to_np(digit.proximal.prev_joint),
            vector_to_np(digit.intermediate.next_joint) - vector_to_np(digit.intermediate.prev_joint),
        )
        dip = vector_angle_deg(
            vector_to_np(digit.intermediate.next_joint) - vector_to_np(digit.intermediate.prev_joint),
            vector_to_np(digit.distal.next_joint) - vector_to_np(digit.distal.prev_joint),
        )
        angles.extend([mcp, pip, dip])
    thumb = hand.digits[0]
    thumb_mcp = vector_angle_deg(
        vector_to_np(thumb.metacarpal.next_joint) - vector_to_np(thumb.metacarpal.prev_joint),
        vector_to_np(thumb.proximal.next_joint) - vector_to_np(thumb.proximal.prev_joint),
    )
    thumb_ip = vector_angle_deg(
        vector_to_np(thumb.proximal.next_joint) - vector_to_np(thumb.proximal.prev_joint),
        vector_to_np(thumb.distal.next_joint) - vector_to_np(thumb.distal.prev_joint),
    )
    thumb_cmc = 0.0
    angles.extend([thumb_mcp, thumb_ip, thumb_cmc])
    return np.array(angles)

def enhanced_leap_to_qpos(hand, arm=None):
    """Enhanced mapping dari Leap Motion ke 23-DOF qpos menggunakan vektor tulang (akurasi lebih baik)."""
    qpos = np.zeros(23)

    def _proj_to_plane(v, n):
        n = _unit(n)
        return v - np.dot(v, n) * n

    def _signed_angle_on_plane(a, b, n):
        a_u = _unit(a); b_u = _unit(b); n_u = _unit(n)
        x = float(np.clip(np.dot(a_u, b_u), -1.0, 1.0))
        y = float(np.dot(n_u, np.cross(a_u, b_u)))
        return math.atan2(y, x)

    try:
        basis = extract_palm_basis_from_hand(hand)
        palm_right = basis[:, 0]
        palm_normal = basis[:, 1]
        palm_direction = basis[:, 2]

        if arm is not None and getattr(arm, 'rotation', None):
            yaw = quaternion_to_yaw(arm.rotation)
            if DESKTOP_MODE_TRANSFORM and DESKTOP_YAW_OFFSET is not None:
                yaw = yaw - DESKTOP_YAW_OFFSET
            try:
                prev_yaw = getattr(enhanced_leap_to_qpos, '_prev_yaw', None)
                if prev_yaw is not None:
                    while yaw - prev_yaw > math.pi:
                        yaw -= 2 * math.pi
                    while yaw - prev_yaw < -math.pi:
                        yaw += 2 * math.pi
                enhanced_leap_to_qpos._prev_yaw = yaw
            except Exception:
                pass
            qpos[0] = np.clip(yaw, -2.5, 2.5)
        else:
            qpos[0] = 0.0

        deviation_angle = math.atan2(palm_normal[0], palm_normal[1])
        dev_gain = float(JOINT_TUNING.get("wrist_dev_gain", 0.35))
        qpos[1] = np.clip(deviation_angle * dev_gain, -0.174533, 0.436332)

        try:
            _prev_r = getattr(enhanced_leap_to_qpos, '_wf_prev_r')
            if float(np.dot(_prev_r, palm_right)) < 0.0:
                palm_right = -palm_right
        except Exception:
            pass
        enhanced_leap_to_qpos._wf_prev_r = palm_right

        wf_angle = 0.0
        have_arm = hasattr(hand, 'arm') and getattr(hand.arm, 'prev_joint', None) is not None and getattr(hand.arm, 'next_joint', None) is not None
        if have_arm:
            arm_dir = _dir(hand.arm.prev_joint, hand.arm.next_joint)
            a0 = _proj_to_plane(arm_dir, palm_right)
            b0 = _proj_to_plane(palm_direction, palm_right)
            if np.linalg.norm(a0) > 1e-6 and np.linalg.norm(b0) > 1e-6:
                wf_angle = _signed_angle_on_plane(a0, b0, palm_right)
            else:
                wf_angle = 0.0
        else:
            pd_u = _unit(palm_direction)
            wf_angle = math.asin(float(np.clip(pd_u[1], -1.0, 1.0)))

        gain_wf = float(JOINT_TUNING.get("wrist_flex_gain", 0.9))
        k = -gain_wf
        qpos[2] = np.clip(k * wf_angle, -0.785398, 0.785398)

        thumb = hand.digits[0]
        t_mc = vector_to_np(thumb.metacarpal.next_joint) - vector_to_np(thumb.metacarpal.prev_joint)
        t_px = vector_to_np(thumb.proximal.next_joint) - vector_to_np(thumb.proximal.prev_joint)
        t_ds = vector_to_np(thumb.distal.next_joint) - vector_to_np(thumb.distal.prev_joint)

        t_mc_u = _unit(t_mc)
        n_palm = _unit(palm_normal)
        r_palm = _unit(palm_right)
        try:
            _prev_n = getattr(enhanced_leap_to_qpos, '_thumb_prev_n')
            if float(np.dot(_prev_n, n_palm)) < 0.0:
                n_palm = -n_palm
        except Exception:
            pass
        enhanced_leap_to_qpos._thumb_prev_n = n_palm
        try:
            _prev_r_t = getattr(enhanced_leap_to_qpos, '_thumb_prev_r')
            if float(np.dot(_prev_r_t, r_palm)) < 0.0:
                r_palm = -r_palm
        except Exception:
            pass
        enhanced_leap_to_qpos._thumb_prev_r = r_palm

        px_u = _unit(t_px)
        radial = float(np.clip(np.dot(px_u, r_palm), -1.0, 1.0))
        base_abd_scale = float(JOINT_TUNING.get("cmc_abd_scale", 0.6))
        scale_abd = base_abd_scale * (2.0 if DESKTOP_MODE_TRANSFORM else 1.0)
        abd_base = THUMB_CMC_ABD0_RAD
        radial_rel = radial - abd_base
        cmc_abd = float(np.clip(-radial_rel * scale_abd, -1.0, 1.0))
        qpos[3] = cmc_abd

        flex_raw = vector_angle_rad(t_mc, t_px)
        base_flex_scale = float(JOINT_TUNING.get("cmc_flex_scale", 1.0))
        scale_flex = base_flex_scale * (1.8 if DESKTOP_MODE_TRANSFORM else 1.0)
        flex_base = THUMB_CMC_FLEX0_RAW
        cmc_flex = -(flex_raw - flex_base) * scale_flex
        qpos[4] = np.clip(cmc_flex, -0.78, 0.85)

        ref_flex = _proj_to_plane(palm_direction, r_palm)
        px_f = _proj_to_plane(t_px, r_palm)
        mp_scale = 0.8 * (1.5 if DESKTOP_MODE_TRANSFORM else 1.0)
        mp_flex = _signed_angle_on_plane(ref_flex, px_f, r_palm) * mp_scale
        qpos[5] = np.clip(mp_flex, -0.698132, 0.698132)

        px_f2 = _proj_to_plane(t_px, r_palm)
        ds_f2 = _proj_to_plane(t_ds, r_palm)
        ip_flex = _signed_angle_on_plane(px_f2, ds_f2, r_palm) * 1.0
        qpos[6] = np.clip(ip_flex, -1.309, 0.436332)

        try:
            if not np.isfinite(qpos[3]):
                cmc_abd_fb = math.asin(float(np.clip(np.dot(_unit(t_mc), n_palm), -1.0, 1.0))) * 0.6
                qpos[3] = np.clip(cmc_abd_fb, -1.0, 1.0)
            if abs(qpos[4]) < 1e-3:
                cmc_flex_fb = (vector_angle_rad(t_mc, palm_direction) - math.pi/2) * -0.8
                qpos[4] = np.clip(cmc_flex_fb, -0.78, 0.7)
            if abs(qpos[5]) < 1e-3:
                ang = vector_angle_rad(t_mc, t_px)
                mp_fb = max(0.0, math.pi - ang) * 0.5
                qpos[5] = np.clip(mp_fb, -0.785398, 0.698132)
            thumb_tip = vector_to_np(thumb.distal.next_joint)
            thumb_base = vector_to_np(thumb.metacarpal.prev_joint)
            tt_vec = thumb_tip - thumb_base
            if not np.isfinite(qpos[3]):
                cmc_abd_fb2 = math.asin(float(np.clip(np.dot(_unit(tt_vec), n_palm), -1.0, 1.0))) * 0.6
                qpos[3] = np.clip(cmc_abd_fb2, -1.0, 1.0)
            if abs(qpos[4]) < 1e-3:
                cmc_flex_fb2 = _signed_angle_on_plane(_proj_to_plane(palm_direction, n_palm), _proj_to_plane(tt_vec, n_palm), n_palm) * -0.8
                qpos[4] = np.clip(cmc_flex_fb2, -0.78, 0.7)
        except Exception:
            pass

        for i in range(4):
            digit = hand.digits[i + 1]
            base = 7 + i * 4

            mc = vector_to_np(digit.metacarpal.next_joint) - vector_to_np(digit.metacarpal.prev_joint)
            px = vector_to_np(digit.proximal.next_joint) - vector_to_np(digit.proximal.prev_joint)
            im = vector_to_np(digit.intermediate.next_joint) - vector_to_np(digit.intermediate.prev_joint)
            ds = vector_to_np(digit.distal.next_joint) - vector_to_np(digit.distal.prev_joint)

            if i == 0:
                flex_key = "index_flex_gain"
                abd_key = "index_abd_gain"
            elif i == 1:
                flex_key = "middle_flex_gain"
                abd_key = "middle_abd_gain"
            elif i == 2:
                flex_key = "ring_flex_gain"
                abd_key = "ring_abd_gain"
            else:
                flex_key = "little_flex_gain"
                abd_key = "little_abd_gain"

            mcp_flex = vector_angle_rad(mc, px)
            pip_flex = vector_angle_rad(px, im)
            dip_flex = vector_angle_rad(im, ds) 
            flex_gain = float(JOINT_TUNING.get(flex_key, 1.0))
            qpos[base + 0] = np.clip(mcp_flex * flex_gain, 0.0, 1.5708)
            base_map = MCP_ABD_BASE
            digit_index = i + 1
            radial = float(np.clip(np.dot(_unit(px), r_palm), -1.0, 1.0))
            if digit_index not in base_map:
                base_map[digit_index] = radial
            radial_rel = radial - base_map.get(digit_index, radial)
            gain_mcp_abd_global = float(JOINT_TUNING.get("mcp_abd_gain", 1.2))
            gain_mcp_abd_finger = float(JOINT_TUNING.get(abd_key, 1.0))
            gain_mcp_abd = gain_mcp_abd_global * gain_mcp_abd_finger
            mcp_abd = float(np.clip(-radial_rel * gain_mcp_abd, -0.35, 0.35))
            flex_factor = float(np.clip(mcp_flex / 1.0, 0.0, 1.0))
            flex_suppress = float(JOINT_TUNING.get("mcp_abd_flex_suppress", 0.4))
            mcp_abd = mcp_abd * (1.0 - flex_suppress * flex_factor)
            qpos[base + 1] = mcp_abd
            qpos[base + 2] = np.clip(pip_flex, 0.0, 1.5708)
            qpos[base + 3] = np.clip(dip_flex, 0.0, 1.5708)

        if DESKTOP_MODE_TRANSFORM:
            try:
                joint_angles = leap_hand_to_joint_angles(hand)
                THUMB_MCP_OFFSET = 90.0
                THUMB_IP_OFFSET = 14.3
                THUMB_CMC_OFFSET = 0.0
                if joint_angles is not None and len(joint_angles) >= 15:
                    b = float(JOINT_ANGLE_OVERRIDE_BLEND)
                    b = float(np.clip(b, 0.0, 1.0))
                    over = {}
                    over[6] = -math.radians(joint_angles[13] - THUMB_IP_OFFSET)

                    over[7] = math.radians(joint_angles[0])
                    over[9] = math.radians(joint_angles[1])
                    over[10] = math.radians(joint_angles[2])

                    over[11] = math.radians(joint_angles[3])
                    over[13] = math.radians(joint_angles[4])
                    over[14] = math.radians(joint_angles[5])

                    over[15] = math.radians(joint_angles[6])
                    over[17] = math.radians(joint_angles[7])
                    over[18] = math.radians(joint_angles[8])

                    over[19] = math.radians(joint_angles[9])
                    over[21] = math.radians(joint_angles[10])
                    over[22] = math.radians(joint_angles[11])

                    for k, v in over.items():
                        if 0 <= int(k) < int(qpos.shape[0]) and np.isfinite(v):
                            qpos[int(k)] = (1.0 - b) * float(qpos[int(k)]) + b * float(v)
            except Exception:
                pass

        try:
            calib = getattr(enhanced_leap_to_qpos, '_calib_qpos', None)
            if calib is not None:
                calib = np.asarray(calib, dtype=float)
                flex_idx = [7, 9, 10, 11, 13, 14, 15, 17, 18, 19, 21, 22]
                for j in flex_idx:
                    if 0 <= j < int(qpos.shape[0]) and j < int(calib.shape[0]):
                        if np.isfinite(qpos[j]) and np.isfinite(calib[j]):
                            qpos[j] = qpos[j] - calib[j]

                dz = float(FINGER_FLEX_DEADZONE_RAD)
                for j in flex_idx:
                    if 0 <= j < int(qpos.shape[0]) and np.isfinite(qpos[j]):
                        if qpos[j] < dz:
                            qpos[j] = 0.0
        except Exception:
            pass

    except Exception:
        try:
            qpos = np.asarray(getattr(enhanced_leap_to_qpos, '_prev_qpos'), dtype=float).copy()
        except Exception:
            qpos = np.zeros(23)

    try:
        prev_qpos = getattr(enhanced_leap_to_qpos, '_prev_qpos', None)
        qpos = np.asarray(qpos, dtype=float)
        if prev_qpos is None:
            qpos = np.where(np.isfinite(qpos), qpos, 0.0)
            enhanced_leap_to_qpos._prev_qpos = qpos.copy()
        else:
            prev_qpos = np.asarray(prev_qpos, dtype=float)
            qpos = np.where(np.isfinite(qpos), qpos, prev_qpos)

            max_step = np.ones_like(qpos) * float(QPOS_MAX_STEP_DEFAULT)
            if qpos.shape[0] > 0:
                max_step[0] = float(QPOS_MAX_STEP_YAW)
            if qpos.shape[0] > 1:
                max_step[1] = float(QPOS_MAX_STEP_WRIST_DEV)
            if qpos.shape[0] > 2:
                max_step[2] = float(QPOS_MAX_STEP_WRIST_FLEX)
            for j in [3, 4]:
                if 0 <= j < int(qpos.shape[0]):
                    max_step[j] = float(QPOS_MAX_STEP_THUMB_CMC)
            for j in [3, 8, 12, 16, 20]:
                if 0 <= j < int(qpos.shape[0]):
                    max_step[j] = float(QPOS_MAX_STEP_MCP_ABD)

            dq = np.clip(qpos - prev_qpos, -max_step, max_step)
            q_clamped = prev_qpos + dq

            alpha = float(QPOS_SMOOTH_ALPHA)
            qpos = prev_qpos * (1.0 - alpha) + q_clamped * alpha

            try:
                if bool(AUTO_RESET_ON_THUMB_STUCK):
                    now = time.time()
                    last_reset_t = float(getattr(enhanced_leap_to_qpos, '_thumb_stuck_last_reset_t', -1e9))
                    if (now - last_reset_t) >= float(THUMB_STUCK_COOLDOWN_S):
                        thumb_idx = np.array([3, 4, 5, 6], dtype=int)
                        other_idx = np.array([0, 1, 2] + list(range(7, 23)), dtype=int)
                        thumb_idx = thumb_idx[thumb_idx < int(qpos.shape[0])]
                        other_idx = other_idx[other_idx < int(qpos.shape[0])]

                        d_thumb = 0.0
                        d_other = 0.0
                        if thumb_idx.size:
                            d_thumb = float(np.max(np.abs(qpos[thumb_idx] - prev_qpos[thumb_idx])))
                        if other_idx.size:
                            d_other = float(np.max(np.abs(qpos[other_idx] - prev_qpos[other_idx])))

                        cnt = int(getattr(enhanced_leap_to_qpos, '_thumb_stuck_count', 0))
                        if (d_thumb <= float(THUMB_STUCK_EPS)) and (d_other >= float(THUMB_STUCK_OTHER_EPS)):
                            cnt += 1
                        else:
                            cnt = 0
                        enhanced_leap_to_qpos._thumb_stuck_count = cnt

                        if cnt >= int(THUMB_STUCK_FRAMES):
                            try:
                                print("[WARN] Auto reset: thumb appears stuck; resetting tracking state.", flush=True)
                            except Exception:
                                pass
                            reset_tracking_state()
                            enhanced_leap_to_qpos._thumb_stuck_last_reset_t = float(now)
                            enhanced_leap_to_qpos._thumb_stuck_count = 0
            except Exception:
                pass

            enhanced_leap_to_qpos._prev_qpos = qpos.copy()
    except Exception:
        try:
            enhanced_leap_to_qpos._prev_qpos = np.asarray(qpos, dtype=float).copy()
        except Exception:
            pass

    return qpos


def reset_tracking_state():
    """Reset semua state internal tracking untuk mengatasi stuck/drift.
    Panggil ini jika joint stuck di nilai tertentu atau delay setelah lama berjalan."""
    global MCP_ABD_BASE, THUMB_CMC_ABD0_RAD, THUMB_CMC_FLEX0_RAW
    attrs_to_reset = [
        '_prev_yaw', '_prev_qpos', '_wf_prev_r',
        '_thumb_prev_n', '_thumb_prev_r',
        '_calib_qpos',
        '_thumb_stuck_count', '_thumb_stuck_last_reset_t',
    ]
    for attr in attrs_to_reset:
        if hasattr(enhanced_leap_to_qpos, attr):
            delattr(enhanced_leap_to_qpos, attr)

    MCP_ABD_BASE = deepcopy(MCP_ABD_BASE_DEFAULT)
    THUMB_CMC_ABD0_RAD = float(THUMB_CMC_ABD0_RAD_DEFAULT)
    THUMB_CMC_FLEX0_RAW = float(THUMB_CMC_FLEX0_RAW_DEFAULT)


def setup_leap_motion_device(connection):
    """Setup and configure Leap Motion device with proper positioning"""
    try:
        t0 = time.time()
        devices = []
        while True:
            try:
                devices = list(connection.get_devices() or [])
            except Exception:
                devices = []

            if devices:
                break

            if (time.time() - t0) >= 2.0:
                try:
                    print(
                        "[WARN] Leap get_devices() returned empty for 2s. Check Ultraleap Tracking is running and the device is connected.",
                        flush=True,
                    )
                except Exception:
                    pass
                return False

            time.sleep(0.1)

        for device in devices:
            try:
                connection.subscribe_events(device)
            except Exception:
                continue
        return True
    except Exception:
        return False


_TRACKING_MODES = {
    leap.TrackingMode.Desktop: "Desktop",
    leap.TrackingMode.HMD: "HMD",
    leap.TrackingMode.ScreenTop: "ScreenTop",
}

class Canvas:
    def __init__(self):
        self.name = "Leap Visualiser"
        self.screen_size = [500, 700]
        self.hands_colour = (255, 255, 255)
        self.font_colour = (0, 255, 44)
        self.hands_format = "Skeleton"
        self.output_image = np.zeros((self.screen_size[0], self.screen_size[1], 3), np.uint8)
        self.tracking_mode = None

    def set_tracking_mode(self, tracking_mode):
        self.tracking_mode = tracking_mode

    def get_joint_position(self, vec):
        if vec is None:
            return None
        half_w = (self.screen_size[1] / 2)
        half_h = (self.screen_size[0] / 2)
        x = int(vec.x + half_w)
        z = int(vec.z + half_h)
        return (x, z)

    def render_hands(self, event):
        self.output_image[:, :] = 0

        mode_text = _TRACKING_MODES.get(self.tracking_mode, str(self.tracking_mode))
        cv2.putText(
            self.output_image,
            f"Tracking Mode: {mode_text}",
            (10, self.screen_size[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            self.font_colour,
            1,
        )

        if len(event.hands) == 0:
            return

        for i in range(len(event.hands)):
            hand = event.hands[i]
            for index_digit in range(0, 5):
                digit = hand.digits[index_digit]
                for index_bone in range(0, 4):
                    bone = digit.bones[index_bone]

                    if self.hands_format == "Dots":
                        prev_joint = self.get_joint_position(bone.prev_joint)
                        next_joint = self.get_joint_position(bone.next_joint)
                        if prev_joint:
                            cv2.circle(self.output_image, prev_joint, 2, self.hands_colour, -1)
                        if next_joint:
                            cv2.circle(self.output_image, next_joint, 2, self.hands_colour, -1)

                    if self.hands_format == "Skeleton":
                        wrist = self.get_joint_position(hand.arm.next_joint)
                        elbow = self.get_joint_position(hand.arm.prev_joint)
                        if wrist:
                            cv2.circle(self.output_image, wrist, 3, self.hands_colour, -1)
                        if elbow:
                            cv2.circle(self.output_image, elbow, 3, self.hands_colour, -1)
                        if wrist and elbow:
                            cv2.line(self.output_image, wrist, elbow, self.hands_colour, 2)

                        bone_start = self.get_joint_position(bone.prev_joint)
                        bone_end = self.get_joint_position(bone.next_joint)
                        if bone_start:
                            cv2.circle(self.output_image, bone_start, 3, self.hands_colour, -1)
                        if bone_end:
                            cv2.circle(self.output_image, bone_end, 3, self.hands_colour, -1)
                        if bone_start and bone_end:
                            cv2.line(self.output_image, bone_start, bone_end, self.hands_colour, 2)

                        if ((index_digit == 0) and (index_bone == 0)) or (
                            (index_digit > 0) and (index_digit < 4) and (index_bone < 2)
                        ):
                            index_digit_next = index_digit + 1
                            digit_next = hand.digits[index_digit_next]
                            bone_next = digit_next.bones[index_bone]
                            bone_next_start = self.get_joint_position(bone_next.prev_joint)
                            if bone_start and bone_next_start:
                                cv2.line(
                                    self.output_image,
                                    bone_start,
                                    bone_next_start,
                                    self.hands_colour,
                                    2,
                                )

                        if index_bone == 0 and bone_start and wrist:
                            cv2.line(self.output_image, bone_start, wrist, self.hands_colour, 2)


def main():
    global DESKTOP_MODE_TRANSFORM
    tracking_mode = leap.TrackingMode.Desktop

    connection = leap.Connection()
    leap_ok = True

    canvas = Canvas() if VISUALIZE_LEAP else None
    if canvas:
        canvas.set_tracking_mode(tracking_mode)
    env = None

    _recording = False
    _record_fh = None
    _record_writer = None
    _record_frame = 0
    _record_session = 0
    _record_path = None

    def _stop_recording():
        nonlocal _recording, _record_fh, _record_writer, _record_path
        if _record_fh is not None:
            try:
                _record_fh.flush()
            except Exception:
                pass
            try:
                _record_fh.close()
            except Exception:
                pass
        if _recording:
            try:
                print(f"[INFO] Recording stopped: {_record_path}", flush=True)
            except Exception:
                pass
        _recording = False
        _record_fh = None
        _record_writer = None
        _record_path = None

    def _start_recording():
        nonlocal _recording, _record_fh, _record_writer, _record_frame, _record_session, _record_path
        if env is None:
            try:
                print("[WARN] Cannot start recording: MyoSuite env not created yet.", flush=True)
            except Exception:
                pass
            return

        rec_dir = os.path.join(_PROJECT_ROOT, "recordings")
        try:
            os.makedirs(rec_dir, exist_ok=True)
        except Exception:
            pass

        ts = time.strftime("%Y%m%d_%H%M%S")
        session = int(_record_session)
        while True:
            fname = f"ctrl_display_{ts}_{session:03d}.csv"
            path = os.path.join(rec_dir, fname)
            if not os.path.exists(path):
                break
            session += 1

        _record_session = session + 1
        _record_frame = 0
        _record_path = path

        try:
            _record_fh = open(path, "w", newline="")
            _record_writer = csv.writer(_record_fh)

            try:
                nu = int(getattr(env.sim.model, 'nu', 0))
            except Exception:
                nu = 0
            names = getattr(listener, '_actuator_names', None)
            if names is None or len(names) != nu:
                names = [f"ctrl_{i}" for i in range(int(nu))]

            header = ["t", "frame"] + [str(nm) for nm in names]
            _record_writer.writerow(header)
            try:
                _record_fh.flush()
            except Exception:
                pass

            _recording = True
            try:
                print(f"[INFO] Recording started: {path}", flush=True)
            except Exception:
                pass
        except Exception as e:
            _stop_recording()
            try:
                print(f"[WARN] Failed to start recording ({type(e).__name__}: {e})", flush=True)
            except Exception:
                pass

    try:
        print("[INFO] Force indicator source: probe (fixed)", flush=True)
    except Exception:
        pass

    class LeapDataListener(leap.Listener):
        def __init__(self, canvas=None, env=None):
            super().__init__()
            self.canvas = canvas
            self.env = env
            self.latest_qpos = None
            self.last_render_time = time.time()
            self._actuator_names = None
            self._ntendon = 0
            self._nu = 0
            self._ctrl_baseline = None
            self._actuator_fmax = None
            self._actuator_to_tendon = None
            self._actuator_actadr = None
            self._probe_model = None
            self._probe_data = None
            self._display_norm_prev = None
            self._init_force_metadata()

        def _init_force_metadata(self):
            if self.env is None:
                return
            try:
                self._ntendon = int(getattr(self.env.sim.model, 'ntendon', 0))
            except Exception:
                self._ntendon = 0
            try:
                self._nu = int(getattr(self.env.sim.model, 'nu', 0))
            except Exception:
                self._nu = 0

            try:
                adr = np.asarray(getattr(self.env.sim.model, 'actuator_actadr', []), dtype=int)
                self._actuator_actadr = adr if adr.size >= int(self._nu) else None
            except Exception:
                self._actuator_actadr = None

            if PROBE_FORCE_DISPLAY and _MUJOCO_NATIVE_AVAILABLE:
                try:
                    model_ptr = getattr(self.env.sim.model, 'ptr', None)
                    if model_ptr is None:
                        model_ptr = getattr(self.env.sim.model, '_model', None)
                    if model_ptr is not None:
                        self._probe_model = model_ptr
                        self._probe_data = mujoco.MjData(model_ptr)
                except Exception:
                    self._probe_model = None
                    self._probe_data = None

            try:
                trnid = np.asarray(self.env.sim.model.actuator_trnid, dtype=int)
                if trnid.ndim == 2 and trnid.shape[0] >= int(self._nu) and int(self._ntendon) > 0:
                    tendon_id = np.asarray(trnid[: int(self._nu), 0], dtype=int)
                    tendon_id = np.where(
                        (tendon_id >= 0) & (tendon_id < int(self._ntendon)),
                        tendon_id,
                        -1,
                    )
                    self._actuator_to_tendon = tendon_id
                else:
                    self._actuator_to_tendon = None
            except Exception:
                self._actuator_to_tendon = None

        def _compute_probe_ctrl(self, qpos_target):
            if self.env is None:
                return None
            try:
                nu = int(self._nu)
                if nu <= 0:
                    return None
                q = np.asarray(qpos_target, dtype=float).reshape(-1)
                base = float(PROBE_BASE_ACT)
                gain = float(PROBE_GAIN)
                base = float(np.clip(base, 0.0, 1.0))
                gain = float(max(0.0, gain))

                flex_vals = []
                finger_flex = np.zeros((4,), dtype=float)
                finger_mcp_flex = np.zeros((4,), dtype=float)
                finger_ip_flex = np.zeros((4,), dtype=float)
                finger_abd = np.zeros((4,), dtype=float)
                finger_abd_pos = np.zeros((4,), dtype=float)
                finger_abd_neg = np.zeros((4,), dtype=float)
                for i in range(4):
                    off = 7 + i * 4
                    if off + 3 < int(q.shape[0]):
                        mcp = float(q[off + 0])
                        abd = float(q[off + 1])
                        pip = float(q[off + 2])
                        dip = float(q[off + 3])
                        flex_vals.extend([mcp, pip, dip])
                        finger_mcp_flex[i] = float(np.clip(mcp / 1.5708, 0.0, 1.0))
                        finger_ip_flex[i] = float(np.clip(np.mean([pip, dip]) / 1.5708, 0.0, 1.0))
                        finger_flex[i] = float(np.clip(np.mean([mcp, pip, dip]) / 1.5708, 0.0, 1.0))
                        abd_s = float(np.clip(abd / 0.35, -1.0, 1.0))
                        finger_abd[i] = float(abs(abd_s))
                        finger_abd_pos[i] = float(np.clip(abd_s, 0.0, 1.0))
                        finger_abd_neg[i] = float(np.clip(-abd_s, 0.0, 1.0))
                if flex_vals:
                    flex_mean = float(np.clip(np.mean(np.asarray(flex_vals)) / 1.5708, 0.0, 1.0))
                else:
                    flex_mean = 0.0
                open_mean = 1.0 - flex_mean

                finger_open = 1.0 - finger_flex

                wrist_dev = float(q[1]) if int(q.shape[0]) > 1 else 0.0
                wrist_dev_n = float(np.clip(abs(wrist_dev) / 0.35, 0.0, 1.0))
                wrist_radial = float(np.clip(wrist_dev / 0.35, 0.0, 1.0))
                wrist_ulnar = float(np.clip((-wrist_dev) / 0.35, 0.0, 1.0))

                yaw_n = float(np.clip(abs(float(q[0])) / 2.5, 0.0, 1.0)) if int(q.shape[0]) > 0 else 0.0

                thumb_cmc_abd = float(q[3]) if int(q.shape[0]) > 3 else 0.0
                thumb_cmc_flex = float(q[4]) if int(q.shape[0]) > 4 else 0.0
                thumb_mcp_flex = float(q[5]) if int(q.shape[0]) > 5 else 0.0
                thumb_ip_flex = float(q[6]) if int(q.shape[0]) > 6 else 0.0

                thumb_mcp_flex_n = float(np.clip(max(0.0, thumb_mcp_flex) / 0.698132, 0.0, 1.0))
                thumb_mcp_ext_n = float(np.clip(max(0.0, -thumb_mcp_flex) / 0.698132, 0.0, 1.0))
                thumb_ip_flex_n = float(np.clip(max(0.0, -thumb_ip_flex) / 1.309, 0.0, 1.0))
                thumb_ip_ext_n = float(np.clip(max(0.0, thumb_ip_flex) / 0.436332, 0.0, 1.0))
                thumb_flex = float(0.5 * thumb_mcp_flex_n + 0.5 * thumb_ip_flex_n)
                thumb_ext = float(0.5 * thumb_mcp_ext_n + 0.5 * thumb_ip_ext_n)
                thumb_cmc_n = float(np.clip((abs(thumb_cmc_abd) + abs(thumb_cmc_flex)) / (2.0 * 0.698132), 0.0, 1.0))
                thumb_mcp_n = float(np.clip(abs(thumb_mcp_flex) / 0.698132, 0.0, 1.0))
                thumb_ip_n = float(np.clip(abs(thumb_ip_flex) / 1.309, 0.0, 1.0))
                wrist_angle = float(q[2]) if int(q.shape[0]) > 2 else 0.0
                wrist_flex = float(np.clip(max(0.0, wrist_angle) / 0.785398, 0.0, 1.0))
                wrist_ext = float(np.clip(max(0.0, -wrist_angle) / 0.785398, 0.0, 1.0))

                names = self._actuator_names
                if names is None or len(names) != nu:
                    names = np.asarray([f"act_{i}" for i in range(nu)])

                ctrl = np.full((nu,), base, dtype=float)
                if str(PROBE_MODE).lower() != 'heuristic':
                    return np.clip(ctrl, 0.0, 1.0)

                for i in range(nu):
                    nm = str(names[i])

                    if nm == 'ECRL' or nm == 'ECRB':
                        ctrl[i] = base + gain * float(0.65 * wrist_ext + 0.35 * wrist_radial)
                        continue
                    if nm == 'ECU':
                        ctrl[i] = base + gain * float(0.65 * wrist_ext + 0.35 * wrist_ulnar)
                        continue
                    if nm == 'FCR':
                        ctrl[i] = base + gain * float(0.65 * wrist_flex + 0.35 * wrist_radial)
                        continue
                    if nm == 'FCU':
                        ctrl[i] = base + gain * float(0.65 * wrist_flex + 0.35 * wrist_ulnar)
                        continue
                    if nm == 'PL':
                        ctrl[i] = base + gain * float(0.75 * wrist_flex + 0.25 * wrist_dev_n)
                        continue
                    if nm == 'PT' or nm == 'PQ':
                        ctrl[i] = base + gain * float(yaw_n)
                        continue

                    if nm == 'OP':
                        ctrl[i] = base + gain * float(0.55 * thumb_cmc_n + 0.45 * thumb_mcp_n)
                        continue
                    if nm == 'FPL':
                        ctrl[i] = base + gain * float(0.35 * thumb_mcp_n + 0.65 * thumb_ip_n)
                        continue
                    if nm == 'EPL':
                        ctrl[i] = base + gain * float(thumb_ext)
                        continue
                    if nm == 'EPB' or nm == 'APL':
                        ctrl[i] = base + gain * float(0.60 * thumb_ext + 0.40 * thumb_cmc_n)
                        continue

                    last_digit = None
                    if nm and nm[-1].isdigit():
                        last_digit = ord(nm[-1]) - 48

                    if last_digit in (2, 3, 4, 5):
                        fi = int(last_digit) - 2
                        if 0 <= fi < 4:
                            if nm.startswith('FDP') or nm.startswith('FDS'):
                                ctrl[i] = base + gain * float(finger_flex[fi])
                                continue
                            if nm.startswith('EDC'):
                                ctrl[i] = base + gain * float(finger_open[fi])
                                continue
                            if nm.startswith('LU_RB'):
                                ctrl[i] = base + gain * float(finger_mcp_flex[fi] * (1.0 - finger_ip_flex[fi]))
                                continue
                            if nm.startswith('RI'):
                                ctrl[i] = base + gain * float(0.80 * finger_abd_pos[fi] + 0.20 * finger_mcp_flex[fi])
                                continue
                            if nm.startswith('UI_UB'):
                                ctrl[i] = base + gain * float(0.80 * finger_abd_neg[fi] + 0.20 * finger_mcp_flex[fi])
                                continue

                    if nm.startswith('EIP'):
                        ctrl[i] = base + gain * float(finger_open[0])
                        continue
                    if nm.startswith('EDM'):
                        ctrl[i] = base + gain * float(finger_open[3])
                        continue

                    is_ext = any(k in nm for k in ["EDC", "EIP", "EDM", "ECU", "ECR", "EPL", "EPB", "APL"])
                    is_flex = any(k in nm for k in ["FDP", "FDS", "FPL", "FCR", "FCU", "PL"])
                    is_thumb = any(k in nm for k in ["FPL", "EPL", "EPB", "APL", "OP"])
                    is_wrist = any(k in nm for k in ["ECR", "ECU", "FCR", "FCU", "PL", "PQ"])

                    if is_thumb and is_ext:
                        ctrl[i] = base + gain * thumb_ext
                    elif is_thumb and is_flex:
                        ctrl[i] = base + gain * thumb_flex
                    elif is_wrist and is_ext:
                        ctrl[i] = base + gain * wrist_ext
                    elif is_wrist and is_flex:
                        ctrl[i] = base + gain * wrist_flex
                    elif is_ext:
                        ctrl[i] = base + gain * open_mean
                    elif is_flex:
                        ctrl[i] = base + gain * flex_mean
                    else:
                        ctrl[i] = base + 0.5 * gain * (0.5 * flex_mean + 0.5 * open_mean)

                return np.clip(ctrl, 0.0, 1.0)
            except Exception:
                return None

        def _update_probe_state(self, qpos_target):
            if not PROBE_FORCE_DISPLAY:
                return
            if not _MUJOCO_NATIVE_AVAILABLE:
                return
            if self._probe_model is None or self._probe_data is None:
                return
            if qpos_target is None:
                return
            try:
                m = self._probe_model
                d = self._probe_data

                q = np.asarray(qpos_target, dtype=float).reshape(-1)
                n = min(int(d.qpos.shape[0]), int(q.shape[0]))
                d.qpos[:n] = q[:n]
                if n < int(d.qpos.shape[0]):
                    d.qpos[n:] = 0.0

                try:
                    d.qvel[:] = 0.0
                except Exception:
                    pass

                ctrl = self._compute_probe_ctrl(q)
                if ctrl is None:
                    return

                try:
                    d.ctrl[:] = 0.0
                except Exception:
                    pass
                nu = min(int(getattr(m, 'nu', 0)), int(d.ctrl.shape[0]), int(ctrl.shape[0]))
                if nu > 0:
                    d.ctrl[:nu] = ctrl[:nu]

                mode = str(PROBE_ACT_MODE).lower()
                if mode in ('copy', 'copy_ctrl', 'direct'):
                    try:
                        act = getattr(d, 'act', None)
                        adr = np.asarray(getattr(m, 'actuator_actadr', []), dtype=int)
                        if act is not None and adr.size >= nu:
                            for i in range(nu):
                                a = int(adr[i])
                                if 0 <= a < int(act.shape[0]):
                                    act[a] = float(d.ctrl[i])
                    except Exception:
                        pass
                    mujoco.mj_forward(m, d)
                else:
                    steps = int(PROBE_SUBSTEPS) if int(PROBE_SUBSTEPS) > 0 else 1
                    for _ in range(steps):
                        try:
                            d.qpos[:n] = q[:n]
                            if n < int(d.qpos.shape[0]):
                                d.qpos[n:] = 0.0
                            d.qvel[:] = 0.0
                        except Exception:
                            pass
                        mujoco.mj_step(m, d)
                    try:
                        d.qpos[:n] = q[:n]
                        if n < int(d.qpos.shape[0]):
                            d.qpos[n:] = 0.0
                        d.qvel[:] = 0.0
                    except Exception:
                        pass
                    mujoco.mj_forward(m, d)
            except Exception:
                pass

            try:
                ctrl_view = getattr(self.env.sim.data, 'ctrl', None)
                if ctrl_view is not None:
                    self._ctrl_baseline = np.zeros_like(np.asarray(ctrl_view, dtype=float))
            except Exception:
                self._ctrl_baseline = None

            fmax = None
            try:
                fr = np.asarray(self.env.sim.model.actuator_forcerange, dtype=float)
                if fr.ndim == 2 and fr.shape[1] >= 2 and fr.shape[0] >= 1:
                    m = min(int(fr.shape[0]), int(self._nu))
                    fmax = np.max(np.abs(fr[:m, :2]), axis=1)
            except Exception:
                fmax = None

            try:
                if fmax is not None:
                    fmax = np.asarray(fmax, dtype=float)
                    _mx = float(np.nanmax(fmax)) if fmax.size else 0.0
                    if (not np.isfinite(_mx)) or (_mx <= 0.0):
                        fmax = None
            except Exception:
                fmax = None

            if fmax is None:
                try:
                    gainprm = np.asarray(self.env.sim.model.actuator_gainprm, dtype=float)
                    if gainprm.ndim == 2 and gainprm.shape[0] >= 1:
                        fmax = gainprm[: self._nu, 0].copy() if gainprm.shape[1] > 0 else None
                except Exception:
                    fmax = None

            try:
                self._actuator_fmax = np.asarray(fmax, dtype=float) if fmax is not None else None
            except Exception:
                self._actuator_fmax = None

            if not _MUJOCO_NATIVE_AVAILABLE:
                return
            try:
                model = getattr(self.env.sim.model, '_model', None)
                if model is None:
                    model = self.env.sim.model

                if self._nu > 0:
                    act_names = []
                    for i in range(self._nu):
                        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
                        act_names.append(name if name is not None else "")
                    self._actuator_names = np.asarray(act_names)
            except Exception:
                self._actuator_names = None

        def _get_force_indicator_source(self):
            if self.env is None:
                return None, None, "none"
            try:
                if self._probe_data is not None and self._probe_model is not None:
                    return self._probe_data, self._probe_model, "probe"
            except Exception:
                pass
            return self.env.sim.data, self.env.sim.model, "main"

        def _reset_ctrl_to_baseline(self):
            if self.env is None:
                return
            if self._ctrl_baseline is None:
                return
            try:
                ctrl = self.env.sim.data.ctrl
                n = min(int(ctrl.shape[0]), int(self._ctrl_baseline.shape[0]))
                ctrl[:n] = self._ctrl_baseline[:n]
            except Exception:
                pass

        def _display_tension_in_ctrl(self):
            if not SHOW_TENSION_IN_CTRL:
                return
            if self.env is None:
                return
            try:
                data = self.env.sim.data
                src_data, _, _ = self._get_force_indicator_source()
                if src_data is None:
                    return

                src_mode = str(TENSION_SOURCE).lower()

                if src_mode == 'probe_ctrl':
                    src_ctrl = getattr(src_data, 'ctrl', None)
                    if src_ctrl is None:
                        return
                    src_ctrl = np.asarray(src_ctrl, dtype=float).reshape(-1)
                    norm = np.clip(src_ctrl, 0.0, 1.0)

                elif src_mode == 'probe_act':
                    act_vec = getattr(src_data, 'act', None)
                    if act_vec is None or self._actuator_actadr is None or int(self._nu) <= 0:
                        return
                    act_vec = np.asarray(act_vec, dtype=float).reshape(-1)
                    adr = np.asarray(self._actuator_actadr, dtype=int)
                    norm = np.zeros((int(self._nu),), dtype=float)
                    for i in range(int(self._nu)):
                        a = int(adr[i]) if i < int(adr.shape[0]) else -1
                        if 0 <= a < int(act_vec.shape[0]):
                            norm[i] = float(act_vec[a])
                    norm = np.clip(norm, 0.0, 1.0)

                else:
                    if src_mode == 'tendon_force':
                        ten_force = getattr(src_data, 'ten_force', None)
                        if ten_force is None:
                            ten_force = getattr(src_data, 'tendon_force', None)
                        force = ten_force
                    else:
                        force = getattr(src_data, 'actuator_force', None)

                    if force is None:
                        return

                    force = np.abs(np.asarray(force, dtype=float))

                    if src_mode == 'tendon_force' and self._actuator_to_tendon is not None and int(self._nu) > 0:
                        mapped = np.zeros((int(self._nu),), dtype=float)
                        tid = np.asarray(self._actuator_to_tendon, dtype=int)
                        valid = (tid >= 0) & (tid < int(force.shape[0]))
                        mapped[valid] = force[tid[valid]]
                        force = mapped

                    fmax = self._actuator_fmax
                    use_fmax = (str(TENSION_NORM_MODE).lower() == 'fmax')
                    if use_fmax and fmax is not None and len(fmax) > 0:
                        m = min(int(force.shape[0]), int(fmax.shape[0]))
                        denom = np.asarray(fmax[:m], dtype=float)
                        denom = np.where(np.isfinite(denom) & (denom > 0), denom, 1.0)
                        norm = np.zeros_like(force, dtype=float)
                        norm[:m] = np.clip(force[:m] / (denom + float(TENSION_EPS)), 0.0, 1.0)
                    else:
                        finite = force[np.isfinite(force)]
                        denom = float(np.max(finite)) if finite.size else 1.0
                        denom = denom if denom > 0 else 1.0
                        norm = np.clip(force / (denom + float(TENSION_EPS)), 0.0, 1.0)

                try:
                    z = float(DISPLAY_ZERO_EPS)
                    if np.isfinite(z) and z > 0:
                        norm = np.where(np.abs(norm) < z, 0.0, norm)
                except Exception:
                    pass

                try:
                    a = float(DISPLAY_SMOOTH_ALPHA)
                    if np.isfinite(a) and (a > 0.0) and (a < 1.0):
                        prev = getattr(self, '_display_norm_prev', None)
                        if prev is not None:
                            prev = np.asarray(prev, dtype=float).reshape(-1)
                            if prev.shape == norm.shape:
                                norm = prev * (1.0 - a) + norm * a
                        self._display_norm_prev = np.asarray(norm, dtype=float).reshape(-1).copy()
                except Exception:
                    pass

                try:
                    z = float(DISPLAY_ZERO_EPS)
                    if np.isfinite(z) and z > 0:
                        norm = np.where(np.abs(norm) < z, 0.0, norm)
                except Exception:
                    pass

                ctrl = data.ctrl
                n = min(int(ctrl.shape[0]), int(norm.shape[0]))
                try:
                    new_ctrl = np.zeros_like(np.asarray(ctrl, dtype=float))
                    if n > 0:
                        new_ctrl[:n] = norm[:n]
                    ctrl[:] = new_ctrl
                except Exception:
                    try:
                        ctrl[:n] = norm[:n]
                    except Exception:
                        pass
            except Exception:
                pass

        def on_tracking_mode_event(self, event):
            if self.canvas:
                self.canvas.set_tracking_mode(event.current_tracking_mode)
        
        def on_tracking_event(self, event):
            if len(event.hands) > 0:
                hand = event.hands[0]
                try:
                    q = enhanced_leap_to_qpos(hand, hand.arm)
                except Exception:
                    q = None
                if q is not None:
                    try:
                        self.latest_qpos = np.asarray(q, dtype=float)
                    except Exception:
                        self.latest_qpos = None

            if self.canvas:
                self.canvas.render_hands(event)


    listener = LeapDataListener(canvas, env)
    connection.add_listener(listener)

    def _handle_keys():
        key = cv2.waitKey(1) & 0xFF
        if key == ord('x'):
            try:
                print("[INFO] Exit requested via key 'x'", flush=True)
            except Exception:
                pass
            return False
        elif key == ord('0'):
            reset_tracking_state()
        elif key == ord('r'):
            if _recording:
                _stop_recording()
            else:
                _start_recording()
        elif key == ord('k'):
            try:
                if listener.latest_qpos is not None:
                    enhanced_leap_to_qpos._calib_qpos = np.asarray(listener.latest_qpos, dtype=float).copy()
                    print('Calibrated open pose baseline')
            except Exception:
                pass
        return True

    try:
        if leap_ok:
            try:
                print("[INFO] Entering Leap main loop (connection.open)", flush=True)
            except Exception:
                pass
            try:
                with connection.open():
                    connection.set_tracking_mode(tracking_mode)
                    DESKTOP_MODE_TRANSFORM = (tracking_mode == leap.TrackingMode.Desktop)

                    leap_ok = setup_leap_motion_device(connection)
                    try:
                        print(f"[INFO] Leap device available: {bool(leap_ok)}", flush=True)
                    except Exception:
                        pass
                    if not leap_ok:
                        try:
                            print("[WARN] No Leap Motion device detected. Exiting (LeapOnly=Yes).", flush=True)
                        except Exception:
                            pass

                        return

                    if ENABLE_MYOSUITE and MYOSUITE_AVAILABLE:
                        try:
                            env = gym.make('myoHandPoseRandom-v0')
                            env.reset()
                            try:
                                env = env.unwrapped
                            except Exception:
                                pass
                        except Exception:
                            env = None

                    try:
                        print(f"[INFO] MyoSuite env created: {env is not None}", flush=True)
                    except Exception:
                        pass
                    if env is not None:
                        try:
                            _nu_dbg = int(getattr(env.sim.model, 'nu', 0))
                            _ntendon_dbg = int(getattr(env.sim.model, 'ntendon', 0))
                            _ctrl_dbg = getattr(env.sim.data, 'ctrl', None)
                            _ctrl_n_dbg = int(np.asarray(_ctrl_dbg).shape[0]) if _ctrl_dbg is not None else 0
                            print(f"[INFO] dims: nu={_nu_dbg} ntendon={_ntendon_dbg} ctrl={_ctrl_n_dbg}", flush=True)
                        except Exception:
                            pass

                    try:
                        listener.env = env
                        listener._init_force_metadata()
                    except Exception:
                        pass

                    running = True
                    last_render_time = time.time()
                    _printed_loop = False
                    while running:
                        if not _printed_loop:
                            try:
                                print("[INFO] Leap loop running", flush=True)
                            except Exception:
                                pass
                            _printed_loop = True
                        if VISUALIZE_LEAP and canvas is not None:
                            cv2.imshow(canvas.name, canvas.output_image)

                        if env is not None and ENABLE_MYOSUITE_RENDER:
                            try:
                                now = time.time()
                                if now - last_render_time >= 1.0 / float(MYOSUITE_RENDER_FPS):
                                    q_target = getattr(listener, 'latest_qpos', None)
                                    if q_target is not None:
                                        qpos_view = env.sim.data.qpos
                                        n = min(len(qpos_view), len(q_target))
                                        qpos_view[:n] = q_target[:n]
                                        if n < len(qpos_view):
                                            qpos_view[n:] = 0.0
                                    if not SHOW_TENSION_IN_CTRL:
                                        listener._reset_ctrl_to_baseline()
                                    env.sim.forward()

                                    try:
                                        listener._update_probe_state(qpos_view)
                                    except Exception:
                                        pass

                                    listener._display_tension_in_ctrl()
                                    if _recording and _record_writer is not None:
                                        try:
                                            ctrl_vec = np.asarray(env.sim.data.ctrl, dtype=float).reshape(-1)
                                            row = [float(now), int(_record_frame)] + [float(x) for x in ctrl_vec]
                                            _record_writer.writerow(row)
                                            _record_frame += 1
                                            if (_record_frame % 30) == 0 and _record_fh is not None:
                                                _record_fh.flush()
                                        except Exception:
                                            pass

                                    env.mj_render()
                                    last_render_time = now
                            except Exception:
                                pass
                        running = _handle_keys()
                        time.sleep(0.001)
            except BaseException as e:
                if isinstance(e, KeyboardInterrupt):
                    raise
                leap_ok = False
                try:
                    print(f"[WARN] Leap connection failed ({type(e).__name__}: {e}).", flush=True)
                except Exception:
                    pass
            try:
                print("[INFO] Exited Leap main loop", flush=True)
            except Exception:
                pass

            if not leap_ok:
                try:
                    print("[WARN] Leap unavailable; exiting (LeapOnly=Yes)", flush=True)
                except Exception:
                    pass
    except KeyboardInterrupt:
        pass
    finally:
        if VISUALIZE_LEAP:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
        try:
            _stop_recording()
        except Exception:
            pass
        if env is not None:
            try:
                env.close()
            except BaseException:
                pass

if __name__ == '__main__':
    main()