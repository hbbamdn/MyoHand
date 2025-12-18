import numpy as np
import time
import os
import glob
import zipfile
import fnmatch
import cv2
import math
import sys
from copy import deepcopy

_PROJECT_ROOT = os.path.dirname(__file__)
_LEAP_LOCAL_SRC = os.path.join(
    _PROJECT_ROOT,
    "leapc-python-bindings",
    "leapc-python-api",
    "src",
)
if _LEAP_LOCAL_SRC not in sys.path:
    sys.path.append(_LEAP_LOCAL_SRC)

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
RECORD_QPOS = True
RECORD_DIR = 'recordings'

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
        with connection.open():
            time.sleep(0.3)
            devices = connection.get_devices()
            if not devices:
                return False

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
    
    if not setup_leap_motion_device(connection):
        return
    
    connection.set_tracking_mode(tracking_mode)
    canvas = Canvas() if VISUALIZE_LEAP else None
    if canvas:
        canvas.set_tracking_mode(tracking_mode)
    env = None
    if ENABLE_MYOSUITE and MYOSUITE_AVAILABLE:
        try:
            env = gym.make('myoHandPoseRandom-v0')
            env.reset()
            try:
                env.mj_render()
            except Exception:
                pass
        except Exception:
            env = None

    class LeapDataListener(leap.Listener):
        def __init__(self, canvas=None, env=None):
            super().__init__()
            self.canvas = canvas
            self.latest_qpos = None
            self.env = env
            self.last_render_time = time.time()
            self.recording = False
            self.rec_start_time = None
            self.rec_t = []
            self.rec_qpos = []

        def start_recording(self):
            self.recording = True
            self.rec_start_time = time.time()
            self.rec_t = []
            self.rec_qpos = []

        def stop_recording(self):
            self.recording = False

        def save_recording(self, out_path):
            if not self.rec_qpos:
                return False
            t = np.asarray(self.rec_t, dtype=float)
            qpos = np.asarray(self.rec_qpos, dtype=float)
            np.savez(out_path, t=t, qpos=qpos)
            return True

        def clear_recording(self):
            self.rec_start_time = None
            self.rec_t = []
            self.rec_qpos = []

        def on_tracking_mode_event(self, event):
            if self.canvas:
                self.canvas.set_tracking_mode(event.current_tracking_mode)
        
        def on_tracking_event(self, event):
            if len(event.hands) > 0:
                hand = event.hands[0]  
                
                try:
                    self.latest_qpos = enhanced_leap_to_qpos(hand, hand.arm)

                except Exception as e:
                    self.latest_qpos = None
                if self.latest_qpos is not None:
                    try:
                        q_target = np.asarray(self.latest_qpos, dtype=float)
                        if self.recording and RECORD_QPOS:
                            now = time.time()
                            if self.rec_start_time is None:
                                self.rec_start_time = now
                            self.rec_t.append(now - self.rec_start_time)
                            self.rec_qpos.append(q_target.copy())

                        if self.env is not None:
                            qpos_view = self.env.sim.data.qpos
                            n = min(len(qpos_view), len(q_target))
                            qpos_view[:n] = q_target[:n]
                            if n < len(qpos_view):
                                qpos_view[n:] = 0.0
                            self.env.sim.forward()
                    except Exception:
                        pass

            else:
                if self.env is not None:
                    try:
                        self.env.sim.forward()
                    except Exception:
                        pass

            if self.canvas:
                self.canvas.render_hands(event)
            if self.env is not None and ENABLE_MYOSUITE_RENDER:
                try:
                    now = time.time()
                    if now - self.last_render_time >= 1.0 / float(MYOSUITE_RENDER_FPS):
                        self.env.mj_render()
                        self.last_render_time = now
                except Exception:
                    pass


    listener = LeapDataListener(canvas, env)
    connection.add_listener(listener)
    try:
        with connection.open():
            connection.set_tracking_mode(tracking_mode)
            DESKTOP_MODE_TRANSFORM = (tracking_mode == leap.TrackingMode.Desktop)
            running = True
            while running:
                if VISUALIZE_LEAP and canvas is not None:
                    cv2.imshow(canvas.name, canvas.output_image)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('x'):
                    running = False
                    break
                elif key == ord('r'):
                    if RECORD_QPOS:
                        if listener.recording:
                            listener.stop_recording()
                            print(f"Recording stopped. Samples: {len(listener.rec_qpos)}")
                        else:
                            listener.start_recording()
                            print("Recording started")
                elif key == ord('p'):
                    if RECORD_QPOS:
                        try:
                            os.makedirs(RECORD_DIR, exist_ok=True)
                            out_path = os.path.join(RECORD_DIR, f"leap_qpos_{int(time.time())}.npz")
                            ok = listener.save_recording(out_path)
                            if ok:
                                print(f"Saved recording: {out_path}")
                            else:
                                print("No recording data to save")
                        except Exception as e:
                            print(f"Failed to save recording: {e}")
                elif key == ord('c'):
                    if RECORD_QPOS:
                        listener.clear_recording()
                        print("Recording buffer cleared")
                elif key == ord('0'):
                    reset_tracking_state()
                elif key == ord('k'):
                    try:
                        if listener.latest_qpos is not None:
                            enhanced_leap_to_qpos._calib_qpos = np.asarray(listener.latest_qpos, dtype=float).copy()
                            print('Calibrated open pose baseline')
                    except Exception:
                        pass
                time.sleep(0.001)
    except KeyboardInterrupt:
        pass
    finally:
        if VISUALIZE_LEAP:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
        if env is not None:
            try:
                env.close()
            except Exception:
                pass

if __name__ == '__main__':
    main()
