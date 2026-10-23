#!/usr/bin/env python3
"""
V-JEPA 2-AC goal-image planner for a Franka in Isaac Sim (ROS 2).

Input : ONE goal image.  Nothing else is scripted (no open/close gripper routine).
Loop  : encode current frame -> CEM over actions with the AC predictor -> execute the first
        action -> wait until the arm has settled and a fresh frame arrived -> re-plan.

Planner follows facebookresearch/vjepa2 notebooks/utils/mpc_utils.py:
  * action = [dx, dy, dz, dgripper]; rotation deltas are fixed to 0
  * xyz sampling std starts at max_action (NOT 1.0), gripper std starts at 1.0
  * xyz clipped to +-max_action, gripper delta clipped to +-0.75
  * elites refit mean/std with momentum (0.15)
  * final action = mean of the last distribution; gripper delta below 0.25 is rounded to 0
  * energy = mean |z_pred - z_goal| over all tokens and channels (L1)

Kinematics: Robotics Toolbox (fkine + jacob0) on 'panda_link8' + an explicit TCP transform.

Run:
  python3 vjepa2_ac_planner.py --ros-args -p goal_image:=goal_image.png -p max_steps:=40
"""
import gc
import os
import time
import warnings

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from sensor_msgs.msg import Image, JointState
from cv_bridge import CvBridge
from scipy.spatial.transform import Rotation as Rot
import roboticstoolbox as rtb
import spatialmath.base as smb

warnings.filterwarnings("ignore", category=FutureWarning)


IMAGE_TOPIC, JOINT_TOPIC, CMD_TOPIC = "/rgb", "/joint_states", "/joint_command"
JOINTS = [f"panda_joint{i}" for i in range(1, 8)]


# ======================================================================================
# Kinematics (library based)
# ======================================================================================
def pose_err(T_cur, T_tgt):
    """6-D error in the base frame: [position error, rotation-vector error]."""
    return np.concatenate([
        T_tgt[:3, 3] - T_cur[:3, 3],
        Rot.from_matrix(T_tgt[:3, :3] @ T_cur[:3, :3].T).as_rotvec(),
    ])


class Kinematics:
    """
    FK/Jacobian come from Robotics Toolbox, always measured at 'panda_link8' (the flange).
    NOTE: rtb's default end-effector (robot.fkine(q) without end=) already contains the Franka
    hand TCP (0.1034 m, -45 deg about z). Using it AND adding your own tool double counts.
    """
    END = "panda_link8"

    def __init__(self, tcp_z, tcp_rz):
        self.robot = rtb.models.Panda()
        self.qlo, self.qhi = self.robot.qlim
        self.tool = smb.transl(0, 0, tcp_z) @ smb.trotz(tcp_rz)    # flange -> TCP
        self.tool_inv = np.linalg.inv(self.tool)

    def fk(self, q):
        return self.robot.fkine(q, end=self.END).A @ self.tool

    def ee_state(self, q, grip):
        """7-D model state: xyz, EXTRINSIC xyz euler (same as the V-JEPA 2-AC code), gripper 0..1."""
        T = self.fk(q)
        eul = Rot.from_matrix(T[:3, :3]).as_euler("xyz")
        return np.concatenate([T[:3, 3], eul, [grip]]).astype(np.float32)

    def ik(self, q0, T_tgt, iters=30, lam=0.05, max_step=0.2):
        """Local damped-least-squares IK started from the current q (no random restarts, no
        elbow flips). Uses rtb fkine/jacob0. Returns (q, position_error_m)."""
        T_w = T_tgt @ self.tool_inv                                # wrist target
        q = np.array(q0, dtype=float)
        for _ in range(iters):
            T_c = self.robot.fkine(q, end=self.END).A
            e = pose_err(T_c, T_w)
            if np.linalg.norm(e[:3]) < 5e-4 and np.linalg.norm(e[3:]) < 2e-3:
                break
            J = self.robot.jacob0(q, end=self.END)
            dq = J.T @ np.linalg.solve(J @ J.T + lam ** 2 * np.eye(6), e)
            q = np.clip(q + np.clip(dq, -max_step, max_step), self.qlo, self.qhi)
        return q, float(np.linalg.norm(pose_err(self.robot.fkine(q, end=self.END).A, T_w)[:3]))


def link_checkpoint(src):
    """torch.hub looks for the AC checkpoint in its cache; symlink the local file there."""
    if not os.path.isfile(src):
        raise FileNotFoundError(f"checkpoint not found: {src}")
    cache_dir = os.path.expanduser("~/.cache/torch/hub/checkpoints")
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, "vjepa2-ac-vitg.pt")
    if not os.path.lexists(cache_path):
        os.symlink(src, cache_path)


# ======================================================================================
# Node
# ======================================================================================
class VJepaMPC(Node):
    def __init__(self):
        super().__init__("vjepa2_ac_mpc")
        P = self.declare_parameter
        self.goal_path = P("goal_image", "goal_image.png").value
        self.max_steps = int(P("max_steps", 30).value)
        self.ckpt = P("ckpt", "/home/ediilebudu/vjepa2-ac-vitg.pt").value
        self.precision = P("precision", "fp32").value
        self.yaw = np.deg2rad(float(P("action_yaw_deg", -120.0).value))   # camera -> base frame (calibrate!)
        self.z_min = float(P("z_min_m", 0.0).value)                     # safety only

        self.H = int(P("horizon", 1).value)
        self.N = int(P("num_samples", 256).value)
        self.iters = int(P("cem_iters", 2).value)
        self.K = int(P("num_elites", 4).value)
        self.chunk = int(P("chunk", 256).value)                          # forward batch size (memory only)
        self.max_xyz = float(P("max_action_m", 0.075).value)
        self.mom_mean = float(P("momentum_mean", 0.15).value)
        self.mom_std = float(P("momentum_std", 0.15).value)
        self.dev = "cuda"

        # ---- gripper: model state 0 = open, 1 = closed ----
        if P("gripper_type", "robotiq").value == "franka":
            self.g_joints, self.g_open, self.g_closed = ["panda_finger_joint1", "panda_finger_joint2"], 0.04, 0.0
        else:
            self.g_joints, self.g_open, self.g_closed = ["finger_joint"], 0.0, 0.8

        self.kin = Kinematics(float(P("tcp_offset_m", 0.2104).value),
                              np.deg2rad(float(P("tcp_rz_deg", -45.0).value)))
        self.settle_tol = float(P("settle_tol", 0.02).value)
        self.settle_timeout = float(P("settle_timeout_s", 4.0).value)
        self.frame_wait = float(P("frame_wait_s", 0.2).value)
        self.max_qd = float(P("max_joint_speed", 0.8).value)             # rad/s, ramp limit on joint targets

        torch.set_float32_matmul_precision("high")
        if self.precision == "fp32":
            self.dtype, self.use_amp = torch.float32, False
        else:
            self.dtype, self.use_amp = torch.bfloat16, True

        # ---- runtime state ----
        self.q, self.grip = None, None
        self.latest_rgb, self.frame_time = None, 0.0
        self.phase, self.min_frame_time, self.t_cmd = "wait_frame", 0.0, 0.0
        self.q_cmd, self.g_pos = None, None
        self.q_ref, self.qd, self.t_ref = None, None, 0.0
        self.step = 0
        self.bridge = CvBridge()

        # ---- model ----
        self.get_logger().info(f"Loading V-JEPA 2-AC ({self.precision}) ...")
        link_checkpoint(self.ckpt)
        self.encoder, self.predictor = torch.hub.load("facebookresearch/vjepa2", "vjepa2_ac_vit_giant")
        self.encoder.eval().to(self.dev, self.dtype)
        self.predictor.eval().to(self.dev, self.dtype)
        if P("compile", False).value:                                    # optional; slow first call
            self.predictor = torch.compile(self.predictor)
        gc.collect()
        torch.cuda.empty_cache()
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.dev).view(3, 1, 1)
        self.std_img = torch.tensor([0.229, 0.224, 0.225], device=self.dev).view(3, 1, 1)

        if not os.path.isfile(self.goal_path):
            raise FileNotFoundError(f"goal image missing: {self.goal_path}")
        bgr = cv2.imread(self.goal_path)
        self.z_goal = self.encode(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).float()   # [1, 256, 1408]
        self.tpf = self.z_goal.shape[1]
        self.get_logger().info(
            f"goal={self.goal_path} | N={self.N} iters={self.iters} K={self.K} H={self.H} "
            f"| forwards/plan={self.N * self.iters}")

        self.create_subscription(Image, IMAGE_TOPIC, self.on_image, QoSProfile(depth=1))
        self.create_subscription(JointState, JOINT_TOPIC, self.on_joints, 10)
        self.pub = self.create_publisher(JointState, CMD_TOPIC, 10)
        self.create_timer(0.05, self.tick)

    # ---------------- callbacks ----------------
    def on_image(self, msg):
        self.latest_rgb = self.bridge.imgmsg_to_cv2(msg, "rgb8")
        self.frame_time = time.monotonic()

    def on_joints(self, msg):
        d = dict(zip(msg.name, msg.position))
        if all(j in d for j in JOINTS):
            self.q = np.array([d[j] for j in JOINTS])
            if len(msg.velocity) == len(msg.name):
                dv = dict(zip(msg.name, msg.velocity))
                self.qd = np.array([dv[j] for j in JOINTS])
            gj = d.get(self.g_joints[0])
            if gj is not None:
                self.grip = float(np.clip((gj - self.g_open) / (self.g_closed - self.g_open), 0.0, 1.0))

    def send(self, q_cmd, g_pos):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = JOINTS + self.g_joints
        msg.position = list(map(float, q_cmd)) + [float(g_pos)] * len(self.g_joints)
        self.pub.publish(msg)

    # ---------------- model ----------------
    @torch.inference_mode()
    def encode(self, rgb):
        """RGB uint8 -> layer-normed latent [1, 256, 1408] (frame duplicated to fill the tubelet)."""
        img = cv2.resize(rgb, (256, 256), interpolation=cv2.INTER_AREA)
        x = torch.from_numpy(img).to(self.dev).permute(2, 0, 1).float() / 255.0
        x = (x - self.mean) / self.std_img
        clip = x[None, :, None].repeat(1, 1, 2, 1, 1).to(self.dtype)
        with torch.autocast("cuda", dtype=self.dtype, enabled=self.use_amp):
            h = self.encoder(clip)
        return F.layer_norm(h.float(), (h.size(-1),))

    @torch.inference_mode()
    def _energy_chunk(self, z_ctx, state, a):
        """a: [n, H, 7]. Roll the predictor H steps, L1 distance of the last latent to the goal."""
        n, H, _ = a.shape
        z_seq = z_ctx.expand(n, -1, -1).to(self.dtype)
        states = torch.as_tensor(state, device=self.dev, dtype=self.dtype).view(1, 1, 7).repeat(n, 1, 1)
        acts = a.to(self.dtype)
        z_pred = None
        for t in range(H):
            with torch.autocast("cuda", dtype=self.dtype, enabled=self.use_amp):
                z_pred = self.predictor(z_seq, acts[:, :t + 1], states)[:, -self.tpf:]
            z_pred = F.layer_norm(z_pred.float(), (z_pred.size(-1),))
            if t < H - 1:
                z_seq = torch.cat([z_seq, z_pred.to(self.dtype)], dim=1)
                nxt = states[:, -1:] + acts[:, t:t + 1]          # rotation deltas are 0
                nxt[..., 6] = nxt[..., 6].clamp(0.0, 1.0)
                states = torch.cat([states, nxt], dim=1)
        return (z_pred - self.z_goal).abs().mean(dim=(1, 2))

    @torch.inference_mode()
    def energy(self, z_ctx, state, a):
        return torch.cat([self._energy_chunk(z_ctx, state, a[i:i + self.chunk])
                          for i in range(0, a.shape[0], self.chunk)])

    @torch.inference_mode()
    def cem(self, z_ctx, state):
        """Returns ([dx, dy, dz, dgrip] in the model frame, best energy)."""
        H, N, dev = self.H, self.N, self.dev
        mean = torch.zeros(H, 4, device=dev)                                         # xyz + gripper
        std = torch.cat([torch.full((H, 3), self.max_xyz, device=dev),
                         torch.ones(H, 1, device=dev)], dim=-1)
        best = None
        for _ in range(self.iters):
            s = torch.randn(N, H, 4, device=dev) * std + mean
            s = torch.cat([s[..., :3].clamp(-self.max_xyz, self.max_xyz),
                           s[..., 3:].clamp(-0.75, 0.75)], dim=-1)
            a = torch.cat([s[..., :3], torch.zeros(N, H, 3, device=dev), s[..., 3:]], dim=-1)  # [N,H,7]
            e = self.energy(z_ctx, state, a)
            vals, idx = torch.topk(e, self.K, largest=False)
            sel = s[idx]
            mean = sel.mean(0) * (1 - self.mom_mean) + mean * self.mom_mean
            std = (sel.std(0) * (1 - self.mom_std) + std * self.mom_std).clamp_min(1e-3)
            best = vals[0]
        xyz, g = mean[0, :3], mean[0, 3]
        g = torch.where(g.abs() < 0.25, torch.zeros_like(g), g)                      # dead-band
        return torch.cat([xyz, g[None]]).cpu().numpy(), float(best)

    # ---------------- control loop ----------------
    def tick(self):
        if self.q is None or self.grip is None or self.latest_rgb is None:
            return
        now = time.monotonic()

        if self.phase == "settling":
            # ramp the joint target toward q_cmd at <= max_joint_speed instead of one big step
            dt = now - self.t_ref
            self.t_ref = now
            self.q_ref = self.q_ref + np.clip(self.q_cmd - self.q_ref, -self.max_qd * dt, self.max_qd * dt)
            self.send(self.q_ref, self.g_pos)
            err = np.max(np.abs(self.q - self.q_cmd))
            moving = self.qd is not None and np.max(np.abs(self.qd)) > 0.02
            arrived = np.allclose(self.q_ref, self.q_cmd, atol=1e-6)
            if (arrived and err < self.settle_tol and not moving) or now - self.t_cmd > self.settle_timeout:
                self.get_logger().info(f"settled in {now - self.t_cmd:.1f}s, residual {err:.3f} rad")
                self.phase, self.min_frame_time = "wait_frame", now + self.frame_wait
            return
        if self.frame_time < self.min_frame_time:        # need a frame captured after the arm stopped
            return

        if self.step >= self.max_steps:
            self.get_logger().info("Reached max_steps, done.")
            raise SystemExit
        self.step += 1

        q, grip = self.q.copy(), self.grip
        t0 = time.monotonic()
        state = self.kin.ee_state(q, grip)
        z_ctx = self.encode(self.latest_rgb)
        t1 = time.monotonic()
        a, e = self.cem(z_ctx, state)
        t2 = time.monotonic()

        # model frame -> robot base frame (rotation about z)
        c, s = np.cos(self.yaw), np.sin(self.yaw)
        d = np.array([c * a[0] - s * a[1], s * a[0] + c * a[1], a[2]])
        d[2] = max(d[2], self.z_min - state[2])

        # IK; if the target is unreachable, shrink the move instead of sending a bad configuration
        scale = 1.0
        for _ in range(4):
            T_tgt = self.kin.fk(q)
            T_tgt[:3, 3] += d * scale
            q_try, ik_err = self.kin.ik(q, T_tgt)
            if ik_err < 5e-3:
                break
            scale *= 0.5
        else:
            self.get_logger().warn(f"target unreachable (IK err {ik_err * 1000:.0f}mm), holding arm pose")
            q_try = q.copy()
        self.q_cmd = q_try
        t3 = time.monotonic()

        g_new = float(np.clip(grip + a[3], 0.0, 1.0))
        self.g_pos = self.g_open + g_new * (self.g_closed - self.g_open)
        self.q_ref, self.t_ref = q.copy(), time.monotonic()      # ramp starts from where the arm is now
        self.phase, self.t_cmd = "settling", time.monotonic()    # timeout counts from the SEND, not from tick start

        warn = f" | move scaled x{scale:.2f}" if scale < 1.0 else ""
        self.get_logger().info(
            f"step {self.step}/{self.max_steps} | energy {e:.4f} | move(m) {np.round(d, 3)} | "
            f"grip {grip:.2f}->{g_new:.2f} | EE {np.round(state[:3], 3)} | "
            f"max dq {np.max(np.abs(self.q_cmd - q)):.2f}rad | "
            f"enc {t1 - t0:.1f}s cem {t2 - t1:.1f}s ik {t3 - t2:.2f}s{warn}")


def main():
    rclpy.init()
    node = VJepaMPC()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
