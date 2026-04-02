import time
import sys
import tty
import termios
import threading

import mujoco.viewer
import mujoco
import numpy as np
from legged_gym import LEGGED_GYM_ROOT_DIR
import torch
import yaml

JOINT_NAMES = [
    'FL_hip_joint', 'FL_thigh_joint', 'FL_calf_joint',
    'FR_hip_joint', 'FR_thigh_joint', 'FR_calf_joint',
    'RL_hip_joint', 'RL_thigh_joint', 'RL_calf_joint',
    'RR_hip_joint', 'RR_thigh_joint', 'RR_calf_joint',
]

ACTUATOR_NAMES = [
    'FL_hip', 'FL_thigh', 'FL_calf',
    'FR_hip', 'FR_thigh', 'FR_calf',
    'RL_hip', 'RL_thigh', 'RL_calf',
    'RR_hip', 'RR_thigh', 'RR_calf',
]

def get_gravity_orientation(quaternion):
    """从 MuJoCo 四元数 (w,x,y,z) 计算投影重力"""
    qw = quaternion[0]
    qx = quaternion[1]
    qy = quaternion[2]
    qz = quaternion[3]

    gravity_orientation = np.zeros(3)

    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)

    return gravity_orientation


def pd_control(target_q, q, kp, target_dq, dq, kd):
    """Calculates torques from position commands"""
    return (target_q - q) * kp + (target_dq - dq) * kd


def quat_to_rpy(quat_wxyz):
    """从 MuJoCo 四元数 (w,x,y,z) 计算 roll, pitch, yaw"""
    w, x, y, z = quat_wxyz
    # roll
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    # pitch
    sinp = 2.0 * (w * y - z * x)
    pitch = np.arcsin(np.clip(sinp, -1.0, 1.0))
    # yaw
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw


class KeyboardController:
    """通过 WASDQER 键盘控制速度指令 (非阻塞)"""

    DEFAULT_VEL_STEP = [0.1, 0.1, 0.2]   # linear_x, linear_y, angular_z 每次按键的增量
    DEFAULT_VEL_MAX = [0.8, 1.0, 2.0]    # 速度上限

    def __init__(self, cmd_array, vel_step=None, vel_max=None):
        self.cmd = cmd_array
        self.vel_step = np.array(vel_step if vel_step is not None else self.DEFAULT_VEL_STEP, dtype=np.float32)
        self.vel_max = np.array(vel_max if vel_max is not None else self.DEFAULT_VEL_MAX, dtype=np.float32)
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            while self._running:
                ch = sys.stdin.read(1).lower()
                updated = True
                if ch == 'w':
                    self.cmd[0] = min(self.cmd[0] + self.vel_step[0], self.vel_max[0])
                elif ch == 's':
                    self.cmd[0] = max(self.cmd[0] - self.vel_step[0], -self.vel_max[0])
                elif ch == 'a':
                    self.cmd[1] = min(self.cmd[1] + self.vel_step[1], self.vel_max[1])
                elif ch == 'd':
                    self.cmd[1] = max(self.cmd[1] - self.vel_step[1], -self.vel_max[1])
                elif ch == 'q':
                    self.cmd[2] = min(self.cmd[2] + self.vel_step[2], self.vel_max[2])
                elif ch == 'e':
                    self.cmd[2] = max(self.cmd[2] - self.vel_step[2], -self.vel_max[2])
                elif ch == 'r':
                    self.cmd[:] = 0.0
                else:
                    updated = False
                if updated:
                    sys.stdout.write(f"\rcmd: vx={self.cmd[0]:+.2f}  vy={self.cmd[1]:+.2f}  yaw={self.cmd[2]:+.2f}  ")
                    sys.stdout.flush()
                elif ch == '\x03':  # Ctrl+C
                    self._running = False
                    break
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    def stop(self):
        self._running = False


if __name__ == "__main__":

    # get config file name from command line
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str, help="config file name in the config folder")
    args = parser.parse_args()
    config_file = args.config_file
    with open(f"{LEGGED_GYM_ROOT_DIR}/deploy_mujoco/configs/{config_file}", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        policy_path = config["policy_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)
        xml_path = config["xml_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)

        simulation_duration = config["simulation_duration"]
        simulation_dt = config["simulation_dt"]
        control_decimation = config["control_decimation"]

        kps = np.ones(12, dtype=np.float32) * np.float32(config["kp"])
        kds = np.ones(12, dtype=np.float32) * np.float32(config["kd"])

        default_angles = np.array(config["default_angles"], dtype=np.float32)

        dof_pos_scale = config["dof_pos_scale"]
        dof_vel_scale = config["dof_vel_scale"]
        action_scale = config["action_scale"]
        cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)
        cmd_max = np.array(config.get("cmd_max", KeyboardController.DEFAULT_VEL_MAX), dtype=np.float32)
        cmd_step = np.array(config.get("cmd_step", KeyboardController.DEFAULT_VEL_STEP), dtype=np.float32)

        num_actions = config["num_actions"]
        num_obs = config["num_obs"]
        clip_obs = config.get("clip_observations", 100.0)
        clip_actions = config.get("clip_actions", 100.0)

        # 站立过渡参数
        standup_duration = config.get("standup_duration", 1.0)
        stabilize_duration = config.get("stabilize_duration", 0.5)

        cmd = np.zeros(3, dtype=np.float32)

    # define context variables
    action = np.zeros(num_actions, dtype=np.float32)
    target_dof_pos = default_angles.copy()
    obs = np.zeros(num_obs, dtype=np.float32)

    counter = 0

    # Load robot model
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt

    # 按名字查找关节和执行器索引
    joint_qpos_addrs = []
    joint_dof_addrs = []
    actuator_ids = []

    for jname, aname in zip(JOINT_NAMES, ACTUATOR_NAMES):
        jid = m.joint(jname).id
        joint_qpos_addrs.append(m.jnt_qposadr[jid])
        joint_dof_addrs.append(m.jnt_dofadr[jid])
        actuator_ids.append(m.actuator(aname).id)

    joint_qpos_addrs = np.array(joint_qpos_addrs)
    joint_dof_addrs = np.array(joint_dof_addrs)
    actuator_ids = np.array(actuator_ids)

    print(f"Joint qpos addrs: {joint_qpos_addrs}")
    print(f"Joint dof addrs:  {joint_dof_addrs}")
    print(f"Actuator ids:     {actuator_ids}")

    # 初始化关节到默认站姿
    for i, addr in enumerate(joint_qpos_addrs):
        d.qpos[addr] = default_angles[i]
    mujoco.mj_forward(m, d)

    # load policy
    policy = torch.jit.load(policy_path)

    # 启动键盘控制
    kb = KeyboardController(cmd, vel_step=cmd_step, vel_max=cmd_max)
    print("键盘控制已启动:")
    print("  W/S: 前进/后退")
    print("  A/D: 左移/右移")
    print("  Q/E: 左转/右转")
    print("  R:   停止 (归零)")
    print("  Ctrl+C: 退出")

    with mujoco.viewer.launch_passive(m, d, show_left_ui=False, show_right_ui=False) as viewer:
        start = time.time()
        while viewer.is_running() and time.time() - start < simulation_duration and kb._running:
            step_start = time.time()

            # 读取关节状态 (按名字索引)
            qj_full = d.qpos[joint_qpos_addrs]
            dqj_full = d.qvel[joint_dof_addrs]

            # PD 控制
            tau = pd_control(target_dof_pos, qj_full, kps, np.zeros_like(kds), dqj_full, kds)
            # 写入执行器 (按名字索引)
            for i, aid in enumerate(actuator_ids):
                d.ctrl[aid] = tau[i]

            mujoco.mj_step(m, d)

            # 使用仿真步之后的状态与 Isaac Gym 的 post_physics_step 对齐。
            qj_full = d.qpos[joint_qpos_addrs]
            dqj_full = d.qvel[joint_dof_addrs]
            counter += 1
            sim_time = counter * simulation_dt

            # ========== Phase 1: 站立过渡 ==========
            if sim_time <= standup_duration:
                rate = min(sim_time / standup_duration, 1.0)
                # 从当前位置渐变到默认站姿
                target_dof_pos = qj_full * (1.0 - rate) + default_angles * rate

            # ========== Phase 2: 稳定 ==========
            elif sim_time <= standup_duration + stabilize_duration:
                target_dof_pos = default_angles.copy()

            # ========== Phase 3: 策略控制 ==========
            elif counter % control_decimation == 0:
                quat = d.qpos[3:7]

                # 倾翻检测
                roll, pitch, yaw = quat_to_rpy(quat)
                if abs(roll) > 0.8 or abs(pitch) > 0.8:
                    print(f"[{sim_time:.2f}s] WARNING: tilt detected! roll={np.degrees(roll):.1f} pitch={np.degrees(pitch):.1f}")

                # 构建观测
                qj = (qj_full - default_angles) * dof_pos_scale
                dqj = dqj_full * dof_vel_scale
                gravity_orientation = get_gravity_orientation(quat)

                obs[:3] = gravity_orientation
                obs[3:6] = cmd * cmd_scale
                obs[6:18] = qj
                obs[18:30] = dqj
                obs[30:42] = action

                # 观测裁剪
                obs = np.clip(obs, -clip_obs, clip_obs)

                obs_tensor = torch.from_numpy(obs).unsqueeze(0)
                # policy inference
                action = policy(obs_tensor).detach().numpy().squeeze()
                action = np.clip(action, -clip_actions, clip_actions)
                # transform action to target_dof_pos
                target_dof_pos = action * action_scale + default_angles

            # Pick up changes to the physics state, apply perturbations, update options from GUI.
            viewer.sync()

            # Rudimentary time keeping, will drift relative to wall clock.
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
