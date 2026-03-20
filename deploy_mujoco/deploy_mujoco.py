import time

import mujoco.viewer
import mujoco
import numpy as np
from legged_gym import LEGGED_GYM_ROOT_DIR
import torch
import yaml

import rospy
from geometry_msgs.msg import Twist

# Go1 关节名称 (与 Isaac Gym URDF 解析顺序一致: FR, FL, RR, RL)
JOINT_NAMES = [
    'FR_hip_joint', 'FR_thigh_joint', 'FR_calf_joint',
    'FL_hip_joint', 'FL_thigh_joint', 'FL_calf_joint',
    'RR_hip_joint', 'RR_thigh_joint', 'RR_calf_joint',
    'RL_hip_joint', 'RL_thigh_joint', 'RL_calf_joint',
]

ACTUATOR_NAMES = [
    'FR_hip', 'FR_thigh', 'FR_calf',
    'FL_hip', 'FL_thigh', 'FL_calf',
    'RR_hip', 'RR_thigh', 'RR_calf',
    'RL_hip', 'RL_thigh', 'RL_calf',
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


def cmd_vel_callback(msg):
    global cmd
    cmd[0] = np.clip(msg.linear.x, -0.8, 0.8)
    cmd[1] = np.clip(msg.linear.y, -1.0, 1.0)
    cmd[2] = np.clip(msg.angular.z, -2.0, 2.0)


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

    # init ROS node and subscriber
    rospy.init_node("deploy_mujoco", anonymous=True)
    rospy.Subscriber("cmd_vel", Twist, cmd_vel_callback)

    with mujoco.viewer.launch_passive(m, d) as viewer:
        start = time.time()
        while viewer.is_running() and time.time() - start < simulation_duration:
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
