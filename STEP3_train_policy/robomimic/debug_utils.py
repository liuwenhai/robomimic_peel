import numpy as np
import pybullet as p
import trimesh

import numpy as np
import os
import cv2
from scipy.spatial.transform import Rotation as Rot
import open3d as o3d
import h5py

import rospy
from sensor_msgs.msg import PointCloud2, JointState
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped, PoseStamped, WrenchStamped, Vector3
from std_msgs.msg import Header
from sensor_msgs import point_cloud2

from inverse_kinematics.pinocchio_model import RobotModel
from collections import deque
from copy import deepcopy
from pynput import keyboard

L515_2_BASE = np.array([[1, 0, 0, 0],
                        [0, -np.sin(70 / 180 * np.pi), np.cos(70 / 180 * np.pi), 0],
                        [0, -np.cos(70 / 180 * np.pi), -np.sin(70 / 180 * np.pi), 0.59],
                        [0, 0, 0, 1]]) # 0.59 0.53


def create_point_cloud_msg(points, now, frame_id='l515'):
    import rospy
    import numpy as np
    import struct  # 用于RGB颜色的打包
    from sensor_msgs.msg import PointCloud2, PointField
    from std_msgs.msg import Header
    """
    将 n x 6 点云数据转换为 sensor_msgs/PointCloud2 消息
    :param points: np.array, n x 6 点云数据，前3列是(x, y, z)，后3列是(r, g, b)，0-1
    :return: sensor_msgs/PointCloud2 消息
    """
    # 创建消息头
    header = Header()
    header.stamp = now
    header.frame_id = frame_id  # 修改为你的坐标系框架ID

    # 定义 PointField
    fields = [
        PointField('x', 0, PointField.FLOAT32, 1),
        PointField('y', 4, PointField.FLOAT32, 1),
        PointField('z', 8, PointField.FLOAT32, 1),
        PointField('rgb', 12, PointField.UINT32, 1),
    ]

    # 构建 PointCloud2 的数据部分
    # point_cloud_data = []

    # for point in points:
    #     x, y, z = point[:3]
    #     r, g, b = point[3:6].astype(np.uint8)  # 转换为uint8类型
    #     rgb = struct.unpack('I', struct.pack('BBBB', b, g, r, 0))[0]  # RGB颜色打包为一个整数
    #     point_cloud_data.append([x, y, z, rgb])

    point_cloud_data = np.zeros((len(points), 4), dtype=np.float32)
    point_cloud_data[:, 0:3] = points[:, 0:3]
    rgb_int = np.zeros((points.shape[0],), dtype=np.uint32)
    rgb_int = (points[:, 3].astype(np.uint32) << 16) | (points[:, 4].astype(np.uint32) << 8) | points[:, 5].astype(
        np.uint32)
    point_cloud_data[:, 3] = rgb_int.view(np.float32)  # 将其作为 float32 存储

    # 使用 PointCloud2 构建点云消息
    point_cloud_msg = PointCloud2()
    point_cloud_msg.header = header
    point_cloud_msg.height = 1  # 表示这是一个无序点云
    point_cloud_msg.width = len(point_cloud_data)
    point_cloud_msg.is_dense = True  # 无NaN点
    point_cloud_msg.is_bigendian = False
    point_cloud_msg.fields = fields
    point_cloud_msg.point_step = 16  # 每个点的字节数 (4个float32，每个4字节)
    point_cloud_msg.row_step = point_cloud_msg.point_step * point_cloud_msg.width
    # point_cloud_msg.data = np.array(point_cloud_data, dtype=np.float32).tobytes()
    point_cloud_msg.data = point_cloud_data.tobytes()

    return point_cloud_msg


class KeyboardCtrl():
    def __init__(self, verbose=False):
        self.finish = False
        self.pause = False
        self.verbose = verbose
        listener = keyboard.Listener(on_press=self._on_press, on_release=self._on_release)
        listener.start()

    def _on_press(self, key):

        if key == keyboard.Key.esc:
            self.finish = True
            # print("finish")

        if key == keyboard.Key.ctrl:
            if self.pause:
                self.pause = False
                if self.verbose:
                    print("Start")
            else:
                self.pause = True
                if self.verbose:
                    print("Pause")

    def _on_release(self, key):
        pass

class PeelFTDataRos():
    def __init__(self):
        rospy.init_node('dataset_ft_peeler')
        self.pc_pub = rospy.Publisher('/point_cloud', PointCloud2, queue_size=10)
        self.wrench_pub = rospy.Publisher('/wrench', WrenchStamped, queue_size=10)

        self.wrench_stamped_msg = WrenchStamped()
        self.wrench_stamped_msg.header.frame_id = "ft_peeler"

        self.br_world2ft = TransformBroadcaster()
        ft_pose = TransformStamped()
        ft_pose.header.frame_id = 'l515'
        ft_pose.child_frame_id = 'ft_peeler'
        self.ft_pose =ft_pose

        self.br_world2l515 = TransformBroadcaster()
        l515_pose = TransformStamped()
        l515_pose.header.frame_id = 'world'
        l515_pose.child_frame_id = 'l515'
        quat = Rot.from_matrix(L515_2_BASE[:3 ,:3]).as_quat()
        l515_pose.transform.translation.x = L515_2_BASE[0 ,3]
        l515_pose.transform.translation.y = L515_2_BASE[1 ,3]
        l515_pose.transform.translation.z = L515_2_BASE[2 ,3]
        l515_pose.transform.rotation.x = quat[0]
        l515_pose.transform.rotation.y = quat[1]
        l515_pose.transform.rotation.z = quat[2]
        l515_pose.transform.rotation.w = quat[3]
        self.l515_pose = l515_pose


    def update(self, clouds, ft_xyz, ft_quat, wrench):
        now = rospy.Time.now()
        # static tf transform
        self.l515_pose.header.stamp = now
        self.br_world2l515.sendTransform(self.l515_pose)

        # update

        cloud_msg = create_point_cloud_msg(clouds, now)
        self.pc_pub.publish(cloud_msg)

        self.ft_pose.transform.translation.x = ft_xyz[0]
        self.ft_pose.transform.translation.y = ft_xyz[1]
        self.ft_pose.transform.translation.z = ft_xyz[2]
        self.ft_pose.transform.rotation.x = ft_quat[0]
        self.ft_pose.transform.rotation.y = ft_quat[1]
        self.ft_pose.transform.rotation.z = ft_quat[2]
        self.ft_pose.transform.rotation.w = ft_quat[3]
        self.ft_pose.header.stamp = now
        self.br_world2ft.sendTransform(self.ft_pose)

        self.wrench_stamped_msg.wrench.force = Vector3(wrench[0] ,wrench[1] ,wrench[2])
        self.wrench_stamped_msg.wrench.torque = Vector3(wrench[3] ,wrench[4] ,wrench[5])
        self.wrench_stamped_msg.header.stamp = now
        self.wrench_pub.publish(self.wrench_stamped_msg)

