#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from super_fast_object_detection.pointcloud2_utils import pointcloud2_to_array
from sensor_msgs.msg import PointCloud2
# from autoware_perception_msgs.msg import ObjectClassification , DetectedObjectKinematics, DetectedObject, DetectedObjects
from autoware_perception_msgs.msg import ObjectClassification, DetectedObjectKinematics, DetectedObject , DetectedObjects
from geometry_msgs.msg import Point, Quaternion, TwistWithCovariance, PoseWithCovariance
from autoware_perception_msgs.msg import Shape

import numpy as np
import sensor_msgs_py.point_cloud2 as pc2
import timeit
import torch
import os
import warnings

from sfa.models.model_utils import create_model
from sfa.utils.demo_utils import parse_demo_configs, do_detect
from sfa.data_process.kitti_bev_utils import makeBEVMap
import sfa.config.kitti_config as cnf
from sfa.data_process.kitti_data_utils import get_filtered_lidar

warnings.filterwarnings("ignore", category=UserWarning)

ID_TO_CLASS_NAME = {
    0: 'pedestrian',
    1: 'car',
    2: 'cyclist',
    -3: 'truck',
    -99: 'tram',
    -1: 'unknown'
}

CLASS_NAME_TO_ENUM = {
    'car': ObjectClassification.CAR,
    'pedestrian': ObjectClassification.PEDESTRIAN,
    'cyclist': ObjectClassification.BICYCLE,
    'truck': ObjectClassification.TRUCK,
    'tram': ObjectClassification.BUS,
    'unknown': ObjectClassification.UNKNOWN
}



def euler_to_quaternion(yaw, pitch, roll):
    qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
    qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
    qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    return [qx, qy, qz, qw]


class SuperFastObjectDetector(Node):
    def __init__(self):
        super().__init__('super_fast_object_detector')
        self.publisher_ = self.create_publisher(DetectedObjects, 'objects', 10)
        self.subscription = self.create_subscription(
            PointCloud2,
            'points_raw',
            self.pointcloud_callback,
            10)

        self.configs = parse_demo_configs()
        self.configs.pretrained_path = os.path.join(
            "/home/star/autoware/src/super_fast_object_detection/checkpoints/fpn_resnet_18/fpn_resnet_18_epoch_300.pth")
            # os.getenv("HOME"),'/autoware/src/super_fast_object_detection/checkpoints/fpn_resnet_18/fpn_resnet_18_epoch_300.pth')
        assert os.path.isfile(self.configs.pretrained_path), f"No file at {self.configs.pretrained_path}"
        self.model = create_model(self.configs)
       
        self.model.load_state_dict(torch.load(self.configs.pretrained_path, map_location='cuda:0'))
        self.configs.device = torch.device('cuda:0' if not self.configs.no_cuda else 'cpu')
       
        self.model = self.model.to(self.configs.device)
        # self.model = self.model.to(self.configs.device).half()  # Use half precision
        self.model = torch.compile(self.model)  # Optional: PyTorch 2.0+ only
        self.model.eval()

        self.get_logger().info("Model loaded and node initialized.")

    def pointcloud_callback(self, scan):
        start = timeit.default_timer()
        gen = []
        # for p in pc2.read_points(scan, field_names=("x", "y", "z", "intensity"), skip_nans=True):
        #     gen.append(np.array([p[0], p[1], p[2], p[3] / 100.0]))
        # gen_numpy = np.array(gen, dtype=np.float32)

        pc_np = pointcloud2_to_array(scan)
        x = pc_np['x']
        y = pc_np['y']
        z = pc_np['z']
        # intensity = pc_np['intensity'] / 100.0

        intensity_raw = pc_np['intensity']
        # Apply min-max normalization to intensity
        intensity_min = np.min(intensity_raw)
        intensity_max = np.max(intensity_raw)
        intensity = (intensity_raw - intensity_min) / (intensity_max - intensity_min + 1e-6)  # Add small epsilon to avoid div-by-zero
        gen_numpy = np.stack((x, y, z, intensity), axis=-1).astype(np.float32)

        stop = timeit.default_timer()
        self.get_logger().info(f"process: {stop - start:.3f} sec")
        start = timeit.default_timer()
        front_lidar = get_filtered_lidar(gen_numpy, cnf.boundary)
        bev_map = makeBEVMap(front_lidar, cnf.boundary)
        bev_map = torch.from_numpy(bev_map).to(self.configs.device).half()
        stop = timeit.default_timer()
        self.get_logger().info(f"BEV cre Time: {stop - start:.3f} sec")
        start = timeit.default_timer()
        with torch.no_grad():
            detections, bev_map, fps = do_detect(self.configs, self.model, bev_map, is_front=True)
        stop = timeit.default_timer()
        self.get_logger().info(f"Infer Time1: {stop - start:.3f} sec")
        start = timeit.default_timer()
        msg = DetectedObjects()
        msg.header.stamp = scan.header.stamp
        msg.header.frame_id = scan.header.frame_id
        

        # self.get_logger().info(f"FPS: {fps:.2f}")
        # self.get_logger().info(f"BEV Map Shape: {bev_map.shape}")
        # detected objects num
        self.get_logger().info(f"Detected Objects: {len(detections)}")


        
        start = timeit.default_timer()
        for j in range(self.configs.num_classes):
            class_name = ID_TO_CLASS_NAME[j]
            for det in detections[j]:
                _score, _x, _y, _z, _h, _w, _l, _yaw = det
                yaw = -_yaw

                x = _y / cnf.BEV_HEIGHT * cnf.bound_size_x + cnf.boundary['minX']
                y = _x / cnf.BEV_WIDTH * cnf.bound_size_y + cnf.boundary['minY']
                z = _z + cnf.boundary['minZ']
                w = _w / cnf.BEV_WIDTH * cnf.bound_size_y
                l = _l / cnf.BEV_HEIGHT * cnf.bound_size_x

                

                obj = DetectedObject()
                obj.existence_probability = float(_score)
                # obj.classification.label = class_name
                # obj.classification.probability = float(_score)

                classification = ObjectClassification()
                classification.probability = float(_score)
                print(f"class_name: {class_name}")
                classification.label = CLASS_NAME_TO_ENUM.get(class_name.lower(), ObjectClassification.UNKNOWN)

                obj.classification = [classification]
                obj.kinematics = DetectedObjectKinematics()
                obj.kinematics.pose_with_covariance.pose.position.x = float(x)
                obj.kinematics.pose_with_covariance.pose.position.y = float(y)
                obj.kinematics.pose_with_covariance.pose.position.z = float(z)

                qx, qy, qz, qw = euler_to_quaternion(yaw, 0, 0)
                obj.kinematics.pose_with_covariance.pose.orientation.x = float(qx)
                obj.kinematics.pose_with_covariance.pose.orientation.y = float(qy)
                obj.kinematics.pose_with_covariance.pose.orientation.z = float(qz)
                obj.kinematics.pose_with_covariance.pose.orientation.w = float(qw)

                obj.shape.dimensions.x = float(l)
                obj.shape.dimensions.y = float(w)
                obj.shape.dimensions.z = float(_h)

                msg.objects.append(obj)


        front_lidar = get_filtered_lidar(gen_numpy, cnf.boundary_back)
        bev_map = makeBEVMap(front_lidar, cnf.boundary_back)
        bev_map = torch.from_numpy(bev_map).to(self.configs.device).half()
        stop = timeit.default_timer()
        # self.get_logger().info(f"BEV cre Time: {stop - start:.3f} sec")
        start = timeit.default_timer()
        with torch.no_grad():
            detections, bev_map, fps = do_detect(self.configs, self.model, bev_map, is_front=True)
        stop = timeit.default_timer()
        # self.get_logger().info(f"Infer Time1: {stop - start:.3f} sec")
        start = timeit.default_timer()
        # msg = DetectedObjects()
        # msg.header.stamp = scan.header.stamp
        # msg.header.frame_id = scan.header.frame_id
        

        # self.get_logger().info(f"FPS: {fps:.2f}")
        # self.get_logger().info(f"BEV Map Shape: {bev_map.shape}")
        # detected objects num
        # self.get_logger().info(f"Detected Objects: {len(detections)}")
        
        start = timeit.default_timer()
        for j in range(self.configs.num_classes):
            class_name = ID_TO_CLASS_NAME[j]
            for det in detections[j]:
                _score, _x, _y, _z, _h, _w, _l, _yaw = det
                yaw = -_yaw

                x = _y / cnf.BEV_HEIGHT * cnf.bound_size_x + cnf.boundary_back['minX']
                y = _x / cnf.BEV_WIDTH * cnf.bound_size_y + cnf.boundary_back['minY']
                z = _z + cnf.boundary_back['minZ']
                w = _w / cnf.BEV_WIDTH * cnf.bound_size_y
                l = _l / cnf.BEV_HEIGHT * cnf.bound_size_x

                

                obj = DetectedObject()
                obj.existence_probability = float(_score)
                # obj.classification.label = class_name
                # obj.classification.probability = float(_score)

                classification = ObjectClassification()
                classification.probability = float(_score)
                print(f"class_name: {class_name}")
                classification.label = CLASS_NAME_TO_ENUM.get(class_name.lower(), ObjectClassification.UNKNOWN)

                obj.classification = [classification]
                obj.kinematics = DetectedObjectKinematics()
                obj.kinematics.pose_with_covariance.pose.position.x = float(x)
                obj.kinematics.pose_with_covariance.pose.position.y = float(y)
                obj.kinematics.pose_with_covariance.pose.position.z = float(z)

                qx, qy, qz, qw = euler_to_quaternion(yaw, 0, 0)
                obj.kinematics.pose_with_covariance.pose.orientation.x = float(qx)
                obj.kinematics.pose_with_covariance.pose.orientation.y = float(qy)
                obj.kinematics.pose_with_covariance.pose.orientation.z = float(qz)
                obj.kinematics.pose_with_covariance.pose.orientation.w = float(qw)

                obj.shape.dimensions.x = float(l)
                obj.shape.dimensions.y = float(w)
                obj.shape.dimensions.z = float(_h)

                msg.objects.append(obj)


        self.publisher_.publish(msg)
        stop = timeit.default_timer()
        self.get_logger().info(f"publi time2: {stop - start:.3f} sec")
        self.get_logger().info(f"------------------------------")


def main(args =None): 
    try:
        rclpy.init(args = args)
        node = SuperFastObjectDetector()
        rclpy.spin(node)
        node.destroy_node()
        rclpy.shutdown()
    except KeyboardInterrupt:
        pass
    except Exception as e : 
        print(e)
        
if __name__ == '__main__':
    main()