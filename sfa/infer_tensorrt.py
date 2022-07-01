"""
# -*- coding: utf-8 -*-
-----------------------------------------------------------------------------------
# Author: Nguyen Mau Dung
# DoC: 2020.08.17
# email: nguyenmaudung93.kstn@gmail.com

# Modifier: Vineet Suryan
# email: suryanvineet47@gmail.com
-----------------------------------------------------------------------------------
# Description: TensorRT testing script
"""

from typing import List
import argparse
import sys
import os
import warnings
import cv2
import time
import torch
import numpy as np
import tensorrt as trt
import pycuda.autoinit  # noqa # pylint: disable=unused-import
import pycuda.driver as cuda
from easydict import EasyDict as edict

warnings.filterwarnings("ignore", category=UserWarning)

src_dir = os.path.dirname(os.path.realpath(__file__))
while not src_dir.endswith("sfa"):
    src_dir = os.path.dirname(src_dir)
if src_dir not in sys.path:
    sys.path.append(src_dir)

import config.kitti_config as cnf
from utils.misc import make_folder
from utils.evaluation_utils import decode, post_processing, draw_predictions, \
                                   convert_det_to_real_values
from utils.torch_utils import _sigmoid
from utils.visualization_utils import merge_rgb_to_bev, show_rgb_image_with_boxes
from data_process.kitti_data_utils import Calibration
from data_process.kitti_dataloader import create_test_dataloader
from data_process.transformation import lidar_to_camera_box


class FPNResnet18TRT:
    """FPN Resnet18 TensorRT inference utility class.
    """
    def __init__(self, engine_path):
        """Initialize.
        """
        # Create a Context on this device,
        self._ctx = cuda.Device(0).make_context()
        self._logger = trt.Logger(trt.Logger.INFO)
        self._stream = cuda.Stream()

        # initiate engine related class attributes
        self._engine = None
        self._context = None
        self._inputs = None
        self._outputs = None
        self._bindings = None

        self._load_model(engine_path)
        self._allocate_buffers()

    def _deserialize_engine(self, trt_engine_path: str) -> trt.tensorrt.ICudaEngine:
        """Deserialize TensorRT Cuda Engine
        Args:
            trt_engine_path (str): path to engine file
        Returns:
            trt.tensorrt.ICudaEngine: deserialized engine
        """
        with open(trt_engine_path, 'rb') as engine_file:
            with trt.Runtime(self._logger) as runtime:
                engine = runtime.deserialize_cuda_engine(engine_file.read())

        return engine
    
    def _allocate_buffers(self) -> None:
        """Allocates memory for inference using TensorRT engine.
        """
        inputs, outputs, bindings = [], [], []
        for binding in self._engine:
            size = trt.volume(self._engine.get_binding_shape(binding))
            dtype = trt.nptype(self._engine.get_binding_dtype(binding))
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            bindings.append(int(device_mem))
            if self._engine.binding_is_input(binding):
                inputs.append({'host': host_mem, 'device': device_mem})
            else:
                outputs.append({'host': host_mem, 'device': device_mem})

        # set buffers
        self._inputs = inputs
        self._outputs = outputs
        self._bindings = bindings

    def _load_model(self, engine_path):
        print("[INFO] Deserializing TensorRT engine ...")
        # build engine with given configs and load it
        if not os.path.exists(engine_path):
            raise FileNotFoundError(f"TensorRT engine does not exist {engine_path}.")

        # deserialize and load engine
        self._engine = self._deserialize_engine(engine_path)\

        if not self._engine:
            raise Exception("[Error] Couldn't deserialize engine successfully !")

        # create execution context
        self._context = self._engine.create_execution_context()
        if not self._context:
            raise Exception(
                "[Error] Couldn't create execution context from engine successfully !")

    def __call__(self, inputs: np.ndarray) -> List[np.ndarray]:
        """Runs inference on the given inputs.
        Args:
            inputs (np.ndarray): channels-first format,
            with/without batch axis
        Returns:
            List[np.ndarray]: inference's output (raw tensorrt output)

        """
        self._ctx.push()

        # copy inputs to input memory
        # without astype gives invalid arg error
        self._inputs[0]['host'] = np.ravel(inputs).astype(np.float32)

        # transfer data to the gpu
        t1 = time.time()
        cuda.memcpy_htod_async(
            self._inputs[0]['device'], self._inputs[0]['host'], self._stream)
        
        # run inference
        self._context.execute_async_v2(bindings=self._bindings,
                                       stream_handle=self._stream.handle)

        # fetch outputs from gpu
        for out in self._outputs:
            cuda.memcpy_dtoh_async(out['host'], out['device'], self._stream)
        t2 = time.time()

        # synchronize stream
        self._stream.synchronize()
        self._ctx.pop()
        return [out['host'] for out in self._outputs], t2 - t1

    def destroy(self):
        """Destroy if any context in the stack.
        """
        try:
            self._ctx.pop()
        except Exception as exception:
            pass


def parse_test_configs():
    parser = argparse.ArgumentParser(description='Testing config for the Implementation')
    parser.add_argument('--saved_fn', type=str, default='fpn_resnet_18', metavar='FN',
                        help='The name using for saving logs, models,...')
    parser.add_argument('-a', '--arch', type=str, default='fpn_resnet_18', metavar='ARCH',
                        help='The name of the model architecture')
    parser.add_argument('--K', type=int, default=50,
                        help='the number of top K')
    parser.add_argument('--no_cuda', action='store_true',
                        help='If true, cuda is not used.')
    parser.add_argument('--gpu_idx', default=0, type=int,
                        help='GPU index to use.')
    parser.add_argument('--num_samples', type=int, default=None,
                        help='Take a subset of the dataset to run and debug')
    parser.add_argument('--num_workers', type=int, default=1,
                        help='Number of threads for loading data')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='mini-batch size (default: 4)')
    parser.add_argument('--peak_thresh', type=float, default=0.2)
    parser.add_argument('--save_test_output', action='store_true',
                        help='If true, the output image of the testing phase will be saved')
    parser.add_argument('--output_format', type=str, default='image', metavar='PATH',
                        help='the type of the test output (support image or video)')
    parser.add_argument('--output_video_fn', type=str, default='out_fpn_resnet_18', metavar='PATH',
                        help='the video filename if the output format is video')
    parser.add_argument('--output-width', type=int, default=608,
                        help='the width of showing output, the height maybe vary')
    parser.add_argument('--onnx_path', type=str, default=None,
                        help='file to save onnx model')
    parser.add_argument('--trt_path', type=str,
                        default='../checkpoints/trt/fpn_resnet18_fp16.engine',
                        help='file to save tensorrt engine')

    configs = edict(vars(parser.parse_args()))
    configs.pin_memory = True
    configs.distributed = False  # For testing on 1 GPU only

    configs.input_size = (608, 608)
    configs.hm_size = (152, 152)
    configs.down_ratio = 4
    configs.max_objects = 50

    configs.imagenet_pretrained = False
    configs.head_conv = 64
    configs.num_classes = 3
    configs.num_center_offset = 2
    configs.num_z = 1
    configs.num_dim = 3
    configs.num_direction = 2  # sin, cos

    configs.heads = {
        'hm_cen': configs.num_classes,
        'cen_offset': configs.num_center_offset,
        'direction': configs.num_direction,
        'z_coor': configs.num_z,
        'dim': configs.num_dim
    }
    configs.num_input_features = 4

    ####################################################################
    ##############Dataset, Checkpoints, and results dir configs#########
    ####################################################################
    configs.root_dir = '../'
    # configs.dataset_dir = os.path.join(configs.root_dir, 'dataset', 'kitti')
    configs.dataset_dir = os.path.join('../../', 'dataset')

    if configs.save_test_output:
        configs.results_dir = os.path.join(configs.root_dir, 'results', configs.saved_fn)
        make_folder(configs.results_dir)

    return configs


if __name__ == '__main__':
    configs = parse_test_configs()
    configs.device = torch.device('cuda:0')

    fpn_resnet = FPNResnet18TRT(configs.trt_path)

    test_dataloader = create_test_dataloader(configs)
    times = []

    print('[INFO] TensorRT warmup ..')
    im = np.zeros((1,3,608,608))
    for i in range(200):
        print('.', end="")
        fpn_resnet(im)
    print('[INFO] Warm up done.')

    for batch_idx, batch_data in enumerate(test_dataloader):
        metadatas, bev_maps, img_rgbs = batch_data
        input_img = bev_maps.float().contiguous()
        
        out_raw, infer_time = fpn_resnet(input_img)
        outputs = [torch.from_numpy(output.reshape((1,-1,152,152))) for output in out_raw]
        outputs[0] = _sigmoid(outputs[0])
        outputs[1] = _sigmoid(outputs[1])

        # detections size (batch_size, K, 10)
        detections = decode(outputs[0], outputs[1], outputs[2], outputs[3],
                            outputs[4], K=configs.K)
        detections = detections.numpy().astype(np.float32)
        detections = post_processing(
            detections, configs.num_classes, configs.down_ratio, configs.peak_thresh)

        detections = detections[0]  # only first batch

        # Draw prediction in the image
        bev_map = (bev_maps.squeeze().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        bev_map = cv2.resize(bev_map, (cnf.BEV_WIDTH, cnf.BEV_HEIGHT))
        bev_map = draw_predictions(bev_map, detections.copy(), configs.num_classes)

        # Rotate the bev_map
        bev_map = cv2.rotate(bev_map, cv2.ROTATE_180)
        img_path = metadatas['img_path'][0]
        img_rgb = img_rgbs[0].numpy()
        img_rgb = cv2.resize(img_rgb, (img_rgb.shape[1], img_rgb.shape[0]))
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        calib = Calibration(img_path.replace(".jpg", ".txt").replace("image_2", "calib"))
        kitti_dets = convert_det_to_real_values(detections)

        if len(kitti_dets) > 0:
            kitti_dets[:, 1:] = lidar_to_camera_box(
                kitti_dets[:, 1:], calib.V2C, calib.R0, calib.P2)
            img_bgr = show_rgb_image_with_boxes(img_bgr, kitti_dets, calib)
        out_img = merge_rgb_to_bev(img_bgr, bev_map, output_width=configs.output_width)
        
        print('\tDone testing the {}th sample, time: {:.1f}ms, speed {:.2f}FPS'.format(
            batch_idx, (infer_time) * 1000, 1 / (infer_time)))
        times.append(infer_time)

        if configs.save_test_output:
            if configs.output_format == 'image':
                img_fn = os.path.basename(metadatas['img_path'][0])[:-4]
                cv2.imwrite(os.path.join(configs.results_dir, '{}.jpg'.format(img_fn)), out_img)
            elif configs.output_format == 'video':
                if out_cap is None:
                    out_cap_h, out_cap_w = out_img.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                    out_cap = cv2.VideoWriter(
                        os.path.join(configs.results_dir, '{}.avi'.format(configs.output_video_fn)),
                        fourcc, 30, (out_cap_w, out_cap_h))
                out_cap.write(out_img)
            else:
                raise TypeError
    fpn_resnet.destroy()
    print(f"average time: {np.average(times) * 1000} ms, speed: {1 / np.average(times)} FPS")
