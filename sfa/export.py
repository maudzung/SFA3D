"""
# -*- coding: utf-8 -*-
-----------------------------------------------------------------------------------
# Author: Vineet Suryan
# DoC: 2022.06.29
# email: suryanvineet47@gmail.com
-----------------------------------------------------------------------------------
# Description: Export ONNX/TensorRT script
"""

import argparse
import sys
import os
import time
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

from easydict import EasyDict as edict
import cv2
import torch
import numpy as np
import onnx
import onnxruntime
import tensorrt as trt
from torchsummary import summary

src_dir = os.path.dirname(os.path.realpath(__file__))
while not src_dir.endswith("sfa"):
    src_dir = os.path.dirname(src_dir)
if src_dir not in sys.path:
    sys.path.append(src_dir)

import config.kitti_config as cnf
from models.model_utils import create_model
from utils.misc import make_folder, time_synchronized
from utils.evaluation_utils import decode, post_processing, draw_predictions, convert_det_to_real_values
from utils.torch_utils import _sigmoid
from utils.visualization_utils import merge_rgb_to_bev, show_rgb_image_with_boxes
from data_process.transformation import lidar_to_camera_box
from data_process.kitti_dataloader import create_test_dataloader
from data_process.kitti_data_utils import Calibration


def parse_test_configs():
    parser = argparse.ArgumentParser(description='Testing config for the Implementation')
    parser.add_argument('--saved_fn', type=str, default='fpn_resnet_18', metavar='FN',
                        help='The name using for saving logs, models,...')
    parser.add_argument('-a', '--arch', type=str, default='fpn_resnet_18', metavar='ARCH',
                        help='The name of the model architecture')
    parser.add_argument('--pretrained_path', type=str,
                        default='../checkpoints/fpn_resnet_18/Model_fpn_resnet_18_epoch_200.pth', metavar='PATH',
                        help='the path of the pretrained checkpoint')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='mini-batch size (default: 4)')
    parser.add_argument('--onnx_path', type=str, default='../checkpoints/onnx/fpn_resnet18.onnx',
                        help='file to save onnx model')
    parser.add_argument('--trt_path', type=str, 
                        default='../checkpoints/trt/fpn_resnet18.engine',
                        help='file to save tensorrt engine')
    parser.add_argument('--fp16', action='store_true',
                        help='If true, fp16 quantization for TensorRT.')

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

    ####################################################################
    ##############Dataset, Checkpoints, and results dir configs#########
    ####################################################################
    configs.root_dir = '../'
    # configs.dataset_dir = os.path.join(configs.root_dir, 'dataset', 'kitti')
    configs.dataset_dir = os.path.join('../../', 'dataset')

    return configs


def convert_to_onnx(model, im, onnx_path=None, simplify=True):
    """Export PyTorch model to ONNX.
    """
    try:
        torch.onnx.export(
                model,
                im,
                onnx_path,
                export_params=True,
                verbose=False,
                opset_version=13,
                do_constant_folding=True,
                input_names=['images'],
                output_names=['output'],
                dynamic_axes=None)
        model_onnx = onnx.load(onnx_path)  # load onnx model
        onnx.checker.check_model(model_onnx)  # check onnx model

        onnx.save(model_onnx, onnx_path)

        if simplify:
            try:
                import onnxsim

                print(f'[INFO] simplifying with onnx-simplifier {onnxsim.__version__}...')
                model_onnx, check = onnxsim.simplify(model_onnx,
                                                     dynamic_input_shape=False,
                                                     input_shapes=None)
                assert check, 'assert check failed'
                onnx.save(model_onnx, onnx_path)
            except Exception as e:
                print(f'[INFO] simplifier failure: {e}')
        return onnx_path
    except Exception as e:
        print((f'[INFO] export failure: {e}'))
    

def convert_to_trt(onnx_path, trt_engine_path, fp16=False):
    """Convert ONNX to TensorRT.
    """
    # Checks if onnx path exists.
    if not os.path.exists(onnx_path):
        raise FileNotFoundError(
            f"[Error] {onnx_path} does not exists.")
    
    # Check if onnx_path is valid.
    if ".onnx" not in onnx_path:
        raise TypeError(
            f"[Error] Expected onnx weight file, instead {onnx_path} is given."
        )

    # Specify that the network should be created with an explicit batch dimension.
    batch_size = 1 << (int)(
        trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

    trt_logger = trt.Logger(trt.Logger.INFO)
    # Build and serialize engine.
    with trt.Builder(trt_logger) as builder, \
         builder.create_network(batch_size) as network, \
         trt.OnnxParser(network, trt_logger) as parser:

        # Setup builder config.
        config = builder.create_builder_config()
        config.max_workspace_size = 256 *  1 << 20  # 256 MB
        builder.max_batch_size = 1

        # FP16 quantization.
        if builder.platform_has_fast_fp16 and fp16:
            trt_engine_path = trt_engine_path.replace('.engine', '_fp16.engine')
            config.flags = 1 << (int)(trt.BuilderFlag.FP16)
        else:
            trt_engine_path = trt_engine_path.replace('.engine', '_fp32.engine')
        if os.path.exists(trt_engine_path):
            print(f"{trt_engine_path} already exists.",
            f"Please delete or change trt_path with --trt_path \"your_engine_file_path.engine\"")
            return
        # Parse onnx model.
        with open(onnx_path, 'rb') as onnx_file:
            if not parser.parse(onnx_file.read()):
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
        
        # Build engine.
        engine = builder.build_engine(network, config)
        with open(trt_engine_path, 'wb') as trt_engine_file:
            trt_engine_file.write(engine.serialize())
        print("[INFO] Engine serialized and saved !")
        return engine

if __name__ == '__main__':
    configs = parse_test_configs()

    model = create_model(configs)
    print('\n\n' + '-*=' * 30 + '\n\n')
    assert os.path.isfile(configs.pretrained_path), "No file at {}".format(configs.pretrained_path)
    model.load_state_dict(torch.load(configs.pretrained_path, map_location='cpu'))
    print('Loaded weights from {}\n'.format(configs.pretrained_path))

    model.eval()
    print('Converting to ONNX...')
    if not os.path.exists(configs.onnx_path):
        im = torch.zeros(1, 3, 608, 608)
        convert_to_onnx(model, im, configs.onnx_path)
    else:
        print("Model already exists.")
    
    print('Converting to TensorRT...')
    convert_to_trt(configs.onnx_path, configs.trt_path, fp16=configs.fp16)