"""
# -*- coding: utf-8 -*-
-----------------------------------------------------------------------------------
# Author: Nguyen Mau Dung
# DoC: 2020.08.17
# email: nguyenmaudung93.kstn@gmail.com

# Modifier: Vineet Suryan
# email: suryanvineet47@gmail.com
-----------------------------------------------------------------------------------
# Description: ONNX Testing script
"""

import argparse
import sys
import os
import warnings
import cv2
import torch
import numpy as np
import onnxruntime
from easydict import EasyDict as edict

warnings.filterwarnings("ignore", category=UserWarning)

src_dir = os.path.dirname(os.path.realpath(__file__))
while not src_dir.endswith("sfa"):
    src_dir = os.path.dirname(src_dir)
if src_dir not in sys.path:
    sys.path.append(src_dir)

# pylint: disable=wrong-import-position
import config.kitti_config as cnf
from models.model_utils import create_model
from utils.misc import make_folder, time_synchronized
from utils.evaluation_utils import decode, post_processing, draw_predictions, \
                                   convert_det_to_real_values
from utils.torch_utils import _sigmoid
from utils.visualization_utils import merge_rgb_to_bev, show_rgb_image_with_boxes
from data_process.kitti_data_utils import Calibration
from data_process.kitti_dataloader import create_test_dataloader
from data_process.transformation import lidar_to_camera_box
from export import convert_to_onnx


def parse_configs():
    """Parse config arguments.
    """
    parser = argparse.ArgumentParser(description='Testing config for the Implementation')
    parser.add_argument('--saved_fn', type=str, default='fpn_resnet_18', metavar='FN',
                        help='The name using for saving logs, models,...')
    parser.add_argument('-a', '--arch', type=str, default='fpn_resnet_18', metavar='ARCH',
                        help='The name of the model architecture')
    parser.add_argument('--pretrained_path', type=str,
                        default='../checkpoints/fpn_resnet_18/Model_fpn_resnet_18_epoch_200.pth',
                        metavar='PATH', help='the path of the pretrained checkpoint')
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
    parser.add_argument('--onnx_path', type=str, default='../checkpoints/onnx/fpn_resnet18.onnx',
                        help='file to save onnx model')

    # pylint: disable=no-member
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
    # pylint: disable=no-member
    configs = parse_configs()

    if not os.path.exists(configs.onnx_path):
        model = create_model(configs)
        print('\n\n' + '-*=' * 30 + '\n\n')
        assert os.path.isfile(
            configs.pretrained_path), "No file at {}".format(configs.pretrained_path)
        model.load_state_dict(torch.load(configs.pretrained_path, map_location='cpu'))
        print('Loaded weights from {}\n'.format(configs.pretrained_path))

        configs.device = torch.device(
            'cpu' if configs.no_cuda else 'cuda:{}'.format(configs.gpu_idx))

        model = model.to(device=configs.device)
        model.eval()
        print('Converting to ONNX...')
        im = torch.zeros(1, 3, 608, 608)
        convert_to_onnx(model, im, configs.onnx_path)

    session = onnxruntime.InferenceSession(configs.onnx_path)
    input_name = session.get_inputs()[0].name
    print("The model expects input shape: ", session.get_inputs()[0].shape)
    test_dataloader = create_test_dataloader(configs)
    times = []
    for batch_idx, batch_data in enumerate(test_dataloader):
        metadatas, bev_maps, img_rgbs = batch_data
        input_bev_maps = bev_maps.float().numpy()
        input_bev_maps = onnxruntime.OrtValue.ortvalue_from_numpy(input_bev_maps, 'cuda', 0)
        t1 = time_synchronized()
        outputs = session.run(None, {input_name: input_bev_maps})
        t2 = time_synchronized()

        outputs = [torch.from_numpy(output) for output in outputs]
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
        times.append(t2 - t1)

        print('Done testing the {}th sample, time: {:.1f}ms, speed {:.2f}FPS'.format(
            batch_idx, (t2 - t1) * 1000, 1 / (t2 - t1)))

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
    times = np.array(times)
    print(f"average time: {np.average(times)}, speed: {1 / np.average(times)}")
