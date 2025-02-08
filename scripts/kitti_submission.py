import sys
sys.path.append('.')

from tqdm import tqdm
import os
import numpy as np
import cv2
import argparse
import torch

from lietorch import SE3
import raft3d.projective_ops as pops
from raft3d.raft3d import RAFT3D

from utils import prepare_images_and_depths
from data_readers.kitti import KITTIEval
import torch.nn.functional as F
from torch.utils.data import DataLoader

from glob import glob
from data_readers.frame_utils import *
from config.config_loader import load_config


@torch.no_grad()
def make_kitti_submission(model, iterations):
    loader_args = {'batch_size': 1, 'shuffle': False, 'num_workers': 1, 'drop_last': False}
    data_loader = DataLoader(KITTIEval(), **loader_args)

    DEPTH_SCALE = .1

    for i_batch, data_blob in enumerate(tqdm(data_loader)):
        image1, image2, disp1, disp2, intrinsics = [item.cuda() for item in data_blob]

        img1 = image1[0].permute(1,2,0).cpu().numpy()
        depth1 = DEPTH_SCALE * (intrinsics[0,0] / disp1)
        depth2 = DEPTH_SCALE * (intrinsics[0,0] / disp2)

        ht, wd = image1.shape[2:]
        image1, image2, depth1, depth2, _ = \
            prepare_images_and_depths(image1, image2, depth1, depth2, depth_scale=1.0)

        Ts = model(image1, image2, depth1, depth2, intrinsics, iters=iterations)
        tau_phi = Ts.log()

        # compute optical flow
        flow, _, _ = pops.induced_flow(Ts, depth1, intrinsics)
        flow = flow[0, :ht, :wd, :2].cpu().numpy()

        # compute disparity change
        coords, _ = pops.projective_transform(Ts, depth1, intrinsics)
        disp2 =  intrinsics[0,0] * coords[:,:ht,:wd,2] * DEPTH_SCALE
        disp1 = disp1[0].cpu().numpy()
        disp2 = disp2[0].cpu().numpy()

        KITTIEval.write_prediction(i_batch, disp1, disp2, flow)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='Training configuration file')
    parser.add_argument('--model', help='path the model weights')

    args = parser.parse_args()
    config = load_config(args)

    model = torch.nn.DataParallel(RAFT3D(config))
    model.load_state_dict(torch.load(args.model))

    model.cuda()
    model.eval()

    if not os.path.isdir('kitti_submission'):
        os.mkdir('kitti_submission')
        os.mkdir('kitti_submission/disp_0')
        os.mkdir('kitti_submission/disp_1')
        os.mkdir('kitti_submission/flow')

    make_kitti_submission(model, config["iterations"])
