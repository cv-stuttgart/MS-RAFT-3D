import sys
sys.path.append('.')

from tqdm import tqdm
import os
import numpy as np
import argparse
import torch

import raft3d.projective_ops as pops
from raft3d.raft3d import RAFT3D

from utils import prepare_images_and_depths
from data_readers.spring import SpringSceneFlowDataset
from torch.utils.data import DataLoader

from config.config_loader import load_config

import flow_IO


def write_spring_prediction(frame_data, flow, disp2):
    frame, scene, cam, direc, _ = frame_data
    scene_save_folder = os.path.join("spring_submission", scene)

    disp2_folder = os.path.join(scene_save_folder, f'disp2_{direc}_{cam}')
    flow_folder = os.path.join(scene_save_folder, f'flow_{direc}_{cam}')
    os.makedirs(disp2_folder, exist_ok=True)
    os.makedirs(flow_folder, exist_ok=True)

    disp2_path = os.path.join(disp2_folder, f'disp2_{direc}_{cam}_{frame:04d}.dsp5')
    flow_path = os.path.join(flow_folder, f'flow_{direc}_{cam}_{frame:04d}.flo5')
    flow_IO.writeFlo5File(flow.squeeze(0), flow_path)
    flow_IO.writeDsp5File(disp2.squeeze(0), disp2_path)


@torch.no_grad()
def spring_submission(model, iterations):
    loader_args = {'batch_size': 1, 'shuffle': False, 'num_workers': 4, 'drop_last': False}
    dataset = SpringSceneFlowDataset(split='test', do_augment=False, use_estimated_disp=True)
    data_loader = DataLoader(dataset, **loader_args)

    depth_scale = 0.15

    for i_batch, test_data_blob in enumerate(tqdm(data_loader)):
        img1, img2, depth1, depth2, intrinsics = [data_item.cuda() for data_item in test_data_blob[:-1]]

        # pad and normalize
        ht, wd = img1.shape[2:]
        img1, img2, depth1, depth2, padding = prepare_images_and_depths(img1, img2, depth1, depth2, depth_scale=depth_scale)

        # run model
        Ts = model(img1, img2, depth1, depth2, intrinsics, iters=iterations)


        # extract 2D and 3D flow from transformation field
        flow, _, _ = pops.induced_flow(Ts, depth1, intrinsics)
        flow = flow[:, :ht, :wd, :2]

        # extract disparites
        coords, _ = pops.projective_transform(Ts, depth1, intrinsics)
        disp2 = depth_scale * intrinsics[0,0] * coords[:, :ht, :wd, 2]

        if torch.isnan(flow).any() or torch.isnan(disp2).any():
            print("Encountered nan prediction in", *[x[0] for x in test_data_blob[-1]])
            exit(0)

        write_spring_prediction([x[0] for x in test_data_blob[-1]], flow.cpu(), disp2.cpu())

    print("Spring submission finished")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='Training configuration file')
    parser.add_argument('--model', help='Path the model weights')
    parser.add_argument('--output_dir', help='Path for the results')

    args = parser.parse_args()
    config = load_config(args)

    model = torch.nn.DataParallel(RAFT3D(config))
    model.load_state_dict(torch.load(args.model))

    model.cuda()
    model.eval()

    spring_submission(model, config["iterations"])
