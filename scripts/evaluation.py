import sys
sys.path.append('.')

from tqdm import tqdm
import numpy as np
import cv2
import argparse
import torch
import torch.nn.functional as F

from lietorch import SE3
import raft3d.projective_ops as pops
from raft3d.raft3d import RAFT3D

from utils import show_image, normalize_image, prepare_images_and_depths
from config.config_loader import load_config
from data_readers.sceneflow import FlyingThingsTest
from data_readers.kitti import KITTIEval
from data_readers.spring import SpringSceneFlowDataset
from torch.utils.data import DataLoader

from glob import glob
from collections import defaultdict
from data_readers.frame_utils import *

from pprint import pprint

# scale input depth maps (scaling is undone before evaluation)
DEPTH_SCALE = 0.2

# exclude pixels with depth > 250
MAX_DEPTH = 250

# exclude extermely fast moving pixels
MAX_FLOW = 250

import flow_errors


@torch.no_grad()
def test_sceneflow(model, iterations):
    loader_args = {'batch_size': 1, 'shuffle': False, 'num_workers': 4, 'drop_last': False}
    train_dataset = FlyingThingsTest()
    train_loader = DataLoader(train_dataset, **loader_args)

    count_all, count_sampled = 0, 0
    metrics = {'all_epe2d': 0.0, 'all_epe3d': 0.0, 'all_1px': 0.0, 'all_5cm': 0.0, 'all_10cm': 0.0, 'flownet3d_epe3d': 0.0, 'flownet3d_5cm': 0.0, 'flownet3d_10cm': 0.0}

    for i_batch, test_data_blob in enumerate(tqdm(train_loader)):
        image1, image2, depth1, depth2, flow2d, flow3d, intrinsics, index = \
            [data_item.cuda() for data_item in test_data_blob]

        mag = torch.sum(flow2d**2, dim=-1).sqrt()
        valid = (mag.reshape(-1) < MAX_FLOW) & (depth1.reshape(-1) < MAX_DEPTH)

        # pad and normalize images
        image1, image2, depth1, depth2, padding = \
            prepare_images_and_depths(image1, image2, depth1, depth2, DEPTH_SCALE)

        # run the model
        Ts = model(image1, image2, depth1, depth2, intrinsics, iters=iterations)

        # use transformation field to extract 2D and 3D flow
        flow2d_est, flow3d_est, _ = pops.induced_flow(Ts, depth1, intrinsics)

        # unpad the flow fields / undo depth scaling
        flow2d_est = flow2d_est[:, :-4, :, :2]
        flow3d_est = flow3d_est[:, :-4] / DEPTH_SCALE

        epe2d = torch.sum((flow2d_est - flow2d)**2, -1).sqrt()
        epe3d = torch.sum((flow3d_est - flow3d)**2, -1).sqrt()

        # our evaluation (use all valid pixels)
        epe2d_all = epe2d.reshape(-1)[valid].double().cpu().numpy()
        epe3d_all = epe3d.reshape(-1)[valid].double().cpu().numpy()

        count_all += epe2d_all.shape[0]
        metrics['all_epe2d'] += epe2d_all.sum()
        metrics['all_epe3d'] += epe3d_all.sum()
        metrics['all_1px'] += np.count_nonzero(epe2d_all < 1.0)
        metrics['all_5cm'] += np.count_nonzero(epe3d_all < .05)
        metrics['all_10cm'] += np.count_nonzero(epe3d_all < .10)

        # FlowNet3D evaluation (only use sampled non-occ pixels)
        epe3d = epe3d[0,index[0,0],index[0,1]]
        epe2d = epe2d[0,index[0,0],index[0,1]]

        epe2d_sampled = epe2d.reshape(-1).double().cpu().numpy()
        epe3d_sampled = epe3d.reshape(-1).double().cpu().numpy()

        count_sampled += epe2d_sampled.shape[0]
        metrics['flownet3d_epe3d'] += epe3d_sampled.mean()
        metrics['flownet3d_5cm'] += (epe3d_sampled < .05).astype(float).mean()
        metrics['flownet3d_10cm'] += (epe3d_sampled < .10).astype(float).mean()

    for key in metrics:
        if key.startswith("all_"):
            metrics[key] /= count_all
        elif key.startswith("flownet3d_"):
            metrics[key] /= (i_batch + 1)

    return metrics


@torch.no_grad()
def test_kitti(model, iterations):
    loader_args = {'batch_size': 1, 'shuffle': False, 'num_workers': 4, 'drop_last': False}
    train_dataset = KITTIEval(do_augment=False, mode="training")
    train_loader = DataLoader(train_dataset, **loader_args)

    metrics = {'D1': 0.0, 'D2': 0.0, 'Fl': 0.0, 'SF': 0.0}

    depth_scale = 0.1

    for i_batch, test_data_blob in enumerate(tqdm(train_loader)):
        image1, image2, disp1, disp2, intrinsics, gt_d1, gt_d2, gt_fl = test_data_blob
        image1, image2, disp1, disp2, intrinsics = [data_item.cuda() for data_item in [image1, image2, disp1, disp2, intrinsics]]

        depth1 = depth_scale * (intrinsics[0,0] / disp1)
        depth2 = depth_scale * (intrinsics[0,0] / disp2)

        ht, wd = image1.shape[2:]
        # pad and normalize images
        image1, image2, depth1, depth2, padding = prepare_images_and_depths(image1, image2, depth1, depth2, depth_scale=1.0)

        # run the model
        Ts = model(image1, image2, depth1, depth2, intrinsics, iters=iterations)

        # use transformation field to extract 2D and 3D flow
        flow, _, _ = pops.induced_flow(Ts, depth1, intrinsics)
        flow = flow[0, :ht, :wd, :2].cpu().numpy()

        coords, _ = pops.projective_transform(Ts, depth1, intrinsics)
        disp2 =  intrinsics[0,0] * coords[:,:ht,:wd,2] * depth_scale
        disp1 = disp1[0].cpu().numpy()
        disp2 = disp2[0].cpu().numpy()

        disp1 = np.pad(disp1, ((KITTIEval.crop,0),(0,0)), mode='edge')
        disp2 = np.pad(disp2, ((KITTIEval.crop, 0), (0,0)), mode='edge')
        flow = np.pad(flow, ((KITTIEval.crop, 0), (0,0),(0,0)), mode='edge')

        d1_badcount, d1_pxcount, d2_badcount, d2_pxcount, fl_badcount, fl_pxcount, sf_badcount, sf_pxcount = flow_errors.compute_SF(disp1, disp2, flow, gt_d1.numpy()[0], gt_d2.numpy()[0], gt_fl.numpy()[0])
        metrics["D1"] += 100 * d1_badcount / d1_pxcount
        metrics["D2"] += 100 * d2_badcount / d2_pxcount
        metrics["Fl"] += 100 * fl_badcount / fl_pxcount
        metrics["SF"] += 100 * sf_badcount / sf_pxcount

    metrics["D1"] /= len(train_dataset)
    metrics["D2"] /= len(train_dataset)
    metrics["Fl"] /= len(train_dataset)
    metrics["SF"] /= len(train_dataset)

    return metrics


def compute_spring_errors(d1_pred, d2_pred, fl_pred, d1_gt, d2_gt, fl_gt):
    def compute_errors(epe, gt):
        valid = ~torch.isnan(epe)

        abs_error = epe[valid].mean()

        kitti = (epe > 3) & (epe > 0.05 * torch.abs(gt))
        kitti_error = 100 * kitti.sum() / valid.sum()

        onepx = epe > 1
        onepx_error = 100 * onepx.sum() / valid.sum()

        return (abs_error, kitti_error, onepx_error), kitti, onepx

    d1_err, d1_kitti, d1_onepx = compute_errors(torch.abs(d1_pred - d1_gt), d1_gt)
    d2_err, d2_kitti, d2_onepx = compute_errors(torch.abs(d2_pred - d2_gt), d2_gt)
    fl_err, fl_kitti, fl_onepx = compute_errors(torch.linalg.norm(fl_pred - fl_gt, dim=-1), torch.linalg.norm(fl_gt, dim=-1))

    valid = ~(torch.isnan(d1_gt) | torch.isnan(d2_gt) | torch.isnan(fl_gt).any(axis=-1))

    sf_kitti = d1_kitti | d2_kitti | fl_kitti
    sf_kitti_error = 100 * sf_kitti.sum() / valid.sum()

    sf_onepx = d1_onepx | d2_onepx | fl_onepx
    sf_onepx_error = 100 * sf_onepx.sum() / valid.sum()

    metrics = {
        'Sf': {
            'kitti': sf_kitti_error,
            '1px': sf_onepx_error
        }
    }

    for name, err in zip(['D1', 'D2', 'Fl'], [d1_err, d2_err, fl_err]):
        metrics[name] = {
            'abs': err[0],
            'kitti': err[1],
            '1px': err[2]
        }

    return metrics


@torch.no_grad()
def test_spring(model, iterations, scenes=None):
    loader_args = {'batch_size': 1, 'shuffle': False, 'num_workers': 4, 'drop_last': False}
    train_dataset = SpringSceneFlowDataset(split='train', do_augment=False, use_estimated_disp=True, max_samples_per_scene=10, eval_mode=True, scenes=scenes)
    train_loader = DataLoader(train_dataset, **loader_args)

    depth_scale = 0.15
    metrics = {'D1': {}, 'D2': {}, 'Fl': {}, 'Sf': {}}

    for i_batch, test_data_blob in enumerate(tqdm(train_loader)):
        img1, img2, depth1, depth2, flow2d_gt, disp1_gt, disp3_gt, intrinsics = [data_item.cuda() for data_item in test_data_blob[:-1]]

        # pad and normalize
        ht, wd = img1.shape[2:]
        img1, img2, depth1, depth2, padding = prepare_images_and_depths(img1, img2, depth1, depth2, depth_scale=depth_scale)

        # run model
        Ts = model(img1, img2, depth1, depth2, intrinsics, iters=iterations)

        # extract 2D and 3D flow from transformation field
        flow2d_est, flow3d_est, _ = pops.induced_flow(Ts, depth1, intrinsics)
        flow2d_est = flow2d_est[:, :ht, :wd, :2]

        # extract disparites
        coords, _ = pops.projective_transform(Ts, depth1, intrinsics)
        disp3 = depth_scale * intrinsics[0,0] * coords[:, :ht, :wd, 2]
        disp1 = depth_scale * intrinsics[0,0] / depth1[:, :ht, :wd]

        errors = compute_spring_errors(disp1, disp3, flow2d_est, disp1_gt, disp3_gt, flow2d_gt)

        for category in errors:
            for error_measure in errors[category]:
                if error_measure not in metrics[category]:
                    metrics[category][error_measure] = 0
                metrics[category][error_measure] += errors[category][error_measure].item()

    for category in metrics:
        for error_measure in metrics[category]:
            metrics[category][error_measure] /= i_batch + 1

    return metrics

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='Training configuration file')
    parser.add_argument('--model', help='path the model weights')

    parser.add_argument('--dataset', default="kitti")

    args = parser.parse_args()
    config = load_config(args)

    model = torch.nn.DataParallel(RAFT3D(config))
    model.load_state_dict(torch.load(args.model), strict=False)

    model.cuda()
    model.eval()

    if "kitti" in args.dataset.lower():
        print("Kitti evaluation:")
        pprint(test_kitti(model, config["iterations"]))
        print()
    if "sceneflow" in args.dataset.lower() or "flyingthings" in args.dataset.lower():
        print("FlyingThings evaluation:")
        pprint(test_sceneflow(model, config["iterations"]))
        print()
    if "spring" in args.dataset.lower():
        print("Spring evaluation:")
        pprint(test_spring(model, config["iterations"]))
        print()
