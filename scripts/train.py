import sys
sys.path.append('.')

import os
import argparse
import cv2
import numpy as np
import warnings
import functools

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler

from lietorch import SE3
import raft3d.projective_ops as pops
from raft3d.raft3d import RAFT3D

from utils import Logger, show_image, normalize_image, fetch_optimizer, sequence_loss, l1_loss, l2_loss, samplewise_l1_loss, samplewise_l2_loss
from evaluation import test_kitti, test_sceneflow, test_spring
from config.config_loader import load_config

from data_readers.sceneflow import SceneFlow
from data_readers.kitti import KITTI
from data_readers.spring import SpringSceneFlowDataset

VAL_FREQ = 5000
SAVE_FREQ = 5000

def fetch_dataloader(config, phase):
    dataset = config['train']['dataset'][phase]
    image_size = config['train']['image_size'][phase]
    batch_size = config['train']['batch_size'][phase]

    assert image_size[0] % 16 == 0 and image_size[1] % 16 == 0, "Training only supports multiples of 16 as image sizes"

    gpuargs = {'shuffle': True, 'num_workers': 4, 'drop_last' : True}

    if dataset == 'things':
        train_dataset = SceneFlow(do_augment=True, image_size=image_size)
    elif dataset == 'kitti':
        train_dataset = KITTI(do_augment=True, image_size=image_size)
    elif dataset == 'spring':
        scenes = SpringSceneFlowDataset.SPRING_SPLIT_TRAIN_SCENES if os.getenv("DATASETS_SPRING_SPLIT", 'False').lower() in ('true', '1', 't') else None
        train_dataset = SpringSceneFlowDataset(do_augment=True, image_size=image_size, scenes=scenes)
    elif dataset == 'spring_subsample':
        assert image_size[0] % 32 == 0 and image_size[1] % 32 == 0, "Training on the subsampled spring dataset only supports multiples of 32 as image sizes"
        scenes = SpringSceneFlowDataset.SPRING_SPLIT_TRAIN_SCENES if os.getenv("DATASETS_SPRING_SPLIT", 'False').lower() in ('true', '1', 't') else None
        train_dataset = SpringSceneFlowDataset(do_augment=True, image_size=image_size, scenes=scenes, subsample_images=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, **gpuargs)
    return train_loader

def save_model(model, optimizer, config, phase, current_step):
    folder = 'checkpoints'
    if config['checkpoint_save_path'] is not None:
        folder = config['checkpoint_save_path']
    folder = os.path.join(folder, config['name'])
    file = f"{config['name']}_{config['train']['dataset'][phase]}_{current_step:06d}.pth"
    file_optimizer = f"{config['name']}_{config['train']['dataset'][phase]}_{current_step:06d}_optimizer.pth"

    if not os.path.isdir(folder):
        os.mkdir(folder)

    torch.save(model.state_dict(), os.path.join(folder, file))
    torch.save(optimizer.state_dict(), os.path.join(folder, file_optimizer))

def fetch_model(config):
    model = torch.nn.DataParallel(RAFT3D(config), device_ids=config["gpus"])
    if config["checkpoint_load_path"] is not None:
        model.load_state_dict(torch.load(config["checkpoint_load_path"]))

    model.cuda()
    model.train()

    return model


def fetch_optimizer(config, phase, model):
    optimizer = optim.Adam(model.parameters(), lr=config["train"]["lr"][phase], weight_decay=config["train"]["wdecay"][phase], eps=config["adamw_eps"])
    if config["checkpoint_load_path"] is not None:
        optimizer_load_path = f"{config['checkpoint_load_path'][:-4]}_optimizer.pth"
        if os.path.isfile(optimizer_load_path):
            optimizer.load_state_dict(torch.load(optimizer_load_path))

    return optimizer


def train_phase(dataloader, model, optimizer, scheduler, phase, logger, config, validation_func):
    num_steps = config["train"]["num_steps"][phase]
    iterations = config["iterations"]

    keep_training = True
    total_steps = 0
    if config["initial_phase"] == phase:
        total_steps = config["initial_step"]

    if config["train"]["loss_fn"][phase] == "l1":
        loss_fn = l1_loss
    elif config["train"]["loss_fn"][phase] == "l2":
        loss_fn = l2_loss
    elif config["train"]["loss_fn"][phase] == "samplewise_l1":
        loss_fn = samplewise_l1_loss
    elif config["train"]["loss_fn"][phase] == "samplewise_l2":
        loss_fn = samplewise_l2_loss
    else:
        raise ValueError(f'Loss function {config["train"]["loss_fn"][phase]} is unknown')

    while keep_training:
        for data_blob in dataloader:
            optimizer.zero_grad()
            image1, image2, depth1, depth2, flow_gt, valid, intrinsics = [x.cuda() for x in data_blob]

            image1 = normalize_image(image1.float())
            image2 = normalize_image(image2.float())

            flow2d_est, flow2d_rev = model(image1, image2, depth1, depth2, intrinsics, iters=iterations, train_mode=True)

            loss, metrics = sequence_loss(flow2d_est, flow2d_rev, flow_gt, valid, loss_fn)

            if torch.isnan(loss):
                print("nan loss during training. Exiting...")
                exit(0)


            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()

            metrics.update({"loss": float(loss.float().item())})
            logger.push(metrics)

            total_steps += 1

            if total_steps % SAVE_FREQ == 0 or total_steps >= num_steps:
                save_model(model, optimizer, config, phase, total_steps)

            if total_steps % VAL_FREQ == 0 or total_steps >= num_steps:
                results = validation_func(model.module, iterations)
                print(results, flush=True)
                logger.write_dict(results)

            if total_steps >= num_steps:
                keep_training = False
                break

def train(config):
    initial_phase = config["initial_phase"]
    passed_steps = initial_step = config["initial_step"]
    if config["initial_phase"] != 0:
        passed_steps += sum(config["train"]["num_steps"][:config["initial_phase"]])
    num_steps = config["train"]["num_steps"]
    num_phases = len(num_steps)

    learning_rate = config["train"]["lr"]
    wdecay = config["train"]["wdecay"]
    gamma = config["train"]["gamma"]
    adamw_eps = config["adamw_eps"]

    iterations = config["iterations"]

    model = fetch_model(config)
    logger = Logger(name=config["name"], start_step=passed_steps)

    for phase in range(initial_phase, num_phases):
        optimizer = fetch_optimizer(config, phase, model)
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer, learning_rate[phase], num_steps[phase], pct_start=0.001, cycle_momentum=False)
        with warnings.catch_warnings():
            # suppress scheduler warning
            warnings.simplefilter("ignore")
            for _ in range(initial_step):
                scheduler.step()

        dataloader = fetch_dataloader(config, phase)

        if 'spring' in config['train']['dataset'][phase] and os.getenv("DATASETS_SPRING_SPLIT", 'False').lower() in ('true', '1', 't'):
            def validation_func(model, it):
                results = test_spring(model, it, SpringSceneFlowDataset.SPRING_SPLIT_TEST_SCENES)
                return {
                    f"{x}_{y}": results[x][y]
                        for x in results
                        for y in results[x]
                }
        else:
            validation_func = test_kitti

        train_phase(dataloader, model, optimizer, scheduler, phase, logger, config, validation_func)

        initial_step = 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='Training configuration file')
    parser.add_argument('--ckpt', help='Checkpoint to restore')
    parser.add_argument('--initial_step', type=int, default=0, help='Number of steps the checkpoint has already trained')
    parser.add_argument('--initial_phase', type=int, default=0, help='Number of phases the checkpoint has already trained')
    parser.add_argument('--save', default='checkpoints', help='Folder for saving checkpoints')

    args = parser.parse_args()

    if not os.path.isdir(args.save):
        os.mkdir(args.save)

    config = load_config(args)

    print(config)
    train(config)
