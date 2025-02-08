# Adapted from  https://github.com/cv-stuttgart/spring_utils

import torch
import torch.utils.data as data
import numpy as np
from glob import glob
import os
from PIL import Image
import flow_IO

from .augmentation import RGBDAugmentor
import raft3d.projective_ops as pops


class SpringSceneFlowDataset(data.Dataset):
    """
    Dataset class for Spring scene flow dataset.
    For train, this dataset returns image1, image2, image3, image4, disp1, disp2, flow and a data tuple (framenum, scene name, left/right cam, FW/BW direction).
    For test, this dataset returns image1, image2, image3, image4 and a data tuple (framenum, scene name, left/right cam, FW/BW direction).
    The images are:
    image1: reference frame
    image2: same time step as reference frame, but other camera
    image3: next/previous time step compared to reference frame, same camera as reference frame
    image4: other time step and other camera than reference frame

    root: root directory of the spring dataset (should contain test/train directories)
    split: train/test split
    subsample_groundtruth: If true, return ground truth such that it has the same dimensions as the images (1920x1080px); if false return full 4K resolution
    """

    SPRING_SPLIT_TRAIN_SCENES = [1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 16, 18, 20, 21, 22, 23, 25, 26, 27, 30, 33, 36, 37, 38, 41, 43, 45, 47]
    SPRING_SPLIT_TEST_SCENES = [12, 15, 17, 24, 32, 39, 44]

    def __init__(self, root=os.getenv("DATASETS_SPRING_ROOT"), split='train', subsample_groundtruth=True, image_size=None, do_augment=True, disp_path=os.getenv("DATASETS_SPRING_DISP"), use_estimated_disp=True, max_samples_per_scene=None, eval_mode=False, scenes=None, subsample_images=False):
        super(SpringSceneFlowDataset, self).__init__()

        assert split in ["train", "test"]
        seq_root = os.path.join(root, split)

        if not os.path.exists(seq_root):
            raise ValueError(f"Spring {split} directory does not exist: {seq_root}")

        if subsample_images:
            image_size = (image_size[0] // 2, image_size[1] // 2)

        if do_augment:
            self.augmentor = RGBDAugmentor(image_size)
        else:
            self.augmentor = None

        self.eval_mode = eval_mode
        self.subsample_groundtruth = subsample_groundtruth
        self.subsample_images = subsample_images
        self.use_estimated_disp = use_estimated_disp
        self.split = split
        self.seq_root = seq_root
        self.disp_root = os.path.join(disp_path, split) if use_estimated_disp else seq_root
        self.data_list = []

        sample_stepsize = None if max_samples_per_scene is None else 3
        sample_slice_end = None if max_samples_per_scene is None else max_samples_per_scene * sample_stepsize

        for scene in sorted(os.listdir(seq_root)):
            if scenes is not None and int(scene) not in scenes:
                continue

            intrinsics_list = []
            with open(os.path.join(seq_root, scene, "cam_data", "intrinsics.txt")) as f:
                for row in f:
                    intrinsics = torch.tensor([float(x) for x in row.split(' ')])
                    intrinsics_list.append(intrinsics)

            for cam in ["left", "right"]:
                images = sorted(glob(os.path.join(seq_root, scene, f"frame_{cam}", '*.png')))
                # forward
                for frame in range(1, len(images))[:sample_slice_end:sample_stepsize]:
                    self.data_list.append((frame, scene, cam, "FW", intrinsics_list[frame - 1]))
                # backward
                for frame in reversed(range(2, len(images)+1)[:sample_slice_end:sample_stepsize]):
                    self.data_list.append((frame, scene, cam, "BW", intrinsics_list[frame - 1]))

        print(f"Spring loaded {len(self.data_list)} datapoints")

    def load_images_depth(self, frame_data):
        frame, scene, cam, direction, _ = frame_data

        if cam == "left":
            othercam = "right"
        else:
            othercam = "left"

        if direction == "FW":
            othertimestep = frame+1
        else:
            othertimestep = frame-1

        # load images
        img1_path = os.path.join(self.seq_root, scene, f"frame_{cam}", f"frame_{cam}_{frame:04d}.png")
        img2_path = os.path.join(self.seq_root, scene, f"frame_{cam}", f"frame_{cam}_{othertimestep:04d}.png")
        img1 = np.asarray(Image.open(img1_path))
        img2 = np.asarray(Image.open(img2_path))
        img1 = torch.from_numpy(img1).permute(2,0,1).float()
        img2 = torch.from_numpy(img2).permute(2,0,1).float()

        # load disparities used for the prediction
        disp1_path = os.path.join(self.disp_root, scene, f"disp1_{cam}", f"disp1_{cam}_{frame:04d}.dsp5")
        disp2_path = os.path.join(self.disp_root, scene, f"disp1_{cam}", f"disp1_{cam}_{othertimestep:04d}.dsp5")
        disp1 = torch.from_numpy(flow_IO.readDispFile(disp1_path)).float()
        disp2 = torch.from_numpy(flow_IO.readDispFile(disp2_path)).float()

        if not self.use_estimated_disp:
            # use only every second value in both spatial directions ==> ground truth will have same dimensions as images
            disp1 = disp1[::2,::2]
            disp2 = disp2[::2,::2]

        if self.subsample_images:
            img1 = img1[:, ::2, ::2]
            img2 = img2[:, ::2, ::2]

            disp1 = disp1[::2, ::2]
            disp2 = disp2[::2, ::2]

        return img1, img2, disp1, disp2

    def load_ground_truth(self, frame_data):
        frame, scene, cam, direction, _ = frame_data

        if direction == "FW":
            othertimestep = frame+1
        else:
            othertimestep = frame-1

        # load disparity ground truth
        disp1_gt_path = os.path.join(self.seq_root, scene, f"disp1_{cam}", f"disp1_{cam}_{frame:04d}.dsp5")
        disp3_gt_path = os.path.join(self.seq_root, scene, f"disp2_{direction}_{cam}", f"disp2_{direction}_{cam}_{frame:04d}.dsp5")
        disp1_gt = torch.from_numpy(flow_IO.readDispFile(disp1_gt_path)).float()
        disp3_gt = torch.from_numpy(flow_IO.readDispFile(disp3_gt_path)).float()

        # load flow ground truth
        flow_gt_path = os.path.join(self.seq_root, scene, f"flow_{direction}_{cam}", f"flow_{direction}_{cam}_{frame:04d}.flo5")
        flow_gt = torch.from_numpy(flow_IO.readFlowFile(flow_gt_path)).float()

        if self.subsample_groundtruth:
            # use only every second value in both spatial directions ==> ground truth will have same dimensions as images
            disp1_gt = disp1_gt[::2,::2]
            disp3_gt = disp3_gt[::2,::2]
            flow_gt = flow_gt[::2,::2]

        if self.subsample_images:
            disp1_gt = disp1_gt[::2,::2]
            disp3_gt = disp3_gt[::2,::2]
            flow_gt = 0.5 * flow_gt[::2,::2]

        return flow_gt, disp1_gt, disp3_gt

    def __getitem__(self, index):
        frame_data = self.data_list[index]
        intrinsics = frame_data[4]

        # load images
        img1, img2, disp1, disp2 = self.load_images_depth(frame_data)

        depth1 = intrinsics[0] / disp1.clip(min=0.001)
        depth2 = intrinsics[0] / disp2.clip(min=0.001)

        if self.split == "test":
            return img1, img2, depth1, depth2, intrinsics, frame_data

        # load ground truth
        flow2d_gt, disp1_gt, disp3_gt = self.load_ground_truth(frame_data)

        depth1_gt = intrinsics[0] / disp1_gt.clip(min=0.001)
        depth3_gt = intrinsics[0] / disp3_gt.clip(min=0.001)

        if self.eval_mode:
            return img1, img2, depth1, depth2, flow2d_gt, disp1_gt, disp3_gt, intrinsics, frame_data


        # SCALE = np.random.uniform(0.1, 0.5)       # FlyingThings
        # SCALE = np.random.uniform(0.08, 0.15)     # KITTI
        SCALE = 1 if self.augmentor is None else 0.1 + 0.1 * torch.rand(1)

        depth1 *= SCALE
        depth2 *= SCALE
        depth1_gt *= SCALE
        depth3_gt *= SCALE

        flowz_gt = (1.0/depth3_gt - 1.0/depth1_gt).unsqueeze(-1)
        flow_xyz_gt = torch.cat([flow2d_gt, flowz_gt], dim=-1)

        if self.augmentor is not None:
            img1, img2, depth1, depth2, flow_xyz_gt, intrinsics = \
                self.augmentor(img1, img2, depth1, depth2, flow_xyz_gt, intrinsics)

        valid = ~torch.isnan(flow_xyz_gt).any(dim=-1).unsqueeze(-1)

        return img1, img2, depth1, depth2, flow_xyz_gt, valid, intrinsics

    def __len__(self):
        return len(self.data_list)
