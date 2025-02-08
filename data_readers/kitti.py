
import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F

import os
import os.path as osp
import cv2
import random
import csv

from glob import glob

from . import frame_utils
from .augmentation import RGBDAugmentor, SparseAugmentor

import flow_IO


class KITTIEval(data.Dataset):

    crop = 80

    def __init__(self, image_size=None, root=os.getenv("DATASETS_KITTI_ROOT"), do_augment=True, mode="testing", disp_train=os.getenv("DATASETS_KITTI_DISP_TRAIN"), disp_test=os.getenv("DATASETS_KITTI_DISP_TEST"), split_dir=os.getenv("DATASETS_KITTI_SPLIT_EVAL")):
        if split_dir == None:
            split_dir = osp.join(root, mode)

        self.init_seed = None

        self.data_list = [ osp.basename(file)[:6]
            for file in sorted(glob(osp.join(split_dir, "image_2", "*10.png"))) ]

        disp_path = disp_test if mode == "testing" else disp_train

        self.image1_list = []
        self.image2_list = []
        self.disp1_ga_list = []
        self.disp2_ga_list = []
        self.calib_list = []

        for data_number in self.data_list:
            self.image1_list.append(osp.join(root, mode, "image_2", f"{data_number}_10.png"))
            self.image2_list.append(osp.join(root, mode, "image_2", f"{data_number}_11.png"))

            self.disp1_ga_list.append(osp.join(disp_path, f"{data_number}_10.png"))
            self.disp2_ga_list.append(osp.join(disp_path, f"{data_number}_11.png"))

            self.calib_list.append(osp.join(root, mode, "calib_cam_to_cam", f"{data_number}.txt"))

        self.intrinsics_list = []
        for calib_file in self.calib_list:
            with open(calib_file) as f:
                reader = csv.reader(f, delimiter=' ')
                for row in reader:
                    if row[0] == 'K_02:':
                        K = np.array(row[1:], dtype=np.float32).reshape(3,3)
                        kvec = np.array([K[0,0], K[1,1], K[0,2], K[1,2]])
                        self.intrinsics_list.append(kvec)

        self.istest = (mode == "testing")
        if not self.istest:
            self.gt_d1_list = []
            self.gt_d2_list = []
            self.gt_fl_list = []

            for data_number in self.data_list:
                self.gt_d1_list.append(osp.join(root, mode, "disp_occ_0", f"{data_number}_10.png"))
                self.gt_d2_list.append(osp.join(root, mode, "disp_occ_1", f"{data_number}_10.png"))
                self.gt_fl_list.append(osp.join(root, mode, "flow_occ", f"{data_number}_10.png"))

    @staticmethod
    def write_prediction(index, disp1, disp2, flow):

        def writeFlowKITTI(filename, uv):
            uv = 64.0 * uv + 2**15
            valid = np.ones([uv.shape[0], uv.shape[1], 1])
            uv = np.concatenate([uv, valid], axis=-1).astype(np.uint16)
            cv2.imwrite(filename, uv[..., ::-1])

        def writeDispKITTI(filename, disp):
            disp = (256 * disp).astype(np.uint16)
            cv2.imwrite(filename, disp)

        disp1 = np.pad(disp1, ((KITTIEval.crop,0),(0,0)), mode='edge')
        disp2 = np.pad(disp2, ((KITTIEval.crop, 0), (0,0)), mode='edge')
        flow = np.pad(flow, ((KITTIEval.crop, 0), (0,0),(0,0)), mode='edge')

        disp1_path = 'kitti_submission/disp_0/%06d_10.png' % index
        disp2_path = 'kitti_submission/disp_1/%06d_10.png' % index
        flow_path = 'kitti_submission/flow/%06d_10.png' % index

        writeDispKITTI(disp1_path, disp1)
        writeDispKITTI(disp2_path, disp2)
        writeFlowKITTI(flow_path, flow)

    def __len__(self):
        return len(self.image1_list)

    def __getitem__(self, index):

        intrinsics = self.intrinsics_list[index]
        image1 = cv2.imread(self.image1_list[index])
        image2 = cv2.imread(self.image2_list[index])

        disp1 = cv2.imread(self.disp1_ga_list[index], cv2.IMREAD_ANYDEPTH) / 256.0
        disp2 = cv2.imread(self.disp2_ga_list[index], cv2.IMREAD_ANYDEPTH) / 256.0

        image1 = image1[self.crop:]
        image2 = image2[self.crop:]
        disp1 = disp1[self.crop:]
        disp2 = disp2[self.crop:]
        intrinsics[3] -= self.crop

        image1 = torch.from_numpy(image1).float().permute(2,0,1)
        image2 = torch.from_numpy(image2).float().permute(2,0,1)
        disp1 = torch.from_numpy(disp1).float()
        disp2 = torch.from_numpy(disp2).float()
        intrinsics = torch.from_numpy(intrinsics).float()

        if self.istest:
            return image1, image2, disp1, disp2, intrinsics

        gt_d1 = flow_IO.readDispFile(self.gt_d1_list[index])
        gt_d2 = flow_IO.readDispFile(self.gt_d2_list[index])
        gt_fl = flow_IO.readFlowFile(self.gt_fl_list[index])

        return image1, image2, disp1, disp2, intrinsics, gt_d1, gt_d2, gt_fl


class KITTI(data.Dataset):
    def __init__(self, image_size=None, root=os.getenv("DATASETS_KITTI_ROOT"), do_augment=True, disp_train=os.getenv("DATASETS_KITTI_DISP_TRAIN"), split_dir=os.getenv("DATASETS_KITTI_SPLIT_TRAIN")):
        import csv

        if split_dir == None:
            split_dir = osp.join(root, "training")

        self.init_seed = None
        self.crop = 80

        if do_augment:
            self.augmentor = SparseAugmentor(image_size)
        else:
            self.augmentor = None

        data_list = [ osp.basename(file)[:6]
            for file in sorted(glob(osp.join(split_dir, "image_2", "*10.png"))) ]

        self.image1_list = []
        self.image2_list = []
        self.disp1_list = []
        self.disp2_list = []
        self.disp1_ga_list = []
        self.disp2_ga_list = []
        self.flow_list = []
        self.calib_list = []

        for data_number in data_list:
            self.image1_list.append(osp.join(root, "training", "image_2", f"{data_number}_10.png"))
            self.image2_list.append(osp.join(root, "training", "image_2", f"{data_number}_11.png"))

            self.disp1_list.append(osp.join(root, "training", "disp_occ_0", f"{data_number}_10.png"))
            self.disp2_list.append(osp.join(root, "training", "disp_occ_1", f"{data_number}_10.png"))

            self.disp1_ga_list.append(osp.join(disp_train, f"{data_number}_10.png"))
            self.disp2_ga_list.append(osp.join(disp_train, f"{data_number}_11.png"))

            self.flow_list.append(osp.join(root, "training", "flow_occ", f"{data_number}_10.png"))
            self.calib_list.append(osp.join(root, "training", "calib_cam_to_cam", f"{data_number}.txt"))

        self.intrinsics_list = []
        for calib_file in self.calib_list:
            with open(calib_file) as f:
                reader = csv.reader(f, delimiter=' ')
                for row in reader:
                    if row[0] == 'K_02:':
                        K = np.array(row[1:], dtype=np.float32).reshape(3,3)
                        kvec = np.array([K[0,0], K[1,1], K[0,2], K[1,2]])
                        self.intrinsics_list.append(kvec)

    def __len__(self):
        return len(self.image1_list)

    def __getitem__(self, index):
        if not self.init_seed:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                torch.manual_seed(worker_info.id)
                np.random.seed(worker_info.id)
                random.seed(worker_info.id)
                self.init_seed = True

        image1 = cv2.imread(self.image1_list[index])
        image2 = cv2.imread(self.image2_list[index])

        disp1 = cv2.imread(self.disp1_list[index], cv2.IMREAD_ANYDEPTH) / 256.0
        disp2 = cv2.imread(self.disp2_list[index], cv2.IMREAD_ANYDEPTH) / 256.0
        disp1_dense = cv2.imread(self.disp1_ga_list[index], cv2.IMREAD_ANYDEPTH) / 256.0
        disp2_dense = cv2.imread(self.disp2_ga_list[index], cv2.IMREAD_ANYDEPTH) / 256.0

        flow, valid = frame_utils.readFlowKITTI(self.flow_list[index])
        intrinsics = self.intrinsics_list[index]

        SCALE = np.random.uniform(0.08, 0.15)

        # crop top 80 pixels, no ground truth information
        image1 = image1[self.crop:]
        image2 = image2[self.crop:]
        disp1 = disp1[self.crop:]
        disp2 = disp2[self.crop:]
        flow = flow[self.crop:]
        valid = valid[self.crop:]
        disp1_dense = disp1_dense[self.crop:]
        disp2_dense = disp2_dense[self.crop:]
        intrinsics[3] -= self.crop

        image1 = torch.from_numpy(image1).float().permute(2,0,1)
        image2 = torch.from_numpy(image2).float().permute(2,0,1)

        disp1 = torch.from_numpy(disp1 / intrinsics[0]) / SCALE
        disp2 = torch.from_numpy(disp2 / intrinsics[0]) / SCALE
        disp1_dense = torch.from_numpy(disp1_dense / intrinsics[0]) / SCALE
        disp2_dense = torch.from_numpy(disp2_dense / intrinsics[0]) / SCALE

        dz = (disp2 - disp1_dense).unsqueeze(dim=-1)
        depth1 = 1.0 / disp1_dense.clamp(min=0.01).float()
        depth2 = 1.0 / disp2_dense.clamp(min=0.01).float()

        intrinsics = torch.from_numpy(intrinsics)
        valid = torch.from_numpy(valid)
        flow = torch.from_numpy(flow)

        valid = valid * (disp2 > 0).float()
        flow = torch.cat([flow, dz], -1)

        if self.augmentor is not None:
            image1, image2, depth1, depth2, flow, valid, intrinsics = \
                self.augmentor(image1, image2, depth1, depth2, flow, valid, intrinsics)

        valid = valid.unsqueeze(-1) > 0.5
        return image1, image2, depth1, depth2, flow, valid, intrinsics
