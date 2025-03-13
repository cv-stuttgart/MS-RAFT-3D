import torch
import torch.nn.functional as F
import cv2
from torch.utils.tensorboard import SummaryWriter


SUM_FREQ = 100

RV_WEIGHT = 0.2     # Flow revision weight
DZ_WEIGHT = 250.0   # Depth weight

def set_to_zero(vec, mask):
    vec[~mask.squeeze(-1)] = 0
    return vec

def l1_loss(flow_est, flow_rev, dz_est, flow_gt, dz_gt, valid_mask):
    loss = torch.mean(set_to_zero((flow_est - flow_gt).abs(), valid_mask))
    loss += RV_WEIGHT * torch.mean(set_to_zero((flow_rev - flow_gt).abs(), valid_mask))
    loss += DZ_WEIGHT * torch.mean(set_to_zero((dz_est - dz_gt).abs(), valid_mask))
    return loss

def l2_loss(flow_est, flow_rev, dz_est, flow_gt, dz_gt, valid_mask):
    est = torch.cat((flow_est, DZ_WEIGHT * dz_est), dim=-1)
    gt = torch.cat((flow_gt, DZ_WEIGHT * dz_gt), dim=-1)

    loss = torch.mean(torch.sum(set_to_zero((est - gt)**2, valid_mask) + 1e-5, dim=-1).sqrt())
    loss += RV_WEIGHT * torch.mean(torch.sum(set_to_zero((flow_rev - flow_gt)**2, valid_mask) + 1e-5, dim=-1).sqrt())
    return loss

def samplewise_l1_loss(flow_est, flow_rev, dz_est, flow_gt, dz_gt, valid_mask):
    valid_ratio = valid_mask.float().mean(dim=[1, 2])
    validity = valid_ratio > 0.1
    scaling_imgwise = torch.zeros_like(valid_ratio)
    scaling_imgwise[validity] = torch.reciprocal(valid_ratio[validity])

    l1_norm = torch.sum((flow_est - flow_gt).abs(), dim=-1)
    l1_norm = l1_norm + RV_WEIGHT * torch.sum((flow_rev - flow_gt).abs(), dim=-1)
    l1_norm = l1_norm + DZ_WEIGHT * (dz_est - dz_gt).abs().squeeze(-1)
    l1_norm = set_to_zero(l1_norm, valid_mask)

    norm_imgwise = torch.mean(l1_norm, dim=[1, 2])
    loss = torch.sum((norm_imgwise + 0.01)**0.7 * scaling_imgwise) / validity.sum()
    return loss

def samplewise_l2_loss(flow_est, flow_rev, dz_est, flow_gt, dz_gt, valid_mask):
    valid_ratio = valid_mask.float().mean(dim=[1, 2])
    validity = valid_ratio > 0.1
    scaling_imgwise = torch.zeros_like(valid_ratio)
    scaling_imgwise[validity] = torch.reciprocal(valid_ratio[validity])

    est = torch.cat((flow_est, DZ_WEIGHT * dz_est), dim=-1)
    gt = torch.cat((flow_gt, DZ_WEIGHT * dz_gt), dim=-1)

    l2_norm = torch.sum((est - gt)**2 + 1e-5, dim=-1).sqrt()
    l2_norm = l2_norm + RV_WEIGHT * torch.sum((flow_rev - flow_gt)**2 + 1e-5, dim=-1).sqrt()
    l2_norm = set_to_zero(l2_norm, valid_mask)

    norm_imgwise = torch.mean(l2_norm, dim=[1, 2])
    loss = torch.sum((norm_imgwise + 0.01)**0.7 * scaling_imgwise) / validity.sum()
    return loss

def sequence_loss(flow2d_est, flow2d_rev, flow_gt, valid_mask, loss_fn, gamma=0.9):
    """ Loss function defined over sequence of flow predictions """

    N = len(flow2d_est)
    loss = 0.0

    for i in range(N):
        w = gamma**(N - i - 1)
        fl_rev = flow2d_rev[i]

        fl_est, dz_est = flow2d_est[i].split([2,1], dim=-1)
        fl_gt, dz_gt = flow_gt.split([2,1], dim=-1)

        loss += w * loss_fn(fl_est, fl_rev, dz_est, fl_gt, dz_gt, valid_mask)

    epe_2d = (fl_est - fl_gt).norm(dim=-1)
    epe_2d = epe_2d.view(-1)[valid_mask.view(-1)]

    epe_dz = (dz_est - dz_gt).norm(dim=-1)
    epe_dz = epe_dz.view(-1)[valid_mask.view(-1)]

    metrics = {
        'epe2d': epe_2d.mean().item(),
        'epedz': epe_dz.mean().item(),
        '1px': (epe_2d < 1).float().mean().item(),
        '3px': (epe_2d < 3).float().mean().item(),
        '5px': (epe_2d < 5).float().mean().item(),
    }

    return loss, metrics

def fetch_optimizer(model, args):
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.00001)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, args.lr, args.num_steps, pct_start=0.001, cycle_momentum=False)
    return optimizer, scheduler

def show_image(image):
    image = image.permute(1, 2, 0).cpu().numpy()
    cv2.imshow('image', image / 255.0)
    cv2.waitKey()

def normalize_image(image):
    image = image[:, [2,1,0]]
    mean = torch.as_tensor([0.485, 0.456, 0.406], device=image.device)
    std = torch.as_tensor([0.229, 0.224, 0.225], device=image.device)
    return (image/255.0).sub_(mean[:, None, None]).div_(std[:, None, None])

def prepare_images_and_depths(image1, image2, depth1, depth2, depth_scale=0.2, coarsest_scale=16):
    """ padding, normalization, and scaling """

    ht, wd = image1.shape[-2:]
    pad_h = (-ht) % coarsest_scale
    pad_w = (-wd) % coarsest_scale

    image1 = F.pad(image1, [0,pad_w,0,pad_h], mode='replicate')
    image2 = F.pad(image2, [0,pad_w,0,pad_h], mode='replicate')
    depth1 = F.pad(depth1[:,None], [0,pad_w,0,pad_h], mode='replicate')[:,0]
    depth2 = F.pad(depth2[:,None], [0,pad_w,0,pad_h], mode='replicate')[:,0]

    depth1 = (depth_scale * depth1).float()
    depth2 = (depth_scale * depth2).float()
    image1 = normalize_image(image1.float())
    image2 = normalize_image(image2.float())

    depth1 = depth1.float()
    depth2 = depth2.float()

    return image1, image2, depth1, depth2, (pad_w, pad_h)

class Logger:
    def __init__(self, name, start_step=0):
        self.total_steps = start_step
        self.running_loss = {}
        self.writer = None
        self.name = name
        self.print_keys = True

    def _print_training_status(self):
        if self.writer is None:
            self.writer = SummaryWriter(comment="_"+self.name)

        metrics_data = [self.running_loss[k]/SUM_FREQ for k in sorted(self.running_loss.keys())]
        training_str = "[{:6d}] ".format(self.total_steps+1)
        metrics_str = ("{:10.4f}, "*len(metrics_data)).format(*metrics_data)

        # print the training status
        if self.print_keys:
            self.print_keys = False
            print(" "*8 + ("{:>11},"*len(metrics_data)).format(*sorted(self.running_loss.keys())))
        print(training_str + metrics_str, flush=True)

        for key in self.running_loss:
            val = self.running_loss[key] / SUM_FREQ
            self.writer.add_scalar(key, val, self.total_steps)
            self.running_loss[key] = 0.0

    def push(self, metrics):

        for key in metrics:
            if key not in self.running_loss:
                self.running_loss[key] = 0.0

            self.running_loss[key] += metrics[key]

        if self.total_steps % SUM_FREQ == SUM_FREQ-1:
            self._print_training_status()
            self.running_loss = {}

        self.total_steps += 1

    def write_dict(self, results):
        if self.writer is None:
            self.writer = SummaryWriter(comment="_"+self.name)

        for key in results:
            self.writer.add_scalar(key, results[key], self.total_steps)
