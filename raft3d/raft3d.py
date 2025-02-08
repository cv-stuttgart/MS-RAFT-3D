import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast

# lietorch for tangent space backpropogation
from lietorch import SE3

from .blocks.extractor import FeatureEncoderBasic, FeatureEncoderCCMR, ContextEncoderBasic, ContextEncoderFPN, ContextEncoderCCMR, ContextEncoderResnet, ContextEncoderResnetFeatureAggregation
from .blocks.corr import CorrBlock, AlternateCorrBlock
from .blocks.update import BasicUpdateBlock, BasicUpdateBlockBilaplacian
from .sampler_ops import depth_sampler, cvx_upsample, upsample_bilinear, upsample_se3, upsample_ae, downsample_depth

from . import projective_ops as pops
from . import se3_field


class RAFT3D(nn.Module):
    def __init__(self, config):
        super(RAFT3D, self).__init__()

        self.config = config
        hdim = config['hidden_dim']
        cdim = config['context_dim']

        # feature network, context network, and update block
        if len(config['iterations']) == 4:
            assert config['feature_encoder'] == 'ccmr', 'The 4-scale model only implements the ccmr feature encoder'
            self.fnet = FeatureEncoderCCMR(norm_fn='instance', four_scales=True)
        elif config['feature_encoder'] == 'basic':
            self.fnet = FeatureEncoderBasic()
        elif config['feature_encoder'] == 'ccmr':
            self.fnet = FeatureEncoderCCMR(norm_fn='instance')
        else:
            raise ValueError(f"Unknown feature encoder {config['feature_encoder']}")

        context_encoder_channels = self.config["context_encoder_dim"] if self.config["context_encoder_dim"] is not None else [64, 96, 128, 192, 256]

        if len(config['iterations']) == 4:
            assert config['context_encoder'] == 'basic', 'The 4-scale model only implements the basic context encoder'
            self.cnet = ContextEncoderBasic(output_dim=hdim+cdim, channels=context_encoder_channels, four_scales=True)
        elif config['context_encoder'] == 'basic':
            self.cnet = ContextEncoderBasic(output_dim=hdim+cdim, channels=context_encoder_channels)
        elif config['context_encoder'] == 'fpn':
            self.cnet = ContextEncoderFPN(output_dim=hdim+cdim, channels=context_encoder_channels)
        elif config['context_encoder'] == 'ccmr':
            self.cnet = ContextEncoderCCMR(output_dim=hdim+cdim, channels=context_encoder_channels)
        elif config['context_encoder'] == 'resnet':
            self.cnet = ContextEncoderResnet(output_dim=hdim+cdim)
        elif config['context_encoder'] == 'resnet_aggregate':
            self.cnet = ContextEncoderResnetFeatureAggregation(output_dim=hdim+cdim)
        else:
            raise ValueError(f"Unknown context encoder {config['context_encoder']}")

        if config['corr'] == 'pre_calc':
            self.corr = CorrBlock
        elif config['corr'] == 'on_demand':
            self.corr = AlternateCorrBlock
        else:
            raise ValueError(f"Unknown corr type {config['corr']}")

        if config['update_operator'] == 'basic':
            self.update_block = BasicUpdateBlock(config, hidden_dim=hdim)
        elif config['update_operator'] == 'bilaplacian':
            self.update_block = BasicUpdateBlockBilaplacian(config, hidden_dim=hdim)
        else:
            raise ValueError(f"Unknown update operator {config['update_operator']}")

        count_params = lambda m: sum(p.numel() for p in m.parameters() if p.requires_grad)
        print(f"Parameter Count: overall={count_params(self)}, fnet={count_params(self.fnet)}, cnet={count_params(self.cnet)}, update_block={count_params(self.update_block)}")

    def initializer(self, image1, scale=16):
        """ Initialize coords and transformation maps """

        batch_size, ch, ht, wd = image1.shape
        device = image1.device

        Ts = SE3.Identity(batch_size, ht//scale, wd//scale, device=device)
        ae = torch.zeros(batch_size, 16, ht//scale, wd//scale, device=device)

        return Ts, ae, self.get_grid(image1, scale)

    def get_grid(self, image1, scale):
        """ Initialize coords """

        batch_size, ch, ht, wd = image1.shape
        device = image1.device

        y0, x0 = torch.meshgrid(torch.arange(ht//scale), torch.arange(wd//scale))
        coords0 = torch.stack([x0, y0], dim=-1).float()
        coords0 = coords0[None].repeat(batch_size, 1, 1, 1).to(device)

        return coords0

    def forward(self, image1, image2, depth1, depth2, intrinsics, iters, train_mode=False):
        """ Estimate optical flow between pair of frames """

        levels = self.config["corr_levels"]
        radius = self.config["corr_radius"]

        se3_lm = self.config["se3_lm"]
        se3_ep = self.config["se3_ep"]
        se3_radius = (lambda level: 2**(level + 4)) if self.config["se3_neighborhood"] is None else (lambda level: self.config["se3_neighborhood"])

        Ts, ae, coords0 = self.initializer(image1, scale=16)

        fmap_pyramid = self.fnet([image1, image2])
        if self.config["train_save_vram"]:
            cmap_pyramid = torch.utils.checkpoint.checkpoint(self.cnet, image1, use_reentrant=False)
        else:
            cmap_pyramid = self.cnet(image1)

        assert len(iters) == len(fmap_pyramid), 'the number of scales should match the number of feature maps.'
        assert len(iters) == len(cmap_pyramid), 'the number of scales should match the number of context maps.'

        flow_est_list = []
        flow_rev_list = []

        for index, (fmap1, fmap2) in enumerate(fmap_pyramid):
            corr_fn = self.corr(fmap1.float(), fmap2.float(), num_levels=levels, radius=radius)
            net, inp = torch.split(cmap_pyramid[index], [self.config['hidden_dim'], self.config['context_dim']], dim=1)
            net = torch.tanh(net)
            inp = torch.relu(inp)

            # intrinsics and depth at current resolution
            scale = 16 // (2 ** index)
            intrinsics_scale = intrinsics / scale
            depth1_scale = downsample_depth(depth1, scale)
            depth1_upsample = downsample_depth(depth1, scale // 2)
            depth2_scale = downsample_depth(depth2, scale)

            for itr in range(iters[index]):
                Ts = Ts.detach()

                if index >= 1 and itr == 0:
                    # Upsample
                    coords0 = self.get_grid(image1, scale)
                    Ts = upsample_se3(Ts, mask)
                    ae = upsample_ae(ae, mask)

                coords1_xyz, _ = pops.projective_transform(Ts, depth1_scale, intrinsics_scale)
                coords1, zinv_proj = coords1_xyz.split([2,1], dim=-1)
                zinv, _ = depth_sampler(1.0/depth2_scale, coords1)

                corr = corr_fn(coords1)
                flow = coords1 - coords0

                dz = zinv.unsqueeze(-1) - zinv_proj
                twist = Ts.log()

                net, mask, ae, delta, weight = \
                    self.update_block(net, inp, corr, flow, dz, twist, ae)

                target = coords1_xyz.permute(0,3,1,2) + delta
                target = target.contiguous()

                # Gauss-Newton step
                Ts = se3_field.step_inplace(Ts, ae, target, weight, depth1_scale, intrinsics_scale, lm=se3_lm, ep=se3_ep, radius=se3_radius(index))

                if train_mode:
                    up_factor = 2**(3 - index)

                    flow2d_rev = target.permute(0, 2, 3, 1)[...,:2] - coords0
                    flow2d_rev = cvx_upsample(2 * flow2d_rev, mask)

                    if index < 3:
                        flow2d_rev = up_factor * upsample_bilinear(flow2d_rev, up_factor)
                        Ts_up = upsample_se3(Ts, mask, bilinear_scale_factor=up_factor)
                    else:
                        Ts_up = upsample_se3(Ts, mask)

                    flow2d_est, flow3d_est, valid = pops.induced_flow(Ts_up, depth1, intrinsics)

                    flow_rev_list.append(flow2d_rev)
                    flow_est_list.append(flow2d_est)

                    if torch.isnan(flow2d_rev).any() or torch.isnan(flow2d_est).any():
                        print(f"nan prediction on level {index} iteration {itr}: isnan(flow)={torch.isnan(flow2d_est).any()}, isnan(revision)={torch.isnan(flow2d_rev).any()}")
                        print("Exiting...")
                        exit(0)

        if train_mode:
            return flow_est_list, flow_rev_list

        if len(iters) < 4:
            Ts_up = upsample_se3(Ts, mask, bilinear_scale_factor=2)
        else:
            Ts_up = upsample_se3(Ts, mask)

        return Ts_up

