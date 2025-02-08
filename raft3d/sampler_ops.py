import torch
import torch.nn.functional as F

from lietorch import SE3

def bilinear_sampler(img, coords, mode='bilinear', mask=False):
    """ Wrapper for grid_sample, uses pixel coordinates """
    H, W = img.shape[-2:]
    xgrid, ygrid = coords.split([1,1], dim=-1)
    xgrid = 2*xgrid/(W-1) - 1
    ygrid = 2*ygrid/(H-1) - 1

    grid = torch.cat([xgrid, ygrid], dim=-1)
    img = F.grid_sample(img, grid, align_corners=True)

    if mask:
        mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
        return img, mask.float()

    return img

def depth_sampler(depths, coords):
    depths_proj, valid = bilinear_sampler(depths[:,None], coords, mask=True)
    return depths_proj.squeeze(dim=1), valid

def cvx_upsample(data, mask, mask_scale=2):
    """ Convex upsampling [H/scale, W/scale, 2] -> [H, W, 2] """
    batch, ht, wd, dim = data.shape
    data = data.permute(0, 3, 1, 2)
    mask = mask.view(batch, 1, 9, mask_scale, mask_scale, ht, wd)
    mask = torch.softmax(mask, dim=2)

    up_data = F.unfold(data, [3,3], padding=1)
    up_data = up_data.view(batch, dim, 9, 1, 1, ht, wd)

    up_data = torch.sum(mask * up_data, dim=2)
    up_data = up_data.permute(0, 4, 2, 5, 3, 1)
    up_data = up_data.reshape(batch, mask_scale*ht, mask_scale*wd, dim)
    return up_data

def upsample_se3(Ts, mask, mask_scale=2, bilinear_scale_factor=None):
    """ upsample a se3 field """
    tau_phi = Ts.log()
    tau_phi = cvx_upsample(tau_phi, mask, mask_scale)
    if bilinear_scale_factor is not None:
        tau_phi = upsample_bilinear(tau_phi, bilinear_scale_factor)
    return SE3.exp(tau_phi)

def upsample_ae(ae, mask, mask_scale=2):
    """ upsample a ae field """
    ae = ae.permute(0, 2, 3, 1)
    ae = cvx_upsample(ae, mask, mask_scale)
    ae = ae.permute(0, 3, 1, 2)
    return ae

def upsample_bilinear(data, scale_factor):
    """ upsample [bt, ht, wd, dim] -> [bt, ht*scale_factor, wd*scale_factor, dim] bilinear """
    data = data.permute(0, 3, 1, 2)
    data = F.interpolate(data, scale_factor=(scale_factor, scale_factor), mode='bilinear', align_corners=True)
    data = data.permute(0, 2, 3, 1)
    return data

def downsample_depth(depth, scale):
    start = (scale - 1) // 2
    step = int(scale)
    return depth[:, start::step, start::step]
