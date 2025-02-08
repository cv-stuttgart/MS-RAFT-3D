import torch
import torch.nn.functional as F

import lietorch_extras
try:
    import alt_cuda_corr
except:
    # alt_cuda_corr is not compiled
    pass


class CorrSampler(torch.autograd.Function):
    """ Index from correlation pyramid """
    @staticmethod
    def forward(ctx, volume, coords, radius):
        ctx.save_for_backward(volume,coords)
        ctx.radius = radius
        corr, = lietorch_extras.corr_index_forward(volume, coords, radius)
        return corr

    @staticmethod
    def backward(ctx, grad_output):
        volume, coords = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        grad_volume, = lietorch_extras.corr_index_backward(volume, coords, grad_output, ctx.radius)
        return grad_volume, None, None


class CorrBlock:
    def __init__(self, fmap1, fmap2, num_levels=2, radius=4):
        self.num_levels = num_levels
        self.radius = radius
        self.corr_pyramid = []

        # all pairs correlation
        corr = CorrBlock.corr(fmap1, fmap2)

        batch, h1, w1, h2, w2 = corr.shape
        corr = corr.reshape(batch*h1*w1, 1, h2, w2)
        
        for i in range(self.num_levels):
            self.corr_pyramid.append(
                corr.view(batch, h1, w1, h2//2**i, w2//2**i))
            corr = F.avg_pool2d(corr, 2, stride=2)
            
    def __call__(self, coords):
        coords = coords.permute(0, 3, 1, 2).contiguous()        # b, 2, h, w

        out_pyramid = []
        bz, _, ht, wd = coords.shape
        for i in range(self.num_levels):
            corr = CorrSampler.apply(self.corr_pyramid[i], coords/2**i, self.radius)
            out_pyramid.append(corr.view(bz, -1, ht, wd))

        return torch.cat(out_pyramid, dim=1)

    @staticmethod
    def corr(fmap1, fmap2):
        batch, dim, ht, wd = fmap1.shape
        fmap1 = fmap1.view(batch, dim, ht*wd)
        fmap2 = fmap2.view(batch, dim, ht*wd)
        corr = torch.matmul(fmap1.transpose(1,2), fmap2)
        return corr.view(batch, ht, wd, ht, wd) / torch.sqrt(torch.tensor(dim, dtype=torch.float))


class AlternateCorrSampler(torch.autograd.Function):
    """ On demand calculation of correlation values """
    @staticmethod
    def forward(ctx, fmap1, fmap2, coords, radius):
        ctx.save_for_backward(fmap1, fmap2, coords)
        ctx.radius = radius
        corr, = alt_cuda_corr.forward(fmap1, fmap2, coords, radius)
        return corr

    @staticmethod
    def backward(ctx, grad_output):
        fmap1, fmap2, coords = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        fmap1_grad, fmap2_grad, coords_grad = alt_cuda_corr.backward(fmap1, fmap2, coords, grad_output, ctx.radius)
        return fmap1_grad, fmap2_grad, coords_grad, None


class AlternateCorrBlock:
    def __init__(self, fmap1, fmap2, num_levels=2, radius=4):
        self.feature_dim = fmap1.shape[1]
        self.num_levels = num_levels
        self.radius = radius

        self.fmap1 = fmap1.permute(0, 2, 3, 1).contiguous()     # b, h, w, c
        self.fmap2 = [fmap2.permute(0, 2, 3, 1).contiguous()]   # b, h, w, c
        for i in range(self.num_levels - 1):
            fmap2 = F.avg_pool2d(fmap2, 2, stride=2)
            self.fmap2.append(fmap2.permute(0, 2, 3, 1).contiguous())

    def __call__(self, coords):
        out_pyramid = []
        bz, ht, wt, _ = coords.shape
        for i in range(self.num_levels):
            coords_i = (coords / 2**i).reshape(bz, 1, ht, wt, 2).contiguous()
            corr = AlternateCorrSampler.apply(self.fmap1, self.fmap2[i], coords_i, self.radius)
            out_pyramid.append(corr.squeeze(1))
        
        out_pyramid = torch.stack(out_pyramid, dim=1)
        out_pyramid = out_pyramid.reshape(bz, -1, ht, wt)
        return out_pyramid / torch.sqrt(torch.tensor(self.feature_dim, dtype=torch.float))
