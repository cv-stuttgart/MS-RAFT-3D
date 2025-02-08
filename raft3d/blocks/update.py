import torch
import torch.nn as nn

# lietorch for tangent space backpropogation
from lietorch import SE3

from .grid import GridFactor
from .gru import ConvGRU

GRAD_CLIP = .01


class GradClip(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x

    @staticmethod
    def backward(ctx, grad_x):
        o = torch.zeros_like(grad_x)
        grad_x = torch.where(grad_x.abs()>GRAD_CLIP, o, grad_x)
        grad_x = torch.where(torch.isnan(grad_x), o, grad_x)
        return grad_x


class GradientClip(nn.Module):
    def __init__(self):
        super(GradientClip, self).__init__()

    def forward(self, x):
        return GradClip.apply(x)


class GridSmoother(nn.Module):
    def __init__(self):
        super(GridSmoother, self).__init__()
        self.sym_factor = None
        self.sym_shape = None

    def forward(self, ae, wxwy):

        factor = GridFactor()
        ae = ae.permute(0,2,3,1)

        wx = wxwy[:,0].unsqueeze(-1)
        wy = wxwy[:,1].unsqueeze(-1)
        wu = torch.ones_like(wx)
        J = torch.ones_like(wu).unsqueeze(-2)

        # residual terms
        ru = ae.unsqueeze(-2)
        rx = torch.zeros_like(ru)
        ry = torch.zeros_like(ru)

        factor.add_factor([J], wu, ru, ftype='u')
        factor.add_factor([J,-J], wx, rx, ftype='h')
        factor.add_factor([J,-J], wy, ry, ftype='v')
        factor._build_factors()

        ae, self.sym_factor, self.sym_shape = factor.solveAAt(sym_factor=self.sym_factor, sym_shape=self.sym_shape)
        ae = ae.squeeze(dim=-2).permute(0,3,1,2).contiguous()

        return ae


class BasicUpdateBlockBilaplacian(nn.Module):
    def __init__(self, args, hidden_dim=128, input_dim=128, mask_scale=2):
        super(BasicUpdateBlockBilaplacian, self).__init__()
        self.args = args

        self.gru = ConvGRU(hidden_dim, dilation=3)

        self.solver = GridSmoother()

        corr_size = (self.args["corr_radius"] * 2 + 1)**2 * self.args["corr_levels"]
        self.corr_enc = nn.Sequential(
            nn.Conv2d(corr_size, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 3*128, 1, padding=0))

        self.flow_enc = nn.Sequential(
            nn.Conv2d(9, 128, 7, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 3*128, 1, padding=0))

        self.ae_enc = nn.Conv2d(16, 3*128, 3, padding=1)

        self.ae = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 16, 1, padding=0),
            GradientClip())

        self.delta = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 3, 1, padding=0),
            GradientClip())

        self.weight = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 3, 1, padding=0),
            GradientClip(),
            nn.Sigmoid())

        self.ae_wts = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 2, 1, padding=0),
            GradientClip(),
            nn.Softplus())

        self.mask = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, mask_scale*mask_scale*9, 1, padding=0),
            GradientClip())

    def forward(self, net, inp, corr, flow, twist, dz, ae, upsample=True):
        motion_info = torch.cat([flow, 10*dz, 10*twist], dim=-1)
        motion_info = motion_info.clamp(-50.0, 50.0).permute(0,3,1,2)

        mot = self.flow_enc(motion_info)
        cor = self.corr_enc(corr)

        ae = self.ae_enc(ae)
        net = self.gru(net, inp, cor, mot, ae)

        ae = self.ae(net)
        mask = self.mask(net)
        delta = self.delta(net)
        weight = self.weight(net)

        edges = 5 * self.ae_wts(net)
        ae = self.solver(ae, edges)

        return net, mask, ae, delta, weight


class BasicUpdateBlock(nn.Module):
    def __init__(self, args, hidden_dim=128, input_dim=128, mask_scale=2):
        super(BasicUpdateBlock, self).__init__()
        self.args = args
        self.gru = ConvGRU(hidden_dim)

        corr_size = (self.args["corr_radius"] * 2 + 1)**2 * self.args["corr_levels"]
        self.corr_enc = nn.Sequential(
            nn.Conv2d(corr_size, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 3*128, 1, padding=0))

        self.flow_enc = nn.Sequential(
            nn.Conv2d(9, 128, 7, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 3*128, 1, padding=0))

        self.ae = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 32, 1, padding=0),
            GradientClip())

        self.delta = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 3, 1, padding=0),
            GradientClip())

        self.weight = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 3, 1, padding=0),
            nn.Sigmoid(),
            GradientClip())

        self.mask = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, mask_scale*mask_scale*9, 1, padding=0),
            GradientClip())

    def forward(self, net, inp, corr, flow, twist, dz, ae, upsample=True):
        motion_info = torch.cat([flow, 10*dz, 10*twist], dim=-1)
        motion_info = motion_info.clamp(-50.0, 50.0).permute(0,3,1,2)

        mot = self.flow_enc(motion_info)
        cor = self.corr_enc(corr)

        net = self.gru(net, inp, cor, mot)

        ae = self.ae(net)
        mask = self.mask(net)
        delta = self.delta(net)
        weight = self.weight(net)

        return net, mask, ae, delta, weight
