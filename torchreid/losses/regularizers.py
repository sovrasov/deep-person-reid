import torch
import torch.nn as nn


class ConvRegularizer(nn.Module):
    def __init__(self, reg_class, controller):
        super().__init__()
        self.reg_instance = reg_class(controller)

    def get_all_conv_layers(self, module):
        if isinstance(module, (nn.Sequential, list)):
            for m in module:
                yield from self.get_all_conv_layers(m)

        if isinstance(module, nn.Conv2d):
            yield module

    def forward(self, net, ignore=False):

        accumulator = torch.tensor(0.0).cuda()

        if ignore:
            return accumulator

        all_mods = [module for module in net.module.modules() if type(module) != nn.Sequential]
        for conv in self.get_all_conv_layers(all_mods):
            accumulator += self.reg_instance(conv.weight)

        return accumulator


class SVMORegularizer(nn.Module):
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def dominant_eigenvalue(self, A):  # A: 'N x N'
        N, _ = A.size()
        x = torch.rand(N, 1, device='cuda')
        Ax = (A @ x)
        AAx = (A @ Ax)
        return AAx.permute(1, 0) @ Ax / (Ax.permute(1, 0) @ Ax)

    def get_singular_values(self, A):  # A: 'M x N, M >= N'
        ATA = A.permute(1, 0) @ A
        N, _ = ATA.size()
        largest = self.dominant_eigenvalue(ATA)
        I = torch.eye(N, device='cuda')  # noqa
        I = I * largest  # noqa
        tmp = self.dominant_eigenvalue(ATA - I)
        return tmp + largest, largest

    def forward(self, W):  # W: 'S x C x H x W'
        # old_W = W
        old_size = W.size()
        if old_size[0] == 1:
            return 0
        W = W.view(old_size[0], -1).permute(1, 0)  # (C x H x W) x S
        smallest, largest = self.get_singular_values(W)
        return (
            self.beta * 10 * (largest - smallest)**2
        ).squeeze()


class NoneRegularizer(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, _):
        return torch.tensor(0.0).cuda()


mapping = {
    False: NoneRegularizer,
    True: SVMORegularizer,
}


def get_regularizer(cfg_reg):
    name = cfg_reg.ow
    return ConvRegularizer(mapping[name], cfg_reg.ow_beta)


class OFPenalty(nn.Module):

    def __init__(self, beta, layers_list=[]):
        super().__init__()
        self.penalty_position = frozenset(layers_list)
        self.beta = beta

    def dominant_eigenvalue(self, A):
        B, N, _ = A.size()
        x = torch.randn(B, N, 1, device='cuda')

        for _ in range(1):
            x = torch.bmm(A, x)
        # x: 'B x N x 1'
        numerator = torch.bmm(
            torch.bmm(A, x).view(B, 1, N),
            x
        ).squeeze()
        denominator = (torch.norm(x.view(B, N), p=2, dim=1) ** 2).squeeze()

        return numerator / denominator

    def get_singular_values(self, A):
        AAT = torch.bmm(A, A.permute(0, 2, 1))
        B, N, _ = AAT.size()
        largest = self.dominant_eigenvalue(AAT)
        I = torch.eye(N, device='cuda').expand(B, N, N)  # noqa
        I = I * largest.view(B, 1, 1).repeat(1, N, N)  # noqa
        tmp = self.dominant_eigenvalue(AAT - I)
        return tmp + largest, largest

    def apply_penalty(self, k, x):

        if isinstance(x, (tuple)):
            if not len(x):
                return 0.
            return sum([self.apply_penalty(k, xx) for xx in x]) / len(x)

        batches, channels, height, width = x.size()
        W = x.view(batches, channels, -1)
        smallest, largest = self.get_singular_values(W)
        singular_penalty = (largest - smallest) * self.beta

        if k == 'intermediate':
            singular_penalty *= 0.01

        return singular_penalty.sum() / (x.size(0))

    def forward(self, inputs):
        singular_penalty = sum([self.apply_penalty(' ', x) for x in inputs])
        return singular_penalty
