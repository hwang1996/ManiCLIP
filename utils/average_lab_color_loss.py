import torch
from torch import nn

class AvgLabLoss(nn.Module):
    def __init__(self):
        super(AvgLabLoss, self).__init__()
        self.criterion = nn.L1Loss()
        self.M = torch.tensor([[0.412453, 0.357580, 0.180423], [0.212671, 0.715160, 0.072169], [0.019334, 0.119193, 0.950227]])

    # cal lab written by liuqk
    def f(self, input):
        output = input * 1
        mask = input > 0.008856
        output[mask] = torch.pow(input[mask], 1 / 3)
        output[~mask] = 7.787 * input[~mask] + 0.137931
        return output

    def rgb2xyz(self, input):
        assert input.size(1) == 3
        M_tmp = self.M.to(input.device).unsqueeze(0)
        M_tmp = M_tmp.repeat(input.size(0), 1, 1)  # BxCxC
        output = torch.einsum('bnc,bchw->bnhw', M_tmp, input)  # BxCxHxW
        M_tmp = M_tmp.sum(dim=2, keepdim=True)  # BxCx1
        M_tmp = M_tmp.unsqueeze(3)  # BxCx1x1
        return output / M_tmp

    def xyz2lab(self, input):
        assert input.size(1) == 3
        output = input * 1
        xyz_f = self.f(input)
        # compute l
        mask = input[:, 1, :, :] > 0.008856
        output[:, 0, :, :][mask] = 116 * xyz_f[:, 1, :, :][mask] - 16
        output[:, 0, :, :][~mask] = 903.3 * input[:, 1, :, :][~mask]
        # compute a
        output[:, 1, :, :] = 500 * (xyz_f[:, 0, :, :] - xyz_f[:, 1, :, :])
        # compute b
        output[:, 2, :, :] = 200 * (xyz_f[:, 1, :, :] - xyz_f[:, 2, :, :])
        return output

    def cal_hair_avg(self, input, mask):
        x = input * mask
        sum = torch.sum(torch.sum(x, dim=2, keepdim=True), dim=3, keepdim=True) # [n,3,1,1]
        mask_sum = torch.sum(torch.sum(mask, dim=2, keepdim=True), dim=3, keepdim=True) # [n,1,1,1]
        mask_sum[mask_sum == 0] = 1
        avg = sum / mask_sum
        return avg

    def forward(self, fake, real, mask_fake, mask_real):
        # the mask is [n,1,h,w]
        # normalize to 0~1
        fake_RGB = (fake + 1) / 2.0
        real_RGB = (real + 1) / 2.0
        # from RGB to Lab by liuqk
        # import pdb; pdb.set_trace()
        fake_xyz = self.rgb2xyz(fake_RGB)
        fake_Lab = self.xyz2lab(fake_xyz)
        real_xyz = self.rgb2xyz(real_RGB)
        real_Lab = self.xyz2lab(real_xyz)
        # cal average value
        fake_Lab_avg = self.cal_hair_avg(fake_Lab, mask_fake)
        real_Lab_avg = self.cal_hair_avg(real_Lab, mask_real)

        loss = self.criterion(fake_Lab_avg, real_Lab_avg)

        return loss