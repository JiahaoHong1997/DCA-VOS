import torch
import math
from torch import nn


class PositionEncoding(nn.Module):
    def __init__(self, d_hid, length=100):
        super(PositionEncoding, self).__init__()

        self.r, self.a = self._get_sinusoid_encoding_table(length, d_hid)
        self.register_buffer('r_table', self.r)
        self.register_buffer('a_table', self.a)

    def _get_sinusoid_encoding_table(self, length, d_hid):
        pos_list = torch.arange(1, length + 1, 1)
        x_pos = pos_list.unsqueeze(1).expand(length, length)
        y_pos = pos_list.unsqueeze(0).expand(length, length)

        r1 = torch.sqrt((x_pos.pow(2) + y_pos.pow(2)).to(torch.float))
        r = (r1-torch.min(r1))/(torch.max(r1)-torch.mean(r1))+0.1
        # print(r)
        # x_pos = (x_pos-torch.min(x_pos))/(torch.max(x_pos)-torch.mean(x_pos))+0.1
        a = torch.acos(x_pos / r1)

        r_sinusoid_table = torch.ones(d_hid, length, length)
        a_sinusoid_table = torch.ones(d_hid, length, length)

        for hid_j in range(d_hid):
            r_sinusoid_table[hid_j] = r / math.pow(10000, 2 * hid_j / (d_hid // 3))
            a_sinusoid_table[hid_j] = a / math.pow(10000, 2 * hid_j / (d_hid // 3))

        r_sinusoid_table[0::2, :, :] = torch.sin(r_sinusoid_table[0::2, :, :]) 
        r_sinusoid_table[1::2, :, :] = torch.cos(r_sinusoid_table[1::2, :, :]) 

        a_sinusoid_table[0::2, :, :] = torch.sin(a_sinusoid_table[0::2, :, :]) 
        a_sinusoid_table[1::2, :, :] = torch.cos(a_sinusoid_table[1::2, :, :]) 

        # from tensorboardX import SummaryWriter
        # from torchvision.utils import make_grid
        # from datetime import datetime
        # TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
        # writer = SummaryWriter(comment=TIMESTAMP)
        
        # x1 = nn.functional.interpolate(r_sinusoid_table.unsqueeze(0), size=(75, 150), mode='bilinear', align_corners=True)
        # x2 = nn.functional.interpolate(a_sinusoid_table.unsqueeze(0), size=(75, 150), mode='bilinear', align_corners=True)

        # writer.add_image('layer1',
        #             make_grid(x1[0].detach().cpu().unsqueeze(dim=1), nrow=20, padding=1, normalize=True,
        #             pad_value=1, scale_each=True, range=(0, 1)))

        # writer.add_image('layer2',
        #             make_grid(x2[0].detach().cpu().unsqueeze(dim=1), nrow=20, padding=1, normalize=True,
        #             pad_value=1, scale_each=True, range=(0, 1)))
        # writer.flush()
        # writer.close()

        return r_sinusoid_table, a_sinusoid_table

    def forward(self, x):
        # print(x.size())
        # print(self.r_table.size())
        # print(self.a_table.size())
        return x + self.r_table[:, :x.size(2), :x.size(3)].clone().detach() + self.a_table[:, :x.size(2), :x.size(3)].clone().detach()
