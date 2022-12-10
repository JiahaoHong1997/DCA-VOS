import torch
from torch import nn
import numpy as np


class TimeEncoding(nn.Module):

    def __init__(self, d_hid, n_position=200):
        super(TimeEncoding, self).__init__()

        # Not a parameter
        self.register_buffer('time_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * hid_j / (d_hid // 3)) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2]) / 10  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2]) / 10  # dim 2i+1

        # x = torch.from_numpy(sinusoid_table)
        # print(x.size())
        # x = torch.from_numpy(sinusoid_table).unsqueeze(2).unsqueeze(3).expand(-1,-1,25,30).transpose(1,0)
        # from tensorboardX import SummaryWriter
        # from torchvision.utils import make_grid
        # from datetime import datetime
        # TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
        # writer = SummaryWriter(comment=TIMESTAMP)

        
        # for i in range(30, 40):
        #     writer.add_image('layer1',
        #             make_grid(x[i][:10].detach().cpu().unsqueeze(dim=1), nrow=20, padding=1, normalize=True,
        #             pad_value=1, scale_each=True, range=(0, 1)), i)

        # writer.flush()
        # writer.close()

        return torch.FloatTensor(sinusoid_table).unsqueeze(0).transpose(1, 2)  # (1,dim,n_position)

    def forward(self, x, idx):
        return x + self.time_table[:, :, idx:idx + 1].expand(-1, -1, x.size(-1)).clone().detach()
