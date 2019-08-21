"""
 Copyright (c) 2019 Intel Corporation
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaCosLoss(nn.Module):
    """Computes the AdaCos loss https://arxiv.org/pdf/1905.00292.pdf"""

    def __init__(self, num_classes=2, use_gpu=True, conf_penalty=False):
        super(AdaCosLoss, self).__init__()
        self.register_buffer('s', torch.full((1,), math.sqrt(2.) * math.log(num_classes - 1)))
        self.register_buffer('theta_med', torch.full((1,), math.pi / 4.))
        self.register_buffer('b_avg', torch.full((1,), num_classes))

    def get_last_info(self):
        return {'s': self.s.item(), 'theta_med': self.theta_med.item(), 'B': self.b_avg.item()}

    def forward(self, cos_theta, target):
        with torch.no_grad():
            index = torch.zeros_like(cos_theta, dtype=torch.uint8)
            index.scatter_(1, target.data.view(-1, 1), 1)

            zero = torch.tensor(0.).to(cos_theta.device)
            theta = torch.where(index, torch.acos(cos_theta), zero)
            theta_med = torch.median(torch.sum(theta, dim=1)).item()
            theta_med = min(math.pi / 4., theta_med)

            b_est = torch.where(index > 0, zero, torch.exp(self.s.item() * cos_theta))
            b_avg = torch.sum(b_est).item() / target.shape[0]

            s = math.log(b_avg) / math.cos(theta_med)
            self.s.data = s * torch.ones_like(self.s.data)
            self.b_avg.data = b_avg * torch.ones_like(self.b_avg.data)
            self.theta_med.data = theta_med * torch.ones_like(self.theta_med.data)

        return torch.unsqueeze(F.cross_entropy(s * cos_theta, target), -1)
