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

import torch
import torch.nn as nn
import torch.nn.functional as F


class DSoftmaxLoss(nn.Module):
    """Computes the D-Softmax loss https://arxiv.org/pdf/1908.01281.pdf"""

    def __init__(self, num_classes=2, s=30., d=0.9, use_gpu=True, conf_penalty=False):
        super(DSoftmaxLoss, self).__init__()
        self.s = s
        self.eps = self.s * d
        self.conf_penalty = conf_penalty
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def get_last_info(self):
        return {}

    def forward(self, cos_theta, target):
        index = torch.zeros_like(cos_theta, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)

        zero = torch.tensor(0.).to(cos_theta.device)
        logits = torch.exp(self.s * cos_theta)
        target_z = torch.sum(torch.where(index, logits, zero), dim=1)
        non_target_z = torch.sum(torch.where(index, zero, logits), dim=1)
        loss_intra = torch.log(1 + self.eps / target_z)
        loss_inter = torch.log(1 + non_target_z)
        loss = loss_intra + loss_inter

        if self.conf_penalty:
            log_probs = self.logsoftmax(self.s * cos_theta)
            probs = torch.exp(log_probs)
            ent = (-probs*torch.log(probs.clamp(min=1e-12))).sum(1)
            loss = F.relu(loss - 0.2 * ent)
            with torch.no_grad():
                nonzero_count = loss.nonzero().size(0)
            return loss.sum() / nonzero_count

        return loss.mean()
